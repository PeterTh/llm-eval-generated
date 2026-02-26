/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
// ============================================================================

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    // Translate triangle to box center
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    // Triangle edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // Test box normals (AABB axes)
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    // Test 9 edge cross products
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) +
                  boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Octree for Spatial Acceleration
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box
        minBound = triangles[0].a;
        maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x);
                minBound.y = std::min(minBound.y, v->y);
                minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x);
                maxBound.y = std::max(maxBound.y, v->y);
                maxBound.z = std::max(maxBound.z, v->z);
            }
        }

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

            // Check which children this triangle overlaps
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        // Check if we can't split further (all triangles in one child)
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true;
                break;
            }
        }

        if (!canSplit) {
            triangleIndices = indices;
            return;
        }

        // Build children
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center;
                Vec3 childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x;
                childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y;
                childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z;
                childMax.z = (i & 4) ? maxBound.z : center.z;

                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
    }

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

// Generate an icosphere by subdividing an icosahedron
class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        // Initial icosahedron vertices
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<Vec3> vertices = {
            Vec3(-1,  t,  0).normalized() * radius,
            Vec3( 1,  t,  0).normalized() * radius,
            Vec3(-1, -t,  0).normalized() * radius,
            Vec3( 1, -t,  0).normalized() * radius,
            Vec3( 0, -1,  t).normalized() * radius,
            Vec3( 0,  1,  t).normalized() * radius,
            Vec3( 0, -1, -t).normalized() * radius,
            Vec3( 0,  1, -t).normalized() * radius,
            Vec3( t,  0, -1).normalized() * radius,
            Vec3( t,  0,  1).normalized() * radius,
            Vec3(-t,  0, -1).normalized() * radius,
            Vec3(-t,  0,  1).normalized() * radius
        };

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto& face : faces) {
                idx_t a = getMidpoint(face[0], face[1]);
                idx_t b = getMidpoint(face[1], face[2]);
                idx_t c = getMidpoint(face[2], face[0]);

                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        // Build triangles (normals pointing inward for a "room")
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Generation
// ============================================================================

class RandomGenerator {
public:
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;

    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }

    std::string get_state() const {
        std::ostringstream oss;
        oss << rng;
        return oss.str();
    }

    void set_state(const std::string& s) {
        std::istringstream iss(s);
        iss >> rng;
    }
};

// Generate a random point inside a triangle using barycentric coordinates
Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const Octree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;  // Ray is blocked
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    // Global problem size
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    // MPI decomposition over triangle index i (rows)
    int mpiRank = 0;
    int mpiSize = 1;
    size_t iBegin = 0;
    size_t iEnd = 0;
    size_t localTriangles = 0;

    std::vector<int> mpiCounts; // per-rank counts for Allgatherv/Gatherv (in elements)
    std::vector<int> mpiDispls;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)

    // Distributed by i-rows: localTriangles x N, row-major
    std::vector<val_t> kijLocal;    // Form factors
    std::vector<int> tauLocal;      // Time delays

    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix) (kept replicated via Allgatherv per timestep)
    std::vector<val_t> distances;   // Computed distances from source (replicated only on rank0 after gather)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex = 0;

    // CUDA buffers
    val_t* d_areas = nullptr;
    val_t* d_rhoLocal = nullptr;
    val_t* d_kijLocal = nullptr;
    int* d_tauLocal = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
    size_t idxLocal2d(size_t localI, size_t j) const { return localI * numTriangles + j; }
};

static inline void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static inline void computeDecomposition(size_t n, int rank, int size, size_t& begin, size_t& end) {
    size_t base = n / static_cast<size_t>(size);
    size_t rem = n % static_cast<size_t>(size);
    begin = static_cast<size_t>(rank) * base + static_cast<size_t>(std::min(rank, static_cast<int>(rem)));
    size_t count = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    end = begin + count;
}

static inline bool cullPair(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    computeDecomposition(state.numTriangles, state.mpiRank, state.mpiSize, state.iBegin, state.iEnd);
    state.localTriangles = state.iEnd - state.iBegin;

    if (state.mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree for spatial acceleration (replicated)
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize distributed matrices and replicated time-series
    state.kijLocal.assign(state.localTriangles * state.numTriangles, ZERO);
    state.tauLocal.assign(state.localTriangles * state.numTriangles, 0);
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);

    // Distances (gathered, then replicated)
    state.distances.assign(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Prepare MPI gather metadata (int counts)
    state.mpiCounts.resize(state.mpiSize);
    state.mpiDispls.resize(state.mpiSize);
    for (int r = 0; r < state.mpiSize; ++r) {
        size_t b = 0, e = 0;
        computeDecomposition(state.numTriangles, r, state.mpiSize, b, e);
        state.mpiCounts[r] = static_cast<int>(e - b);
        state.mpiDispls[r] = static_cast<int>(b);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

static std::string recvRngStateFromRoot(const SimulationState& state) {
    std::string s;
    if (state.mpiRank == 0) return s;
    int len = 0;
    MPI_Recv(&len, 1, MPI_INT, 0, 1001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    s.resize(static_cast<size_t>(len));
    MPI_Recv(s.data(), len, MPI_CHAR, 0, 1002, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return s;
}

static void sendRngStateToRank(int rank, const std::string& s) {
    int len = static_cast<int>(s.size());
    MPI_Send(&len, 1, MPI_INT, rank, 1001, MPI_COMM_WORLD);
    MPI_Send(s.data(), len, MPI_CHAR, rank, 1002, MPI_COMM_WORLD);
}

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij) with MPI+OpenMP...\n");

    // Determine RNG state at the start of this rank's first row in the original sequential order.
    std::string myStartState;
    if (state.mpiRank == 0) {
        RandomGenerator rng(42);
        size_t currentRow = 0;
        for (int r = 0; r < state.mpiSize; ++r) {
            size_t b = 0, e = 0;
            computeDecomposition(state.numTriangles, r, state.mpiSize, b, e);

            while (currentRow < b) {
                // Burn all RNG draws for row currentRow as the sequential version would.
                const Triangle& triI = state.triangles[currentRow];
                for (size_t j = 0; j < state.numTriangles; ++j) {
                    if (currentRow == j) continue;
                    if (cullPair(triI, state.triangles[j])) continue;
                    for (int k = 0; k < 4 * NUM_RAYS; ++k) (void)rng.rand();
                }
                ++currentRow;
            }

            std::string s = rng.get_state();
            if (r == 0) myStartState = s;
            else sendRngStateToRank(r, s);
        }
    } else {
        myStartState = recvRngStateFromRoot(state);
    }

    RandomGenerator baseRng(42);
    if (!myStartState.empty()) baseRng.set_state(myStartState);

    // Precompute row-start RNG engines for deterministic OpenMP across rows.
    std::vector<std::mt19937> rowEng(state.localTriangles);
    for (size_t localI = 0; localI < state.localTriangles; ++localI) {
        rowEng[localI] = baseRng.rng;

        size_t i = state.iBegin + localI;
        const Triangle& triI = state.triangles[i];
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            if (cullPair(triI, state.triangles[j])) continue;
            for (int k = 0; k < 4 * NUM_RAYS; ++k) (void)baseRng.rand();
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t localI = 0; localI < state.localTriangles; ++localI) {
        RandomGenerator rng(42);
        rng.rng = rowEng[localI];

        size_t i = state.iBegin + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kijLocal[state.idxLocal2d(localI, j)] = computeKij(
                i, j, state.triangles, state.octree, rng);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau) with MPI+OpenMP...\n");

    #pragma omp parallel for schedule(static)
    for (size_t localI = 0; localI < state.localTriangles; ++localI) {
        size_t i = state.iBegin + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tauLocal[state.idxLocal2d(localI, j)] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

static void ensureCudaBuffers(SimulationState& state) {
    if (state.d_areas) return;

    int devCount = 0;
    cudaCheck(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    cudaCheck(cudaSetDevice(state.mpiRank % devCount), "cudaSetDevice");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    cudaCheck(cudaMalloc(&state.d_areas, N * sizeof(val_t)), "cudaMalloc d_areas");
    cudaCheck(cudaMalloc(&state.d_rhoLocal, state.localTriangles * sizeof(val_t)), "cudaMalloc d_rhoLocal");
    cudaCheck(cudaMalloc(&state.d_kijLocal, state.localTriangles * N * sizeof(val_t)), "cudaMalloc d_kijLocal");
    cudaCheck(cudaMalloc(&state.d_tauLocal, state.localTriangles * N * sizeof(int)), "cudaMalloc d_tauLocal");
    cudaCheck(cudaMalloc(&state.d_radE, T * N * sizeof(val_t)), "cudaMalloc d_radE");
    cudaCheck(cudaMalloc(&state.d_radB, T * N * sizeof(val_t)), "cudaMalloc d_radB");

    cudaCheck(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice), "H2D areas");

    std::vector<val_t> rhoLocal(state.localTriangles);
    for (size_t localI = 0; localI < state.localTriangles; ++localI) {
        rhoLocal[localI] = state.rho[state.iBegin + localI];
    }
    if (!rhoLocal.empty()) {
        cudaCheck(cudaMemcpy(state.d_rhoLocal, rhoLocal.data(), rhoLocal.size() * sizeof(val_t), cudaMemcpyHostToDevice), "H2D rhoLocal");
    }

    if (!state.kijLocal.empty()) {
        cudaCheck(cudaMemcpy(state.d_kijLocal, state.kijLocal.data(), state.kijLocal.size() * sizeof(val_t), cudaMemcpyHostToDevice), "H2D kijLocal");
    }
    if (!state.tauLocal.empty()) {
        cudaCheck(cudaMemcpy(state.d_tauLocal, state.tauLocal.data(), state.tauLocal.size() * sizeof(int), cudaMemcpyHostToDevice), "H2D tauLocal");
    }

    cudaCheck(cudaMemcpy(state.d_radE, state.radE.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice), "H2D radE");
    cudaCheck(cudaMemcpy(state.d_radB, state.radB.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice), "H2D radB");
}

static void freeCudaBuffers(SimulationState& state) {
    if (state.d_areas) cudaCheck(cudaFree(state.d_areas), "cudaFree d_areas");
    if (state.d_rhoLocal) cudaCheck(cudaFree(state.d_rhoLocal), "cudaFree d_rhoLocal");
    if (state.d_kijLocal) cudaCheck(cudaFree(state.d_kijLocal), "cudaFree d_kijLocal");
    if (state.d_tauLocal) cudaCheck(cudaFree(state.d_tauLocal), "cudaFree d_tauLocal");
    if (state.d_radE) cudaCheck(cudaFree(state.d_radE), "cudaFree d_radE");
    if (state.d_radB) cudaCheck(cudaFree(state.d_radB), "cudaFree d_radB");

    state.d_areas = nullptr;
    state.d_rhoLocal = nullptr;
    state.d_kijLocal = nullptr;
    state.d_tauLocal = nullptr;
    state.d_radE = nullptr;
    state.d_radB = nullptr;
}

__global__ void simStepKernel(size_t t, size_t N, size_t iBegin,
                             const val_t* __restrict__ areas,
                             const val_t* __restrict__ rhoLocal,
                             const val_t* __restrict__ kijLocal,
                             const int* __restrict__ tauLocal,
                             const val_t* __restrict__ radE,
                             val_t* __restrict__ radB) {
    const size_t localI = static_cast<size_t>(blockIdx.x);
    const size_t i = iBegin + localI;

    val_t partial = 0.0f;
    for (size_t j = static_cast<size_t>(threadIdx.x); j < N; j += static_cast<size_t>(blockDim.x)) {
        if (i == j) continue;

        int tauij = tauLocal[localI * N + j];
        if (static_cast<int>(t) < tauij) continue;

        val_t kij = kijLocal[localI * N + j];
        if (kij <= 0.0f) continue;

        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= 0.0f) continue;

        val_t w = kij * areas[j];
        if (w > 1.0f) w = 1.0f;
        partial += w * radJ;
    }

    extern __shared__ val_t sh[];
    sh[threadIdx.x] = partial;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < static_cast<int>(s)) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        radB[t * N + i] = rhoLocal[localI] * sh[0] + radE[t * N + i];
    }
}

__global__ void distancesKernel(size_t N, size_t T, size_t iBegin, size_t localTriangles,
                                size_t sourceIndex,
                                const val_t* __restrict__ radB,
                                val_t* __restrict__ distLocal) {
    size_t localI = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (localI >= localTriangles) return;

    size_t i = iBegin + localI;
    val_t maxCorr = 0.0f;
    int bestT = 0;

    for (size_t t = 0; t < T; ++t) {
        val_t sum = 0.0f;
        for (size_t tt = t; tt < T; ++tt) {
            val_t pB = radB[tt * N + i];
            val_t pS = radB[(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distLocal[localI] = WAVE_SPEED * static_cast<val_t>(bestT);
}

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation with CUDA+MPI...\n");
    ensureCudaBuffers(state);

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    const int threads = 256;
    const size_t shmem = static_cast<size_t>(threads) * sizeof(val_t);

    std::vector<val_t> localRow(state.localTriangles);
    std::vector<val_t> fullRow(N);

    for (size_t t = 0; t < T; ++t) {
        if (state.localTriangles > 0) {
            simStepKernel<<<static_cast<unsigned>(state.localTriangles), threads, shmem>>>(
                t, N, state.iBegin,
                state.d_areas, state.d_rhoLocal, state.d_kijLocal, state.d_tauLocal,
                state.d_radE, state.d_radB);
            cudaCheck(cudaGetLastError(), "simStepKernel launch");
            cudaCheck(cudaDeviceSynchronize(), "simStepKernel sync");

            cudaCheck(cudaMemcpy(localRow.data(), state.d_radB + t * N + state.iBegin,
                                 state.localTriangles * sizeof(val_t), cudaMemcpyDeviceToHost),
                      "D2H radB local row");
        }

        MPI_Allgatherv(localRow.data(), static_cast<int>(state.localTriangles), MPI_FLOAT,
                       fullRow.data(), state.mpiCounts.data(), state.mpiDispls.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        std::memcpy(&state.radB[t * N], fullRow.data(), N * sizeof(val_t));
        cudaCheck(cudaMemcpy(state.d_radB + t * N, fullRow.data(), N * sizeof(val_t), cudaMemcpyHostToDevice),
                  "H2D radB full row");

        if (state.mpiRank == 0 && (((t + 1) % 10 == 0) || (t + 1 == T))) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation with CUDA+MPI...\n");
    ensureCudaBuffers(state);

    val_t* d_distLocal = nullptr;
    cudaCheck(cudaMalloc(&d_distLocal, state.localTriangles * sizeof(val_t)), "cudaMalloc d_distLocal");

    const int threads = 256;
    const int blocks = static_cast<int>((state.localTriangles + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

    if (state.localTriangles > 0) {
        distancesKernel<<<blocks, threads>>>(state.numTriangles, state.numTimesteps, state.iBegin, state.localTriangles,
                                             state.sourceIndex, state.d_radB, d_distLocal);
        cudaCheck(cudaGetLastError(), "distancesKernel launch");
        cudaCheck(cudaDeviceSynchronize(), "distancesKernel sync");
    }

    std::vector<val_t> localDist(state.localTriangles);
    if (state.localTriangles > 0) {
        cudaCheck(cudaMemcpy(localDist.data(), d_distLocal, state.localTriangles * sizeof(val_t), cudaMemcpyDeviceToHost),
                  "D2H distances");
    }

    MPI_Allgatherv(localDist.data(), static_cast<int>(state.localTriangles), MPI_FLOAT,
                   state.distances.data(), state.mpiCounts.data(), state.mpiDispls.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);

    cudaCheck(cudaFree(d_distLocal), "cudaFree d_distLocal");
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    int ok = 1;

    // Global count of non-zero Kij entries (distributed)
    long long localNonZeroKij = 0;
    for (val_t v : state.kijLocal) {
        if (v > EPSILON) localNonZeroKij++;
    }
    long long globalNonZeroKij = 0;
    MPI_Reduce(&localNonZeroKij, &globalNonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (state.mpiRank == 0) {
        printf("\nValidation:\n");

        // Check that distances are non-negative
        bool allNonNegative = true;
        val_t minDist = std::numeric_limits<val_t>::max();
        val_t maxDist = std::numeric_limits<val_t>::lowest();
        val_t sumDist = ZERO;
        int nonZeroCount = 0;

        for (size_t i = 0; i < state.numTriangles; ++i) {
            val_t d = state.distances[i];
            if (d < 0) {
                allNonNegative = false;
                printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
            }
            if (!std::isfinite(d)) {
                printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
                ok = 0;
                break;
            }
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            sumDist += d;
            if (d > EPSILON) nonZeroCount++;
        }

        if (ok) {
            printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
            printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
            printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

            // Check source distance is zero or very small
            val_t srcDist = state.distances[state.sourceIndex];
            if (srcDist > WAVE_SPEED * 2) {
                printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
            }

            // Check radiosity propagation (some triangles should have received energy)
            int receivedEnergy = 0;
            for (size_t i = 0; i < state.numTriangles; ++i) {
                for (size_t t = 0; t < state.numTimesteps; ++t) {
                    if (state.radB[state.idxTN(t, i)] > EPSILON) {
                        receivedEnergy++;
                        break;
                    }
                }
            }

            printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
            if (receivedEnergy == 0) {
                printf("  ERROR: No triangles received energy - simulation failed\n");
                ok = 0;
            }

            printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
                   globalNonZeroKij, state.numTriangles * state.numTriangles,
                   100.0f * static_cast<double>(globalNonZeroKij) / static_cast<double>(state.numTriangles * state.numTriangles));

            if (globalNonZeroKij == 0) {
                printf("  ERROR: All form factors are zero - visibility computation failed\n");
                ok = 0;
            }

            if (!allNonNegative) ok = 0;

            if (ok) printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok != 0;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    // Icosphere: 20 triangles initially, 4x per subdivision
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count will be rounded to nearest icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    int validate = 0;
    int printResults = 0;

    int exitNow = 0;
    int exitCode = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                targetTriangles = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                timesteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                sourceIdx = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
                reflectivity = static_cast<val_t>(atof(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-o") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitNow = 1;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitNow = 1;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exitNow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitNow) {
        MPI_Finalize();
        return exitCode;
    }

    MPI_Bcast(&targetTriangles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sourceIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reflectivity, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d\n", size);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    SimulationState state;
    state.mpiRank = rank;
    state.mpiSize = size;

    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    MPI_Barrier(MPI_COMM_WORLD);

    // Precomputation
    double preStart = MPI_Wtime();
    computeTimeDelays(state);
    computeFormFactors(state);
    MPI_Barrier(MPI_COMM_WORLD);
    double preLocalMs = (MPI_Wtime() - preStart) * 1000.0;
    double preMs = 0.0;
    MPI_Reduce(&preLocalMs, &preMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Simulation
    double simStart = MPI_Wtime();
    runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);
    double simLocalMs = (MPI_Wtime() - simStart) * 1000.0;
    double simMs = 0.0;
    MPI_Reduce(&simLocalMs, &simMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Distance computation
    double distStart = MPI_Wtime();
    computeDistances(state);
    MPI_Barrier(MPI_COMM_WORLD);
    double distLocalMs = (MPI_Wtime() - distStart) * 1000.0;
    double distMs = 0.0;
    MPI_Reduce(&distLocalMs, &distMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long preDuration = static_cast<long>(preMs);
        long simDuration = static_cast<long>(simMs);
        long distDuration = static_cast<long>(distMs);
        long totalTime = preDuration + simDuration + distDuration;

        printf("Precomputation time: %ld ms\n", preDuration);
        printf("Simulation time: %ld ms\n", simDuration);
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n) * static_cast<double>(n);
        double simOps = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(t);
        double distOps = static_cast<double>(n) * static_cast<double>(t) * static_cast<double>(t);

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / static_cast<double>(n));

        // Memory usage (distributed Kij/Tau, replicated time-series)
        size_t memRad = 2 * t * n * sizeof(val_t);
        printf("  Replicated time-series memory: %.2f MB\n", memRad / (1024.0 * 1024.0));

        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
    }

    if (validate) {
        if (!validateResults(state)) {
            exitCode = 1;
        }
    }

    freeCudaBuffers(state);

    MPI_Finalize();
    return exitCode;
}
