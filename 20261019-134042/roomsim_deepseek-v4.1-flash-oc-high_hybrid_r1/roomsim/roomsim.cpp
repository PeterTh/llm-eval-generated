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
 *
 * Parallelization: MPI distributes destination rows of the form-factor and
 * radiosity matrices across ranks, OpenMP parallelizes the CPU visibility /
 * form-factor work inside each rank, and CUDA executes the radiosity wave
 * propagation and cross-correlation kernels on each rank's GPU.
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
#include <utility>
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

// Each non-culled pair consumes two random points per ray, and each random
// point consumes two uniform real numbers.
constexpr size_t DRAWS_PER_PAIR = static_cast<size_t>(NUM_RAYS) * 4;

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
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void skip(unsigned long long n) { rng.discard(n); }
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
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)

    // Locally-owned rows of the distributed Kij / Tau matrices.
    std::vector<val_t> kij;         // localRows x N, row-major
    std::vector<int> tau;           // localRows x N, row-major

    std::vector<val_t> radB;        // Full emission/reflection matrix (T x N)
    std::vector<val_t> radBT;       // Transposed radiosity (N x T) for correlation
    std::vector<val_t> distances;   // Full distance vector (valid on rank 0)
    std::vector<val_t> localDistances;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t rowBegin = 0;            // First globally-owned row
    size_t localRows = 0;           // Number of locally-owned rows

    // GPU buffers
    val_t* d_kij = nullptr;
    int* d_tau = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_radBT = nullptr;

    size_t idxLocal(size_t l, size_t j) const { return l * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int size) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree for spatial acceleration
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Partition destination rows across MPI ranks
    const size_t n = state.numTriangles;
    const size_t base = n / static_cast<size_t>(size);
    const size_t extra = n % static_cast<size_t>(size);
    state.localRows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    state.rowBegin = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);

    state.kij.assign(state.localRows * n, ZERO);
    state.tau.assign(state.localRows * n, 0);
    state.radB.assign(timesteps * n, ZERO);
    state.radBT.assign(n * timesteps, ZERO);
    state.localDistances.assign(state.localRows, ZERO);
    if (rank == 0) {
        state.distances.assign(n, ZERO);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

static inline bool isCulledPair(const Triangle& a, const Triangle& b) {
    return a.normal().dot(b.normal()) > 0.99f;
}

// Build per-row counts and exclusive prefix sums of non-culled pairs.
// This lets every worker position the original sequential RNG stream exactly.
void buildCullPrefix(const SimulationState& state,
                     std::vector<size_t>& rowCounts,
                     std::vector<size_t>& rowPrefix) {
    const size_t n = state.numTriangles;
    rowCounts.assign(n + 1, 0);
    rowPrefix.assign(n + 1, 0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        size_t count = 0;
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            if (!isCulledPair(state.triangles[i], state.triangles[j])) ++count;
        }
        rowCounts[i] = count;
    }

    for (size_t i = 0; i < n; ++i) {
        rowPrefix[i + 1] = rowPrefix[i] + rowCounts[i];
    }
}

// Number of random draws consumed before flattened pair index k.
size_t drawsBeforePair(const SimulationState& state, size_t k,
                       const std::vector<size_t>& rowPrefix) {
    const size_t n = state.numTriangles;
    if (n <= 1) return 0;

    const size_t perRow = n - 1;
    const size_t i = k / perRow;
    const size_t rem = k % perRow;

    size_t nonCulled = rowPrefix[i];
    for (size_t p = 0; p < rem; ++p) {
        const size_t j = (p < i) ? p : p + 1;
        if (!isCulledPair(state.triangles[i], state.triangles[j])) ++nonCulled;
    }
    return nonCulled * DRAWS_PER_PAIR;
}

void computeTimeDelays(SimulationState& state) {
    const size_t n = state.numTriangles;
    const size_t localRows = state.localRows;
    const size_t rowBegin = state.rowBegin;

    #pragma omp parallel for schedule(static)
    for (size_t l = 0; l < localRows; ++l) {
        const size_t i = rowBegin + l;
        const Triangle& triI = state.triangles[i];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            state.tau[state.idxLocal(l, j)] = computeTau(triI, state.triangles[j]);
        }
    }
}

void computeFormFactors(SimulationState& state, int rank) {
    if (rank == 0) {
        printf("Computing form factors (Kij)...\n");
    }

    const size_t n = state.numTriangles;
    if (n <= 1 || state.localRows == 0) return;

    std::vector<size_t> rowCounts, rowPrefix;
    buildCullPrefix(state, rowCounts, rowPrefix);

    const size_t perRow = n - 1;
    const size_t rankStart = state.rowBegin * perRow;
    const size_t rankEnd = (state.rowBegin + state.localRows) * perRow;
    const int numThreads = omp_get_max_threads();

    #pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();
        const size_t start = rankStart + (rankEnd - rankStart) * static_cast<size_t>(tid) / static_cast<size_t>(numThreads);
        const size_t end = rankStart + (rankEnd - rankStart) * static_cast<size_t>(tid + 1) / static_cast<size_t>(numThreads);

        if (start < end) {
            RandomGenerator rng(42);
            rng.skip(static_cast<unsigned long long>(drawsBeforePair(state, start, rowPrefix)));

            size_t i = start / perRow;
            size_t rem = start % perRow;
            for (size_t k = start; k < end; ++k) {
                const size_t j = (rem < i) ? rem : rem + 1;
                state.kij[state.idxLocal(i - state.rowBegin, j)] =
                    computeKij(i, j, state.triangles, state.octree, rng);

                if (++rem == perRow) {
                    rem = 0;
                    ++i;
                }
            }
        }
    }

    if (rank == 0) {
        printf("Form factor computation complete.\n");
    }
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void simulateStepKernel(const val_t* __restrict__ kij,
                                   const int* __restrict__ tau,
                                   const val_t* __restrict__ areas,
                                   const val_t* __restrict__ rho,
                                   val_t* __restrict__ radB,
                                   int n, int t, int rowBegin, int localRows,
                                   int sourceIndex, int timeOff) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= localRows) return;

    const int i = rowBegin + l;
    const size_t kRow = static_cast<size_t>(l) * static_cast<size_t>(n);
    const size_t tRow = static_cast<size_t>(l) * static_cast<size_t>(n);

    val_t sumB = ZERO;
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        const int tauij = tau[tRow + static_cast<size_t>(j)];
        if (t < tauij) continue;

        const val_t kijv = kij[kRow + static_cast<size_t>(j)];
        if (kijv <= ZERO) continue;

        const size_t srcTime = static_cast<size_t>(t) - static_cast<size_t>(tauij);
        const val_t radJ = radB[srcTime * static_cast<size_t>(n) + static_cast<size_t>(j)];
        if (radJ <= ZERO) continue;

        sumB += fminf(kijv * areas[j], ONE) * radJ;
    }

    const val_t emission = (i == sourceIndex && t < timeOff) ? ONE : ZERO;
    radB[static_cast<size_t>(t) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
        rho[i] * sumB + emission;
}

__global__ void correlateKernel(const val_t* __restrict__ radBT,
                                const val_t* __restrict__ src,
                                int rowBegin,
                                int numTimesteps,
                                int numChunks,
                                val_t* __restrict__ outVal,
                                int* __restrict__ outTime) {
    __shared__ val_t sVal[256];
    __shared__ int sTime[256];

    const int tid = threadIdx.x;
    const int l = static_cast<int>(blockIdx.x);
    const int i = rowBegin + l;
    const int chunk = blockIdx.y;
    const int t = chunk * blockDim.x + tid;

    val_t sum = -INFINITY;
    if (t < numTimesteps) {
        const val_t* b = radBT + static_cast<size_t>(i) * static_cast<size_t>(numTimesteps);
        sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            sum += src[tt - t] * b[tt];
        }
    }

    sVal[tid] = sum;
    sTime[tid] = t;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const val_t a = sVal[tid];
            const val_t c = sVal[tid + stride];
            const int ta = sTime[tid];
            const int tc = sTime[tid + stride];
            if (c > a || (c == a && tc < ta)) {
                sVal[tid] = c;
                sTime[tid] = tc;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const size_t pos = static_cast<size_t>(l) * static_cast<size_t>(numChunks) + static_cast<size_t>(chunk);
        outVal[pos] = sVal[0];
        outTime[pos] = sTime[0];
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void uploadToDevice(SimulationState& state) {
    const size_t n = state.numTriangles;
    const size_t localEntries = std::max<size_t>(state.localRows * n, 1);

    cudaMalloc(&state.d_kij, localEntries * sizeof(val_t));
    cudaMalloc(&state.d_tau, localEntries * sizeof(int));
    cudaMalloc(&state.d_areas, n * sizeof(val_t));
    cudaMalloc(&state.d_rho, n * sizeof(val_t));
    cudaMalloc(&state.d_radB, state.numTimesteps * n * sizeof(val_t));
    cudaMalloc(&state.d_radBT, state.numTimesteps * n * sizeof(val_t));

    if (state.localRows > 0) {
        cudaMemcpy(state.d_kij, state.kij.data(), state.localRows * n * sizeof(val_t), cudaMemcpyHostToDevice);
        cudaMemcpy(state.d_tau, state.tau.data(), state.localRows * n * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_rho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemset(state.d_radB, 0, state.numTimesteps * n * sizeof(val_t));
}

void runSimulation(SimulationState& state, const std::vector<int>& counts,
                   const std::vector<int>& displs, int rank) {
    const int n = static_cast<int>(state.numTriangles);
    const int timesteps = static_cast<int>(state.numTimesteps);
    const int rowBegin = static_cast<int>(state.rowBegin);
    const int localRows = static_cast<int>(state.localRows);
    const int timeOff = timesteps / 2;
    const int sourceIndex = static_cast<int>(state.sourceIndex);
    const int threads = 256;
    const int blocks = (localRows + threads - 1) / threads;

    val_t* hLocal = nullptr;
    val_t* hRow = nullptr;
    cudaMallocHost(reinterpret_cast<void**>(&hLocal), std::max<size_t>(state.localRows, 1) * sizeof(val_t));
    cudaMallocHost(reinterpret_cast<void**>(&hRow), state.numTriangles * sizeof(val_t));

    for (int t = 0; t < timesteps; ++t) {
        if (localRows > 0) {
            simulateStepKernel<<<blocks, threads>>>(
                state.d_kij, state.d_tau, state.d_areas, state.d_rho, state.d_radB,
                n, t, rowBegin, localRows, sourceIndex, timeOff);
        }

        if (localRows > 0) {
            cudaMemcpy(hLocal, state.d_radB + static_cast<size_t>(t) * static_cast<size_t>(n) + state.rowBegin,
                       state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost);
        }

        MPI_Allgatherv(hLocal, localRows, MPI_FLOAT, hRow,
                       counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        std::memcpy(state.radB.data() + static_cast<size_t>(t) * state.numTriangles,
                    hRow, state.numTriangles * sizeof(val_t));
        cudaMemcpy(state.d_radB + static_cast<size_t>(t) * static_cast<size_t>(n), hRow,
                   state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice);

        if (rank == 0 && ((t + 1) % 100 == 0 || t + 1 == timesteps)) {
            printf("  Timestep %d/%d\n", t + 1, timesteps);
        }
    }

    cudaFreeHost(hLocal);
    cudaFreeHost(hRow);

    // Transpose the radiosity history for the coalesced correlation kernel.
    const size_t T = state.numTimesteps;
    const size_t N = state.numTriangles;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const val_t* src = state.radB.data() + i;
        val_t* dst = state.radBT.data() + i * T;
        for (size_t t = 0; t < T; ++t) {
            dst[t] = src[t * N];
        }
    }
    cudaMemcpy(state.d_radBT, state.radBT.data(), N * T * sizeof(val_t), cudaMemcpyHostToDevice);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, const std::vector<int>& counts,
                      const std::vector<int>& displs, int rank) {
    const int T = static_cast<int>(state.numTimesteps);
    const int localRows = static_cast<int>(state.localRows);
    const int threads = 256;
    const int numChunks = (T + threads - 1) / threads;

    val_t* d_partialVal = nullptr;
    int* d_partialTime = nullptr;
    const size_t partialCount = std::max<size_t>(static_cast<size_t>(localRows) * static_cast<size_t>(numChunks), 1);

    if (localRows > 0) {
        cudaMalloc(&d_partialVal, partialCount * sizeof(val_t));
        cudaMalloc(&d_partialTime, partialCount * sizeof(int));

        const val_t* src = state.d_radBT + state.sourceIndex * state.numTimesteps;
        dim3 grid(static_cast<unsigned int>(localRows), static_cast<unsigned int>(numChunks));
        correlateKernel<<<grid, threads>>>(state.d_radBT, src, static_cast<int>(state.rowBegin), T, numChunks,
                                           d_partialVal, d_partialTime);

        std::vector<val_t> partialVal(partialCount);
        std::vector<int> partialTime(partialCount);
        cudaMemcpy(partialVal.data(), d_partialVal, partialCount * sizeof(val_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(partialTime.data(), d_partialTime, partialCount * sizeof(int), cudaMemcpyDeviceToHost);

        #pragma omp parallel for schedule(static)
        for (int l = 0; l < localRows; ++l) {
            val_t bestVal = -INFINITY;
            int bestT = 0;
            const size_t base = static_cast<size_t>(l) * static_cast<size_t>(numChunks);
            for (int c = 0; c < numChunks; ++c) {
                const val_t v = partialVal[base + static_cast<size_t>(c)];
                if (v > bestVal) {
                    bestVal = v;
                    bestT = partialTime[base + static_cast<size_t>(c)];
                }
            }
            if (bestVal <= ZERO) bestT = 0;
            state.localDistances[l] = WAVE_SPEED * static_cast<val_t>(bestT);
        }

        cudaFree(d_partialVal);
        cudaFree(d_partialTime);
    }

    MPI_Gatherv(state.localDistances.data(), localRows, MPI_FLOAT,
                (rank == 0 ? state.distances.data() : nullptr),
                counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, size_t nonZeroKijGlobal) {
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
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

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
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    printf("  Non-zero form factors: %zu/%zu (%.2f%%)\n",
           nonZeroKijGlobal, state.numTriangles * state.numTriangles,
           100.0f * static_cast<val_t>(nonZeroKijGlobal) / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKijGlobal == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int localRank = 0;
    MPI_Comm sharedComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &sharedComm);
    MPI_Comm_rank(sharedComm, &localRank);
    MPI_Comm_free(&sharedComm);

    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "CUDA device unavailable\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaSetDevice(localRank % deviceCount);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank, size);

    // Row ownership used by all collectives
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t base = state.numTriangles / static_cast<size_t>(size);
        const size_t extra = state.numTriangles % static_cast<size_t>(size);
        counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < extra ? 1 : 0));
        displs[r] = static_cast<int>(static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra));
    }

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double startPre = MPI_Wtime();

    computeTimeDelays(state);
    computeFormFactors(state, rank);

    double endPre = MPI_Wtime();
    long preDuration = static_cast<long>((endPre - startPre) * 1000.0);
    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    uploadToDevice(state);

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double startSim = MPI_Wtime();

    if (rank == 0) printf("Running wave propagation simulation...\n");
    runSimulation(state, counts, displs, rank);

    double endSim = MPI_Wtime();
    long simDuration = static_cast<long>((endSim - startSim) * 1000.0);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    double startDist = MPI_Wtime();

    if (rank == 0) printf("Computing distances via cross-correlation...\n");
    computeDistances(state, counts, displs, rank);

    cudaDeviceSynchronize();
    double endDist = MPI_Wtime();
    double localElapsed = endDist - startPre;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long distDuration = static_cast<long>((endDist - startDist) * 1000.0);
    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    long long localNonZeroKij = 0;
    if (state.localRows > 0) {
        #pragma omp parallel for reduction(+:localNonZeroKij) schedule(static)
        for (size_t idx = 0; idx < state.kij.size(); ++idx) {
            if (state.kij[idx] > EPSILON) localNonZeroKij++;
        }
    }
    long long nonZeroKijGlobal = 0;
    MPI_Reduce(&localNonZeroKij, &nonZeroKijGlobal, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        // Total time
        long totalTime = static_cast<long>(maxElapsed * 1000.0);
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = static_cast<double>(n * n * t);
        double distOps = static_cast<double>(n * t * t);

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage
        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        // Hash
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        // Validation
        if (validate) {
            if (!validateResults(state, static_cast<size_t>(nonZeroKijGlobal))) {
                rc = 1;
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (state.d_kij) cudaFree(state.d_kij);
    if (state.d_tau) cudaFree(state.d_tau);
    if (state.d_areas) cudaFree(state.d_areas);
    if (state.d_rho) cudaFree(state.d_rho);
    if (state.d_radB) cudaFree(state.d_radB);
    if (state.d_radBT) cudaFree(state.d_radBT);

    MPI_Finalize();
    return rc;
}
