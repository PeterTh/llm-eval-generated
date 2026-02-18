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
// Hybrid MPI + OpenMP + CUDA Helpers
// ============================================================================

static inline void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

static inline void mpiCheck(int e, const char* what) {
    if (e != MPI_SUCCESS) {
        char err[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(e, err, &len);
        fprintf(stderr, "MPI error (%s): %.*s\n", what, len, err);
        std::abort();
    }
}

static inline void decompose1D(size_t n, int rank, int nranks, size_t& start, size_t& count) {
    size_t base = n / static_cast<size_t>(nranks);
    size_t rem = n % static_cast<size_t>(nranks);
    count = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    start = base * static_cast<size_t>(rank) + std::min(rem, static_cast<size_t>(rank));
}

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
    void discard(uint64_t n) { rng.discard(static_cast<unsigned long long>(n)); }
};

static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline val_t rand01(uint32_t x) {
    // 24-bit mantissa uniform in [0,1)
    return static_cast<val_t>((hash32(x) >> 8) * (1.0f / 16777216.0f));
}

// Generate a deterministic "random" point inside a triangle using barycentric coordinates
static inline Vec3 randomPointInTriangle(const Triangle& t, uint32_t seedBase) {
    val_t u = rand01(seedBase);
    val_t v = rand01(seedBase ^ 0x9e3779b9U);
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
                  const Octree& octree) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    const uint32_t base = 42u ^ (static_cast<uint32_t>(idxI) * 0x85ebca6bU) ^ (static_cast<uint32_t>(idxJ) * 0xc2b2ae35U);

    for (int r = 0; r < NUM_RAYS; ++r) {
        const uint32_t s = base ^ (static_cast<uint32_t>(r) * 0x27d4eb2dU);
        Vec3 pI = randomPointInTriangle(triI, s);
        Vec3 pJ = randomPointInTriangle(triJ, s ^ 0x165667b1U);

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
    // MPI decomposition (1D block over receiver triangle index i)
    int mpiRank = 0;
    int mpiSize = 1;
    size_t localStart = 0;
    size_t localCount = 0;

    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;                 // N
    std::vector<val_t> rho;                   // N

    std::vector<val_t> kijLocal;              // (localCount x N) row-major
    std::vector<int>   tauLocal;              // (localCount x N) row-major

    std::vector<val_t> radBLocal;             // (T x localCount) row-major
    std::vector<val_t> sourceSeries;          // (T) source radiosity time-series (replicated)
    std::vector<val_t> distancesLocal;        // localCount

    Octree octree;

    size_t sourceIndex = 0;

    size_t idx2dLocal(size_t li, size_t j) const { return li * numTriangles + j; }
    size_t idxTL(size_t t, size_t li) const { return t * localCount + li; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (replicated across MPI ranks)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    decompose1D(state.numTriangles, state.mpiRank, state.mpiSize, state.localStart, state.localCount);

    if (state.mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }
    state.octree.build(state.triangles);

    state.areas.resize(state.numTriangles);
    state.rho.assign(state.numTriangles, reflectivity);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.kijLocal.assign(state.localCount * state.numTriangles, ZERO);
    state.tauLocal.assign(state.localCount * state.numTriangles, 0);
    state.radBLocal.assign(state.numTimesteps * state.localCount, ZERO);
    state.sourceSeries.assign(state.numTimesteps, ZERO);
    state.distancesLocal.assign(state.localCount, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij) with MPI+OpenMP...\n");

    const size_t N = state.numTriangles;

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t li = 0; li < state.localCount; ++li) {
        const size_t i = state.localStart + li;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            state.kijLocal[state.idx2dLocal(li, j)] = computeKij(i, j, state.triangles, state.octree);
        }
    }
}

int computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau) with MPI+OpenMP...\n");

    const size_t N = state.numTriangles;
    int localMaxTau = 0;

    #pragma omp parallel for schedule(static) reduction(max:localMaxTau)
    for (size_t li = 0; li < state.localCount; ++li) {
        const size_t i = state.localStart + li;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const int tauij = computeTau(state.triangles[i], state.triangles[j]);
            state.tauLocal[state.idx2dLocal(li, j)] = tauij;
            localMaxTau = std::max(localMaxTau, tauij);
        }
    }

    int globalMaxTau = 0;
    mpiCheck(MPI_Allreduce(&localMaxTau, &globalMaxTau, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), "Allreduce maxTau");
    return globalMaxTau;
}

// ============================================================================
// Simulation Phase (Wave Propagation): MPI (distributed i) + CUDA (local rows)
// ============================================================================

__global__ void sim_step_kernel(
    int t,
    int N,
    int localStart,
    int localCount,
    const float* __restrict__ kijLocal,
    const int* __restrict__ tauLocal,
    const float* __restrict__ areas,
    const float* __restrict__ rhoLocal,
    const float* __restrict__ globalRing,
    int ringLen,
    int sourceIndex,
    int timeOff,
    float* __restrict__ outLocalSlice,
    float* __restrict__ radBLocalHist)
{
    const int li = static_cast<int>(blockIdx.x);
    if (li >= localCount) return;
    const int iGlobal = localStart + li;

    float sum = 0.0f;
    const int row = li * N;

    for (int j = static_cast<int>(threadIdx.x); j < N; j += static_cast<int>(blockDim.x)) {
        if (j == iGlobal) continue;
        const int tauij = tauLocal[row + j];
        if (t < tauij) continue;
        const int srcTime = t - tauij;
        const float radJ = globalRing[(srcTime % ringLen) * N + j];
        if (radJ <= 0.0f) continue;
        float w = kijLocal[row + j] * areas[j];
        w = (w < 1.0f) ? w : 1.0f;
        sum += w * radJ;
    }

    __shared__ float shm[256];
    shm[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) shm[threadIdx.x] += shm[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float emit = (iGlobal == sourceIndex && t < timeOff) ? 1.0f : 0.0f;
        const float b = rhoLocal[li] * shm[0] + emit;
        outLocalSlice[li] = b;
        radBLocalHist[t * localCount + li] = b;
    }
}

__global__ void dist_kernel(
    int T,
    int localCount,
    const float* __restrict__ radBLocalHist,
    const float* __restrict__ sourceSeries,
    float* __restrict__ distancesLocal)
{
    const int li = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (li >= localCount) return;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int dt = 0; dt < T; ++dt) {
        float sum = 0.0f;
        for (int tt = dt; tt < T; ++tt) {
            const float pB = radBLocalHist[tt * localCount + li];
            const float pS = sourceSeries[tt - dt];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = dt;
        }
    }

    distancesLocal[li] = WAVE_SPEED * static_cast<float>(bestT);
}

void runSimulation(SimulationState& state, int globalMaxTau) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation with MPI+CUDA...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int localCount = static_cast<int>(state.localCount);
    const int localStart = static_cast<int>(state.localStart);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    cudaCheck(cudaSetDevice(deviceCount ? (state.mpiRank % deviceCount) : 0), "cudaSetDevice");

    // Allgather layout for per-timestep radiosity slices.
    std::vector<int> recvCounts(state.mpiSize);
    {
        int lc = localCount;
        mpiCheck(MPI_Allgather(&lc, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD), "Allgather counts");
    }
    std::vector<int> displs(state.mpiSize, 0);
    for (int r = 1; r < state.mpiSize; ++r) displs[r] = displs[r - 1] + recvCounts[r - 1];

    const int maxNeeded = std::min(globalMaxTau + 1, T);
    const int ringLen = std::max(1, maxNeeded);

    // Host pinned buffers for MPI communication.
    float* hLocal = nullptr;
    float* hGlobal = nullptr;
    cudaCheck(cudaHostAlloc(&hLocal, static_cast<size_t>(localCount) * sizeof(float), cudaHostAllocPortable), "cudaHostAlloc hLocal");
    cudaCheck(cudaHostAlloc(&hGlobal, static_cast<size_t>(N) * sizeof(float), cudaHostAllocPortable), "cudaHostAlloc hGlobal");

    // Device buffers.
    float* dKij = nullptr;
    int*   dTau = nullptr;
    float* dAreas = nullptr;
    float* dRhoLocal = nullptr;
    float* dGlobalRing = nullptr;
    float* dOutLocal = nullptr;
    float* dRadBHist = nullptr;

    cudaCheck(cudaMalloc(&dKij, state.kijLocal.size() * sizeof(float)), "cudaMalloc kij");
    cudaCheck(cudaMalloc(&dTau, state.tauLocal.size() * sizeof(int)), "cudaMalloc tau");
    cudaCheck(cudaMalloc(&dAreas, state.areas.size() * sizeof(float)), "cudaMalloc areas");
    cudaCheck(cudaMalloc(&dRhoLocal, static_cast<size_t>(localCount) * sizeof(float)), "cudaMalloc rhoLocal");
    cudaCheck(cudaMalloc(&dGlobalRing, static_cast<size_t>(ringLen) * static_cast<size_t>(N) * sizeof(float)), "cudaMalloc globalRing");
    cudaCheck(cudaMalloc(&dOutLocal, static_cast<size_t>(localCount) * sizeof(float)), "cudaMalloc outLocal");
    cudaCheck(cudaMalloc(&dRadBHist, state.radBLocal.size() * sizeof(float)), "cudaMalloc radBLocal");

    cudaCheck(cudaMemcpy(dKij, state.kijLocal.data(), state.kijLocal.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy kij");
    cudaCheck(cudaMemcpy(dTau, state.tauLocal.data(), state.tauLocal.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy tau");
    cudaCheck(cudaMemcpy(dAreas, state.areas.data(), state.areas.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy areas");

    std::vector<float> rhoLocal(state.localCount);
    for (size_t li = 0; li < state.localCount; ++li) rhoLocal[li] = state.rho[state.localStart + li];
    cudaCheck(cudaMemcpy(dRhoLocal, rhoLocal.data(), rhoLocal.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy rhoLocal");

    cudaCheck(cudaMemset(dGlobalRing, 0, static_cast<size_t>(ringLen) * static_cast<size_t>(N) * sizeof(float)), "memset globalRing");

    const int timeOff = T / 2;
    constexpr int THREADS = 256;

    for (int t = 0; t < T; ++t) {
        sim_step_kernel<<<static_cast<unsigned int>(localCount), THREADS>>>(
            t, N, localStart, localCount,
            dKij, dTau, dAreas, dRhoLocal,
            dGlobalRing, ringLen,
            static_cast<int>(state.sourceIndex), timeOff,
            dOutLocal, dRadBHist);
        cudaCheck(cudaGetLastError(), "sim_step_kernel launch");

        cudaCheck(cudaMemcpy(hLocal, dOutLocal, static_cast<size_t>(localCount) * sizeof(float), cudaMemcpyDeviceToHost), "copy outLocal");

        mpiCheck(MPI_Allgatherv(hLocal, localCount, MPI_FLOAT, hGlobal, recvCounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD),
                 "Allgatherv radB");

        state.sourceSeries[static_cast<size_t>(t)] = hGlobal[static_cast<int>(state.sourceIndex)];
        cudaCheck(cudaMemcpy(dGlobalRing + (static_cast<size_t>(t % ringLen) * static_cast<size_t>(N)),
                             hGlobal, static_cast<size_t>(N) * sizeof(float), cudaMemcpyHostToDevice),
                  "copy globalRing");
    }

    // Copy full local history back.
    cudaCheck(cudaMemcpy(state.radBLocal.data(), dRadBHist, state.radBLocal.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy radBHist");

    cudaCheck(cudaFree(dKij), "free kij");
    cudaCheck(cudaFree(dTau), "free tau");
    cudaCheck(cudaFree(dAreas), "free areas");
    cudaCheck(cudaFree(dRhoLocal), "free rhoLocal");
    cudaCheck(cudaFree(dGlobalRing), "free globalRing");
    cudaCheck(cudaFree(dOutLocal), "free outLocal");
    cudaCheck(cudaFree(dRadBHist), "free radBHist");

    cudaCheck(cudaFreeHost(hLocal), "freeHost hLocal");
    cudaCheck(cudaFreeHost(hGlobal), "freeHost hGlobal");
}

// ============================================================================
// Distance Computation (Cross-Correlation): CUDA (local i) + MPI gather if needed
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation with CUDA...\n");

    const int T = static_cast<int>(state.numTimesteps);
    const int localCount = static_cast<int>(state.localCount);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    cudaCheck(cudaSetDevice(deviceCount ? (state.mpiRank % deviceCount) : 0), "cudaSetDevice");

    float* dRadBHist = nullptr;
    float* dSource = nullptr;
    float* dDist = nullptr;

    cudaCheck(cudaMalloc(&dRadBHist, state.radBLocal.size() * sizeof(float)), "cudaMalloc radBHist");
    cudaCheck(cudaMalloc(&dSource, state.sourceSeries.size() * sizeof(float)), "cudaMalloc sourceSeries");
    cudaCheck(cudaMalloc(&dDist, state.distancesLocal.size() * sizeof(float)), "cudaMalloc distances");

    cudaCheck(cudaMemcpy(dRadBHist, state.radBLocal.data(), state.radBLocal.size() * sizeof(float), cudaMemcpyHostToDevice), "copy radBHist");
    cudaCheck(cudaMemcpy(dSource, state.sourceSeries.data(), state.sourceSeries.size() * sizeof(float), cudaMemcpyHostToDevice), "copy sourceSeries");

    const int threads = 256;
    const int blocks = (localCount + threads - 1) / threads;
    dist_kernel<<<blocks, threads>>>(T, localCount, dRadBHist, dSource, dDist);
    cudaCheck(cudaGetLastError(), "dist_kernel launch");

    cudaCheck(cudaMemcpy(state.distancesLocal.data(), dDist, state.distancesLocal.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy distances");

    cudaCheck(cudaFree(dRadBHist), "free radBHist");
    cudaCheck(cudaFree(dSource), "free sourceSeries");
    cudaCheck(cudaFree(dDist), "free distances");
}

// ============================================================================
// Validation
// ============================================================================

static void gatherDistancesToRoot(const SimulationState& state, std::vector<val_t>& outDistances) {
    std::vector<int> counts(state.mpiSize);
    {
        int lc = static_cast<int>(state.localCount);
        mpiCheck(MPI_Allgather(&lc, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD), "Allgather dist counts");
    }
    std::vector<int> displs(state.mpiSize, 0);
    for (int r = 1; r < state.mpiSize; ++r) displs[r] = displs[r - 1] + counts[r - 1];

    if (state.mpiRank == 0) outDistances.assign(state.numTriangles, ZERO);

    mpiCheck(MPI_Gatherv(state.distancesLocal.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                         state.mpiRank == 0 ? outDistances.data() : nullptr,
                         counts.data(), displs.data(), MPI_FLOAT,
                         0, MPI_COMM_WORLD),
             "Gatherv distances");
}

bool validateResults(const SimulationState& state) {
    int localAllNonNegative = 1;
    int localAllFinite = 1;
    val_t localMin = std::numeric_limits<val_t>::max();
    val_t localMax = std::numeric_limits<val_t>::lowest();
    double localSum = 0.0;
    int localNonZero = 0;

    for (size_t li = 0; li < state.localCount; ++li) {
        const val_t d = state.distancesLocal[li];
        if (d < 0) localAllNonNegative = 0;
        if (!std::isfinite(d)) localAllFinite = 0;
        localMin = std::min(localMin, d);
        localMax = std::max(localMax, d);
        localSum += static_cast<double>(d);
        if (d > EPSILON) localNonZero++;
    }

    int allNonNegative = 0;
    int allFinite = 0;
    val_t minDist = 0;
    val_t maxDist = 0;
    double sumDist = 0.0;
    int nonZeroCount = 0;

    mpiCheck(MPI_Allreduce(&localAllNonNegative, &allNonNegative, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD), "Allreduce nonneg");
    mpiCheck(MPI_Allreduce(&localAllFinite, &allFinite, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD), "Allreduce finite");
    mpiCheck(MPI_Allreduce(&localMin, &minDist, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD), "Allreduce min");
    mpiCheck(MPI_Allreduce(&localMax, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD), "Allreduce max");
    mpiCheck(MPI_Allreduce(&localSum, &sumDist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "Allreduce sum");
    mpiCheck(MPI_Allreduce(&localNonZero, &nonZeroCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD), "Allreduce nonzero");

    // Source distance
    float srcDistLocal = -1.0f;
    if (state.sourceIndex >= state.localStart && state.sourceIndex < state.localStart + state.localCount) {
        srcDistLocal = state.distancesLocal[state.sourceIndex - state.localStart];
    }
    float srcDist = 0.0f;
    mpiCheck(MPI_Allreduce(&srcDistLocal, &srcDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD), "Allreduce srcDist");

    // Energy reception
    int localReceived = 0;
    for (size_t li = 0; li < state.localCount; ++li) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radBLocal[state.idxTL(t, li)] > EPSILON) {
                localReceived++;
                break;
            }
        }
    }
    int receivedEnergy = 0;
    mpiCheck(MPI_Allreduce(&localReceived, &receivedEnergy, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD), "Allreduce receivedEnergy");

    // Kij sparsity check
    int localNonZeroKij = 0;
    for (size_t k = 0; k < state.kijLocal.size(); ++k) {
        if (state.kijLocal[k] > EPSILON) localNonZeroKij++;
    }
    int nonZeroKij = 0;
    mpiCheck(MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD), "Allreduce nonZeroKij");

    int ok = (allFinite && allNonNegative && receivedEnergy > 0 && nonZeroKij > 0) ? 1 : 0;

    if (state.mpiRank == 0) {
        printf("\nValidation:\n");
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", static_cast<float>(sumDist / static_cast<double>(state.numTriangles)));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
        if (srcDist > WAVE_SPEED * 2) {
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
        }
        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
        printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
               nonZeroKij, state.numTriangles * state.numTriangles,
               100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

        if (!ok) printf("  Validation: FAILED\n");
        else printf("  Validation: PASSED\n");
    }

    mpiCheck(MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast validate");
    return ok != 0;
}

// ============================================================================
// Hash for Verification (distributed XOR reduction)
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t localHash = 0;
    for (size_t li = 0; li < state.localCount; ++li) {
        const size_t i = state.localStart + li;
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distancesLocal[li]);
        localHash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
    }

    uint64_t globalHash = 0;
    mpiCheck(MPI_Reduce(&localHash, &globalHash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD), "Reduce hash");
    return globalHash;
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
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");

    int rank = 0, nranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &nranks), "Comm_size");

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    float reflectivity = 0.8f;
    int validate = 0;
    int printResults = 0;
    int showHelp = 0;
    int argError = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                targetTriangles = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                timesteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                sourceIdx = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
                reflectivity = static_cast<float>(atof(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-o") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                argError = 1;
                break;
            }
        }
    }

    mpiCheck(MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast help");
    mpiCheck(MPI_Bcast(&argError, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast argError");

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (argError) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    mpiCheck(MPI_Bcast(&targetTriangles, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast targetTriangles");
    mpiCheck(MPI_Bcast(&timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast timesteps");
    mpiCheck(MPI_Bcast(&sourceIdx, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast sourceIdx");
    mpiCheck(MPI_Bcast(&reflectivity, 1, MPI_FLOAT, 0, MPI_COMM_WORLD), "Bcast reflectivity");
    mpiCheck(MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast validate");
    mpiCheck(MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast printResults");

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("\n");
    }

    SimulationState state;
    state.mpiRank = rank;
    state.mpiSize = nranks;

    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), static_cast<val_t>(reflectivity));

    // Precomputation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier pre");
    double t0 = MPI_Wtime();
    const int globalMaxTau = computeTimeDelays(state);
    computeFormFactors(state);
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier pre end");
    double t1 = MPI_Wtime();
    double preSec = t1 - t0;
    double preSecMax = 0.0;
    mpiCheck(MPI_Reduce(&preSec, &preSecMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "Reduce pre time");

    // Simulation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier sim");
    t0 = MPI_Wtime();
    runSimulation(state, globalMaxTau);
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier sim end");
    t1 = MPI_Wtime();
    double simSec = t1 - t0;
    double simSecMax = 0.0;
    mpiCheck(MPI_Reduce(&simSec, &simSecMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "Reduce sim time");

    // Distances
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier dist");
    t0 = MPI_Wtime();
    computeDistances(state);
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier dist end");
    t1 = MPI_Wtime();
    double distSec = t1 - t0;
    double distSecMax = 0.0;
    mpiCheck(MPI_Reduce(&distSec, &distSecMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "Reduce dist time");

    // Hash (reduced to rank 0)
    uint64_t hash = computeHash(state);

    if (rank == 0) {
        const long preMs = static_cast<long>(preSecMax * 1000.0);
        const long simMs = static_cast<long>(simSecMax * 1000.0);
        const long distMs = static_cast<long>(distSecMax * 1000.0);
        const long totalMs = preMs + simMs + distMs;

        printf("Precomputation time: %ld ms\n\n", preMs);
        printf("Simulation time: %ld ms\n\n", simMs);
        printf("Distance computation time: %ld ms\n\n", distMs);
        printf("Total computation time: %ld ms\n", totalMs);

        const size_t n = state.numTriangles;
        const size_t tt = state.numTimesteps;
        const double kijOps = static_cast<double>(n) * static_cast<double>(n);
        const double simOps = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(tt);
        const double distOps = static_cast<double>(n) * static_cast<double>(tt) * static_cast<double>(tt);

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", tt);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalMs) / n);

        const size_t memKij = n * n * sizeof(val_t);
        const size_t memTau = n * n * sizeof(int);
        const size_t memRad = 2 * tt * n * sizeof(val_t);
        const size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage (theoretical global): %.2f MB\n", totalMem / (1024.0 * 1024.0));

        printf("  Result hash: %016lX\n\n", hash);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<val_t> distances;
        gatherDistancesToRoot(state, distances);
        if (rank == 0) {
            std::vector<double> distData(distances.begin(), distances.end());
            print_results(distData, "Distances");
        }
    }

    // Validation
    if (validate) {
        const bool ok = validateResults(state);
        if (!ok) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
