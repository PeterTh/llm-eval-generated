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
// Room Response Simulation State (Hybrid MPI + OpenMP + CUDA)
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)

    // Row-block distributed storage: only [localCount x N] is stored per rank.
    std::vector<val_t> kijLocal;    // Form factors (localCount x N)
    std::vector<int> tauLocal;      // Time delays (localCount x N)

    // Replicated time-series (we allgather each timestep so every rank has full history).
    std::vector<val_t> radE;        // Emission radiosity (T x N)
    std::vector<val_t> radB;        // Reflected radiosity (T x N)

    std::vector<val_t> distances;   // Computed distances from source (N)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // MPI configuration / 1D row partition of triangles
    int mpiRank = 0;
    int mpiSize = 1;
    size_t localBegin = 0;
    size_t localCount = 0;
    std::vector<int> counts;        // size=mpiSize, counts in elements
    std::vector<int> displs;        // size=mpiSize, displacements in elements

    size_t idx2dLocal(size_t li, size_t j) const { return li * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// MPI partitioning + CUDA helpers
// ============================================================================

static inline void setupPartition(SimulationState& state) {
    state.counts.assign(state.mpiSize, 0);
    state.displs.assign(state.mpiSize, 0);

    const size_t n = state.numTriangles;
    const size_t base = n / static_cast<size_t>(state.mpiSize);
    const size_t rem  = n % static_cast<size_t>(state.mpiSize);

    size_t off = 0;
    for (int r = 0; r < state.mpiSize; ++r) {
        const size_t cnt = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        state.counts[r] = static_cast<int>(cnt);
        state.displs[r] = static_cast<int>(off);
        off += cnt;
    }

    state.localBegin = static_cast<size_t>(state.displs[state.mpiRank]);
    state.localCount = static_cast<size_t>(state.counts[state.mpiRank]);

    state.kijLocal.assign(state.localCount * state.numTriangles, ZERO);
    state.tauLocal.assign(state.localCount * state.numTriangles, 0);
}

static inline uint32_t seedForPair(size_t i, size_t j) {
    // Deterministic per-(i,j) RNG seed for parallel form-factor computation.
    uint64_t x = 42ULL;
    x ^= (uint64_t)i * 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t)j * 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (uint32_t)x;
}

#define CUDA_CHECK(call) do {     cudaError_t _e = (call);     if (_e != cudaSuccess) {         fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));         MPI_Abort(MPI_COMM_WORLD, 1);     } } while (0)


__inline__ __device__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float blockReduceSum(float v) {
    static __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    v = warpReduceSum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) v = warpReduceSum(v);
    return v;
}

__global__ void simulateTimestepKernel(
    int t,
    size_t n,
    size_t localBegin,
    const int* __restrict__ tauLocal,
    const float* __restrict__ kijLocal,
    const float* __restrict__ areas,
    const float* __restrict__ rho,
    const float* __restrict__ radE,
    float* __restrict__ radB)
{
    const size_t li = static_cast<size_t>(blockIdx.x);
    const size_t gi = localBegin + li;

    const int* tauRow = tauLocal + li * n;
    const float* kijRow = kijLocal + li * n;

    float sum = 0.0f;
    for (size_t j = static_cast<size_t>(threadIdx.x); j < n; j += static_cast<size_t>(blockDim.x)) {
        if (j == gi) continue;
        const int tauij = tauRow[j];
        if (t < tauij) continue;

        const float kij = kijRow[j];
        if (kij <= 0.0f) continue;

        const int srcTime = t - tauij;
        const float radJ = radB[static_cast<size_t>(srcTime) * n + j];
        if (radJ <= 0.0f) continue;

        float ff = kij * areas[j];
        if (ff > 1.0f) ff = 1.0f;
        sum += ff * radJ;
    }

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        radB[static_cast<size_t>(t) * n + gi] = rho[gi] * sum + radE[static_cast<size_t>(t) * n + gi];
    }
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

    if (state.mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (state.mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // MPI partition and distributed matrices
    setupPartition(state);

    // Initialize time-series and outputs (replicated)
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    const size_t timeOn = 0;
    const size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}


// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij)...\n");

    const size_t n = state.numTriangles;
    const size_t begin = state.localBegin;
    const size_t end = begin + state.localCount;

    #pragma omp parallel for schedule(dynamic)
    for (size_t gi = begin; gi < end; ++gi) {
        const size_t li = gi - begin;
        for (size_t j = 0; j < n; ++j) {
            if (gi == j) continue;
            RandomGenerator rng(seedForPair(gi, j));
            state.kijLocal[state.idx2dLocal(li, j)] = computeKij(
                gi, j, state.triangles, state.octree, rng);
        }
        if (state.mpiRank == 0 && ((gi - begin + 1) % 100 == 0 || gi + 1 == end)) {
            printf("  Progress (rank0 local): %zu/%zu triangles\n", gi - begin + 1, state.localCount);
        }
    }
}


void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau)...\n");

    const size_t n = state.numTriangles;
    const size_t begin = state.localBegin;
    const size_t end = begin + state.localCount;

    #pragma omp parallel for schedule(static)
    for (size_t gi = begin; gi < end; ++gi) {
        const size_t li = gi - begin;
        for (size_t j = 0; j < n; ++j) {
            if (gi == j) continue;
            state.tauLocal[state.idx2dLocal(li, j)] = computeTau(
                state.triangles[gi], state.triangles[j]);
        }
    }
}


// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Device selection (one GPU per rank, round-robin).
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (state.mpiRank == 0) fprintf(stderr, "ERROR: No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(state.mpiRank % deviceCount));

    float *d_areas = nullptr, *d_rho = nullptr, *d_radE = nullptr, *d_radB = nullptr;
    float *d_kij = nullptr;
    int *d_tau = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_areas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_rho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_kij, state.kijLocal.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_tau, state.tauLocal.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_radE, (T * n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_radB, (T * n) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kij, state.kijLocal.data(), state.kijLocal.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, state.tauLocal.data(), state.tauLocal.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), (T * n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_radB, 0, (T * n) * sizeof(float)));

    std::vector<val_t> radBLocal(state.localCount, ZERO);

    const int threads = 256;
    const dim3 block(threads);
    const dim3 grid(static_cast<unsigned int>(state.localCount));

    for (size_t t = 0; t < T; ++t) {
        simulateTimestepKernel<<<grid, block>>>(
            static_cast<int>(t), n, state.localBegin,
            d_tau, d_kij, d_areas, d_rho, d_radE, d_radB);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Gather full radiosity for this timestep across ranks.
        CUDA_CHECK(cudaMemcpy(radBLocal.data(), d_radB + t * n + state.localBegin,
                              state.localCount * sizeof(float), cudaMemcpyDeviceToHost));

        val_t* radBt = state.radB.data() + state.idxTN(t, 0);
        MPI_Allgatherv(radBLocal.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                       radBt, state.counts.data(), state.displs.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        // Copy the fully gathered timestep back to device so future timesteps can read it.
        CUDA_CHECK(cudaMemcpy(d_radB + t * n, radBt, n * sizeof(float), cudaMemcpyHostToDevice));

        if (state.mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_radB));
}


// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;

    std::vector<val_t> localDistances(state.localCount, ZERO);

    #pragma omp parallel for schedule(static)
    for (size_t gi = state.localBegin; gi < state.localBegin + state.localCount; ++gi) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;
            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, gi)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        localDistances[gi - state.localBegin] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    MPI_Allgatherv(localDistances.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                   state.distances.data(), state.counts.data(), state.displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
}


// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    if (state.mpiRank == 0) printf("\nValidation:\n");

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
            if (state.mpiRank == 0) printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            if (state.mpiRank == 0) printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    if (state.mpiRank == 0) printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    if (state.mpiRank == 0) printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    if (state.mpiRank == 0) printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        if (state.mpiRank == 0) printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    if (state.mpiRank == 0) printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        if (state.mpiRank == 0) printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }
    // Check Kij matrix (distributed): should have some non-zero entries
    int localNonZeroKij = 0;
    for (size_t i = 0; i < state.kijLocal.size(); ++i) {
        if (state.kijLocal[i] > EPSILON) localNonZeroKij++;
    }
    int nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (state.mpiRank == 0) {
        printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
               nonZeroKij, state.numTriangles * state.numTriangles,
               100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));
    }

    if (nonZeroKij == 0) {
        if (state.mpiRank == 0) {
            printf("  ERROR: All form factors are zero - visibility computation failed\n");
        }
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    if (state.mpiRank == 0) printf("  Validation: PASSED\n");
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) printf("Room Response Simulation Benchmark\n");
    if (mpiRank == 0) printf("===================================\n");
    if (mpiRank == 0) printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    if (mpiRank == 0) printf("Timesteps: %d\n", timesteps);
    if (mpiRank == 0) printf("Source triangle: %d\n", sourceIdx);
    if (mpiRank == 0) printf("Reflectivity: %.2f\n", reflectivity);
    if (mpiRank == 0) printf("Validation: %s\n", validate ? "enabled" : "disabled");
    if (mpiRank == 0) printf("\n");

    // Initialize
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (mpiRank == 0) printf("\n");

    long preMax = 0, simMax = 0, distMax = 0;

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    MPI_Allreduce(&preDuration, &preMax, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (mpiRank == 0) printf("Precomputation time: %ld ms\n", preMax);
    if (mpiRank == 0) printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    MPI_Allreduce(&simDuration, &simMax, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (mpiRank == 0) printf("Simulation time: %ld ms\n", simMax);
    if (mpiRank == 0) printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    MPI_Allreduce(&distDuration, &distMax, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (mpiRank == 0) printf("Distance computation time: %ld ms\n", distMax);
    if (mpiRank == 0) printf("\n");

    // Total time
    long totalTime = preMax + simMax + distMax;
    if (mpiRank == 0) printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    if (mpiRank == 0) printf("\nPerformance:\n");
    if (mpiRank == 0) printf("  Triangles: %zu\n", n);
    if (mpiRank == 0) printf("  Timesteps: %zu\n", t);
    if (mpiRank == 0) printf("  Form factor computations: %.2e\n", kijOps);
    if (mpiRank == 0) printf("  Simulation operations: %.2e\n", simOps);
    if (mpiRank == 0) printf("  Distance computations: %.2e\n", distOps);
    if (mpiRank == 0) printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);    // Memory usage (per-rank for distributed Kij/Tau)
    size_t memKij = state.kijLocal.size() * sizeof(val_t);
    size_t memTau = state.tauLocal.size() * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    if (mpiRank == 0) printf("  Memory usage (per-rank): %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    if (mpiRank == 0) printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults && mpiRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    if (validate) {
        int ok = validateResults(state) ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!allOk) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
