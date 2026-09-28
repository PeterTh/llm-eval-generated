/**
 * Room Response Simulation Benchmark
 *
 * Hybrid-parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy (unconditional MPI + OpenMP + CUDA hybrid):
 *  - MPI: triangle rows are block-distributed across ranks; each rank drives
 *    one GPU (ranks round-robin over the visible devices of their node).
 *    Radiosity rows are allgathered once per timestep, distances at the end.
 *  - CUDA: the dominant O(N^2 * rays) form-factor Monte-Carlo (with an
 *    octree flattened into device arrays for visibility tests), the per-
 *    timestep radiosity gather, and the O(N * T^2) cross-correlation all run
 *    as GPU kernels.
 *  - OpenMP: host-side precomputation (time delays, per-triangle areas) and
 *    validation/statistics reductions.
 *
 * The Monte-Carlo sampling replays the exact std::mt19937 stream of the
 * original sequential code: every non-culled (i, j) pair consumes exactly
 * 4 * NUM_RAYS uniforms, so each rank computes its stream offset from the
 * (cheap, deterministic) cull mask, skips ahead, and generates the random
 * numbers for its rows on the host before feeding them to the GPU kernel.
 * Results are therefore identical regardless of rank/thread/GPU counts.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
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

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const {
        return a == o.a && b == o.b && c == o.c;
    }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction, host only)
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
// Octree for Spatial Acceleration (built on host, flattened for the GPU)
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
};

// Flattened octree node for GPU traversal. A node is a leaf iff triCount > 0
// (mirroring the host octree, where only leaves hold triangle indices).
struct GpuNode {
    val_t cx, cy, cz;   // center
    val_t hx, hy, hz;   // half extent
    int child[8];       // child node indices, -1 if absent
    int triStart;       // offset into triangle index array
    int triCount;       // number of triangle indices (0 for internal nodes)
};

static int flattenOctree(const Octree& node, std::vector<GpuNode>& nodes,
                         std::vector<int>& triIdx) {
    int myIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    GpuNode gn{};
    gn.cx = node.center.x; gn.cy = node.center.y; gn.cz = node.center.z;
    gn.hx = node.halfExtent.x; gn.hy = node.halfExtent.y; gn.hz = node.halfExtent.z;
    gn.triStart = static_cast<int>(triIdx.size());
    gn.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t t : node.triangleIndices) triIdx.push_back(static_cast<int>(t));
    for (int i = 0; i < 8; ++i) gn.child[i] = -1;
    nodes[myIdx] = gn;

    for (int i = 0; i < 8; ++i) {
        if (node.children[i]) {
            int c = flattenOctree(*node.children[i], nodes, triIdx);
            nodes[myIdx].child[i] = c;
        }
    }
    return myIdx;
}

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
// Random Number Generation (exact replay of the sequential mt19937 stream)
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }

    // Skip `count` distribution draws. std::uniform_real_distribution<float>
    // consumes exactly one engine draw per call with this engine (verified at
    // runtime), which lets us use the much faster engine discard.
    void skip(uint64_t count) {
        static const bool oneDrawPerCall = [] {
            std::mt19937 a(12345), b(12345);
            std::uniform_real_distribution<val_t> d(0.0f, 1.0f);
            for (int k = 0; k < 1000; ++k) (void)d(a);
            b.discard(1000);
            return a() == b();
        }();
        if (oneDrawPerCall) {
            rng.discard(count);
        } else {
            for (uint64_t k = 0; k < count; ++k) (void)dist(rng);
        }
    }
};

// Number of uniforms consumed per non-culled (i, j) pair: two barycentric
// samples (u, v) per ray endpoint, for both endpoints, per ray.
constexpr uint64_t RANDS_PER_PAIR = 4ULL * NUM_RAYS;

// Cull predicate of computeKij: pairs facing the same direction consume no
// random numbers and have a zero form factor.
inline bool kijPairIsCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Generate a point inside a triangle from two uniform samples (barycentric)
__host__ __device__ inline Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
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

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                               const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (flattened-octree traversal on the GPU)
// ============================================================================

// Segment-box overlap test against a flattened octree node's AABB
__device__ inline bool rayIntersectsNode(const GpuNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 center{n.cx, n.cy, n.cz};
    Vec3 halfExtent{n.hx, n.hy, n.hz};

    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle,
// using an explicit-stack traversal of the flattened octree.
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const GpuNode* nodes, const int* triIdx,
                             const Triangle* triangles,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[96];
    int sp = 0;
    stack[sp++] = 0;  // root

    while (sp > 0) {
        const GpuNode& n = nodes[stack[--sp]];

        if (n.triCount > 0) {
            // Leaf: test triangles directly
            for (int k = 0; k < n.triCount; ++k) {
                int idx = triIdx[n.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            // Internal node: descend into children whose boxes the ray hits
            for (int cIdx = 0; cIdx < 8; ++cIdx) {
                int ci = n.child[cIdx];
                if (ci >= 0 && rayIntersectsNode(nodes[ci], from, to)) {
                    stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) - CUDA kernel
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// One thread per (i, j) pair of a block of rows starting at global row
// blockRowStart. Culled pairs (and the diagonal) carry slot -1 and a zero
// form factor; every other pair reads its RANDS_PER_PAIR pre-generated
// uniforms (the exact sequential mt19937 stream) at slot * RANDS_PER_PAIR.
__global__ void kijKernel(const Triangle* __restrict__ triangles, int numTriangles,
                          int blockRowStart, int blockRows,
                          const GpuNode* __restrict__ nodes,
                          const int* __restrict__ triIdx,
                          const val_t* __restrict__ rands,
                          const int* __restrict__ slots,
                          val_t* __restrict__ kijBlock) {
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(blockRows) * numTriangles;
    if (tid >= total) return;

    int il = static_cast<int>(tid / numTriangles);
    int j = static_cast<int>(tid % numTriangles);
    int i = blockRowStart + il;

    // Diagonal or culled (same-facing) pair
    int slot = slots[tid];
    if (slot < 0) {
        kijBlock[tid] = ZERO;
        return;
    }

    const Triangle triI = triangles[i];
    const Triangle triJ = triangles[j];

    val_t kij = ZERO;
    const val_t* pairRands = rands + static_cast<size_t>(slot) * RANDS_PER_PAIR;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const val_t* rr = pairRands + 4 * r;
        Vec3 pI = pointInTriangle(triI, rr[0], rr[1]);
        Vec3 pJ = pointInTriangle(triJ, rr[2], rr[3]);

        if (isRayBlocked(pI, pJ, nodes, triIdx, triangles, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijBlock[tid] = kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation (host, OpenMP)
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Simulation Kernel (Wave Propagation) - one block per local triangle row
// ============================================================================

constexpr int SIM_BLOCK = 256;

__global__ void simStepKernel(const val_t* __restrict__ kijLocal,
                              const int* __restrict__ tauLocal,
                              const val_t* __restrict__ areas,
                              const val_t* __restrict__ rho,
                              const val_t* __restrict__ radE,
                              val_t* __restrict__ radB,
                              int numTriangles, int rowStart, int t) {
    int il = blockIdx.x;
    int i = rowStart + il;

    const val_t* kRow = kijLocal + static_cast<size_t>(il) * numTriangles;
    const int* tRow = tauLocal + static_cast<size_t>(il) * numTriangles;

    val_t sum = ZERO;
    for (int j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (j == i) continue;

        int tauij = tRow[j];

        // Skip if wave hasn't yet propagated from j to i (tau >= 1 for i != j,
        // so only rows < t of radB are read here)
        if (t < tauij) continue;

        val_t kij = kRow[j];
        if (kij <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        val_t radJ = radB[static_cast<size_t>(t - tauij) * numTriangles + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sum += fminf(kij * areas[j], ONE) * radJ;
    }

    __shared__ val_t sdata[SIM_BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    // Update radiosity: reflection + emission
    if (threadIdx.x == 0) {
        radB[static_cast<size_t>(t) * numTriangles + i] =
            rho[i] * sdata[0] + radE[static_cast<size_t>(t) * numTriangles + i];
    }
}

// ============================================================================
// Distance Kernels (Cross-Correlation)
// ============================================================================

// One thread per (local row, lag): discrete cross-correlation against the
// source triangle's radiosity time series.
__global__ void corrKernel(const val_t* __restrict__ radB,
                           int numTriangles, int numTimesteps,
                           int rowStart, int numRows, int sourceIndex,
                           val_t* __restrict__ corr) {
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(numRows) * numTimesteps;
    if (tid >= total) return;

    int il = static_cast<int>(tid / numTimesteps);
    int t = static_cast<int>(tid % numTimesteps);
    int i = rowStart + il;

    val_t sum = ZERO;
    for (int tt = t; tt < numTimesteps; ++tt) {
        val_t pB = radB[static_cast<size_t>(tt) * numTriangles + i];
        val_t pS = radB[static_cast<size_t>(tt - t) * numTriangles + sourceIndex];
        sum += pS * pB;
    }
    corr[tid] = sum;
}

// One thread per local row: pick the earliest lag with the strictly largest
// correlation (same tie-breaking as the sequential code).
__global__ void argmaxKernel(const val_t* __restrict__ corr, int numTimesteps,
                             int numRows, val_t* __restrict__ distLocal) {
    int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= numRows) return;

    const val_t* row = corr + static_cast<size_t>(il) * numTimesteps;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < numTimesteps; ++t) {
        if (row[t] > maxCorr) {
            maxCorr = row[t];
            bestT = t;
        }
    }
    distLocal[il] = WAVE_SPEED * static_cast<val_t>(bestT);
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
    std::vector<val_t> kij;         // Form factors (local rows x N, row-major)
    std::vector<int> tau;           // Time delays (local rows x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, full copy)
    std::vector<val_t> distances;   // Computed distances from source (full)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // MPI decomposition: this rank owns global rows [rowStart, rowStart+rowCount)
    int mpiRank = 0;
    int mpiSize = 1;
    int rowStart = 0;
    int rowCount = 0;
    std::vector<int> rowCounts;     // per-rank row counts
    std::vector<int> rowDispls;     // per-rank row offsets

    // Device buffers (one GPU per rank)
    Triangle* dTriangles = nullptr;
    GpuNode* dNodes = nullptr;
    int* dTriIdx = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dKij = nullptr;          // local rows x N
    int* dTau = nullptr;            // local rows x N
    val_t* dRadE = nullptr;         // T x N
    val_t* dRadB = nullptr;         // T x N
    val_t* dCorr = nullptr;         // local rows x T
    val_t* dDist = nullptr;         // local rows

    size_t idxLocal2d(size_t il, size_t j) const { return il * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool root = (state.mpiRank == 0);

    // Generate mesh (deterministic, so every rank builds an identical copy)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (root) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Block-distribute triangle rows across MPI ranks
    int n = static_cast<int>(state.numTriangles);
    int base = n / state.mpiSize;
    int rem = n % state.mpiSize;
    state.rowCounts.resize(state.mpiSize);
    state.rowDispls.resize(state.mpiSize);
    for (int r = 0; r < state.mpiSize; ++r) {
        state.rowCounts[r] = base + (r < rem ? 1 : 0);
        state.rowDispls[r] = r * base + std::min(r, rem);
    }
    state.rowStart = state.rowDispls[state.mpiRank];
    state.rowCount = state.rowCounts[state.mpiRank];

    // Build octree for spatial acceleration
    if (root) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices (Kij and Tau hold only this rank's rows)
    state.kij.resize(static_cast<size_t>(state.rowCount) * state.numTriangles, ZERO);
    state.tau.resize(static_cast<size_t>(state.rowCount) * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// Upload static data and allocate working buffers on this rank's GPU
void setupDevice(SimulationState& state) {
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    size_t localRows = static_cast<size_t>(state.rowCount);

    // Flatten octree for GPU traversal
    std::vector<GpuNode> nodes;
    std::vector<int> triIdx;
    flattenOctree(state.octree, nodes, triIdx);

    CUDA_CHECK(cudaMalloc(&state.dTriangles, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, nodes.size() * sizeof(GpuNode)));
    CUDA_CHECK(cudaMalloc(&state.dTriIdx, std::max<size_t>(triIdx.size(), 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dKij, std::max<size_t>(localRows * n, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTau, std::max<size_t>(localRows * n, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dRadE, t * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, t * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dCorr, std::max<size_t>(localRows * t, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dDist, std::max<size_t>(localRows, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.dTriangles, state.triangles.data(),
                          n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, nodes.data(),
                          nodes.size() * sizeof(GpuNode), cudaMemcpyHostToDevice));
    if (!triIdx.empty()) {
        CUDA_CHECK(cudaMemcpy(state.dTriIdx, triIdx.data(),
                              triIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(),
                          t * n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, t * n * sizeof(val_t)));
}

void teardownDevice(SimulationState& state) {
    cudaFree(state.dTriangles);
    cudaFree(state.dNodes);
    cudaFree(state.dTriIdx);
    cudaFree(state.dAreas);
    cudaFree(state.dRho);
    cudaFree(state.dKij);
    cudaFree(state.dTau);
    cudaFree(state.dRadE);
    cudaFree(state.dRadB);
    cudaFree(state.dCorr);
    cudaFree(state.dDist);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Form factors for this rank's rows on the GPU. The rank first skips the
// mt19937 stream past all rows owned by lower ranks (offset derived from the
// deterministic cull mask), then processes its rows in blocks: generate the
// block's uniforms on the host (exact sequential stream), upload, and launch
// the Monte-Carlo visibility kernel for all pairs of the block.
void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij)...\n");

    const size_t n = state.numTriangles;
    const std::vector<Triangle>& tris = state.triangles;

    // Uniforms consumed by all rows before this rank's first row
    uint64_t pairsBefore = 0;
    #pragma omp parallel for schedule(static) reduction(+:pairsBefore)
    for (int i = 0; i < state.rowStart; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (static_cast<size_t>(i) == j) continue;
            if (!kijPairIsCulled(tris[i], tris[j])) pairsBefore++;
        }
    }

    RandomGenerator rng(42);
    rng.skip(pairsBefore * RANDS_PER_PAIR);

    // Process rows in blocks, capping the random-number staging buffers.
    // Two buffer sets (double buffering on two streams) let the host generate
    // block b+1's random numbers while the GPU processes block b.
    constexpr size_t MAX_RAND_BYTES = 256ull * 1024 * 1024;
    int blockRows = state.rowCount;
    if (blockRows > 0) {
        size_t cap = std::max<size_t>(1, MAX_RAND_BYTES / (n * RANDS_PER_PAIR * sizeof(val_t)));
        blockRows = static_cast<int>(std::min<size_t>(blockRows, cap));
    }
    size_t maxRands = static_cast<size_t>(blockRows) * n * RANDS_PER_PAIR;
    size_t maxSlots = static_cast<size_t>(blockRows) * n;

    cudaStream_t streams[2];
    val_t* dRands[2] = {nullptr, nullptr};
    int* dSlots[2] = {nullptr, nullptr};
    val_t* hRands[2] = {nullptr, nullptr};
    int* hSlots[2] = {nullptr, nullptr};
    if (state.rowCount > 0) {
        for (int s = 0; s < 2; ++s) {
            CUDA_CHECK(cudaStreamCreate(&streams[s]));
            CUDA_CHECK(cudaMalloc(&dRands[s], maxRands * sizeof(val_t)));
            CUDA_CHECK(cudaMalloc(&dSlots[s], maxSlots * sizeof(int)));
            CUDA_CHECK(cudaMallocHost(&hRands[s], maxRands * sizeof(val_t)));
            CUDA_CHECK(cudaMallocHost(&hSlots[s], maxSlots * sizeof(int)));
        }
    }

    int blockIndex = 0;
    for (int b = 0; b < state.rowCount; b += blockRows, ++blockIndex) {
        int rows = std::min(blockRows, state.rowCount - b);
        int firstRow = state.rowStart + b;
        int buf = blockIndex % 2;

        // Wait until this buffer's previous block has been fully consumed
        CUDA_CHECK(cudaStreamSynchronize(streams[buf]));
        int* slots = hSlots[buf];
        val_t* rands = hRands[buf];

        // Assign compacted stream slots to the block's non-culled pairs
        // (row-major, matching the sequential consumption order)
        std::vector<uint64_t> rowRands(rows);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            size_t i = static_cast<size_t>(firstRow + r);
            uint64_t cnt = 0;
            for (size_t j = 0; j < n; ++j) {
                bool consumes = (i != j) && !kijPairIsCulled(tris[i], tris[j]);
                slots[static_cast<size_t>(r) * n + j] = consumes ? 0 : -1;
                if (consumes) cnt++;
            }
            rowRands[r] = cnt;
        }
        uint64_t blockPairs = 0;
        std::vector<uint64_t> rowStartSlot(rows);
        for (int r = 0; r < rows; ++r) {
            rowStartSlot[r] = blockPairs;
            blockPairs += rowRands[r];
        }
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            uint64_t slot = rowStartSlot[r];
            for (size_t j = 0; j < n; ++j) {
                size_t p = static_cast<size_t>(r) * n + j;
                if (slots[p] == 0) slots[p] = static_cast<int>(slot++);
            }
        }

        // Replay the exact sequential random stream for this block. Chunks
        // are generated in parallel: each thread copies the block-start
        // engine, discards up to its chunk offset (discard is ~10x cheaper
        // than generating), then draws its share.
        uint64_t totalDraws = blockPairs * RANDS_PER_PAIR;
        #pragma omp parallel
        {
            int nChunks = omp_get_num_threads();
            int c = omp_get_thread_num();
            uint64_t lo = totalDraws * c / nChunks;
            uint64_t hi = totalDraws * (c + 1) / nChunks;
            if (lo < hi) {
                RandomGenerator local = rng;
                local.skip(lo);
                for (uint64_t k = lo; k < hi; ++k) rands[k] = local.rand();
            }
        }
        rng.skip(totalDraws);

        if (blockPairs > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dRands[buf], rands,
                                       totalDraws * sizeof(val_t),
                                       cudaMemcpyHostToDevice, streams[buf]));
        }
        CUDA_CHECK(cudaMemcpyAsync(dSlots[buf], slots,
                                   static_cast<size_t>(rows) * n * sizeof(int),
                                   cudaMemcpyHostToDevice, streams[buf]));

        long long total = static_cast<long long>(rows) * n;
        int threads = 128;
        long long blocks = (total + threads - 1) / threads;
        kijKernel<<<static_cast<unsigned int>(blocks), threads, 0, streams[buf]>>>(
            state.dTriangles, static_cast<int>(n), firstRow, rows,
            state.dNodes, state.dTriIdx, dRands[buf], dSlots[buf],
            state.dKij + static_cast<size_t>(b) * n);
        CUDA_CHECK(cudaGetLastError());
    }

    if (state.rowCount > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());

        // Keep a host copy of the local rows for validation statistics
        CUDA_CHECK(cudaMemcpy(state.kij.data(), state.dKij,
                              static_cast<size_t>(state.rowCount) * n * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        for (int s = 0; s < 2; ++s) {
            CUDA_CHECK(cudaFree(dRands[s]));
            CUDA_CHECK(cudaFree(dSlots[s]));
            CUDA_CHECK(cudaFreeHost(hRands[s]));
            CUDA_CHECK(cudaFreeHost(hSlots[s]));
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
        }
    }

    if (state.mpiRank == 0) {
        printf("  Progress: %zu/%zu triangles\n", n, n);
    }
}

// Time delays for this rank's rows on the host (OpenMP), then uploaded
void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau)...\n");

    #pragma omp parallel for schedule(static)
    for (int il = 0; il < state.rowCount; ++il) {
        size_t i = static_cast<size_t>(state.rowStart + il);
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idxLocal2d(il, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }

    if (state.rowCount > 0) {
        CUDA_CHECK(cudaMemcpy(state.dTau, state.tau.data(),
                              static_cast<size_t>(state.rowCount) * state.numTriangles * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    const bool root = (state.mpiRank == 0);
    if (root) printf("Running wave propagation simulation...\n");

    int n = static_cast<int>(state.numTriangles);
    std::vector<val_t> localRow(std::max(state.rowCount, 1));
    std::vector<val_t> fullRow(state.numTriangles);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        // Each rank computes radB(t, i) for its own rows on the GPU. Only
        // radB rows < t are read (tau >= 1 for i != j), so all inputs are
        // fully synchronized history.
        if (state.rowCount > 0) {
            simStepKernel<<<state.rowCount, SIM_BLOCK>>>(
                state.dKij, state.dTau, state.dAreas, state.dRho,
                state.dRadE, state.dRadB, n, state.rowStart, static_cast<int>(t));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localRow.data(),
                                  state.dRadB + t * state.numTriangles + state.rowStart,
                                  state.rowCount * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        // Exchange the freshly computed radiosity row so every rank (and its
        // GPU) holds the complete history for subsequent timesteps.
        MPI_Allgatherv(localRow.data(), state.rowCount, MPI_FLOAT,
                       fullRow.data(), state.rowCounts.data(), state.rowDispls.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

        std::memcpy(&state.radB[state.idxTN(t, 0)], fullRow.data(), n * sizeof(val_t));
        CUDA_CHECK(cudaMemcpy(state.dRadB + t * state.numTriangles, fullRow.data(),
                              n * sizeof(val_t), cudaMemcpyHostToDevice));

        if (root && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation...\n");

    int n = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);

    std::vector<val_t> localDist(std::max(state.rowCount, 1));

    if (state.rowCount > 0) {
        long long total = static_cast<long long>(state.rowCount) * T;
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        corrKernel<<<static_cast<unsigned int>(blocks), threads>>>(
            state.dRadB, n, T, state.rowStart, state.rowCount,
            static_cast<int>(state.sourceIndex), state.dCorr);
        CUDA_CHECK(cudaGetLastError());

        int aBlocks = (state.rowCount + 255) / 256;
        argmaxKernel<<<aBlocks, 256>>>(state.dCorr, T, state.rowCount, state.dDist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localDist.data(), state.dDist,
                              state.rowCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    // Gather the full distance vector on every rank
    MPI_Allgatherv(localDist.data(), state.rowCount, MPI_FLOAT,
                   state.distances.data(), state.rowCounts.data(), state.rowDispls.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation (collective: every rank contributes its Kij rows)
// ============================================================================

bool validateResults(const SimulationState& state) {
    const bool root = (state.mpiRank == 0);
    if (root) printf("\nValidation:\n");

    // Check that distances are non-negative (full vector on every rank)
    bool allNonNegative = true;
    bool allFinite = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            if (root) printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            if (root) printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            allFinite = false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    if (!allFinite) return false;

    if (root) {
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

        // Check source distance is zero or very small
        val_t srcDist = state.distances[state.sourceIndex];
        if (srcDist > WAVE_SPEED * 2) {
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
        }
    }

    // Check radiosity propagation (full radB history is replicated per rank)
    int receivedEnergy = 0;
    #pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    if (root) {
        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
        if (receivedEnergy == 0) {
            printf("  ERROR: No triangles received energy - simulation failed\n");
        }
    }
    if (receivedEnergy == 0) return false;

    // Check Kij matrix (each rank counts its local rows, then reduce)
    long long localNonZeroKij = 0;
    size_t localEntries = static_cast<size_t>(state.rowCount) * state.numTriangles;
    #pragma omp parallel for schedule(static) reduction(+:localNonZeroKij)
    for (size_t i = 0; i < localEntries; ++i) {
        if (state.kij[i] > EPSILON) localNonZeroKij++;
    }
    long long nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (root) {
        printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
               nonZeroKij, state.numTriangles * state.numTriangles,
               100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));
        if (nonZeroKij == 0) {
            printf("  ERROR: All form factors are zero - visibility computation failed\n");
        }
    }
    if (nonZeroKij == 0) return false;

    if (!allNonNegative) {
        return false;
    }

    if (root) printf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    // XOR accumulation is order-independent, so this parallelizes exactly
    #pragma omp parallel for schedule(static) reduction(^:hash)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    const bool root = (mpiRank == 0);

    // Bind each rank to a GPU (round-robin over the node's visible devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (root) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(mpiRank % deviceCount));

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelism: %d MPI rank(s) x %d OpenMP thread(s), %d CUDA device(s)/node\n",
               mpiSize, omp_get_max_threads(), deviceCount);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    setupDevice(state);

    if (root) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    long maxPreDuration = 0;
    MPI_Reduce(&preDuration, &maxPreDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Precomputation time: %ld ms\n", maxPreDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    long maxSimDuration = 0;
    MPI_Reduce(&simDuration, &maxSimDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Simulation time: %ld ms\n", maxSimDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    long maxDistDuration = 0;
    MPI_Reduce(&distDuration, &maxDistDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Distance computation time: %ld ms\n", maxDistDuration);
        printf("\n");
    }

    // Total time
    long totalTime = maxPreDuration + maxSimDuration + maxDistDuration;

    if (root) {
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
            // Convert distances to double for output
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
    }

    // Validation (collective across ranks)
    int exitCode = 0;
    if (validate) {
        if (!validateResults(state)) {
            exitCode = 1;
        }
    }

    teardownDevice(state);
    MPI_Finalize();
    return exitCode;
}
