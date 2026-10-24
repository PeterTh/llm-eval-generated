/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of a room impulse response
 * simulation using radiosity-based wave propagation. It models how
 * sound/light waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility
 *    and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * Parallelization strategy:
 *   - MPI: distributes triangle rows of the Kij matrix, the wave propagation
 *     timesteps, and the distance cross-correlation across ranks. Results are
 *     exchanged with Allgatherv so every rank holds the full state.
 *   - OpenMP: parallelizes host-side preprocessing and drives independent
 *     CUDA streams from a small team of threads.
 *   - CUDA: accelerates the visibility/form-factor computation (octree
 *     traversal + ray/triangle tests), the per-timestep radiosity updates,
 *     and the cross-correlation distance computation.
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
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
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

constexpr int GPU_BLOCK_SIZE = 256;
constexpr int GPU_MAX_STREAMS = 4;

// Global rank used to funnel human-readable output through rank 0 only.
static int g_rank = 0;

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
        val_t rr = boxHalfSize.x * std::abs(axis.x) +
                   boxHalfSize.y * std::abs(axis.y) +
                   boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > rr || maxP < -rr);
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
// GPU Data Structures and Kernels
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                    \
                         cudaGetErrorString(_err), __FILE__, __LINE__);         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// POD mirror of Triangle used in device code.
struct GTri {
    float3 a, b, c, n;
};

// Flattened octree node. child[i] is -1 when the child is absent.
struct GNode {
    float3 minBound, maxBound, center, halfExtent;
    int child[8];
    int triStart, triCount;
};

static int flattenOctree(const Octree& node, std::vector<GNode>& nodes,
                         std::vector<int>& triIndices) {
    const int id = static_cast<int>(nodes.size());
    nodes.push_back(GNode{});
    {
        GNode& g = nodes[id];
        g.minBound = make_float3(node.minBound.x, node.minBound.y, node.minBound.z);
        g.maxBound = make_float3(node.maxBound.x, node.maxBound.y, node.maxBound.z);
        g.center = make_float3(node.center.x, node.center.y, node.center.z);
        g.halfExtent = make_float3(node.halfExtent.x, node.halfExtent.y, node.halfExtent.z);
        for (int i = 0; i < 8; ++i) g.child[i] = -1;
        g.triStart = -1;
        g.triCount = 0;
        if (!node.triangleIndices.empty()) {
            g.triStart = static_cast<int>(triIndices.size());
            g.triCount = static_cast<int>(node.triangleIndices.size());
        }
    }

    if (!node.triangleIndices.empty()) {
        for (size_t idx : node.triangleIndices) {
            triIndices.push_back(static_cast<int>(idx));
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                const int childId = flattenOctree(*node.children[i], nodes, triIndices);
                // Re-index after the recursive call, which may reallocate nodes.
                nodes[id].child[i] = childId;
            }
        }
    }
    return id;
}

// SplitMix64-based counter RNG. Results are keyed by (i, j) so they are
// reproducible independently of the MPI rank count or CUDA launch geometry.
__device__ __forceinline__ uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ float rand01(uint64_t& state) {
    return static_cast<float>(splitmix64(state) >> 40) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ bool rayIntersectsBox(const GNode& node,
                                                 const float3& p1,
                                                 const float3& p2) {
    constexpr float EPS = 1e-6f;
    float3 d = make_float3((p2.x - p1.x) * 0.5f,
                           (p2.y - p1.y) * 0.5f,
                           (p2.z - p1.z) * 0.5f);
    float3 c = make_float3(p1.x + d.x - node.center.x,
                           p1.y + d.y - node.center.y,
                           p1.z + d.z - node.center.z);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPS) return false;
    if (fabsf(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPS) return false;
    if (fabsf(d.x * c.y - d.y * c.x) >
        node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPS) return false;

    return true;
}

// Moller-Trumbore ray/triangle intersection.
__device__ __forceinline__ float rayTriangleIntersectDevice(const float3& orig,
                                                            const float3& dir,
                                                            const float3& v0,
                                                            const float3& v1,
                                                            const float3& v2) {
    constexpr float EPS = 1e-6f;
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 pvec = make_float3(dir.y * e2.z - dir.z * e2.y,
                              dir.z * e2.x - dir.x * e2.z,
                              dir.x * e2.y - dir.y * e2.x);
    float det = e1.x * pvec.x + e1.y * pvec.y + e1.z * pvec.z;
    if (fabsf(det) < EPS) return FLT_MAX;

    const float invDet = 1.0f / det;
    float3 tvec = make_float3(orig.x - v0.x, orig.y - v0.y, orig.z - v0.z);
    float u = (tvec.x * pvec.x + tvec.y * pvec.y + tvec.z * pvec.z) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    float3 qvec = make_float3(tvec.y * e1.z - tvec.z * e1.y,
                              tvec.z * e1.x - tvec.x * e1.z,
                              tvec.x * e1.y - tvec.y * e1.x);
    float v = (dir.x * qvec.x + dir.y * qvec.y + dir.z * qvec.z) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return (e2.x * qvec.x + e2.y * qvec.y + e2.z * qvec.z) * invDet;
}

// Octree-accelerated segment blocking test, mirroring Octree::applyToTris.
__device__ __forceinline__ bool isRayBlockedDevice(const float3& from,
                                                   const float3& to,
                                                   const GNode* nodes,
                                                   const int* triIndices,
                                                   const GTri* tris,
                                                   int srcTriIdx,
                                                   int dstTriIdx,
                                                   float rayLen,
                                                   const float3& dirNorm) {
    constexpr float EPS = 1e-6f;
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int ni = stack[--sp];
        const GNode& node = nodes[ni];
        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                const int idx = triIndices[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const GTri& tri = tris[idx];
                const float dist = rayTriangleIntersectDevice(from, dirNorm,
                                                              tri.a, tri.b, tri.c);
                if (dist > EPS && dist < rayLen - EPS) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                const int child = node.child[c];
                if (child >= 0 && rayIntersectsBox(nodes[child], from, to)) {
                    if (sp < 64) stack[sp++] = child;
                }
            }
        }
    }
    return false;
}

__device__ __forceinline__ float cosPhiDevice(const float3& v, const float3& normal) {
    constexpr float EPS = 1e-6f;
    const float vNorm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (vNorm <= EPS) return 0.0f;
    const float d = (v.x * normal.x + v.y * normal.y + v.z * normal.z) / vNorm;
    return fmaxf(0.0f, d);
}

__device__ __forceinline__ float computeKijDevice(int idxI, int idxJ,
                                                  const GTri* tris,
                                                  const GNode* nodes,
                                                  const int* triIndices) {
    constexpr int NRAYS = 16;
    constexpr float PI_F = 3.14159265358979323846f;
    constexpr float INV_NRAYS = 1.0f / static_cast<float>(NRAYS);
    constexpr float EPS = 1e-6f;

    const GTri& triI = tris[idxI];
    const GTri& triJ = tris[idxJ];

    const float normalDot = triI.n.x * triJ.n.x + triI.n.y * triJ.n.y + triI.n.z * triJ.n.z;
    if (normalDot > 0.99f) return 0.0f;

    uint64_t state = static_cast<uint64_t>(static_cast<uint32_t>(idxI)) * 0x9E3779B97F4A7C15ULL ^
                     (static_cast<uint64_t>(static_cast<uint32_t>(idxJ)) + 0x632BE59BD9B4E019ULL);

    float kij = 0.0f;
#pragma unroll
    for (int r = 0; r < NRAYS; ++r) {
        float u = rand01(state);
        float v = rand01(state);
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        float3 pI = make_float3(triI.a.x + (triI.b.x - triI.a.x) * u + (triI.c.x - triI.a.x) * v,
                                triI.a.y + (triI.b.y - triI.a.y) * u + (triI.c.y - triI.a.y) * v,
                                triI.a.z + (triI.b.z - triI.a.z) * u + (triI.c.z - triI.a.z) * v);

        float u2 = rand01(state);
        float v2 = rand01(state);
        if (u2 + v2 > 1.0f) {
            u2 = 1.0f - u2;
            v2 = 1.0f - v2;
        }
        float3 pJ = make_float3(triJ.a.x + (triJ.b.x - triJ.a.x) * u2 + (triJ.c.x - triJ.a.x) * v2,
                                triJ.a.y + (triJ.b.y - triJ.a.y) * u2 + (triJ.c.y - triJ.a.y) * v2,
                                triJ.a.z + (triJ.b.z - triJ.a.z) * u2 + (triJ.c.z - triJ.a.z) * v2);

        float3 vv = make_float3(pJ.x - pI.x, pJ.y - pI.y, pJ.z - pI.z);
        const float distSqr = vv.x * vv.x + vv.y * vv.y + vv.z * vv.z;
        if (distSqr < EPS) continue;

        const float rayLen = sqrtf(distSqr);
        const float3 dirNorm = make_float3(vv.x / rayLen, vv.y / rayLen, vv.z / rayLen);

        if (isRayBlockedDevice(pI, pJ, nodes, triIndices, tris, idxI, idxJ,
                               rayLen, dirNorm)) {
            continue;
        }

        const float cosPhiI = cosPhiDevice(vv, triI.n);
        const float3 negV = make_float3(-vv.x, -vv.y, -vv.z);
        const float cosPhiJ = cosPhiDevice(negV, triJ.n);

        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij += (cosPhiI * cosPhiJ) / (PI_F * distSqr);
    }

    return kij * INV_NRAYS;
}

// One row per blockIdx.y; threads stride over the j dimension.
__global__ void kijKernel(const GTri* __restrict__ tris,
                          const GNode* __restrict__ nodes,
                          const int* __restrict__ triIndices,
                          float* __restrict__ kij,
                          int rowStart, int rowEnd, int n) {
    const int i = rowStart + static_cast<int>(blockIdx.y);
    if (i >= rowEnd) return;

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        const size_t idx = static_cast<size_t>(i) * n + j;
        if (i == j) {
            kij[idx] = 0.0f;
            continue;
        }
        kij[idx] = computeKijDevice(i, j, tris, nodes, triIndices);
    }
}

// One CUDA block per receiving triangle; the block reduces over all emitters.
__global__ void simulationKernel(const float* __restrict__ kij,
                                 const int* __restrict__ tau,
                                 const float* __restrict__ areas,
                                 const float* __restrict__ rho,
                                 const float* __restrict__ radE,
                                 float* __restrict__ radB,
                                 int n, int t, int rowStart, int rowEnd) {
    const int i = rowStart + static_cast<int>(blockIdx.x);
    if (i >= rowEnd) return;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == i) continue;

        const size_t idx = static_cast<size_t>(i) * n + j;
        const int tauij = tau[idx];
        if (t < tauij) continue;

        const float kijv = kij[idx];
        if (kijv <= 0.0f) continue;

        const int srcTime = t - tauij;
        const float radJ = radB[static_cast<size_t>(srcTime) * n + j];
        if (radJ <= 0.0f) continue;

        sum += fminf(kijv * areas[j], 1.0f) * radJ;
    }

    __shared__ float sdata[GPU_BLOCK_SIZE];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = GPU_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        radB[static_cast<size_t>(t) * n + i] =
            rho[i] * sdata[0] + radE[static_cast<size_t>(t) * n + i];
    }
}

// One thread per receiving triangle for the O(T^2) cross-correlation.
__global__ void distanceKernel(const float* __restrict__ radB,
                               float* __restrict__ distances,
                               int n, int numTimesteps, int sourceIndex,
                               int rowStart, int rowEnd) {
    constexpr float WAVE_SPEED_F = 0.5f;
    const int i = rowStart + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rowEnd) return;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int t = 0; t < numTimesteps; ++t) {
        float sum = 0.0f;
        for (int tt = t; tt < numTimesteps; ++tt) {
            const float pB = radB[static_cast<size_t>(tt) * n + i];
            const float pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED_F * static_cast<float>(bestT);
}

struct GpuBuffers {
    GTri* d_tris = nullptr;
    GNode* d_nodes = nullptr;
    int* d_triIndices = nullptr;
    float* d_kij = nullptr;
    int* d_tau = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    float* d_distances = nullptr;

    cudaStream_t streams[GPU_MAX_STREAMS] = {};
    int numStreams = 1;
    int device = 0;
};

// Row-block distribution of work across MPI ranks.
struct Partition {
    size_t start = 0;
    size_t end = 0;
    std::vector<int> rowCounts;    // per-rank triangle counts
    std::vector<int> rowDispls;    // per-rank triangle offsets

    size_t localRows() const { return end - start; }
};

static Partition makePartition(size_t n, int rank, int size) {
    Partition p;
    p.rowCounts.assign(size, 0);
    p.rowDispls.assign(size, 0);

    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        p.rowCounts[r] = static_cast<int>(rows);
        p.rowDispls[r] = static_cast<int>(offset);
        if (r == rank) {
            p.start = offset;
            p.end = offset + rows;
        }
        offset += rows;
    }
    return p;
}

static int chooseStreamCount() {
    const int maxThreads = omp_get_max_threads();
    return std::max(1, std::min(GPU_MAX_STREAMS, maxThreads));
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
// Random Number Generation (host reference implementation)
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
// Visibility Testing (Octree-accelerated, host reference)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
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
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// GPU Setup
// ============================================================================

static void setupGpu(GpuBuffers& gpu, const SimulationState& state) {
    const size_t n = state.numTriangles;
    const size_t t = state.numTimesteps;

    CUDA_CHECK(cudaSetDevice(gpu.device));

    // Triangle array.
    std::vector<GTri> hostTris(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& tri = state.triangles[i];
        hostTris[i].a = make_float3(tri.a.x, tri.a.y, tri.a.z);
        hostTris[i].b = make_float3(tri.b.x, tri.b.y, tri.b.z);
        hostTris[i].c = make_float3(tri.c.x, tri.c.y, tri.c.z);
        hostTris[i].n = make_float3(tri._normal.x, tri._normal.y, tri._normal.z);
    }
    CUDA_CHECK(cudaMalloc(&gpu.d_tris, n * sizeof(GTri)));
    CUDA_CHECK(cudaMemcpy(gpu.d_tris, hostTris.data(), n * sizeof(GTri),
                          cudaMemcpyHostToDevice));

    // Flattened octree.
    std::vector<GNode> hostNodes;
    std::vector<int> hostTriIndices;
    if (n > 0) {
        hostNodes.reserve(n);
        flattenOctree(state.octree, hostNodes, hostTriIndices);
    }
    if (hostNodes.empty()) hostNodes.push_back(GNode{});
    if (hostTriIndices.empty()) hostTriIndices.push_back(0);
    const size_t nodeCount = hostNodes.size();
    const size_t triIndexCount = hostTriIndices.size();

    CUDA_CHECK(cudaMalloc(&gpu.d_nodes, nodeCount * sizeof(GNode)));
    CUDA_CHECK(cudaMemcpy(gpu.d_nodes, hostNodes.data(), nodeCount * sizeof(GNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_triIndices, triIndexCount * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu.d_triIndices, hostTriIndices.data(),
                          triIndexCount * sizeof(int), cudaMemcpyHostToDevice));

    // Per-triangle data.
    CUDA_CHECK(cudaMalloc(&gpu.d_areas, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu.d_areas, state.areas.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_rho, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu.d_rho, state.rho.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    const size_t radCount = std::max<size_t>(t * n, 1);
    CUDA_CHECK(cudaMalloc(&gpu.d_radE, radCount * sizeof(float)));
    if (t * n > 0) {
        CUDA_CHECK(cudaMemcpy(gpu.d_radE, state.radE.data(), t * n * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&gpu.d_radB, radCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu.d_radB, 0, radCount * sizeof(float)));

    // Matrices and distance output.
    CUDA_CHECK(cudaMalloc(&gpu.d_kij, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu.d_kij, 0, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_tau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMemset(gpu.d_tau, 0, n * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_distances, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu.d_distances, 0, n * sizeof(float)));

    // One CUDA stream per OpenMP thread participating in GPU work.
    gpu.numStreams = chooseStreamCount();
#pragma omp parallel num_threads(gpu.numStreams)
    {
        const int tid = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(gpu.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&gpu.streams[tid], cudaStreamNonBlocking));
    }
}

static void shutdownGpu(GpuBuffers& gpu) {
    if (gpu.d_tris) cudaFree(gpu.d_tris);
    if (gpu.d_nodes) cudaFree(gpu.d_nodes);
    if (gpu.d_triIndices) cudaFree(gpu.d_triIndices);
    if (gpu.d_kij) cudaFree(gpu.d_kij);
    if (gpu.d_tau) cudaFree(gpu.d_tau);
    if (gpu.d_areas) cudaFree(gpu.d_areas);
    if (gpu.d_rho) cudaFree(gpu.d_rho);
    if (gpu.d_radE) cudaFree(gpu.d_radE);
    if (gpu.d_radB) cudaFree(gpu.d_radB);
    if (gpu.d_distances) cudaFree(gpu.d_distances);
    for (int i = 0; i < gpu.numStreams; ++i) {
        if (gpu.streams[i]) cudaStreamDestroy(gpu.streams[i]);
    }
    gpu = GpuBuffers{};
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

    if (g_rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree for spatial acceleration
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[static_cast<size_t>(i)] = state.triangles[static_cast<size_t>(i)].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.assign(state.numTriangles * state.numTriangles, ZERO);
    state.tau.assign(state.numTriangles * state.numTriangles, 0);
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state, const Partition& part) {
    if (g_rank == 0) printf("Computing time delays (Tau)...\n");

#pragma omp parallel for schedule(static)
    for (long long ii = static_cast<long long>(part.start);
         ii < static_cast<long long>(part.end); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

void computeFormFactors(SimulationState& state, const Partition& part,
                        GpuBuffers& gpu, bool keepHostRows) {
    if (g_rank == 0) printf("Computing form factors (Kij)...\n");

    const size_t local = part.localRows();
    const int n = static_cast<int>(state.numTriangles);

    if (local > 0) {
        const int ns = gpu.numStreams;
#pragma omp parallel num_threads(ns)
        {
            const int tid = omp_get_thread_num();
            CUDA_CHECK(cudaSetDevice(gpu.device));

            const size_t chunk = (local + static_cast<size_t>(ns) - 1) /
                                 static_cast<size_t>(ns);
            const size_t s = part.start + static_cast<size_t>(tid) * chunk;
            const size_t e = std::min(part.end, s + chunk);
            if (s < e) {
                const int block = GPU_BLOCK_SIZE;
                const int gx = (n + block - 1) / block;
                dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(e - s));
                kijKernel<<<grid, block, 0, gpu.streams[tid]>>>(
                    gpu.d_tris, gpu.d_nodes, gpu.d_triIndices, gpu.d_kij,
                    static_cast<int>(s), static_cast<int>(e), n);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // The host copy is only needed when validation counts the form factors.
        if (keepHostRows) {
            const size_t bytes = local * state.numTriangles * sizeof(float);
            CUDA_CHECK(cudaMemcpy(state.kij.data() + part.start * state.numTriangles,
                                  gpu.d_kij + part.start * state.numTriangles,
                                  bytes, cudaMemcpyDeviceToHost));
        }
    }

    if (g_rank == 0) {
        const size_t n = state.numTriangles;
        for (size_t p = 100; p <= n; p += 100) {
            printf("  Progress: %zu/%zu triangles\n", p, n);
        }
        if (n % 100 != 0) {
            printf("  Progress: %zu/%zu triangles\n", n, n);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, const Partition& part,
                   GpuBuffers& gpu) {
    if (g_rank == 0) printf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const size_t local = part.localRows();
    const int numTimesteps = static_cast<int>(state.numTimesteps);
    const int ns = gpu.numStreams;

    for (int t = 0; t < numTimesteps; ++t) {
        if (local > 0) {
#pragma omp parallel num_threads(ns)
            {
                const int tid = omp_get_thread_num();
                CUDA_CHECK(cudaSetDevice(gpu.device));

                const size_t chunk = (local + static_cast<size_t>(ns) - 1) /
                                     static_cast<size_t>(ns);
                const size_t s = part.start + static_cast<size_t>(tid) * chunk;
                const size_t e = std::min(part.end, s + chunk);
                if (s < e) {
                    const unsigned blocks = static_cast<unsigned>(e - s);
                    simulationKernel<<<blocks, GPU_BLOCK_SIZE, 0,
                                       gpu.streams[tid]>>>(
                        gpu.d_kij, gpu.d_tau, gpu.d_areas, gpu.d_rho,
                        gpu.d_radE, gpu.d_radB, static_cast<int>(n), t,
                        static_cast<int>(s), static_cast<int>(e));
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy this rank's row slice back, exchange it, then refresh the device.
        if (local > 0) {
            CUDA_CHECK(cudaMemcpy(state.radB.data() + static_cast<size_t>(t) * n + part.start,
                                  gpu.d_radB + static_cast<size_t>(t) * n + part.start,
                                  local * sizeof(float), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       state.radB.data() + static_cast<size_t>(t) * n,
                       part.rowCounts.data(), part.rowDispls.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(gpu.d_radB + static_cast<size_t>(t) * n,
                              state.radB.data() + static_cast<size_t>(t) * n,
                              n * sizeof(float), cudaMemcpyHostToDevice));

        if (g_rank == 0 &&
            ((t + 1) % 10 == 0 || t + 1 == numTimesteps)) {
            printf("  Timestep %d/%d\n", t + 1, numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, const Partition& part,
                      GpuBuffers& gpu) {
    if (g_rank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t local = part.localRows();
    const int n = static_cast<int>(state.numTriangles);
    const int ns = gpu.numStreams;

    if (local > 0) {
#pragma omp parallel num_threads(ns)
        {
            const int tid = omp_get_thread_num();
            CUDA_CHECK(cudaSetDevice(gpu.device));

            const size_t chunk = (local + static_cast<size_t>(ns) - 1) /
                                 static_cast<size_t>(ns);
            const size_t s = part.start + static_cast<size_t>(tid) * chunk;
            const size_t e = std::min(part.end, s + chunk);
            if (s < e) {
                const int block = 128;
                const int grid = static_cast<int>((e - s + block - 1) / block);
                distanceKernel<<<grid, block, 0, gpu.streams[tid]>>>(
                    gpu.d_radB, gpu.d_distances, n,
                    static_cast<int>(state.numTimesteps),
                    static_cast<int>(state.sourceIndex),
                    static_cast<int>(s), static_cast<int>(e));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(state.distances.data() + part.start,
                              gpu.d_distances + part.start,
                              local * sizeof(float), cudaMemcpyDeviceToHost));
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.distances.data(), part.rowCounts.data(),
                   part.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, long long globalNonZeroKij) {
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
    printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
           globalNonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * static_cast<double>(globalNonZeroKij) /
               static_cast<double>(state.numTriangles * state.numTriangles));

    if (globalNonZeroKij == 0) {
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
    g_rank = rank;

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

    // Bind each rank to a GPU on its node (round-robin over visible devices).
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &localRank);
    }

    int deviceCount = 0;
    const cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: no CUDA device available (CUDA is required)\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    // Keep the host OpenMP team modest so ranks sharing a node do not
    // oversubscribe it. CUDA work is driven by a subset of these threads.
    // Disable dynamic teams so every requested GPU-driving thread exists and
    // receives its share of the rows.
    omp_set_dynamic(0);
    const int cpuThreads = std::max(1, std::min(omp_get_max_threads(), 16));
    omp_set_num_threads(cpuThreads);

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    const Partition part = makePartition(state.numTriangles, rank, size);

    // Prepare GPU resources (mesh, octree, per-triangle data, streams).
    GpuBuffers gpu;
    gpu.device = device;
    setupGpu(gpu, state);

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, part);

    // Only this rank's Tau rows are needed locally; the remaining device rows
    // stay zero from setupGpu. This keeps precomputation perfectly scalable.
    if (part.localRows() > 0) {
        CUDA_CHECK(cudaMemcpy(gpu.d_tau + part.start * state.numTriangles,
                              state.tau.data() + part.start * state.numTriangles,
                              part.localRows() * state.numTriangles * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    computeFormFactors(state, part, gpu, validate);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, part, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, part, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");

        // Total time
        long totalTime = preDuration + simDuration + distDuration;
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

    int localOk = 1;
    if (validate) {
        // Count non-zero form factors over this rank's rows and reduce.
        long long localNonZeroKij = 0;
        const size_t n = state.numTriangles;
        for (size_t i = part.start; i < part.end; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (state.kij[i * n + j] > EPSILON) localNonZeroKij++;
            }
        }
        long long globalNonZeroKij = 0;
        MPI_Reduce(&localNonZeroKij, &globalNonZeroKij, 1, MPI_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            localOk = validateResults(state, globalNonZeroKij) ? 1 : 0;
        }
        MPI_Bcast(&localOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    shutdownGpu(gpu);

    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_free(&nodeComm);
    }

    MPI_Finalize();
    return (validate && localOk == 0) ? 1 : 0;
}
