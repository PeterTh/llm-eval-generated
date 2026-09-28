/**
 * Room Response Simulation Benchmark (CUDA-parallel implementation)
 *
 * GPU-parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy (CUDA):
 *  - Tau:          one thread per (i,j) pair
 *  - Kij:          one thread per (i,j) pair; visibility rays traverse a
 *                  flattened octree with an explicit stack; Monte Carlo
 *                  sampling uses a deterministic counter-based RNG so every
 *                  pair has an independent random stream
 *  - Propagation:  timesteps are inherently sequential; within a timestep one
 *                  block per receiving triangle with a shared-memory reduction
 *                  over all source triangles; radiosity history stays on device
 *  - Distances:    one thread per triangle computing the full cross-correlation
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
#include <vector>

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ============================================================================
// Vector and Triangle Types (host-side mesh generation / octree build)
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
// GPU Data Structures
// ============================================================================

struct DTri {
    float3 a, b, c, n;
};

// Flattened octree node: leaf iff triCount > 0 (matches host-side semantics)
struct GpuNode {
    float3 center;
    float3 halfExtent;
    int child[8];
    int triStart;
    int triCount;
};

// Recursively flatten the pointer-based octree into arrays
static int flattenOctree(const Octree& node, std::vector<GpuNode>& nodes, std::vector<int>& leafTris) {
    int myIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    GpuNode g;
    g.center = make_float3(node.center.x, node.center.y, node.center.z);
    g.halfExtent = make_float3(node.halfExtent.x, node.halfExtent.y, node.halfExtent.z);
    g.triStart = static_cast<int>(leafTris.size());
    g.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) leafTris.push_back(static_cast<int>(idx));

    for (int i = 0; i < 8; ++i) {
        g.child[i] = node.children[i] ? flattenOctree(*node.children[i], nodes, leafTris) : -1;
    }

    nodes[myIdx] = g;
    return myIdx;
}

// ============================================================================
// Device Vector Helpers
// ============================================================================

__device__ __forceinline__ float3 d_add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 d_sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 d_scale(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 d_neg(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float d_dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 d_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float d_norm(float3 a) { return sqrtf(d_dot(a, a)); }

// ============================================================================
// Device RNG (counter-based, deterministic per triangle pair / ray / sample)
// ============================================================================

__device__ __forceinline__ float d_rand01(unsigned long long ctr) {
    // splitmix64 mix; top 24 bits mapped to [0,1)
    unsigned long long z = ctr + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ __forceinline__ float3 d_pointInTriangle(const DTri& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float3 ab = d_sub(t.b, t.a);
    float3 ac = d_sub(t.c, t.a);
    return d_add(t.a, d_add(d_scale(ab, u), d_scale(ac, v)));
}

// ============================================================================
// Device Ray-Triangle Intersection (Möller-Trumbore) and Visibility
// ============================================================================

__device__ __forceinline__ float d_rayTriangleIntersect(float3 orig, float3 dir,
                                                        float3 v0, float3 v1, float3 v2) {
    const float BIG = 3.402823466e+38f;
    float3 e1 = d_sub(v1, v0);
    float3 e2 = d_sub(v2, v0);
    float3 pvec = d_cross(dir, e2);
    float det = d_dot(e1, pvec);

    if (fabsf(det) < EPSILON) return BIG;

    float invDet = 1.0f / det;
    float3 tvec = d_sub(orig, v0);
    float u = d_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return BIG;

    float3 qvec = d_cross(tvec, e1);
    float v = d_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return BIG;

    return d_dot(e2, qvec) * invDet;
}

// Segment-vs-AABB test (matches host Octree::rayIntersectsBox)
__device__ __forceinline__ bool d_rayIntersectsBox(float3 p1, float3 p2, const GpuNode& nd) {
    float3 d = d_scale(d_sub(p2, p1), 0.5f);
    float3 c = d_sub(d_add(p1, d), nd.center);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    float3 h = nd.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Iterative octree traversal with an explicit stack (tree depth is bounded by
// MAX_OCTREE_LEAF_SIZE, so 64 entries are ample).
__device__ bool d_isRayBlocked(float3 from, float3 to,
                               const GpuNode* __restrict__ nodes,
                               const int* __restrict__ leafTris,
                               const DTri* __restrict__ tris,
                               int srcTriIdx, int dstTriIdx) {
    float3 dir = d_sub(to, from);
    float rayLen = d_norm(dir);
    if (rayLen < EPSILON) return true;
    float3 dirNorm = d_scale(dir, 1.0f / rayLen);

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const GpuNode nd = nodes[stack[--sp]];

        if (nd.triCount > 0) {
            // Leaf node: check triangles directly
            for (int k = 0; k < nd.triCount; ++k) {
                int idx = leafTris[nd.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const DTri tr = tris[idx];
                float dist = d_rayTriangleIntersect(from, dirNorm, tr.a, tr.b, tr.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            // Inner node: descend into children whose box the ray touches
            for (int i = 0; i < 8; ++i) {
                int ci = nd.child[i];
                if (ci >= 0 && d_rayIntersectsBox(from, to, nodes[ci])) {
                    stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Kernels
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ float d_cosPhi(float3 v, float3 normal) {
    float vNorm = d_norm(v);
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, d_dot(v, normal) / vNorm);
}

// Form factor Kij between triangle i (receiver) and triangle j (emitter);
// one thread per (i, j) pair.
__global__ void kijKernel(const DTri* __restrict__ tris,
                          const GpuNode* __restrict__ nodes,
                          const int* __restrict__ leafTris,
                          float* __restrict__ kij, int n) {
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(n) * n;
    if (tid >= total) return;

    int i = static_cast<int>(tid / n);
    int j = static_cast<int>(tid % n);
    if (i == j) {
        kij[tid] = ZERO;
        return;
    }

    const DTri triI = tris[i];
    const DTri triJ = tris[j];

    // Cull triangles facing the same direction
    if (d_dot(triI.n, triJ.n) > 0.99f) {
        kij[tid] = ZERO;
        return;
    }

    float acc = ZERO;
    unsigned long long ctrBase = static_cast<unsigned long long>(tid) * (NUM_RAYS * 4ULL);

    for (int r = 0; r < NUM_RAYS; ++r) {
        unsigned long long c = ctrBase + static_cast<unsigned long long>(r) * 4ULL;
        float3 pI = d_pointInTriangle(triI, d_rand01(c + 0), d_rand01(c + 1));
        float3 pJ = d_pointInTriangle(triJ, d_rand01(c + 2), d_rand01(c + 3));

        if (d_isRayBlocked(pI, pJ, nodes, leafTris, tris, i, j)) continue;

        float3 v = d_sub(pJ, pI);
        float distSqr = d_dot(v, v);
        if (distSqr < EPSILON) continue;

        float cosPhiI = d_cosPhi(v, triI.n);
        float cosPhiJ = d_cosPhi(d_neg(v), triJ.n);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        acc += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[tid] = acc * INV_NUM_RAYS;
}

// Time delays Tau; one thread per (i, j) pair
__global__ void tauKernel(const DTri* __restrict__ tris, int* __restrict__ tau, int n) {
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(n) * n;
    if (tid >= total) return;

    int i = static_cast<int>(tid / n);
    int j = static_cast<int>(tid % n);
    if (i == j) {
        tau[tid] = 0;
        return;
    }

    const DTri ti = tris[i];
    const DTri tj = tris[j];
    float3 ci = d_scale(d_add(d_add(ti.a, ti.b), ti.c), 1.0f / 3.0f);
    float3 cj = d_scale(d_add(d_add(tj.a, tj.b), tj.c), 1.0f / 3.0f);
    float dist = d_norm(d_sub(ci, cj));
    tau[tid] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Precompute clamped propagation weights: kw[i][j] = min(kij * area_j, 1)
__global__ void weightKernel(const float* __restrict__ kij, const float* __restrict__ areas,
                             float* __restrict__ kw, int n) {
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(n) * n;
    if (tid >= total) return;
    int j = static_cast<int>(tid % n);
    kw[tid] = fminf(kij[tid] * areas[j], ONE);
}

// One wave-propagation timestep: one block per receiving triangle i,
// threads reduce contributions over source triangles j.
constexpr int SIM_BLOCK = 256;

__global__ void simStepKernel(const float* __restrict__ kw, const int* __restrict__ tau,
                              const float* __restrict__ rho, const float* __restrict__ radE,
                              float* radB, int n, int t) {
    __shared__ float sdata[SIM_BLOCK];

    int i = blockIdx.x;
    const float* kwRow = kw + static_cast<size_t>(i) * n;
    const int* tauRow = tau + static_cast<size_t>(i) * n;

    float s = ZERO;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == i) continue;

        int tauij = tauRow[j];
        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        float w = kwRow[j];
        if (w <= ZERO) continue;

        // Radiosity of source triangle at time when emission occurred
        float radJ = radB[static_cast<size_t>(t - tauij) * n + j];
        if (radJ <= ZERO) continue;

        s += w * radJ;
    }

    sdata[threadIdx.x] = s;
    __syncthreads();
    for (int off = SIM_BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sdata[threadIdx.x] += sdata[threadIdx.x + off];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Update radiosity: reflection + emission
        radB[static_cast<size_t>(t) * n + i] = rho[i] * sdata[0] + radE[static_cast<size_t>(t) * n + i];
    }
}

// Distance estimation via cross-correlation; one thread per triangle
__global__ void distanceKernel(const float* __restrict__ radB, float* __restrict__ dist,
                               int n, int numTimesteps, int sourceIndex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < numTimesteps; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * n + i];
            float pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    dist[i] = WAVE_SPEED * static_cast<float>(bestT);
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

    // Device-side buffers
    DTri* d_tris = nullptr;
    GpuNode* d_nodes = nullptr;
    int* d_leafTris = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_kij = nullptr;
    float* d_kw = nullptr;
    int* d_tau = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    float* d_dist = nullptr;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Upload mesh, octree, and simulation inputs to the GPU
void initializeGpu(SimulationState& state) {
    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Triangles (positions + normals)
    std::vector<DTri> hostTris(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        hostTris[i].a = make_float3(t.a.x, t.a.y, t.a.z);
        hostTris[i].b = make_float3(t.b.x, t.b.y, t.b.z);
        hostTris[i].c = make_float3(t.c.x, t.c.y, t.c.z);
        hostTris[i].n = make_float3(t._normal.x, t._normal.y, t._normal.z);
    }

    // Flatten octree
    std::vector<GpuNode> nodes;
    std::vector<int> leafTris;
    flattenOctree(state.octree, nodes, leafTris);

    CUDA_CHECK(cudaMalloc(&state.d_tris, n * sizeof(DTri)));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, nodes.size() * sizeof(GpuNode)));
    CUDA_CHECK(cudaMalloc(&state.d_leafTris, std::max<size_t>(leafTris.size(), 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_kij, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_kw, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, T * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, T * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_dist, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(state.d_tris, hostTris.data(), n * sizeof(DTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, nodes.data(), nodes.size() * sizeof(GpuNode), cudaMemcpyHostToDevice));
    if (!leafTris.empty()) {
        CUDA_CHECK(cudaMemcpy(state.d_leafTris, leafTris.data(), leafTris.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), T * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, T * n * sizeof(float)));
}

void freeGpu(SimulationState& state) {
    cudaFree(state.d_tris);
    cudaFree(state.d_nodes);
    cudaFree(state.d_leafTris);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_kij);
    cudaFree(state.d_kw);
    cudaFree(state.d_tau);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_dist);
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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Upload everything to the GPU
    printf("Uploading data to GPU...\n");
    initializeGpu(state);
}

// ============================================================================
// Precomputation Phase (GPU)
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const int n = static_cast<int>(state.numTriangles);
    const long long total = static_cast<long long>(n) * n;
    const int block = 128;
    const long long grid = (total + block - 1) / block;

    kijKernel<<<static_cast<unsigned int>(grid), block>>>(
        state.d_tris, state.d_nodes, state.d_leafTris, state.d_kij, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij,
                          state.numTriangles * state.numTriangles * sizeof(float),
                          cudaMemcpyDeviceToHost));
    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const int n = static_cast<int>(state.numTriangles);
    const long long total = static_cast<long long>(n) * n;
    const int block = 256;
    const long long grid = (total + block - 1) / block;

    tauKernel<<<static_cast<unsigned int>(grid), block>>>(state.d_tris, state.d_tau, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.tau.data(), state.d_tau,
                          state.numTriangles * state.numTriangles * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Simulation Phase (Wave Propagation, GPU)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    // Precompute clamped propagation weights kw[i][j] = min(kij * area_j, 1)
    {
        const long long total = static_cast<long long>(n) * n;
        const int block = 256;
        const long long grid = (total + block - 1) / block;
        weightKernel<<<static_cast<unsigned int>(grid), block>>>(
            state.d_kij, state.d_areas, state.d_kw, n);
        CUDA_CHECK(cudaGetLastError());
    }

    // Timesteps are sequential (each depends on earlier radiosity history);
    // within a timestep all triangles update in parallel on the device.
    for (int t = 0; t < T; ++t) {
        simStepKernel<<<n, SIM_BLOCK>>>(state.d_kw, state.d_tau, state.d_rho,
                                        state.d_radE, state.d_radB, n, t);
        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                          state.numTimesteps * state.numTriangles * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation, GPU)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int block = 128;
    const int grid = (n + block - 1) / block;

    distanceKernel<<<grid, block>>>(state.d_radB, state.d_dist, n,
                                    static_cast<int>(state.numTimesteps),
                                    static_cast<int>(state.sourceIndex));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_dist,
                          state.numTriangles * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
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
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
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

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

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

    // Validation
    bool ok = true;
    if (validate) {
        ok = validateResults(state);
    }

    freeGpu(state);
    return ok ? 0 : 1;
}
