/**
 * Room Response Simulation Benchmark
 *
 * CUDA implementation of a room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * All of the O(N^2) and O(N^2*T) workloads (form factors, time delays, wave
 * propagation and cross-correlation) execute on the GPU.  The random samples
 * used by the form-factor estimator are still generated on the host in the
 * original order so that the numerical results match the sequential reference
 * implementation as closely as possible.
 */

#include <algorithm>
#include <array>
#include <cfloat>
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
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// Vector and Triangle Types (host)
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
// Octree for Spatial Acceleration (host construction)
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
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
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
// Flattened device-side octree representation
// ============================================================================

struct DTri {
    float3 a, b, c;
    float3 n;
    float3 center;
};

struct DNode {
    float3 minBound, maxBound;
    float3 center, halfExtent;
    int child[8];
    int triStart;
    int triCount;
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
// Device helper functions
// ============================================================================

__device__ __forceinline__ float3 dadd(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 dsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 dmul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float ddot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 dcross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

// Möller-Trumbore ray-triangle intersection (same test as the reference)
__device__ __forceinline__ float rayTriangleDist(float3 orig, float3 dir, const DTri& tri) {
    float3 e1 = dsub(tri.b, tri.a);
    float3 e2 = dsub(tri.c, tri.a);
    float3 pvec = dcross(dir, e2);
    float det = ddot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    float3 tvec = dsub(orig, tri.a);
    float u = ddot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    float3 qvec = dcross(tvec, e1);
    float v = ddot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return ddot(e2, qvec) * invDet;
}

__device__ __forceinline__ bool rayIntersectsBox(float3 p1, float3 p2, const DNode& node) {
    float3 d = dmul(dsub(p2, p1), 0.5f);
    float3 c = dsub(dadd(p1, d), node.center);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) >
        node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Iterative octree traversal: returns true if any triangle blocks the segment.
__device__ __forceinline__ bool isRayBlockedDevice(float3 from, float3 to,
                                                   const DTri* tris,
                                                   const DNode* nodes,
                                                   const int* triPool,
                                                   int srcTriIdx, int dstTriIdx) {
    float3 dir = dsub(to, from);
    float rayLen = sqrtf(ddot(dir, dir));
    if (rayLen < EPSILON) return true;
    float3 dirNorm = dmul(dir, 1.0f / rayLen);

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const DNode& node = nodes[stack[--sp]];

        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                int idx = triPool[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                float dist = rayTriangleDist(from, dirNorm, tris[idx]);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ci = node.child[c];
                if (ci >= 0 && rayIntersectsBox(from, to, nodes[ci])) {
                    if (sp < 128) stack[sp++] = ci;
                }
            }
        }
    }

    return false;
}

__device__ __forceinline__ float cosPhiDevice(float3 v, float3 normal) {
    float vNorm = sqrtf(ddot(v, v));
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, ddot(v, normal) / vNorm);
}

// ============================================================================
// CUDA kernels
// ============================================================================

// Form factors for a batch of rows.  Each half-warp (16 lanes) computes the
// 16 rays of one (i, j) pair and reduces them with warp shuffles.
__global__ void kijBatchKernel(const DTri* __restrict__ tris,
                               const DNode* __restrict__ nodes,
                               const int* __restrict__ triPool,
                               const float* __restrict__ samples,
                               int N, int rowStart, int rows,
                               float* __restrict__ kij,
                               int* __restrict__ nonZeroKij) {
    const int lane = static_cast<int>(threadIdx.x) & 15;
    const int group = static_cast<int>(threadIdx.x) >> 4;
    const int groupsPerBlock = static_cast<int>(blockDim.x) >> 4;

    const int pair = static_cast<int>(blockIdx.x) * groupsPerBlock + group;
    const int total = rows * N;
    const bool active = pair < total;

    int i = 0;
    int j = 0;
    if (active) {
        i = rowStart + pair / N;
        j = pair % N;
    }

    float contrib = ZERO;

    if (active && i != j) {
        const DTri ti = tris[i];
        const DTri tj = tris[j];

        // Cull triangles facing the same direction
        if (ddot(ti.n, tj.n) <= 0.99f) {
            const float* s = samples + static_cast<size_t>(pair) * (NUM_RAYS * 6);
            const int r = lane;

            float3 pI = make_float3(s[r * 6 + 0], s[r * 6 + 1], s[r * 6 + 2]);
            float3 pJ = make_float3(s[r * 6 + 3], s[r * 6 + 4], s[r * 6 + 5]);

            if (!isRayBlockedDevice(pI, pJ, tris, nodes, triPool, i, j)) {
                float3 v = dsub(pJ, pI);
                float distSqr = ddot(v, v);

                if (distSqr >= EPSILON) {
                    float cosPhiI = cosPhiDevice(v, ti.n);
                    float cosPhiJ = cosPhiDevice(dmul(v, -1.0f), tj.n);

                    if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                        contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
                    }
                }
            }
        }
    }

    // Sum the 16 rays belonging to this pair (stays within each half-warp).
    for (int offset = 8; offset > 0; offset >>= 1) {
        contrib += __shfl_down_sync(0xffffffffu, contrib, offset, 16);
    }

    if (active && lane == 0) {
        float result = (i == j) ? ZERO : contrib * INV_NUM_RAYS;
        kij[static_cast<size_t>(i) * N + j] = result;
        if (result > EPSILON) {
            atomicAdd(nonZeroKij, 1);
        }
    }
}

__global__ void tauKernel(const DTri* __restrict__ tris, int N,
                          int* __restrict__ tau) {
    int j = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);
    int i = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) +
            static_cast<int>(threadIdx.y);
    if (i >= N || j >= N) return;

    size_t idx = static_cast<size_t>(i) * N + j;
    if (i == j) {
        tau[idx] = 0;
        return;
    }

    float3 d = dsub(tris[i].center, tris[j].center);
    float dist = sqrtf(ddot(d, d));
    tau[idx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

__global__ void simulationStepKernel(const float* __restrict__ kij,
                                     const int* __restrict__ tau,
                                     const float* __restrict__ areas,
                                     const float* __restrict__ rho,
                                     const float* __restrict__ radE,
                                     float* __restrict__ radB,
                                     int N, int t) {
    int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);
    if (i >= N) return;

    float sumB = ZERO;
    const size_t row = static_cast<size_t>(i) * N;

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tau[row + j];
        if (t < tauij) continue;

        float k = kij[row + j];
        if (k <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t) - static_cast<size_t>(tauij);
        float radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    radB[static_cast<size_t>(t) * N + i] = rho[i] * sumB + radE[static_cast<size_t>(t) * N + i];
}

__global__ void distanceKernel(const float* __restrict__ radB,
                               int N, int T, int source,
                               float* __restrict__ distances) {
    int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);
    if (i >= N) return;

    float maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < T; ++t) {
        float sum = ZERO;

        for (int tt = t; tt < T; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * N + i];
            float pS = radB[static_cast<size_t>(tt - t) * N + source];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED * static_cast<float>(bestT);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct DeviceData {
    DTri* tris = nullptr;
    DNode* nodes = nullptr;
    int* triPool = nullptr;
    float* kij = nullptr;
    int* tau = nullptr;
    float* areas = nullptr;
    float* rho = nullptr;
    float* radE = nullptr;
    float* radB = nullptr;
    float* distances = nullptr;
    float* samples = nullptr;
    int* nonZeroKij = nullptr;
    int sampleBatchRows = 1;
};

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity, only downloaded for validation
    std::vector<val_t> distances;   // Computed distances from source
    int nonZeroKij = 0;             // Number of Kij entries > EPSILON

    Octree octree;                  // Spatial acceleration structure
    DeviceData dev;                 // GPU buffers

    size_t sourceIndex = 0;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Device buffer management
// ============================================================================

int getSampleBatchRows(int N) {
    const size_t budget = 64ull * 1024ull * 1024ull;
    size_t bytesPerRow = static_cast<size_t>(N) * NUM_RAYS * 6 * sizeof(float);
    size_t rows = bytesPerRow > 0 ? budget / bytesPerRow : 1;
    if (rows < 1) rows = 1;
    if (rows > static_cast<size_t>(N)) rows = static_cast<size_t>(N);
    return static_cast<int>(rows);
}

int flattenOctree(const Octree& node, std::vector<DNode>& nodes,
                  std::vector<int>& triPool) {
    const int index = static_cast<int>(nodes.size());
    nodes.emplace_back();

    DNode& dn = nodes[index];
    dn.minBound = make_float3(node.minBound.x, node.minBound.y, node.minBound.z);
    dn.maxBound = make_float3(node.maxBound.x, node.maxBound.y, node.maxBound.z);
    dn.center = make_float3(node.center.x, node.center.y, node.center.z);
    dn.halfExtent = make_float3(node.halfExtent.x, node.halfExtent.y, node.halfExtent.z);
    dn.triStart = -1;
    dn.triCount = 0;
    for (int k = 0; k < 8; ++k) dn.child[k] = -1;

    if (!node.triangleIndices.empty()) {
        dn.triStart = static_cast<int>(triPool.size());
        dn.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            triPool.push_back(static_cast<int>(idx));
        }
    } else {
        for (int k = 0; k < 8; ++k) {
            if (node.children[k]) {
                int childIndex = flattenOctree(*node.children[k], nodes, triPool);
                // nodes may have been reallocated by the recursive call
                nodes[index].child[k] = childIndex;
            }
        }
    }
    return index;
}

void allocateDevice(SimulationState& state) {
    const int N = static_cast<int>(state.numTriangles);
    const size_t T = state.numTimesteps;

    std::vector<DTri> deviceTris(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 c = t.center();
        deviceTris[i].a = make_float3(t.a.x, t.a.y, t.a.z);
        deviceTris[i].b = make_float3(t.b.x, t.b.y, t.b.z);
        deviceTris[i].c = make_float3(t.c.x, t.c.y, t.c.z);
        deviceTris[i].n = make_float3(t._normal.x, t._normal.y, t._normal.z);
        deviceTris[i].center = make_float3(c.x, c.y, c.z);
    }

    std::vector<DNode> nodes;
    std::vector<int> triPool;
    if (N > 0) {
        flattenOctree(state.octree, nodes, triPool);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.tris),
                          static_cast<size_t>(N) * sizeof(DTri)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.nodes),
                          nodes.size() * sizeof(DNode)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.triPool),
                          triPool.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.kij),
                          static_cast<size_t>(N) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.tau),
                          static_cast<size_t>(N) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.areas),
                          static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.rho),
                          static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.radE),
                          T * static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.radB),
                          T * static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.distances),
                          static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.nonZeroKij),
                          sizeof(int)));

    state.dev.sampleBatchRows = getSampleBatchRows(N);
    const size_t sampleCount = static_cast<size_t>(state.dev.sampleBatchRows) *
                               static_cast<size_t>(N) * NUM_RAYS * 6;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dev.samples),
                          sampleCount * sizeof(float)));

    if (N > 0) {
        CUDA_CHECK(cudaMemcpy(state.dev.tris, deviceTris.data(),
                              static_cast<size_t>(N) * sizeof(DTri),
                              cudaMemcpyHostToDevice));
        if (!nodes.empty()) {
            CUDA_CHECK(cudaMemcpy(state.dev.nodes, nodes.data(),
                                  nodes.size() * sizeof(DNode),
                                  cudaMemcpyHostToDevice));
        }
        if (!triPool.empty()) {
            CUDA_CHECK(cudaMemcpy(state.dev.triPool, triPool.data(),
                                  triPool.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(state.dev.areas, state.areas.data(),
                              static_cast<size_t>(N) * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.dev.rho, state.rho.data(),
                              static_cast<size_t>(N) * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    if (T * static_cast<size_t>(N) > 0) {
        CUDA_CHECK(cudaMemcpy(state.dev.radE, state.radE.data(),
                              T * static_cast<size_t>(N) * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(state.dev.radB, 0,
                              T * static_cast<size_t>(N) * sizeof(float)));
    }
    CUDA_CHECK(cudaMemset(state.dev.nonZeroKij, 0, sizeof(int)));
}

void releaseDevice(SimulationState& state) {
    cudaFree(state.dev.tris);
    cudaFree(state.dev.nodes);
    cudaFree(state.dev.triPool);
    cudaFree(state.dev.kij);
    cudaFree(state.dev.tau);
    cudaFree(state.dev.areas);
    cudaFree(state.dev.rho);
    cudaFree(state.dev.radE);
    cudaFree(state.dev.radB);
    cudaFree(state.dev.distances);
    cudaFree(state.dev.samples);
    cudaFree(state.dev.nonZeroKij);
    state.dev = DeviceData{};
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

    // Initialize matrices on the host side (radiosities only; the large Kij
    // and Tau matrices live exclusively on the GPU).
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Upload geometry and allocate all GPU work buffers
    allocateDevice(state);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    RandomGenerator rng(42);

    const int N = static_cast<int>(state.numTriangles);
    const int batchRows = state.dev.sampleBatchRows;
    const size_t perPairFloats = static_cast<size_t>(NUM_RAYS) * 6;

    // The samples are produced in exactly the same order as the sequential
    // reference (row-major over pairs, ray-major within a pair, and no random
    // draws for culled pairs).
    std::vector<float> samples(static_cast<size_t>(batchRows) *
                               static_cast<size_t>(N) * perPairFloats);

    const int threads = 256;
    const int groupsPerBlock = threads / NUM_RAYS;

    for (int rowStart = 0; rowStart < N; rowStart += batchRows) {
        const int rows = std::min(batchRows, N - rowStart);

        for (int li = 0; li < rows; ++li) {
            const int i = rowStart + li;
            const Triangle& triI = state.triangles[i];

            for (int j = 0; j < N; ++j) {
                if (j == i) continue;

                const Triangle& triJ = state.triangles[j];
                if (triI.normal().dot(triJ.normal()) > 0.99f) continue;

                float* s = samples.data() +
                           (static_cast<size_t>(li) * N + j) * perPairFloats;

                for (int r = 0; r < NUM_RAYS; ++r) {
                    Vec3 pI = randomPointInTriangle(triI, rng);
                    Vec3 pJ = randomPointInTriangle(triJ, rng);

                    s[r * 6 + 0] = pI.x;
                    s[r * 6 + 1] = pI.y;
                    s[r * 6 + 2] = pI.z;
                    s[r * 6 + 3] = pJ.x;
                    s[r * 6 + 4] = pJ.y;
                    s[r * 6 + 5] = pJ.z;
                }
            }
        }

        const int pairCount = rows * N;
        CUDA_CHECK(cudaMemcpy(state.dev.samples, samples.data(),
                              static_cast<size_t>(pairCount) * perPairFloats * sizeof(float),
                              cudaMemcpyHostToDevice));

        const int blocks = (pairCount + groupsPerBlock - 1) / groupsPerBlock;
        kijBatchKernel<<<blocks, threads>>>(
            state.dev.tris, state.dev.nodes, state.dev.triPool,
            state.dev.samples, N, rowStart, rows,
            state.dev.kij, state.dev.nonZeroKij);
        CUDA_CHECK(cudaGetLastError());

        if ((rowStart + rows) % 100 == 0 || rowStart + rows == N) {
            printf("  Progress: %d/%d triangles\n", rowStart + rows, N);
        }
    }

    CUDA_CHECK(cudaMemcpy(&state.nonZeroKij, state.dev.nonZeroKij,
                          sizeof(int), cudaMemcpyDeviceToHost));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const int N = static_cast<int>(state.numTriangles);
    if (N == 0) return;

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    tauKernel<<<grid, block>>>(state.dev.tris, N, state.dev.tau);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int threads = 256;

    for (int t = 0; t < T; ++t) {
        const int blocks = (N + threads - 1) / threads;
        simulationStepKernel<<<blocks, threads>>>(
            state.dev.kij, state.dev.tau, state.dev.areas, state.dev.rho,
            state.dev.radE, state.dev.radB, N, t);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    if (N == 0) return;

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    distanceKernel<<<blocks, threads>>>(
        state.dev.radB, N, T, static_cast<int>(state.sourceIndex),
        state.dev.distances);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dev.distances,
                          static_cast<size_t>(N) * sizeof(float),
                          cudaMemcpyDeviceToHost));
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
    int nonZeroKij = state.nonZeroKij;
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
    if (validate) {
        state.radB.resize(t * n);
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dev.radB,
                              t * n * sizeof(float), cudaMemcpyDeviceToHost));
        if (!validateResults(state)) {
            releaseDevice(state);
            return 1;
        }
    }

    releaseDevice(state);
    return 0;
}
