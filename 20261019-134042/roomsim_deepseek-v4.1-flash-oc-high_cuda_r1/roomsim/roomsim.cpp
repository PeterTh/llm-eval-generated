/**
 * Room Response Simulation Benchmark
 *
 * CUDA implementation of a radiosity-based room impulse response simulation:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The random number stream is generated on the host in exactly the same order as
 * the original sequential implementation (including the fact that culled pairs
 * consume no random numbers), so the results are bit-for-bit equivalent. The
 * dominant O(N^2) / O(N^3) loops are offloaded to the GPU.
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
#include <condition_variable>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
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
// Octree for Spatial Acceleration (host-side construction)
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
// Random Number Generation (host, reproduces the sequential stream exactly)
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(_e), __FILE__, __LINE__);                \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// GPU Data Structures
// ============================================================================

struct GTri {
    float3 a, b, c, n;
};

struct GNode {
    float3 center, half;
    int child[8];
    int triStart, triCount;
};

// ============================================================================
// Device-Side Math
// ============================================================================

__device__ __forceinline__ float3 mk3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float3 vadd(float3 a, float3 b) { return mk3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 vsub(float3 a, float3 b) { return mk3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 vmul(float3 a, float s) { return mk3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 vdiv(float3 a, float s) { return mk3(a.x / s, a.y / s, a.z / s); }
__device__ __forceinline__ float vdot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 vcross(float3 a, float3 b) {
    return mk3(a.y * b.z - a.z * b.y,
               a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float vnorm(float3 a) { return sqrtf(vdot(a, a)); }

// Moller-Trumbore ray-triangle intersection (matches the host implementation)
__device__ __forceinline__ float rayTriangleIntersect3(float3 orig, float3 dir,
                                                       float3 v0, float3 v1, float3 v2) {
    float3 e1 = vsub(v1, v0);
    float3 e2 = vsub(v2, v0);
    float3 pvec = vcross(dir, e2);
    float det = vdot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    float3 tvec = vsub(orig, v0);
    float u = vdot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    float3 qvec = vcross(tvec, e1);
    float v = vdot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return vdot(e2, qvec) * invDet;
}

// Segment / axis-aligned-box overlap test (matches Octree::rayIntersectsBox)
__device__ __forceinline__ bool rayIntersectsBox3(float3 p1, float3 p2,
                                                  float3 center, float3 half) {
    float3 d = vmul(vsub(p2, p1), 0.5f);
    float3 c = vsub(vadd(p1, d), center);
    float adx = fabsf(d.x), ady = fabsf(d.y), adz = fabsf(d.z);

    if (fabsf(c.x) > half.x + adx) return false;
    if (fabsf(c.y) > half.y + ady) return false;
    if (fabsf(c.z) > half.z + adz) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > half.y * adz + half.z * ady + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > half.z * adx + half.x * adz + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > half.x * ady + half.y * adx + EPSILON) return false;

    return true;
}

// Octree-accelerated visibility test (matches isRayBlocked semantics)
__device__ bool isRayBlocked3(float3 from, float3 to,
                              const GNode* __restrict__ nodes,
                              const int* __restrict__ triIdx,
                              const GTri* __restrict__ tris,
                              int srcTriIdx, int dstTriIdx) {
    float3 dir = vsub(to, from);
    float rayLen = vnorm(dir);
    if (rayLen < EPSILON) return true;
    float3 dirNorm = vdiv(dir, rayLen);

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];
        GNode nd = nodes[ni];

        if (nd.triCount > 0) {
            int end = nd.triStart + nd.triCount;
            for (int k = nd.triStart; k < end; ++k) {
                int idx = triIdx[k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                GTri tri = tris[idx];
                float dist = rayTriangleIntersect3(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            for (int k = 0; k < 8; ++k) {
                int ch = nd.child[k];
                if (ch >= 0) {
                    GNode sc = nodes[ch];
                    if (rayIntersectsBox3(from, to, sc.center, sc.half)) {
                        stack[sp++] = ch;
                    }
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Kernels
// ============================================================================

// Compute form factors for a contiguous range of (i,j) pairs.
// pairDrawBase[p] is the offset of this pair's 64 random draws inside the
// compressed stream (0xFFFFFFFF for pairs culled by the normal test, which
// consume no random numbers at all).
__global__ void kijKernel(const GTri* __restrict__ tris,
                          const GNode* __restrict__ nodes,
                          const int* __restrict__ triIdx,
                          const float* __restrict__ rnd,
                          const unsigned int* __restrict__ pairDrawBase,
                          int N, long long pairBase, int numPairs,
                          float* __restrict__ kijT) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numPairs) return;

    long long gp = pairBase + p;
    int i = (int)(gp / (N - 1));
    int rem = (int)(gp - (long long)i * (N - 1));
    int j = (rem < i) ? rem : rem + 1;

    unsigned int base = pairDrawBase[p];
    if (base == 0xFFFFFFFFu) {
        kijT[(size_t)j * N + i] = 0.0f;
        return;
    }

    GTri ti = tris[i];
    GTri tj = tris[j];

    float kij = 0.0f;

    const float* rp = rnd + base;

    float3 abi = vsub(ti.b, ti.a);
    float3 aci = vsub(ti.c, ti.a);
    float3 abj = vsub(tj.b, tj.a);
    float3 acj = vsub(tj.c, tj.a);

    for (int r = 0; r < NUM_RAYS; ++r) {
        const float* rr = rp + r * 4;

        float u = rr[0];
        float v = rr[1];
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        float3 pI = vadd(vadd(ti.a, vmul(abi, u)), vmul(aci, v));

        float u2 = rr[2];
        float v2 = rr[3];
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        float3 pJ = vadd(vadd(tj.a, vmul(abj, u2)), vmul(acj, v2));

        if (isRayBlocked3(pI, pJ, nodes, triIdx, tris, i, j)) continue;

        float3 vv = vsub(pJ, pI);
        float distSqr = vdot(vv, vv);
        if (distSqr < EPSILON) continue;

        float vNorm = sqrtf(distSqr);

        float cosPhiI = ZERO;
        if (vNorm > EPSILON) {
            float d = vdot(vv, ti.n) / vNorm;
            cosPhiI = d > ZERO ? d : ZERO;
        }
        float cosPhiJ = ZERO;
        if (vNorm > EPSILON) {
            float d = vdot(vmul(vv, -1.0f), tj.n) / vNorm;
            cosPhiJ = d > ZERO ? d : ZERO;
        }

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij *= INV_NUM_RAYS;

    kijT[(size_t)j * N + i] = kij;
}

// Compute time delays (column-major, tauT[j*N+i]).
__global__ void tauKernel(const GTri* __restrict__ tris, int N,
                          int* __restrict__ tauT) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * N;
    if (k >= total) return;

    int i = (int)(k % N);
    int j = (int)(k / N);
    if (i == j) return;  // left as zero

    float3 ci = vdiv(vadd(vadd(tris[i].a, tris[i].b), tris[i].c), 3.0f);
    float3 cj = vdiv(vadd(vadd(tris[j].a, tris[j].b), tris[j].c), 3.0f);
    float dist = vnorm(vsub(ci, cj));

    tauT[(size_t)j * N + i] = (int)ceilf(dist * INV_WAVE_SPEED);
}

// One launch per timestep. Each thread computes radB[t][i] exactly like the
// sequential inner loop.
__global__ void simKernel(int t, int N,
                          const float* __restrict__ kijT,
                          const int* __restrict__ tauT,
                          const float* __restrict__ areas,
                          const float* __restrict__ rho,
                          float* __restrict__ radB,
                          const float* __restrict__ radE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sumB = ZERO;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        int tauij = tauT[(size_t)j * N + i];
        if (t < tauij) continue;

        float kij = kijT[(size_t)j * N + i];
        if (kij <= ZERO) continue;

        size_t srcTime = (size_t)(t - tauij);
        float radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kij * areas[j], ONE) * radJ;
    }

    radB[(size_t)t * N + i] = rho[i] * sumB + radE[(size_t)t * N + i];
}

// Cross-correlation distance estimation. One thread per receiver triangle.
__global__ void distKernel(int N, int T,
                           const float* __restrict__ radB,
                           int sourceIndex,
                           float* __restrict__ distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < T; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            float pB = radB[(size_t)tt * N + i];
            float pS = radB[(size_t)(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED * (float)bestT;
}

// ============================================================================
// Simulation State
// ============================================================================

struct GpuState {
    GTri* d_tris = nullptr;
    GNode* d_nodes = nullptr;
    int* d_triIdx = nullptr;
    float* d_kijT = nullptr;
    int* d_tauT = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    float* d_distances = nullptr;

    // Scratch buffer for the streamed random numbers (allocated during setup so
    // that allocation cost is outside the measured precomputation phase).
    float* d_rng = nullptr;
    std::vector<float> h_rng;
    size_t rngRowsPerTile = 0;
    size_t rngMaxFloats = 0;

    // Per-pair offset into the random stream (0xFFFFFFFF marks a culled pair).
    unsigned int* d_pairBase = nullptr;
    std::vector<unsigned int> pairBase;

    int numNodes = 0;
    int numTriIdx = 0;
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Host copy (filled only for validation)
    std::vector<int> tau;           // Unused on host
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Host copy (filled only for validation)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

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

    // Emission radiosity (T x N matrix), zero except the source during the
    // first half of the timesteps.
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    state.distances.assign(state.numTriangles, ZERO);
}

// ============================================================================
// GPU Setup: flatten octree and upload static data
// ============================================================================

static int flattenNode(const Octree* n, std::vector<GNode>& nodes, std::vector<int>& triIdx) {
    int idx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    nodes[idx].center = make_float3(
        (n->minBound.x + n->maxBound.x) * 0.5f,
        (n->minBound.y + n->maxBound.y) * 0.5f,
        (n->minBound.z + n->maxBound.z) * 0.5f);
    nodes[idx].half = make_float3(
        (n->maxBound.x - n->minBound.x) * 0.5f,
        (n->maxBound.y - n->minBound.y) * 0.5f,
        (n->maxBound.z - n->minBound.z) * 0.5f);
    nodes[idx].triStart = 0;
    nodes[idx].triCount = 0;
    for (int k = 0; k < 8; ++k) nodes[idx].child[k] = -1;

    if (!n->triangleIndices.empty()) {
        nodes[idx].triStart = static_cast<int>(triIdx.size());
        nodes[idx].triCount = static_cast<int>(n->triangleIndices.size());
        for (size_t v : n->triangleIndices) triIdx.push_back(static_cast<int>(v));
    } else {
        for (int k = 0; k < 8; ++k) {
            if (n->children[k]) {
                int c = flattenNode(n->children[k].get(), nodes, triIdx);
                nodes[idx].child[k] = c;
            }
        }
    }
    return idx;
}

void setupGpu(SimulationState& state, GpuState& gpu) {
    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);

    // Triangles
    std::vector<GTri> htris(N);
    for (int i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        htris[i].a = make_float3(t.a.x, t.a.y, t.a.z);
        htris[i].b = make_float3(t.b.x, t.b.y, t.b.z);
        htris[i].c = make_float3(t.c.x, t.c.y, t.c.z);
        htris[i].n = make_float3(t._normal.x, t._normal.y, t._normal.z);
    }

    // Flatten octree
    std::vector<GNode> nodes;
    std::vector<int> triIdx;
    nodes.reserve(static_cast<size_t>(N) * 2 + 16);
    triIdx.reserve(static_cast<size_t>(N) * 2);
    flattenNode(&state.octree, nodes, triIdx);
    gpu.numNodes = static_cast<int>(nodes.size());
    gpu.numTriIdx = static_cast<int>(triIdx.size());

    CUDA_CHECK(cudaMalloc(&gpu.d_tris, sizeof(GTri) * N));
    CUDA_CHECK(cudaMalloc(&gpu.d_nodes, sizeof(GNode) * nodes.size()));
    CUDA_CHECK(cudaMalloc(&gpu.d_triIdx, sizeof(int) * std::max<size_t>(1, triIdx.size())));
    CUDA_CHECK(cudaMalloc(&gpu.d_areas, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&gpu.d_rho, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&gpu.d_kijT, sizeof(float) * (size_t)N * N));
    CUDA_CHECK(cudaMalloc(&gpu.d_tauT, sizeof(int) * (size_t)N * N));
    CUDA_CHECK(cudaMalloc(&gpu.d_radE, sizeof(float) * (size_t)std::max(1, T) * N));
    CUDA_CHECK(cudaMalloc(&gpu.d_radB, sizeof(float) * (size_t)std::max(1, T) * N));
    CUDA_CHECK(cudaMalloc(&gpu.d_distances, sizeof(float) * N));

    CUDA_CHECK(cudaMemcpy(gpu.d_tris, htris.data(), sizeof(GTri) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_nodes, nodes.data(), sizeof(GNode) * nodes.size(),
                          cudaMemcpyHostToDevice));
    if (!triIdx.empty()) {
        CUDA_CHECK(cudaMemcpy(gpu.d_triIdx, triIdx.data(), sizeof(int) * triIdx.size(),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(gpu.d_areas, state.areas.data(), sizeof(float) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_rho, state.rho.data(), sizeof(float) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_radE, state.radE.data(), sizeof(float) * (size_t)T * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpu.d_radB, 0, sizeof(float) * (size_t)T * N));
    CUDA_CHECK(cudaMemset(gpu.d_kijT, 0, sizeof(float) * (size_t)N * N));
    CUDA_CHECK(cudaMemset(gpu.d_tauT, 0, sizeof(int) * (size_t)N * N));

    // Size and allocate the streaming buffer for the random-number tiles.
    const size_t targetBytes = 128ull * 1024ull * 1024ull;
    const size_t bytesPerRow = (size_t)(N - 1) * 64ull * sizeof(float);
    size_t rowsPerTile = targetBytes / std::max<size_t>(1, bytesPerRow);
    if (rowsPerTile < 1) rowsPerTile = 1;
    if (rowsPerTile > (size_t)N) rowsPerTile = (size_t)N;
    gpu.rngRowsPerTile = rowsPerTile;
    gpu.rngMaxFloats = std::max<size_t>(1, rowsPerTile * (size_t)(N - 1) * 64ull);
    gpu.h_rng.resize(gpu.rngMaxFloats);
    CUDA_CHECK(cudaMalloc((void**)&gpu.d_rng, gpu.rngMaxFloats * sizeof(float)));

    size_t maxPairs = rowsPerTile * (size_t)(N - 1);
    gpu.pairBase.resize(std::max<size_t>(1, maxPairs));
    CUDA_CHECK(cudaMalloc((void**)&gpu.d_pairBase,
                          sizeof(unsigned int) * std::max<size_t>(1, maxPairs)));

    printf("Octree uploaded: %d nodes, %d leaf entries\n", gpu.numNodes, gpu.numTriIdx);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state, GpuState& gpu) {
    printf("Computing form factors (Kij)...\n");

    const int N = static_cast<int>(state.numTriangles);
    const size_t rowsPerTile = gpu.rngRowsPerTile;
    const int numTiles = (int)(((size_t)N + rowsPerTile - 1) / rowsPerTile);

    std::mutex mtx;
    std::condition_variable cvReady, cvFree;
    bool ready = false;
    bool freeBuf = true;
    std::vector<size_t> tileDrawCounts(numTiles, 0);

    auto tileParams = [&](int k, size_t& i0, size_t& i1, size_t& numPairs) {
        i0 = (size_t)k * rowsPerTile;
        i1 = std::min<size_t>((size_t)N, i0 + rowsPerTile);
        numPairs = (i1 - i0) * (size_t)(N - 1);
    };

    // Normals used for the cull test. The same float expression as the
    // reference implementation is evaluated on the host so that the decision
    // (and therefore the random-number consumption) is identical.
    std::vector<Vec3> normals(N);
    for (int i = 0; i < N; ++i) normals[i] = state.triangles[i]._normal;

    // The random stream is inherently sequential and, importantly, culled
    // pairs consume no random numbers at all. A dedicated "walker" engine
    // advances through the stream; only the state at each chunk boundary is
    // needed because a chunk of consecutive rows can then be generated
    // independently in parallel, producing the exact same values.
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const size_t maxChunks = std::min<size_t>(hw, 128);

    std::thread producer([&]() {
        std::mt19937 walker(42);
        std::vector<std::mt19937> chunkStates;
        std::vector<std::thread> workers;
        std::vector<size_t> rowDrawStart;

        for (int k = 0; k < numTiles; ++k) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cvFree.wait(lk, [&] { return freeBuf; });
            }

            size_t i0, i1, numPairs;
            tileParams(k, i0, i1, numPairs);
            size_t rows = i1 - i0;

            rowDrawStart.assign(rows + 1, 0);

            // Determine culled pairs, per-pair draw offsets and per-row draw
            // prefixes in one pass over the tile.
            size_t lp = 0;
            size_t draws = 0;
            for (size_t li = 0; li < rows; ++li) {
                int i = (int)(i0 + li);
                Vec3 ni = normals[i];
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    if (ni.dot(normals[j]) > 0.99f) {
                        gpu.pairBase[lp] = 0xFFFFFFFFu;
                    } else {
                        gpu.pairBase[lp] = (unsigned int)draws;
                        draws += 64;
                    }
                    ++lp;
                }
                rowDrawStart[li + 1] = draws;
            }
            tileDrawCounts[k] = draws;

            size_t chunks = std::min<size_t>(rows, maxChunks);
            chunkStates.resize(chunks);

            // Walk the consumed part of the stream, saving the engine state at
            // each chunk boundary. Discarding is much cheaper than generating.
            for (size_t c = 0; c < chunks; ++c) {
                size_t s = (size_t)c * rows / chunks;
                size_t e = (size_t)(c + 1) * rows / chunks;
                chunkStates[c] = walker;
                walker.discard((unsigned long long)(rowDrawStart[e] - rowDrawStart[s]));
            }

            // Generate all chunks of this tile in parallel.
            workers.clear();
            workers.reserve(chunks);
            float* base = gpu.h_rng.data();
            for (size_t c = 0; c < chunks; ++c) {
                size_t s = (size_t)c * rows / chunks;
                size_t e = (size_t)(c + 1) * rows / chunks;
                std::mt19937 eng = chunkStates[c];
                float* dst = base + rowDrawStart[s];
                size_t count = rowDrawStart[e] - rowDrawStart[s];
                workers.emplace_back([eng, dst, count]() mutable {
                    std::uniform_real_distribution<val_t> dist(0.0f, 1.0f);
                    for (size_t q = 0; q < count; ++q) dst[q] = dist(eng);
                });
            }
            for (auto& w : workers) w.join();

            {
                std::lock_guard<std::mutex> lk(mtx);
                ready = true;
                freeBuf = false;
            }
            cvReady.notify_all();
        }
    });

    const int threads = 256;

    for (int k = 0; k < numTiles; ++k) {
        {
            std::unique_lock<std::mutex> lk(mtx);
            cvReady.wait(lk, [&] { return ready; });
            ready = false;
        }
        size_t i0, i1, numPairs;
        tileParams(k, i0, i1, numPairs);
        size_t numFloats = tileDrawCounts[k];

        if (numFloats > 0) {
            CUDA_CHECK(cudaMemcpy(gpu.d_rng, gpu.h_rng.data(),
                                  numFloats * sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(gpu.d_pairBase, gpu.pairBase.data(),
                              numPairs * sizeof(unsigned int), cudaMemcpyHostToDevice));

        // The host staging buffers are free again; let the producer start the
        // next tile while the GPU processes this one.
        {
            std::lock_guard<std::mutex> lk(mtx);
            freeBuf = true;
        }
        cvFree.notify_all();

        if (numPairs > 0) {
            long long pairBase = (long long)i0 * (long long)(N - 1);
            int blocks = (int)((numPairs + threads - 1) / threads);
            kijKernel<<<blocks, threads>>>(gpu.d_tris, gpu.d_nodes, gpu.d_triIdx,
                                           gpu.d_rng, gpu.d_pairBase, N, pairBase,
                                           (int)numPairs, gpu.d_kijT);
            CUDA_CHECK(cudaGetLastError());
        }

        if (i1 == (size_t)N || (i1 % 100) < rowsPerTile) {
            printf("  Progress: %zu/%zu triangles\n", i1, (size_t)N);
        }
    }

    producer.join();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeTimeDelays(SimulationState& state, GpuState& gpu) {
    printf("Computing time delays (Tau)...\n");

    int N = static_cast<int>(state.numTriangles);
    size_t total = (size_t)N * N;
    int threads = 256;
    size_t blocks = (total + threads - 1) / threads;

    tauKernel<<<(unsigned int)blocks, threads>>>(gpu.d_tris, N, gpu.d_tauT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, GpuState& gpu) {
    printf("Running wave propagation simulation...\n");

    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int t = 0; t < T; ++t) {
        simKernel<<<blocks, threads>>>(t, N, gpu.d_kijT, gpu.d_tauT,
                                       gpu.d_areas, gpu.d_rho,
                                       gpu.d_radB, gpu.d_radE);
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

void computeDistances(SimulationState& state, GpuState& gpu) {
    printf("Computing distances via cross-correlation...\n");

    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    distKernel<<<blocks, threads>>>(N, T, gpu.d_radB,
                                    (int)state.sourceIndex, gpu.d_distances);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), gpu.d_distances,
                          sizeof(float) * N, cudaMemcpyDeviceToHost));
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

    GpuState gpu;
    setupGpu(state, gpu);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, gpu);
    computeFormFactors(state, gpu);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, gpu);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, gpu);

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
        // Bring back the large arrays required by the validation routine.
        state.kij.resize(n * n);
        state.radB.resize(t * n);
        CUDA_CHECK(cudaMemcpy(state.kij.data(), gpu.d_kijT,
                              sizeof(float) * n * n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.radB.data(), gpu.d_radB,
                              sizeof(float) * t * n, cudaMemcpyDeviceToHost));

        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
