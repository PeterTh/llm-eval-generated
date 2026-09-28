/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how
 * sound/light waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility
 *    and geometry (Monte-Carlo ray sampling against an octree, on the GPU).
 * 2. Wave propagation using radiosity equations with time delays (Tau).
 * 3. Distance estimation via cross-correlation of radiosity values.
 *
 * Parallelization strategy:
 *  - MPI distributes triangles (rows i of the N x N problem) across ranks.
 *    Each rank owns a contiguous row range and is bound to one GPU.
 *  - CUDA kernels perform the heavy O(N^2), O(N^2*T) and O(N*T^2) numerical
 *    work for each rank's row range on its GPU.
 *  - OpenMP parallelizes host-side setup/reduction loops (mesh attribute
 *    computation, octree flattening prep, validation/hash reductions).
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
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

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t cuda_check_err__ = (call);                                    \
        if (cuda_check_err__ != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
                    cudaGetErrorString(cuda_check_err__), __FILE__, __LINE__);    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                         \
    } while (0)

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
};

// ============================================================================
// Flattened Octree (device-friendly, array-of-structs -> struct-of-arrays)
// ============================================================================

struct FlatOctreeHost {
    std::vector<float> centerX, centerY, centerZ;
    std::vector<float> halfX, halfY, halfZ;
    std::vector<int32_t> children;   // numNodes * 8, -1 if absent
    std::vector<int32_t> leafStart;  // -1 if internal node
    std::vector<int32_t> leafCount;  // 0 if internal node
    std::vector<int32_t> leafTriIdx; // concatenated leaf triangle indices
};

int flattenOctreeNode(const Octree& node, FlatOctreeHost& flat) {
    int myIdx = static_cast<int>(flat.centerX.size());
    flat.centerX.push_back(node.center.x);
    flat.centerY.push_back(node.center.y);
    flat.centerZ.push_back(node.center.z);
    flat.halfX.push_back(node.halfExtent.x);
    flat.halfY.push_back(node.halfExtent.y);
    flat.halfZ.push_back(node.halfExtent.z);
    flat.children.resize(flat.children.size() + 8, -1);

    if (!node.triangleIndices.empty()) {
        flat.leafStart.push_back(static_cast<int32_t>(flat.leafTriIdx.size()));
        flat.leafCount.push_back(static_cast<int32_t>(node.triangleIndices.size()));
        for (size_t idx : node.triangleIndices) {
            flat.leafTriIdx.push_back(static_cast<int32_t>(idx));
        }
    } else {
        flat.leafStart.push_back(-1);
        flat.leafCount.push_back(0);
    }

    for (int c = 0; c < 8; ++c) {
        if (node.children[c]) {
            int childIdx = flattenOctreeNode(*node.children[c], flat);
            flat.children[myIdx * 8 + c] = childIdx;
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
// CUDA device math helpers
// ============================================================================

__device__ __forceinline__ float3 f3add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 f3sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 f3scale(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 f3neg(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__device__ __forceinline__ float f3dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 f3cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Möller-Trumbore ray-triangle intersection
__device__ float dRayTriangleIntersect(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2) {
    float3 e1 = f3sub(v1, v0);
    float3 e2 = f3sub(v2, v0);
    float3 pvec = f3cross(dir, e2);
    float det = f3dot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    float3 tvec = f3sub(orig, v0);
    float u = f3dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    float3 qvec = f3cross(tvec, e1);
    float v = f3dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return f3dot(e2, qvec) * invDet;
}

// Ray-box overlap test (matches Octree::rayIntersectsBox)
__device__ bool dRayIntersectsBox(float3 p1, float3 p2, float3 center, float3 halfExtent) {
    float3 d = f3scale(f3sub(p2, p1), 0.5f);
    float3 c = f3sub(f3add(p1, d), center);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

__device__ __forceinline__ float dCosPhi(float3 v, float3 normal) {
    float vNorm = sqrtf(f3dot(v, v));
    if (vNorm <= EPSILON) return 0.0f;
    float c = f3dot(v, normal) / vNorm;
    return c > 0.0f ? c : 0.0f;
}

struct GPUOctreeArrays {
    const float* centerX; const float* centerY; const float* centerZ;
    const float* halfX;   const float* halfY;   const float* halfZ;
    const int32_t* children;
    const int32_t* leafStart;
    const int32_t* leafCount;
    const int32_t* leafTriIdx;
};

struct GPUTriArrays {
    const float3* a;
    const float3* b;
    const float3* c;
    const float3* normal;
};

// Iterative (stack-based) octree traversal: is the segment from-to blocked
// by any triangle other than srcIdx/dstIdx? Equivalent to isRayBlocked().
__device__ bool dIsRayBlocked(const GPUOctreeArrays& oct, const GPUTriArrays& tri,
                               float3 from, float3 to, int srcIdx, int dstIdx) {
    float3 dir = f3sub(to, from);
    float rayLen = sqrtf(f3dot(dir, dir));
    if (rayLen < EPSILON) return true;
    float3 dirNorm = f3scale(dir, 1.0f / rayLen);

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int node = stack[--sp];
        float3 center = make_float3(oct.centerX[node], oct.centerY[node], oct.centerZ[node]);
        float3 half = make_float3(oct.halfX[node], oct.halfY[node], oct.halfZ[node]);

        if (!dRayIntersectsBox(from, to, center, half)) continue;

        int lstart = oct.leafStart[node];
        if (lstart >= 0) {
            int lcount = oct.leafCount[node];
            for (int k = 0; k < lcount; ++k) {
                int idx = oct.leafTriIdx[lstart + k];
                if (idx == srcIdx || idx == dstIdx) continue;
                float dist = dRayTriangleIntersect(from, dirNorm, tri.a[idx], tri.b[idx], tri.c[idx]);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int child = oct.children[node * 8 + c];
                if (child >= 0 && sp < 64) stack[sp++] = child;
            }
        }
    }
    return false;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Form factor Kij computation (Monte-Carlo visibility sampling).
// One thread handles one (i, j) pair. Each thread seeds its own RNG stream
// from (i, j) so that the result is independent of scheduling order.
__global__ void kernelComputeFormFactors(int rowStart, int rowCount, int N,
                                          GPUOctreeArrays oct, GPUTriArrays tri,
                                          float* kijOut, unsigned long long* nonZeroCounter) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rowCount * (long long)N;
    if (tid >= total) return;

    int iLocal = (int)(tid / N);
    int j = (int)(tid % N);
    int i = rowStart + iLocal;

    if (i == j) { kijOut[tid] = 0.0f; return; }

    float3 ni = tri.normal[i];
    float3 nj = tri.normal[j];
    if (f3dot(ni, nj) > 0.99f) { kijOut[tid] = 0.0f; return; }

    curandStatePhilox4_32_10_t rng;
    curand_init(42ULL, (unsigned long long)i * (unsigned long long)N + (unsigned long long)j, 0, &rng);

    float3 ai = tri.a[i], bi = tri.b[i], ci = tri.c[i];
    float3 aj = tri.a[j], bj = tri.b[j], cj = tri.c[j];
    float3 abI = f3sub(bi, ai), acI = f3sub(ci, ai);
    float3 abJ = f3sub(bj, aj), acJ = f3sub(cj, aj);

    float kij = 0.0f;
    for (int r = 0; r < NUM_RAYS; ++r) {
        float u1 = curand_uniform(&rng);
        float v1 = curand_uniform(&rng);
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }
        float3 pI = f3add(ai, f3add(f3scale(abI, u1), f3scale(acI, v1)));

        float u2 = curand_uniform(&rng);
        float v2 = curand_uniform(&rng);
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        float3 pJ = f3add(aj, f3add(f3scale(abJ, u2), f3scale(acJ, v2)));

        if (dIsRayBlocked(oct, tri, pI, pJ, i, j)) continue;

        float3 v = f3sub(pJ, pI);
        float distSqr = f3dot(v, v);
        if (distSqr < EPSILON) continue;

        float cosPhiI = dCosPhi(v, ni);
        float cosPhiJ = dCosPhi(f3neg(v), nj);
        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }
    kij *= INV_NUM_RAYS;
    kijOut[tid] = kij;
    if (kij > EPSILON) atomicAdd(nonZeroCounter, 1ULL);
}

// Time delay (Tau) computation.
__global__ void kernelComputeTau(int rowStart, int rowCount, int N,
                                  const float3* __restrict__ centers, int32_t* tauOut) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rowCount * (long long)N;
    if (tid >= total) return;

    int iLocal = (int)(tid / N);
    int j = (int)(tid % N);
    int i = rowStart + iLocal;

    if (i == j) { tauOut[tid] = 0; return; }

    float3 ci = centers[i], cj = centers[j];
    float dx = ci.x - cj.x, dy = ci.y - cj.y, dz = ci.z - cj.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    tauOut[tid] = (int32_t)ceilf(dist * INV_WAVE_SPEED);
}

// One wave-propagation timestep for this rank's local rows. Depends only on
// radBFull rows strictly before t (tau >= 1 for i != j), so it is safe to
// compute all local i in parallel for a fixed t.
__global__ void kernelSimStep(int rowStart, int rowCount, int N, unsigned long long t,
                               const float* __restrict__ kijLocal, const int32_t* __restrict__ tauLocal,
                               const float* __restrict__ areas, const float* __restrict__ rho,
                               const float* __restrict__ radERow, const float* __restrict__ radBFull,
                               float* __restrict__ radBRowOut) {
    int iLocal = blockIdx.x * blockDim.x + threadIdx.x;
    if (iLocal >= rowCount) return;
    int i = rowStart + iLocal;

    const float* kijRow = kijLocal + (long long)iLocal * N;
    const int32_t* tauRow = tauLocal + (long long)iLocal * N;

    float sumB = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        int32_t tauij = tauRow[j];
        if ((long long)t < tauij) continue;

        float kij = kijRow[j];
        if (kij <= 0.0f) continue;

        unsigned long long srcTime = t - (unsigned long long)tauij;
        float radJ = radBFull[srcTime * (unsigned long long)N + j];
        if (radJ <= 0.0f) continue;

        float contrib = kij * areas[j];
        if (contrib > 1.0f) contrib = 1.0f;
        sumB += contrib * radJ;
    }

    radBRowOut[iLocal] = rho[i] * sumB + radERow[iLocal];
}

// Distance estimation via discrete cross-correlation against the source.
__global__ void kernelComputeDistances(int rowStart, int rowCount, int N, unsigned long long T,
                                        unsigned long long sourceIndex,
                                        const float* __restrict__ radBFull, float* __restrict__ distOut) {
    int iLocal = blockIdx.x * blockDim.x + threadIdx.x;
    if (iLocal >= rowCount) return;
    int i = rowStart + iLocal;

    float maxCorr = 0.0f;
    unsigned long long bestT = 0;

    for (unsigned long long t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (unsigned long long tt = t; tt < T; ++tt) {
            float pB = radBFull[tt * (unsigned long long)N + i];
            float pS = radBFull[(tt - t) * (unsigned long long)N + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distOut[iLocal] = WAVE_SPEED * (float)bestT;
}

// ============================================================================
// GPU resource context (device buffers), one per MPI rank
// ============================================================================

struct GPUContext {
    // Flattened octree
    float* d_centerX = nullptr; float* d_centerY = nullptr; float* d_centerZ = nullptr;
    float* d_halfX = nullptr;   float* d_halfY = nullptr;   float* d_halfZ = nullptr;
    int32_t* d_children = nullptr;
    int32_t* d_leafStart = nullptr;
    int32_t* d_leafCount = nullptr;
    int32_t* d_leafTriIdx = nullptr;

    // Triangle data (all N triangles, replicated per rank)
    float3* d_triA = nullptr;
    float3* d_triB = nullptr;
    float3* d_triC = nullptr;
    float3* d_triNormal = nullptr;
    float3* d_triCenter = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;

    // Local row-range matrices (rowCount x N)
    float* d_kijLocal = nullptr;
    int32_t* d_tauLocal = nullptr;

    // Radiosity
    float* d_radELocal = nullptr; // T x rowCount (only this rank's rows)
    float* d_radBFull = nullptr;  // T x N (replicated/synced across ranks)
    float* d_rowBuf = nullptr;    // rowCount scratch (sim output / distances output)

    unsigned long long* d_nonZeroCounter = nullptr;
};

void freeGPUContext(GPUContext& g) {
    cudaFree(g.d_centerX); cudaFree(g.d_centerY); cudaFree(g.d_centerZ);
    cudaFree(g.d_halfX);   cudaFree(g.d_halfY);   cudaFree(g.d_halfZ);
    cudaFree(g.d_children); cudaFree(g.d_leafStart); cudaFree(g.d_leafCount); cudaFree(g.d_leafTriIdx);
    cudaFree(g.d_triA); cudaFree(g.d_triB); cudaFree(g.d_triC); cudaFree(g.d_triNormal); cudaFree(g.d_triCenter);
    cudaFree(g.d_areas); cudaFree(g.d_rho);
    cudaFree(g.d_kijLocal); cudaFree(g.d_tauLocal);
    cudaFree(g.d_radELocal); cudaFree(g.d_radBFull); cudaFree(g.d_rowBuf);
    cudaFree(g.d_nonZeroCounter);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> distances;   // Computed distances from source (full, gathered)

    Octree octree;                  // Spatial acceleration structure (host)
    GPUContext gpu;                 // Device buffers for this rank

    size_t sourceIndex = 0;

    // MPI decomposition (row range of triangle indices owned by this rank)
    int rank = 0;
    int numRanks = 1;
    size_t rowStart = 0;
    size_t rowCount = 0;
    std::vector<int> recvCounts;
    std::vector<int> displs;

    unsigned long long nonZeroKij = 0;
};

// ============================================================================
// GPU Upload
// ============================================================================

void uploadToGPU(SimulationState& state, const std::vector<val_t>& radELocalHost) {
    GPUContext& g = state.gpu;
    const size_t N = state.numTriangles;
    const size_t rowCount = state.rowCount;
    const size_t T = state.numTimesteps;

    FlatOctreeHost flat;
    flattenOctreeNode(state.octree, flat);
    const size_t numNodes = flat.centerX.size();

    CUDA_CHECK(cudaMalloc(&g.d_centerX, numNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_centerY, numNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_centerZ, numNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_halfX, numNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_halfY, numNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_halfZ, numNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_children, numNodes * 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g.d_leafStart, numNodes * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g.d_leafCount, numNodes * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g.d_leafTriIdx, std::max<size_t>(flat.leafTriIdx.size(), 1) * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(g.d_centerX, flat.centerX.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_centerY, flat.centerY.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_centerZ, flat.centerZ.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_halfX, flat.halfX.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_halfY, flat.halfY.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_halfZ, flat.halfZ.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_children, flat.children.data(), numNodes * 8 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_leafStart, flat.leafStart.data(), numNodes * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_leafCount, flat.leafCount.data(), numNodes * sizeof(int32_t), cudaMemcpyHostToDevice));
    if (!flat.leafTriIdx.empty()) {
        CUDA_CHECK(cudaMemcpy(g.d_leafTriIdx, flat.leafTriIdx.data(), flat.leafTriIdx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    std::vector<float3> triA(N), triB(N), triC(N), triNormal(N), triCenter(N);
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        const Triangle& tr = state.triangles[i];
        triA[i] = make_float3(tr.a.x, tr.a.y, tr.a.z);
        triB[i] = make_float3(tr.b.x, tr.b.y, tr.b.z);
        triC[i] = make_float3(tr.c.x, tr.c.y, tr.c.z);
        Vec3 nrm = tr.normal();
        triNormal[i] = make_float3(nrm.x, nrm.y, nrm.z);
        Vec3 ctr = tr.center();
        triCenter[i] = make_float3(ctr.x, ctr.y, ctr.z);
    }

    CUDA_CHECK(cudaMalloc(&g.d_triA, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&g.d_triB, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&g.d_triC, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&g.d_triNormal, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&g.d_triCenter, N * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(g.d_triA, triA.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_triB, triB.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_triC, triC.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_triNormal, triNormal.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_triCenter, triCenter.data(), N * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&g.d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g.d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&g.d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g.d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&g.d_kijLocal, std::max<size_t>(rowCount * N, 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_tauLocal, std::max<size_t>(rowCount * N, 1) * sizeof(int32_t)));

    CUDA_CHECK(cudaMalloc(&g.d_radELocal, std::max<size_t>(T * rowCount, 1) * sizeof(float)));
    if (T * rowCount > 0) {
        CUDA_CHECK(cudaMemcpy(g.d_radELocal, radELocalHost.data(), T * rowCount * sizeof(float), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&g.d_radBFull, T * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(g.d_radBFull, 0, T * N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&g.d_rowBuf, std::max<size_t>(rowCount, 1) * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&g.d_nonZeroCounter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g.d_nonZeroCounter, 0, sizeof(unsigned long long)));
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (deterministic, identical on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (state.rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree redundantly on every rank (cheap relative to O(N^2) work,
    // avoids serializing a pointer-based tree over MPI).
    state.octree.build(state.triangles);

    // Initialize areas (OpenMP across triangles)
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.assign(state.numTriangles, reflectivity);
    state.distances.assign(state.numTriangles, ZERO);

    // Row decomposition of triangle indices across MPI ranks
    const size_t N = state.numTriangles;
    const size_t base = N / static_cast<size_t>(state.numRanks);
    const size_t rem = N % static_cast<size_t>(state.numRanks);
    state.recvCounts.resize(state.numRanks);
    state.displs.resize(state.numRanks);
    size_t offset = 0;
    for (int r = 0; r < state.numRanks; ++r) {
        size_t cnt = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        state.recvCounts[r] = static_cast<int>(cnt);
        state.displs[r] = static_cast<int>(offset);
        offset += cnt;
    }
    state.rowStart = static_cast<size_t>(state.displs[state.rank]);
    state.rowCount = static_cast<size_t>(state.recvCounts[state.rank]);

    // Source emission (active for first half of timesteps), local rows only
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    std::vector<val_t> radELocalHost(timesteps * state.rowCount, ZERO);
    if (state.sourceIndex >= state.rowStart && state.sourceIndex < state.rowStart + state.rowCount) {
        size_t localIdx = state.sourceIndex - state.rowStart;
        for (size_t t = timeOn; t < timeOff; ++t) {
            radELocalHost[t * state.rowCount + localIdx] = ONE;
        }
    }

    uploadToGPU(state, radELocalHost);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");
    GPUContext& g = state.gpu;
    const int N = static_cast<int>(state.numTriangles);
    const int rowCount = static_cast<int>(state.rowCount);
    const long long total = (long long)rowCount * (long long)N;

    if (total > 0) {
        GPUOctreeArrays oct{g.d_centerX, g.d_centerY, g.d_centerZ, g.d_halfX, g.d_halfY, g.d_halfZ,
                             g.d_children, g.d_leafStart, g.d_leafCount, g.d_leafTriIdx};
        GPUTriArrays tri{g.d_triA, g.d_triB, g.d_triC, g.d_triNormal};

        const int threads = 256;
        const long long blocks = (total + threads - 1) / threads;
        kernelComputeFormFactors<<<static_cast<unsigned int>(blocks), threads>>>(
            static_cast<int>(state.rowStart), rowCount, N, oct, tri, g.d_kijLocal, g.d_nonZeroCounter);
        CUDA_CHECK(cudaGetLastError());
    }

    unsigned long long localNonZero = 0;
    CUDA_CHECK(cudaMemcpy(&localNonZero, g.d_nonZeroCounter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    unsigned long long globalNonZero = 0;
    MPI_Allreduce(&localNonZero, &globalNonZero, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = globalNonZero;

    if (state.rank == 0) printf("  Form factor computation complete\n");
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    GPUContext& g = state.gpu;
    const int N = static_cast<int>(state.numTriangles);
    const int rowCount = static_cast<int>(state.rowCount);
    const long long total = (long long)rowCount * (long long)N;

    if (total > 0) {
        const int threads = 256;
        const long long blocks = (total + threads - 1) / threads;
        kernelComputeTau<<<static_cast<unsigned int>(blocks), threads>>>(
            static_cast<int>(state.rowStart), rowCount, N, g.d_triCenter, g.d_tauLocal);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");
    GPUContext& g = state.gpu;
    const int N = static_cast<int>(state.numTriangles);
    const int rowCount = static_cast<int>(state.rowCount);
    const size_t T = state.numTimesteps;

    std::vector<float> hostRowFull(N);
    std::vector<float> hostRowLocal(std::max(rowCount, 1));

    const int threads = 256;
    const int blocks = (rowCount + threads - 1) / threads;

    for (size_t t = 0; t < T; ++t) {
        if (rowCount > 0) {
            const float* radERow = g.d_radELocal + t * static_cast<size_t>(rowCount);
            kernelSimStep<<<blocks, threads>>>(
                static_cast<int>(state.rowStart), rowCount, N, static_cast<unsigned long long>(t),
                g.d_kijLocal, g.d_tauLocal, g.d_areas, g.d_rho, radERow, g.d_radBFull, g.d_rowBuf);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hostRowLocal.data(), g.d_rowBuf, static_cast<size_t>(rowCount) * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(rowCount > 0 ? hostRowLocal.data() : nullptr, rowCount, MPI_FLOAT,
                        hostRowFull.data(), state.recvCounts.data(), state.displs.data(), MPI_FLOAT,
                        MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(g.d_radBFull + t * static_cast<size_t>(N), hostRowFull.data(),
                               static_cast<size_t>(N) * sizeof(float), cudaMemcpyHostToDevice));

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    GPUContext& g = state.gpu;
    const int N = static_cast<int>(state.numTriangles);
    const int rowCount = static_cast<int>(state.rowCount);
    const size_t T = state.numTimesteps;

    std::vector<float> localDist(std::max(rowCount, 1), 0.0f);
    if (rowCount > 0) {
        const int threads = 256;
        const int blocks = (rowCount + threads - 1) / threads;
        kernelComputeDistances<<<blocks, threads>>>(
            static_cast<int>(state.rowStart), rowCount, N, static_cast<unsigned long long>(T),
            static_cast<unsigned long long>(state.sourceIndex), g.d_radBFull, g.d_rowBuf);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localDist.data(), g.d_rowBuf, static_cast<size_t>(rowCount) * sizeof(float),
                               cudaMemcpyDeviceToHost));
    }

    std::vector<float> fullDist(N);
    MPI_Allgatherv(rowCount > 0 ? localDist.data() : nullptr, rowCount, MPI_FLOAT,
                    fullDist.data(), state.recvCounts.data(), state.displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

    state.distances.assign(fullDist.begin(), fullDist.end());
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, const std::vector<val_t>& radBFullHost) {
    printf("\nValidation:\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < N; ++i) {
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(N));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, N);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    #pragma omp parallel for reduction(+:receivedEnergy)
    for (size_t i = 0; i < N; ++i) {
        bool received = false;
        for (size_t t = 0; t < T; ++t) {
            if (radBFullHost[t * N + i] > EPSILON) {
                received = true;
                break;
            }
        }
        if (received) receivedEnergy++;
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, N);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries), reduced across ranks
    size_t totalKijEntries = N * N;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           state.nonZeroKij, totalKijEntries,
           100.0 * static_cast<double>(state.nonZeroKij) / static_cast<double>(totalKijEntries));

    if (state.nonZeroKij == 0) {
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices visible: %d\n",
               numRanks, omp_get_max_threads(), deviceCount);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.numRanks = numRanks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (rank == 0) printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    long maxPreDuration = 0;
    MPI_Reduce(&preDuration, &maxPreDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
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

    if (rank == 0) {
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

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", maxDistDuration);
        printf("\n");
    }

    int exitCode = 0;

    if (rank == 0) {
        // Total time
        long totalTime = maxPreDuration + maxSimDuration + maxDistDuration;
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

        // Memory usage (aggregate across all ranks, matches original single-process metric)
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

        // Validation (needs the full T x N radiosity matrix, gathered from this rank's GPU)
        if (validate) {
            std::vector<val_t> radBFullHost(state.numTimesteps * state.numTriangles);
            CUDA_CHECK(cudaMemcpy(radBFullHost.data(), state.gpu.d_radBFull,
                                   radBFullHost.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
            if (!validateResults(state, radBFullHost)) {
                exitCode = 1;
            }
        }
    }

    freeGPUContext(state.gpu);

    int broadcastExitCode = exitCode;
    MPI_Bcast(&broadcastExitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return broadcastExitCode;
}
