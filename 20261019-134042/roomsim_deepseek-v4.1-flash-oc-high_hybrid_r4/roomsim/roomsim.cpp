/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of the radiosity-based room
 * impulse response simulation.
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
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

int g_world_rank = 0;
int g_world_size = 1;

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

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

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
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

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

        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

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

#pragma omp parallel for schedule(dynamic) if(indices.size() > 128)
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
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
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

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

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

        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// GPU Data Layout and Device Helpers
// ============================================================================

struct GpuOctreeNode {
    float3 center;
    float3 halfExtent;
    int firstTri;
    int numTris;
    int child[8];
};

__device__ __forceinline__ float3 d_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 d_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 d_mul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 d_div_number(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__device__ __forceinline__ float d_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 d_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float3 d_neg(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __forceinline__ bool d_rayIntersectsBox(float3 p1, float3 p2,
                                                   float3 center, float3 halfExtent) {
    float3 d = d_mul(d_sub(p2, p1), 0.5f);
    float3 c = d_sub(d_add(p1, d), center);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

__device__ __forceinline__ float d_rayTriangleIntersect(float3 orig, float3 dir,
                                                        float3 v0, float3 v1, float3 v2) {
    float3 e1 = d_sub(v1, v0);
    float3 e2 = d_sub(v2, v0);
    float3 pvec = d_cross(dir, e2);
    float det = d_dot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    float3 tvec = d_sub(orig, v0);
    float u = d_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    float3 qvec = d_cross(tvec, e1);
    float v = d_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return d_dot(e2, qvec) * invDet;
}

__device__ __forceinline__ bool d_isRayBlocked(float3 from, float3 to,
                                               const GpuOctreeNode* __restrict__ nodes,
                                               const int* __restrict__ triIdx,
                                               const float3* __restrict__ A,
                                               const float3* __restrict__ B,
                                               const float3* __restrict__ C,
                                               int srcTriIdx, int dstTriIdx) {
    float3 dir = d_sub(to, from);
    float rayLen = sqrtf(d_dot(dir, dir));
    if (rayLen < EPSILON) return true;
    float3 dirNorm = d_mul(dir, 1.0f / rayLen);

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];
        const GpuOctreeNode& nd = nodes[ni];

        if (nd.numTris > 0) {
            int base = nd.firstTri;
            for (int k = 0; k < nd.numTris; ++k) {
                int idx = triIdx[base + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                float dist = d_rayTriangleIntersect(from, dirNorm, A[idx], B[idx], C[idx]);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ch = nd.child[c];
                if (ch >= 0 && d_rayIntersectsBox(from, to, nodes[ch].center, nodes[ch].halfExtent)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

__device__ __forceinline__ float d_cosPhi(float3 v, float3 normal) {
    float vNorm = sqrtf(d_dot(v, v));
    if (vNorm <= EPSILON) return ZERO;
    float c = d_dot(v, normal) / vNorm;
    return c > ZERO ? c : ZERO;
}

__device__ __forceinline__ uint32_t d_pcg(uint32_t& state) {
    state = state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float d_rand(uint32_t& state) {
    return static_cast<float>(d_pcg(state) >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ uint32_t d_seed(int i, int j, int n) {
    uint64_t z = (static_cast<uint64_t>(static_cast<uint32_t>(i)) * 0x9E3779B97F4A7C15ULL)
               ^ (static_cast<uint64_t>(static_cast<uint32_t>(j)) * 0xD1B54A32D192ED03ULL)
               ^ (static_cast<uint64_t>(static_cast<uint32_t>(n)) << 32)
               ^ 0xFULL;
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    uint32_t s = static_cast<uint32_t>(z) ^ static_cast<uint32_t>(z >> 32);
    return s ? s : 0x9E3779B9u;
}

__device__ __forceinline__ float3 d_randomPointInTriangle(float3 a, float3 ab, float3 ac,
                                                          uint32_t& rng) {
    float u = d_rand(rng);
    float v = d_rand(rng);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return d_add(d_add(a, d_mul(ab, u)), d_mul(ac, v));
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeKijKernel(const float3* __restrict__ A,
                                 const float3* __restrict__ B,
                                 const float3* __restrict__ C,
                                 const float3* __restrict__ Nrm,
                                 const GpuOctreeNode* __restrict__ nodes,
                                 const int* __restrict__ triIdx,
                                 int n, int rowStart,
                                 float* __restrict__ kijOut,
                                 int* __restrict__ nonZeroCount) {
    int lr = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    int i = rowStart + lr;

    float kij = ZERO;
    bool nonzero = false;

    if (j < n && i != j) {
        float3 ni = Nrm[i];
        float3 nj = Nrm[j];
        if (d_dot(ni, nj) <= 0.99f) {
            float3 ai = A[i], bi = B[i], ci = C[i];
            float3 abi = d_sub(bi, ai);
            float3 aci = d_sub(ci, ai);
            float3 aj = A[j], bj = B[j], cj = C[j];
            float3 abj = d_sub(bj, aj);
            float3 acj = d_sub(cj, aj);

            uint32_t rng = d_seed(i, j, n);

            for (int r = 0; r < NUM_RAYS; ++r) {
                float3 pI = d_randomPointInTriangle(ai, abi, aci, rng);
                float3 pJ = d_randomPointInTriangle(aj, abj, acj, rng);

                if (d_isRayBlocked(pI, pJ, nodes, triIdx, A, B, C, i, j)) continue;

                float3 v = d_sub(pJ, pI);
                float distSqr = d_dot(v, v);
                if (distSqr < EPSILON) continue;

                float cosPhiI = d_cosPhi(v, ni);
                float cosPhiJ = d_cosPhi(d_neg(v), nj);

                if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

                kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
            kij *= INV_NUM_RAYS;
            nonzero = (kij > EPSILON);
        }
    }

    if (j < n) kijOut[static_cast<size_t>(lr) * n + j] = kij;

    __shared__ int blockNonZero;
    if (threadIdx.x == 0) blockNonZero = 0;
    __syncthreads();

    unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if ((threadIdx.x & 31) == 0) {
        int cnt = __popc(mask);
        if (cnt) atomicAdd(&blockNonZero, cnt);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockNonZero > 0) {
        atomicAdd(nonZeroCount, blockNonZero);
    }
}

__global__ void computeTauKernel(const float3* __restrict__ A,
                                 const float3* __restrict__ B,
                                 const float3* __restrict__ C,
                                 int n, int rowStart,
                                 int* __restrict__ tauOut) {
    int lr = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int i = rowStart + lr;
    float3 ci = d_div_number(d_add(d_add(A[i], B[i]), C[i]), 3.0f);
    float3 cj = d_div_number(d_add(d_add(A[j], B[j]), C[j]), 3.0f);
    float3 d = d_sub(ci, cj);
    float dist = sqrtf(d_dot(d, d));
    tauOut[static_cast<size_t>(lr) * n + j] =
        static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

__global__ void simulateKernel(const float* __restrict__ kij,
                               const int* __restrict__ tau,
                               const float* __restrict__ radB,
                               const float* __restrict__ radE,
                               const float* __restrict__ areas,
                               const float* __restrict__ rho,
                               int n, int rowStart, int t,
                               float* __restrict__ radBOut) {
    int lr = blockIdx.x;
    int i = rowStart + lr;
    int tid = threadIdx.x;

    float sum = ZERO;
    const float* krow = kij + static_cast<size_t>(lr) * n;
    const int* trow = tau + static_cast<size_t>(lr) * n;

    for (int j = tid; j < n; j += blockDim.x) {
        int tauij = trow[j];
        if (t < tauij) continue;
        float k = krow[j];
        if (k <= ZERO) continue;

        int srcTime = t - tauij;
        float radJ = radB[static_cast<size_t>(srcTime) * n + j];
        if (radJ <= ZERO) continue;

        float m = k * areas[j];
        if (m > ONE) m = ONE;
        sum += m * radJ;
    }

    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }

    __shared__ float warpSums[8];
    if ((tid & 31) == 0) warpSums[tid >> 5] = sum;
    __syncthreads();

    if (tid < 8) {
        float v = warpSums[tid];
        for (int off = 4; off > 0; off >>= 1) {
            v += __shfl_down_sync(0x000000ffu, v, off);
        }
        if (tid == 0) {
            radBOut[static_cast<size_t>(t) * n + i] =
                rho[i] * v + radE[static_cast<size_t>(t) * n + i];
        }
    }
}

__global__ void computeDistancesKernel(const float* __restrict__ radB,
                                       int n, int timesteps,
                                       int rowStart, int rowCount, int sourceIndex,
                                       float* __restrict__ distances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rowCount) return;
    int i = rowStart + idx;

    float maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < timesteps; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < timesteps; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * n + i];
            float pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[idx] = WAVE_SPEED * static_cast<float>(bestT);
}

// ============================================================================
// Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;

    Octree octree;

    size_t sourceIndex = 0;

    size_t rowStart = 0;
    size_t rowCount = 0;
    std::vector<int> rowCounts;
    std::vector<int> rowDispls;
    int localNonZeroKij = 0;

    float3* d_a = nullptr;
    float3* d_b = nullptr;
    float3* d_c = nullptr;
    float3* d_n = nullptr;
    GpuOctreeNode* d_nodes = nullptr;
    int* d_triIdx = nullptr;

    float* d_kij = nullptr;
    int* d_tau = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_dist = nullptr;
    int* d_nonZeroKij = nullptr;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

static void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// ============================================================================
// Octree Flattening
// ============================================================================

static int flattenOctree(const Octree& node,
                         std::vector<GpuOctreeNode>& nodes,
                         std::vector<int>& triIndices) {
    int myIndex = static_cast<int>(nodes.size());
    nodes.emplace_back();

    GpuOctreeNode tmp{};
    tmp.center = make_float3(node.center.x, node.center.y, node.center.z);
    tmp.halfExtent = make_float3(node.halfExtent.x, node.halfExtent.y, node.halfExtent.z);
    tmp.firstTri = 0;
    tmp.numTris = 0;
    for (int c = 0; c < 8; ++c) tmp.child[c] = -1;

    if (!node.triangleIndices.empty()) {
        tmp.firstTri = static_cast<int>(triIndices.size());
        tmp.numTris = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            triIndices.push_back(static_cast<int>(idx));
        }
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                tmp.child[c] = flattenOctree(*node.children[c], nodes, triIndices);
            }
        }
    }

    nodes[myIndex] = tmp;
    return myIndex;
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (g_world_rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    state.octree.build(state.triangles);

    size_t n = state.numTriangles;
    size_t base = n / static_cast<size_t>(g_world_size);
    size_t rem = n % static_cast<size_t>(g_world_size);
    state.rowStart = static_cast<size_t>(g_world_rank) * base +
                     std::min(static_cast<size_t>(g_world_rank), rem);
    state.rowCount = base + (static_cast<size_t>(g_world_rank) < rem ? 1 : 0);

    state.rowCounts.resize(g_world_size);
    state.rowDispls.resize(g_world_size);
    for (int r = 0; r < g_world_size; ++r) {
        state.rowCounts[r] = static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
        state.rowDispls[r] = static_cast<int>(static_cast<size_t>(r) * base +
                                             std::min(static_cast<size_t>(r), rem));
    }

    state.areas.resize(n);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.assign(n, reflectivity);

    state.radE.assign(timesteps * n, ZERO);
    state.radB.assign(timesteps * n, ZERO);
    state.distances.assign(n, ZERO);

    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    std::vector<float3> hA(n), hB(n), hC(n), hN(n);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Triangle& tri = state.triangles[i];
        hA[i] = make_float3(tri.a.x, tri.a.y, tri.a.z);
        hB[i] = make_float3(tri.b.x, tri.b.y, tri.b.z);
        hC[i] = make_float3(tri.c.x, tri.c.y, tri.c.z);
        hN[i] = make_float3(tri._normal.x, tri._normal.y, tri._normal.z);
    }

    cudaCheck(cudaMalloc(&state.d_a, n * sizeof(float3)), "cudaMalloc a");
    cudaCheck(cudaMalloc(&state.d_b, n * sizeof(float3)), "cudaMalloc b");
    cudaCheck(cudaMalloc(&state.d_c, n * sizeof(float3)), "cudaMalloc c");
    cudaCheck(cudaMalloc(&state.d_n, n * sizeof(float3)), "cudaMalloc n");
    cudaCheck(cudaMemcpy(state.d_a, hA.data(), n * sizeof(float3), cudaMemcpyHostToDevice), "copy a");
    cudaCheck(cudaMemcpy(state.d_b, hB.data(), n * sizeof(float3), cudaMemcpyHostToDevice), "copy b");
    cudaCheck(cudaMemcpy(state.d_c, hC.data(), n * sizeof(float3), cudaMemcpyHostToDevice), "copy c");
    cudaCheck(cudaMemcpy(state.d_n, hN.data(), n * sizeof(float3), cudaMemcpyHostToDevice), "copy n");

    std::vector<GpuOctreeNode> hNodes;
    std::vector<int> hTriIdx;
    hNodes.reserve(n * 2 + 8);
    hTriIdx.reserve(n);
    flattenOctree(state.octree, hNodes, hTriIdx);

    cudaCheck(cudaMalloc(&state.d_nodes, hNodes.size() * sizeof(GpuOctreeNode)), "cudaMalloc nodes");
    cudaCheck(cudaMalloc(&state.d_triIdx, hTriIdx.size() * sizeof(int)), "cudaMalloc triIdx");
    cudaCheck(cudaMemcpy(state.d_nodes, hNodes.data(), hNodes.size() * sizeof(GpuOctreeNode),
                         cudaMemcpyHostToDevice), "copy nodes");
    cudaCheck(cudaMemcpy(state.d_triIdx, hTriIdx.data(), hTriIdx.size() * sizeof(int),
                         cudaMemcpyHostToDevice), "copy triIdx");

    size_t matElems = state.rowCount * n;
    size_t matBytes = matElems * sizeof(float);
    size_t tauBytes = matElems * sizeof(int);
    cudaCheck(cudaMalloc(&state.d_kij, matBytes), "cudaMalloc kij");
    cudaCheck(cudaMalloc(&state.d_tau, tauBytes), "cudaMalloc tau");

    cudaCheck(cudaMalloc(&state.d_radE, timesteps * n * sizeof(float)), "cudaMalloc radE");
    cudaCheck(cudaMalloc(&state.d_radB, timesteps * n * sizeof(float)), "cudaMalloc radB");
    cudaCheck(cudaMalloc(&state.d_areas, n * sizeof(float)), "cudaMalloc areas");
    cudaCheck(cudaMalloc(&state.d_rho, n * sizeof(float)), "cudaMalloc rho");
    cudaCheck(cudaMalloc(&state.d_dist, state.rowCount * sizeof(float)), "cudaMalloc dist");
    cudaCheck(cudaMalloc(&state.d_nonZeroKij, sizeof(int)), "cudaMalloc nonZero");

    if (timesteps * n > 0) {
        cudaCheck(cudaMemcpy(state.d_radE, state.radE.data(), timesteps * n * sizeof(float),
                             cudaMemcpyHostToDevice), "copy radE");
        cudaCheck(cudaMemset(state.d_radB, 0, timesteps * n * sizeof(float)), "memset radB");
    }
    cudaCheck(cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice), "copy areas");
    cudaCheck(cudaMemcpy(state.d_rho, state.rho.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice), "copy rho");
}

static void freeDeviceState(SimulationState& state) {
    cudaFree(state.d_a);
    cudaFree(state.d_b);
    cudaFree(state.d_c);
    cudaFree(state.d_n);
    cudaFree(state.d_nodes);
    cudaFree(state.d_triIdx);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_dist);
    cudaFree(state.d_nonZeroKij);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    if (g_world_rank == 0) printf("Computing time delays (Tau)...\n");
    if (state.rowCount == 0) return;

    size_t n = state.numTriangles;
    dim3 block(256);
    dim3 grid(static_cast<unsigned>(state.rowCount),
              static_cast<unsigned>((n + block.x - 1) / block.x));

    computeTauKernel<<<grid, block>>>(state.d_a, state.d_b, state.d_c,
                                      static_cast<int>(n),
                                      static_cast<int>(state.rowStart),
                                      state.d_tau);
    cudaCheck(cudaGetLastError(), "computeTauKernel");
}

void computeFormFactors(SimulationState& state) {
    if (g_world_rank == 0) printf("Computing form factors (Kij)...\n");

    cudaCheck(cudaMemset(state.d_nonZeroKij, 0, sizeof(int)), "memset nonZero");
    state.localNonZeroKij = 0;
    if (state.rowCount == 0) return;

    size_t n = state.numTriangles;
    dim3 block(256);
    dim3 grid(static_cast<unsigned>(state.rowCount),
              static_cast<unsigned>((n + block.x - 1) / block.x));

    computeKijKernel<<<grid, block>>>(state.d_a, state.d_b, state.d_c, state.d_n,
                                      state.d_nodes, state.d_triIdx,
                                      static_cast<int>(n),
                                      static_cast<int>(state.rowStart),
                                      state.d_kij, state.d_nonZeroKij);
    cudaCheck(cudaGetLastError(), "computeKijKernel");
    cudaCheck(cudaMemcpy(&state.localNonZeroKij, state.d_nonZeroKij, sizeof(int),
                         cudaMemcpyDeviceToHost), "copy nonZero");

    if (g_world_rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, n);
            }
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (g_world_rank == 0) printf("Running wave propagation simulation...\n");

    size_t n = state.numTriangles;
    int timesteps = static_cast<int>(state.numTimesteps);

    for (int t = 0; t < timesteps; ++t) {
        if (state.rowCount > 0) {
            int block = 256;
            dim3 grid(static_cast<unsigned>(state.rowCount));
            simulateKernel<<<grid, block>>>(state.d_kij, state.d_tau, state.d_radB,
                                            state.d_radE, state.d_areas, state.d_rho,
                                            static_cast<int>(n),
                                            static_cast<int>(state.rowStart), t,
                                            state.d_radB);
            cudaCheck(cudaGetLastError(), "simulateKernel");
        }

        size_t rowOff = static_cast<size_t>(t) * n;
        if (state.rowCount > 0) {
            cudaCheck(cudaMemcpy(state.radB.data() + rowOff + state.rowStart,
                                 state.d_radB + rowOff + state.rowStart,
                                 state.rowCount * sizeof(float),
                                 cudaMemcpyDeviceToHost), "copy radB row local");
        }

        MPI_Allgatherv(MPI_IN_PLACE, state.rowCounts[g_world_rank], MPI_FLOAT,
                       state.radB.data() + rowOff,
                       state.rowCounts.data(), state.rowDispls.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

        if (n > 0) {
            cudaCheck(cudaMemcpy(state.d_radB + rowOff, state.radB.data() + rowOff,
                                 n * sizeof(float), cudaMemcpyHostToDevice),
                      "copy radB row back");
        }

        if (g_world_rank == 0 && (((t + 1) % 10 == 0) || (t + 1 == timesteps))) {
            printf("  Timestep %d/%d\n", t + 1, timesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (g_world_rank == 0) printf("Computing distances via cross-correlation...\n");

    std::vector<val_t> distLocal(state.rowCount, ZERO);

    if (state.rowCount > 0) {
        int block = 128;
        dim3 grid(static_cast<unsigned>((state.rowCount + block - 1) / block));
        computeDistancesKernel<<<grid, block>>>(state.d_radB,
                                                static_cast<int>(state.numTriangles),
                                                static_cast<int>(state.numTimesteps),
                                                static_cast<int>(state.rowStart),
                                                static_cast<int>(state.rowCount),
                                                static_cast<int>(state.sourceIndex),
                                                state.d_dist);
        cudaCheck(cudaGetLastError(), "computeDistancesKernel");
        cudaCheck(cudaMemcpy(distLocal.data(), state.d_dist,
                             state.rowCount * sizeof(float), cudaMemcpyDeviceToHost),
                  "copy dist");
    }

    MPI_Gatherv(distLocal.data(), state.rowCounts[g_world_rank], MPI_FLOAT,
                state.distances.data(), state.rowCounts.data(), state.rowDispls.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, int globalNonZeroKij) {
    printf("\nValidation:\n");

    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;
    int negativeCount = 0;

#pragma omp parallel for schedule(static) reduction(+:sumDist,nonZeroCount,negativeCount) \
    reduction(min:minDist) reduction(max:maxDist)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) negativeCount++;
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }
    bool allNonNegative = (negativeCount == 0);

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

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

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           globalNonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * globalNonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world_size);

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &localSize);
    }

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No CUDA device available: %s\n",
                cudaGetErrorString(devErr == cudaSuccess ? cudaErrorNoDevice : devErr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    if (getenv("OMP_NUM_THREADS") == nullptr && localSize > 0) {
        int procs = omp_get_num_procs();
        int threads = procs / localSize;
        if (threads < 1) threads = 1;
        omp_set_num_threads(threads);
    }

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

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
            if (g_world_rank == 0) printUsage(argv[0]);
            if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);
            MPI_Finalize();
            return 0;
        } else {
            if (g_world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (g_world_rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (g_world_rank == 0) printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (g_world_rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (g_world_rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (g_world_rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    int globalNonZeroKij = 0;
    MPI_Reduce(&state.localNonZeroKij, &globalNonZeroKij, 1, MPI_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (g_world_rank == 0) {
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

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

        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        if (validate) {
            if (!validateResults(state, globalNonZeroKij)) {
                exitCode = 1;
            }
        }
    }

    freeDeviceState(state);
    if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
