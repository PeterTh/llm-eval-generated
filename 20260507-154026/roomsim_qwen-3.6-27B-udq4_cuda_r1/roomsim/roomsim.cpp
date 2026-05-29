/**
 * Room Response Simulation Benchmark - CUDA Parallelized
 * 
 * This is a GPU-accelerated implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 * Parallelization strategy:
 *   - Form factors: 2D thread grid over (i,j) pairs
 *   - Time delays: 2D thread grid over (i,j) pairs
 *   - Simulation: 1D thread grid per timestep over triangles
 *   - Distances: 1D thread grid over triangles
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
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err_));               \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

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

// ============================================================================
// Vector and Triangle Types (CPU)
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
// GPU Data Structures
// ============================================================================

struct __align__(16) TriData {
    float ax, ay, az;
    float bx, by, bz;
    float cx, cy, cz;
    float nx, ny, nz;
};

struct __align__(64) OctreeNode {
    float minBound[3];
    float maxBound[3];
    float center[3];
    float halfExtent[3];
    int children[8];
    int triStart;
    int triCount;
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction) - CPU only
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
// Octree for Spatial Acceleration (CPU build, GPU traversal)
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
// Mesh Generation: Icosphere (CPU only)
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
// Random Number Generation (CPU only)
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// ============================================================================
// GPU Device Functions
// ============================================================================

__device__ __forceinline__ float d_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 d_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float d_norm(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// MÃ¶ller-Trumbore ray-triangle intersection
__device__ float rayTriangleIntersectGPU(float3 orig, float3 dir,
                                          float3 v0, float3 v1, float3 v2) {
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 pvec = d_cross(dir, e2);
    float det = d_dot(e1, pvec);

    if (fabsf(det) < EPSILON) return HUGE_VALF;

    float invDet = 1.0f / det;
    float3 tvec = make_float3(orig.x - v0.x, orig.y - v0.y, orig.z - v0.z);
    float u = d_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return HUGE_VALF;

    float3 qvec = d_cross(tvec, e1);
    float v = d_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return HUGE_VALF;

    return d_dot(e2, qvec) * invDet;
}

// Segment-box intersection test (matches CPU version exactly)
__device__ bool rayIntersectsBoxGPU(float3 p1, float3 p2, const OctreeNode& node) {
    float dx = (p2.x - p1.x) * 0.5f;
    float dy = (p2.y - p1.y) * 0.5f;
    float dz = (p2.z - p1.z) * 0.5f;
    float cx = p1.x + dx - node.center[0];
    float cy = p1.y + dy - node.center[1];
    float cz = p1.z + dz - node.center[2];
    float adx = fabsf(dx);
    float ady = fabsf(dy);
    float adz = fabsf(dz);

    if (fabsf(cx) > node.halfExtent[0] + adx) return false;
    if (fabsf(cy) > node.halfExtent[1] + ady) return false;
    if (fabsf(cz) > node.halfExtent[2] + adz) return false;

    if (fabsf(dy * cz - dz * cy) > node.halfExtent[1] * adz + node.halfExtent[2] * ady + EPSILON) return false;
    if (fabsf(dz * cx - dx * cz) > node.halfExtent[2] * adx + node.halfExtent[0] * adz + EPSILON) return false;
    if (fabsf(dx * cy - dy * cx) > node.halfExtent[0] * ady + node.halfExtent[1] * adx + EPSILON) return false;

    return true;
}

// Generate random point in triangle using pre-computed barycentric coords
__device__ float3 randomPointInTriangleGPU(const TriData& tri, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float abx = tri.bx - tri.ax;
    float aby = tri.by - tri.ay;
    float abz = tri.bz - tri.az;
    float acx = tri.cx - tri.ax;
    float acy = tri.cy - tri.ay;
    float acz = tri.cz - tri.az;
    return make_float3(
        tri.ax + abx * u + acx * v,
        tri.ay + aby * u + acy * v,
        tri.az + abz * u + acz * v
    );
}

// Iterative octree traversal for visibility check
__device__ bool isRayBlockedGPU(
    const TriData* triangles,
    const OctreeNode* nodes,
    const int* triIndices,
    float3 from, float3 to,
    int srcTriIdx, int dstTriIdx)
{
    float3 dir = make_float3(to.x - from.x, to.y - from.y, to.z - from.z);
    float rayLen = d_norm(dir);
    if (rayLen < EPSILON) return true;
    float3 dirNorm = make_float3(dir.x / rayLen, dir.y / rayLen, dir.z / rayLen);

    // Iterative DFS with fixed-size stack
    constexpr int MAX_STACK = 64;
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = 0;  // Push root

    while (sp > 0) {
        int nodeIdx = stack[--sp];
        const OctreeNode& node = nodes[nodeIdx];

        // Check bounding box
        if (!rayIntersectsBoxGPU(from, to, node)) continue;

        // Leaf node: check triangles
        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                int triIdx = triIndices[node.triStart + k];
                if (triIdx == srcTriIdx || triIdx == dstTriIdx) continue;

                const TriData& tri = triangles[triIdx];
                float3 v0 = make_float3(tri.ax, tri.ay, tri.az);
                float3 v1 = make_float3(tri.bx, tri.by, tri.bz);
                float3 v2 = make_float3(tri.cx, tri.cy, tri.cz);

                float dist = rayTriangleIntersectGPU(from, dirNorm, v0, v1, v2);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
            continue;
        }

        // Internal node: push children (reverse order for DFS)
        for (int i = 7; i >= 0; --i) {
            if (node.children[i] >= 0) {
                stack[sp++] = node.children[i];
            }
        }
    }

    return false;
}

// ============================================================================
// CUDA Kernel: Form Factor Computation
// ============================================================================

__global__ void computeFormFactorsKernel(
    const TriData* triangles,
    const OctreeNode* octreeNodes,
    const int* octreeTriIndices,
    const float* randomNumbers,
    const int* randOffsets,
    int N,
    float* kij)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * N;
    if (idx >= total) return;

    int i = static_cast<int>(idx / N);
    int j = static_cast<int>(idx % N);

    if (i == j) {
        kij[idx] = ZERO;
        return;
    }

    const TriData& triI = triangles[i];
    const TriData& triJ = triangles[j];

    // Cull triangles facing the same direction
    float normalDot = triI.nx * triJ.nx + triI.ny * triJ.ny + triI.nz * triJ.nz;
    if (normalDot > 0.99f) {
        kij[idx] = ZERO;
        return;
    }

    int randOffset = randOffsets[idx];
    if (randOffset < 0) {
        kij[idx] = ZERO;
        return;
    }

    float kijVal = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        int base = randOffset + r * 4;
        float uI = randomNumbers[base + 0];
        float vI = randomNumbers[base + 1];
        float uJ = randomNumbers[base + 2];
        float vJ = randomNumbers[base + 3];

        float3 pI = randomPointInTriangleGPU(triI, uI, vI);
        float3 pJ = randomPointInTriangleGPU(triJ, uJ, vJ);

        if (isRayBlockedGPU(triangles, octreeNodes, octreeTriIndices, pI, pJ, i, j))
            continue;

        float3 v = make_float3(pJ.x - pI.x, pJ.y - pI.y, pJ.z - pI.z);
        float distSqr = v.x * v.x + v.y * v.y + v.z * v.z;
        if (distSqr < EPSILON) continue;

        float vNorm = sqrtf(distSqr);
        float cosPhiI = fmaxf(ZERO, (v.x * triI.nx + v.y * triI.ny + v.z * triI.nz) / vNorm);
        float cosPhiJ = fmaxf(ZERO, (-v.x * triJ.nx - v.y * triJ.ny - v.z * triJ.nz) / vNorm);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kijVal += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[idx] = kijVal * INV_NUM_RAYS;
}

// ============================================================================
// CUDA Kernel: Time Delay Computation
// ============================================================================

__global__ void computeTimeDelaysKernel(
    const float* triCenters,
    int N,
    int* tau)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * N;
    if (idx >= total) return;

    int i = static_cast<int>(idx / N);
    int j = static_cast<int>(idx % N);

    if (i == j) {
        tau[idx] = 0;
        return;
    }

    float dx = triCenters[i * 3 + 0] - triCenters[j * 3 + 0];
    float dy = triCenters[i * 3 + 1] - triCenters[j * 3 + 1];
    float dz = triCenters[i * 3 + 2] - triCenters[j * 3 + 2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    tau[idx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// CUDA Kernel: Wave Propagation Simulation (one timestep)
// ============================================================================

__global__ void runSimulationKernel(
    int N, int T,
    int t,
    const float* areas,
    const float* rho,
    const float* kij,
    const int* tau,
    const float* radE,
    float* radB)
{
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) return;

    float sumB = ZERO;

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tau[i * N + j];

        if (t < tauij) continue;

        float kijVal = kij[i * N + j];
        if (kijVal <= ZERO) continue;

        int srcTime = t - tauij;
        float radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kijVal * areas[j], ONE) * radJ;
    }

    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

// ============================================================================
// CUDA Kernel: Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void computeDistancesKernel(
    int N, int T,
    const float* radB,
    int sourceIndex,
    float* distances)
{
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) return;

    float maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < T; ++t) {
        float sum = ZERO;

        for (int tt = t; tt < T; ++tt) {
            float pB = radB[tt * N + i];
            float pS = radB[(tt - t) * N + sourceIndex];
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
// Host Helpers
// ============================================================================

// Convert CPU Triangle to GPU TriData
__host__ TriData cpuToGpuTriangle(const Triangle& t) {
    TriData td;
    td.ax = t.a.x; td.ay = t.a.y; td.az = t.a.z;
    td.bx = t.b.x; td.by = t.b.y; td.bz = t.b.z;
    td.cx = t.c.x; td.cy = t.c.y; td.cz = t.c.z;
    td.nx = t._normal.x; td.ny = t._normal.y; td.nz = t._normal.z;
    return td;
}

// Flatten the pointer-based octree into linear arrays for GPU
static int flattenOctreeRecursive(const Octree& octree,
                                   std::vector<OctreeNode>& nodes,
                                   std::vector<int>& allTriIndices) {
    int idx = static_cast<int>(nodes.size());
    OctreeNode node;
    memset(&node, 0, sizeof(node));

    node.minBound[0] = octree.minBound.x;
    node.minBound[1] = octree.minBound.y;
    node.minBound[2] = octree.minBound.z;
    node.maxBound[0] = octree.maxBound.x;
    node.maxBound[1] = octree.maxBound.y;
    node.maxBound[2] = octree.maxBound.z;
    node.center[0] = octree.center.x;
    node.center[1] = octree.center.y;
    node.center[2] = octree.center.z;
    node.halfExtent[0] = octree.halfExtent.x;
    node.halfExtent[1] = octree.halfExtent.y;
    node.halfExtent[2] = octree.halfExtent.z;

    node.triStart = static_cast<int>(allTriIndices.size());
    node.triCount = static_cast<int>(octree.triangleIndices.size());
    for (size_t triIdx : octree.triangleIndices) {
        allTriIndices.push_back(static_cast<int>(triIdx));
    }

    for (int i = 0; i < 8; ++i) node.children[i] = -1;

    for (int i = 0; i < 8; ++i) {
        if (octree.children[i]) {
            node.children[i] = flattenOctreeRecursive(*octree.children[i], nodes, allTriIndices);
        }
    }

    nodes.push_back(node);
    return idx;
}

void flattenOctree(const Octree& octree,
                   std::vector<OctreeNode>& nodes,
                   std::vector<int>& allTriIndices) {
    nodes.clear();
    allTriIndices.clear();
    if (octree.allTriangles == nullptr || octree.allTriangles->empty()) return;
    flattenOctreeRecursive(octree, nodes, allTriIndices);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;
    std::vector<int> tau;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;

    Octree octree;

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    printf("Building octree...\n");
    state.octree.build(state.triangles);

    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.resize(state.numTriangles, reflectivity);

    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase (CUDA)
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    size_t N = state.numTriangles;

    // Compute triangle centers
    std::vector<float> triCenters(N * 3);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& tri = state.triangles[i];
        triCenters[i * 3 + 0] = (tri.a.x + tri.b.x + tri.c.x) / 3.0f;
        triCenters[i * 3 + 1] = (tri.a.y + tri.b.y + tri.c.y) / 3.0f;
        triCenters[i * 3 + 2] = (tri.a.z + tri.b.z + tri.c.z) / 3.0f;
    }

    // Allocate GPU memory
    float* d_triCenters = nullptr;
    int* d_tau = nullptr;
    CUDA_CHECK(cudaMalloc(&d_triCenters, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tau, N * N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_triCenters, triCenters.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = static_cast<int>((N * N + blockSize - 1) / blockSize);
    computeTimeDelaysKernel<<<gridSize, blockSize>>>(d_triCenters, static_cast<int>(N), d_tau);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(state.tau.data(), d_tau, N * N * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_triCenters));
    CUDA_CHECK(cudaFree(d_tau));
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    size_t N = state.numTriangles;

    // Pre-compute random numbers in the same order as the original sequential code
    RandomGenerator rng(42);
    std::vector<int> randOffsets(N * N, -1);
    size_t totalRands = 0;

    // First pass: determine which (i,j) pairs need random numbers
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const Triangle& triI = state.triangles[i];
            const Triangle& triJ = state.triangles[j];
            if (triI.normal().dot(triJ.normal()) > 0.99f) continue;
            randOffsets[i * N + j] = static_cast<int>(totalRands);
            totalRands += 4 * NUM_RAYS;
        }
    }

    // Second pass: generate random numbers
    std::vector<float> randomNumbers(totalRands);
    size_t offset = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const Triangle& triI = state.triangles[i];
            const Triangle& triJ = state.triangles[j];
            if (triI.normal().dot(triJ.normal()) > 0.99f) continue;
            for (int r = 0; r < NUM_RAYS; ++r) {
                randomNumbers[offset++] = rng.rand();
                randomNumbers[offset++] = rng.rand();
                randomNumbers[offset++] = rng.rand();
                randomNumbers[offset++] = rng.rand();
            }
        }
    }

    // Flatten octree for GPU
    std::vector<OctreeNode> octreeNodes;
    std::vector<int> octreeTriIndices;
    flattenOctree(state.octree, octreeNodes, octreeTriIndices);

    // Convert triangles to GPU format
    std::vector<TriData> triData(N);
    for (size_t i = 0; i < N; ++i) {
        triData[i] = cpuToGpuTriangle(state.triangles[i]);
    }

    // Allocate GPU memory
    TriData* d_triangles = nullptr;
    OctreeNode* d_octreeNodes = nullptr;
    int* d_octreeTriIndices = nullptr;
    float* d_randomNumbers = nullptr;
    int* d_randOffsets = nullptr;
    float* d_kij = nullptr;

    CUDA_CHECK(cudaMalloc(&d_triangles, N * sizeof(TriData)));
    CUDA_CHECK(cudaMalloc(&d_octreeNodes, octreeNodes.size() * sizeof(OctreeNode)));
    CUDA_CHECK(cudaMalloc(&d_octreeTriIndices, octreeTriIndices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_randomNumbers, totalRands * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_randOffsets, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_kij, N * N * sizeof(float)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_triangles, triData.data(), N * sizeof(TriData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_octreeNodes, octreeNodes.data(), octreeNodes.size() * sizeof(OctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_octreeTriIndices, octreeTriIndices.data(), octreeTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_randomNumbers, randomNumbers.data(), totalRands * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_randOffsets, randOffsets.data(), N * N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = static_cast<int>((N * N + blockSize - 1) / blockSize);
    computeFormFactorsKernel<<<gridSize, blockSize>>>(
        d_triangles, d_octreeNodes, d_octreeTriIndices,
        d_randomNumbers, d_randOffsets, static_cast<int>(N), d_kij);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(state.kij.data(), d_kij, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_octreeNodes));
    CUDA_CHECK(cudaFree(d_octreeTriIndices));
    CUDA_CHECK(cudaFree(d_randomNumbers));
    CUDA_CHECK(cudaFree(d_randOffsets));
    CUDA_CHECK(cudaFree(d_kij));

    printf("  Form factors computed on GPU.\n");
}

// ============================================================================
// Simulation Phase (CUDA)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    // Allocate GPU memory
    float* d_kij = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    int* d_tau = nullptr;

    CUDA_CHECK(cudaMalloc(&d_kij, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radE, T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tau, N * N * sizeof(int)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_kij, state.kij.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), T * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_radB, 0, T * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tau, state.tau.data(), N * N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch one kernel per timestep
    int blockSize = 256;
    int gridSize = static_cast<int>((N + blockSize - 1) / blockSize);

    for (size_t t = 0; t < T; ++t) {
        runSimulationKernel<<<gridSize, blockSize>>>(
            static_cast<int>(N), static_cast<int>(T),
            static_cast<int>(t),
            d_areas, d_rho, d_kij, d_tau, d_radE, d_radB);

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB, T * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_radB));
    CUDA_CHECK(cudaFree(d_tau));
}

// ============================================================================
// Distance Computation (CUDA)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    float* d_radB = nullptr;
    float* d_distances = nullptr;

    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), T * N * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = static_cast<int>((N + blockSize - 1) / blockSize);
    computeDistancesKernel<<<gridSize, blockSize>>>(
        static_cast<int>(N), static_cast<int>(T),
        d_radB, static_cast<int>(state.sourceIndex), d_distances);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), d_distances, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_radB));
    CUDA_CHECK(cudaFree(d_distances));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

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

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

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
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    if (validate) {
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
