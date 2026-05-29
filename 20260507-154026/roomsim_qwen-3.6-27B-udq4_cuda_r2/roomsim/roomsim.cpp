/**
 * Room Response Simulation Benchmark - CUDA Parallelized
 * 
 * Parallelizes the radiosity-based wave propagation simulation using CUDA:
 * 1. Form factors (Kij) - parallel over all triangle pairs
 * 2. Wave propagation - parallel over triangles per timestep
 * 3. Distance estimation - parallel over triangles
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// CUDA Vector and Triangle Types
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
    Vec3 a, b, c, _normal;

    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) * (ONE / 3.0f); }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }
};

// ============================================================================
// Host-side Octree (used only for host-side debugging/compatibility)
// ============================================================================

static bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
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
    val_t r = boxHalfSize.x * fabsf(triNormal.x) +
              boxHalfSize.y * fabsf(triNormal.y) +
              boxHalfSize.z * fabsf(triNormal.z);
    if (fabsf(d) > r) return false;

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                val_t p0 = crossAxis.dot(v0);
                val_t p1 = crossAxis.dot(v1);
                val_t p2 = crossAxis.dot(v2);
                auto [minP, maxP] = minMax3(p0, p1, p2);
                val_t rr = boxHalfSize.x * fabsf(crossAxis.x) +
                           boxHalfSize.y * fabsf(crossAxis.y) +
                           boxHalfSize.z * fabsf(crossAxis.z);
                if (minP > rr || maxP < -rr) return false;
            }
        }
    }
    return true;
}

class HostOctree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<HostOctree> children[8];
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;

    HostOctree() : allTriangles(nullptr) {}

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

        if (indices.size() <= 8 ||
            (maxBound - minBound).norm() < 0.5f) {
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

                children[i] = std::make_unique<HostOctree>();
                children[i]->allTriangles = allTriangles;
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
    }

public:
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
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
// Per-thread RNG (XORShift64) for GPU
// ============================================================================

__device__ __forceinline__ uint64_t xorshift64(uint64_t& state) {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}

__device__ __forceinline__ float xorshift64f(uint64_t& state) {
    return (xorshift64(state) >> 11) * (1.0f / (1ULL << 53));
}

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ Vec3 randomPointInTriangle(const Triangle& t, uint64_t& rngState) {
    float u = xorshift64f(rngState);
    float v = xorshift64f(rngState);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Brute-force visibility test on GPU (checks all triangles)
__device__ __forceinline__ bool isRayBlockedDevice(
    const Vec3& from, const Vec3& to,
    const Triangle* triangles,
    size_t numTriangles,
    size_t srcTriIdx, size_t dstTriIdx) 
{
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    for (size_t k = 0; k < numTriangles; k++) {
        if (k == srcTriIdx || k == dstTriIdx) continue;
        const Triangle& tri = triangles[k];
        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cp = v.dot(normal) / vNorm;
    return cp > ZERO ? cp : ZERO;
}

__device__ __forceinline__ val_t computeKijDevice(
    size_t idxI, size_t idxJ,
    const Triangle* triangles,
    size_t numTriangles,
    uint64_t rngState) 
{
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rngState);
        Vec3 pJ = randomPointInTriangle(triJ, rngState);

        if (isRayBlockedDevice(pI, pJ, triangles, numTriangles, idxI, idxJ)) continue;

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
// CUDA Kernels
// ============================================================================

// Kernel 1: Compute Form Factors (Kij) - parallel over (i,j) pairs
__global__ void computeKijKernel(
    size_t numTriangles,
    val_t* kij,
    const Triangle* triangles) 
{
    size_t total = numTriangles * numTriangles;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t i = idx / numTriangles;
    size_t j = idx % numTriangles;
    if (i == j) return;

    uint64_t rngState = 42ULL + (i * numTriangles + j) * 2654435761ULL;
    val_t result = computeKijDevice(i, j, triangles, numTriangles, rngState);
    kij[idx] = result;
}

// Kernel 2: Compute Time Delays (Tau) - parallel over (i,j) pairs
__global__ void computeTauKernel(
    size_t numTriangles,
    int* tau,
    const Triangle* triangles) 
{
    size_t total = numTriangles * numTriangles;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t i = idx / numTriangles;
    size_t j = idx % numTriangles;
    if (i == j) { tau[idx] = 0; return; }

    val_t dist = (triangles[i].center() - triangles[j].center()).norm();
    tau[idx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Kernel 3: Simulation timestep - parallel over triangles
__global__ void simulateKernel(
    size_t numTriangles,
    size_t timestep,
    const val_t* radB,
    const val_t* radE,
    val_t* radBOut,
    const val_t* rho,
    const val_t* kij,
    const int* tau,
    size_t n) 
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t sumB = ZERO;

    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;

        int tauij = tau[i * n + j];
        if (static_cast<int>(timestep) < tauij) continue;

        val_t k = kij[i * n + j];
        if (k <= ZERO) continue;

        size_t srcTime = timestep - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * n + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(k * rho[j], ONE) * radJ;
    }

    radBOut[timestep * n + i] = rho[i] * sumB + radE[timestep * n + i];
}

// Kernel 4: Distance computation - parallel over triangles
__global__ void computeDistancesKernel(
    size_t numTriangles,
    size_t numTimesteps,
    const val_t* radB,
    val_t* distances,
    size_t n,
    size_t t,
    size_t sourceIndex) 
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (size_t ts = 0; ts < t; ++ts) {
        val_t sum = ZERO;

        for (size_t tt = ts; tt < t; ++tt) {
            val_t pB = radB[tt * n + i];
            val_t pS = radB[(tt - ts) * n + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(ts);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Mesh Generation: Icosphere (host-only)
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
// Simulation State
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

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization (host)
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
// CUDA Execution
// ============================================================================

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void computeFormFactorsGPU(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");
    size_t n = state.numTriangles;
    size_t total = n * n;

    Triangle* dTriangles = nullptr;
    val_t* dkij = nullptr;

    checkCuda(cudaMalloc(&dTriangles, n * sizeof(Triangle)), "malloc triangles");
    checkCuda(cudaMalloc(&dkij, total * sizeof(val_t)), "malloc kij");
    checkCuda(cudaMemset(dkij, 0, total * sizeof(val_t)), "memset kij");
    checkCuda(cudaMemcpy(dTriangles, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice), "copy triangles");

    int blockSize = 256;
    int numBlocks = std::min((int)((total + blockSize - 1) / blockSize), 65535);

    computeKijKernel<<<numBlocks, blockSize>>>(n, dkij, dTriangles);
    checkCuda(cudaGetLastError(), "launch kij kernel");
    checkCuda(cudaDeviceSynchronize(), "sync kij kernel");

    checkCuda(cudaMemcpy(state.kij.data(), dkij, total * sizeof(val_t), cudaMemcpyDeviceToHost), "copy kij results");

    for (size_t i = 0; i < n; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == n) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, n);
        }
    }

    cudaFree(dTriangles);
    cudaFree(dkij);
}

void computeTimeDelaysGPU(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");
    size_t n = state.numTriangles;
    size_t total = n * n;

    Triangle* dTriangles = nullptr;
    int* dtau = nullptr;

    checkCuda(cudaMalloc(&dTriangles, n * sizeof(Triangle)), "malloc triangles");
    checkCuda(cudaMalloc(&dtau, total * sizeof(int)), "malloc tau");
    checkCuda(cudaMemset(dtau, 0, total * sizeof(int)), "memset tau");
    checkCuda(cudaMemcpy(dTriangles, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice), "copy triangles");

    int blockSize = 256;
    int numBlocks = std::min((int)((total + blockSize - 1) / blockSize), 65535);

    computeTauKernel<<<numBlocks, blockSize>>>(n, dtau, dTriangles);
    checkCuda(cudaGetLastError(), "launch tau kernel");
    checkCuda(cudaDeviceSynchronize(), "sync tau kernel");

    checkCuda(cudaMemcpy(state.tau.data(), dtau, total * sizeof(int), cudaMemcpyDeviceToHost), "copy tau results");

    cudaFree(dTriangles);
    cudaFree(dtau);
}

void runSimulationGPU(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;

    val_t* dRadB = nullptr;
    val_t* dRadBOut = nullptr;
    val_t* dRadE = nullptr;
    val_t* dRho = nullptr;
    val_t* dKij = nullptr;
    int* dTau = nullptr;

    size_t radSize = t * n * sizeof(val_t);
    checkCuda(cudaMalloc(&dRadB, radSize), "malloc radB");
    checkCuda(cudaMalloc(&dRadBOut, radSize), "malloc radBOut");
    checkCuda(cudaMalloc(&dRadE, radSize), "malloc radE");
    checkCuda(cudaMalloc(&dRho, n * sizeof(val_t)), "malloc rho");
    checkCuda(cudaMalloc(&dKij, n * n * sizeof(val_t)), "malloc kij");
    checkCuda(cudaMalloc(&dTau, n * n * sizeof(int)), "malloc tau");

    checkCuda(cudaMemcpy(dRadE, state.radE.data(), radSize, cudaMemcpyHostToDevice), "copy radE");
    checkCuda(cudaMemcpy(dRho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice), "copy rho");
    checkCuda(cudaMemcpy(dKij, state.kij.data(), n * n * sizeof(val_t), cudaMemcpyHostToDevice), "copy kij");
    checkCuda(cudaMemcpy(dTau, state.tau.data(), n * n * sizeof(int), cudaMemcpyHostToDevice), "copy tau");
    checkCuda(cudaMemset(dRadB, 0, radSize), "memset radB");
    checkCuda(cudaMemset(dRadBOut, 0, radSize), "memset radBOut");

    int blockSize = 256;
    int numBlocks = std::min((int)((n + blockSize - 1) / blockSize), 65535);

    for (size_t ts = 0; ts < t; ++ts) {
        simulateKernel<<<numBlocks, blockSize>>>(n, ts, dRadB, dRadE, dRadBOut, dRho, dKij, dTau, n);
        checkCuda(cudaGetLastError(), "launch simulate kernel");
        checkCuda(cudaDeviceSynchronize(), "sync simulate kernel");

        std::swap(dRadB, dRadBOut);

        if ((ts + 1) % 10 == 0 || ts + 1 == t) {
            printf("  Timestep %zu/%zu\n", ts + 1, t);
        }
    }

    // After swap, results are in dRadBOut
    checkCuda(cudaMemcpy(state.radB.data(), dRadBOut, radSize, cudaMemcpyDeviceToHost), "copy radB results");

    cudaFree(dRadB);
    cudaFree(dRadBOut);
    cudaFree(dRadE);
    cudaFree(dRho);
    cudaFree(dKij);
    cudaFree(dTau);
}

void computeDistancesGPU(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;

    val_t* dRadB = nullptr;
    val_t* dDistances = nullptr;

    size_t radSize = t * n * sizeof(val_t);
    checkCuda(cudaMalloc(&dRadB, radSize), "malloc radB");
    checkCuda(cudaMalloc(&dDistances, n * sizeof(val_t)), "malloc distances");
    checkCuda(cudaMemcpy(dRadB, state.radB.data(), radSize, cudaMemcpyHostToDevice), "copy radB");

    int blockSize = 256;
    int numBlocks = std::min((int)((n + blockSize - 1) / blockSize), 65535);

    computeDistancesKernel<<<numBlocks, blockSize>>>(n, t, dRadB, dDistances, n, t, state.sourceIndex);
    checkCuda(cudaGetLastError(), "launch distances kernel");
    checkCuda(cudaDeviceSynchronize(), "sync distances kernel");

    checkCuda(cudaMemcpy(state.distances.data(), dDistances, n * sizeof(val_t), cudaMemcpyDeviceToHost), "copy distances results");

    cudaFree(dRadB);
    cudaFree(dDistances);
}

// ============================================================================
// Validation (host)
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

int getSubdivisionsForTriangleCount(int targetTriangles) {
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

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
    printf("Backend: CUDA GPU\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();
    computeTimeDelaysGPU(state);
    computeFormFactorsGPU(state);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();
    runSimulationGPU(state);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();
    computeDistancesGPU(state);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
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
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
