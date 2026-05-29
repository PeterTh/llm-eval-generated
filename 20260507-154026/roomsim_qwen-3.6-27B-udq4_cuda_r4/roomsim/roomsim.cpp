/**
 * Room Response Simulation Benchmark - CUDA Parallel Version
 * 
 * GPU-parallelized implementation of room impulse response simulation using
 * radiosity-based wave propagation. Models how sound/light waves propagate
 * between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles (parallel on GPU)
 * 2. Wave propagation using radiosity equations with time delays (parallel per timestep)
 * 3. Distance estimation via cross-correlation of radiosity values (parallel on GPU)
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

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                    \
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
// Vector and Triangle Types (__host__ __device__)
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z)
        : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const {
        return {x + o.x, y + o.y, z + o.z};
    }
    __host__ __device__ Vec3 operator-(const Vec3& o) const {
        return {x - o.x, y - o.y, z - o.z};
    }
    __host__ __device__ Vec3 operator*(val_t s) const {
        return {x * s, y * s, z * s};
    }
    __host__ __device__ Vec3 operator/(val_t s) const {
        return {x / s, y / s, z / s};
    }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const {
        return x * o.x + y * o.y + z * o.z;
    }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const {
        return x * x + y * y + z * z;
    }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON &&
               fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const {
        return a == o.a && b == o.b && c == o.c;
    }
};

// ============================================================================
// Triangle-Box Overlap Test (host-only, for octree construction)
// ============================================================================

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize,
                         const Triangle& tri) {
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

    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * fabsf(axis.x) +
                  boxHalfSize.y * fabsf(axis.y) +
                  boxHalfSize.z * fabsf(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
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
// Octree for Spatial Acceleration (host-only)
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
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin,
                   const Vec3& nodeMax) {
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
                childCenter.x +=
                    (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y +=
                    (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z +=
                    (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() &&
                childIndices[i].size() < indices.size()) {
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
// Mesh Generation: Icosphere (host-only)
// ============================================================================

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<Vec3> vertices = {
            Vec3(-1, t, 0).normalized() * radius,
            Vec3(1, t, 0).normalized() * radius,
            Vec3(-1, -t, 0).normalized() * radius,
            Vec3(1, -t, 0).normalized() * radius,
            Vec3(0, -1, t).normalized() * radius,
            Vec3(0, 1, t).normalized() * radius,
            Vec3(0, -1, -t).normalized() * radius,
            Vec3(0, 1, -t).normalized() * radius,
            Vec3(t, 0, -1).normalized() * radius,
            Vec3(t, 0, 1).normalized() * radius,
            Vec3(-t, 0, -1).normalized() * radius,
            Vec3(-t, 0, 1).normalized() * radius};

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1},   {0, 1, 7},  {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4},   {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2},    {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
            {4, 9, 5}, {2, 4, 11},   {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};

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
            triangles.emplace_back(vertices[face[2]], vertices[face[1]],
                                   vertices[face[0]]);
        }
    }
};

// ============================================================================
// CUDA Device Code
// ============================================================================

// ---- Device RNG (xorshift32) ----

__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ __forceinline__ float xorshift_float(uint32_t& state) {
    return (xorshift32(state) >> 8) / 16777216.0f;
}

// ---- Device Utility Functions ----

__device__ Vec3 randomPointInTriangleDev(const Triangle& t, uint32_t& state) {
    float u = xorshift_float(state);
    float v = xorshift_float(state);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ float
rayTriangleIntersectDev(const Vec3& orig, const Vec3& dir, const Vec3& v0,
                        const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    float det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return 1e38f;

    float invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e38f;

    Vec3 qvec = tvec.cross(e1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e38f;

    return e2.dot(qvec) * invDet;
}

__device__ float cosPhiDev(const Vec3& v, const Vec3& normal) {
    float vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

__device__ bool isRayBlockedDev(
    const Vec3& from, const Vec3& to, const Triangle* __restrict__ triangles,
    size_t srcTriIdx, size_t dstTriIdx, size_t numTriangles) {
    Vec3 dir = to - from;
    float rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    for (size_t k = 0; k < numTriangles; k++) {
        if (k == srcTriIdx || k == dstTriIdx) continue;

        const Triangle& tri = triangles[k];
        float dist =
            rayTriangleIntersectDev(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true; // Ray is blocked
        }
    }
    return false;
}

// ---- Device Computation Functions ----

__device__ float computeKijDev(
    size_t idxI, size_t idxJ, const Triangle* __restrict__ triangles,
    uint32_t seed, size_t numTriangles) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    float kij = ZERO;
    uint32_t state = seed;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangleDev(triI, state);
        Vec3 pJ = randomPointInTriangleDev(triJ, state);

        if (isRayBlockedDev(pI, pJ, triangles, idxI, idxJ, numTriangles))
            continue;

        Vec3 v = pJ - pI;
        float distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        float cosPhiI = cosPhiDev(v, triI.normal());
        float cosPhiJ = cosPhiDev(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

__device__ int computeTauDev(const Triangle& triI, const Triangle& triJ) {
    float dist = (triI.center() - triJ.center()).norm();
    return (int)ceilf(dist * INV_WAVE_SPEED);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Kernel: Compute form factors Kij (each thread handles one (i,j) pair)
__global__ void computeFormFactorsKernel(
    const Triangle* __restrict__ triangles, size_t numTriangles,
    float* __restrict__ kij) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = numTriangles * numTriangles;

    if (idx >= total) return;

    size_t i = idx / numTriangles;
    size_t j = idx % numTriangles;

    if (i == j) {
        kij[idx] = ZERO;
        return;
    }

    // Deterministic per-thread seed based on (i, j)
    uint32_t seed = 0x12345678u + (i * numTriangles + j) * 0x9e3779b9u;

    kij[idx] = computeKijDev(i, j, triangles, seed, numTriangles);
}

// Kernel: Compute time delays Tau (each thread handles one (i,j) pair)
__global__ void computeTimeDelaysKernel(
    const Triangle* __restrict__ triangles, size_t numTriangles,
    int* __restrict__ tau) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = numTriangles * numTriangles;

    if (idx >= total) return;

    size_t i = idx / numTriangles;
    size_t j = idx % numTriangles;

    if (i == j) {
        tau[idx] = 0;
        return;
    }

    tau[idx] = computeTauDev(triangles[i], triangles[j]);
}

// Kernel: Run simulation for one timestep (each thread handles one triangle)
__global__ void runSimulationKernel(
    size_t t, size_t numTriangles, const float* __restrict__ kij,
    const int* __restrict__ tau, float* __restrict__ radB,
    const float* __restrict__ radE, const float* __restrict__ areas,
    const float* __restrict__ rho) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    float sumB = ZERO;

    for (size_t j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        size_t idx2d = i * numTriangles + j;
        int tauij = tau[idx2d];

        if ((int)t < tauij) continue;

        float kij_val = kij[idx2d];
        if (kij_val <= ZERO) continue;

        size_t srcTime = t - (size_t)tauij;
        float radJ = radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kij_val * areas[j], ONE) * radJ;
    }

    radB[t * numTriangles + i] =
        rho[i] * sumB + radE[t * numTriangles + i];
}

// Kernel: Compute distances (each thread handles one triangle)
__global__ void computeDistancesKernel(
    size_t numTriangles, size_t numTimesteps,
    const float* __restrict__ radB, size_t sourceIndex,
    float* __restrict__ distances) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    float maxCorr = ZERO;
    int bestT = 0;

    for (size_t t = 0; t < numTimesteps; ++t) {
        float sum = ZERO;

        for (size_t tt = t; tt < numTimesteps; ++tt) {
            float pB = radB[tt * numTriangles + i];
            float pS = radB[(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = (int)t;
        }
    }

    distances[i] = WAVE_SPEED * (float)bestT;
}

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

    Octree octree;

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions,
                          size_t timesteps, size_t sourceIdx,
                          val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n",
           state.numTriangles);

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
// GPU Computation Functions
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");

    // GPU memory
    Triangle* d_triangles = nullptr;
    float* d_kij = nullptr;

    CUDA_CHECK(cudaMalloc(&d_triangles,
                          state.numTriangles * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_kij, state.kij.size() * sizeof(float)));

    // Transfer triangles to GPU
    CUDA_CHECK(cudaMemcpy(d_triangles, state.triangles.data(),
                          state.numTriangles * sizeof(Triangle),
                          cudaMemcpyHostToDevice));

    // Launch kernel
    size_t total = state.numTriangles * state.numTriangles;
    int blockSize = 256;
    int gridSize = (int)((total + blockSize - 1) / blockSize);

    computeFormFactorsKernel<<<gridSize, blockSize>>>(
        d_triangles, state.numTriangles, d_kij);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(state.kij.data(), d_kij,
                          state.kij.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_kij));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");

    Triangle* d_triangles = nullptr;
    int* d_tau = nullptr;

    CUDA_CHECK(cudaMalloc(&d_triangles,
                          state.numTriangles * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_tau, state.tau.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_triangles, state.triangles.data(),
                          state.numTriangles * sizeof(Triangle),
                          cudaMemcpyHostToDevice));

    size_t total = state.numTriangles * state.numTriangles;
    int blockSize = 256;
    int gridSize = (int)((total + blockSize - 1) / blockSize);

    computeTimeDelaysKernel<<<gridSize, blockSize>>>(
        d_triangles, state.numTriangles, d_tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.tau.data(), d_tau,
                          state.tau.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_tau));
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");

    // Allocate GPU memory
    float* d_kij = nullptr;
    int* d_tau = nullptr;
    float* d_radB = nullptr;
    float* d_radE = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;

    CUDA_CHECK(cudaMalloc(&d_kij, state.kij.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tau, state.tau.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radB, state.radB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radE, state.radE.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_areas, state.numTriangles * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_rho, state.numTriangles * sizeof(float)));

    // Transfer data to GPU
    CUDA_CHECK(cudaMemcpy(d_kij, state.kij.data(),
                          state.kij.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, state.tau.data(),
                          state.tau.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(),
                          state.radB.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(),
                          state.radE.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(),
                          state.numTriangles * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(),
                          state.numTriangles * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch one kernel per timestep
    int blockSize = 256;
    int gridSize =
        (int)((state.numTriangles + blockSize - 1) / blockSize);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        runSimulationKernel<<<gridSize, blockSize>>>(
            t, state.numTriangles, d_kij, d_tau, d_radB, d_radE, d_areas,
            d_rho);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy radB back
    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB,
                          state.radB.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_radB));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");

    float* d_radB = nullptr;
    float* d_distances = nullptr;

    CUDA_CHECK(cudaMalloc(&d_radB, state.radB.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_distances, state.numTriangles * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(),
                          state.radB.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize =
        (int)((state.numTriangles + blockSize - 1) / blockSize);

    computeDistancesKernel<<<gridSize, blockSize>>>(
        state.numTriangles, state.numTimesteps, d_radB, state.sourceIndex,
        d_distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), d_distances,
                          state.numTriangles * sizeof(float),
                          cudaMemcpyDeviceToHost));

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
    printf("  Average distance: %.4f\n",
           sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount,
           state.numTriangles);

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n",
               srcDist);
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

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy,
           state.numTriangles);

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
           100.0f * nonZeroKij /
               static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation "
               "failed\n");
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
        const uint32_t* ptr =
            reinterpret_cast<const uint32_t*>(&state.distances[i]);
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
    printf("               Actual count will be rounded to nearest icosphere "
           "level:\n");
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

    printf("Room Response Simulation Benchmark (CUDA)\n");
    printf("==========================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n",
           targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    SimulationState state;
    initializeSimulation(state, subdivisions,
                         static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(endPre -
                                                              startPre)
            .count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(endSim -
                                                              startSim)
            .count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(endDist -
                                                              startDist)
            .count();

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
    printf("  Total time per triangle: %.4f ms\n",
           static_cast<double>(totalTime) / n);

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
        std::vector<double> distData(state.distances.begin(),
                                     state.distances.end());
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
