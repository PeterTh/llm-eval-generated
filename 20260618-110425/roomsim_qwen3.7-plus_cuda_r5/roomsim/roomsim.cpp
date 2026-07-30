/**
 * Room Response Simulation Benchmark - CUDA Parallel Version
 *
 * CUDA-parallelized implementation of room impulse response simulation
 * using radiosity-based wave propagation. All compute-intensive phases
 * run on GPU:
 *   1. Form factors (Kij) - brute-force visibility per pair on GPU
 *   2. Time delays (Tau) - parallel computation
 *   3. Wave propagation - parallel across triangles per timestep
 *   4. Distance estimation via cross-correlation - parallel per triangle
 *
 * Random numbers are pre-generated on CPU to maintain deterministic results.
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
// CUDA error checking macro
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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
// Vector and Triangle Types (host+device compatible)
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
    __host__ __device__ Triangle(const Vec3& a_, const Vec3& b_, const Vec3& c_)
        : a(a_), b(b_), c(c_), _normal((b_ - a_).cross(c_ - a_).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
// Random Number Generation (CPU side, pre-generate for GPU)
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// ============================================================================
// CUDA Kernels
// ============================================================================

// Kernel: Compute time delays (Tau)
__global__ void computeTauKernel(
    const Triangle* __restrict__ triangles,
    int* __restrict__ tau,
    size_t N)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t totalPairs = N * N;
    if (idx >= totalPairs) return;

    size_t i = idx / N;
    size_t j = idx % N;
    if (i == j) { tau[idx] = 0; return; }

    Vec3 ci = triangles[i].center();
    Vec3 cj = triangles[j].center();
    val_t dist = (ci - cj).norm();
    tau[idx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Device function: Möller-Trumbore ray-triangle intersection
__device__ val_t rayTriangleIntersectDevice(
    const Vec3& orig, const Vec3& dir,
    const Vec3& v0, const Vec3& v1, const Vec3& v2)
{
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return 1e30f;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e30f;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e30f;

    return e2.dot(qvec) * invDet;
}

// Kernel: Compute form factors (Kij) with brute-force visibility
__global__ void computeKijKernel(
    const Triangle* __restrict__ triangles,
    const val_t* __restrict__ randVals,  // Pre-generated random numbers
    val_t* __restrict__ kij_out,
    size_t N)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t totalPairs = N * N;
    if (idx >= totalPairs) return;

    size_t i = idx / N;
    size_t j = idx % N;
    if (i == j) { kij_out[idx] = ZERO; return; }

    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) { kij_out[idx] = ZERO; return; }

    // Random numbers for this pair: 2 values per ray * NUM_RAYS rays
    // Each pair (i,j) uses randVals[(i*N+j) * 2*NUM_RAYS ... + 2*NUM_RAYS-1]
    size_t randOffset = idx * (2 * NUM_RAYS);

    val_t result = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t uI = randVals[randOffset + 2*r];
        val_t vI = randVals[randOffset + 2*r + 1];
        val_t uJ = randVals[randOffset + 2*NUM_RAYS + 2*r];
        val_t vJ = randVals[randOffset + 2*NUM_RAYS + 2*r + 1];

        // Generate random point in triangle I
        if (uI + vI > 1.0f) { uI = 1.0f - uI; vI = 1.0f - vI; }
        Vec3 abI = triI.b - triI.a;
        Vec3 acI = triI.c - triI.a;
        Vec3 pI = triI.a + abI * uI + acI * vI;

        // Generate random point in triangle J
        if (uJ + vJ > 1.0f) { uJ = 1.0f - uJ; vJ = 1.0f - vJ; }
        Vec3 abJ = triJ.b - triJ.a;
        Vec3 acJ = triJ.c - triJ.a;
        Vec3 pJ = triJ.a + abJ * uJ + acJ * vJ;

        // Brute-force visibility check
        Vec3 dir = pJ - pI;
        val_t rayLen = dir.norm();
        if (rayLen < EPSILON) continue;
        Vec3 dirNorm = dir / rayLen;

        bool blocked = false;
        for (size_t k = 0; k < N; ++k) {
            if (k == i || k == j) continue;
            val_t dist = rayTriangleIntersectDevice(pI, dirNorm,
                triangles[k].a, triangles[k].b, triangles[k].c);
            if (dist > EPSILON && dist < rayLen - EPSILON) {
                blocked = true;
                break;
            }
        }
        if (blocked) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t vNormI = v.norm();
        val_t cosPhiI = (vNormI > EPSILON) ? fmaxf(ZERO, v.dot(triI.normal()) / vNormI) : ZERO;

        Vec3 negV = -v;
        val_t vNormJ = negV.norm();
        val_t cosPhiJ = (vNormJ > EPSILON) ? fmaxf(ZERO, negV.dot(triJ.normal()) / vNormJ) : ZERO;

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        result += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij_out[idx] = result * INV_NUM_RAYS;
}

// Kernel: Wave propagation for one timestep
__global__ void wavePropagationKernel(
    const val_t* __restrict__ kij,
    const int* __restrict__ tau,
    const val_t* __restrict__ areas,
    const val_t* __restrict__ rho,
    const val_t* __restrict__ radE,
    const val_t* __restrict__ radB,
    val_t* __restrict__ radB_new,
    size_t N, int t)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        size_t pairIdx = i * N + j;
        int tauij = tau[pairIdx];

        if (t < tauij) continue;

        val_t k = kij[pairIdx];
        if (k <= ZERO) continue;

        size_t srcTime = (size_t)(t - tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    radB_new[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

// Kernel: Cross-correlation distance computation
__global__ void computeDistancesKernel(
    const val_t* __restrict__ radB,
    val_t* __restrict__ distances,
    size_t N, size_t T, size_t sourceIndex)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (size_t t = 0; t < T; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < T; ++tt) {
            val_t pB = radB[tt * N + i];
            val_t pS = radB[(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
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

    size_t sourceIndex;

    // GPU pointers
    Triangle* d_triangles = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_kij = nullptr;
    int* d_tau = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;
    val_t* d_rand = nullptr;

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

    // Initialize areas
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

    // Allocate GPU memory
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    CUDA_CHECK(cudaMalloc(&state.d_triangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_kij, N * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, N * sizeof(val_t)));

    // Upload triangle data, areas, rho, radE
    CUDA_CHECK(cudaMemcpy(state.d_triangles, state.triangles.data(),
                          N * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(),
                          N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(),
                          N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(),
                          T * N * sizeof(val_t), cudaMemcpyHostToDevice));

    // Zero-initialize GPU buffers
    CUDA_CHECK(cudaMemset(state.d_kij, 0, N * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_tau, 0, N * N * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_distances, 0, N * sizeof(val_t)));

    printf("GPU memory allocated\n");
}

// ============================================================================
// Precomputation Phase (GPU)
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");

    size_t N = state.numTriangles;
    size_t totalPairs = N * N;
    int blockSize = 256;
    int gridSize = (int)((totalPairs + blockSize - 1) / blockSize);

    computeTauKernel<<<gridSize, blockSize>>>(
        state.d_triangles, state.d_tau, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download tau for CPU-side use if needed
    CUDA_CHECK(cudaMemcpy(state.tau.data(), state.d_tau,
                          N * N * sizeof(int), cudaMemcpyDeviceToHost));
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");

    size_t N = state.numTriangles;

    // Pre-generate random numbers on CPU (same order as original)
    printf("  Pre-generating random numbers...\n");
    RandomGenerator rng(42);
    size_t totalRands = N * N * 2 * NUM_RAYS;
    std::vector<val_t> randVals(totalRands);

    // Generate in the same order as the original sequential code
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) {
                // Still consume random numbers to keep RNG state consistent
                // Actually the original skips i==j with `continue`, so no random numbers consumed
                continue;
            }
            size_t pairIdx = i * N + j;
            for (int r = 0; r < NUM_RAYS; ++r) {
                randVals[pairIdx * 2 * NUM_RAYS + 2*r]     = rng.rand(); // uI
                randVals[pairIdx * 2 * NUM_RAYS + 2*r + 1] = rng.rand(); // vI
                randVals[pairIdx * 2 * NUM_RAYS + 2*NUM_RAYS + 2*r]     = rng.rand(); // uJ
                randVals[pairIdx * 2 * NUM_RAYS + 2*NUM_RAYS + 2*r + 1] = rng.rand(); // vJ
            }
        }
    }

    // Upload random numbers to GPU
    CUDA_CHECK(cudaMalloc(&state.d_rand, totalRands * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_rand, randVals.data(),
                          totalRands * sizeof(val_t), cudaMemcpyHostToDevice));

    // Free CPU-side random data
    randVals.clear();
    randVals.shrink_to_fit();

    // Launch kernel
    size_t totalPairs = N * N;
    int blockSize = 128;
    int gridSize = (int)((totalPairs + blockSize - 1) / blockSize);

    computeKijKernel<<<gridSize, blockSize>>>(
        state.d_triangles, state.d_rand, state.d_kij, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download kij for CPU-side validation/output
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij,
                          N * N * sizeof(val_t), cudaMemcpyDeviceToHost));

    // Free random numbers on GPU
    CUDA_CHECK(cudaFree(state.d_rand));
    state.d_rand = nullptr;
}

// ============================================================================
// Simulation Phase (GPU)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;
    int blockSize = 256;
    int gridSize = (int)((N + blockSize - 1) / blockSize);

    // Upload radE
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(),
                          T * N * sizeof(val_t), cudaMemcpyHostToDevice));

    for (size_t t = 0; t < T; ++t) {
        wavePropagationKernel<<<gridSize, blockSize>>>(
            state.d_kij, state.d_tau, state.d_areas, state.d_rho,
            state.d_radE, state.d_radB, state.d_radB,
            N, (int)t);
        CUDA_CHECK(cudaDeviceSynchronize());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    // Download radB for CPU-side validation/distance computation
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                          T * N * sizeof(val_t), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (GPU)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;
    int blockSize = 256;
    int gridSize = (int)((N + blockSize - 1) / blockSize);

    // Upload radB (already on GPU from simulation, but re-upload to be safe)
    CUDA_CHECK(cudaMemcpy(state.d_radB, state.radB.data(),
                          T * N * sizeof(val_t), cudaMemcpyHostToDevice));

    computeDistancesKernel<<<gridSize, blockSize>>>(
        state.d_radB, state.d_distances, N, T, state.sourceIndex);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download distances
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances,
                          N * sizeof(val_t), cudaMemcpyDeviceToHost));
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
// Cleanup GPU resources
// ============================================================================

void cleanupGPU(SimulationState& state) {
    if (state.d_triangles) cudaFree(state.d_triangles);
    if (state.d_areas) cudaFree(state.d_areas);
    if (state.d_rho) cudaFree(state.d_rho);
    if (state.d_kij) cudaFree(state.d_kij);
    if (state.d_tau) cudaFree(state.d_tau);
    if (state.d_radE) cudaFree(state.d_radE);
    if (state.d_radB) cudaFree(state.d_radB);
    if (state.d_distances) cudaFree(state.d_distances);
    if (state.d_rand) cudaFree(state.d_rand);
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

    printf("Room Response Simulation Benchmark (CUDA)\n");
    printf("==========================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Print GPU info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("\n");

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
        if (!validateResults(state)) {
            cleanupGPU(state);
            return 1;
        }
    }

    // Cleanup GPU resources
    cleanupGPU(state);

    return 0;
}
