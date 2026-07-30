/**
 * Room Response Simulation Benchmark - CUDA Parallelized
 * 
 * GPU-parallelized implementation of room impulse response simulation
 * using radiosity-based wave propagation. Parallelization strategy:
 * - Form factors (Kij): one thread per (i,j) pair with brute-force visibility
 * - Time delays (Tau): one thread per (i,j) pair
 * - Wave propagation: one thread per triangle per timestep
 * - Cross-correlation distances: one thread per triangle
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

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
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
// Host-side Vector and Triangle Types (for mesh generation)
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
// Mesh Generation: Icosphere (CPU-side, same as original)
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
// Device-side triangle data structure (padded for alignment)
// ============================================================================

struct DTriangle {
    float ax, ay, az, _pad0;
    float bx, by, bz, _pad1;
    float cx, cy, cz, _pad2;
    float nx, ny, nz, _pad3;
};

// ============================================================================
// Device-side vector operations
// ============================================================================

__device__ inline float3 d_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 d_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 d_mul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float d_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 d_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ inline float d_norm(float3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ inline float3 d_getA(const DTriangle& t) { return make_float3(t.ax, t.ay, t.az); }
__device__ inline float3 d_getB(const DTriangle& t) { return make_float3(t.bx, t.by, t.bz); }
__device__ inline float3 d_getC(const DTriangle& t) { return make_float3(t.cx, t.cy, t.cz); }
__device__ inline float3 d_getN(const DTriangle& t) { return make_float3(t.nx, t.ny, t.nz); }

// ============================================================================
// Device-side RNG (xorshift32)
// ============================================================================

__device__ inline uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ inline float d_randf(uint32_t& state) {
    return (xorshift32(state) & 0x00FFFFFFu) / (float)0x01000000u;
}

// ============================================================================
// Device-side geometric functions
// ============================================================================

__device__ inline float3 d_randomPointInTriangle(const DTriangle& tri, uint32_t& rngState) {
    float u = d_randf(rngState);
    float v = d_randf(rngState);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float3 a = d_getA(tri);
    float3 ab = d_sub(d_getB(tri), a);
    float3 ac = d_sub(d_getC(tri), a);
    return d_add(a, d_add(d_mul(ab, u), d_mul(ac, v)));
}

__device__ inline val_t d_cosPhi(float3 v, float3 normal) {
    float vNorm = d_norm(v);
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, d_dot(v, normal) / vNorm);
}

__device__ inline val_t d_rayTriangleIntersect(float3 orig, float3 dir,
                                                float3 v0, float3 v1, float3 v2) {
    float3 e1 = d_sub(v1, v0);
    float3 e2 = d_sub(v2, v0);
    float3 pvec = d_cross(dir, e2);
    float det = d_dot(e1, pvec);

    if (fabsf(det) < EPSILON) return 1e30f;

    float invDet = 1.0f / det;
    float3 tvec = d_sub(orig, v0);
    float u = d_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e30f;

    float3 qvec = d_cross(tvec, e1);
    float v = d_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e30f;

    return d_dot(e2, qvec) * invDet;
}

__device__ inline bool d_isRayBlocked(float3 from, float3 to,
                                       const DTriangle* __restrict__ triangles,
                                       size_t numTriangles,
                                       size_t srcTriIdx, size_t dstTriIdx) {
    float3 dir = d_sub(to, from);
    float rayLen = d_norm(dir);
    if (rayLen < EPSILON) return true;
    float3 dirNorm = d_mul(dir, 1.0f / rayLen);

    for (size_t idx = 0; idx < numTriangles; ++idx) {
        if (idx == srcTriIdx || idx == dstTriIdx) continue;
        float dist = d_rayTriangleIntersect(from, dirNorm,
                                             d_getA(triangles[idx]),
                                             d_getB(triangles[idx]),
                                             d_getC(triangles[idx]));
        if (dist > EPSILON && dist < rayLen - EPSILON) return true;
    }
    return false;
}

// ============================================================================
// CUDA Kernel: Compute Tau (time delays) - one thread per (i,j) pair
// ============================================================================

__global__ void computeTauKernel(const DTriangle* __restrict__ triangles,
                                  int* __restrict__ tau, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * N;
    if (idx >= total) return;

    size_t i = idx / N;
    size_t j = idx % N;
    if (i == j) { tau[idx] = 0; return; }

    float3 ci = d_mul(d_add(d_add(d_getA(triangles[i]), d_getB(triangles[i])),
                             d_getC(triangles[i])), 1.0f / 3.0f);
    float3 cj = d_mul(d_add(d_add(d_getA(triangles[j]), d_getB(triangles[j])),
                             d_getC(triangles[j])), 1.0f / 3.0f);
    float dist = d_norm(d_sub(ci, cj));
    tau[idx] = (int)ceilf(dist * INV_WAVE_SPEED);
}

// ============================================================================
// CUDA Kernel: Compute Kij (form factors) - one thread per (i,j) pair
// ============================================================================

__global__ void computeKijKernel(const DTriangle* __restrict__ triangles,
                                  val_t* __restrict__ kij, size_t N,
                                  uint32_t rngSeed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * N;
    if (idx >= total) return;

    size_t i = idx / N;
    size_t j = idx % N;
    if (i == j) { kij[idx] = ZERO; return; }

    const DTriangle triI = triangles[i];
    const DTriangle triJ = triangles[j];

    float3 nI = d_getN(triI);
    float3 nJ = d_getN(triJ);

    // Cull triangles facing the same direction
    if (d_dot(nI, nJ) > 0.99f) { kij[idx] = ZERO; return; }

    // Deterministic per-(i,j) RNG seed using hash mixing
    uint32_t h = rngSeed;
    h ^= (uint32_t)(i * 2654435761u);
    h ^= (uint32_t)(j * 2246822519u);
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    uint32_t rngState = h ? h : 1u;

    val_t result = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        float3 pI = d_randomPointInTriangle(triI, rngState);
        float3 pJ = d_randomPointInTriangle(triJ, rngState);

        if (d_isRayBlocked(pI, pJ, triangles, N, i, j)) continue;

        float3 v = d_sub(pJ, pI);
        float distSqr = d_dot(v, v);
        if (distSqr < EPSILON) continue;

        float cosPhiI = d_cosPhi(v, nI);
        float cosPhiJ = d_cosPhi(d_mul(v, -1.0f), nJ);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        result += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[idx] = result * INV_NUM_RAYS;
}

// ============================================================================
// CUDA Kernel: Simulation step - one thread per triangle, per timestep
// ============================================================================

__global__ void simulationStepKernel(const val_t* __restrict__ kij,
                                      const int* __restrict__ tau,
                                      const val_t* __restrict__ areas,
                                      const val_t* __restrict__ rho,
                                      const val_t* __restrict__ radE,
                                      val_t* __restrict__ radB,
                                      size_t N, size_t t) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        size_t ij = i * N + j;
        int tauij = tau[ij];

        if ((int)t < tauij) continue;

        val_t k = kij[ij];
        if (k <= ZERO) continue;

        size_t srcTime = t - (size_t)tauij;
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

// ============================================================================
// CUDA Kernel: Cross-correlation distances - one thread per triangle
// ============================================================================

__global__ void computeDistancesKernel(const val_t* __restrict__ radB,
                                        val_t* __restrict__ distances,
                                        size_t N, size_t T, size_t sourceIndex) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
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
            bestT = (int)t;
        }
    }

    distances[i] = WAVE_SPEED * (val_t)bestT;
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
    DTriangle* d_triangles;
    val_t* d_kij;
    int* d_tau;
    val_t* d_areas;
    val_t* d_rho;
    val_t* d_radE;
    val_t* d_radB;
    val_t* d_distances;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization (CPU-side mesh generation + GPU memory allocation)
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Compute areas
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

    // Set source emission
    size_t timeOff = timesteps / 2;
    for (size_t t = 0; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Convert triangles to device format
    size_t N = state.numTriangles;
    std::vector<DTriangle> h_triangles(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& tri = state.triangles[i];
        h_triangles[i].ax = tri.a.x; h_triangles[i].ay = tri.a.y; h_triangles[i].az = tri.a.z; h_triangles[i]._pad0 = 0;
        h_triangles[i].bx = tri.b.x; h_triangles[i].by = tri.b.y; h_triangles[i].bz = tri.b.z; h_triangles[i]._pad1 = 0;
        h_triangles[i].cx = tri.c.x; h_triangles[i].cy = tri.c.y; h_triangles[i].cz = tri.c.z; h_triangles[i]._pad2 = 0;
        h_triangles[i].nx = tri._normal.x; h_triangles[i].ny = tri._normal.y; h_triangles[i].nz = tri._normal.z; h_triangles[i]._pad3 = 0;
    }

    // Allocate GPU memory
    size_t T = state.numTimesteps;
    CUDA_CHECK(cudaMalloc(&state.d_triangles, N * sizeof(DTriangle)));
    CUDA_CHECK(cudaMalloc(&state.d_kij, N * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, N * sizeof(val_t)));

    // Upload data to GPU
    CUDA_CHECK(cudaMemcpy(state.d_triangles, h_triangles.data(), N * sizeof(DTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, T * N * sizeof(val_t)));
}

// ============================================================================
// GPU Computation Functions
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    size_t N = state.numTriangles;
    size_t total = N * N;
    int blockSize = 256;
    int gridSize = (int)((total + blockSize - 1) / blockSize);

    computeTauKernel<<<gridSize, blockSize>>>(state.d_triangles, state.d_tau, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    size_t N = state.numTriangles;
    size_t total = N * N;
    int blockSize = 256;
    int gridSize = (int)((total + blockSize - 1) / blockSize);

    computeKijKernel<<<gridSize, blockSize>>>(state.d_triangles, state.d_kij, N, 42u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Progress: %zu/%zu triangles\n", N, N);
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;
    int blockSize = 256;
    int gridSize = (int)((N + blockSize - 1) / blockSize);

    for (size_t t = 0; t < T; ++t) {
        simulationStepKernel<<<gridSize, blockSize>>>(
            state.d_kij, state.d_tau, state.d_areas, state.d_rho,
            state.d_radE, state.d_radB, N, t);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;
    int blockSize = 256;
    int gridSize = (int)((N + blockSize - 1) / blockSize);

    computeDistancesKernel<<<gridSize, blockSize>>>(
        state.d_radB, state.d_distances, N, T, state.sourceIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results to CPU
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances,
                           N * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij,
                           N * N * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.tau.data(), state.d_tau,
                           N * N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                           T * N * sizeof(val_t), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Cleanup GPU memory
// ============================================================================

void cleanupSimulation(SimulationState& state) {
    cudaFree(state.d_triangles);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_distances);
}

// ============================================================================
// Validation (CPU-side, same as original)
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

    // Select GPU device
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

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
            cleanupSimulation(state);
            return 1;
        }
    }

    // Cleanup
    cleanupSimulation(state);

    return 0;
}
