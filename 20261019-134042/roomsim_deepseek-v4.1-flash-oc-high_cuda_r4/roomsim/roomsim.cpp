/**
 * Room Response Simulation Benchmark
 * 
 * CUDA implementation of room impulse response simulation using
 * radiosity-based wave propagation.
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
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            std::exit(1);                                                       \
        }                                                                       \
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
// CUDA Device Helpers
// ============================================================================

__device__ __forceinline__ uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ val_t uniform_from_bits(uint32_t bits) {
    return static_cast<val_t>(bits) * 2.3283064365386963e-10f;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeKijKernel(const Triangle* __restrict__ triangles,
                                 val_t* __restrict__ kij,
                                 unsigned int n,
                                 uint64_t seed) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y;
    if (j >= n) return;

    val_t result = ZERO;
    if (i != j) {
        const Triangle& ti = triangles[i];
        const Triangle& tj = triangles[j];

        val_t nix = ti._normal.x, niy = ti._normal.y, niz = ti._normal.z;
        val_t njx = tj._normal.x, njy = tj._normal.y, njz = tj._normal.z;

        if (nix * njx + niy * njy + niz * njz <= 0.99f) {
            val_t aix = ti.a.x, aiy = ti.a.y, aiz = ti.a.z;
            val_t abix = ti.b.x - aix, abiy = ti.b.y - aiy, abiz = ti.b.z - aiz;
            val_t acix = ti.c.x - aix, aciy = ti.c.y - aiy, aciz = ti.c.z - aiz;

            val_t ajx = tj.a.x, ajy = tj.a.y, ajz = tj.a.z;
            val_t abjx = tj.b.x - ajx, abjy = tj.b.y - ajy, abjz = tj.b.z - ajz;
            val_t acjx = tj.c.x - ajx, acjy = tj.c.y - ajy, acjz = tj.c.z - ajz;

            uint64_t state = (((uint64_t)i << 32) ^ static_cast<uint64_t>(j)) ^ seed;
            val_t acc = ZERO;

#pragma unroll
            for (int r = 0; r < NUM_RAYS; ++r) {
                uint64_t r1 = splitmix64(state);
                uint64_t r2 = splitmix64(state);

                val_t u = uniform_from_bits(static_cast<uint32_t>(r1));
                val_t v = uniform_from_bits(static_cast<uint32_t>(r1 >> 32));
                if (u + v > ONE) {
                    u = ONE - u;
                    v = ONE - v;
                }

                val_t u2 = uniform_from_bits(static_cast<uint32_t>(r2));
                val_t v2 = uniform_from_bits(static_cast<uint32_t>(r2 >> 32));
                if (u2 + v2 > ONE) {
                    u2 = ONE - u2;
                    v2 = ONE - v2;
                }

                val_t pix = aix + abix * u + acix * v;
                val_t piy = aiy + abiy * u + aciy * v;
                val_t piz = aiz + abiz * u + aciz * v;
                val_t pjx = ajx + abjx * u2 + acjx * v2;
                val_t pjy = ajy + abjy * u2 + acjy * v2;
                val_t pjz = ajz + abjz * u2 + acjz * v2;

                val_t vx = pjx - pix;
                val_t vy = pjy - piy;
                val_t vz = pjz - piz;
                val_t distSqr = vx * vx + vy * vy + vz * vz;
                if (distSqr < EPSILON) continue;

                val_t invDist = 1.0f / sqrtf(distSqr);
                val_t cosI = (vx * nix + vy * niy + vz * niz) * invDist;
                if (cosI < ZERO) cosI = ZERO;
                val_t cosJ = (-vx * njx - vy * njy - vz * njz) * invDist;
                if (cosJ < ZERO) cosJ = ZERO;

                if (cosI <= ZERO || cosJ <= ZERO) continue;

                acc += (cosI * cosJ) / (PI * distSqr);
            }

            result = acc * INV_NUM_RAYS;
        }
    }

    kij[static_cast<size_t>(i) * n + j] = result;
}

__global__ void computeTauKernel(const Triangle* __restrict__ triangles,
                                 uint16_t* __restrict__ tau,
                                 unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y;
    if (j >= n) return;

    uint16_t result = 0;
    if (i != j) {
        const Triangle& ti = triangles[i];
        const Triangle& tj = triangles[j];

        val_t cix = (ti.a.x + ti.b.x + ti.c.x) / 3.0f;
        val_t ciy = (ti.a.y + ti.b.y + ti.c.y) / 3.0f;
        val_t ciz = (ti.a.z + ti.b.z + ti.c.z) / 3.0f;
        val_t cjx = (tj.a.x + tj.b.x + tj.c.x) / 3.0f;
        val_t cjy = (tj.a.y + tj.b.y + tj.c.y) / 3.0f;
        val_t cjz = (tj.a.z + tj.b.z + tj.c.z) / 3.0f;

        val_t dx = cix - cjx;
        val_t dy = ciy - cjy;
        val_t dz = ciz - cjz;
        val_t dist = sqrtf(dx * dx + dy * dy + dz * dz);
        result = static_cast<uint16_t>(ceilf(dist * INV_WAVE_SPEED));
    }

    tau[static_cast<size_t>(i) * n + j] = result;
}

__global__ void simulateStepKernel(const val_t* __restrict__ kij,
                                   const uint16_t* __restrict__ tau,
                                   const val_t* __restrict__ areas,
                                   const val_t* __restrict__ rho,
                                   const val_t* __restrict__ radE,
                                   val_t* __restrict__ radB,
                                   unsigned int n,
                                   unsigned int t) {
    unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int i = warp;
    if (i >= n) return;

    val_t sum = ZERO;
    size_t row = static_cast<size_t>(i) * n;

    for (unsigned int j = lane; j < n; j += 32u) {
        if (j == i) continue;

        unsigned int tauij = tau[row + j];
        if (t < tauij) continue;

        val_t kijv = kij[row + j];
        if (kijv <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t) - tauij;
        val_t radJ = radB[srcTime * n + j];
        if (radJ <= ZERO) continue;

        sum += fminf(kijv * areas[j], ONE) * radJ;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
        size_t idx = static_cast<size_t>(t) * n + i;
        radB[idx] = rho[i] * sum + radE[idx];
    }
}

__global__ void correlationKernel(const val_t* __restrict__ radB,
                                  val_t* __restrict__ corr,
                                  unsigned int n,
                                  unsigned int numTimesteps,
                                  unsigned int sourceIndex) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t = blockIdx.y;
    if (i >= n) return;

    val_t sum = ZERO;
    for (unsigned int tt = t; tt < numTimesteps; ++tt) {
        val_t pB = radB[static_cast<size_t>(tt) * n + i];
        val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
        sum += pS * pB;
    }

    corr[static_cast<size_t>(t) * n + i] = sum;
}

__global__ void maxCorrelationKernel(const val_t* __restrict__ corr,
                                     val_t* __restrict__ distances,
                                     unsigned int n,
                                     unsigned int numTimesteps) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t maxCorr = ZERO;
    unsigned int bestT = 0;

    for (unsigned int t = 0; t < numTimesteps; ++t) {
        val_t sum = corr[static_cast<size_t>(t) * n + i];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;

    size_t sourceIndex = 0;

    Triangle* d_triangles = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_kij = nullptr;
    uint16_t* d_tau = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }

    ~SimulationState() {
        cudaFree(d_triangles);
        cudaFree(d_areas);
        cudaFree(d_rho);
        cudaFree(d_kij);
        cudaFree(d_tau);
        cudaFree(d_radE);
        cudaFree(d_radB);
        cudaFree(d_distances);
    }
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

    size_t n = state.numTriangles;

    state.areas.resize(n);
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.resize(n, reflectivity);

    state.kij.resize(n * n, ZERO);
    state.radE.resize(timesteps * n, ZERO);
    state.radB.resize(timesteps * n, ZERO);
    state.distances.resize(n, ZERO);

    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    size_t nn = n * n;
    size_t tn = timesteps * n;

    CUDA_CHECK(cudaMalloc(&state.d_triangles, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_kij, nn * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, nn * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, tn * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, tn * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.d_triangles, state.triangles.data(),
                          n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(),
                          tn * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, tn * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    unsigned int n = static_cast<unsigned int>(state.numTriangles);
    unsigned int threads = 256;
    dim3 block(threads);
    dim3 grid((n + threads - 1) / threads, n);

    computeKijKernel<<<grid, block>>>(state.d_triangles, state.d_kij, n,
                                      22ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Progress: %u/%u triangles\n", n, n);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    unsigned int n = static_cast<unsigned int>(state.numTriangles);
    unsigned int threads = 256;
    dim3 block(threads);
    dim3 grid((n + threads - 1) / threads, n);

    computeTauKernel<<<grid, block>>>(state.d_triangles, state.d_tau, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    unsigned int n = static_cast<unsigned int>(state.numTriangles);
    unsigned int blocks = (n + 7u) / 8u;
    unsigned int threads = 256;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulateStepKernel<<<blocks, threads>>>(
            state.d_kij, state.d_tau, state.d_areas, state.d_rho,
            state.d_radE, state.d_radB,
            n, static_cast<unsigned int>(t));

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    unsigned int n = static_cast<unsigned int>(state.numTriangles);
    unsigned int numTimesteps = static_cast<unsigned int>(state.numTimesteps);
    unsigned int threads = 256;

    size_t corrCount = static_cast<size_t>(n) * numTimesteps;
    val_t* d_corr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_corr, corrCount * sizeof(val_t)));

    dim3 block(threads);
    dim3 grid((n + threads - 1) / threads, numTimesteps);

    correlationKernel<<<grid, block>>>(state.d_radB, d_corr, n, numTimesteps,
                                       static_cast<unsigned int>(state.sourceIndex));
    CUDA_CHECK(cudaGetLastError());

    unsigned int distBlocks = (n + threads - 1) / threads;
    maxCorrelationKernel<<<distBlocks, block>>>(d_corr, state.d_distances,
                                                n, numTimesteps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances,
                          n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_corr));
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

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

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
        CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij,
                              n * n * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                              t * n * sizeof(val_t), cudaMemcpyDeviceToHost));
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
