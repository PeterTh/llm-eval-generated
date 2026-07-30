/**
 * Room Response Simulation Benchmark - CUDA Parallel Implementation
 *
 * GPU-parallelized implementation of room impulse response simulation using
 * radiosity-based wave propagation. All major computation phases are
 * offloaded to CUDA kernels for maximum parallel scalability.
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
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// CUDA Device Types
// ============================================================================

struct alignas(16) Vec3f {
    val_t x, y, z;
};

struct alignas(32) TriangleF {
    Vec3f a, b, c, normal;
};

// ============================================================================
// CUDA Device Helper Functions
// ============================================================================

__device__ __forceinline__ Vec3f make_vec3(val_t x, val_t y, val_t z) {
    return {x, y, z};
}

__device__ __forceinline__ Vec3f vec_add(const Vec3f& a, const Vec3f& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ Vec3f vec_sub(const Vec3f& a, const Vec3f& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ Vec3f vec_mul(const Vec3f& a, val_t s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ __forceinline__ Vec3f vec_div(const Vec3f& a, val_t s) {
    return {a.x / s, a.y / s, a.z / s};
}

__device__ __forceinline__ Vec3f vec_neg(const Vec3f& a) {
    return {-a.x, -a.y, -a.z};
}

__device__ __forceinline__ val_t vec_dot(const Vec3f& a, const Vec3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3f vec_cross(const Vec3f& a, const Vec3f& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ __forceinline__ val_t vec_sqnorm(const Vec3f& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ __forceinline__ val_t vec_norm(const Vec3f& v) {
    return sqrtf(vec_sqnorm(v));
}

__device__ __forceinline__ Vec3f vec_normalize(const Vec3f& v) {
    val_t n = vec_norm(v);
    return n > EPSILON ? vec_div(v, n) : make_vec3(0, 0, 0);
}

// Möller-Trumbore ray-triangle intersection
__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3f& orig, const Vec3f& dir,
                                                      const Vec3f& v0, const Vec3f& v1, const Vec3f& v2) {
    Vec3f e1 = vec_sub(v1, v0);
    Vec3f e2 = vec_sub(v2, v0);
    Vec3f pvec = vec_cross(dir, e2);
    val_t det = vec_dot(e1, pvec);

    if (fabsf(det) < EPSILON) return 1e30f;

    val_t invDet = 1.0f / det;
    Vec3f tvec = vec_sub(orig, v0);
    val_t u = vec_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e30f;

    Vec3f qvec = vec_cross(tvec, e1);
    val_t v = vec_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e30f;

    return vec_dot(e2, qvec) * invDet;
}

// Cosine of angle between vector and normal
__device__ __forceinline__ val_t cosPhi(const Vec3f& v, const Vec3f& normal) {
    val_t vNorm = vec_norm(v);
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, vec_dot(v, normal) / vNorm);
}

// Random point in triangle using barycentric coordinates
__device__ __forceinline__ Vec3f randomPointInTriangle(const TriangleF& tri, curandStatePhilox4_32_10_t& state) {
    val_t u = curand_uniform(&state);
    val_t v = curand_uniform(&state);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3f ab = vec_sub(tri.b, tri.a);
    Vec3f ac = vec_sub(tri.c, tri.a);
    return vec_add(tri.a, vec_add(vec_mul(ab, u), vec_mul(ac, v)));
}

// Check if ray between two points is blocked by any triangle (brute force on GPU)
__device__ __forceinline__ bool isRayBlocked(const Vec3f& from, const Vec3f& to,
                                              const TriangleF* d_triangles,
                                              size_t numTriangles,
                                              size_t srcTriIdx, size_t dstTriIdx) {
    Vec3f dir = vec_sub(to, from);
    val_t rayLen = vec_norm(dir);
    if (rayLen < EPSILON) return true;
    Vec3f dirNorm = vec_div(dir, rayLen);

    for (size_t k = 0; k < numTriangles; ++k) {
        if (k == srcTriIdx || k == dstTriIdx) continue;

        val_t dist = rayTriangleIntersect(from, dirNorm,
                                          d_triangles[k].a, d_triangles[k].b, d_triangles[k].c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// CUDA Kernel: Form Factor Computation (parallel over all i,j pairs)
// ============================================================================

__global__ void computeFormFactorsKernel(const TriangleF* d_triangles,
                                          const val_t* __restrict__ d_areas,
                                          val_t* __restrict__ d_kij,
                                          size_t numTriangles) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = numTriangles * numTriangles;

    if (idx >= total) return;

    size_t i = idx / numTriangles;
    size_t j = idx % numTriangles;

    if (i == j) {
        d_kij[idx] = ZERO;
        return;
    }

    const TriangleF& triI = d_triangles[i];
    const TriangleF& triJ = d_triangles[j];

    // Cull triangles facing the same direction
    if (vec_dot(triI.normal, triJ.normal) > 0.99f) {
        d_kij[idx] = ZERO;
        return;
    }

    // Initialize RNG with deterministic seed per (i,j) pair
    curandStatePhilox4_32_10_t state;
    curand_init(42, static_cast<unsigned long long>(i) * 1000003ULL + j, 0, &state);

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3f pI = randomPointInTriangle(triI, state);
        Vec3f pJ = randomPointInTriangle(triJ, state);

        if (isRayBlocked(pI, pJ, d_triangles, numTriangles, i, j)) continue;

        Vec3f v = vec_sub(pJ, pI);
        val_t distSqr = vec_sqnorm(v);
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal);
        val_t cosPhiJ = cosPhi(vec_neg(v), triJ.normal);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    d_kij[idx] = kij * INV_NUM_RAYS;
}

// ============================================================================
// CUDA Kernel: Time Delay Computation (parallel over all i,j pairs)
// ============================================================================

__global__ void computeTimeDelaysKernel(const TriangleF* d_triangles,
                                         int* __restrict__ d_tau,
                                         size_t numTriangles) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = numTriangles * numTriangles;

    if (idx >= total) return;

    size_t i = idx / numTriangles;
    size_t j = idx % numTriangles;

    if (i == j) {
        d_tau[idx] = 0;
        return;
    }

    // Compute center of each triangle
    Vec3f centerI = vec_div(vec_add(vec_add(d_triangles[i].a, d_triangles[i].b), d_triangles[i].c), 3.0f);
    Vec3f centerJ = vec_div(vec_add(vec_add(d_triangles[j].a, d_triangles[j].b), d_triangles[j].c), 3.0f);

    val_t dist = vec_norm(vec_sub(centerI, centerJ));
    d_tau[idx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// CUDA Kernel: Simulation Step (parallel over triangles for one timestep)
// ============================================================================

__global__ void simulationStepKernel(const val_t* d_radE,
                                      const val_t* d_radB,
                                      const val_t* d_rho,
                                      const val_t* d_areas,
                                      const val_t* d_kij,
                                      const int* d_tau,
                                      val_t* d_radB_out,
                                      size_t numTriangles,
                                      size_t t) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= numTriangles) return;

    val_t sumB = ZERO;

    for (size_t j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        size_t idx2d = i * numTriangles + j;
        int tauij = d_tau[idx2d];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        val_t kij = d_kij[idx2d];
        if (kij <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = d_radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(kij * d_areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission
    d_radB_out[t * numTriangles + i] = d_rho[i] * sumB + d_radE[t * numTriangles + i];
}

// ============================================================================
// CUDA Kernel: Distance Computation (parallel over triangles)
// ============================================================================

__global__ void computeDistancesKernel(const val_t* __restrict__ d_radB,
                                        val_t* __restrict__ d_distances,
                                        size_t numTriangles,
                                        size_t numTimesteps,
                                        size_t sourceIndex) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= numTriangles) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (size_t t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;

        for (size_t tt = t; tt < numTimesteps; ++tt) {
            val_t pB = d_radB[tt * numTriangles + i];
            val_t pS = d_radB[(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    d_distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Host-side Vec3 (for mesh generation only)
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
};

// ============================================================================
// Mesh Generation: Icosphere (CPU-only, then transferred to GPU)
// ============================================================================

class IcosphereMesh {
public:
    std::vector<TriangleF> triangles;

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

        // Build triangles with aligned TriangleF structure
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            Vec3 va = vertices[face[2]];
            Vec3 vb = vertices[face[1]];
            Vec3 vc = vertices[face[0]];
            Vec3 edge1 = vb - va;
            Vec3 edge2 = vc - va;
            Vec3 n = edge1.cross(edge2);
            val_t nlen = n.norm();
            if (nlen > EPSILON) {
                n = n / nlen;
            } else {
                n = Vec3();
            }
            triangles.push_back({{va.x, va.y, va.z}, {vb.x, vb.y, vb.z},
                                 {vc.x, vc.y, vc.z}, {n.x, n.y, n.z}});
        }
    }
};

// ============================================================================
// CUDA Memory Management Helpers
// ============================================================================

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ============================================================================
// Simulation State (Host-side for validation/results)
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<TriangleF> triangles;
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
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh on CPU
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        Vec3f ab = {state.triangles[i].b.x - state.triangles[i].a.x,
                    state.triangles[i].b.y - state.triangles[i].a.y,
                    state.triangles[i].b.z - state.triangles[i].a.z};
        Vec3f ac = {state.triangles[i].c.x - state.triangles[i].a.x,
                    state.triangles[i].c.y - state.triangles[i].a.y,
                    state.triangles[i].c.z - state.triangles[i].a.z};
        Vec3f cr = {ab.y * ac.z - ab.z * ac.y,
                    ab.z * ac.x - ab.x * ac.z,
                    ab.x * ac.y - ab.y * ac.x};
        state.areas[i] = 0.5f * std::sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// CUDA-accelerated Precomputation: Form Factors
// ============================================================================

void computeFormFactorsCUDA(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");

    size_t n = state.numTriangles;
    size_t totalPairs = n * n;

    // Allocate device memory
    TriangleF* d_triangles = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_kij = nullptr;

    checkCuda(cudaMalloc(&d_triangles, n * sizeof(TriangleF)), "malloc triangles");
    checkCuda(cudaMalloc(&d_areas, n * sizeof(val_t)), "malloc areas");
    checkCuda(cudaMalloc(&d_kij, totalPairs * sizeof(val_t)), "malloc kij");

    // Copy data to device
    checkCuda(cudaMemcpy(d_triangles, state.triangles.data(),
                         n * sizeof(TriangleF), cudaMemcpyHostToDevice), "memcpy triangles");
    checkCuda(cudaMemcpy(d_areas, state.areas.data(),
                         n * sizeof(val_t), cudaMemcpyHostToDevice), "memcpy areas");

    // Launch kernel
    const int blockSize = 256;
    int numBlocks = static_cast<int>((totalPairs + blockSize - 1) / blockSize);
    const int maxGridSize = 65535;
    if (numBlocks > maxGridSize) numBlocks = maxGridSize;

    computeFormFactorsKernel<<<numBlocks, blockSize>>>(d_triangles, d_areas, d_kij, n);
    checkCuda(cudaGetLastError(), "form factors kernel launch");
    checkCuda(cudaDeviceSynchronize(), "form factors sync");

    // Copy result back
    checkCuda(cudaMemcpy(state.kij.data(), d_kij,
                         totalPairs * sizeof(val_t), cudaMemcpyDeviceToHost), "memcpy kij back");

    // Free device memory
    cudaFree(d_triangles);
    cudaFree(d_areas);
    cudaFree(d_kij);
}

// ============================================================================
// CUDA-accelerated Precomputation: Time Delays
// ============================================================================

void computeTimeDelaysCUDA(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");

    size_t n = state.numTriangles;
    size_t totalPairs = n * n;

    // Allocate device memory
    TriangleF* d_triangles = nullptr;
    int* d_tau = nullptr;

    checkCuda(cudaMalloc(&d_triangles, n * sizeof(TriangleF)), "malloc triangles");
    checkCuda(cudaMalloc(&d_tau, totalPairs * sizeof(int)), "malloc tau");

    // Copy data to device
    checkCuda(cudaMemcpy(d_triangles, state.triangles.data(),
                         n * sizeof(TriangleF), cudaMemcpyHostToDevice), "memcpy triangles");

    // Launch kernel
    const int blockSize = 256;
    int numBlocks = static_cast<int>((totalPairs + blockSize - 1) / blockSize);
    const int maxGridSize = 65535;
    if (numBlocks > maxGridSize) numBlocks = maxGridSize;

    computeTimeDelaysKernel<<<numBlocks, blockSize>>>(d_triangles, d_tau, n);
    checkCuda(cudaGetLastError(), "time delays kernel launch");
    checkCuda(cudaDeviceSynchronize(), "time delays sync");

    // Copy result back
    checkCuda(cudaMemcpy(state.tau.data(), d_tau,
                         totalPairs * sizeof(int), cudaMemcpyDeviceToHost), "memcpy tau back");

    // Free device memory
    cudaFree(d_triangles);
    cudaFree(d_tau);
}

// ============================================================================
// CUDA-accelerated Simulation (Wave Propagation)
// ============================================================================

void runSimulationCUDA(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");

    size_t n = state.numTriangles;
    size_t T = state.numTimesteps;

    // Allocate device memory
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_kij = nullptr;
    int* d_tau = nullptr;

    checkCuda(cudaMalloc(&d_radE, T * n * sizeof(val_t)), "malloc radE");
    checkCuda(cudaMalloc(&d_radB, T * n * sizeof(val_t)), "malloc radB");
    checkCuda(cudaMalloc(&d_rho, n * sizeof(val_t)), "malloc rho");
    checkCuda(cudaMalloc(&d_areas, n * sizeof(val_t)), "malloc areas");
    checkCuda(cudaMalloc(&d_kij, n * n * sizeof(val_t)), "malloc kij");
    checkCuda(cudaMalloc(&d_tau, n * n * sizeof(int)), "malloc tau");

    // Copy data to device
    checkCuda(cudaMemcpy(d_radE, state.radE.data(),
                         T * n * sizeof(val_t), cudaMemcpyHostToDevice), "memcpy radE");
    checkCuda(cudaMemset(d_radB, 0, T * n * sizeof(val_t)), "memset radB");
    checkCuda(cudaMemcpy(d_rho, state.rho.data(),
                         n * sizeof(val_t), cudaMemcpyHostToDevice), "memcpy rho");
    checkCuda(cudaMemcpy(d_areas, state.areas.data(),
                         n * sizeof(val_t), cudaMemcpyHostToDevice), "memcpy areas");
    checkCuda(cudaMemcpy(d_kij, state.kij.data(),
                         n * n * sizeof(val_t), cudaMemcpyHostToDevice), "memcpy kij");
    checkCuda(cudaMemcpy(d_tau, state.tau.data(),
                         n * n * sizeof(int), cudaMemcpyHostToDevice), "memcpy tau");

    // Launch simulation step kernel for each timestep
    const int blockSize = 256;
    int numBlocks = static_cast<int>((n + blockSize - 1) / blockSize);

    for (size_t t = 0; t < T; ++t) {
        simulationStepKernel<<<numBlocks, blockSize>>>(
            d_radE, d_radB, d_rho, d_areas, d_kij, d_tau, d_radB, n, t);
        checkCuda(cudaGetLastError(), "simulation step kernel launch");
    }
    checkCuda(cudaDeviceSynchronize(), "simulation sync");

    // Copy result back
    checkCuda(cudaMemcpy(state.radB.data(), d_radB,
                         T * n * sizeof(val_t), cudaMemcpyDeviceToHost), "memcpy radB back");

    // Free device memory
    cudaFree(d_radE);
    cudaFree(d_radB);
    cudaFree(d_rho);
    cudaFree(d_areas);
    cudaFree(d_kij);
    cudaFree(d_tau);
}

// ============================================================================
// CUDA-accelerated Distance Computation
// ============================================================================

void computeDistancesCUDA(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");

    size_t n = state.numTriangles;
    size_t T = state.numTimesteps;

    // Allocate device memory
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;

    checkCuda(cudaMalloc(&d_radB, T * n * sizeof(val_t)), "malloc radB");
    checkCuda(cudaMalloc(&d_distances, n * sizeof(val_t)), "malloc distances");

    // Copy data to device
    checkCuda(cudaMemcpy(d_radB, state.radB.data(),
                         T * n * sizeof(val_t), cudaMemcpyHostToDevice), "memcpy radB");

    // Launch kernel
    const int blockSize = 256;
    int numBlocks = static_cast<int>((n + blockSize - 1) / blockSize);

    computeDistancesKernel<<<numBlocks, blockSize>>>(
        d_radB, d_distances, n, T, state.sourceIndex);
    checkCuda(cudaGetLastError(), "distances kernel launch");
    checkCuda(cudaDeviceSynchronize(), "distances sync");

    // Copy result back
    checkCuda(cudaMemcpy(state.distances.data(), d_distances,
                         n * sizeof(val_t), cudaMemcpyDeviceToHost), "memcpy distances back");

    // Free device memory
    cudaFree(d_radB);
    cudaFree(d_distances);
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

    printf("Room Response Simulation Benchmark (CUDA)\n");
    printf("==========================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Print GPU info
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s (%zu MB, %d SMs)\n", prop.name, prop.totalGlobalMem / (1024*1024), prop.multiProcessorCount);
    } else {
        fprintf(stderr, "No CUDA devices available!\n");
        return 1;
    }
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelaysCUDA(state);
    computeFormFactorsCUDA(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulationCUDA(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistancesCUDA(state);

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
