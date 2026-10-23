#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err_),         \
                    __FILE__, __LINE__);                                                  \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

// Shared-memory stencil tiles.  The host selects a one-z-plane tile for small
// grids (more blocks, lower latency) and a multi-z-plane tile for large grids
// (less halo traffic, more work per block).
constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;
constexpr int MAX_TILE_Z = 8;
constexpr int SHARED_X = TILE_X + 2;
constexpr int SHARED_Y = TILE_Y + 2;
constexpr int MAX_BLOCK_THREADS = TILE_X * TILE_Y * 2;

__device__ __forceinline__ int smemIndex(const int x, const int y, const int z) {
    return (z * SHARED_Y + y) * SHARED_X + x;
}

__device__ __forceinline__ double laplacianFromTile(const double* tile, const int si,
                                                    const double invDx2,
                                                    const double invDy2,
                                                    const double invDz2) {
    const double cv = tile[si];
    const double cxx = fma(-cv, 2.0, tile[si + 1] + tile[si - 1]) * invDx2;
    const double cyy = fma(-cv, 2.0, tile[si + SHARED_X] + tile[si - SHARED_X]) * invDy2;
    const double czz = fma(-cv, 2.0, tile[si + SHARED_X * SHARED_Y] +
                                    tile[si - SHARED_X * SHARED_Y]) * invDz2;
    return (cxx + cyy) + czz;
}

// Load a halo tile from a global field with clamped (replicated) boundaries.
template <int TILE_Z>
__device__ __forceinline__ void loadTile(double* tile, const double* field,
                                         const int bx, const int by, const int bz,
                                         const int nx, const int ny, const int nz,
                                         const int tid, const int nthreads) {
    constexpr int elems = SHARED_X * SHARED_Y * (TILE_Z + 2);
    for (int i = tid; i < elems; i += nthreads) {
        const int sx = i % SHARED_X;
        const int rem = i / SHARED_X;
        const int sy = rem % SHARED_Y;
        const int sz = rem / SHARED_Y;

        int gx = bx + sx - 1;
        int gy = by + sy - 1;
        int gz = bz + sz - 1;
        gx = gx < 0 ? 0 : (gx >= nx ? nx - 1 : gx);
        gy = gy < 0 ? 0 : (gy >= ny ? ny - 1 : gy);
        gz = gz < 0 ? 0 : (gz >= nz ? nz - 1 : gz);

        tile[i] = field[((size_t)gz * (size_t)ny + (size_t)gy) * (size_t)nx + (size_t)gx];
    }
}

__global__ __launch_bounds__(MAX_BLOCK_THREADS)
void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
    size_t linear_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; linear_id < vol; linear_id += stride) {
        const size_t linear = linear_id;
        const double pseudo = (((linear + 1) * (size_t)1299709) % vol) / (double)vol;
        c[linear] = -1.0 + 2.0 * pseudo;
    }
}

template <int TILE_Z, int THREAD_Z>
__global__ __launch_bounds__(MAX_BLOCK_THREADS)
void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                    const int nx, const int ny, const int nz,
                                    const double invDx2, const double invDy2,
                                    const double invDz2, const double gamma,
                                    const double e_AA, const double e_BB,
                                    const double e_AB) {
    constexpr int Z_PER_THREAD = TILE_Z / THREAD_Z;
    __shared__ double tile[SHARED_X * SHARED_Y * (TILE_Z + 2)];

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;
    const int bz = blockIdx.z * TILE_Z;
    const int tid = threadIdx.x + threadIdx.y * TILE_X +
                    threadIdx.z * TILE_X * TILE_Y;
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;

    loadTile<TILE_Z>(tile, c, bx, by, bz, nx, ny, nz, tid, nthreads);
    __syncthreads();

    const int x = bx + threadIdx.x;
    const int y = by + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }

    for (int k = 0; k < Z_PER_THREAD; ++k) {
        const int lz = threadIdx.z + k * THREAD_Z;
        const int z = bz + lz;
        if (z >= nz) {
            continue;
        }

        const int si = smemIndex(threadIdx.x + 1, threadIdx.y + 1, lz + 1);
        const double cv = tile[si];
        const double lap = laplacianFromTile(tile, si, invDx2, invDy2, invDz2);

        // Explicit FMA sequence matching the reference CPU build's contracted
        // floating-point operations, preserving bit-for-bit results.
        const double t0 = fma(cv + 1.0, e_AA, (cv - 1.0) * e_BB);
        const double t1 = fma(-(cv + cv), e_AB, t0);
        const double p = fma(4.5, t1, 3.0 * cv);
        const double q = fma(cv * cv, cv, p);
        const size_t idx = ((size_t)z * (size_t)ny + (size_t)y) * (size_t)nx + (size_t)x;
        mu[idx] = fma(-gamma, lap, q);
    }
}

template <int TILE_Z, int THREAD_Z>
__global__ __launch_bounds__(MAX_BLOCK_THREADS)
void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                              const double* __restrict__ mu,
                              const int nx, const int ny, const int nz,
                              const double D, const double dt,
                              const double invDx2, const double invDy2,
                              const double invDz2) {
    constexpr int Z_PER_THREAD = TILE_Z / THREAD_Z;
    __shared__ double tile[SHARED_X * SHARED_Y * (TILE_Z + 2)];

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;
    const int bz = blockIdx.z * TILE_Z;
    const int tid = threadIdx.x + threadIdx.y * TILE_X +
                    threadIdx.z * TILE_X * TILE_Y;
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;

    loadTile<TILE_Z>(tile, mu, bx, by, bz, nx, ny, nz, tid, nthreads);
    __syncthreads();

    const int x = bx + threadIdx.x;
    const int y = by + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }

    for (int k = 0; k < Z_PER_THREAD; ++k) {
        const int lz = threadIdx.z + k * THREAD_Z;
        const int z = bz + lz;
        if (z >= nz) {
            continue;
        }

        const int si = smemIndex(threadIdx.x + 1, threadIdx.y + 1, lz + 1);
        const double lap = laplacianFromTile(tile, si, invDx2, invDy2, invDz2);
        const size_t idx = ((size_t)z * (size_t)ny + (size_t)y) * (size_t)nx + (size_t)x;
        cnew[idx] = fma(dt * D, lap, cold[idx]);
    }
}

// The program always executes on the GPU; no host fallback is provided.

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
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

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);

    const size_t gridSize = nx * ny * nz;

    // Device buffers
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    // Initialize concentration field directly on the device.
    printf("Initializing concentration field...\n");
    {
        const int threads = 256;
        const int blocks = (int)std::min<size_t>((gridSize + threads - 1) / threads, 1u << 20);
        initializeConcentrationKernel<<<blocks, threads>>>(d_cold, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Pick the tile shape that gives enough resident blocks for the grid size.
    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    const size_t largeGridThreshold = (size_t)smCount * 1024u * 32u;
    const bool useLargeTiles = gridSize >= largeGridThreshold;

    const int tileZ = useLargeTiles ? MAX_TILE_Z : 1;
    const int threadZ = useLargeTiles ? 2 : 1;
    const dim3 block(TILE_X, TILE_Y, threadZ);
    const dim3 grid((unsigned)((nx + TILE_X - 1) / TILE_X),
                    (unsigned)((ny + TILE_Y - 1) / TILE_Y),
                    (unsigned)((nz + tileZ - 1) / tileZ));

    // Make sure initialization completes before timing starts.
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (useLargeTiles) {
            computeChemicalPotentialKernel<MAX_TILE_Z, 2><<<grid, block>>>(
                d_cold, d_mu, (int)nx, (int)ny, (int)nz, invDx2, invDy2, invDz2,
                gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdateKernel<MAX_TILE_Z, 2><<<grid, block>>>(
                d_cnew, d_cold, d_mu, (int)nx, (int)ny, (int)nz, D, dt,
                invDx2, invDy2, invDz2);
        } else {
            computeChemicalPotentialKernel<1, 1><<<grid, block>>>(
                d_cold, d_mu, (int)nx, (int)ny, (int)nz, invDx2, invDy2, invDz2,
                gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdateKernel<1, 1><<<grid, block>>>(
                d_cnew, d_cold, d_mu, (int)nx, (int)ny, (int)nz, D, dt,
                invDx2, invDy2, invDz2);
        }

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Bring the final concentration field back to the host for output/validation.
    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
