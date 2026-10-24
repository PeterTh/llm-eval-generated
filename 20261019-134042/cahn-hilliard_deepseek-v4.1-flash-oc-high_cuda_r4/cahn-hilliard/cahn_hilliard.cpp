#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Tile dimensions for the CUDA kernels.  Each block computes a BX x BY x BZ
// tile of the grid; the shared-memory tile additionally holds one layer of
// halo cells in every direction.
constexpr int BX = 16;
constexpr int BY = 8;
constexpr int BZ = 4;

constexpr int SX = BX + 2;
constexpr int SY = BY + 2;
constexpr int SZ = BZ + 2;
constexpr int TILE_ELEMS = SX * SY * SZ;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

__device__ __forceinline__ int clampIndex(const int v, const int n) {
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

// Load a (BX+2) x (BY+2) x (BZ+2) tile into shared memory, replicating the
// clamped boundary conditions used by the original serial implementation.
__device__ __forceinline__ void loadTile(const double* __restrict__ src,
                                         double* __restrict__ tile,
                                         const int nx, const int ny, const int nz,
                                         const int baseX, const int baseY, const int baseZ) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    for (int sz = tz; sz < SZ; sz += BZ) {
        const int gz = clampIndex(baseZ + sz - 1, nz);
        for (int sy = ty; sy < SY; sy += BY) {
            const int gy = clampIndex(baseY + sy - 1, ny);
            const int row = (gz * ny + gy) * nx;
            const int srow = (sz * SY + sy) * SX;
            for (int sx = tx; sx < SX; sx += BX) {
                const int gx = clampIndex(baseX + sx - 1, nx);
                tile[srow + sx] = src[row + gx];
            }
        }
    }
}

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const int nx, const int ny, const int nz,
                                               const double invdx2, const double invdy2,
                                               const double invdz2,
                                               const double gamma, const double e_AA,
                                               const double e_BB, const double e_AB) {
    __shared__ double tile[TILE_ELEMS];

    const int baseX = blockIdx.x * BX;
    const int baseY = blockIdx.y * BY;
    const int baseZ = blockIdx.z * BZ;

    loadTile(c, tile, nx, ny, nz, baseX, baseY, baseZ);
    __syncthreads();

    const int x = baseX + threadIdx.x;
    const int y = baseY + threadIdx.y;
    const int z = baseZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const int c0 = ((threadIdx.z + 1) * SY + (threadIdx.y + 1)) * SX + (threadIdx.x + 1);
    const double center = tile[c0];
    const double cxx = (tile[c0 + 1] + tile[c0 - 1] - 2.0 * center) * invdx2;
    const double cyy = (tile[c0 + SX] + tile[c0 - SX] - 2.0 * center) * invdy2;
    const double czz = (tile[c0 + SY * SX] + tile[c0 - SY * SX] - 2.0 * center) * invdz2;
    const double lap = cxx + cyy + czz;

    const double cv = center;
    mu[(z * ny + y) * nx + x] =
        4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
        + 3.0 * cv + cv * cv * cv
        - gamma * lap;
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const int nx, const int ny, const int nz,
                                         const double D, const double dt,
                                         const double invdx2, const double invdy2,
                                         const double invdz2) {
    __shared__ double tile[TILE_ELEMS];

    const int baseX = blockIdx.x * BX;
    const int baseY = blockIdx.y * BY;
    const int baseZ = blockIdx.z * BZ;

    loadTile(mu, tile, nx, ny, nz, baseX, baseY, baseZ);
    __syncthreads();

    const int x = baseX + threadIdx.x;
    const int y = baseY + threadIdx.y;
    const int z = baseZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const int c0 = ((threadIdx.z + 1) * SY + (threadIdx.y + 1)) * SX + (threadIdx.x + 1);
    const double center = tile[c0];
    const double cxx = (tile[c0 + 1] + tile[c0 - 1] - 2.0 * center) * invdx2;
    const double cyy = (tile[c0 + SX] + tile[c0 - SX] - 2.0 * center) * invdy2;
    const double czz = (tile[c0 + SY * SX] + tile[c0 - SY * SX] - 2.0 * center) * invdz2;
    const double lap = cxx + cyy + czz;

    const size_t idx = (static_cast<size_t>(z) * ny + y) * nx + x;
    cnew[idx] = cold[idx] + dt * D * lap;
}

// Initialize concentration field
__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                               const int nx, const int ny, const int nz,
                                               const unsigned long long vol) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const unsigned long long linear_id =
        (static_cast<unsigned long long>(z) * ny + y) * nx + x;
    const double pseudo =
        static_cast<double>(((linear_id + 1ull) * 1299709ull) % vol) /
        static_cast<double>(vol);
    c[(static_cast<size_t>(z) * ny + y) * nx + x] = -1.0 + 2.0 * pseudo;
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate device arrays
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    {
        const dim3 block(8, 8, 4);
        const dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                        (unsigned)((ny + block.y - 1) / block.y),
                        (unsigned)((nz + block.z - 1) / block.z));
        initializeConcentrationKernel<<<grid, block>>>(d_cold, (int)nx, (int)ny, (int)nz,
                                                       (unsigned long long)gridSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    const dim3 block(BX, BY, BZ);
    const dim3 grid((unsigned)((nx + BX - 1) / BX),
                    (unsigned)((ny + BY - 1) / BY),
                    (unsigned)((nz + BZ - 1) / BZ));
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, (int)nx, (int)ny, (int)nz,
                                                        invdx2, invdy2, invdz2,
                                                        gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, (int)nx, (int)ny, (int)nz,
                                                  D, dt, invdx2, invdy2, invdz2);
        
        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const double elapsedSec = duration.count() / 1e6;
    
    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = (elapsedSec > 0.0) ? cellUpdates / elapsedSec / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy the final field back to the host
    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    
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
