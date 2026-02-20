#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

constexpr int BX = 32;
constexpr int BY = 4;
constexpr int BZ = 2;

__global__ void stencil7_shared(const Real* __restrict__ in, Real* __restrict__ out, int nx, int ny, int nz) {
    constexpr int SHX = BX + 2;
    constexpr int SHY = BY + 2;
    constexpr int SHZ = BZ + 2;

    __shared__ Real sh[SHX * SHY * SHZ];

    const int x0 = (int)blockIdx.x * BX;
    const int y0 = (int)blockIdx.y * BY;
    const int z0 = (int)blockIdx.z * BZ;

    const int tid = (int)threadIdx.x + BX * ((int)threadIdx.y + BY * (int)threadIdx.z);
    const int nThreads = BX * BY * BZ;

    for (int i = tid; i < SHX * SHY * SHZ; i += nThreads) {
        const int sx = i % SHX;
        const int sy = (i / SHX) % SHY;
        const int sz = i / (SHX * SHY);

        const int gx = x0 + sx - 1;
        const int gy = y0 + sy - 1;
        const int gz = z0 + sz - 1;

        Real v = 0.0;
        if ((unsigned)gx < (unsigned)nx && (unsigned)gy < (unsigned)ny && (unsigned)gz < (unsigned)nz) {
            const size_t gidx = (size_t)gz * (size_t)nx * (size_t)ny + (size_t)gy * (size_t)nx + (size_t)gx;
            v = in[gidx];
        }
        sh[i] = v;
    }

    __syncthreads();

    const int x = x0 + (int)threadIdx.x;
    const int y = y0 + (int)threadIdx.y;
    const int z = z0 + (int)threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t gidx = (size_t)z * (size_t)nx * (size_t)ny + (size_t)y * (size_t)nx + (size_t)x;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        out[gidx] = in[gidx];
        return;
    }

    const int sidx = ((int)threadIdx.z + 1) * (SHX * SHY) + ((int)threadIdx.y + 1) * SHX + ((int)threadIdx.x + 1);

    const Real center = sh[sidx];
    const Real left = sh[sidx - 1];
    const Real right = sh[sidx + 1];
    const Real front = sh[sidx - SHX];
    const Real back = sh[sidx + SHX];
    const Real bottom = sh[sidx - SHX * SHY];
    const Real top = sh[sidx + SHX * SHY];

    out[gidx] = (center + left + right + front + back + bottom + top) / 7.0;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        std::fprintf(stderr, "No CUDA device found\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));  // initialize runtime
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&d_grid1, bytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, bytes));
    CUDA_CHECK(cudaMemcpy(d_grid1, grid1.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(BX, BY, BZ);
    const dim3 grid((unsigned)((nx + BX - 1) / BX), (unsigned)((ny + BY - 1) / BY), (unsigned)((nz + BZ - 1) / BZ));

    // Run stencil iterations
    printf("Running stencil computation...\n");

    cudaEvent_t startEv{}, stopEv{};
    CUDA_CHECK(cudaEventCreate(&startEv));
    CUDA_CHECK(cudaEventCreate(&stopEv));
    CUDA_CHECK(cudaEventRecord(startEv));

    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencil7_shared<<<grid, block>>>(d_grid1, d_grid2, (int)nx, (int)ny, (int)nz);
        } else {
            stencil7_shared<<<grid, block>>>(d_grid2, d_grid1, (int)nx, (int)ny, (int)nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stopEv));
    CUDA_CHECK(cudaEventSynchronize(stopEv));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEv, stopEv));

    printf("Computation time: %.3f ms\n", (double)elapsedMs);

    // Calculate performance metrics
    const double seconds = (double)elapsedMs / 1000.0;
    const double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * (double)iterations;
    const double mcups = (seconds > 0.0) ? (cellUpdates / seconds / 1e6) : 0.0;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy result back for optional printing/validation
    std::vector<Real>& finalHost = (iterations % 2 == 0) ? grid1 : grid2;
    const Real* finalDev = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(finalHost.data(), finalDev, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(startEv));
    CUDA_CHECK(cudaEventDestroy(stopEv));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    // Print results for external validation
    const std::vector<Real>& finalGrid = finalHost;
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
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
