#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err), #call);         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// 3D index calculation
inline __host__ __device__ constexpr size_t idx3(const size_t x, const size_t y,
                                                 const size_t z, const size_t nx,
                                                 const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel: 7-point stencil with boundary copying
// Each thread block uses shared memory tiling with halo regions for
// coalesced global memory access and reduced global memory traffic.
__global__ void stencilKernel(const Real * __restrict__ input,
                              Real * __restrict__ output, const size_t nx,
                              const size_t ny, const size_t nz) {
    // Shared memory tile with 1-cell halo on each face
    constexpr int TX = 32;
    constexpr int TY = 8;
    constexpr int TZ = 4;
    __shared__ Real sTile[TX + 2][TY + 2][TZ + 2];

    // Position within the block (0-based)
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tz = threadIdx.z;

    // Block origin in global coordinates
    unsigned int bx = blockIdx.x * TX;
    unsigned int by = blockIdx.y * TY;
    unsigned int bz = blockIdx.z * TZ;

    // Load tile + halo into shared memory (each thread loads one element)
    unsigned int gx = bx + tx - 1;
    unsigned int gy = by + ty - 1;
    unsigned int gz = bz + tz - 1;

    if (gx < nx && gy < ny && gz < nz) {
        sTile[tx][ty][tz] = input[gz * (nx * ny) + gy * nx + gx];
    } else {
        sTile[tx][ty][tz] = 0.0; // Out-of-bounds (shouldn't be accessed)
    }
    __syncthreads();

    // Compute stencil for interior points within this block's core tile
    if (tx > 0 && tx < TX && ty > 0 && ty < TY && tz > 0 && tz < TZ) {
        // Global coordinates of this interior point
        unsigned int ix = bx + tx - 1;
        unsigned int iy = by + ty - 1;
        unsigned int iz = bz + tz - 1;

        if (ix > 0 && ix < nx - 1 && iy > 0 && iy < ny - 1 && iz > 0 &&
            iz < nz - 1) {
            // Read from shared memory (all accesses are coalesced via shared)
            Real center = sTile[tx][ty][tz];
            Real left = sTile[tx - 1][ty][tz];
            Real right = sTile[tx + 1][ty][tz];
            Real front = sTile[tx][ty - 1][tz];
            Real back = sTile[tx][ty + 1][tz];
            Real bottom = sTile[tx][ty][tz - 1];
            Real top = sTile[tx][ty][tz + 1];

            output[iz * (nx * ny) + iy * nx + ix] =
                (center + left + right + front + back + bottom + top) / 7.0;
        }
    }

    // Copy boundary values (points on the block's core tile that are on
    // the global boundary)
    if (tx < TX && ty < TY && tz < TZ) {
        unsigned int ix = bx + tx;
        unsigned int iy = by + ty;
        unsigned int iz = bz + tz;

        if (ix < nx && iy < ny && iz < nz &&
            (ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1 || iz == 0 ||
             iz == nz - 1)) {
            output[iz * (nx * ny) + iy * nx + ix] =
                input[iz * (nx * ny) + iy * nx + ix];
        }
    }
}

// CUDA kernel: initialize grid on device
__global__ void initKernel(Real *grid, const size_t nx, const size_t ny,
                           const size_t nz) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;

    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        grid[i] = static_cast<Real>(i % 19);
    }
}

// CUDA kernel: validate result on device (check for NaN/Inf)
__global__ void validateKernel(const Real *grid, const size_t total,
                               int *hasError) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        Real val = grid[i];
        if (isnan(val) || isinf(val)) {
            *hasError = 1;
            return;
        }
    }
}

void printUsage(const char *progName) {
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

int main(int argc, char **argv) {
    size_t nx = 128;
    size_t ny = 0; // Will be set to nx if not specified
    size_t nz = 0; // Will be set to nx if not specified
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

    if (ny == 0)
        ny = nx;
    if (nz == 0)
        nz = nx;

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Detect GPU
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);

    size_t gridSize = nx * ny * nz;
    size_t gridBytes = gridSize * sizeof(Real);

    // Allocate device memory (double buffering)
    Real *d_grid1 = nullptr;
    Real *d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridBytes));

    // Initialize grid on GPU
    printf("Initializing grid...\n");
    {
        const int blockSize = 256;
        const int numBlocks =
            std::min(static_cast<int>((gridSize + blockSize - 1) / blockSize),
                     65535);
        initKernel<<<numBlocks, blockSize>>>(d_grid1, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Configure kernel launch parameters
    // Block size: 32x8x4 = 1024 threads, matching shared memory tile
    dim3 blockSize(32, 8, 4);
    // Grid covers the entire domain; each block processes a 32x8x4 tile
    dim3 gridDim(
        static_cast<unsigned int>((nx + blockSize.x - 1) / blockSize.x),
        static_cast<unsigned int>((ny + blockSize.y - 1) / blockSize.y),
        static_cast<unsigned int>((nz + blockSize.z - 1) / blockSize.z));

    // Run stencil iterations on GPU (data stays on device throughout)
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<gridDim, blockSize>>>(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilKernel<<<gridDim, blockSize>>>(d_grid2, d_grid1, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Determine which buffer holds the final result
    Real *d_final = (iterations % 2 == 0) ? d_grid2 : d_grid1;

    // Copy final result back to host for validation/printing
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridBytes,
                          cudaMemcpyDeviceToHost));

    // Calculate performance metrics
    double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) *
                         iterations;
    double mcups =
        cellUpdates / (duration.count() / 1000.0) / 1e6; // Million cell updates/s
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");

        // 1. Check for NaN/Inf using GPU kernel
        int h_hasError = 0;
        int *d_hasError = nullptr;
        CUDA_CHECK(cudaMalloc(&d_hasError, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_hasError, 0, sizeof(int)));

        {
            const int blockSize = 256;
            const int numBlocks =
                std::min(static_cast<int>((gridSize + blockSize - 1) / blockSize),
                         65535);
            validateKernel<<<numBlocks, blockSize>>>(d_final, gridSize,
                                                     d_hasError);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_hasError, d_hasError, sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(d_hasError));

        if (h_hasError) {
            printf("Validation failed: found NaN or Inf value\n");
            CUDA_CHECK(cudaFree(d_grid1));
            CUDA_CHECK(cudaFree(d_grid2));
            return 1;
        }

        // 2. Compute min/max on host
        Real minVal = finalGrid[0];
        Real maxVal = finalGrid[0];
        for (const auto &val : finalGrid) {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }

        printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

        if (maxVal > 1e6 || minVal < -1e6) {
            printf("Validation failed: values out of expected range\n");
            CUDA_CHECK(cudaFree(d_grid1));
            CUDA_CHECK(cudaFree(d_grid2));
            return 1;
        }

        printf("Validation: PASSED\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    return 0;
}
