#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// Block tile dimensions for the shared-memory stencil kernel.
// The block has (BX+2)*(BY+2)*(BZ+2) threads and a shared-memory tile of
// (BX+2)*(BY+2)*(BZ+2) doubles.
constexpr int BX = 8;
constexpr int BY = 8;
constexpr int BZ = 8;

// ---------------------------------------------------------------------------
// Kernel: initialise the grid
// ---------------------------------------------------------------------------
__global__ void initKernel(double* grid, size_t nx, size_t ny, size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        size_t idx = z * nx * ny + y * nx + x;
        grid[idx] = static_cast<double>(idx % 19);
    }
}

// ---------------------------------------------------------------------------
// Kernel: 7-point stencil with shared-memory tiling
//
// Each block covers a tile of BX*BY*BZ interior cells plus a one-cell halo
// on every face.  Threads in the halo (tx==0, tx==BX+1, …) only load data
// into shared memory; threads with 1<=tx<=BX compute outputs.
// ---------------------------------------------------------------------------
__global__ void stencilKernel(const double* __restrict__ input,
                              double* __restrict__ output,
                              size_t nx, size_t ny, size_t nz) {
    __shared__ double sdata[BX + 2][BY + 2][BZ + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Global coordinates for loading (with -1 halo offset).
    int gx = static_cast<int>(blockIdx.x) * BX + tx - 1;
    int gy = static_cast<int>(blockIdx.y) * BY + ty - 1;
    int gz = static_cast<int>(blockIdx.z) * BZ + tz - 1;

    // Load into shared memory with bounds checking.
    if (gx >= 0 && gx < static_cast<int>(nx) &&
        gy >= 0 && gy < static_cast<int>(ny) &&
        gz >= 0 && gz < static_cast<int>(nz)) {
        sdata[tx][ty][tz] =
            input[static_cast<size_t>(gz) * nx * ny +
                  static_cast<size_t>(gy) * nx + static_cast<size_t>(gx)];
    } else {
        sdata[tx][ty][tz] = 0.0;
    }

    __syncthreads();

    // Output coordinate (same as load coordinate).
    int ox = gx;
    int oy = gy;
    int oz = gz;

    if (ox >= 0 && ox < static_cast<int>(nx) &&
        oy >= 0 && oy < static_cast<int>(ny) &&
        oz >= 0 && oz < static_cast<int>(nz)) {
        size_t idx =
            static_cast<size_t>(oz) * nx * ny +
            static_cast<size_t>(oy) * nx + static_cast<size_t>(ox);

        // Boundary cells: copy from input (reads global memory).
        if (ox == 0 || ox == static_cast<int>(nx) - 1 ||
            oy == 0 || oy == static_cast<int>(ny) - 1 ||
            oz == 0 || oz == static_cast<int>(nz) - 1) {
            output[idx] = input[idx];
        }
        // Interior cells: 7-point stencil from shared memory.
        else if (tx >= 1 && tx <= BX && ty >= 1 && ty <= BY && tz >= 1 &&
                 tz <= BZ) {
            output[idx] = (sdata[tx][ty][tz] +       // center
                           sdata[tx - 1][ty][tz] +   // left  (x-1)
                           sdata[tx + 1][ty][tz] +   // right (x+1)
                           sdata[tx][ty - 1][tz] +   // front (y-1)
                           sdata[tx][ty + 1][tz] +   // back  (y+1)
                           sdata[tx][ty][tz - 1] +   // bottom(z-1)
                           sdata[tx][ty][tz + 1]) /
                          7.0;                        // top   (z+1)
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side validation (unchanged semantics).
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<Real>& grid) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

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
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
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

    // ----- Device allocation -----
    double* d_grid1 = nullptr;
    double* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(double)));

    // ----- Initialise on device -----
    printf("Initializing grid...\n");
    {
        dim3 iblk(8, 8, 8);
        dim3 igrd((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
        initKernel<<<igrd, iblk>>>(d_grid1, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ----- Stencil iterations -----
    printf("Running stencil computation...\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    dim3 sblk(BX + 2, BY + 2, BZ + 2);
    dim3 sgrd((nx + BX - 1) / BX, (ny + BY - 1) / BY, (nz + BZ - 1) / BZ);

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<sgrd, sblk>>>(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilKernel<<<sgrd, sblk>>>(d_grid2, d_grid1, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    printf("Computation time: %.1f ms\n", elapsedMs);

    // ----- Performance metrics -----
    double cellUpdates =
        static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // ----- Copy result back -----
    std::vector<Real> finalGrid(gridSize);
    const double* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final,
                          gridSize * sizeof(double), cudaMemcpyDeviceToHost));

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // ----- Cleanup -----
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
