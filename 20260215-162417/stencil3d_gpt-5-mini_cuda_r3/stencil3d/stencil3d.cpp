#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline __host__ __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA error checking
static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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

// Device kernel: compute interior stencil points
__global__ void stencilKernel(const Real* input, Real* output, int nx, int ny, int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // maps to [0..nx-3]
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // maps to [0..ny-3]
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // maps to [0..nz-3]

    int gx = ix + 1; // actual x in [1..nx-2]
    int gy = iy + 1;
    int gz = iz + 1;

    if (gx >= 1 && gx <= nx - 2 && gy >= 1 && gy <= ny - 2 && gz >= 1 && gz <= nz - 2) {
        const size_t idx = static_cast<size_t>(gz) * (static_cast<size_t>(nx) * static_cast<size_t>(ny))
                         + static_cast<size_t>(gy) * static_cast<size_t>(nx) + static_cast<size_t>(gx);

        const Real center = input[idx];
        const Real left = input[idx3(gx-1, gy, gz, nx, ny)];
        const Real right = input[idx3(gx+1, gy, gz, nx, ny)];
        const Real front = input[idx3(gx, gy-1, gz, nx, ny)];
        const Real back = input[idx3(gx, gy+1, gz, nx, ny)];
        const Real bottom = input[idx3(gx, gy, gz-1, nx, ny)];
        const Real top = input[idx3(gx, gy, gz+1, nx, ny)];

        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// Device kernel: copy boundary values from input to output for all boundary indices
__global__ void copyBoundaryKernel(const Real* input, Real* output, int nx, int ny, int nz) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    if (i >= total) return;

    int z = static_cast<int>(i / (static_cast<size_t>(nx) * static_cast<size_t>(ny)));
    int rem = static_cast<int>(i % (static_cast<size_t>(nx) * static_cast<size_t>(ny)));
    int y = rem / nx;
    int x = rem % nx;

    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
        output[i] = input[i];
    }
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

    // Device buffers
    Real *d_grid1 = nullptr, *d_grid2 = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_grid1), gridSize * sizeof(Real)), "alloc d_grid1");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_grid2), gridSize * sizeof(Real)), "alloc d_grid2");

    // Copy initial grid to device
    cudaCheck(cudaMemcpy(d_grid1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice), "copy to d_grid1");

    // Launch configuration
    int bx = 8, by = 8, bz = 8; // 512 threads per block
    dim3 blockDim(bx, by, bz);
    dim3 gridDim((int)((nx - 2 + bx - 1) / bx), (int)((ny - 2 + by - 1) / by), (int)((nz - 2 + bz - 1) / bz));

    // Boundary copy config (1D)
    size_t threads1D = 256;
    size_t blocks1D = (gridSize + threads1D - 1) / threads1D;

    // Run stencil iterations
    printf("Running stencil computation (CUDA)...\n");
    auto start = std::chrono::high_resolution_clock::now();

    Real* d_in = d_grid1;
    Real* d_out = d_grid2;

    for (int iter = 0; iter < iterations; ++iter) {
        // compute interior
        stencilKernel<<<gridDim, blockDim>>>(d_in, d_out, (int)nx, (int)ny, (int)nz);
        cudaCheck(cudaGetLastError(), "stencilKernel launch");

        // copy boundaries from input to output
        copyBoundaryKernel<<<(unsigned int)blocks1D, (unsigned int)threads1D>>>(d_in, d_out, (int)nx, (int)ny, (int)nz);
        cudaCheck(cudaGetLastError(), "copyBoundaryKernel launch");

        cudaCheck(cudaDeviceSynchronize(), "kernel sync");

        // swap
        Real* tmp = d_in; d_in = d_out; d_out = tmp;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy back final result
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_in; // d_in points to the latest input after swaps
    Real* host_final = (iterations % 2 == 0) ? grid1.data() : grid2.data();
    // Note: after loop, d_in points to the last-written buffer, so final is d_in if iterations odd.
    if (iterations % 2 == 0) {
        // even iterations: original d_grid1 still holds initial and was not last-written, but swap logic ensures d_grid1 is final
        cudaCheck(cudaMemcpy(host_final, d_grid1, gridSize * sizeof(Real), cudaMemcpyDeviceToHost), "copy final even");
    } else {
        cudaCheck(cudaMemcpy(host_final, d_in, gridSize * sizeof(Real), cudaMemcpyDeviceToHost), "copy final odd");
    }

    // Free device memory
    cudaCheck(cudaFree(d_grid1), "free d_grid1");
    cudaCheck(cudaFree(d_grid2), "free d_grid2");

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
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
