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

// 3D index calculation
inline __host__ __device__ size_t idx3(const int x, const int y, const int z, const int nx, const int ny) noexcept {
    return static_cast<size_t>(z) * (static_cast<size_t>(nx) * static_cast<size_t>(ny)) + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(nx), static_cast<int>(ny));
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// CUDA kernel implementing the 7-point stencil (handles boundaries by copying)
__global__ void stencilKernel(const Real* input, Real* output, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t id = idx3(x, y, z, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        output[id] = input[id];
        return;
    }

    const size_t center = id;
    const size_t left   = idx3(x - 1, y, z, nx, ny);
    const size_t right  = idx3(x + 1, y, z, nx, ny);
    const size_t front  = idx3(x, y - 1, z, nx, ny);
    const size_t back   = idx3(x, y + 1, z, nx, ny);
    const size_t bottom = idx3(x, y, z - 1, nx, ny);
    const size_t top    = idx3(x, y, z + 1, nx, ny);

    output[center] = (input[center] + input[left] + input[right] + input[front] + input[back] + input[bottom] + input[top]) / 7.0;
}

inline void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(e));
        exit(1);
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
    
    // Prepare GPU
    double bytes = static_cast<double>(gridSize) * sizeof(Real);
    Real* d_in = nullptr;
    Real* d_out = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_in), static_cast<size_t>(bytes)), "alloc d_in");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_out), static_cast<size_t>(bytes)), "alloc d_out");
    cudaCheck(cudaMemcpy(d_in, grid1.data(), static_cast<size_t>(bytes), cudaMemcpyHostToDevice), "copy to device");

    // Launch parameters: tune block size for 3D
    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (nz + block.z - 1) / block.z);

    // Run stencil iterations on GPU
    printf("Running stencil computation on GPU...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        stencilKernel<<<grid, block>>>(d_in, d_out, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
        cudaCheck(cudaGetLastError(), "kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "kernel sync");
        // swap pointers
        Real* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy final result back to host (d_in holds the latest result)
    cudaCheck(cudaMemcpy(grid1.data(), d_in, static_cast<size_t>(bytes), cudaMemcpyDeviceToHost), "copy to host");

    // If original code expected final in grid2 when iterations odd, keep same semantics by copying
    if (iterations % 2 != 0) {
        grid2 = grid1;
    }

    // Free device memory
    cudaCheck(cudaFree(d_in), "free d_in");
    cudaCheck(cudaFree(d_out), "free d_out");

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
