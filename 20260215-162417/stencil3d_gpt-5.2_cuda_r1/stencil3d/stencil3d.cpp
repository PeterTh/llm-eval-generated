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

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

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

__device__ __forceinline__ size_t didx3(const int x, const int y, const int z, const int nx, const int ny) {
    return static_cast<size_t>(z) * (static_cast<size_t>(nx) * static_cast<size_t>(ny)) +
           static_cast<size_t>(y) * static_cast<size_t>(nx) +
           static_cast<size_t>(x);
}

// 7-point stencil computation (GPU)
__global__ void stencil7_kernel(const Real* __restrict__ input, Real* __restrict__ output,
                                const int nx, const int ny, const int nz) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    const int z = static_cast<int>(blockIdx.z) * static_cast<int>(blockDim.z) + static_cast<int>(threadIdx.z);

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = didx3(x, y, z, nx, ny);

    // Copy boundary values unchanged
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left   = input[didx3(x - 1, y, z, nx, ny)];
    const Real right  = input[didx3(x + 1, y, z, nx, ny)];
    const Real front  = input[didx3(x, y - 1, z, nx, ny)];
    const Real back   = input[didx3(x, y + 1, z, nx, ny)];
    const Real bottom = input[didx3(x, y, z - 1, nx, ny)];
    const Real top    = input[didx3(x, y, z + 1, nx, ny)];

    // Simple averaging stencil
    output[idx] = (center + left + right + front + back + bottom + top) * (Real(1.0) / Real(7.0));
}

static long runStencilGPU(const std::vector<Real>& h_input, std::vector<Real>& h_output,
                          const int nx, const int ny, const int nz, const int iterations) {
    const size_t gridSize = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const size_t bytes = gridSize * sizeof(Real);

    Real* d_a = nullptr;
    Real* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // Use a 3D block for coalesced X loads and good occupancy
    dim3 block(32, 4, 2); // 256 threads
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));

    Real* in = d_a;
    Real* out = d_b;
    for (int iter = 0; iter < iterations; ++iter) {
        stencil7_kernel<<<grid, block>>>(in, out, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
        Real* tmp = in;
        in = out;
        out = tmp;
    }

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));

    // Copy back final grid (note: if iterations==0, in still points to d_a)
    CUDA_CHECK(cudaMemcpy(h_output.data(), in, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return static_cast<long>(ms + 0.5f);
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
    
    // Run stencil iterations (GPU)
    printf("Running stencil computation...\n");

    const long durationMs = runStencilGPU(grid1, grid2, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), iterations);
    printf("Computation time: %ld ms\n", durationMs);

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    const double seconds = std::max(1.0, static_cast<double>(durationMs)) / 1000.0;
    double mcups = cellUpdates / seconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    const std::vector<Real>& finalGrid = grid2;
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
