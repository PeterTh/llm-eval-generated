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

namespace {

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr unsigned int blockX = 32;
constexpr unsigned int blockY = 4;
constexpr unsigned int blockZ = 2;

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t elementCount) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elementCount; index += stride) {
        grid[index] = static_cast<Real>(index % 19);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx,
                              const size_t ny,
                              const size_t nz,
                              const size_t planeSize) {
    const size_t xStride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t yStride = static_cast<size_t>(blockDim.y) * gridDim.y;
    const size_t zStride = static_cast<size_t>(blockDim.z) * gridDim.z;

    for (size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
         z < nz; z += zStride) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += yStride) {
            for (size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 x < nx; x += xStride) {
                const size_t index = z * planeSize + y * nx + x;

                if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
                    output[index] = input[index];
                } else {
                    output[index] = (input[index] +
                                     input[index - 1] + input[index + 1] +
                                     input[index - nx] + input[index + nx] +
                                     input[index - planeSize] + input[index + planeSize]) / 7.0;
                }
            }
        }
    }
}

} // namespace

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 7-point stencil computation on the GPU. The boundary copy is fused with
// the stencil so every grid element is written exactly once per iteration.
void stencilIteration(const Real* input,
                      Real* output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const dim3 block(blockX, blockY, blockZ);
    const auto ceilDiv = [](const size_t value, const unsigned int divisor) {
        return static_cast<unsigned int>((value + divisor - 1) / divisor);
    };
    const unsigned int gridZ = std::min(ceilDiv(nz, blockZ), 65535u);
    const dim3 grid(ceilDiv(nx, blockX), ceilDiv(ny, blockY), gridZ);
    stencilKernel<<<grid, block>>>(input, output, nx, ny, nz, nx * ny);
    checkCuda(cudaGetLastError(), "stencil kernel launch");
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
    
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        fprintf(stderr, "Grid dimensions must be positive and iterations non-negative\n");
        return 1;
    }

    // Keep both buffers resident on the GPU for the entire computation.
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    checkCuda(cudaMalloc(&deviceGrid1, gridSize * sizeof(Real)), "allocating device grid 1");
    checkCuda(cudaMalloc(&deviceGrid2, gridSize * sizeof(Real)), "allocating device grid 2");

    // Initialize
    printf("Initializing grid...\n");
    const size_t initBlocks = std::min<size_t>((gridSize + 255) / 256, 65535);
    initializeGridKernel<<<static_cast<unsigned int>(std::max<size_t>(initBlocks, 1)), 256>>>(deviceGrid1, gridSize);
    checkCuda(cudaGetLastError(), "initialization kernel launch");
    checkCuda(cudaDeviceSynchronize(), "initialization kernel");

    // Run stencil iterations
    printf("Running stencil computation...\n");
    cudaEvent_t start;
    cudaEvent_t end;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    checkCuda(cudaEventRecord(start), "recording start event");
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            stencilIteration(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
    }

    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "waiting for stencil kernels");
    float elapsedMilliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, end), "measuring stencil time");
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");

    const auto duration = std::chrono::milliseconds(static_cast<long long>(elapsedMilliseconds));

    const Real* finalDeviceGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
    checkCuda(cudaMemcpy(grid1.data(), finalDeviceGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost),
              "copying final grid to host");
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = grid1;
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            checkCuda(cudaFree(deviceGrid1), "freeing device grid 1");
            checkCuda(cudaFree(deviceGrid2), "freeing device grid 2");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            checkCuda(cudaFree(deviceGrid1), "freeing device grid 1");
            checkCuda(cudaFree(deviceGrid2), "freeing device grid 2");
            return 1;
        }
    }

    checkCuda(cudaFree(deviceGrid1), "freeing device grid 1");
    checkCuda(cudaFree(deviceGrid2), "freeing device grid 2");
    return 0;
}
