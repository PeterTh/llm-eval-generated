#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr unsigned int kBlockX = 32;
constexpr unsigned int kBlockY = 4;
constexpr unsigned int kBlockZ = 2;

bool checkCuda(const cudaError_t status, const char* expression, const char* file, const int line) {
    if (status == cudaSuccess) {
        return true;
    }

    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                 file, line, expression, cudaGetErrorString(status));
    return false;
}

#define CUDA_CHECK(expression) \
    do { \
        if (!checkCuda((expression), #expression, __FILE__, __LINE__)) { \
            return 1; \
        } \
    } while (false)

__global__ void initializeGrid(Real* const grid, const size_t nx, const size_t ny,
                               const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        const size_t index = z * (nx * ny) + y * nx + x;
        grid[index] = static_cast<Real>(index % 19);
    }
}

// Each launch updates every cell.  The branch is taken only by threads on the
// six outer faces, so interior warps execute the seven-point stencil directly.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t index = z * plane + y * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
        output[index] = input[index];
        return;
    }

    output[index] = (input[index] + input[index - 1] + input[index + 1]
                     + input[index - nx] + input[index + nx]
                     + input[index - plane] + input[index + plane]) / 7.0;
}

dim3 gridDimensions(const size_t nx, const size_t ny, const size_t nz) {
    return dim3(static_cast<unsigned int>((nx + kBlockX - 1) / kBlockX),
                static_cast<unsigned int>((ny + kBlockY - 1) / kBlockY),
                static_cast<unsigned int>((nz + kBlockZ - 1) / kBlockZ));
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // 1. No NaN or Inf values.
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded).
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified.
    size_t nz = 0;  // Will be set to nx if not specified.
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0
        || nx > std::numeric_limits<size_t>::max() / ny
        || nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions and iteration count must be valid.\n");
        return 1;
    }

    const size_t gridSize = nx * ny * nz;
    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid = gridDimensions(nx, ny, nz);

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    std::printf("Initializing grid...\n");

    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1, gridSize * sizeof(*grid1)));
    CUDA_CHECK(cudaMalloc(&grid2, gridSize * sizeof(*grid2)));

    initializeGrid<<<grid, block>>>(grid1, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    std::printf("Running stencil computation...\n");

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencilIteration<<<grid, block>>>(grid1, grid2, nx, ny, nz);
        } else {
            stencilIteration<<<grid, block>>>(grid2, grid1, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, end));
    std::printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMilliseconds));

    const size_t interiorCells = (nx > 2 && ny > 2 && nz > 2)
                               ? (nx - 2) * (ny - 2) * (nz - 2)
                               : 0;
    const double cellUpdates = static_cast<double>(interiorCells)
                               * static_cast<double>(iterations);
    const double mcups = elapsedMilliseconds > 0.0F
                       ? cellUpdates / (static_cast<double>(elapsedMilliseconds) * 1000.0)
                       : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        std::vector<Real> finalGrid(gridSize);
        const Real* const finalDeviceGrid = (iterations & 1) == 0 ? grid1 : grid2;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDeviceGrid, gridSize * sizeof(*grid1),
                              cudaMemcpyDeviceToHost));

        if (printResults) {
            print_results(finalGrid, "Grid");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (!validateResult(finalGrid, nx, ny, nz)) {
                std::printf("Validation: FAILED\n");
                CUDA_CHECK(cudaEventDestroy(start));
                CUDA_CHECK(cudaEventDestroy(end));
                CUDA_CHECK(cudaFree(grid1));
                CUDA_CHECK(cudaFree(grid2));
                return 1;
            }
            std::printf("Validation: PASSED\n");
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaFree(grid2));
    return 0;
}
