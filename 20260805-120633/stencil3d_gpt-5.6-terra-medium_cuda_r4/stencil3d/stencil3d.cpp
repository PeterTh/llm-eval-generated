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

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        const cudaError_t cudaStatus = (call);                                                   \
        if (cudaStatus != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(cudaStatus));                                        \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                        \
    } while (false)

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = static_cast<Real>(idx % 19);
            }
        }
    }
}

// Each thread produces one element.  A 32-wide x dimension gives fully
// coalesced loads/stores for the contiguous x dimension, while the small y/z
// tile exposes enough parallelism without using shared memory for a stencil
// whose neighbors are read only once per output.
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
    const size_t idx = z * plane + y * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
        output[idx] = input[idx];
        return;
    }

    output[idx] = (input[idx] + input[idx - 1] + input[idx + 1] +
                   input[idx - nx] + input[idx + nx] +
                   input[idx - plane] + input[idx + plane]) / 7.0;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

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

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
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
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        std::fprintf(stderr, "Grid dimensions must be positive and iterations must be non-negative.\n");
        return 1;
    }

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    std::vector<Real> hostGrid(gridSize);
    std::printf("Initializing grid...\n");
    initializeGrid(hostGrid, nx, ny, nz);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, bytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, bytes));
    CUDA_CHECK(cudaMemcpy(deviceGrid1, hostGrid.data(), bytes, cudaMemcpyHostToDevice));

    // CUDA is the sole computation path.  The launch covers the complete grid
    // so every output buffer, including its boundary cells, is defined each step.
    const dim3 block(32, 4, 2);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>((nz + block.z - 1) / block.z));
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    std::printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencilIteration<<<grid, block>>>(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            stencilIteration<<<grid, block>>>(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    std::printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMs));

    const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    const double mcups = elapsedMs > 0.0f ? cellUpdates / (static_cast<double>(elapsedMs) * 1.0e3) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const Real* finalDeviceGrid = (iterations & 1) == 0 ? deviceGrid1 : deviceGrid2;
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), finalDeviceGrid, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));

    if (printResults) {
        print_results(hostGrid, "Grid");
    }
    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(hostGrid, nx, ny, nz)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }
    return 0;
}
