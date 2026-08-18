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

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 4;
constexpr int BLOCK_Z = 4;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* operation) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation)                                                   \
    do {                                                                        \
        const cudaError_t cuda_check_error = (operation);                       \
        if (cuda_check_error != cudaSuccess)                                    \
            cudaFailure(cuda_check_error, #operation);                          \
    } while (false)

void initializeGrid(std::vector<Real>& grid) {
    for (size_t i = 0; i < grid.size(); ++i)
        grid[i] = static_cast<Real>(i % 19);
}

// A block computes a 32x4x4 interior brick.  The complete one-cell halo is
// cooperatively staged in shared memory, reducing the seven input loads per
// output to about 2.4 loads for full blocks.
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z, 2)
void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                   const size_t nx, const size_t ny, const size_t nz) {
    constexpr int TILE_X = BLOCK_X + 2;
    constexpr int TILE_Y = BLOCK_Y + 2;
    constexpr int TILE_Z = BLOCK_Z + 2;
    constexpr int TILE_SIZE = TILE_X * TILE_Y * TILE_Z;
    __shared__ Real tile[TILE_SIZE];

    const int thread = (threadIdx.z * BLOCK_Y + threadIdx.y) * BLOCK_X +
                       threadIdx.x;
    constexpr int THREADS = BLOCK_X * BLOCK_Y * BLOCK_Z;
    const size_t x0 = static_cast<size_t>(blockIdx.x) * BLOCK_X;
    const size_t y0 = static_cast<size_t>(blockIdx.y) * BLOCK_Y;
    const size_t z0 = static_cast<size_t>(blockIdx.z) * BLOCK_Z;

    for (int linear = thread; linear < TILE_SIZE; linear += THREADS) {
        const int tx = linear % TILE_X;
        const int yz = linear / TILE_X;
        const int ty = yz % TILE_Y;
        const int tz = yz / TILE_Y;
        const size_t gx = x0 + static_cast<size_t>(tx);
        const size_t gy = y0 + static_cast<size_t>(ty);
        const size_t gz = z0 + static_cast<size_t>(tz);
        tile[linear] = (gx < nx && gy < ny && gz < nz)
                           ? input[(gz * ny + gy) * nx + gx]
                           : Real(0);
    }
    __syncthreads();

    const size_t x = x0 + static_cast<size_t>(threadIdx.x) + 1;
    const size_t y = y0 + static_cast<size_t>(threadIdx.y) + 1;
    const size_t z = z0 + static_cast<size_t>(threadIdx.z) + 1;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1)
        return;

    const int sx = threadIdx.x + 1;
    const int sy = threadIdx.y + 1;
    const int sz = threadIdx.z + 1;
    const int center = (sz * TILE_Y + sy) * TILE_X + sx;
    const Real sum = tile[center] + tile[center - 1] + tile[center + 1] +
                     tile[center - TILE_X] + tile[center + TILE_X] +
                     tile[center - TILE_X * TILE_Y] +
                     tile[center + TILE_X * TILE_Y];
    output[(z * ny + y) * nx + x] = sum / Real(7);
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real val : grid) {
        if (!std::isfinite(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto extrema = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *extrema.first, *extrema.second);
    if (*extrema.second > 1e6 || *extrema.first < -1e6) {
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

} // namespace

int main(int argc, char** argv) {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc)
            nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc)
            ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc)
            nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (std::strcmp(argv[i], "-r") == 0)
            printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
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
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr, "Grid dimensions must be at least 2 and must fit in memory; iterations must be non-negative.\n");
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
    initializeGrid(hostGrid);

    Real *deviceGrid0 = nullptr, *deviceGrid1 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid0, bytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid1, bytes));
    CUDA_CHECK(cudaMemcpy(deviceGrid0, hostGrid.data(), bytes,
                          cudaMemcpyHostToDevice));
    // Both buffers start with identical boundary values. Kernels subsequently
    // update only the interior, so boundary-copy kernels are unnecessary.
    CUDA_CHECK(cudaMemcpy(deviceGrid1, deviceGrid0, bytes,
                          cudaMemcpyDeviceToDevice));

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const dim3 grid(static_cast<unsigned int>((nx - 2 + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny - 2 + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((nz - 2 + BLOCK_Z - 1) / BLOCK_Z));
    Real* input = deviceGrid0;
    Real* output = deviceGrid1;

    cudaStream_t stream;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Capture the complete time-step chain once.  This substantially reduces
    // CPU launch overhead for small/medium grids without changing dependency
    // ordering between alternating buffers.
    cudaGraph_t graphHandle = nullptr;
    cudaGraphExec_t graphExecutable = nullptr;
    if (iterations > 0 && hasInterior) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int iter = 0; iter < iterations; ++iter) {
            stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, nz);
            std::swap(input, output);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graphHandle));
        CUDA_CHECK(cudaGraphInstantiate(&graphExecutable, graphHandle, nullptr,
                                        nullptr, 0));
    }
    CUDA_CHECK(cudaGetLastError());

    std::printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (graphExecutable != nullptr)
        CUDA_CHECK(cudaGraphLaunch(graphExecutable, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    std::printf("Computation time: %.3f ms\n", elapsedMs);
    const double cellUpdates = static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
    const double mcups = elapsedMs > 0.0f ? cellUpdates / (elapsedMs * 1000.0) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate)
        CUDA_CHECK(cudaMemcpy(hostGrid.data(), input, bytes,
                              cudaMemcpyDeviceToHost));
    if (printResults)
        print_results(hostGrid, "Grid");

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(hostGrid);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (graphExecutable != nullptr)
        CUDA_CHECK(cudaGraphExecDestroy(graphExecutable));
    if (graphHandle != nullptr)
        CUDA_CHECK(cudaGraphDestroy(graphHandle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceGrid0));
    CUDA_CHECK(cudaFree(deviceGrid1));
    return valid ? 0 : 1;
}
