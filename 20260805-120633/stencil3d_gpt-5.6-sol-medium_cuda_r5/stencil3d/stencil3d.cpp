#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

[[noreturn]] void cudaFailure(const char* operation, cudaError_t error) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation)                                                   \
    do {                                                                        \
        const cudaError_t cuda_check_error = (operation);                        \
        if (cuda_check_error != cudaSuccess)                                     \
            cudaFailure(#operation, cuda_check_error);                           \
    } while (false)

// X spans a full warp for coalesced rows. The 256-thread 3D block provides
// enough blocks for full occupancy and lets the read-only cache reuse adjacent
// X/Y/Z values without a block-wide shared-memory barrier.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 4;
constexpr int BLOCK_Z = 2;
constexpr int BLOCK_THREADS = BLOCK_X * BLOCK_Y * BLOCK_Z;

__global__ void initializeGrids(Real* __restrict__ first,
                                Real* __restrict__ second,
                                size_t count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        const Real value = static_cast<Real>(index % 19);
        first[index] = value;
        second[index] = value;
    }
}

// The initialized output buffer already contains the immutable boundary, so
// this kernel only writes interior points and no boundary-copy pass is needed.
// Neighboring threads' loads are serviced by the GPU's unified data cache.
__global__ __launch_bounds__(BLOCK_THREADS, 4)
void stencilKernel(const Real* __restrict__ input,
                   Real* __restrict__ output,
                   int nx, int ny, int nz) {
    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + threadIdx.x + 1;
    const int y = static_cast<int>(blockIdx.y) * BLOCK_Y + threadIdx.y + 1;
    const int z = static_cast<int>(blockIdx.z) * BLOCK_Z + threadIdx.z + 1;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1)
        return;

    const size_t planeStride = static_cast<size_t>(nx) * ny;
    const size_t global = static_cast<size_t>(z) * planeStride
                        + static_cast<size_t>(y) * nx + x;

    // Keep the source expression's evaluation order for equivalent FP results.
    output[global] = (input[global]
                    + input[global - 1]
                    + input[global + 1]
                    + input[global - nx]
                    + input[global + nx]
                    + input[global - planeStride]
                    + input[global + planeStride]) / 7.0;
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto range = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *range.first, *range.second);
    if (*range.second > 1e6 || *range.first < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
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
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::strtoull(argv[++i], nullptr, 10);
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
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0
        || nx > static_cast<size_t>(std::numeric_limits<int>::max())
        || ny > static_cast<size_t>(std::numeric_limits<int>::max())
        || nz > static_cast<size_t>(std::numeric_limits<int>::max())
        || nx > std::numeric_limits<size_t>::max() / ny
        || nx * ny > std::numeric_limits<size_t>::max() / nz
        || nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr,
                     "Grid dimensions must be in [3, INT_MAX], their product "
                     "must fit in size_t, and iterations must be nonnegative.\n");
        return 1;
    }

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    std::printf("Initializing grid...\n");

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, bytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr int initializationThreads = 256;
    const unsigned initializationBlocks = static_cast<unsigned>(
        std::min<size_t>((gridSize + initializationThreads - 1)
                         / initializationThreads, 4096));
    initializeGrids<<<initializationBlocks, initializationThreads, 0, stream>>>(
        deviceGrid1, deviceGrid2, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned>((nx - 2 + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned>((ny - 2 + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned>((nz - 2 + BLOCK_Z - 1) / BLOCK_Z));

    // Capture the full alternating sequence once. Graph upload and setup are
    // deliberately outside the measured stencil interval.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const Real* source = (iteration & 1) ? deviceGrid2 : deviceGrid1;
            Real* destination = (iteration & 1) ? deviceGrid1 : deviceGrid2;
            stencilKernel<<<grid, block, 0, stream>>>(
                source, destination, static_cast<int>(nx),
                static_cast<int>(ny), static_cast<int>(nz));
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::printf("Running stencil computation...\n");
    cudaEvent_t start;
    cudaEvent_t finish;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&finish));
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (iterations > 0)
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaEventRecord(finish, stream));
    CUDA_CHECK(cudaEventSynchronize(finish));
    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, finish));

    std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    const double cellUpdates = static_cast<double>(nx - 2)
                             * static_cast<double>(ny - 2)
                             * static_cast<double>(nz - 2)
                             * static_cast<double>(iterations);
    const double mcups = elapsedMilliseconds > 0.0f
        ? cellUpdates / (static_cast<double>(elapsedMilliseconds) * 1000.0)
        : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    int exitCode = 0;
    if (printResults || validate) {
        std::vector<Real> finalGrid(gridSize);
        const Real* finalDeviceGrid = (iterations & 1) ? deviceGrid2 : deviceGrid1;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDeviceGrid, bytes,
                              cudaMemcpyDeviceToHost));
        if (printResults)
            print_results(finalGrid, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(finalGrid)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(finish));
    if (graphExec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));
    return exitCode;
}
