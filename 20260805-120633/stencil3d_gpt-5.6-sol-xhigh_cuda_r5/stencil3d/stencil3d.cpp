#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

constexpr unsigned kBlockX = 32;
constexpr unsigned kBlockY = 4;
constexpr unsigned kBlockZ = 2;
constexpr unsigned kThreadsPerBlock = kBlockX * kBlockY * kBlockZ;

__global__ void initializeGrids(Real* __restrict__ grid1,
                                Real* __restrict__ grid2,
                                const size_t elementCount) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elementCount) {
        const Real value = static_cast<Real>(index % 19);
        grid1[index] = value;
        grid2[index] = value;
    }
}

// For this benchmark all operands are finite and nonnegative. A reciprocal
// estimate followed by an FMA residual correction produces the same correctly
// rounded result as division by 7, without invoking CUDA's general FP64 divider.
__device__ __forceinline__ Real averageOfSeven(const Real sum) {
    constexpr Real reciprocal = 0x1.2492492492492p-3;
    const Real estimate = sum * reciprocal;
    const Real remainder = __fma_rn(-7.0, estimate, sum);
    return __fma_rn(remainder, reciprocal, estimate);
}

// Boundaries are invariant. Both buffers are initialized identically, allowing
// every iteration to update only the interior and avoid a full boundary-copy
// pass. A warp spans consecutive X coordinates, so all seven reads and the
// write are coalesced; adjacent warps provide cache reuse in Y and Z.
__global__ __launch_bounds__(kThreadsPerBlock, 2)
void stencilIteration(const Real* __restrict__ input,
                      Real* __restrict__ output,
                      const unsigned nx, const unsigned ny, const unsigned nz,
                      const size_t planeStride) {
    const unsigned x = blockIdx.x * kBlockX + threadIdx.x + 1;
    const unsigned y = blockIdx.y * kBlockY + threadIdx.y + 1;
    const unsigned z = blockIdx.z * kBlockZ + threadIdx.z + 1;

    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) {
        return;
    }

    const size_t index = static_cast<size_t>(z) * planeStride
                       + static_cast<size_t>(y) * nx + x;
    const Real sum = input[index] + input[index - 1] + input[index + 1]
                   + input[index - nx] + input[index + nx]
                   + input[index - planeStride] + input[index + planeStride];
    output[index] = averageOfSeven(sum);
}

bool checkedGridSize(const size_t nx, const size_t ny, const size_t nz,
                     size_t& elementCount, size_t& byteCount) {
    if (nx > std::numeric_limits<size_t>::max() / ny) {
        return false;
    }
    const size_t plane = nx * ny;
    if (plane > std::numeric_limits<size_t>::max() / nz) {
        return false;
    }
    elementCount = plane * nz;
    if (elementCount > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        return false;
    }
    byteCount = elementCount * sizeof(Real);
    return true;
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto bounds = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *bounds.first, *bounds.second);
    if (*bounds.second > 1e6 || *bounds.first < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
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
    if (nx < 3 || ny < 3 || nz < 3
        || nx > std::numeric_limits<unsigned>::max()
        || ny > std::numeric_limits<unsigned>::max()
        || nz > std::numeric_limits<unsigned>::max()
        || iterations < 0) {
        std::fprintf(stderr, "Grid dimensions must be in [3, 2^32-1] and iterations must be nonnegative.\n");
        return 1;
    }

    size_t elementCount = 0;
    size_t byteCount = 0;
    if (!checkedGridSize(nx, ny, nz, elementCount, byteCount)) {
        std::fprintf(stderr, "Grid size is too large.\n");
        return 1;
    }

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    std::printf("Initializing grid...\n");

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, byteCount));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, byteCount));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    constexpr unsigned kInitializeThreads = 256;
    const size_t initializeBlocks = (elementCount + kInitializeThreads - 1) / kInitializeThreads;
    if (initializeBlocks > std::numeric_limits<unsigned>::max()) {
        std::fprintf(stderr, "Grid requires too many CUDA blocks.\n");
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(deviceGrid2));
        CUDA_CHECK(cudaFree(deviceGrid1));
        return 1;
    }
    initializeGrids<<<static_cast<unsigned>(initializeBlocks), kInitializeThreads, 0, stream>>>(
        deviceGrid1, deviceGrid2, elementCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 blocks(static_cast<unsigned>((nx - 2 + kBlockX - 1) / kBlockX),
                      static_cast<unsigned>((ny - 2 + kBlockY - 1) / kBlockY),
                      static_cast<unsigned>((nz - 2 + kBlockZ - 1) / kBlockZ));
    const size_t planeStride = nx * ny;

    // Capture the complete iteration chain once. A single graph submission
    // removes per-iteration CPU launch latency, especially on smaller grids.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExecution = nullptr;
    Real* input = deviceGrid1;
    Real* output = deviceGrid2;
    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stencilIteration<<<blocks, block, 0, stream>>>(
                input, output, static_cast<unsigned>(nx), static_cast<unsigned>(ny),
                static_cast<unsigned>(nz), planeStride);
            std::swap(input, output);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaGraphInstantiate(&graphExecution, graph, nullptr, nullptr, 0));
    }

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    std::printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    if (iterations > 0) {
        CUDA_CHECK(cudaGraphLaunch(graphExecution, stream));
    }
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);

    const double cellUpdates = static_cast<double>(nx - 2)
                             * static_cast<double>(ny - 2)
                             * static_cast<double>(nz - 2) * iterations;
    const double mcups = elapsedMilliseconds > 0.0f
        ? cellUpdates / (static_cast<double>(elapsedMilliseconds) * 1.0e3)
        : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(elementCount);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), input, byteCount, cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(finalGrid);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    if (graphExecution != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graphExecution));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceGrid2));
    CUDA_CHECK(cudaFree(deviceGrid1));
    return valid ? 0 : 1;
}
