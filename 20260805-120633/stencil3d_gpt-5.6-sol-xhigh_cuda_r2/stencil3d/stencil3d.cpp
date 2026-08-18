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

constexpr int kBlockX = 32;
constexpr int kInitThreads = 256;
constexpr size_t kLargeGridThreshold = 8ULL * 1024ULL * 1024ULL;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file,
                 line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(const cudaError_t error, const char* expression,
                      const char* file, const int line) {
    if (error != cudaSuccess) {
        cudaFailure(error, expression, file, line);
    }
}

#define CUDA_CHECK(expression) \
    cudaCheck((expression), #expression, __FILE__, __LINE__)

__global__ void initializeGrids(Real* __restrict__ grid1,
                                Real* __restrict__ grid2,
                                const size_t elementCount) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = first; index < elementCount; index += stride) {
        const Real value = static_cast<Real>(index % 19);
        grid1[index] = value;
        grid2[index] = value;
    }
}

// X is mapped to complete warps for fully coalesced access.  On large grids a
// thread advances through a short Z run, retaining bottom and centre in
// registers.  The cache hierarchy supplies the overlapping X/Y reads more
// cheaply than an explicitly synchronized shared-memory tile on modern GPUs.
template <typename Index, int BlockY, int ZChunk>
__global__ __launch_bounds__(kBlockX * BlockY, 8)
void stencilKernel(const Real* __restrict__ input,
                   Real* __restrict__ output, const Index nx, const Index ny,
                   const Index nz) {
    const Index xGridStride = static_cast<Index>(gridDim.x) * kBlockX;
    const Index yGridStride = static_cast<Index>(gridDim.y) * BlockY;
    const Index zGridStride = static_cast<Index>(gridDim.z) * ZChunk;

    for (Index xBase = static_cast<Index>(blockIdx.x) * kBlockX + 1;
         xBase < nx - 1; xBase += xGridStride) {
        const Index x = xBase + static_cast<Index>(threadIdx.x);
        if (x >= nx - 1) continue;

        for (Index yBase = static_cast<Index>(blockIdx.y) * BlockY + 1;
             yBase < ny - 1; yBase += yGridStride) {
            const Index y = yBase + static_cast<Index>(threadIdx.y);
            if (y >= ny - 1) continue;

            const Index planeStride = nx * ny;
            for (Index zBegin =
                     static_cast<Index>(blockIdx.z) * ZChunk + 1;
                 zBegin < nz - 1; zBegin += zGridStride) {
                const Index zEnd =
                    min(zBegin + static_cast<Index>(ZChunk), nz - 1);
                Index index = zBegin * planeStride + y * nx + x;
                Real bottom = input[index - planeStride];
                Real center = input[index];

#pragma unroll
                for (Index z = zBegin; z < zEnd; ++z) {
                    const Real left = input[index - 1];
                    const Real right = input[index + 1];
                    const Real front = input[index - nx];
                    const Real back = input[index + nx];
                    const Real top = input[index + planeStride];
                    output[index] =
                        (center + left + right + front + back + bottom + top) /
                        7.0;
                    bottom = center;
                    center = top;
                    index += planeStride;
                }
            }
        }
    }
}

struct GraphRun {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    Real* finalGrid = nullptr;
};

template <typename Index, int BlockY, int ZChunk>
GraphRun captureStencilGraph(Real* grid1, Real* grid2, const size_t nx,
                             const size_t ny, const size_t nz,
                             const int iterations,
                             const cudaDeviceProp& deviceProperties,
                             const cudaStream_t stream) {
    const size_t requestedBlocksX =
        (nx - 2 + static_cast<size_t>(kBlockX) - 1) / kBlockX;
    const size_t requestedBlocksY =
        (ny - 2 + static_cast<size_t>(BlockY) - 1) / BlockY;
    const size_t requestedBlocksZ =
        (nz - 2 + static_cast<size_t>(ZChunk) - 1) / ZChunk;
    const unsigned int blocksX = static_cast<unsigned int>(std::min<size_t>(
        requestedBlocksX, deviceProperties.maxGridSize[0]));
    const unsigned int blocksY = static_cast<unsigned int>(std::min<size_t>(
        requestedBlocksY, deviceProperties.maxGridSize[1]));
    const unsigned int blocksZ = static_cast<unsigned int>(std::min<size_t>(
        requestedBlocksZ, deviceProperties.maxGridSize[2]));
    const dim3 block(kBlockX, BlockY);
    const dim3 grid(blocksX, blocksY, blocksZ);
    const Index kernelNx = static_cast<Index>(nx);
    const Index kernelNy = static_cast<Index>(ny);
    const Index kernelNz = static_cast<Index>(nz);

    GraphRun run;
    Real* input = grid1;
    Real* output = grid2;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        stencilKernel<Index, BlockY, ZChunk><<<grid, block, 0, stream>>>(
            input, output, kernelNx, kernelNy, kernelNz);
        std::swap(input, output);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &run.graph));
    CUDA_CHECK(cudaGraphInstantiate(&run.executable, run.graph, nullptr,
                                    nullptr, 0));
    run.finalGrid = input;
    return run;
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (std::isnan(value) || std::isinf(value)) {
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

    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-x") == 0 && argument + 1 < argc) {
            nx = std::strtoull(argv[++argument], nullptr, 10);
        } else if (std::strcmp(argv[argument], "-y") == 0 &&
                   argument + 1 < argc) {
            ny = std::strtoull(argv[++argument], nullptr, 10);
        } else if (std::strcmp(argv[argument], "-z") == 0 &&
                   argument + 1 < argc) {
            nz = std::strtoull(argv[++argument], nullptr, 10);
        } else if (std::strcmp(argv[argument], "-i") == 0 &&
                   argument + 1 < argc) {
            iterations = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[argument]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0) {
        std::fprintf(stderr,
                     "Grid dimensions must be at least 2 and iterations must "
                     "be non-negative.\n");
        return 1;
    }
    if (ny > std::numeric_limits<size_t>::max() / nx ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny)) {
        std::fprintf(stderr, "Grid size exceeds addressable memory.\n");
        return 1;
    }

    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr, "Grid byte size exceeds addressable memory.\n");
        return 1;
    }
    const size_t gridBytes = gridSize * sizeof(Real);

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    int device = 0;
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, gridBytes));

    std::printf("Initializing grid...\n");
    const size_t requiredInitBlocks =
        (gridSize + kInitThreads - 1) / kInitThreads;
    const size_t usefulInitBlocks =
        static_cast<size_t>(deviceProperties.multiProcessorCount) * 32;
    const unsigned int initBlocks = static_cast<unsigned int>(std::min(
        {requiredInitBlocks, usefulInitBlocks,
         static_cast<size_t>(deviceProperties.maxGridSize[0])}));
    initializeGrids<<<initBlocks, kInitThreads>>>(deviceGrid1, deviceGrid2,
                                                  gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Running stencil computation...\n");
    cudaStream_t stream = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    GraphRun run;
    run.finalGrid = deviceGrid1;
    const size_t interiorCells =
        (nx - 2) * static_cast<size_t>(ny - 2) * static_cast<size_t>(nz - 2);
    if (interiorCells != 0 && iterations != 0) {
        if (gridSize <= std::numeric_limits<unsigned int>::max()) {
            if (interiorCells < kLargeGridThreshold) {
                run = captureStencilGraph<unsigned int, 2, 1>(
                    deviceGrid1, deviceGrid2, nx, ny, nz, iterations,
                    deviceProperties, stream);
            } else {
                run = captureStencilGraph<unsigned int, 4, 8>(
                    deviceGrid1, deviceGrid2, nx, ny, nz, iterations,
                    deviceProperties, stream);
            }
        } else {
            run = captureStencilGraph<size_t, 4, 8>(
                deviceGrid1, deviceGrid2, nx, ny, nz, iterations,
                deviceProperties, stream);
        }
    }

    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    if (run.executable != nullptr) {
        CUDA_CHECK(cudaGraphLaunch(run.executable, stream));
    }
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(
        cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));

    std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    const double cellUpdates = static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
    const double elapsedSeconds = elapsedMilliseconds / 1000.0;
    const double mcups =
        elapsedSeconds > 0.0 ? cellUpdates / elapsedSeconds / 1e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), run.finalGrid, gridBytes,
                              cudaMemcpyDeviceToHost));
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

    if (run.executable != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(run.executable));
        CUDA_CHECK(cudaGraphDestroy(run.graph));
    }
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));
    return valid ? 0 : 1;
}
