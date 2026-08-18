#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// One warp spans X so every center and neighbor stream is coalesced.  Four
// warps per block provide full occupancy while retaining good Y/Z cache reuse.
constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kThreadsPerBlock = kBlockX * kBlockY;

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t error) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        cudaFailure(operation, error);
    }
}

template <bool ComputePotential>
__device__ __forceinline__ void updateCell(
    double* __restrict__ output, const double* __restrict__ stencilInput,
    const double* __restrict__ oldConcentration, const size_t nx,
    const size_t ny, const size_t nz, const size_t planeStride, const size_t x,
    const size_t y, const size_t z) {
    const size_t xMinus = x == 0 ? x : x - 1;
    const size_t xPlus = x + 1 == nx ? x : x + 1;
    const size_t yMinus = y == 0 ? y : y - 1;
    const size_t yPlus = y + 1 == ny ? y : y + 1;
    const size_t zMinus = z == 0 ? z : z - 1;
    const size_t zPlus = z + 1 == nz ? z : z + 1;
    const size_t index = z * planeStride + y * nx + x;
    const double value = stencilInput[index];

    // dx == dy == dz == 1.0 in the original benchmark.  Keep the same
    // per-axis association and clamped boundary conditions.
    const double xx = stencilInput[z * planeStride + y * nx + xPlus] +
                      stencilInput[z * planeStride + y * nx + xMinus] -
                      2.0 * value;
    const double yy = stencilInput[z * planeStride + yPlus * nx + x] +
                      stencilInput[z * planeStride + yMinus * nx + x] -
                      2.0 * value;
    const double zz = stencilInput[zPlus * planeStride + y * nx + x] +
                      stencilInput[zMinus * planeStride + y * nx + x] -
                      2.0 * value;
    const double laplacian = xx + yy + zz;

    if constexpr (ComputePotential) {
        // e_AA=e_BB=-2/9 and e_AB=2/9 reduce the original bulk-energy
        // derivative algebraically to c^3-c.
        output[index] = value * value * value - value - 0.5 * laplacian;
    } else {
        // D=1 and dt=0.01.
        output[index] = oldConcentration[index] + 0.01 * laplacian;
    }
}

// Both phases are the same seven-point stencil.  Specializing the bulk term
// at compile time keeps these hot-path kernels at 28 registers on sm_86 and
// avoids runtime branches.  The potential remains resident on the GPU and is
// immediately consumed by the update kernel.
template <bool ComputePotential>
__global__ __launch_bounds__(kThreadsPerBlock) void stencilKernel(
    double* __restrict__ output, const double* __restrict__ stencilInput,
    const double* __restrict__ oldConcentration, const size_t nx,
    const size_t ny, const size_t nz, const size_t planeStride) {
    const size_t x = static_cast<size_t>(blockIdx.x) * kBlockX + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + threadIdx.y;
    const size_t z = blockIdx.z;
    if (x < nx && y < ny) {
        updateCell<ComputePotential>(output, stencilInput, oldConcentration, nx,
                                     ny, nz, planeStride, x, y, z);
    }
}

// The fast kernel covers every domain that fits CUDA's 3-D launch limits.
// Grid-stride fallback kernels preserve GPU parallelism for more elongated
// domains without burdening the common path with extra loops and registers.
template <bool ComputePotential>
__global__ __launch_bounds__(kThreadsPerBlock) void largeGridStencilKernel(
    double* __restrict__ output, const double* __restrict__ stencilInput,
    const double* __restrict__ oldConcentration, const size_t nx,
    const size_t ny, const size_t nz, const size_t planeStride) {
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * kBlockY) {
            for (size_t x =
                     static_cast<size_t>(blockIdx.x) * kBlockX + threadIdx.x;
                 x < nx; x += static_cast<size_t>(gridDim.x) * kBlockX) {
                updateCell<ComputePotential>(output, stencilInput,
                                             oldConcentration, nx, ny, nz,
                                             planeStride, x, y, z);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& concentration) {
    const size_t volume = concentration.size();
    for (size_t linearId = 0; linearId < volume; ++linearId) {
        const double pseudo =
            ((linearId + 1) * size_t{1299709} % volume) /
            static_cast<double>(volume);
        concentration[linearId] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& concentration) {
    for (const double value : concentration) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto [minimum, maximum] =
        std::minmax_element(concentration.begin(), concentration.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *minimum, *maximum);
    if (*maximum > 10.0 || *minimum < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

bool parsePositiveSize(const char* text, size_t& value) {
    if (text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseNonnegativeInt(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool checkedVolume(const size_t nx, const size_t ny, const size_t nz,
                   size_t& planeStride, size_t& volume) {
    if (nx > std::numeric_limits<size_t>::max() / ny) {
        return false;
    }
    planeStride = nx * ny;
    if (planeStride > std::numeric_limits<size_t>::max() / nz) {
        return false;
    }
    volume = planeStride * nz;
    return volume <= std::numeric_limits<size_t>::max() / sizeof(double);
}

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-x") == 0 && argument + 1 < argc) {
            if (!parsePositiveSize(argv[++argument], nx)) {
                std::fprintf(stderr, "Invalid X grid size\n");
                return EXIT_FAILURE;
            }
        } else if (std::strcmp(argv[argument], "-y") == 0 &&
                   argument + 1 < argc) {
            if (!parsePositiveSize(argv[++argument], ny)) {
                std::fprintf(stderr, "Invalid Y grid size\n");
                return EXIT_FAILURE;
            }
        } else if (std::strcmp(argv[argument], "-z") == 0 &&
                   argument + 1 < argc) {
            if (!parsePositiveSize(argv[++argument], nz)) {
                std::fprintf(stderr, "Invalid Z grid size\n");
                return EXIT_FAILURE;
            }
        } else if (std::strcmp(argv[argument], "-i") == 0 &&
                   argument + 1 < argc) {
            if (!parseNonnegativeInt(argv[++argument], iterations)) {
                std::fprintf(stderr, "Invalid number of time steps\n");
                return EXIT_FAILURE;
            }
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[argument]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    size_t planeStride = 0;
    size_t gridSize = 0;
    if (!checkedVolume(nx, ny, nz, planeStride, gridSize)) {
        std::fprintf(stderr, "Grid dimensions are too large\n");
        return EXIT_FAILURE;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    int device = 0;
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDevice(&device), "selecting the CUDA device");
    cudaCheck(cudaGetDeviceProperties(&properties, device),
              "querying the CUDA device");

    const size_t blocksX = (nx + kBlockX - 1) / kBlockX;
    const size_t blocksY = (ny + kBlockY - 1) / kBlockY;
    const size_t launchBlocksX =
        std::min(blocksX, static_cast<size_t>(properties.maxGridSize[0]));
    const size_t launchBlocksY =
        std::min(blocksY, static_cast<size_t>(properties.maxGridSize[1]));
    const size_t launchBlocksZ =
        std::min(nz, static_cast<size_t>(properties.maxGridSize[2]));
    const bool useLargeGridKernel = launchBlocksX != blocksX ||
                                    launchBlocksY != blocksY ||
                                    launchBlocksZ != nz;
    std::printf("CUDA device: %s\n", properties.name);

    std::vector<double> concentration(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration);

    double* deviceCurrent = nullptr;
    double* deviceNext = nullptr;
    double* devicePotential = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    cudaCheck(cudaMalloc(&deviceCurrent, bytes), "allocating the current field");
    cudaCheck(cudaMalloc(&deviceNext, bytes), "allocating the next field");
    cudaCheck(cudaMalloc(&devicePotential, bytes),
              "allocating the chemical-potential field");
    cudaCheck(cudaMemcpy(deviceCurrent, concentration.data(), bytes,
                         cudaMemcpyHostToDevice),
              "copying the initial field to the GPU");
    if (useLargeGridKernel) {
        cudaCheck(cudaFuncSetCacheConfig(largeGridStencilKernel<true>,
                                         cudaFuncCachePreferL1),
                  "configuring the large-grid potential kernel");
        cudaCheck(cudaFuncSetCacheConfig(largeGridStencilKernel<false>,
                                         cudaFuncCachePreferL1),
                  "configuring the large-grid update kernel");
    } else {
        cudaCheck(cudaFuncSetCacheConfig(stencilKernel<true>,
                                         cudaFuncCachePreferL1),
                  "configuring the CUDA potential kernel");
        cudaCheck(cudaFuncSetCacheConfig(stencilKernel<false>,
                                         cudaFuncCachePreferL1),
                  "configuring the CUDA update kernel");
    }

    cudaStream_t stream = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "creating the CUDA stream");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExecution = nullptr;
    const dim3 block(kBlockX, kBlockY, 1);
    const dim3 grid(static_cast<unsigned int>(launchBlocksX),
                    static_cast<unsigned int>(launchBlocksY),
                    static_cast<unsigned int>(launchBlocksZ));

    if (iterations > 0) {
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                  "beginning CUDA Graph capture");
        for (int step = 0; step < iterations; ++step) {
            if (useLargeGridKernel) {
                largeGridStencilKernel<true><<<grid, block, 0, stream>>>(
                    devicePotential, deviceCurrent, nullptr, nx, ny, nz,
                    planeStride);
                largeGridStencilKernel<false><<<grid, block, 0, stream>>>(
                    deviceNext, devicePotential, deviceCurrent, nx, ny, nz,
                    planeStride);
            } else {
                stencilKernel<true><<<grid, block, 0, stream>>>(
                    devicePotential, deviceCurrent, nullptr, nx, ny, nz,
                    planeStride);
                stencilKernel<false><<<grid, block, 0, stream>>>(
                    deviceNext, devicePotential, deviceCurrent, nx, ny, nz,
                    planeStride);
            }
            std::swap(deviceCurrent, deviceNext);
        }
        cudaCheck(cudaGetLastError(), "capturing the CUDA stencil launches");
        cudaCheck(cudaStreamEndCapture(stream, &graph),
                  "ending CUDA Graph capture");
        cudaCheck(cudaGraphInstantiate(&graphExecution, graph, nullptr, nullptr,
                                       0),
                  "instantiating the CUDA Graph");
    }

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaCheck(cudaEventCreate(&startEvent), "creating the start event");
    cudaCheck(cudaEventCreate(&stopEvent), "creating the stop event");

    std::printf("Running Cahn-Hilliard simulation...\n");
    cudaCheck(cudaEventRecord(startEvent, stream), "recording the start event");
    if (iterations > 0) {
        cudaCheck(cudaGraphLaunch(graphExecution, stream),
                  "launching the CUDA Graph");
    }
    cudaCheck(cudaEventRecord(stopEvent, stream), "recording the stop event");
    cudaCheck(cudaEventSynchronize(stopEvent), "waiting for GPU computation");

    float elapsedMilliseconds = 0.0F;
    cudaCheck(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent),
              "measuring GPU computation time");
    std::printf("Computation time: %.3f ms\n",
                static_cast<double>(elapsedMilliseconds));
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = elapsedMilliseconds > 0.0F
                             ? cellUpdates / elapsedMilliseconds / 1000.0
                             : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    cudaCheck(cudaMemcpyAsync(concentration.data(), deviceCurrent, bytes,
                              cudaMemcpyDeviceToHost, stream),
              "copying the final field to the CPU");
    cudaCheck(cudaStreamSynchronize(stream), "waiting for the final field");

    cudaCheck(cudaEventDestroy(startEvent), "destroying the start event");
    cudaCheck(cudaEventDestroy(stopEvent), "destroying the stop event");
    if (graphExecution != nullptr) {
        cudaCheck(cudaGraphExecDestroy(graphExecution),
                  "destroying the executable CUDA Graph");
    }
    if (graph != nullptr) {
        cudaCheck(cudaGraphDestroy(graph), "destroying the CUDA Graph");
    }
    cudaCheck(cudaStreamDestroy(stream), "destroying the CUDA stream");
    cudaCheck(cudaFree(devicePotential),
              "freeing the chemical-potential field");
    cudaCheck(cudaFree(deviceNext), "freeing the next field");
    cudaCheck(cudaFree(deviceCurrent), "freeing the current field");

    if (printResults) {
        print_results(concentration, "Concentration");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(concentration)) {
            std::printf("Validation: PASSED\n");
            return EXIT_SUCCESS;
        }
        std::printf("Validation: FAILED\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
