#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

// A compact, nearly cubic tile minimizes halo traffic.  256 threads allow
// several resident blocks per SM, which is important for hiding stencil-load
// latency on both small and large grids.
constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_Z = 4;
constexpr int BLOCK_THREADS = BLOCK_X * BLOCK_Y * BLOCK_Z;
constexpr int TILE_X = BLOCK_X + 2;
constexpr int TILE_Y = BLOCK_Y + 2;
constexpr int TILE_Z = BLOCK_Z + 2;
constexpr int TILE_SIZE = TILE_X * TILE_Y * TILE_Z;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

__device__ __forceinline__ int tileIndex(const int x, const int y, const int z) {
    return (z * TILE_Y + y) * TILE_X + x;
}

__device__ __forceinline__ unsigned clampCoordinate(const long long coordinate,
                                                     const unsigned extent) {
    if (coordinate <= 0) {
        return 0;
    }
    const unsigned value = static_cast<unsigned>(coordinate);
    return value < extent ? value : extent - 1;
}

__global__ __launch_bounds__(BLOCK_THREADS)
void computeChemicalPotential(double* __restrict__ mu,
                              const double* __restrict__ concentration,
                              const unsigned nx, const unsigned ny,
                              const unsigned nz, const size_t plane) {
    __shared__ double tile[TILE_SIZE];

    const int tid = static_cast<int>(threadIdx.x)
                  + BLOCK_X * (static_cast<int>(threadIdx.y)
                  + BLOCK_Y * static_cast<int>(threadIdx.z));
    const unsigned originX = blockIdx.x * BLOCK_X;
    const unsigned originY = blockIdx.y * BLOCK_Y;
    const unsigned originZ = blockIdx.z * BLOCK_Z;

    // Stage the input and its clamped one-cell halo.  Neighboring threads and
    // all three stencil directions then reuse these values from shared memory.
    for (int linear = tid; linear < TILE_SIZE; linear += BLOCK_THREADS) {
        const int localX = linear % TILE_X;
        const int yz = linear / TILE_X;
        const int localY = yz % TILE_Y;
        const int localZ = yz / TILE_Y;
        const unsigned x = clampCoordinate(
            static_cast<long long>(originX) + localX - 1, nx);
        const unsigned y = clampCoordinate(
            static_cast<long long>(originY) + localY - 1, ny);
        const unsigned z = clampCoordinate(
            static_cast<long long>(originZ) + localZ - 1, nz);
        tile[linear] = concentration[static_cast<size_t>(z) * plane
                                   + static_cast<size_t>(y) * nx + x];
    }
    __syncthreads();

    const unsigned x = originX + threadIdx.x;
    const unsigned y = originY + threadIdx.y;
    const unsigned z = originZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int centerIndex = tileIndex(static_cast<int>(threadIdx.x) + 1,
                                      static_cast<int>(threadIdx.y) + 1,
                                      static_cast<int>(threadIdx.z) + 1);
    const double center = tile[centerIndex];
    const double cxx = tile[centerIndex + 1] + tile[centerIndex - 1]
                     - 2.0 * center;
    const double cyy = tile[centerIndex + TILE_X] + tile[centerIndex - TILE_X]
                     - 2.0 * center;
    constexpr int tilePlane = TILE_X * TILE_Y;
    const double czz = tile[centerIndex + tilePlane]
                     + tile[centerIndex - tilePlane] - 2.0 * center;

    // The original fixed e_AA/e_BB/e_AB expression simplifies to c^3-c.
    const double bulk = center * center * center - center;
    const size_t globalIndex = static_cast<size_t>(z) * plane
                             + static_cast<size_t>(y) * nx + x;
    mu[globalIndex] = bulk - 0.5 * (cxx + cyy + czz);
}

__global__ __launch_bounds__(BLOCK_THREADS)
void updateConcentration(double* __restrict__ output,
                         const double* __restrict__ input,
                         const double* __restrict__ mu,
                         const unsigned nx, const unsigned ny,
                         const unsigned nz, const size_t plane) {
    __shared__ double tile[TILE_SIZE];

    const int tid = static_cast<int>(threadIdx.x)
                  + BLOCK_X * (static_cast<int>(threadIdx.y)
                  + BLOCK_Y * static_cast<int>(threadIdx.z));
    const unsigned originX = blockIdx.x * BLOCK_X;
    const unsigned originY = blockIdx.y * BLOCK_Y;
    const unsigned originZ = blockIdx.z * BLOCK_Z;

    for (int linear = tid; linear < TILE_SIZE; linear += BLOCK_THREADS) {
        const int localX = linear % TILE_X;
        const int yz = linear / TILE_X;
        const int localY = yz % TILE_Y;
        const int localZ = yz / TILE_Y;
        const unsigned x = clampCoordinate(
            static_cast<long long>(originX) + localX - 1, nx);
        const unsigned y = clampCoordinate(
            static_cast<long long>(originY) + localY - 1, ny);
        const unsigned z = clampCoordinate(
            static_cast<long long>(originZ) + localZ - 1, nz);
        tile[linear] = mu[static_cast<size_t>(z) * plane
                          + static_cast<size_t>(y) * nx + x];
    }
    __syncthreads();

    const unsigned x = originX + threadIdx.x;
    const unsigned y = originY + threadIdx.y;
    const unsigned z = originZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int centerIndex = tileIndex(static_cast<int>(threadIdx.x) + 1,
                                      static_cast<int>(threadIdx.y) + 1,
                                      static_cast<int>(threadIdx.z) + 1);
    const double center = tile[centerIndex];
    constexpr int tilePlane = TILE_X * TILE_Y;
    const double laplacianMu =
        (tile[centerIndex + 1] + tile[centerIndex - 1] - 2.0 * center) +
        (tile[centerIndex + TILE_X] + tile[centerIndex - TILE_X]
         - 2.0 * center) +
        (tile[centerIndex + tilePlane] + tile[centerIndex - tilePlane]
         - 2.0 * center);

    const size_t globalIndex = static_cast<size_t>(z) * plane
                             + static_cast<size_t>(y) * nx + x;
    output[globalIndex] = input[globalIndex] + 0.01 * laplacianMu;
}

void initializeConcentration(std::vector<double>& concentration) {
    const size_t volume = concentration.size();
    for (size_t linearId = 0; linearId < volume; ++linearId) {
        const double pseudo = ((linearId + 1) * size_t{1299709} % volume)
                            / static_cast<double>(volume);
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

bool checkedMultiply(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
            return EXIT_SUCCESS;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        std::fprintf(stderr,
                     "Grid dimensions must be positive and iterations non-negative.\n");
        return EXIT_FAILURE;
    }
    if (nx > std::numeric_limits<unsigned>::max()
        || ny > std::numeric_limits<unsigned>::max()
        || nz > std::numeric_limits<unsigned>::max()) {
        std::fprintf(stderr, "Requested grid dimensions are too large.\n");
        return EXIT_FAILURE;
    }

    size_t plane = 0;
    size_t gridSize = 0;
    if (!checkedMultiply(nx, ny, plane) || !checkedMultiply(plane, nz, gridSize)
        || gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Requested grid is too large.\n");
        return EXIT_FAILURE;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> concentration(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration);

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    const size_t blocksX = (nx + BLOCK_X - 1) / BLOCK_X;
    const size_t blocksY = (ny + BLOCK_Y - 1) / BLOCK_Y;
    const size_t blocksZ = (nz + BLOCK_Z - 1) / BLOCK_Z;
    if (blocksX > static_cast<size_t>(deviceProperties.maxGridSize[0])
        || blocksY > static_cast<size_t>(deviceProperties.maxGridSize[1])
        || blocksZ > static_cast<size_t>(deviceProperties.maxGridSize[2])) {
        std::fprintf(stderr, "Requested grid exceeds this GPU's launch limits.\n");
        return EXIT_FAILURE;
    }

    const unsigned deviceNx = static_cast<unsigned>(nx);
    const unsigned deviceNy = static_cast<unsigned>(ny);
    const unsigned deviceNz = static_cast<unsigned>(nz);
    const size_t bytes = gridSize * sizeof(double);
    double* deviceCurrent = nullptr;
    double* deviceNext = nullptr;
    double* deviceMu = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCurrent), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceNext), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMu), bytes));
    CUDA_CHECK(cudaMemcpy(deviceCurrent, concentration.data(), bytes,
                          cudaMemcpyHostToDevice));

    const dim3 threads(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 blocks(static_cast<unsigned>(blocksX),
                      static_cast<unsigned>(blocksY),
                      static_cast<unsigned>(blocksZ));
    cudaStream_t computeStream{};
    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    cudaGraph_t simulationGraph{};
    cudaGraphExec_t simulationGraphExecutable{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // Capture the complete dependency chain so all iteration data stays on the
    // GPU and short grids do not become dominated by host launch latency.
    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(computeStream,
                                          cudaStreamCaptureModeThreadLocal));
        for (int step = 0; step < iterations; ++step) {
            computeChemicalPotential<<<blocks, threads, 0, computeStream>>>(
                deviceMu, deviceCurrent, deviceNx, deviceNy, deviceNz, plane);
            updateConcentration<<<blocks, threads, 0, computeStream>>>(
                deviceNext, deviceCurrent, deviceMu,
                deviceNx, deviceNy, deviceNz, plane);
            std::swap(deviceCurrent, deviceNext);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(computeStream, &simulationGraph));
        CUDA_CHECK(cudaGraphInstantiate(&simulationGraphExecutable,
                                        simulationGraph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(simulationGraphExecutable, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(startEvent, computeStream));
    if (iterations > 0) {
        CUDA_CHECK(cudaGraphLaunch(simulationGraphExecutable, computeStream));
    }
    CUDA_CHECK(cudaEventRecord(stopEvent, computeStream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaMemcpy(concentration.data(), deviceCurrent, bytes,
                          cudaMemcpyDeviceToHost));

    std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    const double cellUpdates = static_cast<double>(gridSize)
                             * static_cast<double>(iterations);
    const double mcups = cellUpdates
                       / (static_cast<double>(elapsedMilliseconds) * 1000.0);
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (iterations > 0) {
        CUDA_CHECK(cudaGraphExecDestroy(simulationGraphExecutable));
        CUDA_CHECK(cudaGraphDestroy(simulationGraph));
    }
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(deviceCurrent));
    CUDA_CHECK(cudaFree(deviceNext));
    CUDA_CHECK(cudaFree(deviceMu));

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
