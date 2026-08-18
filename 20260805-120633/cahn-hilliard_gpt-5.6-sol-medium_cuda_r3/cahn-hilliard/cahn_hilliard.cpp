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

namespace {

constexpr unsigned BLOCK_X = 32;
constexpr unsigned BLOCK_Y = 4;
constexpr unsigned BLOCK_Z = 2;
constexpr unsigned SHARED_X = BLOCK_X + 2;
constexpr unsigned SHARED_Y = BLOCK_Y + 2;
constexpr unsigned SHARED_Z = BLOCK_Z + 2;
constexpr unsigned SHARED_SIZE = SHARED_X * SHARED_Y * SHARED_Z;

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                     expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

// All threads cooperatively stage a block and its one-cell halo.  Clamping
// while loading gives exactly the zero-normal-gradient boundary treatment used
// by the original implementation.
__device__ __forceinline__ void loadTile(const double* __restrict__ field,
                                         double* __restrict__ tile,
                                         size_t nx, size_t ny, size_t nz) {
    const unsigned thread = (threadIdx.z * BLOCK_Y + threadIdx.y) * BLOCK_X + threadIdx.x;
    constexpr unsigned threads = BLOCK_X * BLOCK_Y * BLOCK_Z;

    const size_t originX = static_cast<size_t>(blockIdx.x) * BLOCK_X;
    const size_t originY = static_cast<size_t>(blockIdx.y) * BLOCK_Y;
    const size_t originZ = static_cast<size_t>(blockIdx.z) * BLOCK_Z;

    for (unsigned linear = thread; linear < SHARED_SIZE; linear += threads) {
        const unsigned sx = linear % SHARED_X;
        const unsigned remainder = linear / SHARED_X;
        const unsigned sy = remainder % SHARED_Y;
        const unsigned sz = remainder / SHARED_Y;

        size_t x = (sx == 0 && originX == 0) ? 0 : originX + sx - 1;
        size_t y = (sy == 0 && originY == 0) ? 0 : originY + sy - 1;
        size_t z = (sz == 0 && originZ == 0) ? 0 : originZ + sz - 1;
        x = min(x, nx - 1);
        y = min(y, ny - 1);
        z = min(z, nz - 1);
        tile[linear] = field[(z * ny + y) * nx + x];
    }
}

__device__ __forceinline__ double tileLaplacian(const double* __restrict__ tile,
                                                unsigned sx, unsigned sy, unsigned sz) {
    const unsigned center = (sz * SHARED_Y + sy) * SHARED_X + sx;
    const double value = tile[center];
    const double cxx = tile[center + 1] + tile[center - 1] - 2.0 * value;
    const double cyy = tile[center + SHARED_X] + tile[center - SHARED_X] - 2.0 * value;
    constexpr unsigned plane = SHARED_X * SHARED_Y;
    const double czz = tile[center + plane] + tile[center - plane] - 2.0 * value;
    return cxx + cyy + czz;
}

__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void chemicalPotentialKernel(const double* __restrict__ concentration,
                             double* __restrict__ potential,
                             size_t nx, size_t ny, size_t nz) {
    __shared__ double tile[SHARED_SIZE];
    loadTile(concentration, tile, nx, ny, nz);
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t index = (z * ny + y) * nx + x;
    const double cv = concentration[index];
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB =  (2.0 / 9.0);
    potential[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                     + 3.0 * cv + cv * cv * cv
                     - 0.5 * tileLaplacian(tile, threadIdx.x + 1,
                                           threadIdx.y + 1, threadIdx.z + 1);
}

__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void updateKernel(double* __restrict__ next,
                  const double* __restrict__ current,
                  const double* __restrict__ potential,
                  size_t nx, size_t ny, size_t nz) {
    __shared__ double tile[SHARED_SIZE];
    loadTile(potential, tile, nx, ny, nz);
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t index = (z * ny + y) * nx + x;
    next[index] = current[index]
                + 0.01 * tileLaplacian(tile, threadIdx.x + 1,
                                       threadIdx.y + 1, threadIdx.z + 1);
}

void initializeConcentration(std::vector<double>& c, size_t nx, size_t ny, size_t nz) {
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = idx3(x, y, z, nx, ny);
                const size_t linearId = z * (nx * ny) + y * nx + x;
                const double pseudo = (((linearId + 1) * 1299709) % volume)
                                    / static_cast<double>(volume);
                c[index] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c) {
    for (const double value : c) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto extrema = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *extrema.first, *extrema.second);
    if (*extrema.second > 10.0 || *extrema.first < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
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
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions and iteration count must be valid and non-negative.\n");
        return 1;
    }

    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Requested grid is too large.\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> concentration(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration, nx, ny, nz);

    double* deviceCurrent = nullptr;
    double* deviceNext = nullptr;
    double* devicePotential = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&deviceCurrent, bytes));
    CUDA_CHECK(cudaMalloc(&deviceNext, bytes));
    CUDA_CHECK(cudaMalloc(&devicePotential, bytes));
    CUDA_CHECK(cudaMemcpy(deviceCurrent, concentration.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned>((nz + BLOCK_Z - 1) / BLOCK_Z));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Materialize the CUDA module and warm both kernels before measurement.
    // The output is intentionally discarded, so deviceCurrent is unchanged.
    chemicalPotentialKernel<<<grid, block, 0, stream>>>(deviceCurrent, devicePotential, nx, ny, nz);
    updateKernel<<<grid, block, 0, stream>>>(deviceNext, deviceCurrent, devicePotential, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // A graph preserves the strict potential/update ordering while amortizing
    // host launch overhead across the complete time integration.
    cudaGraph_t graphHandle = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int step = 0; step < iterations; ++step) {
            chemicalPotentialKernel<<<grid, block, 0, stream>>>(deviceCurrent, devicePotential, nx, ny, nz);
            updateKernel<<<grid, block, 0, stream>>>(deviceNext, deviceCurrent, devicePotential, nx, ny, nz);
            std::swap(deviceCurrent, deviceNext);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graphHandle));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graphHandle, 0));
    }

    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (iterations > 0) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    std::printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMs));

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = elapsedMs > 0.0F ? cellUpdates / (static_cast<double>(elapsedMs) * 1000.0) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(concentration.data(), deviceCurrent, bytes, cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        print_results(concentration, "Concentration");
    }

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(concentration);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (graphExec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graphHandle != nullptr) CUDA_CHECK(cudaGraphDestroy(graphHandle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(devicePotential));
    CUDA_CHECK(cudaFree(deviceNext));
    CUDA_CHECK(cudaFree(deviceCurrent));
    return valid ? 0 : 1;
}
