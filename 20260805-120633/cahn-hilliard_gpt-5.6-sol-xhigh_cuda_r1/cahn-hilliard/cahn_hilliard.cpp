#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

// A block covers four z planes.  Keeping only x and y in the CUDA block
// gives each thread useful work in all four planes while retaining enough
// resident blocks to hide global-memory latency.
constexpr int kBlockX = 16;
constexpr int kBlockY = 8;
constexpr int kTileZ = 4;
constexpr int kThreads = kBlockX * kBlockY;
constexpr int kSharedX = kBlockX + 2;
constexpr int kSharedY = kBlockY + 2;
constexpr int kSharedZ = kTileZ + 2;
constexpr int kCoreCells = kBlockX * kBlockY * kTileZ;

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

__device__ __forceinline__ constexpr int sharedIndex(const int x, const int y,
                                                      const int z) {
    return (z * kSharedY + y) * kSharedX + x;
}

__device__ __forceinline__ size_t boundedIndex(const size_t index,
                                                const size_t extent) {
    return index < extent ? index : extent - 1;
}

// Load one 16x8x4 tile plus the six face halos required by a 7-point
// Laplacian.  Edges and corners are deliberately not loaded: the stencil
// never reads them, saving about 11% of the tile's global loads.
__device__ __forceinline__ void loadTile(const double* __restrict__ input,
                                         double* __restrict__ tile,
                                         const size_t nx, const size_t ny,
                                         const size_t nz) {
    const int tid = static_cast<int>(threadIdx.y) * kBlockX +
                    static_cast<int>(threadIdx.x);
    const size_t baseX = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t baseY = static_cast<size_t>(blockIdx.y) * kBlockY;
    const size_t baseZ = static_cast<size_t>(blockIdx.z) * kTileZ;
    const size_t plane = nx * ny;

    // The 512 interior values are loaded by 128 threads in four fully
    // coalesced passes.
#pragma unroll
    for (int local = tid; local < kCoreCells; local += kThreads) {
        const int lx = local % kBlockX;
        const int ly = (local / kBlockX) % kBlockY;
        const int lz = local / (kBlockX * kBlockY);
        const size_t gx = boundedIndex(baseX + static_cast<size_t>(lx), nx);
        const size_t gy = boundedIndex(baseY + static_cast<size_t>(ly), ny);
        const size_t gz = boundedIndex(baseZ + static_cast<size_t>(lz), nz);
        tile[sharedIndex(lx + 1, ly + 1, lz + 1)] =
            input[gz * plane + gy * nx + gx];
    }

    // X faces.
    if (tid < kBlockY * kTileZ) {
        const int ly = tid % kBlockY;
        const int lz = tid / kBlockY;
        const size_t gy = boundedIndex(baseY + static_cast<size_t>(ly), ny);
        const size_t gz = boundedIndex(baseZ + static_cast<size_t>(lz), nz);
        const size_t leftX = baseX == 0 ? 0 : baseX - 1;
        const size_t rightX = boundedIndex(baseX + kBlockX, nx);
        const size_t row = gz * plane + gy * nx;
        tile[sharedIndex(0, ly + 1, lz + 1)] = input[row + leftX];
        tile[sharedIndex(kBlockX + 1, ly + 1, lz + 1)] = input[row + rightX];
    }

    // Y faces.
    if (tid < kBlockX * kTileZ) {
        const int lx = tid % kBlockX;
        const int lz = tid / kBlockX;
        const size_t gx = boundedIndex(baseX + static_cast<size_t>(lx), nx);
        const size_t gz = boundedIndex(baseZ + static_cast<size_t>(lz), nz);
        const size_t lowerY = baseY == 0 ? 0 : baseY - 1;
        const size_t upperY = boundedIndex(baseY + kBlockY, ny);
        tile[sharedIndex(lx + 1, 0, lz + 1)] =
            input[gz * plane + lowerY * nx + gx];
        tile[sharedIndex(lx + 1, kBlockY + 1, lz + 1)] =
            input[gz * plane + upperY * nx + gx];
    }

    // Z faces.
    if (tid < kBlockX * kBlockY) {
        const int lx = tid % kBlockX;
        const int ly = tid / kBlockX;
        const size_t gx = boundedIndex(baseX + static_cast<size_t>(lx), nx);
        const size_t gy = boundedIndex(baseY + static_cast<size_t>(ly), ny);
        const size_t lowerZ = baseZ == 0 ? 0 : baseZ - 1;
        const size_t upperZ = boundedIndex(baseZ + kTileZ, nz);
        tile[sharedIndex(lx + 1, ly + 1, 0)] =
            input[lowerZ * plane + gy * nx + gx];
        tile[sharedIndex(lx + 1, ly + 1, kTileZ + 1)] =
            input[upperZ * plane + gy * nx + gx];
    }

    __syncthreads();
}

__global__ __launch_bounds__(kThreads)
void chemicalPotentialKernel(const double* __restrict__ concentration,
                             double* __restrict__ chemicalPotential,
                             const size_t nx, const size_t ny,
                             const size_t nz) {
    __shared__ double tile[kSharedX * kSharedY * kSharedZ];
    loadTile(concentration, tile, nx, ny, nz);

    const int lx = static_cast<int>(threadIdx.x);
    const int ly = static_cast<int>(threadIdx.y);
    const size_t gx = static_cast<size_t>(blockIdx.x) * kBlockX + lx;
    const size_t gy = static_cast<size_t>(blockIdx.y) * kBlockY + ly;
    const size_t baseZ = static_cast<size_t>(blockIdx.z) * kTileZ;
    const size_t plane = nx * ny;

    if (gx >= nx || gy >= ny) {
        return;
    }

    // With eAA=eBB=-2/9 and eAB=2/9, the bulk free-energy derivative
    // reduces algebraically to c^3-c.  Using the reduced form removes six
    // double-precision operations from this hot kernel.
    constexpr double gamma = 0.5;

#pragma unroll
    for (int lz = 0; lz < kTileZ; ++lz) {
        const size_t gz = baseZ + static_cast<size_t>(lz);
        if (gz < nz) {
            const int center = sharedIndex(lx + 1, ly + 1, lz + 1);
            const double cv = tile[center];
            const double cxx = tile[center + 1] + tile[center - 1] - 2.0 * cv;
            const double cyy = tile[center + kSharedX] +
                               tile[center - kSharedX] - 2.0 * cv;
            const double czz = tile[center + kSharedX * kSharedY] +
                               tile[center - kSharedX * kSharedY] - 2.0 * cv;
            const double laplacian = cxx + cyy + czz;
            chemicalPotential[gz * plane + gy * nx + gx] =
                cv * cv * cv - cv - gamma * laplacian;
        }
    }
}

__global__ __launch_bounds__(kThreads)
void updateKernel(double* __restrict__ nextConcentration,
                  const double* __restrict__ concentration,
                  const double* __restrict__ chemicalPotential,
                  const size_t nx, const size_t ny, const size_t nz) {
    __shared__ double tile[kSharedX * kSharedY * kSharedZ];
    loadTile(chemicalPotential, tile, nx, ny, nz);

    const int lx = static_cast<int>(threadIdx.x);
    const int ly = static_cast<int>(threadIdx.y);
    const size_t gx = static_cast<size_t>(blockIdx.x) * kBlockX + lx;
    const size_t gy = static_cast<size_t>(blockIdx.y) * kBlockY + ly;
    const size_t baseZ = static_cast<size_t>(blockIdx.z) * kTileZ;
    const size_t plane = nx * ny;

    if (gx >= nx || gy >= ny) {
        return;
    }

    constexpr double dt = 0.01;
    constexpr double diffusion = 1.0;

#pragma unroll
    for (int lz = 0; lz < kTileZ; ++lz) {
        const size_t gz = baseZ + static_cast<size_t>(lz);
        if (gz < nz) {
            const int center = sharedIndex(lx + 1, ly + 1, lz + 1);
            const double value = tile[center];
            const double mxx = tile[center + 1] + tile[center - 1] - 2.0 * value;
            const double myy = tile[center + kSharedX] +
                               tile[center - kSharedX] - 2.0 * value;
            const double mzz = tile[center + kSharedX * kSharedY] +
                               tile[center - kSharedX * kSharedY] - 2.0 * value;
            const double laplacian = mxx + myy + mzz;
            const size_t index = gz * plane + gy * nx + gx;
            nextConcentration[index] =
                concentration[index] + dt * diffusion * laplacian;
        }
    }
}

void initializeConcentration(std::vector<double>& concentration,
                             const size_t nx, const size_t ny,
                             const size_t nz) {
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = z * (nx * ny) + y * nx + x;
                const double pseudo =
                    (((linearId + 1) * 1299709) % volume) /
                    static_cast<double>(volume);
                concentration[linearId] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& concentration) {
    for (const double value : concentration) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minValue = concentration[0];
    double maxValue = concentration[0];
    for (const double value : concentration) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 10.0 || minValue < -10.0) {
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

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    std::vector<double> concentration(gridSize);

    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration, nx, ny, nz);

    // Device allocation and graph construction are setup costs, matching the
    // original benchmark's timed region, which contained time steps only.
    double* deviceCurrent = nullptr;
    double* deviceNext = nullptr;
    double* deviceMu = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCurrent), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceNext), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMu), bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(deviceCurrent, concentration.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const dim3 block(kBlockX, kBlockY, 1);
    const dim3 grid(static_cast<unsigned int>((nx + kBlockX - 1) / kBlockX),
                    static_cast<unsigned int>((ny + kBlockY - 1) / kBlockY),
                    static_cast<unsigned int>((nz + kTileZ - 1) / kTileZ));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    double* finalDeviceResult = deviceCurrent;
    double* nextDeviceResult = deviceNext;
    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int step = 0; step < iterations; ++step) {
            chemicalPotentialKernel<<<grid, block, 0, stream>>>(
                finalDeviceResult, deviceMu, nx, ny, nz);
            updateKernel<<<grid, block, 0, stream>>>(
                nextDeviceResult, finalDeviceResult, deviceMu, nx, ny, nz);
            std::swap(finalDeviceResult, nextDeviceResult);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (iterations > 0) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
    const long wholeMilliseconds = static_cast<long>(elapsedMilliseconds);
    std::printf("Computation time: %ld ms\n", wholeMilliseconds);

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double elapsedSeconds = static_cast<double>(elapsedMilliseconds) / 1000.0;
    const double mcups = cellUpdates / elapsedSeconds / 1.0e6;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    CUDA_CHECK(cudaMemcpyAsync(concentration.data(), finalDeviceResult, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    if (graphExec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(deviceMu));
    CUDA_CHECK(cudaFree(deviceNext));
    CUDA_CHECK(cudaFree(deviceCurrent));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (printResults) {
        print_results(concentration, "Concentration");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(concentration)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
