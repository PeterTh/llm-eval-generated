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

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        const cudaError_t cuda_status_ = (call);                                  \
        if (cuda_status_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(cuda_status_));                       \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                        \
    } while (false)

// A wide X dimension gives fully coalesced global accesses.  The complete tile,
// including its one-cell halo, is staged once and reused for the seven-point
// stencil.  256 threads/block also provides ample occupancy on current GPUs.
constexpr int TILE_X = 32;
constexpr int TILE_Y = 4;
constexpr int TILE_Z = 2;
constexpr int SHARED_X = TILE_X + 2;
constexpr int SHARED_Y = TILE_Y + 2;
constexpr int SHARED_Z = TILE_Z + 2;
constexpr int TILE_ELEMENTS = SHARED_X * SHARED_Y * SHARED_Z;

__device__ __forceinline__ size_t index3(size_t x, size_t y, size_t z,
                                         size_t nx, size_t ny) {
    return (z * ny + y) * nx + x;
}

__device__ __forceinline__ void loadTile(const double* __restrict__ input,
                                         double* tile, size_t nx, size_t ny,
                                         size_t nz) {
    const int thread = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x +
                       threadIdx.x;
    constexpr int THREADS = TILE_X * TILE_Y * TILE_Z;
    const long long ox = static_cast<long long>(blockIdx.x) * TILE_X;
    const long long oy = static_cast<long long>(blockIdx.y) * TILE_Y;
    const long long oz = static_cast<long long>(blockIdx.z) * TILE_Z;

    for (int local = thread; local < TILE_ELEMENTS; local += THREADS) {
        const int lx = local % SHARED_X;
        const int yz = local / SHARED_X;
        const int ly = yz % SHARED_Y;
        const int lz = yz / SHARED_Y;
        const size_t gx = static_cast<size_t>(max(0LL, min(ox + lx - 1,
                                              static_cast<long long>(nx) - 1)));
        const size_t gy = static_cast<size_t>(max(0LL, min(oy + ly - 1,
                                              static_cast<long long>(ny) - 1)));
        const size_t gz = static_cast<size_t>(max(0LL, min(oz + lz - 1,
                                              static_cast<long long>(nz) - 1)));
        tile[local] = input[index3(gx, gy, gz, nx, ny)];
    }
}

__device__ __forceinline__ double tileLaplacian(const double* tile, int sx,
                                                int sy, int sz) {
    const int center = (sz * SHARED_Y + sy) * SHARED_X + sx;
    return (tile[center + 1] + tile[center - 1] - 2.0 * tile[center]) +
           (tile[center + SHARED_X] + tile[center - SHARED_X] -
            2.0 * tile[center]) +
           (tile[center + SHARED_X * SHARED_Y] +
            tile[center - SHARED_X * SHARED_Y] - 2.0 * tile[center]);
}

__global__ void initializeKernel(double* concentration, size_t volume) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; i < volume; i += stride) {
        const double pseudo = ((i + 1) * static_cast<size_t>(1299709) % volume) /
                              static_cast<double>(volume);
        concentration[i] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                        double* __restrict__ mu, size_t nx,
                                        size_t ny, size_t nz) {
    __shared__ double tile[TILE_ELEMENTS];
    loadTile(c, tile, nx, ny, nz);
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * TILE_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * TILE_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * TILE_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const int sx = threadIdx.x + 1;
    const int sy = threadIdx.y + 1;
    const int sz = threadIdx.z + 1;
    const size_t i = index3(x, y, z, nx, ny);
    const double cv = tile[(sz * SHARED_Y + sy) * SHARED_X + sx];
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = 2.0 / 9.0;
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                   2.0 * cv * e_AB) +
            3.0 * cv + cv * cv * cv -
            0.5 * tileLaplacian(tile, sx, sy, sz);
}

__global__ void updateKernel(double* __restrict__ cnew,
                             const double* __restrict__ cold,
                             const double* __restrict__ mu, size_t nx,
                             size_t ny, size_t nz) {
    __shared__ double tile[TILE_ELEMENTS];
    loadTile(mu, tile, nx, ny, nz);
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * TILE_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * TILE_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * TILE_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t i = index3(x, y, z, nx, ny);
    cnew[i] = cold[i] + 0.01 * tileLaplacian(
        tile, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1);
}

bool validateResult(const std::vector<double>& c) {
    for (const double val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto bounds = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *bounds.first,
                *bounds.second);
    if (*bounds.second > 10.0 || *bounds.first < -10.0) {
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

int main(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
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
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions and iterations must be valid non-negative values\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Requested grid is too large\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const dim3 block(TILE_X, TILE_Y, TILE_Z);
    const size_t gx = (nx + TILE_X - 1) / TILE_X;
    const size_t gy = (ny + TILE_Y - 1) / TILE_Y;
    const size_t gz = (nz + TILE_Z - 1) / TILE_Z;
    if (gx > static_cast<size_t>(properties.maxGridSize[0]) ||
        gy > static_cast<size_t>(properties.maxGridSize[1]) ||
        gz > static_cast<size_t>(properties.maxGridSize[2])) {
        std::fprintf(stderr, "Grid dimensions exceed this CUDA device's launch limits\n");
        return 1;
    }
    const dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy),
                    static_cast<unsigned>(gz));

    double *dCold = nullptr, *dNew = nullptr, *dMu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dCold, bytes));
    CUDA_CHECK(cudaMalloc(&dNew, bytes));
    CUDA_CHECK(cudaMalloc(&dMu, bytes));

    std::printf("Initializing concentration field...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>(
        (gridSize + 255) / 256, static_cast<size_t>(properties.multiProcessorCount) * 32));
    initializeKernel<<<initBlocks, 256>>>(dCold, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int t = 0; t < iterations; ++t) {
            chemicalPotentialKernel<<<grid, block, 0, stream>>>(dCold, dMu, nx, ny, nz);
            updateKernel<<<grid, block, 0, stream>>>(dNew, dCold, dMu, nx, ny, nz);
            std::swap(dCold, dNew);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    std::printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (iterations > 0) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    std::printf("Computation time: %.3f ms\n", elapsedMs);
    const double seconds = static_cast<double>(elapsedMs) * 1.0e-3;
    const double mcups = seconds > 0.0
        ? static_cast<double>(gridSize) * iterations / seconds / 1.0e6
        : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    int result = 0;
    if (printResults || validate) {
        std::vector<double> concentration(gridSize);
        CUDA_CHECK(cudaMemcpy(concentration.data(), dCold, bytes,
                              cudaMemcpyDeviceToHost));
        if (printResults) print_results(concentration, "Concentration");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(concentration);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dMu));
    CUDA_CHECK(cudaFree(dNew));
    CUDA_CHECK(cudaFree(dCold));
    return result;
}
