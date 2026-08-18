#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 4;
constexpr int BLOCK_Z = 2;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t cuda_status_ = (call);                                \
        if (cuda_status_ != cudaSuccess) {                                      \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(cuda_status_));            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

// Load one block and its six one-cell halos.  Clamping is done as data is
// loaded, so the stencil itself has no boundary branches.
__device__ __forceinline__ void loadTile(
    const double* __restrict__ field,
    double (&tile)[BLOCK_Z + 2][BLOCK_Y + 2][BLOCK_X + 2],
    const int nx, const int ny, const int nz) {
    const int lx = static_cast<int>(threadIdx.x);
    const int ly = static_cast<int>(threadIdx.y);
    const int lz = static_cast<int>(threadIdx.z);
    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + lx;
    const int y = static_cast<int>(blockIdx.y) * BLOCK_Y + ly;
    const int z = static_cast<int>(blockIdx.z) * BLOCK_Z + lz;

    const int cx = min(x, nx - 1);
    const int cy = min(y, ny - 1);
    const int cz = min(z, nz - 1);
    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t center = static_cast<size_t>(cz) * plane +
                          static_cast<size_t>(cy) * nx + cx;
    tile[lz + 1][ly + 1][lx + 1] = field[center];

    if (lx == 0) {
        const int hx = max(x - 1, 0);
        tile[lz + 1][ly + 1][0] = field[static_cast<size_t>(cz) * plane +
                                           static_cast<size_t>(cy) * nx + hx];
    }
    if (lx == BLOCK_X - 1) {
        const int hx = min(x + 1, nx - 1);
        tile[lz + 1][ly + 1][BLOCK_X + 1] =
            field[static_cast<size_t>(cz) * plane +
                  static_cast<size_t>(cy) * nx + hx];
    }
    if (ly == 0) {
        const int hy = max(y - 1, 0);
        tile[lz + 1][0][lx + 1] = field[static_cast<size_t>(cz) * plane +
                                             static_cast<size_t>(hy) * nx + cx];
    }
    if (ly == BLOCK_Y - 1) {
        const int hy = min(y + 1, ny - 1);
        tile[lz + 1][BLOCK_Y + 1][lx + 1] =
            field[static_cast<size_t>(cz) * plane +
                  static_cast<size_t>(hy) * nx + cx];
    }
    if (lz == 0) {
        const int hz = max(z - 1, 0);
        tile[0][ly + 1][lx + 1] = field[static_cast<size_t>(hz) * plane +
                                             static_cast<size_t>(cy) * nx + cx];
    }
    if (lz == BLOCK_Z - 1) {
        const int hz = min(z + 1, nz - 1);
        tile[BLOCK_Z + 1][ly + 1][lx + 1] =
            field[static_cast<size_t>(hz) * plane +
                  static_cast<size_t>(cy) * nx + cx];
    }
    __syncthreads();
}

__device__ __forceinline__ double tileLaplacian(
    const double (&tile)[BLOCK_Z + 2][BLOCK_Y + 2][BLOCK_X + 2],
    const double invDx2, const double invDy2, const double invDz2) {
    const int x = static_cast<int>(threadIdx.x) + 1;
    const int y = static_cast<int>(threadIdx.y) + 1;
    const int z = static_cast<int>(threadIdx.z) + 1;
    const double center = tile[z][y][x];
    const double cxx = (tile[z][y][x + 1] + tile[z][y][x - 1] -
                        2.0 * center) * invDx2;
    const double cyy = (tile[z][y + 1][x] + tile[z][y - 1][x] -
                        2.0 * center) * invDy2;
    const double czz = (tile[z + 1][y][x] + tile[z - 1][y][x] -
                        2.0 * center) * invDz2;
    return cxx + cyy + czz;
}

__global__ void chemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const int nx, const int ny, const int nz,
    const double invDx2, const double invDy2, const double invDz2,
    const double gamma, const double eAA, const double eBB,
    const double eAB) {
    __shared__ double tile[BLOCK_Z + 2][BLOCK_Y + 2][BLOCK_X + 2];
    loadTile(c, tile, nx, ny, nz);

    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const int z = static_cast<int>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t index = (static_cast<size_t>(z) * ny + y) * nx + x;
    const double cv = c[index];
    mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB -
                       2.0 * cv * eAB) +
                3.0 * cv + cv * cv * cv -
                gamma * tileLaplacian(tile, invDx2, invDy2, invDz2);
}

__global__ void updateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu, const int nx, const int ny, const int nz,
    const double dtD, const double invDx2, const double invDy2,
    const double invDz2) {
    __shared__ double tile[BLOCK_Z + 2][BLOCK_Y + 2][BLOCK_X + 2];
    loadTile(mu, tile, nx, ny, nz);

    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const int z = static_cast<int>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t index = (static_cast<size_t>(z) * ny + y) * nx + x;
    cnew[index] = cold[index] +
                  dtD * tileLaplacian(tile, invDx2, invDy2, invDz2);
}

void initializeConcentration(std::vector<double>& c, const size_t volume) {
    for (size_t i = 0; i < volume; ++i) {
        const double pseudo = (((i + 1) * 1299709) % volume) /
                              static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c) {
    for (const double value : c) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto limits = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n",
                *limits.first, *limits.second);
    if (*limits.second > 10.0 || *limits.first < -10.0) {
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
        nx > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nz > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions and iteration count must be valid non-negative values.\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Grid is too large.\n");
        return 1;
    }
    const size_t bytes = gridSize * sizeof(double);

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;

    std::vector<double> result(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(result, gridSize);

    double* dCold = nullptr;
    double* dNew = nullptr;
    double* dMu = nullptr;
    CUDA_CHECK(cudaMalloc(&dCold, bytes));
    CUDA_CHECK(cudaMalloc(&dNew, bytes));
    CUDA_CHECK(cudaMalloc(&dMu, bytes));
    CUDA_CHECK(cudaMemcpy(dCold, result.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid((nx + BLOCK_X - 1) / BLOCK_X,
                    (ny + BLOCK_Y - 1) / BLOCK_Y,
                    (nz + BLOCK_Z - 1) / BLOCK_Z);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int step = 0; step < iterations; ++step) {
            chemicalPotentialKernel<<<grid, block, 0, stream>>>(
                dCold, dMu, static_cast<int>(nx), static_cast<int>(ny),
                static_cast<int>(nz), 1.0 / (dx * dx), 1.0 / (dy * dy),
                1.0 / (dz * dz), gamma, eAA, eBB, eAB);
            updateKernel<<<grid, block, 0, stream>>>(
                dNew, dCold, dMu, static_cast<int>(nx), static_cast<int>(ny),
                static_cast<int>(nz), dt * diffusion, 1.0 / (dx * dx),
                1.0 / (dy * dy), 1.0 / (dz * dz));
            std::swap(dCold, dNew);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    std::printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    if (iterations > 0) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::printf("Computation time: %ld ms\n", duration.count());

    const double seconds = std::chrono::duration<double>(end - start).count();
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1.0e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(result.data(), dCold, bytes, cudaMemcpyDeviceToHost));
    }
    if (printResults) print_results(result, "Concentration");

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(result);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    if (graphExec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dMu));
    CUDA_CHECK(cudaFree(dNew));
    CUDA_CHECK(cudaFree(dCold));
    return valid ? 0 : 1;
}
