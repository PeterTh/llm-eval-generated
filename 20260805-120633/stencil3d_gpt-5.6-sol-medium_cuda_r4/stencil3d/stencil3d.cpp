#include <algorithm>
#include <chrono>
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

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kZChunk = 16;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_result = (expression);                      \
        if (cuda_check_result != cudaSuccess) {                                  \
            cudaFailure(cuda_check_result, #expression, __FILE__, __LINE__);     \
        }                                                                        \
    } while (false)

__global__ void initializeGrids(Real* const grid1, Real* const grid2,
                                const size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; i < count; i += stride) {
        const Real value = static_cast<Real>(i % 19);
        grid1[i] = value;
        grid2[i] = value;
    }
}

// Each block advances through a short z slab.  The current z value and its two
// z-neighbours remain in registers, while the x/y neighbourhood is shared by
// the whole block.  Compared with a conventional 3-D block this removes most
// redundant global loads without sacrificing the number of resident blocks.
__global__ __launch_bounds__(kBlockX * kBlockY)
void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                   const int nx, const int ny, const int nz,
                   const size_t plane) {
    __shared__ Real tile[kBlockY + 2][kBlockX + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = 1 + static_cast<int>(blockIdx.x) * kBlockX + tx;
    const int y = 1 + static_cast<int>(blockIdx.y) * kBlockY + ty;
    const int zFirst = 1 + static_cast<int>(blockIdx.z) * kZChunk;
    const bool active = x < nx - 1 && y < ny - 1;

    const size_t xy = static_cast<size_t>(y) * static_cast<size_t>(nx) +
                      static_cast<size_t>(x);
    const size_t first = static_cast<size_t>(zFirst) * plane + xy;

    Real below = 0.0;
    Real center = 0.0;
    Real above = 0.0;
    if (active) {
        below = input[first - plane];
        center = input[first];
        above = input[first + plane];
    }

#pragma unroll
    for (int dz = 0; dz < kZChunk; ++dz) {
        const int z = zFirst + dz;
        if (z >= nz - 1) {
            break;
        }

        if (active) {
            tile[ty + 1][tx + 1] = center;

            if (tx == 0) {
                tile[ty + 1][0] = input[first + static_cast<size_t>(dz) * plane - 1];
            }
            if (tx == kBlockX - 1 || x == nx - 2) {
                tile[ty + 1][tx + 2] =
                    input[first + static_cast<size_t>(dz) * plane + 1];
            }
            if (ty == 0) {
                tile[0][tx + 1] =
                    input[first + static_cast<size_t>(dz) * plane - nx];
            }
            if (ty == kBlockY - 1 || y == ny - 2) {
                tile[ty + 2][tx + 1] =
                    input[first + static_cast<size_t>(dz) * plane + nx];
            }
        }
        __syncthreads();

        if (active) {
            // Preserve the source program's operation order for equivalent
            // double-precision numerical results.
            output[first + static_cast<size_t>(dz) * plane] =
                (center + tile[ty + 1][tx] + tile[ty + 1][tx + 2] +
                 tile[ty][tx + 1] + tile[ty + 2][tx + 1] + below + above) /
                7.0;
        }
        __syncthreads();

        below = center;
        center = above;
        if (active && dz + 1 < kZChunk && z + 1 < nz - 1) {
            above = input[first + static_cast<size_t>(dz + 2) * plane];
        }
    }
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

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto limits = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *limits.first, *limits.second);
    if (*limits.second > 1e6 || *limits.first < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

bool parsePositiveSize(const char* text, size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (text == end || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    size_t nxArg = 128;
    size_t nyArg = 0;
    size_t nzArg = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        size_t* destination = nullptr;
        if (std::strcmp(argv[i], "-x") == 0) destination = &nxArg;
        else if (std::strcmp(argv[i], "-y") == 0) destination = &nyArg;
        else if (std::strcmp(argv[i], "-z") == 0) destination = &nzArg;

        if (destination != nullptr) {
            if (++i >= argc || !parsePositiveSize(argv[i], *destination)) {
                std::fprintf(stderr, "Invalid grid dimension\n");
                return EXIT_FAILURE;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const long parsed = std::strtol(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || parsed < 0 ||
                parsed > std::numeric_limits<int>::max()) {
                std::fprintf(stderr, "Invalid iteration count\n");
                return EXIT_FAILURE;
            }
            iterations = static_cast<int>(parsed);
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

    if (nyArg == 0) nyArg = nxArg;
    if (nzArg == 0) nzArg = nxArg;
    if (nxArg < 3 || nyArg < 3 || nzArg < 3) {
        std::fprintf(stderr, "All grid dimensions must be at least 3\n");
        return EXIT_FAILURE;
    }

    const int nx = static_cast<int>(nxArg);
    const int ny = static_cast<int>(nyArg);
    const int nz = static_cast<int>(nzArg);
    const size_t plane = nxArg * nyArg;
    if (nxArg != 0 && plane / nxArg != nyArg ||
        nzArg > std::numeric_limits<size_t>::max() / plane) {
        std::fprintf(stderr, "Grid size is too large\n");
        return EXIT_FAILURE;
    }
    const size_t gridSize = plane * nzArg;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr, "Grid allocation is too large\n");
        return EXIT_FAILURE;
    }
    const size_t bytes = gridSize * sizeof(Real);

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %d x %d x %d\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    std::printf("Initializing grid...\n");

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, bytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, bytes));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int initThreads = 256;
    const size_t requiredBlocks = (gridSize + initThreads - 1) / initThreads;
    const int initBlocks = static_cast<int>(std::min<size_t>(
        requiredBlocks, static_cast<size_t>(properties.multiProcessorCount) * 8));
    initializeGrids<<<initBlocks, initThreads>>>(deviceGrid1, deviceGrid2, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((nx - 2 + kBlockX - 1) / kBlockX,
                    (ny - 2 + kBlockY - 1) / kBlockY,
                    (nz - 2 + kZChunk - 1) / kZChunk);
    CUDA_CHECK(cudaFuncSetCacheConfig(stencilKernel, cudaFuncCachePreferShared));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        Real* source = deviceGrid1;
        Real* destination = deviceGrid2;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stencilKernel<<<grid, block, 0, stream>>>(source, destination, nx, ny,
                                                      nz, plane);
            std::swap(source, destination);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    std::printf("Running stencil computation...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    if (iterations > 0) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::printf("Computation time: %lld ms\n", static_cast<long long>(milliseconds));
    const double cellUpdates = static_cast<double>(nx - 2) * (ny - 2) *
                               (nz - 2) * iterations;
    const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    int returnCode = EXIT_SUCCESS;
    if (printResults || validate) {
        std::vector<Real> finalGrid(gridSize);
        const Real* finalDeviceGrid =
            (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDeviceGrid, bytes,
                              cudaMemcpyDeviceToHost));
        if (printResults) {
            print_results(finalGrid, "Grid");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(finalGrid)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                returnCode = EXIT_FAILURE;
            }
        }
    }

    if (graphExec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceGrid2));
    CUDA_CHECK(cudaFree(deviceGrid1));
    return returnCode;
}
