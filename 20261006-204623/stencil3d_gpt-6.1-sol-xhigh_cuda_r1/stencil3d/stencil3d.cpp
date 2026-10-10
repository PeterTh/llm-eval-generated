#include <algorithm>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

size_t parseSize(const char* text, const char* option) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || errno == ERANGE ||
        value > std::numeric_limits<size_t>::max()) {
        fprintf(stderr, "Invalid value for %s: %s\n", option, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(value);
}

// Both buffers receive identical fixed boundaries. No boundary copy is needed
// during the iterations, because only interior cells are subsequently written.
__global__ void initializeGrids(Real* first, Real* second, const size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

// Since seven is an odd integer, a normal binary quotient cannot be halfway
// between doubles. The exact fused residual and one correction therefore give
// the same rounding as division for these sums (zero, or normal values <= 126).
__device__ __forceinline__ Real averageSeven(const Real sum) {
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = __dmul_rn(sum, reciprocal);
    const Real residual = __fma_rn(-7.0, quotient, sum);
    return __fma_rn(residual, reciprocal, quotient);
}

// For narrow XY planes, distribute threads across Z as well so that a warp
// contains many interior points rather than mostly boundary lanes.
template <typename Index>
__global__ void stencilFlat(const Real* __restrict__ input, Real* __restrict__ output,
                            const Index nx, const Index plane, const Index count) {
    for (size_t position = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < count; position += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const Index index = static_cast<Index>(position);
        if (index < plane || index >= count - plane) continue;
        const Index xy = index % plane;
        if (xy < nx || xy >= plane - nx) continue;
        const Index x = xy % nx;
        if (x == 0 || x == nx - 1) continue;
        const Real sum = input[index] + input[index - 1] + input[index + 1] +
                         input[index - nx] + input[index + nx] +
                         input[index - plane] + input[index + plane];
        output[index] = averageSeven(sum);
    }
}

// Adjacent lanes traverse adjacent X/Y cells. Each thread advances through a
// short Z column, retaining the previous two plane values in registers.
// Independent Z tiles expose parallelism even for grids with small XY planes.
template <int Depth, typename Index>
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 const Index nx, const Index plane, const Index nz) {
    const Index xy = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (xy < nx || xy >= plane - nx) return;
    const Index x = xy % nx;
    if (x == 0 || x == nx - 1) return;

    // Grid-striding in Z also supports very long, narrow volumes.
    for (Index firstZ = static_cast<Index>(blockIdx.y) * Depth + 1;
         firstZ < nz - 1; firstZ += static_cast<Index>(gridDim.y) * Depth) {
        const Index endZ = firstZ + Depth < nz - 1 ? firstZ + Depth : nz - 1;
        Index index = firstZ * plane + xy;
        Real bottom = input[index - plane];
        Real center = input[index];
        #pragma unroll 1
        for (Index z = firstZ; z < endZ; ++z, index += plane) {
            const Real top = input[index + plane];
            const Real sum = center + input[index - 1] + input[index + 1] +
                             input[index - nx] + input[index + nx] + bottom + top;
            output[index] = averageSeven(sum);
            bottom = center;
            center = top;
        }
    }
}

template <int Depth, typename Index>
void launchStencil(const Real* input, Real* output, const size_t nx,
                   const size_t ny, const size_t nz, cudaStream_t stream) {
    const size_t plane = nx * ny;
    if (plane < 128) {
        const size_t count = plane * nz;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<size_t>((count + 255) / 256, 65535));
        stencilFlat<Index><<<blocks, 256, 0, stream>>>(input, output,
            static_cast<Index>(nx), static_cast<Index>(plane), static_cast<Index>(count));
        return;
    }
    const unsigned int threads = plane >= 65536 ? 256 : 128;
    const size_t planeBlocks = (plane + threads - 1) / threads;
    const size_t zTiles = (nz - 2 + Depth - 1) / Depth;
    const dim3 blocks(static_cast<unsigned int>(planeBlocks),
                      static_cast<unsigned int>(std::min<size_t>(zTiles, 65535)));
    stencilIteration<Depth, Index><<<blocks, threads, 0, stream>>>(
        input, output, static_cast<Index>(nx), static_cast<Index>(plane), static_cast<Index>(nz));
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = parseSize(argv[++i], "-x");
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = parseSize(argv[++i], "-y");
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = parseSize(argv[++i], "-z");
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            const size_t value = parseSize(argv[++i], "-i");
            if (value > INT_MAX) {
                fprintf(stderr, "Iteration count is too large\n");
                return 1;
            }
            iterations = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz / sizeof(Real)) {
        fprintf(stderr, "Grid dimensions must be positive and fit in memory\n");
        return 1;
    }

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;

    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&grid1, bytes));
    CUDA_CHECK(cudaMalloc(&grid2, bytes));

    printf("Initializing grid...\n");
    const unsigned int initBlocks = static_cast<unsigned int>(
        std::min<size_t>((gridSize + 255) / 256, 65535));
    initializeGrids<<<initBlocks, 256, 0, stream>>>(grid1, grid2, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Small grids need more independent Z tiles to occupy the GPU.
    const bool useColumns = gridSize >= 1024 * 1024;
    // Use cheaper 32-bit index arithmetic when the entire allocation fits;
    // retain a 64-bit kernel for GPUs capable of holding larger grids.
    auto launch = [&](const Real* input, Real* output) {
        if (gridSize <= std::numeric_limits<unsigned int>::max()) {
            if (useColumns) launchStencil<8, unsigned int>(input, output, nx, ny, nz, stream);
            else launchStencil<1, unsigned int>(input, output, nx, ny, nz, stream);
        } else {
            if (useColumns) launchStencil<8, size_t>(input, output, nx, ny, nz, stream);
            else launchStencil<1, size_t>(input, output, nx, ny, nz, stream);
        }
    };

    // Replay a bounded batch of iterations to reduce CPU launch overhead. An
    // even-sized batch always starts and finishes in the same buffer.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    const int batch = std::min(iterations - iterations % 2, 64);
    if (hasInterior && batch > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < batch; i += 2) {
            launch(grid1, grid2);
            launch(grid2, grid1);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    printf("Running stencil computation...\n");
    const auto start = std::chrono::steady_clock::now();
    if (hasInterior) {
        int iter = 0;
        if (executable != nullptr) {
            for (; iter <= iterations - batch; iter += batch) {
                CUDA_CHECK(cudaGraphLaunch(executable, stream));
            }
        }
        for (; iter < iterations; ++iter) {
            launch(iter % 2 == 0 ? grid1 : grid2, iter % 2 == 0 ? grid2 : grid1);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %.3f ms\n", seconds * 1000.0);

    const double cellUpdates = hasInterior ?
        static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations : 0.0;
    const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        CUDA_CHECK(cudaMemcpyAsync(finalGrid.data(), iterations % 2 == 0 ? grid1 : grid2,
                                   bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (executable != nullptr) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(grid2));
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
