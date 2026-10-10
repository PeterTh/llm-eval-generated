#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// Initialize both buffers so that boundaries need no work in subsequent steps.
__global__ void initializeGrid(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        first[i] = second[i] = Real(i % 19);
    }
}

constexpr unsigned tileX = 32;
constexpr unsigned tileY = 8;
constexpr unsigned tileZ = 4;

// A warp traverses contiguous X values. Each thread processes a short Z column,
// retaining the previous and current planes in registers. Independent Z tiles
// keep enough blocks available even for wide, shallow grids.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * tileX + threadIdx.x;
    const size_t y = size_t(blockIdx.y) * tileY + threadIdx.y + 1;
    const size_t zBegin = size_t(blockIdx.z) * tileZ + 1;
    if (x == 0 || x >= nx - 1 || y >= ny - 1) return;

    const size_t plane = nx * ny;
    size_t index = zBegin * plane + y * nx + x;
    Real bottom = input[index - plane];
    Real center = input[index];
    #pragma unroll
    for (unsigned step = 0; step < tileZ; ++step) {
        if (zBegin + step >= nz - 1) break;
        const Real top = input[index + plane];
        // Preserve the original left-to-right addition order and double precision.
        const Real sum = center + input[index - 1] + input[index + 1]
                       + input[index - nx] + input[index + nx] + bottom + top;
        // Correct the reciprocal product using its exact residual. For the
        // bounded, nonnegative stencil values this rounds identically to / 7,
        // without CUDA's general-purpose division (including exceptional cases).
        constexpr Real reciprocal = 1.0 / 7.0;
        const Real quotient = sum * reciprocal;
        const Real residual = __fma_rn(-quotient, 7.0, sum);
        output[index] = __fma_rn(residual, reciprocal, quotient);
        bottom = center;
        center = top;
        index += plane;
    }
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
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1, bytes));
    CUDA_CHECK(cudaMalloc(&grid2, bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize + 255) / 256, 65535));
    initializeGrid<<<initBlocks, 256, 0, stream>>>(grid1, grid2, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t blocksX = (nx + tileX - 1) / tileX;
    const size_t blocksY = hasInterior ? (ny - 2 + tileY - 1) / tileY : 1;
    const size_t blocksZ = hasInterior ? (nz - 2 + tileZ - 1) / tileZ : 1;
    cudaDeviceProp properties;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (hasInterior && (blocksX > size_t(properties.maxGridSize[0]) ||
                        blocksY > size_t(properties.maxGridSize[1]) ||
                        blocksZ > size_t(properties.maxGridSize[2]))) {
        fprintf(stderr, "Grid exceeds CUDA launch dimensions\n");
        return 1;
    }
    const dim3 blocks(static_cast<unsigned>(blocksX), static_cast<unsigned>(blocksY),
                      static_cast<unsigned>(blocksZ));
    const dim3 threads(tileX, tileY);

    // Capture an even number of steps so graph replays always start with grid1.
    // Bounded graph size avoids setup cost growing with the iteration count.
    const int batch = std::min(iterations, 32) & ~1;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    if (hasInterior && batch > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int step = 0; step < batch; ++step) {
            stencilIteration<<<blocks, threads, 0, stream>>>(
                step % 2 == 0 ? grid1 : grid2, step % 2 == 0 ? grid2 : grid1,
                nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executableGraph, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (hasInterior) {
        int iter = 0;
        if (batch > 0) {
            for (; iter <= iterations - batch; iter += batch) {
                CUDA_CHECK(cudaGraphLaunch(executableGraph, stream));
            }
        }
        for (; iter < iterations; ++iter) {
            stencilIteration<<<blocks, threads, 0, stream>>>(
                iter % 2 == 0 ? grid1 : grid2, iter % 2 == 0 ? grid2 : grid1,
                nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    printf("Computation time: %.3f ms\n", milliseconds);
    const double cellUpdates = hasInterior ? double(nx - 2) * double(ny - 2) *
                                            double(nz - 2) * iterations : 0.0;
    const double mcups = milliseconds > 0 ? cellUpdates / (milliseconds * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), iterations % 2 == 0 ? grid1 : grid2,
                              bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    if (executableGraph) CUDA_CHECK(cudaGraphExecDestroy(executableGraph));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaFree(grid2));

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
