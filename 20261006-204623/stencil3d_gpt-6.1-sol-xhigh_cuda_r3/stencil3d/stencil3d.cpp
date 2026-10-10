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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

struct DeviceGrids {
    Real* first = nullptr;
    Real* second = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;

    ~DeviceGrids() {
        if (graphExec) cudaGraphExecDestroy(graphExec);
        if (graph) cudaGraphDestroy(graph);
        if (first) cudaFree(first);
        if (second) cudaFree(second);
        if (stream) cudaStreamDestroy(stream);
    }
};

__global__ void initializeGrid(Real* first, Real* second, size_t count) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(idx % 19);
        first[idx] = value;
        // Both buffers retain the same fixed boundary for every iteration.
        second[idx] = value;
    }
}

constexpr unsigned tileX = 64;
constexpr unsigned tileY = 2;
constexpr unsigned tileZ = 4;
constexpr unsigned blockThreads = tileX * tileY;

__device__ __forceinline__ Real divideBySeven(Real value) {
    // Nonzero stencil sums and quotients are well within the normal range.
    // A rounded reciprocal followed by an exact FMA residual and correction
    // gives the same rounded quotient as division by 7, without recomputing
    // the reciprocal in every cell. Division by this odd integer has no ties.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = __dmul_rn(value, reciprocal);
    const Real residual = __fma_rn(-quotient, 7.0, value);
    return __fma_rn(residual, reciprocal, quotient);
}

// On regular grids, each warp accesses consecutive X coordinates. A rolling
// Z window reuses neighbors in registers, and independent Z tiles expose ample
// parallelism.
template <bool FlatTiles>
__global__ __launch_bounds__(blockThreads)
void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                      size_t nx, size_t ny, size_t nz, size_t tilesY, size_t tilesYZ) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    // Normal grids use direct 3D indexing. Flattening with a grid-stride loop
    // also supports dimensions exceeding CUDA's Y/Z launch limits.
    const size_t firstTile = FlatTiles ? blockIdx.y : 0;
    const size_t tileCount = FlatTiles ? tilesYZ : 1;
    const size_t tileStride = FlatTiles ? gridDim.y : 1;
    for (size_t tile = firstTile; tile < tileCount; tile += tileStride) {
        const size_t yTile = FlatTiles ? tile % tilesY : blockIdx.y;
        const size_t zTile = FlatTiles ? tile / tilesY : blockIdx.z;
        const size_t y = yTile * blockDim.y + threadIdx.y + 1;
        if (y >= ny - 1) continue;
        const size_t z = (zTile * blockDim.z + threadIdx.z) * tileZ + 1;
        if (z >= nz - 1) continue;
        size_t idx = z * plane + y * nx + x;
        Real bottom = input[idx - plane];
        Real center = input[idx];
#pragma unroll
        for (unsigned step = 0; step < tileZ; ++step) {
            if (z + step < nz - 1) {
                const Real top = input[idx + plane];
                // Keep the CPU's addition order and correctly rounded division.
                const Real sum = center + input[idx - 1] + input[idx + 1]
                               + input[idx - nx] + input[idx + nx] + bottom + top;
                output[idx] = divideBySeven(sum);
                bottom = center;
                center = top;
                idx += plane;
            }
        }
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
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > maxCells / ny || nx * ny > maxCells / nz) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    const size_t gridBytes = gridSize * sizeof(Real);
    DeviceGrids grids;
    CUDA_CHECK(cudaStreamCreateWithFlags(&grids.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&grids.first), gridBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&grids.second), gridBytes));
    
    // Initialize
    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize + 255) / 256, 65535));
    initializeGrid<<<initBlocks, 256, 0, grids.stream>>>(grids.first, grids.second, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(grids.stream));

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    dim3 threads(tileX, tileY);
    if (hasInterior) {
        // Redistribute threads to keep lanes active on thin grids. Remaining
        // threads process independent groups of Z cells.
        while (threads.x > 1 && threads.x / 2 >= nx - 2) threads.x /= 2;
        threads.y = blockThreads / threads.x;
        while (threads.y > 1 && threads.y / 2 >= ny - 2) threads.y /= 2;
        // CUDA permits at most 64 threads in the block's Z dimension.
        threads.z = std::min(64u, blockThreads / (threads.x * threads.y));
    }
    const size_t tilesY = hasInterior ? (ny - 2 + threads.y - 1) / threads.y : 0;
    const size_t tilesZ = hasInterior ? (nz - 2 + threads.z * tileZ - 1) / (threads.z * tileZ) : 0;
    const size_t tilesYZ = tilesY * tilesZ;
    const bool flatTiles = tilesY > 65535 || tilesZ > 65535;
    const dim3 blocks(hasInterior ? static_cast<unsigned>((nx - 2 + threads.x - 1) / threads.x) : 1,
                      flatTiles ? static_cast<unsigned>(std::min<size_t>(tilesYZ, 65535))
                                : static_cast<unsigned>(tilesY),
                      flatTiles ? 1 : static_cast<unsigned>(tilesZ));
    Real* input = grids.first;
    Real* output = grids.second;

    const auto launchIteration = [&](Real* source, Real* destination) {
        if (flatTiles) {
            stencilIteration<true><<<blocks, threads, 0, grids.stream>>>(source, destination, nx, ny, nz, tilesY, tilesYZ);
        } else {
            stencilIteration<false><<<blocks, threads, 0, grids.stream>>>(source, destination, nx, ny, nz, tilesY, tilesYZ);
        }
    };

    // An even batch always returns to the first buffer, so the same graph can
    // be replayed without changing kernel parameters or transferring data.
    // Limit graph size independently of the requested iteration count.
    const int graphSteps = hasInterior && iterations >= 4 ? std::min(iterations - iterations % 2, 128) : 0;
    if (graphSteps != 0) {
        CUDA_CHECK(cudaStreamBeginCapture(grids.stream, cudaStreamCaptureModeThreadLocal));
        for (int iter = 0; iter < graphSteps; iter += 2) {
            launchIteration(grids.first, grids.second);
            launchIteration(grids.second, grids.first);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(grids.stream, &grids.graph));
        CUDA_CHECK(cudaGraphInstantiate(&grids.graphExec, grids.graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(grids.graphExec, grids.stream));
        CUDA_CHECK(cudaStreamSynchronize(grids.stream));
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (hasInterior) {
        if (graphSteps != 0) {
            for (int batch = 0; batch < iterations / graphSteps; ++batch) {
                CUDA_CHECK(cudaGraphLaunch(grids.graphExec, grids.stream));
            }
        }
        const int remaining = graphSteps != 0 ? iterations % graphSteps : iterations;
        for (int iter = 0; iter < remaining; ++iter) {
            launchIteration(input, output);
            std::swap(input, output);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(grids.stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate performance metrics
    const double cellUpdates = hasInterior ? static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations : 0.0;
    const double mcups = milliseconds > 0.0 ? cellUpdates / (milliseconds * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), input, gridBytes, cudaMemcpyDeviceToHost));
    }
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
