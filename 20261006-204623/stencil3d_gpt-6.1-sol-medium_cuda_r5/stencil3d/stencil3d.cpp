#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// Every CUDA failure is fatal: there is deliberately no CPU fallback.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrids(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const Real value = Real(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

constexpr unsigned TileX = 32;
constexpr unsigned TileY = 4;
constexpr unsigned TileZ = 8;

// Adjacent lanes access adjacent X cells. Each thread sweeps a short Z
// column, retaining the previous and current planes in registers. Separate
// Z tiles expose enough independent blocks even for very deep grids.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t tilesX, size_t tilesY, size_t tiles) {
    const size_t plane = nx * ny;
    for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const size_t tx = tile % tilesX;
        const size_t yz = tile / tilesX;
        const size_t y = (yz % tilesY) * TileY + threadIdx.y + 1;
        const size_t x = tx * TileX + threadIdx.x;
        const size_t zBegin = (yz / tilesY) * TileZ + 1;
        if (x == 0 || x >= nx - 1 || y >= ny - 1) continue;

        size_t index = zBegin * plane + y * nx + x;
        Real bottom = input[index - plane];
        Real center = input[index];
        const size_t zEnd = (zBegin + TileZ < nz - 1) ? zBegin + TileZ : nz - 1;
        for (size_t z = zBegin; z < zEnd; ++z, index += plane) {
            const Real top = input[index + plane];
            // Preserve the CPU expression's summation order and precision.
            output[index] = (center + input[index - 1] + input[index + 1]
                          + input[index - nx] + input[index + nx]
                          + bottom + top) / 7.0;
            bottom = center;
            center = top;
        }
    }
    // Both buffers were initialized identically, so boundaries need no writes.
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
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Create CUDA stream");
    checkCuda(cudaMalloc(&grid1, bytes), "Allocate first GPU grid");
    checkCuda(cudaMalloc(&grid2, bytes), "Allocate second GPU grid");

    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize - 1) / 256 + 1, 65535));
    initializeGrids<<<initBlocks, 256, 0, stream>>>(grid1, grid2, gridSize);
    checkCuda(cudaGetLastError(), "Launch grid initialization");
    checkCuda(cudaStreamSynchronize(stream), "Initialize GPU grids");

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t tilesX = (nx - 1) / TileX + 1;
    const size_t tilesY = hasInterior ? (ny - 3) / TileY + 1 : 0;
    const size_t tilesZ = hasInterior ? (nz - 3) / TileZ + 1 : 0;
    const size_t tiles = tilesX * tilesY * tilesZ;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(tiles, 65535));
    const dim3 threads(TileX, TileY);
    auto launchIteration = [&](int iter) {
        const Real* input = (iter % 2 == 0) ? grid1 : grid2;
        Real* output = (iter % 2 == 0) ? grid2 : grid1;
        stencilIteration<<<blocks, threads, 0, stream>>>(input, output, nx, ny, nz,
                                                       tilesX, tilesY, tiles);
    };

    // A bounded graph amortizes host launch overhead without allocating one
    // graph node per iteration for arbitrarily long runs. An even batch keeps
    // the ping-pong buffer orientation unchanged between graph launches.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    int batch = std::min(iterations, 32);
    if (batch > 1) batch -= batch % 2;
    if (hasInterior && batch > 0) {
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "Begin stencil graph");
        for (int iter = 0; iter < batch; ++iter) launchIteration(iter);
        checkCuda(cudaStreamEndCapture(stream, &graph), "Capture stencil graph");
        checkCuda(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0), "Instantiate stencil graph");
        checkCuda(cudaGraphUpload(executableGraph, stream), "Upload stencil graph");
        checkCuda(cudaStreamSynchronize(stream), "Prepare stencil graph");
    }

    printf("Running stencil computation...\n");
    const auto start = std::chrono::steady_clock::now();
    if (hasInterior && batch > 0) {
        const int batches = iterations / batch;
        for (int i = 0; i < batches; ++i) {
            checkCuda(cudaGraphLaunch(executableGraph, stream), "Launch stencil graph");
        }
        for (int iter = batches * batch; iter < iterations; ++iter) launchIteration(iter);
        checkCuda(cudaGetLastError(), "Launch stencil iteration");
    }
    checkCuda(cudaStreamSynchronize(stream), "Complete stencil computation");
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %.3f ms\n", seconds * 1000.0);

    const double cellUpdates = hasInterior ? double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations : 0.0;
    const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Transfer only when host-side result reporting or validation needs it.
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        const Real* result = (iterations % 2 == 0) ? grid1 : grid2;
        checkCuda(cudaMemcpyAsync(finalGrid.data(), result, bytes, cudaMemcpyDeviceToHost, stream), "Copy final grid");
        checkCuda(cudaStreamSynchronize(stream), "Retrieve final grid");
    }
    if (executableGraph) checkCuda(cudaGraphExecDestroy(executableGraph), "Destroy stencil graph executable");
    if (graph) checkCuda(cudaGraphDestroy(graph), "Destroy stencil graph");
    checkCuda(cudaFree(grid2), "Free second GPU grid");
    checkCuda(cudaFree(grid1), "Free first GPU grid");
    checkCuda(cudaStreamDestroy(stream), "Destroy CUDA stream");

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
