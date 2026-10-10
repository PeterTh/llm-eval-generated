#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// Own all CUDA resources, including on an error path.
struct GpuState {
    Real* first = nullptr;
    Real* second = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~GpuState() {
        if (stream) cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (first) cudaFree(first);
        if (second) cudaFree(second);
        if (stream) cudaStreamDestroy(stream);
    }
};

__global__ void initializeGrid(Real* first, Real* second, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

// Warps span contiguous X coordinates. Each thread walks a short Z column,
// retaining the previous/current/next planes in registers. All boundaries
// remain at their initial values in both buffers throughout the computation.
constexpr unsigned zTile = 16;
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t begin = static_cast<size_t>(blockIdx.z) * zTile + 1;
    if (x >= nx - 1 || y >= ny - 1 || begin >= nz - 1) return;
    const size_t plane = nx * ny;
    size_t index = begin * plane + y * nx + x;
    Real bottom = input[index - plane];
    Real center = input[index];
    const size_t end = (begin + zTile < nz - 1) ? begin + zTile : nz - 1;
    for (size_t z = begin; z < end; ++z, index += plane) {
        const Real top = input[index + plane];
        // Explicit correctly rounded division preserves the CPU expression.
        output[index] = __ddiv_rn(center + input[index - 1] + input[index + 1]
                                + input[index - nx] + input[index + nx]
                                + bottom + top, 7.0);
        bottom = center;
        center = top;
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

int run(int argc, char** argv) {
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
    GpuState gpu;
    cudaCheck(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
    cudaCheck(cudaMalloc(&gpu.first, bytes));
    cudaCheck(cudaMalloc(&gpu.second, bytes));

    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize - 1) / 256 + 1, 65535));
    initializeGrid<<<initBlocks, 256, 0, gpu.stream>>>(gpu.first, gpu.second, gridSize);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(gpu.stream));

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const dim3 block(32, 8);
    dim3 blocks;
    if (hasInterior) {
        const size_t bx = (nx - 3) / block.x + 1;
        const size_t by = (ny - 3) / block.y + 1;
        const size_t bz = (nz - 3) / zTile + 1;
        cudaDeviceProp properties;
        int device;
        cudaCheck(cudaGetDevice(&device));
        cudaCheck(cudaGetDeviceProperties(&properties, device));
        if (bx > static_cast<size_t>(properties.maxGridSize[0]) ||
            by > static_cast<size_t>(properties.maxGridSize[1]) ||
            bz > static_cast<size_t>(properties.maxGridSize[2])) {
            throw std::runtime_error("Grid exceeds CUDA launch dimensions");
        }
        blocks = dim3(static_cast<unsigned>(bx), static_cast<unsigned>(by), static_cast<unsigned>(bz));
    }

    // An even-length graph can be replayed without changing buffer parity.
    // Bound its size so very long runs do not need a graph node per iteration.
    const int batch = std::min(iterations - iterations % 2, 64);
    if (hasInterior && batch > 0) {
        cudaCheck(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeThreadLocal));
        for (int iter = 0; iter < batch; ++iter) {
            Real* input = (iter % 2 == 0) ? gpu.first : gpu.second;
            Real* output = (iter % 2 == 0) ? gpu.second : gpu.first;
            stencilIteration<<<blocks, block, 0, gpu.stream>>>(input, output, nx, ny, nz);
        }
        cudaCheck(cudaStreamEndCapture(gpu.stream, &gpu.graph));
        cudaCheck(cudaGraphInstantiate(&gpu.executable, gpu.graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(gpu.executable, gpu.stream));
        cudaCheck(cudaStreamSynchronize(gpu.stream));
    }

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (hasInterior) {
        int iter = 0;
        if (batch > 0) {
            for (; iter <= iterations - batch; iter += batch) {
                cudaCheck(cudaGraphLaunch(gpu.executable, gpu.stream));
            }
        }
        for (; iter < iterations; ++iter) {
            Real* input = (iter % 2 == 0) ? gpu.first : gpu.second;
            Real* output = (iter % 2 == 0) ? gpu.second : gpu.first;
            stencilIteration<<<blocks, block, 0, gpu.stream>>>(input, output, nx, ny, nz);
            cudaCheck(cudaGetLastError());
        }
    }
    cudaCheck(cudaStreamSynchronize(gpu.stream));
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    const double cellUpdates = hasInterior
        ? static_cast<double>(nx - 2) * static_cast<double>(ny - 2) * static_cast<double>(nz - 2) * iterations
        : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? cellUpdates / seconds / 1e6 : 0.0);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        const Real* result = (iterations % 2 == 0) ? gpu.first : gpu.second;
        cudaCheck(cudaMemcpyAsync(finalGrid.data(), result, bytes, cudaMemcpyDeviceToHost, gpu.stream));
        cudaCheck(cudaStreamSynchronize(gpu.stream));
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

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "Stencil failed: %s\n", error.what());
        return 1;
    }
}
