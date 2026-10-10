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

struct CudaResources {
    Real* grids[2] = {nullptr, nullptr};
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~CudaResources() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        for (Real* grid : grids) {
            if (grid) cudaFree(grid);
        }
        if (stream) cudaStreamDestroy(stream);
    }
};

__global__ void initializeGrids(Real* first, Real* second, size_t count,
                                bool initializeSecond) {
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        second[i] = initializeSecond ? value : 0.0;
    }
}

// Adjacent lanes visit adjacent x coordinates. Each thread walks a short z
// column, retaining the previous/current/next planes in registers. The read-only
// cache supplies the x/y neighbors shared by nearby threads and blocks.
template <unsigned ZTile>
__device__ __forceinline__ void stencilColumn(const Real* __restrict__ input,
                                             Real* __restrict__ output,
                                             size_t nx, size_t ny, size_t nz,
                                             size_t x, size_t y, size_t zBegin) {
    const size_t plane = nx * ny;
    size_t index = zBegin * plane + y * nx + x;
    Real bottom = input[index - plane];
    Real center = input[index];
    const size_t zEnd = min(zBegin + ZTile, nz - 1);
    for (size_t z = zBegin; z < zEnd; ++z, index += plane) {
        const Real top = input[index + plane];
        const Real left = input[index - 1];
        const Real right = input[index + 1];
        const Real front = input[index - nx];
        const Real back = input[index + nx];
        const Real sum = center + left + right + front + back + bottom + top;
        // Division by the fixed integer 7: correct the rounded reciprocal
        // product using its exact residual. For this bounded, nonnegative
        // stencil, the two explicit FMAs give the same rounding as sum / 7.0
        // without the general divider's reciprocal refinement and range checks.
        constexpr Real reciprocal = 0x1.2492492492492p-3;
        const Real quotient = sum * reciprocal;
        const Real residual = __fma_rn(-quotient, 7.0, sum);
        output[index] = __fma_rn(residual, reciprocal, quotient);
        bottom = center;
        center = top;
    }
}

template <unsigned ZTile, bool GridStride = false>
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t yBegin = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t zBegin = size_t(blockIdx.z) * ZTile + 1;
    if (x == 0 || x >= nx - 1 || yBegin >= ny - 1) return;

    if constexpr (GridStride) {
        // CUDA limits the y/z launch dimensions to 65535 blocks. Stride over
        // additional tiles for long, narrow grids without limiting grid size.
        for (size_t y = yBegin; y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
            for (size_t z = zBegin; z < nz - 1; z += size_t(gridDim.z) * ZTile) {
                stencilColumn<ZTile>(input, output, nx, ny, nz, x, y, z);
            }
        }
    } else {
        stencilColumn<ZTile>(input, output, nx, ny, nz, x, yBegin, zBegin);
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
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    CudaResources gpu;
    checkCuda(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking), "creating stream");
    checkCuda(cudaMalloc(&gpu.grids[0], bytes), "allocating first grid");
    checkCuda(cudaMalloc(&gpu.grids[1], bytes), "allocating second grid");
    
    // Initialize
    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize + 255) / 256, 65535));
    initializeGrids<<<initBlocks, 256, 0, gpu.stream>>>(gpu.grids[0], gpu.grids[1], gridSize, iterations > 0);
    checkCuda(cudaGetLastError(), "initializing grids");

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    constexpr unsigned zTile = 16;
    const dim3 threads(64, 2);
    dim3 blocks;
    bool gridStride = false;
    if (hasInterior) {
        const size_t bx = (nx + threads.x - 1) / threads.x;
        const size_t by = (ny - 2 + threads.y - 1) / threads.y;
        const size_t bz = (nz - 2 + zTile - 1) / zTile;
        gridStride = by > 65535 || bz > 65535;
        blocks = dim3(static_cast<unsigned>(bx),
                      static_cast<unsigned>(std::min<size_t>(by, 65535)),
                      static_cast<unsigned>(std::min<size_t>(bz, 65535)));
    }
    auto launchIteration = [&](int iter) {
        if (gridStride) {
            stencilIteration<zTile, true><<<blocks, threads, 0, gpu.stream>>>(
                gpu.grids[iter & 1], gpu.grids[(iter & 1) ^ 1], nx, ny, nz);
        } else {
            stencilIteration<zTile><<<blocks, threads, 0, gpu.stream>>>(
                gpu.grids[iter & 1], gpu.grids[(iter & 1) ^ 1], nx, ny, nz);
        }
    };

    // An even-sized graph returns to the same ping-pong buffers, so it can be
    // replayed without rebuilding or modifying pointers. Bound graph size even
    // for very long runs, and launch any remaining iterations normally.
    const int graphIterations = hasInterior ? std::min(iterations, 32) / 2 * 2 : 0;
    if (graphIterations > 0) {
        checkCuda(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeThreadLocal), "starting graph capture");
        for (int iter = 0; iter < graphIterations; ++iter) launchIteration(iter);
        checkCuda(cudaStreamEndCapture(gpu.stream, &gpu.graph), "finishing graph capture");
        checkCuda(cudaGraphInstantiate(&gpu.executable, gpu.graph, nullptr, nullptr, 0), "instantiating graph");
        checkCuda(cudaGraphUpload(gpu.executable, gpu.stream), "uploading graph");
    }
    checkCuda(cudaStreamSynchronize(gpu.stream), "finishing initialization");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (hasInterior) {
        int iter = 0;
        if (graphIterations > 0) {
            for (; iter <= iterations - graphIterations; iter += graphIterations) {
                checkCuda(cudaGraphLaunch(gpu.executable, gpu.stream), "launching stencil graph");
            }
        }
        for (; iter < iterations; ++iter) launchIteration(iter);
        checkCuda(cudaGetLastError(), "launching stencil");
    }
    checkCuda(cudaStreamSynchronize(gpu.stream), "running stencil");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %lld ms\n", static_cast<long long>(duration.count()));
    
    // Calculate performance metrics
    double cellUpdates = hasInterior ? double(nx-2) * double(ny-2) * double(nz-2) * std::max(iterations, 0) : 0.0;
    double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), gpu.grids[iterations % 2 == 0 ? 0 : 1], bytes,
                             cudaMemcpyDeviceToHost), "copying results to host");
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
