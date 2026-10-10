#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// CUDA resources are released even if an allocation or launch fails.
struct DeviceGrid {
    Real* data = nullptr;
    explicit DeviceGrid(size_t bytes) {
        checkCuda(cudaMalloc(&data, bytes), "Allocating device grid");
    }
    ~DeviceGrid() { if (data) cudaFree(data); }
    DeviceGrid(const DeviceGrid&) = delete;
    DeviceGrid& operator=(const DeviceGrid&) = delete;
};

struct CudaEvent {
    cudaEvent_t event = nullptr;
    CudaEvent() { checkCuda(cudaEventCreate(&event), "Creating timing event"); }
    ~CudaEvent() { if (event) cudaEventDestroy(event); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};

struct CudaExecution {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CudaExecution() {
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "Creating CUDA stream");
    }
    ~CudaExecution() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
    }
    CudaExecution(const CudaExecution&) = delete;
    CudaExecution& operator=(const CudaExecution&) = delete;
};

__global__ void initializeGrid(Real* __restrict__ first, Real* __restrict__ second,
                               size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        // Initialize both sets of boundaries once; kernels only write interiors.
        second[i] = value;
    }
}

constexpr unsigned tileX = 32;
constexpr unsigned tileY = 4;
constexpr unsigned tileZ = 8;

__device__ __forceinline__ Real divideBySeven(Real sum) {
    constexpr Real reciprocal = 0x1.2492492492492p-3;
    const Real quotient = __dmul_rn(sum, reciprocal);
    // All stencil sums are bounded, nonnegative normal doubles (or zero).
    // The fused residual is exact here. Its correction gives the same rounded
    // result as division by seven, without repeatedly refining a reciprocal.
    const Real residual = __fma_rn(-7.0, quotient, sum);
    return __fma_rn(residual, reciprocal, quotient);
}

// Adjacent lanes access adjacent X coordinates. Each thread streams a short
// Z column, retaining the bottom and center values in registers between planes.
// Grid-stride tiles also support dimensions exceeding CUDA's launch limits.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    for (size_t firstZ = size_t(blockIdx.z) * tileZ + 1; firstZ < nz - 1;
         firstZ += size_t(gridDim.z) * tileZ) {
        const size_t endZ = firstZ + tileZ < nz - 1 ? firstZ + tileZ : nz - 1;
        for (size_t y = size_t(blockIdx.y) * tileY + threadIdx.y + 1; y < ny - 1;
             y += size_t(gridDim.y) * tileY) {
            for (size_t x = size_t(blockIdx.x) * tileX + threadIdx.x; x < nx - 1;
                 x += size_t(gridDim.x) * tileX) {
                if (x == 0) continue;
                size_t idx = firstZ * plane + y * nx + x;
                Real bottom = input[idx - plane];
                Real center = input[idx];
                for (size_t z = firstZ; z < endZ; ++z, idx += plane) {
                    const Real top = input[idx + plane];
                    output[idx] = divideBySeven(center + input[idx - 1] + input[idx + 1]
                                           + input[idx - nx] + input[idx + nx]
                                           + bottom + top);
                    bottom = center;
                    center = top;
                }
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

    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > maxElements / ny || nx * ny > maxElements / nz) {
        fprintf(stderr, "Grid dimensions must be positive and fit in memory; iterations must be nonnegative.\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    try {
        const size_t gridSize = nx * ny * nz;
        const size_t bytes = gridSize * sizeof(Real);
        DeviceGrid grid1(bytes);
        DeviceGrid grid2(bytes);
        CudaExecution execution;
        CudaEvent start;
        CudaEvent end;

        printf("Initializing grid...\n");
        const unsigned initBlocks = static_cast<unsigned>(
            std::min<size_t>((gridSize - 1) / 256 + 1, 65535));
        initializeGrid<<<initBlocks, 256, 0, execution.stream>>>(grid1.data, grid2.data, gridSize);
        checkCuda(cudaGetLastError(), "Launching grid initialization");
        checkCuda(cudaStreamSynchronize(execution.stream), "Initializing device grids");

        const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
        const size_t tilesX = (nx - 1) / tileX + 1;
        const size_t tilesY = hasInterior ? (ny - 3) / tileY + 1 : 0;
        const size_t tilesZ = hasInterior ? (nz - 3) / tileZ + 1 : 0;
        const dim3 blocks(static_cast<unsigned>(std::min<size_t>(tilesX, 65535)),
                          static_cast<unsigned>(std::min<size_t>(tilesY, 65535)),
                          static_cast<unsigned>(std::min<size_t>(tilesZ, 65535)));
        const dim3 threads(tileX, tileY);
        const auto launchIteration = [&](int iter) {
            const Real* input = (iter % 2 == 0) ? grid1.data : grid2.data;
            Real* output = (iter % 2 == 0) ? grid2.data : grid1.data;
            stencilIteration<<<blocks, threads, 0, execution.stream>>>(
                input, output, nx, ny, nz);
            checkCuda(cudaGetLastError(), "Launching stencil kernel");
        };

        // Replay batches of an even number of steps to reduce launch overhead.
        // An even batch always returns the latest values to grid1.
        const int batch = hasInterior ? std::min(iterations & ~1, 64) : 0;
        if (batch > 0) {
            checkCuda(cudaStreamBeginCapture(execution.stream, cudaStreamCaptureModeThreadLocal),
                      "Starting stencil graph capture");
            for (int iter = 0; iter < batch; ++iter) launchIteration(iter);
            checkCuda(cudaStreamEndCapture(execution.stream, &execution.graph),
                      "Completing stencil graph capture");
            checkCuda(cudaGraphInstantiate(&execution.executable, execution.graph, nullptr, nullptr, 0),
                      "Instantiating stencil graph");
            checkCuda(cudaGraphUpload(execution.executable, execution.stream), "Uploading stencil graph");
            checkCuda(cudaStreamSynchronize(execution.stream), "Preparing stencil graph");
        }

        printf("Running stencil computation...\n");
        checkCuda(cudaEventRecord(start.event, execution.stream), "Recording start time");
        if (hasInterior) {
            const int batches = batch > 0 ? iterations / batch : 0;
            for (int b = 0; b < batches; ++b) {
                checkCuda(cudaGraphLaunch(execution.executable, execution.stream), "Launching stencil graph");
            }
            for (int iter = batches * batch; iter < iterations; ++iter) launchIteration(iter);
        }
        checkCuda(cudaEventRecord(end.event, execution.stream), "Recording end time");
        checkCuda(cudaEventSynchronize(end.event), "Completing stencil computation");
        float milliseconds = 0;
        checkCuda(cudaEventElapsedTime(&milliseconds, start.event, end.event), "Reading elapsed time");
        printf("Computation time: %.3f ms\n", static_cast<double>(milliseconds));

        const double cellUpdates = hasInterior
            ? double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations : 0;
        const double mcups = milliseconds > 0 ? cellUpdates / (double(milliseconds) * 1000.0) : 0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // There are no host/device transfers between steps. Download once only
        // when the caller needs the final grid for validation or result output.
        if (printResults || validate) {
            std::vector<Real> finalGrid(gridSize);
            const Real* finalDevice = (iterations % 2 == 0) ? grid1.data : grid2.data;
            checkCuda(cudaMemcpy(finalGrid.data(), finalDevice, bytes, cudaMemcpyDeviceToHost),
                      "Downloading stencil result");
            if (printResults) print_results(finalGrid, "Grid");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(finalGrid, nx, ny, nz);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                return valid ? 0 : 1;
            }
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Stencil computation failed: %s\n", error.what());
        return 1;
    }
    
    return 0;
}
