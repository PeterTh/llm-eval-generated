#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
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

struct CudaResources {
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~CudaResources() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
        if (grid1) cudaFree(grid1);
        if (grid2) cudaFree(grid2);
    }
};

__global__ void initializeGrid(Real* __restrict__ grid1,
                               Real* __restrict__ grid2, size_t count) {
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const Real value = Real(i % 19);
        // Boundaries are identical in both buffers and never written again.
        grid1[i] = value;
        grid2[i] = value;
    }
}

constexpr unsigned TileX = 32;
constexpr unsigned TileY = 4;
constexpr unsigned TileZ = 8;

template<bool BoundedRange>
__device__ __forceinline__ Real averageSeven(Real sum) {
    // Grid values stay in [0, 18]. A two-double reciprocal approximates 1/7
    // to 106 bits; the FMA rounds its product just once. Since division by
    // the odd integer 7 cannot produce a halfway tie, this gives exactly the
    // same double as sum / 7.0 throughout the normal range used here.
    constexpr Real reciprocal = 1.0 / 7.0;
    if constexpr (BoundedRange) {
        constexpr Real reciprocalTail = 0x1.2492492492492p-57;
        return __fma_rn(sum, reciprocal, sum * reciprocalTail);
    } else {
        // An exact residual correction also handles zero and gradual
        // underflow during exceptionally long runs.
        const Real quotient = __dmul_rn(sum, reciprocal);
        const Real residual = __fma_rn(-quotient, 7.0, sum);
        return __fma_rn(residual, reciprocal, quotient);
    }
}

// Each warp accesses consecutive X coordinates. A thread advances through
// several Z planes, keeping the center and Z neighbors in registers.
template<unsigned Z, bool BoundedRange>
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * TileX + threadIdx.x;
    if (x == 0 || x >= nx - 1) return;
    const size_t plane = nx * ny;

    // Grid strides also support thin grids exceeding CUDA's Y/Z grid limits.
    for (size_t y = size_t(blockIdx.y) * TileY + threadIdx.y + 1;
         y < ny - 1; y += size_t(gridDim.y) * TileY) {
        for (size_t firstZ = size_t(blockIdx.z) * Z + 1;
             firstZ < nz - 1; firstZ += size_t(gridDim.z) * Z) {
            size_t i = firstZ * plane + y * nx + x;
            Real bottom = input[i - plane];
            Real center = input[i];
            Real top = input[i + plane];

            #pragma unroll
            for (unsigned dz = 0; dz < Z; ++dz) {
                if (firstZ + dz >= nz - 1) break;
                const Real left = input[i - 1];
                const Real right = input[i + 1];
                const Real front = input[i - nx];
                const Real back = input[i + nx];
                output[i] = averageSeven<BoundedRange>(center + left + right + front + back + bottom + top);
                bottom = center;
                center = top;
                if (dz + 1 < Z && firstZ + dz + 1 < nz - 1)
                    top = input[i + 2 * plane];
                i += plane;
            }
        }
    }
}

// Avoid mostly inactive warps for narrow grids; neighboring threads still
// access neighboring elements of the original, unpadded grid.
template<bool BoundedRange>
__global__ void stencilIterationLinear(const Real* __restrict__ input,
                                       Real* __restrict__ output,
                                       size_t nx, size_t ny, size_t count) {
    const size_t plane = nx * ny;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const size_t x = i % nx;
        const size_t y = (i / nx) % ny;
        if (i >= plane && i < count - plane &&
            x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
            output[i] = averageSeven<BoundedRange>(input[i] + input[i - 1] + input[i + 1] +
                                                  input[i - nx] + input[i + nx] +
                                                  input[i - plane] + input[i + plane]);
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
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    CudaResources gpu;
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&gpu.start));
    CUDA_CHECK(cudaEventCreate(&gpu.stop));
    CUDA_CHECK(cudaMalloc(&gpu.grid1, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.grid2, bytes));
    
    // Initialize
    printf("Initializing grid...\n");
    const unsigned linearBlocks = unsigned(std::min(
        (gridSize + 255) / 256, size_t(properties.multiProcessorCount) * 32));
    initializeGrid<<<linearBlocks, 256, 0, gpu.stream>>>(gpu.grid1, gpu.grid2, gridSize);
    CUDA_CHECK(cudaGetLastError());

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const bool useLinear = nx < TileX || ny < TileY;
    // Smaller Z tiles expose more blocks and improve occupancy on small grids.
    const bool useSinglePlane = gridSize <= size_t(128) * 128 * 128;
    const unsigned tileZ = useSinglePlane ? 1 : TileZ;
    dim3 blocks(1, 1, 1);
    if (hasInterior) {
        blocks = dim3(unsigned((nx - 1 + TileX - 1) / TileX),
                      unsigned(std::min((ny - 2 + TileY - 1) / TileY,
                                        size_t(properties.maxGridSize[1]))),
                      unsigned(std::min((nz - 2 + tileZ - 1) / tileZ,
                                        size_t(properties.maxGridSize[2]))));
    }

    auto enqueueIteration = [&](int iter) {
        const Real* input = (iter % 2 == 0) ? gpu.grid1 : gpu.grid2;
        Real* output = (iter % 2 == 0) ? gpu.grid2 : gpu.grid1;
        auto launch = [&](auto boundedRange) {
            constexpr bool bounded = decltype(boundedRange)::value;
            if (useLinear) {
                stencilIterationLinear<bounded><<<linearBlocks, 256, 0, gpu.stream>>>(
                    input, output, nx, ny, gridSize);
            } else if (useSinglePlane) {
                stencilIteration<1, bounded><<<blocks, dim3(TileX, TileY), 0, gpu.stream>>>(
                    input, output, nx, ny, nz);
            } else {
                stencilIteration<TileZ, bounded><<<blocks, dim3(TileX, TileY), 0, gpu.stream>>>(
                    input, output, nx, ny, nz);
            }
        };
        // The initial center/left/right sum is at least 3. Subsequently each
        // sum retains at least the previous center, giving a lower bound of
        // 3 / 7^(iter). Through 340 iterations this exceeds 2^-960, keeping
        // the two-double reciprocal's low product normal. Longer runs use
        // residual correction, without a per-cell range check in either case.
        if (iterations <= 340) launch(std::true_type{});
        else launch(std::false_type{});
        CUDA_CHECK(cudaGetLastError());
    };

    // Capture at most 64 iterations to bound graph construction/storage for
    // long runs. Repeated batches are even-sized, preserving buffer parity.
    const int batch = std::min(iterations, 64);
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    if (hasInterior && batch > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeThreadLocal));
        for (int iter = 0; iter < batch; ++iter) enqueueIteration(iter);
        CUDA_CHECK(cudaStreamEndCapture(gpu.stream, &gpu.graph));
        CUDA_CHECK(cudaGraphInstantiate(&gpu.executable, gpu.graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(gpu.executable, gpu.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(gpu.start, gpu.stream));
    if (hasInterior && batch > 0) {
        const int batches = iterations / batch;
        for (int replay = 0; replay < batches; ++replay)
            CUDA_CHECK(cudaGraphLaunch(gpu.executable, gpu.stream));
        for (int iter = batches * batch; iter < iterations; ++iter)
            enqueueIteration(iter);
    }
    CUDA_CHECK(cudaEventRecord(gpu.stop, gpu.stream));
    CUDA_CHECK(cudaEventSynchronize(gpu.stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, gpu.start, gpu.stop));

    printf("Computation time: %.3f ms\n", double(milliseconds));
    
    // Calculate performance metrics
    const double cellUpdates = hasInterior ?
        double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations : 0.0;
    const double mcups = milliseconds > 0 ? cellUpdates / (milliseconds * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        const Real* finalDeviceGrid = (iterations % 2 == 0) ? gpu.grid1 : gpu.grid2;
        CUDA_CHECK(cudaMemcpyAsync(finalGrid.data(), finalDeviceGrid, bytes,
                                   cudaMemcpyDeviceToHost, gpu.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
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
