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

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// The benchmark uses unit grid spacing. Keep each directional difference and
// their addition in the same order as the original double-precision stencil.
template <typename Index>
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                           Index index, Index xp, Index xn,
                                           Index yp, Index yn, double center,
                                           double previous, double next) {
    const double cxx = field[index + xp] + field[index - xn] - 2.0 * center;
    const double cyy = field[index + yp] + field[index - yn] - 2.0 * center;
    const double czz = next + previous - 2.0 * center;
    return cxx + cyy + czz;
}

template <bool Chemical>
__device__ __forceinline__ double stencilResult(double center, double lap,
                                                const double* __restrict__ cold,
                                                size_t index) {
    if constexpr (Chemical) {
        constexpr double e_AA = -(2.0 / 9.0);
        constexpr double e_BB = -(2.0 / 9.0);
        constexpr double e_AB = 2.0 / 9.0;
        return 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB
                      - 2.0 * center * e_AB)
               + 3.0 * center + center * center * center - 0.5 * lap;
    } else {
        return cold[index] + 0.01 * lap;
    }
}

// For small or thin grids, one thread per cell exposes enough parallel work.
template <bool Chemical, typename Index>
__global__ void pointStencil(const double* __restrict__ field,
                             const double* __restrict__ cold,
                             double* __restrict__ output,
                             Index nx, Index ny, Index nz, Index volume) {
    const Index plane = nx * ny;
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < volume; linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const Index index = static_cast<Index>(linear);
        const Index x = index % nx;
        const Index y = (index / nx) % ny;
        const Index z = index / plane;
        const double center = field[index];
        const double previous = field[index - (z > 0 ? plane : 0)];
        const double next = field[index + (z + 1 < nz ? plane : 0)];
        const double lap = laplacian<Index>(field, index, x + 1 < nx ? 1 : 0,
                                            x > 0 ? 1 : 0, y + 1 < ny ? nx : 0,
                                            y > 0 ? nx : 0, center, previous, next);
        output[index] = stencilResult<Chemical>(center, lap, cold, index);
    }
}

// Adjacent threads traverse adjacent XY cells. Marching through a short Z strip
// reuses the two Z neighbors in registers instead of repeatedly loading them.
template <bool Chemical, unsigned Strip, typename Index>
__global__ void columnStencil(const double* __restrict__ field,
                              const double* __restrict__ cold,
                              double* __restrict__ output,
                              Index nx, Index ny, Index nz) {
    const Index plane = nx * ny;
    const Index xy = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (xy >= plane) return;
    const Index x = xy % nx;
    const Index y = xy / nx;
    const Index xp = x + 1 < nx ? 1 : 0;
    const Index xn = x > 0 ? 1 : 0;
    const Index yp = y + 1 < ny ? nx : 0;
    const Index yn = y > 0 ? nx : 0;
    for (size_t zbase = static_cast<size_t>(blockIdx.y) * Strip; zbase < nz;
         zbase += static_cast<size_t>(gridDim.y) * Strip) {
        Index index = static_cast<Index>(zbase) * plane + xy;
        double center = field[index];
        double previous = field[index - (zbase > 0 ? plane : 0)];
        #pragma unroll
        for (unsigned k = 0; k < Strip; ++k) {
            const size_t z = zbase + k;
            if (z < nz) {
                const double next = field[index + (z + 1 < nz ? plane : 0)];
                const double lap = laplacian(field, index, xp, xn, yp, yn,
                                             center, previous, next);
                output[index] = stencilResult<Chemical>(center, lap, cold, index);
                previous = center;
                center = next;
                index += plane;
            }
        }
    }
}

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < volume; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const double pseudo = (((index + 1) * 1299709) % volume)
                              / static_cast<double>(volume);
        c[index] = -1.0 + 2.0 * pseudo;
    }
}

template <typename Index>
void launchStepImpl(double*& cold, double*& cnew, double* mu, Index nx, Index ny,
                    Index nz, Index volume, bool useColumns, dim3 grid,
                    unsigned blockSize, cudaStream_t stream) {
    if (useColumns) {
        columnStencil<true, 8><<<grid, blockSize, 0, stream>>>(cold, cold, mu, nx, ny, nz);
        columnStencil<false, 8><<<grid, blockSize, 0, stream>>>(mu, cold, cnew, nx, ny, nz);
    } else {
        pointStencil<true><<<grid, blockSize, 0, stream>>>(cold, cold, mu, nx, ny, nz, volume);
        pointStencil<false><<<grid, blockSize, 0, stream>>>(mu, cold, cnew, nx, ny, nz, volume);
    }
    std::swap(cold, cnew);
}

void launchStep(double*& cold, double*& cnew, double* mu, size_t nx, size_t ny,
                size_t nz, size_t volume, bool useColumns, dim3 grid,
                unsigned blockSize, cudaStream_t stream) {
    // Most fields fit in 32-bit indices, avoiding expensive 64-bit division.
    // Keep a 64-bit path for grids that exceed that range.
    if (volume <= std::numeric_limits<unsigned>::max()) {
        launchStepImpl(cold, cnew, mu, static_cast<unsigned>(nx),
                       static_cast<unsigned>(ny), static_cast<unsigned>(nz),
                       static_cast<unsigned>(volume), useColumns, grid, blockSize, stream);
    } else {
        launchStepImpl(cold, cnew, mu, nx, ny, nz, volume, useColumns, grid, blockSize, stream);
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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

    const size_t maxSize = std::numeric_limits<size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxSize / ny
        || nx * ny > maxSize / nz
        || nx * ny * nz > maxSize / (3 * sizeof(double))) {
        fprintf(stderr, "Invalid grid dimensions or grid size overflow\n");
        return 1;
    }
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "get device");
    checkCuda(cudaGetDeviceProperties(&properties, device), "get device properties");
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    double* storage = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&storage), 3 * bytes), "allocate fields");
    double* cold = storage;
    double* cnew = storage + gridSize;
    double* mu = storage + 2 * gridSize;

    // Smaller blocks give more independent work on devices with few FP64 units;
    // wider blocks suit devices where memory bandwidth dominates the stencil.
    const unsigned blockSize = properties.singleToDoublePrecisionPerfRatio > 8 ? 64 : 128;
    const unsigned pointBlocks = static_cast<unsigned>(std::min(
        (gridSize - 1) / blockSize + 1, static_cast<size_t>(properties.maxGridSize[0])));
    const bool useColumns = gridSize >= 1048576 && nx * ny >= blockSize && nz > 1
        && (nx * ny - 1) / blockSize + 1 <= static_cast<size_t>(properties.maxGridSize[0]);
    const dim3 grid = useColumns
        ? dim3(static_cast<unsigned>((nx * ny - 1) / blockSize + 1),
               static_cast<unsigned>(std::min((nz - 1) / 8 + 1,
                   static_cast<size_t>(properties.maxGridSize[1]))))
        : dim3(pointBlocks);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    const unsigned initBlocks = std::min(pointBlocks,
                                         static_cast<unsigned>(properties.multiProcessorCount * 32));
    initializeConcentration<<<initBlocks, 128, 0, stream>>>(cold, gridSize);
    checkCuda(cudaGetLastError(), "launch initialization");

    // An even number of steps leaves the ping-pong pointers unchanged, so a
    // bounded graph can be replayed for arbitrarily many time steps. Capture
    // and instantiate outside the timed region, like the field allocation.
    const int graphSteps = iterations > 1 ? (std::min(iterations, 32) & ~1) : 0;
    const int graphReplays = graphSteps > 0 ? iterations / graphSteps : 0;
    const int tailSteps = std::max(iterations, 0) - graphReplays * graphSteps;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    cudaGraph_t tailGraph = nullptr;
    cudaGraphExec_t executableTail = nullptr;
    auto captureSteps = [&](int steps, cudaGraph_t& capturedGraph, cudaGraphExec_t& executable) {
        double* input = cold;
        double* output = cnew;
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        for (int t = 0; t < steps; ++t) {
            launchStep(input, output, mu, nx, ny, nz, gridSize, useColumns, grid, blockSize, stream);
        }
        checkCuda(cudaGetLastError(), "capture stencil kernels");
        checkCuda(cudaStreamEndCapture(stream, &capturedGraph), "end capture");
        checkCuda(cudaGraphInstantiate(&executable, capturedGraph, nullptr, nullptr, 0), "instantiate graph");
        checkCuda(cudaGraphUpload(executable, stream), "upload graph");
    };
    if (graphSteps > 0) captureSteps(graphSteps, graph, executableGraph);
    // Batch the remainder too, including a single step or an odd final step.
    if (tailSteps > 0) captureSteps(tailSteps, tailGraph, executableTail);
    checkCuda(cudaStreamSynchronize(stream), "initialize fields");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::steady_clock::now();
    
    for (int replay = 0; replay < graphReplays; ++replay) {
        checkCuda(cudaGraphLaunch(executableGraph, stream), "launch simulation graph");
    }
    if (tailSteps > 0) {
        checkCuda(cudaGraphLaunch(executableTail, stream), "launch final steps");
        if (tailSteps & 1) std::swap(cold, cnew);
    }
    checkCuda(cudaGetLastError(), "launch stencil kernels");
    checkCuda(cudaStreamSynchronize(stream), "run simulation");
    
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Only transfer the final field when it is needed for output or validation.
    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "copy result");
    }
    if (executableGraph) checkCuda(cudaGraphExecDestroy(executableGraph), "destroy executable graph");
    if (graph) checkCuda(cudaGraphDestroy(graph), "destroy graph");
    if (executableTail) checkCuda(cudaGraphExecDestroy(executableTail), "destroy final executable graph");
    if (tailGraph) checkCuda(cudaGraphDestroy(tailGraph), "destroy final graph");
    checkCuda(cudaFree(storage), "free fields");
    checkCuda(cudaStreamDestroy(stream), "destroy stream");
    
    // Print results for external validation
    if (printResults) {
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, nx, ny, nz);
        
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
