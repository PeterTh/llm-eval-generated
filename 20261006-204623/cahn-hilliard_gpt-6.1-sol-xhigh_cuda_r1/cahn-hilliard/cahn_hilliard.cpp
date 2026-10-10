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

namespace {

constexpr double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
constexpr double gamma = 0.5, D = 1.0;
constexpr unsigned int blockX = 32, planesPerBlock = 4;

struct KernelLaunch {
    dim3 grid;
    dim3 block;
    bool linear;
};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// Own resources on normal and validation-failure return paths.
struct CudaResources {
    double* a = nullptr;
    double* b = nullptr;
    double* mu = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraph_t tailGraph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaGraphExec_t tailExecutable = nullptr;

    ~CudaResources() {
        if (tailExecutable) cudaGraphExecDestroy(tailExecutable);
        if (executable) cudaGraphExecDestroy(executable);
        if (tailGraph) cudaGraphDestroy(tailGraph);
        if (graph) cudaGraphDestroy(graph);
        if (mu) cudaFree(mu);
        if (b) cudaFree(b);
        if (a) cudaFree(a);
        if (stream) cudaStreamDestroy(stream);
    }
};

__device__ __forceinline__ double laplacian(
    double center, double xp, double xn, double yp, double yn, double zp, double zn) {
    const double cxx = (xp + xn - 2.0 * center) / (dx * dx);
    const double cyy = (yp + yn - 2.0 * center) / (dy * dy);
    const double czz = (zp + zn - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

__device__ __forceinline__ double chemicalPotential(double center, double lap) {
    return 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB
                  - 2.0 * center * e_AB)
           + 3.0 * center + center * center * center - gamma * lap;
}

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * 1299709) % volume) / static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Pack narrow grids into full warps rather than leaving most X lanes idle.
template <bool potential>
__global__ void linearStencilPass(const double* __restrict__ input,
                                  const double* __restrict__ oldConcentration,
                                  double* __restrict__ output,
                                  size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const size_t volume = plane * nz;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = i / nx;
        const size_t x = i - row * nx;
        const size_t z = row / ny;
        const size_t y = row - z * ny;
        const double center = input[i];
        const double lap = laplacian(center,
            input[i + (x + 1 < nx ? 1 : 0)], input[i - (x > 0 ? 1 : 0)],
            input[i + (y + 1 < ny ? nx : 0)], input[i - (y > 0 ? nx : 0)],
            input[i + (z + 1 < nz ? plane : 0)], input[i - (z > 0 ? plane : 0)]);
        if constexpr (potential) {
            output[i] = chemicalPotential(center, lap);
        } else {
            output[i] = oldConcentration[i] + dt * D * lap;
        }
    }
}

// Each thread streams several Z planes, retaining neighboring values in
// registers. Adjacent lanes access adjacent X cells in both stencil passes.
template <bool potential>
__global__ void stencilPass(const double* __restrict__ input,
                            const double* __restrict__ oldConcentration,
                            double* __restrict__ output,
                            size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockX + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    const size_t xMinus = x > 0 ? 1 : 0;
    const size_t xPlus = x + 1 < nx ? 1 : 0;
    for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
        const size_t yMinus = y > 0 ? nx : 0;
        const size_t yPlus = y + 1 < ny ? nx : 0;
        for (size_t firstZ = static_cast<size_t>(blockIdx.z) * planesPerBlock;
             firstZ < nz; firstZ += static_cast<size_t>(gridDim.z) * planesPerBlock) {
            const size_t lastZ = firstZ + planesPerBlock < nz ? firstZ + planesPerBlock : nz;
            size_t index = firstZ * plane + y * nx + x;
            double center = input[index];
            double previous = input[index - (firstZ > 0 ? plane : 0)];
            double next = input[index + (firstZ + 1 < nz ? plane : 0)];
            for (size_t z = firstZ; z < lastZ; ++z) {
                const double lap = laplacian(center, input[index + xPlus],
                                            input[index - xMinus], input[index + yPlus],
                                            input[index - yMinus], next, previous);
                if constexpr (potential) {
                    output[index] = chemicalPotential(center, lap);
                } else {
                    output[index] = oldConcentration[index] + dt * D * lap;
                }
                if (z + 1 < lastZ) {
                    index += plane;
                    previous = center;
                    center = next;
                    next = input[index + (z + 2 < nz ? plane : 0)];
                }
            }
        }
    }
}

void launchStep(double*& cold, double*& cnew, double* mu,
                size_t nx, size_t ny, size_t nz, const KernelLaunch& launch, cudaStream_t stream) {
    if (launch.linear) {
        linearStencilPass<true><<<launch.grid, launch.block, 0, stream>>>(cold, nullptr, mu, nx, ny, nz);
    } else {
        stencilPass<true><<<launch.grid, launch.block, 0, stream>>>(cold, nullptr, mu, nx, ny, nz);
    }
    CUDA_CHECK(cudaGetLastError());
    // Stream ordering enforces the global dependency between the two passes.
    if (launch.linear) {
        linearStencilPass<false><<<launch.grid, launch.block, 0, stream>>>(mu, cold, cnew, nx, ny, nz);
    } else {
        stencilPass<false><<<launch.grid, launch.block, 0, stream>>>(mu, cold, cnew, nx, ny, nz);
    }
    CUDA_CHECK(cudaGetLastError());
    std::swap(cold, cnew);
}

void captureSteps(cudaGraph_t& graph, cudaGraphExec_t& executable,
                  int steps, double*& cold, double*& cnew, double* mu,
                  size_t nx, size_t ny, size_t nz, const KernelLaunch& launch, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int t = 0; t < steps; ++t) {
        launchStep(cold, cnew, mu, nx, ny, nz, launch, stream);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(executable, stream));
}

} // namespace

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
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (nx == 0 || nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or overflowing grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    CudaResources cuda;
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    CUDA_CHECK(cudaStreamCreateWithFlags(&cuda.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&cuda.a, bytes));
    CUDA_CHECK(cudaMalloc(&cuda.b, bytes));
    CUDA_CHECK(cudaMalloc(&cuda.mu, bytes));
    double* cold = cuda.a;
    double* cnew = cuda.b;

    const unsigned int initializationBlocks = static_cast<unsigned int>(
        std::min(1 + (gridSize - 1) / 256,
                 static_cast<size_t>(properties.multiProcessorCount) * 32));
    KernelLaunch launch{};
    launch.linear = nx < blockX / 2;
    if (launch.linear) {
        launch.block = dim3(256);
        launch.grid = dim3(initializationBlocks);
    } else {
        const unsigned int rowsPerBlock = gridSize >= 1024 * 1024 ? 8 : 4;
        launch.block = dim3(blockX, rowsPerBlock);
        const size_t xBlocks = 1 + (nx - 1) / blockX;
        if (xBlocks > static_cast<size_t>(properties.maxGridSize[0])) {
            fprintf(stderr, "Grid X dimension exceeds CUDA launch limits\n");
            return 1;
        }
        launch.grid = dim3(static_cast<unsigned int>(xBlocks),
            static_cast<unsigned int>(std::min(1 + (ny - 1) / rowsPerBlock,
                                              static_cast<size_t>(properties.maxGridSize[1]))),
            static_cast<unsigned int>(std::min(1 + (nz - 1) / planesPerBlock,
                                              static_cast<size_t>(properties.maxGridSize[2]))));
    }

    printf("Initializing concentration field...\n");
    initializeConcentration<<<initializationBlocks, 256, 0, cuda.stream>>>(cold, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream));

    // Even batches return to the same buffer pair and can be replayed. Keep
    // graph size bounded independently of the number of simulation steps.
    const int batchSteps = iterations > 0 ? std::min(iterations, 32) : 0;
    const int batches = batchSteps > 0 ? iterations / batchSteps : 0;
    const int tailSteps = batchSteps > 0 ? iterations % batchSteps : 0;
    if (batchSteps > 0) {
        captureSteps(cuda.graph, cuda.executable, batchSteps, cold, cnew, cuda.mu,
                     nx, ny, nz, launch, cuda.stream);
    }
    if (tailSteps > 0) {
        captureSteps(cuda.tailGraph, cuda.tailExecutable, tailSteps, cold, cnew, cuda.mu,
                     nx, ny, nz, launch, cuda.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream));

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::steady_clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        CUDA_CHECK(cudaGraphLaunch(cuda.executable, cuda.stream));
    }
    if (tailSteps > 0) {
        CUDA_CHECK(cudaGraphLaunch(cuda.tailExecutable, cuda.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream));
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy only the final field, and only if host-side output is requested.
    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        CUDA_CHECK(cudaMemcpyAsync(result.data(), cold, bytes, cudaMemcpyDeviceToHost, cuda.stream));
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream));
    }

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
