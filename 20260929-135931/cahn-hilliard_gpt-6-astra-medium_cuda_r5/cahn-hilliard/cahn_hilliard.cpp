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

// Fail explicitly if CUDA is unavailable; the simulation has no CPU fallback.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

__device__ __forceinline__ double computeLaplacian(
    const double* __restrict__ c, size_t i, size_t x, size_t y, size_t z,
    size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = 2.0 * c[i];
    // The benchmark uses unit spacing in all three directions.
    const double cxx = (c[x + 1 < nx ? i + 1 : i] + c[x ? i - 1 : i] - center);
    const double cyy = (c[y + 1 < ny ? i + nx : i] + c[y ? i - nx : i] - center);
    const double czz = (c[z + 1 < nz ? i + plane : i] + c[z ? i - plane : i] - center);
    return cxx + cyy + czz;
}

// Warps follow the contiguous X dimension. Separate launches provide the global
// synchronization required between the two stencil passes.
template <bool chemicalPotential>
__global__ void stencil(const double* __restrict__ cold,
                        const double* __restrict__ input,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    for (size_t z = size_t(blockIdx.z) * blockDim.z + threadIdx.z;
         z < nz; z += size_t(gridDim.z) * blockDim.z) {
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += size_t(gridDim.y) * blockDim.y) {
            const size_t i = (z * ny + y) * nx + x;
            const double lap = computeLaplacian(input, i, x, y, z, nx, ny, nz);
            const double cv = cold[i];
            if constexpr (chemicalPotential) {
                constexpr double e_AA = -(2.0 / 9.0);
                constexpr double e_BB = -(2.0 / 9.0);
                constexpr double e_AB = (2.0 / 9.0);
                output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                          + 3.0 * cv + cv * cv * cv - 0.5 * lap;
            } else {
                output[i] = cv + 0.01 * lap;
            }
        }
    }
}

__global__ void initializeConcentration(double* c, size_t vol) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vol; i += size_t(gridDim.x) * blockDim.x) {
        const double pseudo = (((i + 1) * 1299709) % vol) / static_cast<double>(vol);
        c[i] = -1.0 + 2.0 * pseudo;
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
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    CUDA_CHECK(cudaSetDevice(0));
    const dim3 block(32, 4, 2);
    if ((nx + block.x - 1) / block.x > size_t(device.maxGridSize[0])) {
        fprintf(stderr, "X dimension exceeds CUDA grid limits\n");
        return 1;
    }
    const dim3 grid((nx + block.x - 1) / block.x,
                   std::min((ny + block.y - 1) / block.y, size_t(device.maxGridSize[1])),
                   std::min((nz + block.z - 1) / block.z, size_t(device.maxGridSize[2])));
    double *coldDevice, *cnewDevice, *muDevice;
    CUDA_CHECK(cudaMalloc(&coldDevice, bytes));
    CUDA_CHECK(cudaMalloc(&cnewDevice, bytes));
    CUDA_CHECK(cudaMalloc(&muDevice, bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Initializing concentration field...\n");
    initializeConcentration<<<std::min(size_t(65535), (gridSize + 255) / 256), 256, 0, stream>>>(coldDevice, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto step = [&]() {
        stencil<true><<<grid, block, 0, stream>>>(coldDevice, coldDevice, muDevice, nx, ny, nz);
        stencil<false><<<grid, block, 0, stream>>>(coldDevice, muDevice, cnewDevice, nx, ny, nz);
        std::swap(coldDevice, cnewDevice);
    };
    // An even number of steps restores the buffer addresses, so the graph can
    // be replayed without patching kernel arguments. Bound graph setup/storage
    // independently of the requested simulation length.
    const int graphSteps = std::min(32, iterations > 0 ? (iterations / 2) * 2 : 0);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    if (graphSteps > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int t = 0; t < graphSteps; ++t) step();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executableGraph, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    int t = 0;
    if (graphSteps > 0) {
        for (; t <= iterations - graphSteps; t += graphSteps)
            CUDA_CHECK(cudaGraphLaunch(executableGraph, stream));
    }
    for (; t < iterations; ++t) step();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (milliseconds / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Transfer only the final field, and only when host-side output needs it.
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), coldDevice, bytes, cudaMemcpyDeviceToHost));
    }
    if (executableGraph) CUDA_CHECK(cudaGraphExecDestroy(executableGraph));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(muDevice));
    CUDA_CHECK(cudaFree(cnewDevice));
    CUDA_CHECK(cudaFree(coldDevice));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
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
