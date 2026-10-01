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

// Fail explicitly on CUDA errors; execution never falls back to the CPU.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// Adjacent lanes access adjacent X cells. Neighbor loads are shared through
// the GPU caches; each pass writes its output exactly once.
__device__ __forceinline__ double computeLaplacian(
        const double* __restrict__ c, size_t i, size_t x, size_t y, size_t z,
        size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = 2.0 * c[i];
    const double cxx = c[i + (x + 1 < nx ? 1 : 0)]
                     + c[i - (x > 0 ? 1 : 0)] - center;
    const double cyy = c[i + (y + 1 < ny ? nx : 0)]
                     + c[i - (y > 0 ? nx : 0)] - center;
    const double czz = c[i + (z + 1 < nz ? plane : 0)]
                     + c[i - (z > 0 ? plane : 0)] - center;
    // The benchmark's fixed spacings dx, dy, dz are all 1.0.
    return cxx + cyy + czz;
}

// Specialize the two stages so the update does not evaluate the potential.
// A kernel boundary supplies the global barrier required between stages.
template<bool chemicalPotential>
__global__ void stencil(const double* __restrict__ source,
                        const double* __restrict__ cold,
                        double* __restrict__ destination,
                        size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    for (size_t z = size_t(blockIdx.z) * blockDim.z + threadIdx.z;
         z < nz; z += size_t(gridDim.z) * blockDim.z) {
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += size_t(gridDim.y) * blockDim.y) {
            const size_t i = (z * ny + y) * nx + x;
            const double laplacian = computeLaplacian(source, i, x, y, z, nx, ny, nz);
            if constexpr (chemicalPotential) {
                const double cv = source[i];
                constexpr double e_AA = -(2.0 / 9.0);
                constexpr double e_BB = -(2.0 / 9.0);
                constexpr double e_AB = 2.0 / 9.0;
                destination[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                                       - 2.0 * cv * e_AB)
                               + 3.0 * cv + cv * cv * cv - 0.5 * laplacian;
            } else {
                destination[i] = cold[i] + 0.01 * laplacian;
            }
        }
    }
}

__global__ void initializeConcentration(double* c, size_t vol) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vol; i += size_t(blockDim.x) * gridDim.x) {
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
    // Keep warps contiguous in X on normal grids, but distribute threads
    // over the remaining dimensions for thin and singleton grids.
    unsigned bx = 1, by = 1;
    const unsigned maxX = (ny == 1 && nz == 1) ? 256 : 64;
    while (bx < nx && bx < maxX) bx *= 2;
    while (by < ny && bx * by < 256) by *= 2;
    const dim3 block(bx, by, std::min(64u, 256 / (bx * by)));
    if ((nx - 1) / block.x + 1 > 2147483647ULL) {
        fprintf(stderr, "Grid X dimension exceeds CUDA launch limits\n");
        return 1;
    }
    const dim3 grid(static_cast<unsigned>((nx - 1) / block.x + 1),
                    static_cast<unsigned>(std::min<size_t>((ny - 1) / block.y + 1, 65535)),
                    static_cast<unsigned>(std::min<size_t>((nz - 1) / block.z + 1, 65535)));
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Initializing concentration field...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize - 1) / 256 + 1, 65535));
    initializeConcentration<<<initBlocks, 256, 0, stream>>>(cold, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto step = [&]() {
        stencil<true><<<grid, block, 0, stream>>>(cold, nullptr, mu, nx, ny, nz);
        stencil<false><<<grid, block, 0, stream>>>(mu, cold, cnew, nx, ny, nz);
        std::swap(cold, cnew);
    };

    // Capture an even number of steps so replay always starts with the same
    // buffer parity. Graph storage stays bounded regardless of step count.
    constexpr int batchSteps = 32;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    if (iterations >= 2 * batchSteps) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int t = 0; t < batchSteps; ++t) step();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executableGraph, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    int t = 0;
    if (executableGraph) {
        for (; t <= iterations - batchSteps; t += batchSteps)
            CUDA_CHECK(cudaGraphLaunch(executableGraph, stream));
    }
    for (; t < iterations; ++t) step();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Transfer only the final field, and only when it is requested.
    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost));
    }
    if (executableGraph) CUDA_CHECK(cudaGraphExecDestroy(executableGraph));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(cold));

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
