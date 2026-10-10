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

namespace {
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// The benchmark uses unit spacing. Keep the original evaluation order and
// constants, rather than simplifying the chemical potential polynomial.
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = 2.0 / 9.0;
constexpr double gamma = 0.5;
constexpr double dt = 0.01;
constexpr unsigned planesPerTile = 4;

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += size_t(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * size_t(1299709)) % volume) /
                              static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Adjacent lanes traverse X, so every stencil direction is coalesced. Each
// thread processes a short Z column, retaining the previous/current planes
// in registers. Separate launches supply the global barrier between passes.
template <bool chemical>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ cold,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    const size_t xm = x > 0 ? 1 : 0;
    const size_t xp = x + 1 < nx ? 1 : 0;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny; y += size_t(gridDim.y) * blockDim.y) {
        const size_t ym = y > 0 ? nx : 0;
        const size_t yp = y + 1 < ny ? nx : 0;
        for (size_t zbase = size_t(blockIdx.z) * planesPerTile;
             zbase < nz; zbase += size_t(gridDim.z) * planesPerTile) {
            size_t i = zbase * plane + y * nx + x;
            double center = input[i];
            double previous = input[i - (zbase > 0 ? plane : 0)];
            #pragma unroll
            for (unsigned k = 0; k < planesPerTile; ++k) {
                const size_t z = zbase + k;
                if (z >= nz) break;
                const double next = input[i + (z + 1 < nz ? plane : 0)];
                const double cxx = input[i + xp] + input[i - xm] - 2.0 * center;
                const double cyy = input[i + yp] + input[i - ym] - 2.0 * center;
                const double czz = next + previous - 2.0 * center;
                const double laplacian = cxx + cyy + czz;
                if constexpr (chemical) {
                    output[i] = 4.5 * ((center + 1.0) * e_AA +
                                      (center - 1.0) * e_BB - 2.0 * center * e_AB)
                              + 3.0 * center + center * center * center
                              - gamma * laplacian;
                } else {
                    output[i] = cold[i] + dt * laplacian;
                }
                previous = center;
                center = next;
                i += plane;
            }
        }
    }
}

void launchStep(double* cold, double* cnew, double* mu,
                size_t nx, size_t ny, size_t nz, dim3 blocks, dim3 threads,
                cudaStream_t stream) {
    stencil<true><<<blocks, threads, 0, stream>>>(cold, nullptr, mu, nx, ny, nz);
    stencil<false><<<blocks, threads, 0, stream>>>(mu, cold, cnew, nx, ny, nz);
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
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / (3 * sizeof(double))) {
        fprintf(stderr, "Invalid or oversized grid\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double* storage = nullptr;
    CUDA_CHECK(cudaMalloc(&storage, 3 * bytes));
    double* cold = storage;
    double* cnew = storage + gridSize;
    double* mu = storage + 2 * gridSize;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const dim3 threads(32, 4);
    const dim3 blocks(static_cast<unsigned>((nx + 31) / 32),
                      static_cast<unsigned>(std::min(size_t(65535), (ny + 3) / 4)),
                      static_cast<unsigned>(std::min(size_t(65535), (nz + 3) / 4)));

    printf("Initializing concentration field...\n");
    const unsigned initBlocks = static_cast<unsigned>(
        std::min(size_t(65535), (gridSize + 255) / 256));
    initializeConcentration<<<initBlocks, 256, 0, stream>>>(cold, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // A pair of steps returns to the original buffer arrangement. Replaying
    // this graph avoids per-step host synchronization and launch overhead.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    if (iterations >= 4) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launchStep(cold, cnew, mu, nx, ny, nz, blocks, threads, stream);
        launchStep(cnew, cold, mu, nx, ny, nz, blocks, threads, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executableGraph, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::steady_clock::now();
    int t = 0;
    if (executableGraph) {
        for (; t + 1 < iterations; t += 2) {
            CUDA_CHECK(cudaGraphLaunch(executableGraph, stream));
        }
    }
    for (; t < iterations; ++t) {
        launchStep(cold, cnew, mu, nx, ny, nz, blocks, threads, stream);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / seconds / 1e6);

    // Transfer only the final field, and only when host output is requested.
    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost));
    }
    if (executableGraph) CUDA_CHECK(cudaGraphExecDestroy(executableGraph));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(storage));

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
