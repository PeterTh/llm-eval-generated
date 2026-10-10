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

// All simulation storage stays on the device until results are requested.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * 1299709) % volume) / static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// A warp spans contiguous X cells. Each thread rolls through four Z planes,
// retaining its Z neighbors in registers between planes. The two passes share
// this stencil implementation and preserve the original arithmetic order.
template <bool chemical>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ concentration,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t plane = nx * ny;
    const size_t firstZ = static_cast<size_t>(blockIdx.z) * 4;
    size_t i = firstZ * plane + y * nx + x;
    const size_t xm = x > 0 ? 1 : 0;
    const size_t xp = x + 1 < nx ? 1 : 0;
    const size_t ym = y > 0 ? nx : 0;
    const size_t yp = y + 1 < ny ? nx : 0;
    double center = input[i];
    double previous = input[firstZ > 0 ? i - plane : i];
    for (size_t z = firstZ; z < nz && z < firstZ + 4; ++z, i += plane) {
        const double next = input[z + 1 < nz ? i + plane : i];
        const double cxx = input[i + xp] + input[i - xm] - 2.0 * center;
        const double cyy = input[i + yp] + input[i - ym] - 2.0 * center;
        const double czz = next + previous - 2.0 * center;
        const double laplacian = cxx + cyy + czz;
        if constexpr (chemical) {
            constexpr double e_AA = -(2.0 / 9.0);
            constexpr double e_BB = -(2.0 / 9.0);
            constexpr double e_AB = 2.0 / 9.0;
            output[i] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB
                              - 2.0 * center * e_AB)
                      + 3.0 * center + center * center * center - 0.5 * laplacian;
        } else {
            output[i] = concentration[i] + 0.01 * laplacian;
        }
        previous = center;
        center = next;
    }
}

void launchStep(const double* oldField, double* newField, double* potential,
                size_t nx, size_t ny, size_t nz, dim3 blocks, dim3 threads,
                cudaStream_t stream) {
    stencil<true><<<blocks, threads, 0, stream>>>(oldField, nullptr, potential, nx, ny, nz);
    stencil<false><<<blocks, threads, 0, stream>>>(potential, oldField, newField, nx, ny, nz);
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
    
    if (nx == 0 || nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or overflowing grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    CUDA_CHECK(cudaSetDevice(0));
    const dim3 threads(32, 4);
    const size_t bx = (nx + threads.x - 1) / threads.x;
    const size_t by = (ny + threads.y - 1) / threads.y;
    const size_t bz = (nz + 3) / 4;
    if (bx > static_cast<size_t>(device.maxGridSize[0]) ||
        by > static_cast<size_t>(device.maxGridSize[1]) ||
        bz > static_cast<size_t>(device.maxGridSize[2])) {
        fprintf(stderr, "Grid dimensions exceed CUDA launch limits\n");
        return 1;
    }
    const dim3 blocks(static_cast<unsigned>(bx), static_cast<unsigned>(by),
                      static_cast<unsigned>(bz));
    double *fieldA, *fieldB, *potential;
    CUDA_CHECK(cudaMalloc(&fieldA, bytes));
    CUDA_CHECK(cudaMalloc(&fieldB, bytes));
    CUDA_CHECK(cudaMalloc(&potential, bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Initializing concentration field...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize + 255) / 256, 65535));
    initializeConcentration<<<initBlocks, 256, 0, stream>>>(fieldA, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Two steps restore the pointer ordering, so one small graph can be replayed
    // for any iteration count without building a graph proportional to it.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    if (iterations >= 2) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launchStep(fieldA, fieldB, potential, nx, ny, nz, blocks, threads, stream);
        launchStep(fieldB, fieldA, potential, nx, ny, nz, blocks, threads, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executableGraph, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations / 2; ++t) {
        CUDA_CHECK(cudaGraphLaunch(executableGraph, stream));
    }
    double* finalField = fieldA;
    if (iterations > 0 && iterations % 2 != 0) {
        launchStep(fieldA, fieldB, potential, nx, ny, nz, blocks, threads, stream);
        CUDA_CHECK(cudaGetLastError());
        finalField = fieldB;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / seconds / 1e6);

    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), finalField, bytes, cudaMemcpyDeviceToHost));
    }
    if (executableGraph) CUDA_CHECK(cudaGraphExecDestroy(executableGraph));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(potential));
    CUDA_CHECK(cudaFree(fieldB));
    CUDA_CHECK(cudaFree(fieldA));

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
