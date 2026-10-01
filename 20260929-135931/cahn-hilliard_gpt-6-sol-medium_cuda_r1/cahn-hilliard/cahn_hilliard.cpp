#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Both stencil stages use the same clamped (zero-flux) boundary conditions.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             size_t i, size_t nx, size_t ny, size_t volume) {
    const size_t plane = nx * ny;
    const size_t x = i % nx;
    const size_t y = (i / nx) % ny;
    const double center = field[i];
    const double cxx = field[x + 1 < nx ? i + 1 : i] + field[x ? i - 1 : i] - 2.0 * center;
    const double cyy = field[y + 1 < ny ? i + nx : i] + field[y ? i - nx : i] - 2.0 * center;
    const double czz = field[i + plane < volume ? i + plane : i] + field[i >= plane ? i - plane : i] - 2.0 * center;
    return cxx + cyy + czz;
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t volume) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < volume; i += stride) {
        const double cv = c[i];
        const double e_AA = -(2.0 / 9.0);
        const double e_BB = -(2.0 / 9.0);
        const double e_AB = 2.0 / 9.0;
        mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c, i, nx, ny, volume);
    }
}

__global__ void concentrationUpdate(const double* __restrict__ old, const double* __restrict__ mu,
                                    double* __restrict__ next, size_t nx, size_t ny, size_t volume) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < volume; i += stride) {
        next[i] = old[i] + 0.01 * laplacian(mu, i, nx, ny, volume);
    }
}

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// Preserve the original deterministic initialization for every linear cell index.
__global__ void initializeConcentration(double* c, size_t volume) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < volume; i += stride) {
        const double pseudo = (((i + 1) * 1299709) % volume) / static_cast<double>(volume);
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
    
    size_t gridSize = nx * ny * nz;
    
    // Keep initialization and all time-step data on the GPU.
    double *deviceOld = nullptr, *deviceNew = nullptr, *deviceMu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceOld), bytes), "cudaMalloc concentration");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceNew), bytes), "cudaMalloc next concentration");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceMu), bytes), "cudaMalloc chemical potential");
    constexpr unsigned int blockSize = 256;
    const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>((gridSize + blockSize - 1) / blockSize, 65535));
    printf("Initializing concentration field...\n");
    initializeConcentration<<<blocks, blockSize>>>(deviceOld, gridSize);
    checkCuda(cudaGetLastError(), "CUDA initialization launch");
    checkCuda(cudaDeviceSynchronize(), "CUDA initialization");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        chemicalPotential<<<blocks, blockSize>>>(deviceOld, deviceMu, nx, ny, gridSize);
        concentrationUpdate<<<blocks, blockSize>>>(deviceOld, deviceMu, deviceNew, nx, ny, gridSize);
        std::swap(deviceOld, deviceNew);
    }
    checkCuda(cudaGetLastError(), "CUDA stencil launch");
    checkCuda(cudaDeviceSynchronize(), "CUDA stencil execution");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / elapsedSeconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), deviceOld, bytes, cudaMemcpyDeviceToHost), "download concentration");
    }
    checkCuda(cudaFree(deviceOld), "cudaFree concentration");
    checkCuda(cudaFree(deviceNew), "cudaFree next concentration");
    checkCuda(cudaFree(deviceMu), "cudaFree chemical potential");
    
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
