#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

inline void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

__device__ __forceinline__ double laplacian(const double* __restrict__ a, size_t i,
        size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const size_t x = i % nx, y = (i / nx) % ny, z = i / plane;
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz), zn = z - (z > 0);
    const double v = a[i];
    return (a[z * plane + y * nx + xp] + a[z * plane + y * nx + xn] - 2.0 * v)
         + (a[z * plane + yp * nx + x] + a[z * plane + yn * nx + x] - 2.0 * v)
         + (a[zp * plane + y * nx + x] + a[zn * plane + y * nx + x] - 2.0 * v);
}

__global__ void initializeKernel(double* c, size_t n, size_t volume) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const size_t pseudo = (((i + 1) * 1299709) % volume);
        c[i] = -1.0 + 2.0 * (static_cast<double>(pseudo) / static_cast<double>(volume));
    }
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
        size_t n, size_t nx, size_t ny, size_t nz) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const double v = c[i];
        // These constants are the original benchmark parameters (dx=dy=dz=1).
        mu[i] = 4.5 * ((v + 1.0) * (-2.0 / 9.0) + (v - 1.0) * (-2.0 / 9.0)
                       - 2.0 * v * (2.0 / 9.0)) + 3.0 * v + v * v * v
                - 0.5 * laplacian(c, i, nx, ny, nz);
    }
}

__global__ void updateKernel(double* __restrict__ next, const double* __restrict__ current,
        const double* __restrict__ mu, size_t n, size_t nx, size_t ny, size_t nz) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) next[i] = current[i] + 0.01 * laplacian(mu, i, nx, ny, nz);
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
    
    // Physical parameters
    size_t gridSize = nx * ny * nz;
    
    // Keep all evolving fields on the GPU; only copy the final field back.
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    cudaCheck(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    cudaCheck(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    cudaCheck(cudaMalloc(&d_mu, gridSize * sizeof(double)));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((gridSize + threads - 1) / threads);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeKernel<<<blocks, threads>>>(d_cold, gridSize, gridSize);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    std::vector<double> cold(gridSize);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventRecord(start));
    
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, threads>>>(d_cold, d_mu, gridSize, nx, ny, nz);
        updateKernel<<<blocks, threads>>>(d_cnew, d_cold, d_mu, gridSize, nx, ny, nz);
        cudaCheck(cudaGetLastError());
        std::swap(d_cold, d_cnew);
    }
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end));
    float elapsedMs = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsedMs, start, end));
    const long durationMs = static_cast<long>(elapsedMs);
    cudaCheck(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);

    printf("Computation time: %ld ms\n", durationMs);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
