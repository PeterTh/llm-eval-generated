#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void initialize(double* c, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x) {
        double pseudo = (((i + 1) * (size_t)1299709) % n) / static_cast<double>(n);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

__device__ __forceinline__ double laplacian(const double* a, size_t i, size_t x,
                                            size_t y, size_t z, size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = a[i];
    const double xx = a[x + 1 < nx ? i + 1 : i] + a[x ? i - 1 : i] - 2.0 * center;
    const double yy = a[y + 1 < ny ? i + nx : i] + a[y ? i - nx : i] - 2.0 * center;
    const double zz = a[z + 1 < nz ? i + plane : i] + a[z ? i - plane : i] - 2.0 * center;
    return xx + yy + zz;
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t nz, size_t n) {
    const size_t plane = nx * ny;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x) {
        const size_t z = i / plane, y = (i % plane) / nx, x = i % nx;
        const double cv = c[i];
        mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c, i, x, y, z, nx, ny, nz);
    }
}

__global__ void update(double* __restrict__ next, const double* __restrict__ c,
                       const double* __restrict__ mu, size_t nx, size_t ny, size_t nz, size_t n) {
    const size_t plane = nx * ny;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x) {
        const size_t z = i / plane, y = (i % plane) / nx, x = i % nx;
        next[i] = c[i] + 0.01 * laplacian(mu, i, x, y, z, nx, ny, nz);
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
    
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny * nz > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Invalid grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double *cold, *cnew, *mu;
    checkCuda(cudaMalloc(&cold, bytes), "allocating concentration");
    checkCuda(cudaMalloc(&cnew, bytes), "allocating next concentration");
    checkCuda(cudaMalloc(&mu, bytes), "allocating chemical potential");
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min((gridSize - 1) / threads + 1, (size_t)65535));

    printf("Initializing concentration field...\n");
    initialize<<<blocks, threads>>>(cold, gridSize);
    checkCuda(cudaGetLastError(), "launching initialization");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration");

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotential<<<blocks, threads>>>(cold, mu, nx, ny, nz, gridSize);
        update<<<blocks, threads>>>(cnew, cold, mu, nx, ny, nz, gridSize);
        std::swap(cold, cnew);
    }
    checkCuda(cudaGetLastError(), "launching simulation");
    checkCuda(cudaDeviceSynchronize(), "running simulation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "copying results");
    }
    checkCuda(cudaFree(cold), "freeing concentration");
    checkCuda(cudaFree(cnew), "freeing next concentration");
    checkCuda(cudaFree(mu), "freeing chemical potential");

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
