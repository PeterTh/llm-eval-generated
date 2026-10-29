#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Each pass reads a complete field and writes a separate field. Kernel launch
// ordering in the default stream supplies the required time-step barriers.
static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__global__ void initialize(double* c, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += size_t(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * size_t(1299709)) % n) / static_cast<double>(n);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

__device__ inline double laplacian(const double* __restrict__ field, size_t i,
                                    size_t x, size_t y, size_t z,
                                    size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = field[i];
    const double cxx = field[i + (x + 1 < nx ? 1 : 0)] + field[i - (x > 0 ? 1 : 0)] - 2.0 * center;
    const double cyy = field[i + (y + 1 < ny ? nx : 0)] + field[i - (y > 0 ? nx : 0)] - 2.0 * center;
    const double czz = field[i + (z + 1 < nz ? plane : 0)] + field[i - (z > 0 ? plane : 0)] - 2.0 * center;
    return cxx + cyy + czz;
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t nz, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += size_t(blockDim.x) * gridDim.x) {
        const size_t x = i % nx;
        const size_t y = (i / nx) % ny;
        const size_t z = i / (nx * ny);
        const double cv = c[i];
        const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
        mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c, i, x, y, z, nx, ny, nz);
    }
}

__global__ void update(double* __restrict__ next, const double* __restrict__ c,
                       const double* __restrict__ mu, size_t nx, size_t ny, size_t nz, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += size_t(blockDim.x) * gridDim.x) {
        const size_t x = i % nx;
        const size_t y = (i / nx) % ny;
        const size_t z = i / (nx * ny);
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
    
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Invalid or excessively large grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;

    double *cold, *cnew, *mu;
    const size_t bytes = gridSize * sizeof(double);
    cudaCheck(cudaMalloc(&cold, bytes), "allocating concentration");
    cudaCheck(cudaMalloc(&cnew, bytes), "allocating next concentration");
    cudaCheck(cudaMalloc(&mu, bytes), "allocating chemical potential");
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((gridSize - 1) / threads + 1, 65535));

    printf("Initializing concentration field...\n");
    initialize<<<blocks, threads>>>(cold, gridSize);
    cudaCheck(cudaGetLastError(), "launching initialization");
    cudaCheck(cudaDeviceSynchronize(), "initializing concentration");

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotential<<<blocks, threads>>>(cold, mu, nx, ny, nz, gridSize);
        update<<<blocks, threads>>>(cnew, cold, mu, nx, ny, nz, gridSize);
        std::swap(cold, cnew);
    }
    cudaCheck(cudaGetLastError(), "launching simulation");
    cudaCheck(cudaDeviceSynchronize(), "running simulation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        cudaCheck(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "copying result");
    }
    cudaCheck(cudaFree(cold), "freeing concentration");
    cudaCheck(cudaFree(cnew), "freeing next concentration");
    cudaCheck(cudaFree(mu), "freeing chemical potential");

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
