#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

inline void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ double laplacian(const double* __restrict__ a, size_t idx,
        size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz,
        double idx2, double idy2, double idz2) {
    const size_t plane = nx * ny;
    const size_t xp = idx + (x + 1 < nx), xn = idx - (x > 0);
    const size_t yp = idx + (y + 1 < ny) * nx, yn = idx - (y > 0) * nx;
    const size_t zp = idx + (z + 1 < nz) * plane, zn = idx - (z > 0) * plane;
    const double center = a[idx];
    const double xx = (a[xp] + a[xn] - 2.0 * center) * idx2;
    const double yy = (a[yp] + a[yn] - 2.0 * center) * idy2;
    const double zz = (a[zp] + a[zn] - 2.0 * center) * idz2;
    return xx + yy + zz;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
        size_t nx, size_t ny, size_t nz, double idx2, double idy2, double idz2,
        double gamma, double eAA, double eBB, double eAB, size_t count) {
    const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    const size_t x = idx % nx, y = (idx / nx) % ny, z = idx / (nx * ny);
    const double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
            + 3.0 * cv + cv * cv * cv
            - gamma * laplacian(c, idx, x, y, z, nx, ny, nz, idx2, idy2, idz2);
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
        const double* __restrict__ mu, size_t nx, size_t ny, size_t nz,
        double idx2, double idy2, double idz2, double scale, size_t count) {
    const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    const size_t x = idx % nx, y = (idx / nx) % ny, z = idx / (nx * ny);
    cnew[idx] = cold[idx] + scale * laplacian(mu, idx, x, y, z, nx, ny, nz, idx2, idy2, idz2);
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = z * (nx * ny) + y * nx + x;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
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
    
    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate host and device arrays.
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    double *deviceCold = nullptr, *deviceNew = nullptr, *deviceMu = nullptr;
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceCold), gridSize * sizeof(double)), "allocating concentration");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceNew), gridSize * sizeof(double)), "allocating next concentration");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceMu), gridSize * sizeof(double)), "allocating chemical potential");
    cudaCheck(cudaMemcpy(deviceCold, cold.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice), "copying initial concentration");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    cudaCheck(cudaDeviceSynchronize(), "waiting before simulation");
    auto start = std::chrono::high_resolution_clock::now();

    constexpr unsigned int blockSize = 256;
    const unsigned int gridBlocks = static_cast<unsigned int>((gridSize + blockSize - 1) / blockSize);
    
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<gridBlocks, blockSize>>>(deviceCold, deviceMu, nx, ny, nz,
            1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz), gamma, e_AA, e_BB, e_AB, gridSize);
        cudaCheck(cudaGetLastError(), "launching chemical potential kernel");
        updateKernel<<<gridBlocks, blockSize>>>(deviceNew, deviceCold, deviceMu, nx, ny, nz,
            1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz), dt * D, gridSize);
        cudaCheck(cudaGetLastError(), "launching concentration update kernel");
        std::swap(deviceCold, deviceNew);
    }
    
    cudaCheck(cudaDeviceSynchronize(), "completing simulation");
    auto end = std::chrono::high_resolution_clock::now();
    cudaCheck(cudaMemcpy(cold.data(), deviceCold, gridSize * sizeof(double), cudaMemcpyDeviceToHost), "copying final concentration");
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        cudaFree(deviceCold);
        cudaFree(deviceNew);
        cudaFree(deviceMu);
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    cudaFree(deviceCold);
    cudaFree(deviceNew);
    cudaFree(deviceMu);
    
    return 0;
}
