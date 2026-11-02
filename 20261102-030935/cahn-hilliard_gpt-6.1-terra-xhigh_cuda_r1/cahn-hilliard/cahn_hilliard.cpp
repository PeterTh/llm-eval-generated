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

// CUDA errors are fatal: never silently fall back to a different implementation.
static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Each warp traverses a contiguous portion of a row. The second grid
// dimension traverses all (y,z) rows, including grids taller than 65535 blocks.
__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += size_t(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * size_t(1299709)) % volume) /
                              static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

__device__ __forceinline__ double laplacian(const double* field, size_t i,
                                            size_t x, size_t y, size_t z,
                                            size_t nx, size_t ny, size_t nz,
                                            size_t plane) {
    const double center = field[i];
    const double xx = field[i + (x + 1 < nx ? 1 : 0)] +
                      field[i - (x > 0 ? 1 : 0)] - 2.0 * center;
    const double yy = field[i + (y + 1 < ny ? nx : 0)] +
                      field[i - (y > 0 ? nx : 0)] - 2.0 * center;
    const double zz = field[i + (z + 1 < nz ? plane : 0)] +
                      field[i - (z > 0 ? plane : 0)] - 2.0 * center;
    return xx + yy + zz;
}

__global__ void computeChemicalPotential(const double* __restrict__ c,
                                          double* __restrict__ mu,
                                          size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         row < ny * nz; row += size_t(gridDim.y) * blockDim.y) {
        const size_t z = ny * nz <= 0xffffffffull
            ? static_cast<unsigned int>(row) / static_cast<unsigned int>(ny)
            : row / ny;
        const size_t y = row - z * ny;
        const size_t i = row * nx + x;
        const double cv = c[i];
        constexpr double e_AA = -(2.0 / 9.0);
        constexpr double e_BB = -(2.0 / 9.0);
        constexpr double e_AB = (2.0 / 9.0);
        mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - 0.5 * laplacian(c, i, x, y, z, nx, ny, nz, plane);
    }
}

__global__ void cahnHilliardUpdate(double* __restrict__ cnew,
                                    const double* __restrict__ cold,
                                    const double* __restrict__ mu,
                                    size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         row < ny * nz; row += size_t(gridDim.y) * blockDim.y) {
        const size_t z = ny * nz <= 0xffffffffull
            ? static_cast<unsigned int>(row) / static_cast<unsigned int>(ny)
            : row / ny;
        const size_t y = row - z * ny;
        const size_t i = row * nx + x;
        cnew[i] = cold[i] + 0.01 * laplacian(mu, i, x, y, z, nx, ny, nz, plane);
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
        fprintf(stderr, "Invalid or too large grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    checkCuda(cudaMalloc(&cold, bytes), "Allocating concentration field");
    checkCuda(cudaMalloc(&cnew, bytes), "Allocating updated field");
    checkCuda(cudaMalloc(&mu, bytes), "Allocating chemical potential");

    printf("Initializing concentration field...\n");
    const unsigned int initBlocks = static_cast<unsigned int>(std::min<size_t>((gridSize - 1) / 256 + 1, 65535));
    initializeConcentration<<<initBlocks, 256>>>(cold, gridSize);
    checkCuda(cudaGetLastError(), "Launching initialization");
    checkCuda(cudaDeviceSynchronize(), "Initializing concentration field");

    const dim3 block(32, 4);
    const size_t xBlocks = (nx - 1) / block.x + 1;
    if (xBlocks > 2147483647u) {
        fprintf(stderr, "Grid size exceeds CUDA launch limits\n");
        return 1;
    }
    const dim3 grid(static_cast<unsigned int>(xBlocks),
                    static_cast<unsigned int>(std::min<size_t>((ny * nz - 1) / block.y + 1, 65535)));
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential<<<grid, block>>>(cold, mu, nx, ny, nz);
        cahnHilliardUpdate<<<grid, block>>>(cnew, cold, mu, nx, ny, nz);
        std::swap(cold, cnew);
    }
    checkCuda(cudaGetLastError(), "Launching simulation");
    checkCuda(cudaDeviceSynchronize(), "Running simulation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", duration.count());

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "Copying results");
    }
    checkCuda(cudaFree(cold), "Freeing concentration field");
    checkCuda(cudaFree(cnew), "Freeing updated field");
    checkCuda(cudaFree(mu), "Freeing chemical potential");

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
