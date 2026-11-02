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

// Each thread updates one cell. The y and z grid-stride loops also cover
// domains larger than CUDA's limits on those grid dimensions.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
        size_t i, size_t x, size_t y, size_t z,
        size_t nx, size_t ny, size_t nz, size_t plane) {
    const double center = field[i];
    const double cxx = field[i + (x + 1 < nx ? 1 : 0)]
                     + field[i - (x > 0 ? 1 : 0)] - 2.0 * center;
    const double cyy = field[i + (y + 1 < ny ? nx : 0)]
                     + field[i - (y > 0 ? nx : 0)] - 2.0 * center;
    const double czz = field[i + (z + 1 < nz ? plane : 0)]
                     + field[i - (z > 0 ? plane : 0)] - 2.0 * center;
    return cxx + cyy + czz;
}

__global__ void chemicalPotential(const double* __restrict__ c,
                                  double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            const size_t i = z * plane + y * nx + x;
            const double cv = c[i];
            const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
            mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv
                    - 0.5 * laplacian(c, i, x, y, z, nx, ny, nz, plane);
        }
    }
}

__global__ void updateConcentration(double* __restrict__ cnew,
                                    const double* __restrict__ cold,
                                    const double* __restrict__ mu,
                                    size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            const size_t i = z * plane + y * nx + x;
            cnew[i] = cold[i] + 0.01 * laplacian(mu, i, x, y, z, nx, ny, nz, plane);
        }
    }
}

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* c, size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    const size_t volume = plane * nz;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            const size_t i = z * plane + y * nx + x;
            const double pseudo = ((i + 1) * size_t{1299709} % volume) / static_cast<double>(volume);
            c[i] = -1.0 + 2.0 * pseudo;
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
    
    if (nx == 0 || ny > std::numeric_limits<size_t>::max() / nx ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny) ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or too large grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    std::vector<double> cold;

    printf("Initializing concentration field...\n");

    double *d_cold, *d_cnew, *d_mu;
    checkCuda(cudaMalloc(&d_cold, bytes), "allocating concentration");
    checkCuda(cudaMalloc(&d_cnew, bytes), "allocating next concentration");
    checkCuda(cudaMalloc(&d_mu, bytes), "allocating chemical potential");

    // Keep adjacent x coordinates in the same warp for coalesced accesses.
    unsigned int bx = 32;
    while (bx > 1 && bx / 2 >= nx) bx /= 2;
    const dim3 block(bx, 128 / bx);
    const size_t blocksX = (nx - 1) / block.x + 1;
    if (blocksX > 2147483647u) {
        fprintf(stderr, "Grid x dimension exceeds CUDA launch limits\n");
        return 1;
    }
    const dim3 grid(static_cast<unsigned int>(blocksX),
                    static_cast<unsigned int>(std::min<size_t>((ny - 1) / block.y + 1, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(nz, 65535)));

    initializeConcentration<<<grid, block>>>(d_cold, nx, ny, nz);
    checkCuda(cudaGetLastError(), "launching initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration");

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotential<<<grid, block>>>(d_cold, d_mu, nx, ny, nz);
        updateConcentration<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz);
        std::swap(d_cold, d_cnew);
    }
    checkCuda(cudaGetLastError(), "launching simulation kernels");
    checkCuda(cudaDeviceSynchronize(), "running simulation kernels");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / elapsedSeconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    if (printResults || validate) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost),
                  "copying final concentration");
    }
    checkCuda(cudaFree(d_cold), "freeing concentration");
    checkCuda(cudaFree(d_cnew), "freeing next concentration");
    checkCuda(cudaFree(d_mu), "freeing chemical potential");

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
