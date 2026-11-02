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

// Each thread handles one cell. X is contiguous so a warp accesses adjacent
// cells, while the Y and Z grid strides also support unusually tall grids.
#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void initializeConcentration(double* c, size_t nx, size_t ny, size_t nz, size_t vol) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            const size_t i = z * plane + y * nx + x;
            const double pseudo = ((i + 1) * size_t(1299709) % vol) / static_cast<double>(vol);
            c[i] = -1.0 + 2.0 * pseudo;
        }
    }
}

__device__ __forceinline__ double laplacian(const double* __restrict__ a, size_t i,
                                             size_t x, size_t y, size_t z,
                                             size_t nx, size_t ny, size_t nz, size_t plane) {
    const double center = a[i];
    const double cxx = a[x + 1 < nx ? i + 1 : i] + a[x > 0 ? i - 1 : i] - 2.0 * center;
    const double cyy = a[y + 1 < ny ? i + nx : i] + a[y > 0 ? i - nx : i] - 2.0 * center;
    const double czz = a[z + 1 < nz ? i + plane : i] + a[z > 0 ? i - plane : i] - 2.0 * center;
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                          size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            const size_t i = z * plane + y * nx + x;
            const double cv = c[i];
            mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c, i, x, y, z, nx, ny, nz, plane);
        }
    }
}

__global__ void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                                   const double* __restrict__ mu, size_t nx, size_t ny, size_t nz) {
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
        fprintf(stderr, "Invalid or too large grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    const dim3 block(32, 4);
    const size_t blocksX = (nx - 1) / block.x + 1;
    if (blocksX > 2147483647u) {
        fprintf(stderr, "Grid size X exceeds CUDA launch limits\n");
        return 1;
    }
    const dim3 grid(static_cast<unsigned int>(blocksX),
                    static_cast<unsigned int>(std::min<size_t>((ny - 1) / block.y + 1, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(nz, 65535)));

    double *cold, *cnew, *mu;
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));

    printf("Initializing concentration field...\n");
    initializeConcentration<<<grid, block>>>(cold, nx, ny, nz, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential<<<grid, block>>>(cold, mu, nx, ny, nz);
        cahnHilliardUpdate<<<grid, block>>>(cnew, cold, mu, nx, ny, nz);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();

    printf("Computation time: %.3f ms\n", seconds * 1000.0);

    // Calculate performance using the full timer resolution.
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));

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
