#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Each warp processes one contiguous row segment. The row loop also permits
// grids with more rows than CUDA's grid.y limit.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
        size_t idx, size_t x, size_t y, size_t z,
        size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = field[idx];
    const double cxx = field[x + 1 < nx ? idx + 1 : idx]
                     + field[x > 0 ? idx - 1 : idx] - 2.0 * center;
    const double cyy = field[y + 1 < ny ? idx + nx : idx]
                     + field[y > 0 ? idx - nx : idx] - 2.0 * center;
    const double czz = field[z + 1 < nz ? idx + plane : idx]
                     + field[z > 0 ? idx - plane : idx] - 2.0 * center;
    return cxx + cyy + czz;
}

template <typename Index>
__global__ void chemicalPotential(const double* __restrict__ c,
                                  double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t rows = ny * nz;
    for (size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         row < rows; row += static_cast<size_t>(gridDim.y) * blockDim.y) {
        const Index y = static_cast<Index>(row) % static_cast<Index>(ny);
        const Index z = static_cast<Index>(row) / static_cast<Index>(ny);
        const size_t idx = row * nx + x;
        const double cv = c[idx];
        const double e_AA = -(2.0 / 9.0);
        const double e_BB = -(2.0 / 9.0);
        const double e_AB = (2.0 / 9.0);
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv
                - 0.5 * laplacian(c, idx, x, y, z, nx, ny, nz);
    }
}

template <typename Index>
__global__ void updateConcentration(double* __restrict__ cnew,
                                    const double* __restrict__ cold,
                                    const double* __restrict__ mu,
                                    size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t rows = ny * nz;
    for (size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         row < rows; row += static_cast<size_t>(gridDim.y) * blockDim.y) {
        const Index y = static_cast<Index>(row) % static_cast<Index>(ny);
        const Index z = static_cast<Index>(row) / static_cast<Index>(ny);
        const size_t idx = row * nx + x;
        cnew[idx] = cold[idx] + 0.01 * laplacian(mu, idx, x, y, z, nx, ny, nz);
    }
}

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
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
    
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Invalid grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;

    std::vector<double> cold(gridSize);
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    double *d_cold, *d_cnew, *d_mu;
    const size_t bytes = gridSize * sizeof(double);
    checkCuda(cudaMalloc(&d_cold, bytes), "allocating concentration");
    checkCuda(cudaMalloc(&d_cnew, bytes), "allocating next concentration");
    checkCuda(cudaMalloc(&d_mu, bytes), "allocating chemical potential");
    checkCuda(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice),
              "copying initial concentration");

    const dim3 block(32, 4);
    const size_t xBlocks = (nx - 1) / block.x + 1;
    if (xBlocks > 2147483647ULL) {
        fprintf(stderr, "Grid size exceeds CUDA launch limits\n");
        return 1;
    }
    const dim3 grid(static_cast<unsigned>(xBlocks),
                    static_cast<unsigned>(std::min<size_t>((ny * nz - 1) / block.y + 1, 65535)));

    // Load the kernels before timing so module loading and JIT compilation are
    // excluded, just as allocation and initialization are.
    const bool use32bit = gridSize <= UINT32_MAX;
    cudaFuncAttributes attributes;
    if (use32bit) {
        checkCuda(cudaFuncGetAttributes(&attributes, chemicalPotential<uint32_t>), "loading chemical potential kernel");
        checkCuda(cudaFuncGetAttributes(&attributes, updateConcentration<uint32_t>), "loading update kernel");
    } else {
        checkCuda(cudaFuncGetAttributes(&attributes, chemicalPotential<size_t>), "loading chemical potential kernel");
        checkCuda(cudaFuncGetAttributes(&attributes, updateConcentration<size_t>), "loading update kernel");
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        if (use32bit) {
            chemicalPotential<uint32_t><<<grid, block>>>(d_cold, d_mu, nx, ny, nz);
            updateConcentration<uint32_t><<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz);
        } else {
            chemicalPotential<size_t><<<grid, block>>>(d_cold, d_mu, nx, ny, nz);
            updateConcentration<size_t><<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz);
        }
        std::swap(d_cold, d_cnew);
    }
    checkCuda(cudaGetLastError(), "launching simulation kernels");
    checkCuda(cudaDeviceSynchronize(), "running simulation kernels");
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Computation time: %.3f ms\n", elapsed_ms);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost),
                  "copying final concentration");
    }
    checkCuda(cudaFree(d_cold), "freeing concentration");
    checkCuda(cudaFree(d_cnew), "freeing next concentration");
    checkCuda(cudaFree(d_mu), "freeing chemical potential");

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (elapsed_ms / 1000.0) / 1e6;
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
