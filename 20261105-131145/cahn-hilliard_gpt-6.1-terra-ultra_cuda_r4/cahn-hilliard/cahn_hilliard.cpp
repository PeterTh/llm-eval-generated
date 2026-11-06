#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Each block covers a strip of rows, with adjacent lanes traversing x.
// The y and z loops also handle grids larger than CUDA's grid dimension limits.
constexpr unsigned int BLOCK_X = 32;
constexpr unsigned int BLOCK_Y = 4;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
        size_t index, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = field[index];
    const double cxx = field[index + (x + 1 < nx ? 1 : 0)] +
                       field[index - (x > 0 ? 1 : 0)] - 2.0 * center;
    const double cyy = field[index + (y + 1 < ny ? nx : 0)] +
                       field[index - (y > 0 ? nx : 0)] - 2.0 * center;
    const double czz = field[index + (z + 1 < nz ? plane : 0)] +
                       field[index - (z > 0 ? plane : 0)] - 2.0 * center;
    // dx = dy = dz = 1 in this benchmark.
    return cxx + cyy + czz;
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * BLOCK_Y) {
            const size_t index = z * plane + y * nx + x;
            const double cv = c[index];
            mu[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv
                      - 0.5 * laplacian(c, index, x, y, z, nx, ny, nz);
        }
    }
}

__global__ void updateConcentration(double* __restrict__ cnew, const double* __restrict__ cold,
                                    const double* __restrict__ mu, size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * BLOCK_Y) {
            const size_t index = z * plane + y * nx + x;
            cnew[index] = cold[index] + 0.01 * laplacian(mu, index, x, y, z, nx, ny, nz);
        }
    }
}

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
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
    
    const size_t gridSize = nx * ny * nz;
    std::vector<double> cold(gridSize);

    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    double *d_cold, *d_cnew, *d_mu;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>(std::min<size_t>((ny + BLOCK_Y - 1) / BLOCK_Y, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(nz, 65535)));

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotential<<<grid, block>>>(d_cold, d_mu, nx, ny, nz);
        updateConcentration<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz);
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = (double)gridSize * iterations;
    double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / elapsedSeconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

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
