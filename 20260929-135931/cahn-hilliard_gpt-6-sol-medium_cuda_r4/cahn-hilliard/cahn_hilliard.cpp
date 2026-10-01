#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// The benchmark uses unit spacing on all three axes. Each thread handles one
// cell; clamped neighbor indices reproduce the original boundary condition.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                            size_t i, size_t x, size_t y, size_t z,
                                            size_t nx, size_t ny, size_t nz, size_t plane) {
    const double center = field[i];
    const double xx = (field[x + 1 < nx ? i + 1 : i] + field[x ? i - 1 : i] - 2.0 * center);
    const double yy = (field[y + 1 < ny ? i + nx : i] + field[y ? i - nx : i] - 2.0 * center);
    const double zz = (field[z + 1 < nz ? i + plane : i] + field[z ? i - plane : i] - 2.0 * center);
    return xx + yy + zz;
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t nz, double gamma,
                                  double e_AA, double e_BB, double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z;
    if (x >= nx || y >= ny) return;
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
          + 3.0 * cv + cv * cv * cv
          - gamma * laplacian(c, i, x, y, z, nx, ny, nz, plane);
}

__global__ void concentrationUpdate(double* __restrict__ next, const double* __restrict__ current,
                                    const double* __restrict__ mu, size_t nx, size_t ny,
                                    size_t nz, double dt, double D) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z;
    if (x >= nx || y >= ny) return;
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    next[i] = current[i] + dt * D * laplacian(mu, i, x, y, z, nx, ny, nz, plane);
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
    
    // Physical parameters
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    double *deviceCurrent, *deviceNext, *deviceMu;
    const size_t bytes = gridSize * sizeof(double);
    checkCuda(cudaMalloc(&deviceCurrent, bytes), "allocate concentration");
    checkCuda(cudaMalloc(&deviceNext, bytes), "allocate next concentration");
    checkCuda(cudaMalloc(&deviceMu, bytes), "allocate chemical potential");
    checkCuda(cudaMemcpy(deviceCurrent, cold.data(), bytes, cudaMemcpyHostToDevice),
              "copy initial concentration");
    const dim3 threads(32, 4);
    const dim3 blocks((nx + threads.x - 1) / threads.x,
                      (ny + threads.y - 1) / threads.y, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        chemicalPotential<<<blocks, threads>>>(deviceCurrent, deviceMu, nx, ny, nz,
                                                gamma, e_AA, e_BB, e_AB);
        concentrationUpdate<<<blocks, threads>>>(deviceNext, deviceCurrent, deviceMu,
                                                 nx, ny, nz, dt, D);
        std::swap(deviceCurrent, deviceNext);
    }
    checkCuda(cudaGetLastError(), "launch simulation kernels");
    checkCuda(cudaDeviceSynchronize(), "run simulation kernels");
    
    auto end = std::chrono::high_resolution_clock::now();
    const double durationMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(cold.data(), deviceCurrent, bytes, cudaMemcpyDeviceToHost),
                  "copy final concentration");
    }
    checkCuda(cudaFree(deviceCurrent), "free concentration");
    checkCuda(cudaFree(deviceNext), "free next concentration");
    checkCuda(cudaFree(deviceMu), "free chemical potential");
    
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
