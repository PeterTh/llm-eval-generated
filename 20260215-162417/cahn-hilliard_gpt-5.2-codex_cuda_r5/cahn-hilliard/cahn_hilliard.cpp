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

inline void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(result));
        std::exit(1);
    }
}

__device__ __forceinline__ size_t idx3_device(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__device__ __forceinline__ double computeLaplacianDevice(const double* c, const size_t nx, const size_t ny, const size_t nz,
                                                         const double dx, const double dy, const double dz,
                                                         const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t zp = (z + 1 < nz) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const size_t idx = idx3_device(x, y, z, nx, ny);
    const double cxx = (c[idx3_device(xp, y, z, nx, ny)] + c[idx3_device(xn, y, z, nx, ny)] -
                        2.0 * c[idx]) / (dx * dx);
    const double cyy = (c[idx3_device(x, yp, z, nx, ny)] + c[idx3_device(x, yn, z, nx, ny)] -
                        2.0 * c[idx]) / (dy * dy);
    const double czz = (c[idx3_device(x, y, zp, nx, ny)] + c[idx3_device(x, y, zn, nx, ny)] -
                        2.0 * c[idx]) / (dz * dz);

    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* c, const size_t nx, const size_t ny, const size_t nz, const size_t vol) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= vol) {
        return;
    }
    const unsigned long long linear_id = static_cast<unsigned long long>(idx);
    const double pseudo = static_cast<double>(((linear_id + 1ULL) * 1299709ULL) % static_cast<unsigned long long>(vol)) /
                          static_cast<double>(vol);
    c[idx] = -1.0 + 2.0 * pseudo;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                               const size_t vol) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= vol) {
        return;
    }
    const size_t plane = nx * ny;
    const size_t z = idx / plane;
    const size_t rem = idx - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    const double cv = c[idx];

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * computeLaplacianDevice(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                        const double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double D, const double dt, const double dx, const double dy, const double dz,
                                        const size_t vol) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= vol) {
        return;
    }
    const size_t plane = nx * ny;
    const size_t z = idx / plane;
    const size_t rem = idx - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;

    cnew[idx] = cold[idx] + dt * D * computeLaplacianDevice(mu, nx, ny, nz, dx, dy, dz, x, y, z);
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
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    checkCuda(cudaMalloc(&d_cold, gridSize * sizeof(double)), "cudaMalloc d_cold");
    checkCuda(cudaMalloc(&d_cnew, gridSize * sizeof(double)), "cudaMalloc d_cnew");
    checkCuda(cudaMalloc(&d_mu, gridSize * sizeof(double)), "cudaMalloc d_mu");
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    const int threads = 256;
    const int blocks = static_cast<int>((gridSize + threads - 1) / threads);
    initializeConcentrationKernel<<<blocks, threads>>>(d_cold, nx, ny, nz, gridSize);
    checkCuda(cudaGetLastError(), "initializeConcentrationKernel");
    checkCuda(cudaDeviceSynchronize(), "initializeConcentrationKernel sync");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    checkCuda(cudaDeviceSynchronize(), "pre-sim sync");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<blocks, threads>>>(d_cold, d_mu, nx, ny, nz, dx, dy, dz,
                                                            gamma, e_AA, e_BB, e_AB, gridSize);
        checkCuda(cudaGetLastError(), "computeChemicalPotentialKernel");
        
        // Update concentration
        cahnHilliardUpdateKernel<<<blocks, threads>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz, gridSize);
        checkCuda(cudaGetLastError(), "cahnHilliardUpdateKernel");
        
        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaDeviceSynchronize(), "kernel sync");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    checkCuda(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy d_cold");
    checkCuda(cudaFree(d_mu), "cudaFree d_mu");
    checkCuda(cudaFree(d_cnew), "cudaFree d_cnew");
    checkCuda(cudaFree(d_cold), "cudaFree d_cold");

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
