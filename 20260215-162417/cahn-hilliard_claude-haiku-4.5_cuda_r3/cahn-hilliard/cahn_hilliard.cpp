#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Macro for CUDA error checking
#define CUDA_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device version of 3D index calculation
__device__ inline size_t d_idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions (device kernel)
__device__ double d_computeLaplacian(const double* c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[d_idx3(xp, y, z, nx, ny)] + c[d_idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[d_idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[d_idx3(x, yp, z, nx, ny)] + c[d_idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[d_idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[d_idx3(x, y, zp, nx, ny)] + c[d_idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[d_idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Kernel to compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = d_idx3(x, y, z, nx, ny);
        const double cv = c[idx];
        
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * d_computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// Compute chemical potential (CPU wrapper)
void computeChemicalPotential(const double* d_c, double* d_mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz + blockDim.z - 1) / blockDim.z);
    
    computeChemicalPotentialKernel<<<gridDim, blockDim>>>(d_c, d_mu, nx, ny, nz, 
                                                           dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel for Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold,
                                         const double* mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt, const double dx, const double dy, const double dz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = d_idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * 
                   d_computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// Cahn-Hilliard update step (CPU wrapper)
void cahnHilliardUpdate(double* d_cnew, const double* d_cold,
                        const double* d_mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz + blockDim.z - 1) / blockDim.z);
    
    cahnHilliardUpdateKernel<<<gridDim, blockDim>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to initialize concentration field
__global__ void initializeConcentrationKernel(double* c, const size_t nx, const size_t ny, const size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = d_idx3(x, y, z, nx, ny);
        const size_t vol = nx * ny * nz;
        const size_t linear_id = z * (nx * ny) + y * nx + x;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[idx] = -1.0 + 2.0 * pseudo;
    }
}

// Initialize concentration field (CPU wrapper)
void initializeConcentration(double* d_c, const size_t nx, const size_t ny, const size_t nz) {
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz + blockDim.z - 1) / blockDim.z);
    
    initializeConcentrationKernel<<<gridDim, blockDim>>>(d_c, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
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
    size_t bytes = gridSize * sizeof(double);
    
    // Allocate host arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    
    // Allocate device arrays
    double* d_cold;
    double* d_cnew;
    double* d_mu;
    
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    
    // Initialize concentration field on GPU
    printf("Initializing concentration field...\n");
    initializeConcentration(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(d_cold, d_mu, nx, ny, nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdate(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
        
        // Swap buffers (swap device pointers)
        std::swap(d_cold, d_cnew);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    
    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
        // Clean up GPU memory
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    // Clean up GPU memory
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
    return 0;
}
