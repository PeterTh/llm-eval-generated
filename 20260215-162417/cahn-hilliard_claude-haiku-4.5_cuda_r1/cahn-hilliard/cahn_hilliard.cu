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

// CUDA kernel: Compute Laplacian
__global__ void computeLaplacianKernel(const double* c, double* laplacian, 
                                       const size_t nx, const size_t ny, const size_t nz,
                                       const double dx_inv_sq, const double dy_inv_sq, const double dz_inv_sq) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const size_t idx = z * (nx * ny) + y * nx + x;
    const size_t idx_xp = z * (nx * ny) + y * nx + xp;
    const size_t idx_xn = z * (nx * ny) + y * nx + xn;
    const size_t idx_yp = z * (nx * ny) + yp * nx + x;
    const size_t idx_yn = z * (nx * ny) + yn * nx + x;
    const size_t idx_zp = zp * (nx * ny) + y * nx + x;
    const size_t idx_zn = zn * (nx * ny) + y * nx + x;
    
    const double cxx = (c[idx_xp] + c[idx_xn] - 2.0 * c[idx]) * dx_inv_sq;
    const double cyy = (c[idx_yp] + c[idx_yn] - 2.0 * c[idx]) * dy_inv_sq;
    const double czz = (c[idx_zp] + c[idx_zn] - 2.0 * c[idx]) * dz_inv_sq;
    
    laplacian[idx] = cxx + cyy + czz;
}

// CUDA kernel: Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* c, const double* laplacian_c, double* mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    const size_t idx = z * (nx * ny) + y * nx + x;
    const double cv = c[idx];
    
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * laplacian_c[idx];
}

// CUDA kernel: Cahn-Hilliard update
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* laplacian_mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D_dt) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    const size_t idx = z * (nx * ny) + y * nx + x;
    cnew[idx] = cold[idx] + D_dt * laplacian_mu[idx];
}

// CUDA kernel: Initialize concentration field
__global__ void initializeConcentrationKernel(double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    const size_t vol = nx * ny * nz;
    const size_t linear_id = z * (nx * ny) + y * nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[linear_id] = -1.0 + 2.0 * pseudo;
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

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
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
    
    printf("Cahn-Hilliard Phase Separation Benchmark (CUDA)\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dx_inv_sq = 1.0 / (dx * dx);
    const double dy_inv_sq = 1.0 / (dy * dy);
    const double dz_inv_sq = 1.0 / (dz * dz);
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    const double D_dt = D * dt;
    
    size_t gridSize = nx * ny * nz;
    size_t gridBytes = gridSize * sizeof(double);
    
    // Allocate host arrays
    std::vector<double> cold_host(gridSize);
    std::vector<double> cnew_host(gridSize);
    
    // Allocate device arrays
    double* cold_dev = nullptr;
    double* cnew_dev = nullptr;
    double* mu_dev = nullptr;
    double* laplacian_c_dev = nullptr;
    double* laplacian_mu_dev = nullptr;
    
    checkCudaError(cudaMalloc(&cold_dev, gridBytes), "Allocate cold");
    checkCudaError(cudaMalloc(&cnew_dev, gridBytes), "Allocate cnew");
    checkCudaError(cudaMalloc(&mu_dev, gridBytes), "Allocate mu");
    checkCudaError(cudaMalloc(&laplacian_c_dev, gridBytes), "Allocate laplacian_c");
    checkCudaError(cudaMalloc(&laplacian_mu_dev, gridBytes), "Allocate laplacian_mu");
    
    // Configure thread blocks and grid
    const dim3 blockDim(8, 8, 8);
    const dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                       (ny + blockDim.y - 1) / blockDim.y,
                       (nz + blockDim.z - 1) / blockDim.z);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<gridDim, blockDim>>>(cold_dev, nx, ny, nz);
    checkCudaError(cudaDeviceSynchronize(), "Initialize kernel");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute Laplacian of concentration
        computeLaplacianKernel<<<gridDim, blockDim>>>(cold_dev, laplacian_c_dev, nx, ny, nz, dx_inv_sq, dy_inv_sq, dz_inv_sq);
        checkCudaError(cudaDeviceSynchronize(), "Laplacian c kernel");
        
        // Compute chemical potential
        computeChemicalPotentialKernel<<<gridDim, blockDim>>>(cold_dev, laplacian_c_dev, mu_dev, nx, ny, nz, gamma, e_AA, e_BB, e_AB);
        checkCudaError(cudaDeviceSynchronize(), "Chemical potential kernel");
        
        // Compute Laplacian of chemical potential
        computeLaplacianKernel<<<gridDim, blockDim>>>(mu_dev, laplacian_mu_dev, nx, ny, nz, dx_inv_sq, dy_inv_sq, dz_inv_sq);
        checkCudaError(cudaDeviceSynchronize(), "Laplacian mu kernel");
        
        // Update concentration
        cahnHilliardUpdateKernel<<<gridDim, blockDim>>>(cnew_dev, cold_dev, laplacian_mu_dev, nx, ny, nz, D_dt);
        checkCudaError(cudaDeviceSynchronize(), "Update kernel");
        
        // Swap buffers
        std::swap(cold_dev, cnew_dev);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(cold_host.data(), cold_dev, gridBytes, cudaMemcpyDeviceToHost), "Copy result");
    
    // Print results for external validation
    if (printResults) {
        print_results(cold_host, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold_host, nx, ny, nz);
        
        // Cleanup
        cudaFree(cold_dev);
        cudaFree(cnew_dev);
        cudaFree(mu_dev);
        cudaFree(laplacian_c_dev);
        cudaFree(laplacian_mu_dev);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    // Cleanup
    cudaFree(cold_dev);
    cudaFree(cnew_dev);
    cudaFree(mu_dev);
    cudaFree(laplacian_c_dev);
    cudaFree(laplacian_mu_dev);
    
    return 0;
}
