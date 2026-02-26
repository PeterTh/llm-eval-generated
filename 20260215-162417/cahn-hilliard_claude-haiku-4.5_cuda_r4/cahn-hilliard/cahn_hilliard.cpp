#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr __device__ __host__ size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel: Compute Laplacian with clamped boundary conditions
__global__ void computeLaplacianKernel(const double* c, double* laplacian,
                                       const size_t nx, const size_t ny, const size_t nz,
                                       const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t xp = (x < nx - 1) ? x + 1 : x;
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t zp = (z < nz - 1) ? z + 1 : z;
        const size_t xn = (x > 0) ? x - 1 : 0;
        const size_t yn = (y > 0) ? y - 1 : 0;
        const size_t zn = (z > 0) ? z - 1 : 0;
        
        const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                      2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
        const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                      2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
        const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                      2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
        
        laplacian[idx3(x, y, z, nx, ny)] = cxx + cyy + czz;
    }
}

// CUDA kernel: Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* c, const double* laplacian_c, double* mu,
                                                const size_t nx, const size_t ny, const size_t nz,
                                                const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        const double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * laplacian_c[idx];
    }
}

// CUDA kernel: Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* laplacian_mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * laplacian_mu[idx];
    }
}

// CUDA kernel: Initialize concentration field
__global__ void initializeConcentrationKernel(double* c, const size_t nx, const size_t ny, const size_t nz, const size_t vol) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        const size_t linear_id = z * (nx * ny) + y * nx + x;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[idx] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const double* c, const size_t gridSize, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    std::vector<double> h_c(gridSize);
    cudaMemcpy(h_c.data(), c, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (const auto& val : h_c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    double minVal = h_c[0];
    double maxVal = h_c[0];
    for (const auto& val : h_c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
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
    size_t memSize = gridSize * sizeof(double);
    
    // Allocate device arrays
    double *d_cold, *d_cnew, *d_mu, *d_laplacian_c, *d_laplacian_mu;
    cudaMalloc(&d_cold, memSize);
    cudaMalloc(&d_cnew, memSize);
    cudaMalloc(&d_mu, memSize);
    cudaMalloc(&d_laplacian_c, memSize);
    cudaMalloc(&d_laplacian_mu, memSize);
    
    // Initialize concentration field on GPU
    printf("Initializing concentration field...\n");
    dim3 blockSize(8, 8, 8);
    dim3 gridDim((nx + blockSize.x - 1) / blockSize.x,
                 (ny + blockSize.y - 1) / blockSize.y,
                 (nz + blockSize.z - 1) / blockSize.z);
    
    initializeConcentrationKernel<<<gridDim, blockSize>>>(d_cold, nx, ny, nz, gridSize);
    cudaDeviceSynchronize();
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute Laplacian of concentration
        computeLaplacianKernel<<<gridDim, blockSize>>>(d_cold, d_laplacian_c, nx, ny, nz, dx, dy, dz);
        cudaDeviceSynchronize();
        
        // Compute chemical potential
        computeChemicalPotentialKernel<<<gridDim, blockSize>>>(d_cold, d_laplacian_c, d_mu, nx, ny, nz, gamma, e_AA, e_BB, e_AB);
        cudaDeviceSynchronize();
        
        // Compute Laplacian of mu
        computeLaplacianKernel<<<gridDim, blockSize>>>(d_mu, d_laplacian_mu, nx, ny, nz, dx, dy, dz);
        cudaDeviceSynchronize();
        
        // Update concentration
        cahnHilliardUpdateKernel<<<gridDim, blockSize>>>(d_cnew, d_cold, d_laplacian_mu, nx, ny, nz, D, dt);
        cudaDeviceSynchronize();
        
        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> h_cold(gridSize);
        cudaMemcpy(h_cold.data(), d_cold, memSize, cudaMemcpyDeviceToHost);
        print_results(h_cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(d_cold, gridSize, nx, ny, nz);
        
        cudaFree(d_cold);
        cudaFree(d_cnew);
        cudaFree(d_mu);
        cudaFree(d_laplacian_c);
        cudaFree(d_laplacian_mu);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);
    cudaFree(d_laplacian_c);
    cudaFree(d_laplacian_mu);
    
    return 0;
}
