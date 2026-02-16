#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../common/results_output.hpp"

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
__device__ double computeLaplacian(const double* c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
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
    
    return cxx + cyy + czz;
}

// CUDA kernel for chemical potential
__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                             size_t nx, size_t ny, size_t nz,
                             double dx, double dy, double dz,
                             double gamma, double e_AA, double e_BB, double e_AB) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = nx * ny * nz;
    if (tid >= gridSize) return;
    size_t z = tid / (nx * ny);
    size_t y = (tid % (nx * ny)) / nx;
    size_t x = tid % nx;
    const double cv = c[tid];
    mu[tid] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

// CUDA kernel for Cahn-Hilliard update
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                       size_t nx, size_t ny, size_t nz,
                       double D, double dt, double dx, double dy, double dz) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = nx * ny * nz;
    if (tid >= gridSize) return;
    size_t z = tid / (nx * ny);
    size_t y = (tid % (nx * ny)) / nx;
    size_t x = tid % nx;
    cnew[tid] = cold[tid] + dt * D * computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
}

// Host wrappers
void computeChemicalPotentialCUDA(const double* c, double* mu,
                             size_t nx, size_t ny, size_t nz,
                             double dx, double dy, double dz,
                             double gamma, double e_AA, double e_BB, double e_AB, cudaStream_t stream) {
    size_t gridSize = nx * ny * nz;
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    computeChemicalPotentialKernel<<<numBlocks, blockSize, 0, stream>>>(c, mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
}

void cahnHilliardUpdateCUDA(double* cnew, const double* cold, const double* mu,
                       size_t nx, size_t ny, size_t nz,
                       double D, double dt, double dx, double dy, double dz, cudaStream_t stream) {
    size_t gridSize = nx * ny * nz;
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    cahnHilliardUpdateKernel<<<numBlocks, blockSize, 0, stream>>>(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);
}

        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
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

void initializeConcentrationCUDA(double* c, const size_t nx, const size_t ny, const size_t nz) {
    std::vector<double> host_c(nx * ny * nz);
    initializeConcentration(host_c, nx, ny, nz);
    cudaMemcpy(c, host_c.data(), sizeof(double) * nx * ny * nz, cudaMemcpyHostToDevice);
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
    // Parse command line arguments ... (unchanged)
    // ...
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    // Physical parameters ... (unchanged)
    // ...
    size_t gridSize = nx * ny * nz;
    // Allocate device arrays
    double *d_cold, *d_cnew, *d_mu;
    cudaMalloc(&d_cold, sizeof(double) * gridSize);
    cudaMalloc(&d_cnew, sizeof(double) * gridSize);
    cudaMalloc(&d_mu, sizeof(double) * gridSize);
    // Initialize concentration field on device
    printf("Initializing concentration field...\n");
    initializeConcentrationCUDA(d_cold, nx, ny, nz);
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    cudaStream_t stream = nullptr;
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialCUDA(d_cold, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB, stream);
        // Update concentration
        cahnHilliardUpdateCUDA(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz, stream);
        // Swap buffers
        double* tmp = d_cold;
        d_cold = d_cnew;
        d_cnew = tmp;
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", duration.count());
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    // Print results for external validation
    std::vector<double> host_cold(gridSize);
    cudaMemcpy(host_cold.data(), d_cold, sizeof(double) * gridSize, cudaMemcpyDeviceToHost);
    if (printResults) {
        print_results(host_cold, "Concentration");
    }
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(host_cold, nx, ny, nz);
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
    return 0;
}
