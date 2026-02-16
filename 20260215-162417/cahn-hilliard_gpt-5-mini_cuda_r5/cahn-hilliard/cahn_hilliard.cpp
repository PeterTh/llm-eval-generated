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
inline __host__ __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device Laplacian with clamped boundary conditions
inline __device__ __host__ double computeLaplacian_device(const double* c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double center = c[idx3(x, y, z, nx, ny)];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * center) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * center) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * center) / (dz * dz);
    
    return cxx + cyy + czz;
}

// CUDA kernel: compute chemical potential
__global__ void kernel_computeChemicalPotential(const double* c, double* mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t N = nx * ny * nz;
    if (idx >= N) return;
    size_t z = idx / (nx * ny);
    size_t r = idx % (nx * ny);
    size_t y = r / nx;
    size_t x = r % nx;

    const double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * computeLaplacian_device(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

// CUDA kernel: CH update
__global__ void kernel_cahnHilliardUpdate(double* cnew, const double* cold, const double* mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t N = nx * ny * nz;
    if (idx >= N) return;
    size_t z = idx / (nx * ny);
    size_t r = idx % (nx * ny);
    size_t y = r / nx;
    size_t x = r % nx;

    double lap = computeLaplacian_device(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    cnew[idx] = cold[idx] + dt * D * lap;
}

// CUDA kernel: initialize concentration using same pseudo-random formula
__global__ void kernel_initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t N = nx * ny * nz;
    if (idx >= N) return;
    const size_t vol = N;
    const size_t linear_id = idx;
    const double pseudo = ((((linear_id + 1ULL) * 1299709ULL) % vol) / static_cast<double>(vol));
    c[idx] = -1.0 + 2.0 * pseudo;
}

// Validate device buffer (host function, checks managed memory contents)
bool validateResult_ptr(const double* c_ptr, const size_t N, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (size_t i = 0; i < N; ++i) {
        double val = c_ptr[i];
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    double minVal = c_ptr[0];
    double maxVal = c_ptr[0];
    for (size_t i = 0; i < N; ++i) {
        double val = c_ptr[i];
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
    
    // Allocate unified memory for GPU
    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    cudaMallocManaged(&cold, gridSize * sizeof(double));
    cudaMallocManaged(&cnew, gridSize * sizeof(double));
    cudaMallocManaged(&mu, gridSize * sizeof(double));

    // Initialize concentration field on GPU
    printf("Initializing concentration field...\n");
    const int threads = 256;
    const int blocks = (int)((gridSize + threads - 1) / threads);
    kernel_initializeConcentration<<<blocks, threads>>>(cold, nx, ny, nz);
    cudaDeviceSynchronize();
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential on GPU
        kernel_computeChemicalPotential<<<blocks, threads>>>(cold, mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        // Update concentration on GPU
        kernel_cahnHilliardUpdate<<<blocks, threads>>>(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);
        // Wait for kernels to finish
        cudaDeviceSynchronize();
        // Swap buffers
        std::swap(cold, cnew);
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
        std::vector<double> tmp(cold, cold + gridSize);
        print_results(tmp, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult_ptr(cold, gridSize, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            cudaFree(cold);
            cudaFree(cnew);
            cudaFree(mu);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            cudaFree(cold);
            cudaFree(cnew);
            cudaFree(mu);
            return 1;
        }
    }
    
    cudaFree(cold);
    cudaFree(cnew);
    cudaFree(mu);
    return 0;
}
