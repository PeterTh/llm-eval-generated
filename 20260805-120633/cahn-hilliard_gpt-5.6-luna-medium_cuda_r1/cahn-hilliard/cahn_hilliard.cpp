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
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// The x dimension is the contiguous dimension.  A 32x1x4 block therefore
// gives coalesced loads while providing enough independent work to hide the
// latency of the seven-point stencil.
__device__ __forceinline__ double computeLaplacian(const double* c, const size_t nx, const size_t ny, const size_t nz,
                                                   const double invDx2, const double invDy2, const double invDz2,
                                                   const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const size_t center = idx3(x, y, z, nx, ny);
    const double cv = c[center];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * cv) * invDx2;
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * cv) * invDy2;
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * cv) * invDz2;
    
    return cxx + cyy + czz;
}

// Compute chemical potential
__global__ void computeChemicalPotential(const double* c, double* mu,
                                          const size_t nx, const size_t ny, const size_t nz,
                                          const double invDx2, const double invDy2, const double invDz2,
                                          const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t idx = idx3(x, y, z, nx, ny);
    const double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
            + 3.0 * cv + cv * cv * cv
            - gamma * computeLaplacian(c, nx, ny, nz, invDx2, invDy2, invDz2, x, y, z);
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdate(double* cnew, const double* cold, const double* mu,
                                    const size_t nx, const size_t ny, const size_t nz,
                                    const double Ddt, const double invDx2, const double invDy2, const double invDz2) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t idx = idx3(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + Ddt * computeLaplacian(mu, nx, ny, nz, invDx2, invDy2, invDz2, x, y, z);
}

// Initialize concentration field
__global__ void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz, const size_t vol) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t linear_id = idx3(x, y, z, nx, ny);
    const double pseudo = ((((linear_id + 1) * static_cast<size_t>(1299709)) % vol) / static_cast<double>(vol));
    c[linear_id] = -1.0 + 2.0 * pseudo;
}

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t error) {
    fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFailure(operation, error);
}

void checkKernel(const char* operation) {
    checkCuda(cudaGetLastError(), operation);
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
    
    // Keep the full working set on the device.  Only the final field is
    // copied back, so timestep iterations do not incur PCIe transfers.
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cold), gridSize * sizeof(double)), "cudaMalloc(d_cold)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cnew), gridSize * sizeof(double)), "cudaMalloc(d_cnew)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_mu), gridSize * sizeof(double)), "cudaMalloc(d_mu)");

    const dim3 block(32, 1, 4);
    const dim3 grid((static_cast<unsigned int>(nx) + block.x - 1) / block.x,
                    static_cast<unsigned int>(ny),
                    (static_cast<unsigned int>(nz) + block.z - 1) / block.z);

    // Initialize concentration field directly on the GPU.
    printf("Initializing concentration field...\n");
    initializeConcentration<<<grid, block>>>(d_cold, nx, ny, nz, gridSize);
    checkKernel("initializeConcentration");
    checkCuda(cudaDeviceSynchronize(), "initializeConcentration synchronization");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential<<<grid, block>>>(d_cold, d_mu, nx, ny, nz,
                                                  1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz),
                                                  gamma, e_AA, e_BB, e_AB);
        checkKernel("computeChemicalPotential");
        
        // Update concentration
        cahnHilliardUpdate<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz,
                                             D * dt, 1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz));
        checkKernel("cahnHilliardUpdate");
        
        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaDeviceSynchronize(), "simulation synchronization");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    checkCuda(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost),
              "cudaMemcpy(final concentration)");
    checkCuda(cudaFree(d_cold), "cudaFree(d_cold)");
    checkCuda(cudaFree(d_cnew), "cudaFree(d_cnew)");
    checkCuda(cudaFree(d_mu), "cudaFree(d_mu)");
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
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
