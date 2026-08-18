#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

__device__ __forceinline__ size_t idx3(const size_t x, const size_t y, const size_t z,
                                       const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__device__ __forceinline__ double computeLaplacian(const double* c, const size_t nx, const size_t ny,
                                                   const size_t nz, const double dx, const double dy,
                                                   const double dz, const size_t x, const size_t y,
                                                   const size_t z) {
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

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                                                const size_t nx, const size_t ny, const size_t nz,
                                                const double dx, const double dy, const double dz,
                                                const double gamma, const double e_AA,
                                                const double e_BB, const double e_AB) {
    for (size_t z = blockIdx.z * blockDim.z + threadIdx.z; z < nz; z += blockDim.z * gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += blockDim.y * gridDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx; x += blockDim.x * gridDim.x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                                          const size_t nx, const size_t ny, const size_t nz,
                                          const double D, const double dt, const double dx,
                                          const double dy, const double dz) {
    for (size_t z = blockIdx.z * blockDim.z + threadIdx.z; z < nz; z += blockDim.z * gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += blockDim.y * gridDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx; x += blockDim.x * gridDim.x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
__global__ void initializeConcentrationKernel(double* c, const size_t nx, const size_t ny,
                                               const size_t nz) {
    const size_t vol = nx * ny * nz;
    for (size_t z = blockIdx.z * blockDim.z + threadIdx.z; z < nz; z += blockDim.z * gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += blockDim.y * gridDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx; x += blockDim.x * gridDim.x) {
                const size_t linear_id = idx3(x, y, z, nx, ny);
                const double pseudo = (((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol);
                c[linear_id] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

[[noreturn]] void cudaFailure(const cudaError_t error, const char* operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFailure(error, operation);
}

dim3 launchGrid(const size_t nx, const size_t ny, const size_t nz, const dim3 block) {
    constexpr unsigned int maxGrid = 65535;
    return dim3(static_cast<unsigned int>(std::min((nx + block.x - 1) / block.x, size_t(maxGrid))),
                static_cast<unsigned int>(std::min((ny + block.y - 1) / block.y, size_t(maxGrid))),
                static_cast<unsigned int>(std::min((nz + block.z - 1) / block.z, size_t(maxGrid))));
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
    
    // Keep the simulation state on the GPU for the entire time integration.
    std::vector<double> cold(gridSize);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    checkCuda(cudaMalloc(&d_cold, gridSize * sizeof(double)), "allocating concentration buffer");
    checkCuda(cudaMalloc(&d_cnew, gridSize * sizeof(double)), "allocating concentration buffer");
    checkCuda(cudaMalloc(&d_mu, gridSize * sizeof(double)), "allocating chemical-potential buffer");

    const dim3 block(8, 8, 4);
    const dim3 grid = launchGrid(nx, ny, nz, block);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<grid, block>>>(d_cold, nx, ny, nz);
    checkCuda(cudaGetLastError(), "launching initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration field");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, nz, dx, dy, dz,
                                                        gamma, e_AA, e_BB, e_AB);
        checkCuda(cudaGetLastError(), "launching chemical-potential kernel");
        
        // Update concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt,
                                                  dx, dy, dz);
        checkCuda(cudaGetLastError(), "launching Cahn-Hilliard update kernel");
        
        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaDeviceSynchronize(), "running Cahn-Hilliard simulation");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());

    checkCuda(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost),
              "copying final concentration field");
    
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

        cudaFree(d_cold);
        cudaFree(d_cnew);
        cudaFree(d_mu);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    checkCuda(cudaFree(d_cold), "freeing concentration buffer");
    checkCuda(cudaFree(d_cnew), "freeing concentration buffer");
    checkCuda(cudaFree(d_mu), "freeing chemical-potential buffer");

    return 0;
}
