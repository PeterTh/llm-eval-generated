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

__device__ inline size_t deviceIdx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * nx * ny + y * nx + x;
}

__global__ void initializeKernel(double* c, size_t nx, size_t ny, size_t nz, size_t vol) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vol) return;
    const size_t pseudo = (((i + 1) * 1299709) % vol);
    c[i] = -1.0 + 2.0 * (static_cast<double>(pseudo) / static_cast<double>(vol));
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        size_t nx, size_t ny, size_t nz,
                                        double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (i >= vol) return;
    const size_t x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz), zn = z - (z > 0);
    const double cv = c[i];
    const double lap = c[deviceIdx3(xp,y,z,nx,ny)] + c[deviceIdx3(xn,y,z,nx,ny)] - 2.0*cv
                     + c[deviceIdx3(x,yp,z,nx,ny)] + c[deviceIdx3(x,yn,z,nx,ny)] - 2.0*cv
                     + c[deviceIdx3(x,y,zp,nx,ny)] + c[deviceIdx3(x,y,zn,nx,ny)] - 2.0*cv;
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
          + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu, size_t nx, size_t ny, size_t nz,
                             double D, double dt) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (i >= vol) return;
    const size_t x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz), zn = z - (z > 0);
    const double center = mu[i];
    const double lap = mu[deviceIdx3(xp,y,z,nx,ny)] + mu[deviceIdx3(xn,y,z,nx,ny)] - 2.0*center
                     + mu[deviceIdx3(x,yp,z,nx,ny)] + mu[deviceIdx3(x,yn,z,nx,ny)] - 2.0*center
                     + mu[deviceIdx3(x,y,zp,nx,ny)] + mu[deviceIdx3(x,y,zn,nx,ny)] - 2.0*center;
    cnew[i] = cold[i] + dt * D * lap;
}

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
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

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
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
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 0; z < nz; ++z) {
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
    
    double *deviceCold = nullptr, *deviceCnew = nullptr, *deviceMu = nullptr;
    cudaCheck(cudaMalloc(&deviceCold, gridSize * sizeof(double)));
    cudaCheck(cudaMalloc(&deviceCnew, gridSize * sizeof(double)));
    cudaCheck(cudaMalloc(&deviceMu, gridSize * sizeof(double)));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    constexpr int threads = 256;
    const int blocks = static_cast<int>((gridSize + threads - 1) / threads);
    initializeKernel<<<blocks, threads>>>(deviceCold, nx, ny, nz, gridSize);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        chemicalPotentialKernel<<<blocks, threads>>>(deviceCold, deviceMu, nx, ny, nz,
                                                     gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        updateKernel<<<blocks, threads>>>(deviceCnew, deviceCold, deviceMu, nx, ny, nz, D, dt);
        
        // Swap buffers
        std::swap(deviceCold, deviceCnew);
    }
    
    cudaCheck(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<double> cold(gridSize);
    cudaCheck(cudaMemcpy(cold.data(), deviceCold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
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
    
    cudaCheck(cudaFree(deviceCold));
    cudaCheck(cudaFree(deviceCnew));
    cudaCheck(cudaFree(deviceMu));
    return 0;
}
