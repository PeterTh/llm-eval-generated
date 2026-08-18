#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

__device__ __forceinline__ size_t deviceIdx3(size_t x, size_t y, size_t z,
                                              size_t nx, size_t ny) {
    return z * nx * ny + y * nx + x;
}

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             size_t x, size_t y, size_t z,
                                             size_t nx, size_t ny, size_t nz,
                                             double invDx2, double invDy2,
                                             double invDz2) {
    const size_t i = deviceIdx3(x, y, z, nx, ny);
    const size_t xp = deviceIdx3(x + (x + 1 < nx), y, z, nx, ny);
    const size_t xn = deviceIdx3(x - (x > 0), y, z, nx, ny);
    const size_t yp = deviceIdx3(x, y + (y + 1 < ny), z, nx, ny);
    const size_t yn = deviceIdx3(x, y - (y > 0), z, nx, ny);
    const size_t zp = deviceIdx3(x, y, z + (z + 1 < nz), nx, ny);
    const size_t zn = deviceIdx3(x, y, z - (z > 0), nx, ny);
    const double center = field[i];
    return (field[xp] + field[xn] - 2.0 * center) * invDx2
         + (field[yp] + field[yn] - 2.0 * center) * invDy2
         + (field[zp] + field[zn] - 2.0 * center) * invDz2;
}

__global__ void initializeKernel(double* __restrict__ c, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const size_t pseudo = ((i + 1) * 1299709ull) % n;
        c[i] = -1.0 + 2.0 * (static_cast<double>(pseudo) / static_cast<double>(n));
    }
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                         double* __restrict__ mu, size_t nx,
                                         size_t ny, size_t nz, double gamma,
                                         double eAA, double eBB, double eAB) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = nx * ny * nz;
    if (i < n) {
        const size_t plane = nx * ny;
        const size_t z = i / plane;
        const size_t rem = i - z * plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        const double cv = c[i];
        mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
              + 3.0 * cv + cv * cv * cv
              - gamma * laplacian(c, x, y, z, nx, ny, nz, 1.0, 1.0, 1.0);
    }
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                              const double* __restrict__ mu, size_t nx, size_t ny,
                              size_t nz, double dtD) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = nx * ny * nz;
    if (i < n) {
        const size_t plane = nx * ny;
        const size_t z = i / plane;
        const size_t rem = i - z * plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        cnew[i] = cold[i] + dtD * laplacian(mu, x, y, z, nx, ny, nz, 1.0, 1.0, 1.0);
    }
}

inline void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
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
    
    // Keep the working set on the GPU for the complete simulation.
    std::vector<double> cold(gridSize);
    cudaDeviceProp device{};
    cudaCheck(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
    printf("CUDA device: %s\n", device.name);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    cudaCheck(cudaMalloc(&d_cold, bytes), "cudaMalloc(d_cold)");
    cudaCheck(cudaMalloc(&d_cnew, bytes), "cudaMalloc(d_cnew)");
    cudaCheck(cudaMalloc(&d_mu, bytes), "cudaMalloc(d_mu)");

    printf("Initializing concentration field...\n");
    constexpr int blockSize = 256;
    const int gridBlocks = static_cast<int>((gridSize + blockSize - 1) / blockSize);
    initializeKernel<<<gridBlocks, blockSize>>>(d_cold, gridSize);
    cudaCheck(cudaGetLastError(), "initializeKernel launch");
    cudaCheck(cudaDeviceSynchronize(), "initializeKernel");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<gridBlocks, blockSize>>>(d_cold, d_mu, nx, ny, nz,
                                                            gamma, e_AA, e_BB, e_AB);
        cudaCheck(cudaGetLastError(), "chemicalPotentialKernel launch");
        updateKernel<<<gridBlocks, blockSize>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D * dt);
        cudaCheck(cudaGetLastError(), "updateKernel launch");
        std::swap(d_cold, d_cnew);
    }
    cudaCheck(cudaDeviceSynchronize(), "simulation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    cudaCheck(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy result");
    cudaCheck(cudaFree(d_cold), "cudaFree(d_cold)");
    cudaCheck(cudaFree(d_cnew), "cudaFree(d_cnew)");
    cudaCheck(cudaFree(d_mu), "cudaFree(d_mu)");
    
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
