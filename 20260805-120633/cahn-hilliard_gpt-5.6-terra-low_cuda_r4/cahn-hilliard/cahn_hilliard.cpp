#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation.  This is used by both host code and CUDA kernels.
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                  const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__device__ __forceinline__ double computeLaplacian(const double* const c, const size_t nx, const size_t ny,
                                                    const size_t nz, const double invDx2, const double invDy2,
                                                    const double invDz2, const size_t x, const size_t y,
                                                    const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) * invDx2;
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) * invDy2;
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) * invDz2;
    
    return cxx + cyy + czz;
}

// Each cell is handled independently by one CUDA thread.
__global__ void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double invDx2, const double invDy2, const double invDz2,
                                         const double gamma, const double e_AA, const double e_BB,
                                         const double e_AB) {
    const size_t x = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t y = blockIdx.y * static_cast<size_t>(blockDim.y) + threadIdx.y;
    const size_t z = blockIdx.z * static_cast<size_t>(blockDim.z) + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        const double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv
                - gamma * computeLaplacian(c, nx, ny, nz, invDx2, invDy2, invDz2, x, y, z);
    }
}

__global__ void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                                   const double* __restrict__ mu, const size_t nx,
                                   const size_t ny, const size_t nz, const double D, const double dt,
                                   const double invDx2, const double invDy2, const double invDz2) {
    const size_t x = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t y = blockIdx.y * static_cast<size_t>(blockDim.y) + threadIdx.y;
    const size_t z = blockIdx.z * static_cast<size_t>(blockDim.z) + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D *
                    computeLaplacian(mu, nx, ny, nz, invDx2, invDy2, invDz2, x, y, z);
    }
}

inline void checkCuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
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
    
    // Initialize on the host once, then retain all working fields on the GPU.
    // This avoids PCIe transfers inside the timed time-stepping loop.
    std::vector<double> cold(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    checkCuda(cudaMalloc(&d_cold, bytes), "allocating concentration field");
    checkCuda(cudaMalloc(&d_cnew, bytes), "allocating update field");
    checkCuda(cudaMalloc(&d_mu, bytes), "allocating chemical potential field");
    checkCuda(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice), "uploading concentration field");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    // X-major blocks preserve fully coalesced accesses for the contiguous
    // dimension and avoid integer division when reconstructing coordinates.
    constexpr unsigned int blockX = 32;
    constexpr unsigned int blockY = 4;
    constexpr unsigned int blockZ = 2;
    const dim3 threads(blockX, blockY, blockZ);
    const dim3 blocks(static_cast<unsigned int>((nx + blockX - 1) / blockX),
                      static_cast<unsigned int>((ny + blockY - 1) / blockY),
                      static_cast<unsigned int>((nz + blockZ - 1) / blockZ));

    checkCuda(cudaDeviceSynchronize(), "synchronizing before timing");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential<<<blocks, threads>>>(d_cold, d_mu, nx, ny, nz,
                                                               invDx2, invDy2, invDz2, gamma, e_AA, e_BB, e_AB);
        cahnHilliardUpdate<<<blocks, threads>>>(d_cnew, d_cold, d_mu, nx, ny, nz,
                                                        D, dt, invDx2, invDy2, invDz2);
        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaGetLastError(), "launching simulation kernels");
    checkCuda(cudaDeviceSynchronize(), "synchronizing simulation kernels");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    if (printResults || validate) {
        checkCuda(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost), "downloading concentration field");
    }

    checkCuda(cudaFree(d_mu), "freeing chemical potential field");
    checkCuda(cudaFree(d_cnew), "freeing update field");
    checkCuda(cudaFree(d_cold), "freeing concentration field");

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
