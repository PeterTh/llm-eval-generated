#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Thread block shape. x is the contiguous dimension, so each warp reads and
// writes consecutive doubles (fully coalesced accesses).
constexpr unsigned int BLOCK_X = 32;
constexpr unsigned int BLOCK_Y = 4;
constexpr unsigned int BLOCK_Z = 4;

// Physical parameters (compile-time constants, identical to the original values)
namespace phys {
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gamma = 0.5;
constexpr double D = 1.0;
}  // namespace phys

// Grid extents passed by value to the kernels
struct SimParams {
    size_t nx, ny, nz;
};

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
__device__ inline double computeLaplacian(const double* c, const SimParams& p,
                                          const size_t x, const size_t y, const size_t z) {
    const size_t nx = p.nx, ny = p.ny, nz = p.nz;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double center = c[idx3(x, y, z, nx, ny)];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * center) / (phys::dx * phys::dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * center) / (phys::dy * phys::dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * center) / (phys::dz * phys::dz);

    return cxx + cyy + czz;
}

// Initialize concentration field
__global__ void initConcentrationKernel(double* __restrict__ c, const SimParams p) {
    const size_t vol = p.nx * p.ny * p.nz;
    const size_t x0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y0 = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z0 = blockIdx.z * blockDim.z + threadIdx.z;
    const size_t xs = gridDim.x * blockDim.x;
    const size_t ys = gridDim.y * blockDim.y;
    const size_t zs = gridDim.z * blockDim.z;

    for (size_t z = z0; z < p.nz; z += zs) {
        for (size_t y = y0; y < p.ny; y += ys) {
            for (size_t x = x0; x < p.nx; x += xs) {
                const size_t idx = idx3(x, y, z, p.nx, p.ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = idx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu, const SimParams p) {
    const size_t x0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y0 = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z0 = blockIdx.z * blockDim.z + threadIdx.z;
    const size_t xs = gridDim.x * blockDim.x;
    const size_t ys = gridDim.y * blockDim.y;
    const size_t zs = gridDim.z * blockDim.z;

    for (size_t z = z0; z < p.nz; z += zs) {
        for (size_t y = y0; y < p.ny; y += ys) {
            for (size_t x = x0; x < p.nx; x += xs) {
                const size_t idx = idx3(x, y, z, p.nx, p.ny);
                const double cv = c[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * phys::e_AA + (cv - 1.0) * phys::e_BB - 2.0 * cv * phys::e_AB)
                          + 3.0 * cv + cv * cv * cv
                          - phys::gamma * computeLaplacian(c, p, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(const double* __restrict__ cold, double* __restrict__ cnew,
                                         const double* __restrict__ mu, const SimParams p) {
    const size_t x0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y0 = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z0 = blockIdx.z * blockDim.z + threadIdx.z;
    const size_t xs = gridDim.x * blockDim.x;
    const size_t ys = gridDim.y * blockDim.y;
    const size_t zs = gridDim.z * blockDim.z;

    for (size_t z = z0; z < p.nz; z += zs) {
        for (size_t y = y0; y < p.ny; y += ys) {
            for (size_t x = x0; x < p.nx; x += xs) {
                const size_t idx = idx3(x, y, z, p.nx, p.ny);
                cnew[idx] = cold[idx] + phys::dt * phys::D * computeLaplacian(mu, p, x, y, z);
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

    const size_t gridSize = nx * ny * nz;
    if (gridSize == 0) {
        fprintf(stderr, "Error: grid dimensions must be non-zero\n");
        return 1;
    }
    
    const SimParams p{nx, ny, nz};

    // Allocate device arrays
    const size_t bytes = gridSize * sizeof(double);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));

    // Grid is capped at 65535 blocks in y/z; kernels grid-stride over any excess
    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const unsigned int gx = static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X);
    const unsigned int gy = static_cast<unsigned int>(std::min<size_t>((ny + BLOCK_Y - 1) / BLOCK_Y, 65535));
    const unsigned int gz = static_cast<unsigned int>(std::min<size_t>((nz + BLOCK_Z - 1) / BLOCK_Z, 65535));
    const dim3 grid(gx, gy, gz);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initConcentrationKernel<<<grid, block>>>(d_cold, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation (all data stays on the device)
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, p);
        CUDA_CHECK(cudaGetLastError());

        // Update concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cold, d_cnew, d_mu, p);
        CUDA_CHECK(cudaGetLastError());
        
        // Swap buffers
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

    // Copy result back to host only when it is needed
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
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
