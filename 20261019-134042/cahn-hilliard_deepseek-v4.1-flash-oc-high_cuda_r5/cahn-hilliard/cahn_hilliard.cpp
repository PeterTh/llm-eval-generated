#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) cudaCheck((expr), __FILE__, __LINE__)

#define ZCOARSEN 8

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const int nx, const int ny, const int nz,
                                               const double gamma, const double coeff0, const double coeff1) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }
    const int z0 = (blockIdx.z * blockDim.z + threadIdx.z) * ZCOARSEN;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t base = static_cast<size_t>(y) * nx + x;
    const size_t offxp = (x + 1 < nx) ? 1 : 0;
    const size_t offxn = (x > 0) ? 1 : 0;
    const size_t offyp = (y + 1 < ny) ? static_cast<size_t>(nx) : 0;
    const size_t offyn = (y > 0) ? static_cast<size_t>(nx) : 0;

    #pragma unroll
    for (int k = 0; k < ZCOARSEN; ++k) {
        const int z = z0 + k;
        if (z >= nz) {
            break;
        }
        const size_t idx = static_cast<size_t>(z) * plane + base;
        const size_t offzp = (z + 1 < nz) ? plane : 0;
        const size_t offzn = (z > 0) ? plane : 0;

        const double cv = c[idx];
        const double s = c[idx + offxp] + c[idx - offxn]
                       + c[idx + offyp] + c[idx - offyn]
                       + c[idx + offzp] + c[idx - offzn];

        mu[idx] = (cv * cv * cv + coeff1 * cv + coeff0) - gamma * s;
    }
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ c, const double* __restrict__ mu,
                                         const int nx, const int ny, const int nz,
                                         const double dtD) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }
    const int z0 = (blockIdx.z * blockDim.z + threadIdx.z) * ZCOARSEN;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t base = static_cast<size_t>(y) * nx + x;
    const size_t offxp = (x + 1 < nx) ? 1 : 0;
    const size_t offxn = (x > 0) ? 1 : 0;
    const size_t offyp = (y + 1 < ny) ? static_cast<size_t>(nx) : 0;
    const size_t offyn = (y > 0) ? static_cast<size_t>(nx) : 0;

    #pragma unroll
    for (int k = 0; k < ZCOARSEN; ++k) {
        const int z = z0 + k;
        if (z >= nz) {
            break;
        }
        const size_t idx = static_cast<size_t>(z) * plane + base;
        const size_t offzp = (z + 1 < nz) ? plane : 0;
        const size_t offzn = (z > 0) ? plane : 0;

        const double cv = mu[idx];
        const double s = mu[idx + offxp] + mu[idx - offxn]
                       + mu[idx + offyp] + mu[idx - offyn]
                       + mu[idx + offzp] + mu[idx - offzn];

        c[idx] = c[idx] + dtD * (s - 6.0 * cv);
    }
}

// Initialize concentration field
__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                              const int nx, const int ny, const int nz,
                                              const unsigned long long vol) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x;
    const unsigned long long linear_id = static_cast<unsigned long long>(idx);
    const double pseudo = static_cast<double>(((linear_id + 1ull) * 1299709ull) % vol) /
                          static_cast<double>(vol);
    c[idx] = -1.0 + 2.0 * pseudo;
}

static inline dim3 makeBlockDim() {
    return dim3(32, 8, 1);
}

static inline dim3 makeGridDim(const size_t nx, const size_t ny, const size_t nz) {
    const dim3 block = makeBlockDim();
    return dim3(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                static_cast<unsigned int>((ny + block.y - 1) / block.y),
                static_cast<unsigned int>((nz + static_cast<size_t>(block.z) * ZCOARSEN - 1) /
                                          (static_cast<size_t>(block.z) * ZCOARSEN)));
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

    const double coeff0 = 4.5 * (e_AA - e_BB);
    const double coeff1 = 3.0 + 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 6.0 * gamma;
    const double dtD = dt * D;
    
    size_t gridSize = nx * ny * nz;
    
    CUDA_CHECK(cudaSetDevice(0));

    double* d_c = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), gridSize * sizeof(double)));

    const dim3 block = makeBlockDim();
    const dim3 grid = makeGridDim(nx, ny, nz);
    const dim3 gridInit(grid.x, grid.y, static_cast<unsigned int>(nz));

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<gridInit, block>>>(d_c,
                                                   static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz),
                                                   static_cast<unsigned long long>(gridSize));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<grid, block>>>(d_c, d_mu,
                                                        static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz),
                                                        gamma, coeff0, coeff1);
        
        // Update concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(d_c, d_mu,
                                                  static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz),
                                                  dtD);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> host_c;
    if (printResults || validate) {
        host_c.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(host_c.data(), d_c, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Print results for external validation
    if (printResults) {
        print_results(host_c, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(host_c, nx, ny, nz);
        
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
