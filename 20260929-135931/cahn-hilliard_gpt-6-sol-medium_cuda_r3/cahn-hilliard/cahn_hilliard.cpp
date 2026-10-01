#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Unit grid spacing is fixed by the benchmark. A separate potential pass is
// needed because each update reads the neighboring potentials from one time step.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                            size_t x, size_t y, size_t z,
                                            size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const double center = field[i];
    const double xx = (field[i + (x + 1 < nx ? 1 : 0)] +
                       field[i - (x > 0 ? 1 : 0)] - 2.0 * center);
    const double yy = (field[i + (y + 1 < ny ? nx : 0)] +
                       field[i - (y > 0 ? nx : 0)] - 2.0 * center);
    const double zz = (field[i + (z + 1 < nz ? plane : 0)] +
                       field[i - (z > 0 ? plane : 0)] - 2.0 * center);
    return xx + yy + zz;
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t i = (z * ny + y) * nx + x;
    const double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * (-2.0 / 9.0) + (cv - 1.0) * (-2.0 / 9.0)
                   - 2.0 * cv * (2.0 / 9.0))
          + 3.0 * cv + cv * cv * cv
          - 0.5 * laplacian(c, x, y, z, nx, ny, nz);
}

__global__ void concentrationUpdate(const double* __restrict__ cold,
                                    const double* __restrict__ mu,
                                    double* __restrict__ cnew,
                                    size_t nx, size_t ny, size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t i = (z * ny + y) * nx + x;
    cnew[i] = cold[i] + 0.01 * laplacian(mu, x, y, z, nx, ny, nz);
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = (z * ny + y) * nx + x;
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
    
    size_t gridSize = nx * ny * nz;
    
    // Keep all three stencil fields on the GPU throughout the simulation.
    std::vector<double> cold(gridSize);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cold), bytes), "cudaMalloc concentration");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cnew), bytes), "cudaMalloc next concentration");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_mu), bytes), "cudaMalloc potential");
    checkCuda(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice), "copy initial concentration");

    const dim3 block(32, 4, 2);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (nz + block.z - 1) / block.z);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        chemicalPotential<<<grid, block>>>(d_cold, d_mu, nx, ny, nz);
        concentrationUpdate<<<grid, block>>>(d_cold, d_mu, d_cnew, nx, ny, nz);
        std::swap(d_cold, d_cnew);
    }
    checkCuda(cudaGetLastError(), "launch Cahn-Hilliard kernels");
    checkCuda(cudaDeviceSynchronize(), "run Cahn-Hilliard kernels");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost),
                  "copy final concentration");
    }
    checkCuda(cudaFree(d_cold), "cudaFree concentration");
    checkCuda(cudaFree(d_cnew), "cudaFree next concentration");
    checkCuda(cudaFree(d_mu), "cudaFree potential");
    
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
