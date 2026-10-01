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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each thread owns one cell. Adjacent threads access adjacent X cells, and the
// two kernels keep all intermediate fields in device memory between steps.
__device__ __forceinline__ double computeLaplacian(const double* __restrict__ field,
                                                    size_t nx, size_t ny, size_t nz,
                                                    size_t x, size_t y, size_t z,
                                                    size_t index, double dx, double dy, double dz) {
    const size_t plane = nx * ny;
    const double center = field[index];
    const double cxx = (field[index + (x + 1 < nx)] + field[index - (x > 0)] - 2.0 * center) / (dx * dx);
    const double cyy = (field[index + (y + 1 < ny) * nx] + field[index - (y > 0) * nx] - 2.0 * center) / (dy * dy);
    const double czz = (field[index + (z + 1 < nz) * plane] + field[index - (z > 0) * plane] - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                         size_t nx, size_t ny, size_t nz,
                                         double dx, double dy, double dz,
                                         double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t index = z * nx * ny + y * nx + x;
    const double cv = c[index];
    mu[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * computeLaplacian(c, nx, ny, nz, x, y, z, index, dx, dy, dz);
}

__global__ void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                                   const double* __restrict__ mu,
                                   size_t nx, size_t ny, size_t nz,
                                   double D, double dt, double dx, double dy, double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t index = z * nx * ny + y * nx + x;
    cnew[index] = cold[index] + dt * D *
                  computeLaplacian(mu, nx, ny, nz, x, y, z, index, dx, dy, dz);
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
    
    // Initialize on the host with the original deterministic sequence.
    std::vector<double> cold(gridSize);
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    double *deviceCold = nullptr, *deviceNew = nullptr, *deviceMu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    checkCuda(cudaMalloc(&deviceCold, bytes), "allocate concentration");
    checkCuda(cudaMalloc(&deviceNew, bytes), "allocate next concentration");
    checkCuda(cudaMalloc(&deviceMu, bytes), "allocate chemical potential");
    checkCuda(cudaMemcpy(deviceCold, cold.data(), bytes, cudaMemcpyHostToDevice), "upload concentration");

    const dim3 block(32, 4, 2);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (nz + block.z - 1) / block.z);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential<<<grid, block>>>(deviceCold, deviceMu, nx, ny, nz, dx, dy, dz,
                                                  gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdate<<<grid, block>>>(deviceNew, deviceCold, deviceMu,
                                            nx, ny, nz, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(deviceCold, deviceNew);
    }

    checkCuda(cudaGetLastError(), "launch simulation kernels");
    checkCuda(cudaDeviceSynchronize(), "run simulation kernels");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(cold.data(), deviceCold, bytes, cudaMemcpyDeviceToHost), "download concentration");
    }

    checkCuda(cudaFree(deviceCold), "free concentration");
    checkCuda(cudaFree(deviceNew), "free next concentration");
    checkCuda(cudaFree(deviceMu), "free chemical potential");
    
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
