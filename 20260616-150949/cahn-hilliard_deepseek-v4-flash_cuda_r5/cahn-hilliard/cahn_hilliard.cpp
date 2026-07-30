#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 3D index calculation (host+device)
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                  const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device function: Laplacian with clamped boundary conditions
__device__ double deviceLaplacian(const double* __restrict__ c,
                                  const size_t nx, const size_t ny, const size_t nz,
                                  const double idx2x, const double idx2y, const double idx2z,
                                  const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const size_t base = idx3(x, y, z, nx, ny);
    const double c_center = c[base];

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                        2.0 * c_center) * idx2x;
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                        2.0 * c_center) * idx2y;
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                        2.0 * c_center) * idx2z;

    return cxx + cyy + czz;
}

// CUDA kernel: Compute chemical potential
__global__ void chemicalPotentialKernel(double* __restrict__ mu, const double* __restrict__ c,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double idx2x, const double idx2y, const double idx2z,
                                        const double gamma,
                                        const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        const double cv = c[idx];

        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                  + 3.0 * cv + cv * cv * cv
                  - gamma * deviceLaplacian(c, nx, ny, nz, idx2x, idx2y, idx2z, x, y, z);
    }
}

// CUDA kernel: Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double Ddt,
                                         const double idx2x, const double idx2y, const double idx2z) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + Ddt * deviceLaplacian(mu, nx, ny, nz, idx2x, idx2y, idx2z, x, y, z);
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
    
    // Precompute reciprocal squares to avoid division in kernels
    const double idx2x = 1.0 / (dx * dx);
    const double idx2y = 1.0 / (dy * dy);
    const double idx2z = 1.0 / (dz * dz);
    const double Ddt = dt * D;
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate host arrays
    std::vector<double> cold(gridSize);
    
    // Initialize concentration field on host
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    
    // Allocate device memory
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu,  gridSize * sizeof(double)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), gridSize * sizeof(double),
                          cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters (32 x 4 x 4 = 512 threads/block)
    dim3 block(32, 4, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);
    
    // Run simulation on GPU
    printf("Running Cahn-Hilliard simulation on GPU...\n");
    
    // Create CUDA events for accurate GPU timing
    cudaEvent_t cudaStart, cudaStop;
    CUDA_CHECK(cudaEventCreate(&cudaStart));
    CUDA_CHECK(cudaEventCreate(&cudaStop));
    
    CUDA_CHECK(cudaEventRecord(cudaStart));
    
    // Pointer swap tracking: keep original allocations for cleanup
    double *d_cur = d_cold;
    double *d_nxt = d_cnew;
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential on GPU
        chemicalPotentialKernel<<<grid, block>>>(d_mu, d_cur, nx, ny, nz,
                                                  idx2x, idx2y, idx2z,
                                                  gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
        
        // Update concentration on GPU
        cahnHilliardUpdateKernel<<<grid, block>>>(d_nxt, d_cur, d_mu, nx, ny, nz,
                                                   Ddt, idx2x, idx2y, idx2z);
        CUDA_CHECK(cudaGetLastError());
        
        // Swap device pointers
        double *tmp = d_cur;
        d_cur = d_nxt;
        d_nxt = tmp;
    }
    
    CUDA_CHECK(cudaEventRecord(cudaStop));
    CUDA_CHECK(cudaEventSynchronize(cudaStop));
    
    float gpuMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, cudaStart, cudaStop));
    
    CUDA_CHECK(cudaEventDestroy(cudaStart));
    CUDA_CHECK(cudaEventDestroy(cudaStop));
    
    // Copy final result back to host (d_cur holds latest state after swaps)
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cur, gridSize * sizeof(double),
                          cudaMemcpyDeviceToHost));
    
    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
    long durationMs = static_cast<long>(gpuMs);
    printf("Computation time: %ld ms\n", durationMs);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (gpuMs / 1000.0) / 1e6;
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
