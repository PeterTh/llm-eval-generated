#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ __forceinline__ size_t idx3_device(const size_t x, const size_t y, const size_t z,
                                              const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__device__ __forceinline__ double computeLaplacianDevice(const double* __restrict__ c,
                                                         const size_t nx, const size_t ny, const size_t nz,
                                                         const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                                         const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[idx3_device(xp, y, z, nx, ny)] + c[idx3_device(xn, y, z, nx, ny)]
                        - 2.0 * c[idx3_device(x, y, z, nx, ny)]) * inv_dx2;
    const double cyy = (c[idx3_device(x, yp, z, nx, ny)] + c[idx3_device(x, yn, z, nx, ny)]
                        - 2.0 * c[idx3_device(x, y, z, nx, ny)]) * inv_dy2;
    const double czz = (c[idx3_device(x, y, zp, nx, ny)] + c[idx3_device(x, y, zn, nx, ny)]
                        - 2.0 * c[idx3_device(x, y, z, nx, ny)]) * inv_dz2;

    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (idx >= vol) {
        return;
    }
    const unsigned long long linear_id = static_cast<unsigned long long>(idx);
    const unsigned long long mul = (linear_id + 1ULL) * 1299709ULL;
    const double pseudo = static_cast<double>(mul % static_cast<unsigned long long>(vol)) / static_cast<double>(vol);
    c[idx] = -1.0 + 2.0 * pseudo;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                               const double gamma, const double e_AA,
                                               const double e_BB, const double e_AB) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (idx >= vol) {
        return;
    }

    const size_t xy = nx * ny;
    const size_t z = idx / xy;
    const size_t rem = idx - z * xy;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;

    const double cv = c[idx];
    const double lap = computeLaplacianDevice(c, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * lap;
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                        const double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double D, const double dt,
                                        const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (idx >= vol) {
        return;
    }

    const size_t xy = nx * ny;
    const size_t z = idx / xy;
    const size_t rem = idx - z * xy;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;

    const double lap = computeLaplacianDevice(mu, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);
    cnew[idx] = cold[idx] + dt * D * lap;
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
    
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cold), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cnew), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), bytes));

    const int blockSize = 256;
    const int gridBlocks = static_cast<int>((gridSize + blockSize - 1) / blockSize);

    // Initialize concentration field on GPU
    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<gridBlocks, blockSize>>>(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<gridBlocks, blockSize>>>(d_cold, d_mu, nx, ny, nz,
                                                                  inv_dx2, inv_dy2, inv_dz2,
                                                                  gamma, e_AA, e_BB, e_AB);
        cahnHilliardUpdateKernel<<<gridBlocks, blockSize>>>(d_cnew, d_cold, d_mu, nx, ny, nz,
                                                           D, dt, inv_dx2, inv_dy2, inv_dz2);
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaGetLastError());

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    const double duration_ms = static_cast<double>(elapsedMs);
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double seconds = duration_ms / 1000.0;
    const double mcups = (seconds > 0.0) ? (cellUpdates / seconds / 1e6) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<double> cold(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));

        if (printResults) {
            print_results(cold, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(cold, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    return exitCode;
}
