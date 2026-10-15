#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

// Laplacian with clamped boundary conditions
__device__ __forceinline__ double computeLaplacian(const double* __restrict__ c, const int nx, const int ny, const int nz,
                                                   const double idx2, const double idy2, const double idz2,
                                                   const int x, const int y, const int z) {
    const size_t sx = 1, sy = nx, sz = (size_t)nx * ny;
    const size_t i = z * sz + y * sy + x;
    const size_t ip = (x < nx - 1) ? i + sx : i;
    const size_t in = (x > 0) ? i - sx : i;
    const size_t jp = (y < ny - 1) ? i + sy : i;
    const size_t jn = (y > 0) ? i - sy : i;
    const size_t kp = (z < nz - 1) ? i + sz : i;
    const size_t kn = (z > 0) ? i - sz : i;
    const double c0 = __ldg(c + i);

    const double cxx = (__ldg(c + ip) + __ldg(c + in) - 2.0 * c0) * idx2;
    const double cyy = (__ldg(c + jp) + __ldg(c + jn) - 2.0 * c0) * idy2;
    const double czz = (__ldg(c + kp) + __ldg(c + kn) - 2.0 * c0) * idz2;
    return cxx + cyy + czz;
}

// Chemical potential kernel
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const int nx, const int ny, const int nz,
                                        const double idx2, const double idy2, const double idz2,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t idx = (size_t)z * nx * ny + (size_t)y * nx + x;
    const double cv = __ldg(c + idx);
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * computeLaplacian(c, nx, ny, nz, idx2, idy2, idz2, x, y, z);
}

// Cahn-Hilliard update kernel
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             const int nx, const int ny, const int nz,
                             const double Ddt, const double idx2, const double idy2, const double idz2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t idx = (size_t)z * nx * ny + (size_t)y * nx + x;
    cnew[idx] = __ldg(cold + idx) + Ddt * computeLaplacian(mu, nx, ny, nz, idx2, idy2, idz2, x, y, z);
}

// Initialize concentration field
__global__ void initKernel(double* __restrict__ c, const size_t vol) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vol) return;
    const double pseudo = ((((i + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[i] = -1.0 + 2.0 * pseudo;
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
    
    // Allocate arrays on the device
    std::vector<double> hresult(gridSize);
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, gridSize * sizeof(double)));

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initKernel<<<(unsigned)((gridSize + 255) / 256), 256>>>(cold, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 block(32, 4, 2);
    const dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                    (unsigned)((ny + block.y - 1) / block.y),
                    (unsigned)((nz + block.z - 1) / block.z));
    const double idx2 = 1.0 / (dx * dx), idy2 = 1.0 / (dy * dy), idz2 = 1.0 / (dz * dz);
    const double Ddt = dt * D;

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<grid, block>>>(cold, mu, (int)nx, (int)ny, (int)nz, idx2, idy2, idz2,
                                                 gamma, e_AA, e_BB, e_AB);
        updateKernel<<<grid, block>>>(cnew, cold, mu, (int)nx, (int)ny, (int)nz, Ddt, idx2, idy2, idz2);
        std::swap(cold, cnew);
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
    
    CUDA_CHECK(cudaMemcpy(hresult.data(), cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(cold);
    cudaFree(cnew);
    cudaFree(mu);

    // Print results for external validation
    if (printResults) {
        print_results(hresult, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(hresult, nx, ny, nz);
        
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
