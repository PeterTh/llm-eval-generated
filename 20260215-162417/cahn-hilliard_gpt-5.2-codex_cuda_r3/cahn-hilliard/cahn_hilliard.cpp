#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

inline void checkCuda(const cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

__device__ __forceinline__ int idx3d(const int x, const int y, const int z, const int nx, const int ny) {
    return (z * ny + y) * nx + x;
}

__device__ __forceinline__ double laplacian(const double* __restrict__ c, const int nx, const int ny, const int nz,
                                            const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                            const int x, const int y, const int z) {
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zp = (z < nz - 1) ? z + 1 : z;
    const int zn = (z > 0) ? z - 1 : 0;

    const int base = (z * ny + y) * nx;
    const int idx = base + x;
    const int idx_xp = base + xp;
    const int idx_xn = base + xn;

    const int base_yp = (z * ny + yp) * nx;
    const int base_yn = (z * ny + yn) * nx;

    const int base_zp = (zp * ny + y) * nx;
    const int base_zn = (zn * ny + y) * nx;

    const double cxx = (c[idx_xp] + c[idx_xn] - 2.0 * c[idx]) * inv_dx2;
    const double cyy = (c[base_yp + x] + c[base_yn + x] - 2.0 * c[idx]) * inv_dy2;
    const double czz = (c[base_zp + x] + c[base_zn + x] - 2.0 * c[idx]) * inv_dz2;

    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* c, const int nx, const int ny, const int nz, const unsigned long long vol) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const unsigned long long nxny = static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
    const unsigned long long linear_id = static_cast<unsigned long long>(z) * nxny +
                                         static_cast<unsigned long long>(y) * static_cast<unsigned long long>(nx) +
                                         static_cast<unsigned long long>(x);
    const unsigned long long value = (linear_id + 1ULL) * 1299709ULL;
    const double pseudo = static_cast<double>(value % vol) / static_cast<double>(vol);

    const int idx = idx3d(x, y, z, nx, ny);
    c[idx] = -1.0 + 2.0 * pseudo;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const int nx, const int ny, const int nz,
                                               const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int idx = idx3d(x, y, z, nx, ny);
    const double cv = c[idx];
    const double lap = laplacian(c, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
            + 3.0 * cv + cv * cv * cv
            - gamma * lap;
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                        const double* __restrict__ mu,
                                        const int nx, const int ny, const int nz,
                                        const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                        const double D, const double dt) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int idx = idx3d(x, y, z, nx, ny);
    const double lap = laplacian(mu, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);
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

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    const int nx_i = static_cast<int>(nx);
    const int ny_i = static_cast<int>(ny);
    const int nz_i = static_cast<int>(nz);

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const dim3 block(8, 8, 4);
    const dim3 grid((nx_i + block.x - 1) / block.x,
                    (ny_i + block.y - 1) / block.y,
                    (nz_i + block.z - 1) / block.z);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    checkCuda(cudaMalloc(&d_cold, bytes), "cudaMalloc d_cold");
    checkCuda(cudaMalloc(&d_cnew, bytes), "cudaMalloc d_cnew");
    checkCuda(cudaMalloc(&d_mu, bytes), "cudaMalloc d_mu");

    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<grid, block>>>(d_cold, nx_i, ny_i, nz_i, static_cast<unsigned long long>(gridSize));
    checkCuda(cudaGetLastError(), "initializeConcentrationKernel");
    checkCuda(cudaDeviceSynchronize(), "initializeConcentrationKernel sync");

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx_i, ny_i, nz_i,
                                                        inv_dx2, inv_dy2, inv_dz2,
                                                        gamma, e_AA, e_BB, e_AB);
        checkCuda(cudaGetLastError(), "computeChemicalPotentialKernel");

        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx_i, ny_i, nz_i,
                                                  inv_dx2, inv_dy2, inv_dz2, D, dt);
        checkCuda(cudaGetLastError(), "cahnHilliardUpdateKernel");

        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaDeviceSynchronize(), "simulation sync");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = static_cast<double>(gridSize) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold(gridSize);
    checkCuda(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_cold");

    checkCuda(cudaFree(d_mu), "cudaFree d_mu");
    checkCuda(cudaFree(d_cnew), "cudaFree d_cnew");
    checkCuda(cudaFree(d_cold), "cudaFree d_cold");

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
