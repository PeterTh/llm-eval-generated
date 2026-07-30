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
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Device function: compute Laplacian with clamped boundary conditions
__device__ __forceinline__ double deviceLaplacian(
    const double* __restrict__ field,
    const size_t x, const size_t y, const size_t z,
    const size_t nx, const size_t ny, const size_t nz,
    const double inv_dx2, const double inv_dy2, const double inv_dz2)
{
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const size_t nxy = nx * ny;
    const size_t idx_c  = z * nxy + y * nx + x;
    const size_t idx_xp = z * nxy + y * nx + xp;
    const size_t idx_xn = z * nxy + y * nx + xn;
    const size_t idx_yp = z * nxy + yp * nx + x;
    const size_t idx_yn = z * nxy + yn * nx + x;
    const size_t idx_zp = zp * nxy + y * nx + x;
    const size_t idx_zn = zn * nxy + y * nx + x;

    const double d2x = (field[idx_xp] + field[idx_xn] - 2.0 * field[idx_c]) * inv_dx2;
    const double d2y = (field[idx_yp] + field[idx_yn] - 2.0 * field[idx_c]) * inv_dy2;
    const double d2z = (field[idx_zp] + field[idx_zn] - 2.0 * field[idx_c]) * inv_dz2;

    return d2x + d2y + d2z;
}

// Kernel: Initialize concentration field
__global__ void initConcentrationKernel(
    double* __restrict__ c,
    const size_t nx, const size_t ny, const size_t nz,
    const size_t vol)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    if (tid >= total) return;

    const double pseudo = ((((tid + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[tid] = -1.0 + 2.0 * pseudo;
}

// Kernel: Compute chemical potential
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c,
    double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t nz,
    const double inv_dx2, const double inv_dy2, const double inv_dz2,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    if (tid >= total) return;

    const size_t x = tid % nx;
    const size_t y = (tid / nx) % ny;
    const size_t z = tid / (nx * ny);

    const double cv = c[tid];
    const double lap = deviceLaplacian(c, x, y, z, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2);

    mu[tid] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * lap;
}

// Kernel: Cahn-Hilliard update
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew,
    const double* __restrict__ cold,
    const double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t nz,
    const double dt_D,
    const double inv_dx2, const double inv_dy2, const double inv_dz2)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    if (tid >= total) return;

    const size_t x = tid % nx;
    const size_t y = (tid / nx) % ny;
    const size_t z = tid / (nx * ny);

    const double lap = deviceLaplacian(mu, x, y, z, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2);
    cnew[tid] = cold[tid] + dt_D * lap;
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

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

    printf("Cahn-Hilliard Phase Separation Benchmark (CUDA)\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

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
    const double dt_D = dt * D;

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    const int blockSize = 256;
    const int numBlocks = ((int)gridSize + blockSize - 1) / blockSize;

    // Allocate device memory
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));

    // Initialize concentration field on device
    printf("Initializing concentration field...\n");
    initConcentrationKernel<<<numBlocks, blockSize>>>(d_cold, nx, ny, nz, gridSize);
    CUDA_CHECK(cudaGetLastError());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<numBlocks, blockSize>>>(
            d_cold, d_mu, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
            gamma, e_AA, e_BB, e_AB);

        cahnHilliardUpdateKernel<<<numBlocks, blockSize>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, dt_D,
            inv_dx2, inv_dy2, inv_dz2);

        // Swap device pointers
        double* tmp = d_cold;
        d_cold = d_cnew;
        d_cnew = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy result back to host
    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));

    if (printResults) {
        print_results(cold, "Concentration");
    }

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(d_cold));
            CUDA_CHECK(cudaFree(d_cnew));
            CUDA_CHECK(cudaFree(d_mu));
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    return 0;
}
