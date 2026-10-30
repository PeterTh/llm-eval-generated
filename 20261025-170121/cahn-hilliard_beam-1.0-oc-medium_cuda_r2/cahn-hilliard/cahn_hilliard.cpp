#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

inline __device__ size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

inline __device__ double laplacian(const double* c, const size_t nx, const size_t ny, const size_t nz,
                                 const double dx, const double dy, const double dz,
                                 const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    return cxx + cyy + czz;
}

__global__ void k_init(double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    while (i < vol) {
        const double pseudo = ((((i + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[i] = -1.0 + 2.0 * pseudo;
        i += stride;
    }
}

__global__ void k_chempot(const double* c, double* mu, const size_t nx, const size_t ny, const size_t nz,
                          const double dx, const double dy, const double dz,
                          const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t vol = nx * ny * nz;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    while (i < vol) {
        const size_t z = i / (nx * ny);
        const size_t rem = i % (nx * ny);
        const size_t y = rem / nx;
        const size_t x = rem % nx;
        const double cv = c[i];
        mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
               + 3.0 * cv + cv * cv * cv
               - gamma * laplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
        i += stride;
    }
}

__global__ void k_update(double* cnew, const double* cold, const double* mu,
                         const size_t nx, const size_t ny, const size_t nz,
                         const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t vol = nx * ny * nz;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    while (i < vol) {
        const size_t z = i / (nx * ny);
        const size_t rem = i % (nx * ny);
        const size_t y = rem / nx;
        const size_t x = rem % nx;
        cnew[i] = cold[i] + dt * D * laplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
        i += stride;
    }
}

int main(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printf("Usage: %s [options]\n", argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;
    size_t gridSize = nx * ny * nz;
    double *d_cold, *d_cnew, *d_mu;
    cudaMalloc(&d_cold, gridSize * sizeof(double));
    cudaMalloc(&d_cnew, gridSize * sizeof(double));
    cudaMalloc(&d_mu, gridSize * sizeof(double));
    int threads = 256;
    int blocks = (gridSize + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    printf("Initializing concentration field...\n");
    k_init<<<blocks, threads>>>(d_cold, nx, ny, nz);
    cudaDeviceSynchronize();
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        k_chempot<<<blocks, threads>>>(d_cold, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        k_update<<<blocks, threads>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
        std::swap(d_cold, d_cnew);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    std::vector<double> cold(gridSize);
    cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (printResults) print_results(cold, "Concentration");
    if (validate) {
        printf("Validating result...\n");
        bool valid = true;
        for (auto v : cold) { if (std::isnan(v) || std::isinf(v)) { printf("Validation failed: found NaN or Inf value\n"); valid = false; break; } }
        if (valid) {
            double minVal = cold[0], maxVal = cold[0];
            for (auto v : cold) { minVal = std::min(minVal, v); maxVal = std::max(maxVal, v); }
            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0) { printf("Validation failed: values out of expected range\n"); valid = false; }
        }
        if (valid) { printf("Validation: PASSED\n"); } else { printf("Validation: FAILED\n"); }
    }
    cudaFree(d_cold); cudaFree(d_cnew); cudaFree(d_mu);
    return validate ? 0 : 0;
}
