#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned int threadsPerBlock = 256;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                      \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(error));                 \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

__device__ __forceinline__ size_t idx3(const size_t x, const size_t y,
                                       const size_t z, const size_t nx,
                                       const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// The clamped neighbor selection exactly matches the original CPU stencil.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                            const size_t nx, const size_t ny,
                                            const size_t nz, const double dx2,
                                            const double dy2, const double dz2,
                                            const size_t x, const size_t y,
                                            const size_t z) noexcept {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t zp = (z + 1 < nz) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    const size_t center = idx3(x, y, z, nx, ny);

    const double cxx = (field[idx3(xp, y, z, nx, ny)] +
                        field[idx3(xn, y, z, nx, ny)] - 2.0 * field[center]) / dx2;
    const double cyy = (field[idx3(x, yp, z, nx, ny)] +
                        field[idx3(x, yn, z, nx, ny)] - 2.0 * field[center]) / dy2;
    const double czz = (field[idx3(x, y, zp, nx, ny)] +
                        field[idx3(x, y, zn, nx, ny)] - 2.0 * field[center]) / dz2;
    return cxx + cyy + czz;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                        double* __restrict__ mu, size_t count,
                                        size_t nx, size_t ny, size_t nz,
                                        double dx, double dy, double dz,
                                        double gamma, double e_AA, double e_BB,
                                        double e_AB) {
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count; linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t plane = nx * ny;
        const size_t z = linear / plane;
        const size_t rem = linear - z * plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        const double cv = c[linear];
        mu[linear] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                            2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv -
                     gamma * laplacian(c, nx, ny, nz, dx * dx, dy * dy, dz * dz,
                                       x, y, z);
    }
}

__global__ void updateKernel(double* __restrict__ cnew,
                             const double* __restrict__ cold,
                             const double* __restrict__ mu, size_t count,
                             size_t nx, size_t ny, size_t nz, double D, double dt,
                             double dx, double dy, double dz) {
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count; linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t plane = nx * ny;
        const size_t z = linear / plane;
        const size_t rem = linear - z * plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        cnew[linear] = cold[linear] + dt * D *
                       laplacian(mu, nx, ny, nz, dx * dx, dy * dy, dz * dz,
                                 x, y, z);
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx,
                             const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) /
                                       static_cast<double>(vol));
                c[linear_id] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c) {
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

} // namespace

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
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    const double gamma = 0.5, D = 1.0;
    const size_t gridSize = nx * ny * nz;
    std::vector<double> cold(gridSize), cnew(gridSize), mu(gridSize);
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>(
        (gridSize + threadsPerBlock - 1) / threadsPerBlock, 65535));
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, threadsPerBlock>>>(
            d_cold, d_mu, gridSize, nx, ny, nz, dx, dy, dz,
            gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
        updateKernel<<<blocks, threadsPerBlock>>>(
            d_cnew, d_cold, d_mu, gridSize, nx, ny, nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", duration.count());
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Performance: %.3f MCellUpdates/s\n",
           static_cast<double>(gridSize) * iterations / seconds / 1e6);
    if (printResults) print_results(cold, "Concentration");
    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(cold);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
