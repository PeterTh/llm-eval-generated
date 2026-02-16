#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include "../common/results_output.hpp"

// Error checking helper
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)

// 3D index calculation (host)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device helper: compute clamped neighbor index and Laplacian
__device__ inline double deviceLaplacian(const double* c, size_t nx, size_t ny, size_t nz,
                                         double dx, double dy, double dz, size_t x, size_t y, size_t z) {
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t zp = (z < nz - 1) ? z + 1 : z;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zn = (z > 0) ? z - 1 : 0;

    size_t idx = z * (nx * ny) + y * nx + x;
    size_t idx_xp = z * (nx * ny) + y * nx + xp;
    size_t idx_xn = z * (nx * ny) + y * nx + xn;
    size_t idx_yp = z * (nx * ny) + yp * nx + x;
    size_t idx_yn = z * (nx * ny) + yn * nx + x;
    size_t idx_zp = zp * (nx * ny) + y * nx + x;
    size_t idx_zn = zn * (nx * ny) + y * nx + x;

    double cxx = (c[idx_xp] + c[idx_xn] - 2.0 * c[idx]) / (dx * dx);
    double cyy = (c[idx_yp] + c[idx_yn] - 2.0 * c[idx]) / (dy * dy);
    double czz = (c[idx_zp] + c[idx_zn] - 2.0 * c[idx]) / (dz * dz);
    return cxx + cyy + czz;
}

// Kernel: compute chemical potential mu from concentration c
__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                                               size_t nx, size_t ny, size_t nz,
                                               double dx, double dy, double dz,
                                               double gamma, double e_AA, double e_BB, double e_AB) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vol = nx * ny * nz;
    if (idx >= vol) return;

    size_t z = idx / (nx * ny);
    size_t rem = idx % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;

    double cv = c[idx];
    double lap = deviceLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv - gamma * lap;
}

// Kernel: update concentration using Laplacian of mu
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                                         size_t nx, size_t ny, size_t nz,
                                         double D, double dt, double dx, double dy, double dz) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vol = nx * ny * nz;
    if (idx >= vol) return;

    size_t z = idx / (nx * ny);
    size_t rem = idx % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;

    double lap_mu = deviceLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    cnew[idx] = cold[idx] + dt * D * lap_mu;
}

// Kernel: initialize concentration deterministically
__global__ void initializeConcentrationKernel(double* c, size_t nx, size_t ny, size_t nz) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vol = nx * ny * nz;
    if (idx >= vol) return;
    size_t z = idx / (nx * ny);
    size_t rem = idx % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;

    size_t linear_id = z * (nx * ny) + y * nx + x;
    double pseudo = ((((linear_id + 1ULL) * 1299709ULL) % vol) / static_cast<double>(vol));
    c[idx] = -1.0 + 2.0 * pseudo;
}

// Host-side validation (unchanged semantics)
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

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
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

    size_t gridSize = nx * ny * nz;

    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);

    // Device buffers
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    // Initialize on device for best performance
    printf("Initializing concentration field...\n");
    size_t block = 256;
    size_t grid = (gridSize + block - 1) / block;
    initializeConcentrationKernel<<<grid, block>>>(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Optional copy back to host for potential external printing/validation later
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Main time-stepping loop on device
    for (int t = 0; t < iterations; ++t) {
        // compute mu from d_cold -> d_mu
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // update concentration d_cnew = d_cold + dt*D*laplacian(d_mu)
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        // swap pointers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = static_cast<double>(gridSize) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults) {
        print_results(cold, "Concentration");
    }

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        if (valid) {
            printf("Validation: PASSED\n");
            CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_cnew)); CUDA_CHECK(cudaFree(d_mu));
            return 0;
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_cnew)); CUDA_CHECK(cudaFree(d_mu));
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_cnew)); CUDA_CHECK(cudaFree(d_mu));
    return 0;
}
