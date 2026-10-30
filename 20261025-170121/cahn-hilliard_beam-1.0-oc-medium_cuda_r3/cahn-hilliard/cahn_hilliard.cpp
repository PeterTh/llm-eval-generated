#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

inline __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline __device__ double computeLaplacian(const double* c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

__global__ void kInitialize(double* c, size_t nx, size_t ny, size_t nz, size_t vol) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    if (i >= total) return;
    size_t z = i / (nx * ny);
    size_t rem = i % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;
    const size_t linear_id = z * (nx * ny) + y * nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[i] = -1.0 + 2.0 * pseudo;
}

__global__ void kChemPot(const double* c, double* mu, size_t nx, size_t ny, size_t nz,
                         double dx, double dy, double dz, double gamma, double e_AA, double e_BB, double e_AB) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    if (i >= total) return;
    size_t z = i / (nx * ny);
    size_t rem = i % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;
    const double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

__global__ void kUpdate(double* cnew, const double* cold, const double* mu, size_t nx, size_t ny, size_t nz,
                        double D, double dt, double dx, double dy, double dz) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    if (i >= total) return;
    size_t z = i / (nx * ny);
    size_t rem = i % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;
    cnew[i] = cold[i] + dt * D * computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
}

void initializeConcentration(double* c, size_t nx, size_t ny, size_t nz) {
    size_t vol = nx * ny * nz;
    size_t total = vol;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kInitialize<<<blocks, threads>>>(c, nx, ny, nz, vol);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeChemicalPotential(const double* c, double* mu, size_t nx, size_t ny, size_t nz,
                              double dx, double dy, double dz, double gamma, double e_AA, double e_BB, double e_AB) {
    size_t total = nx * ny * nz;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kChemPot<<<blocks, threads>>>(c, mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cahnHilliardUpdate(double* cnew, const double* cold, const double* mu, size_t nx, size_t ny, size_t nz,
                        double D, double dt, double dx, double dy, double dz) {
    size_t total = nx * ny * nz;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kUpdate<<<blocks, threads>>>(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool validateResult(const std::vector<double>& c, size_t nx, size_t ny, size_t nz) {
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

    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    printf("Initializing concentration field...\n");
    initializeConcentration(d_cold, nx, ny, nz);

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential(d_cold, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        cahnHilliardUpdate(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));

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
        }
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        return valid ? 0 : 1;
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    return 0;
}
