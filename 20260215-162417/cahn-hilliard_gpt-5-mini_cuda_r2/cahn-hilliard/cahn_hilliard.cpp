#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Helper to compute 3D index
inline __host__ __device__ size_t idx3_d(const int x, const int y, const int z, const int nx, const int ny) {
    return static_cast<size_t>(z) * (static_cast<size_t>(nx) * static_cast<size_t>(ny)) + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

// CUDA kernel: compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               int nx, int ny, int nz,
                                               double dx, double dy, double dz,
                                               double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t gridSize = static_cast<size_t>(nx) * nx * 0 + static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz); // placeholder to avoid warning
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    for (; gid < total; gid += blockDim.x * gridDim.x) {
        int x = gid % nx;
        int y = (gid / nx) % ny;
        int z = gid / (nx * ny);
        double cv = c[gid];
        int xp = (x < nx - 1) ? x + 1 : x;
        int yp = (y < ny - 1) ? y + 1 : y;
        int zp = (z < nz - 1) ? z + 1 : z;
        int xn = (x > 0) ? x - 1 : 0;
        int yn = (y > 0) ? y - 1 : 0;
        int zn = (z > 0) ? z - 1 : 0;

        double cxx = (c[idx3_d(xp, y, z, nx, ny)] + c[idx3_d(xn, y, z, nx, ny)] - 2.0 * c[gid]) / (dx * dx);
        double cyy = (c[idx3_d(x, yp, z, nx, ny)] + c[idx3_d(x, yn, z, nx, ny)] - 2.0 * c[gid]) / (dy * dy);
        double czz = (c[idx3_d(x, y, zp, nx, ny)] + c[idx3_d(x, y, zn, nx, ny)] - 2.0 * c[gid]) / (dz * dz);

        mu[gid] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                  + 3.0 * cv + cv * cv * cv
                  - gamma * (cxx + cyy + czz);
    }
}

// CUDA kernel: update concentration
__global__ void updateKernel(const double* __restrict__ cold, const double* __restrict__ mu, double* __restrict__ cnew,
                             int nx, int ny, int nz, double D, double dt, double dx, double dy, double dz) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    for (; gid < total; gid += blockDim.x * gridDim.x) {
        int x = gid % nx;
        int y = (gid / nx) % ny;
        int z = gid / (nx * ny);

        int xp = (x < nx - 1) ? x + 1 : x;
        int yp = (y < ny - 1) ? y + 1 : y;
        int zp = (z < nz - 1) ? z + 1 : z;
        int xn = (x > 0) ? x - 1 : 0;
        int yn = (y > 0) ? y - 1 : 0;
        int zn = (z > 0) ? z - 1 : 0;

        double mxx = (mu[idx3_d(xp, y, z, nx, ny)] + mu[idx3_d(xn, y, z, nx, ny)] - 2.0 * mu[gid]) / (dx * dx);
        double myy = (mu[idx3_d(x, yp, z, nx, ny)] + mu[idx3_d(x, yn, z, nx, ny)] - 2.0 * mu[gid]) / (dy * dy);
        double mzz = (mu[idx3_d(x, y, zp, nx, ny)] + mu[idx3_d(x, y, zn, nx, ny)] - 2.0 * mu[gid]) / (dz * dz);

        cnew[gid] = cold[gid] + dt * D * (mxx + myy + mzz);
    }
}

// Initialize concentration field (host)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_d(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(nx), static_cast<int>(ny));
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
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

    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    // Allocate device memory
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice));

    // Determine launch configuration
    int threads = 256;
    int blocks = static_cast<int>((gridSize + threads - 1) / threads);
    // Cap blocks to a reasonable number to avoid excessive launch counts
    int maxBlocks = 65535;
    if (blocks > maxBlocks) blocks = maxBlocks;

    printf("Running Cahn-Hilliard simulation (GPU)...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<blocks, threads>>>(d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        updateKernel<<<blocks, threads>>>(d_cold, d_mu, d_cnew, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        // swap pointers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
