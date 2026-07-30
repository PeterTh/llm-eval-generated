#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation - host and device
inline __host__ __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions - device version
__device__ inline double computeLaplacian(const double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz,
                                          const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (__ldg(&c[idx3(xp, y, z, nx, ny)]) + __ldg(&c[idx3(xn, y, z, nx, ny)]) -
                  2.0 * __ldg(&c[idx3(x, y, z, nx, ny)])) / (dx * dx);
    const double cyy = (__ldg(&c[idx3(x, yp, z, nx, ny)]) + __ldg(&c[idx3(x, yn, z, nx, ny)]) -
                  2.0 * __ldg(&c[idx3(x, y, z, nx, ny)])) / (dy * dy);
    const double czz = (__ldg(&c[idx3(x, y, zp, nx, ny)]) + __ldg(&c[idx3(x, y, zn, nx, ny)]) -
                  2.0 * __ldg(&c[idx3(x, y, z, nx, ny)])) / (dz * dz);

    return cxx + cyy + czz;
}

// Initialize concentration field kernel
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t linear_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (linear_id < vol) {
        // Generate pseudo-random value in [-1, 1]
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[linear_id] = -1.0 + 2.0 * pseudo;
    }
}

// Compute chemical potential kernel
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                                const size_t nx, const size_t ny, const size_t nz,
                                                const double dx, const double dy, const double dz,
                                                const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t vol = nx * ny * nz;
    const size_t linear_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (linear_id < vol) {
        const size_t x = linear_id % nx;
        const size_t y = (linear_id / nx) % ny;
        const size_t z = linear_id / (nx * ny);

        const double cv = __ldg(&c[linear_id]);

        mu[linear_id] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// Cahn-Hilliard update step kernel
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t vol = nx * ny * nz;
    const size_t linear_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (linear_id < vol) {
        const size_t x = linear_id % nx;
        const size_t y = (linear_id / nx) % ny;
        const size_t z = linear_id / (nx * ny);

        cnew[linear_id] = __ldg(&cold[linear_id]) + dt * D *
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// CUDA helper for error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err), #call);         \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// Host-side Laplacian for validation (kept for reference, not used in GPU path)
double computeLaplacianHost(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
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

    // Allocate GPU memory
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    // Determine launch configuration
    const int blockSize = 256;
    const int numBlocks = (static_cast<int>(gridSize) + blockSize - 1) / blockSize;

    // Initialize concentration field on GPU
    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<numBlocks, blockSize>>>(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<numBlocks, blockSize>>>(
            d_cold, d_mu, nx, ny, nz, dx, dy, dz,
            gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // Update concentration
        cahnHilliardUpdateKernel<<<numBlocks, blockSize>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        // Swap pointers on GPU (no data movement needed)
        double* temp = d_cold;
        d_cold = d_cnew;
        d_cnew = temp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Copy result back to host for validation/output
    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
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
        } else {
            printf("Validation: FAILED\n");
        }

        // Free GPU memory
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));

        return valid ? 0 : 1;
    }

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    return 0;
}
