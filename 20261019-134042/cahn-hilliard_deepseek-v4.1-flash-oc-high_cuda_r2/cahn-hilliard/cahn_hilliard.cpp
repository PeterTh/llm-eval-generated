#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// Compute Laplacian with clamped boundary conditions on device.
__device__ __forceinline__ double computeLaplacian(const double* __restrict__ c,
                                                   const size_t idx,
                                                   const int x, const int y, const int z,
                                                   const int nx, const int ny, const int nz,
                                                   const ptrdiff_t sx, const ptrdiff_t sy, const ptrdiff_t sz,
                                                   const double invdx2, const double invdy2, const double invdz2,
                                                   const double centerCoeff) {
    double xm, xp, ym, yp, zm, zp;

    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1) {
        // Interior fast path: direct neighbour offsets, no clamping math.
        xm = c[idx - sx];
        xp = c[idx + sx];
        ym = c[idx - sy];
        yp = c[idx + sy];
        zm = c[idx - sz];
        zp = c[idx + sz];
    } else {
        // Clamped boundary conditions (edge values are replicated).
        const int im = (x > 0) ? x - 1 : x;
        const int ip = (x < nx - 1) ? x + 1 : x;
        const int jm = (y > 0) ? y - 1 : y;
        const int jp = (y < ny - 1) ? y + 1 : y;
        const int km = (z > 0) ? z - 1 : z;
        const int kp = (z < nz - 1) ? z + 1 : z;

        xm = c[idx + static_cast<ptrdiff_t>(im - x) * sx];
        xp = c[idx + static_cast<ptrdiff_t>(ip - x) * sx];
        ym = c[idx + static_cast<ptrdiff_t>(jm - y) * sy];
        yp = c[idx + static_cast<ptrdiff_t>(jp - y) * sy];
        zm = c[idx + static_cast<ptrdiff_t>(km - z) * sz];
        zp = c[idx + static_cast<ptrdiff_t>(kp - z) * sz];
    }

    const double cc = c[idx];

    return (xm + xp) * invdx2 + (ym + yp) * invdy2 + (zm + zp) * invdz2
         - cc * centerCoeff;
}

// Compute chemical potential on device.
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const int nx, const int ny, const int nz,
                                               const double invdx2, const double invdy2, const double invdz2,
                                               const double centerCoeff,
                                               const double gamma, const double coeffA, const double coeffB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const ptrdiff_t sx = 1;
    const ptrdiff_t sy = nx;
    const ptrdiff_t sz = static_cast<ptrdiff_t>(nx) * ny;
    const size_t idx = static_cast<size_t>(z) * static_cast<size_t>(sz) +
                       static_cast<size_t>(y) * static_cast<size_t>(nx) +
                       static_cast<size_t>(x);

    const double cv = c[idx];
    const double lap = computeLaplacian(c, idx, x, y, z, nx, ny, nz, sx, sy, sz,
                                        invdx2, invdy2, invdz2, centerCoeff);

    mu[idx] = coeffA * cv + coeffB + cv * cv * cv
            - gamma * lap;
}

// Cahn-Hilliard update step on device.
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const int nx, const int ny, const int nz,
                                         const double D, const double dt,
                                         const double invdx2, const double invdy2, const double invdz2,
                                         const double centerCoeff) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const ptrdiff_t sx = 1;
    const ptrdiff_t sy = nx;
    const ptrdiff_t sz = static_cast<ptrdiff_t>(nx) * ny;
    const size_t idx = static_cast<size_t>(z) * static_cast<size_t>(sz) +
                       static_cast<size_t>(y) * static_cast<size_t>(nx) +
                       static_cast<size_t>(x);

    const double lap = computeLaplacian(mu, idx, x, y, z, nx, ny, nz, sx, sy, sz,
                                        invdx2, invdy2, invdz2, centerCoeff);

    cnew[idx] = cold[idx] + dt * D * lap;
}

// Initialize concentration field on device.
__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                              const unsigned long long vol) {
    const unsigned long long linear_id =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_id >= vol) return;

    // Generate pseudo-random value in [-1, 1] (identical to reference implementation)
    const double pseudo = static_cast<double>(((linear_id + 1ULL) * 1299709ULL) % vol) /
                          static_cast<double>(vol);
    c[linear_id] = -1.0 + 2.0 * pseudo;
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

    // Fold the constant interaction terms of the chemical potential into two
    // coefficients (algebraically identical to the reference expression).
    const double coeffA = 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 3.0;
    const double coeffB = 4.5 * (e_AA - e_BB);

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double centerCoeff = 2.0 * (invdx2 + invdy2 + invdz2);

    size_t gridSize = nx * ny * nz;

    // Host buffer for the final result
    std::vector<double> cold(gridSize);

    // Device arrays
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    CUDA_CHECK(cudaSetDevice(0));

    if (gridSize > 0) {
        const size_t bytes = gridSize * sizeof(double);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cold), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cnew), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), bytes));

        // Initialize concentration field on the device
        printf("Initializing concentration field...\n");
        const unsigned int initThreads = 256;
        const unsigned int initBlocks =
            static_cast<unsigned int>((gridSize + initThreads - 1) / initThreads);
        initializeConcentrationKernel<<<initBlocks, initThreads>>>(
            d_cold, static_cast<unsigned long long>(gridSize));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        printf("Initializing concentration field...\n");
    }

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (gridSize > 0) {
        const dim3 block(32, 4, 2);
        const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                        static_cast<unsigned int>((ny + block.y - 1) / block.y),
                        static_cast<unsigned int>((nz + block.z - 1) / block.z));

        const int inx = static_cast<int>(nx);
        const int iny = static_cast<int>(ny);
        const int inz = static_cast<int>(nz);

        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotentialKernel<<<grid, block>>>(
                d_cold, d_mu, inx, iny, inz,
                invdx2, invdy2, invdz2, centerCoeff, gamma, coeffA, coeffB);

            // Update concentration
            cahnHilliardUpdateKernel<<<grid, block>>>(
                d_cnew, d_cold, d_mu, inx, iny, inz,
                D, dt, invdx2, invdy2, invdz2, centerCoeff);

            // Swap buffers
            std::swap(d_cold, d_cnew);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy the final concentration field back to the host
    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
    }

    printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
