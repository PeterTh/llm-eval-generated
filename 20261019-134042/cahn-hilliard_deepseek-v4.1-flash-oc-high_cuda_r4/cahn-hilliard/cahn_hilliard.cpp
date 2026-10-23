#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(err_));                                           \
            std::exit(EXIT_FAILURE);                                                          \
        }                                                                                     \
    } while (0)

// One thread per grid cell. A 16x2x16 block gives fully occupied SMs while
// keeping memory accesses coalesced along x.
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 2;
constexpr int BLOCK_Z = 16;
constexpr int BLOCK_SIZE = BLOCK_X * BLOCK_Y * BLOCK_Z;
constexpr int MIN_BLOCKS_PER_SM = 3;

__device__ __forceinline__ size_t globalIndex(int x, int y, int z, int nx, int ny) {
    return (static_cast<size_t>(z) * ny + y) * nx + x;
}

__global__ void initializeConcentrationKernel(double* __restrict__ c, size_t vol) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vol) {
        return;
    }
    const double pseudo =
        ((((i + 1) * static_cast<size_t>(1299709)) % vol) / static_cast<double>(vol));
    c[i] = -1.0 + 2.0 * pseudo;
}

// mu = f(c) - gamma * laplacian(c)
//
// The polynomial part of the reference chemical-potential expression is
// rearranged into coefficient form (mathematically identical):
//   4.5*((c+1)*eAA + (c-1)*eBB - 2*c*eAB) + 3*c + c^3
//     = c^3 + (4.5*(eAA+eBB-2*eAB)+3)*c + 4.5*(eAA-eBB)
// and evaluated with two FMAs.
template <bool UNIT_SPACING>
__global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM) void computeMuKernel(
    const double* __restrict__ c, double* __restrict__ mu, int nx, int ny, int nz,
    double invdx2, double invdy2, double invdz2, double invdiag, double gamma, double coefB,
    double coefC) {
    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const int z = static_cast<int>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int zp = (z < nz - 1) ? z + 1 : z;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zn = (z > 0) ? z - 1 : 0;

    const size_t idx = globalIndex(x, y, z, nx, ny);
    const double cv = c[idx];
    const double xsum =
        c[globalIndex(xp, y, z, nx, ny)] + c[globalIndex(xn, y, z, nx, ny)];
    const double ysum =
        c[globalIndex(x, yp, z, nx, ny)] + c[globalIndex(x, yn, z, nx, ny)];
    const double zsum =
        c[globalIndex(x, y, zp, nx, ny)] + c[globalIndex(x, y, zn, nx, ny)];

    double lap;
    if constexpr (UNIT_SPACING) {
        lap = fma(-6.0, cv, xsum + ysum + zsum);
    } else {
        lap = fma(-cv, invdiag, xsum * invdx2 + ysum * invdy2 + zsum * invdz2);
    }

    const double poly = fma(cv, fma(cv, cv, coefB), coefC);
    mu[idx] = fma(-gamma, lap, poly);
}

// c = c + dt*D*laplacian(mu)
// The update only reads c at the same cell, so it is safe to update in place.
template <bool UNIT_SPACING>
__global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM) void updateKernel(
    double* __restrict__ c, const double* __restrict__ mu, int nx, int ny, int nz,
    double invdx2, double invdy2, double invdz2, double invdiag, double dtD) {
    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const int z = static_cast<int>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int zp = (z < nz - 1) ? z + 1 : z;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zn = (z > 0) ? z - 1 : 0;

    const size_t idx = globalIndex(x, y, z, nx, ny);
    const double muc = mu[idx];
    const double xsum =
        mu[globalIndex(xp, y, z, nx, ny)] + mu[globalIndex(xn, y, z, nx, ny)];
    const double ysum =
        mu[globalIndex(x, yp, z, nx, ny)] + mu[globalIndex(x, yn, z, nx, ny)];
    const double zsum =
        mu[globalIndex(x, y, zp, nx, ny)] + mu[globalIndex(x, y, zn, nx, ny)];

    double lap;
    if constexpr (UNIT_SPACING) {
        lap = fma(-6.0, muc, xsum + ysum + zsum);
    } else {
        lap = fma(-muc, invdiag, xsum * invdx2 + ysum * invdy2 + zsum * invdz2);
    }

    c[idx] = fma(dtD, lap, c[idx]);
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double invdiag = 2.0 * (invdx2 + invdy2 + invdz2);
    const double coefB = 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 3.0;
    const double coefC = 4.5 * (e_AA - e_BB);
    const double dtD = dt * D;
    const bool unitSpacing = (dx == 1.0 && dy == 1.0 && dz == 1.0);

    CUDA_CHECK(cudaSetDevice(0));

    double* d_c = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    // Initialize concentration field on the device.
    printf("Initializing concentration field...\n");
    const unsigned int initBlocks = static_cast<unsigned int>((gridSize + 255) / 256);
    if (initBlocks > 0) {
        initializeConcentrationKernel<<<initBlocks, 256>>>(d_c, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((nz + BLOCK_Z - 1) / BLOCK_Z));
    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (unitSpacing) {
            computeMuKernel<true><<<grid, block>>>(d_c, d_mu, inx, iny, inz, invdx2, invdy2,
                                                   invdz2, invdiag, gamma, coefB, coefC);
            updateKernel<true><<<grid, block>>>(d_c, d_mu, inx, iny, inz, invdx2, invdy2,
                                                invdz2, invdiag, dtD);
        } else {
            computeMuKernel<false><<<grid, block>>>(d_c, d_mu, inx, iny, inz, invdx2, invdy2,
                                                    invdz2, invdiag, gamma, coefB, coefC);
            updateKernel<false><<<grid, block>>>(d_c, d_mu, inx, iny, inz, invdx2, invdy2,
                                                 invdz2, invdiag, dtD);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();

    printf("Computation time: %.3f ms\n", elapsedMs);

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = elapsedMs > 0.0 ? cellUpdates / (elapsedMs / 1000.0) / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Only transfer the final field back when it is actually observed.
    std::vector<double> hostCold;
    if (printResults || validate) {
        hostCold.resize(gridSize);
        CUDA_CHECK(
            cudaMemcpy(hostCold.data(), d_c, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_mu));

    // Print results for external validation
    if (printResults) {
        print_results(hostCold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(hostCold, nx, ny, nz);

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
