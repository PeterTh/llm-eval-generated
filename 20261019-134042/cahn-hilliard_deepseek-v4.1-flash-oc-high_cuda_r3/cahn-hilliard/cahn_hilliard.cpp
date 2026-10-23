#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Thread-block dimensions. BX is a multiple of the warp size so that accesses
// along the contiguous X dimension are fully coalesced.
constexpr int BX = 32;
constexpr int BY = 2;
constexpr int BZ = 2;

// Compute the Laplacian directly from global memory. Neighbouring threads
// share stencil values through the read-only/L1 cache, and the clamped
// boundary conditions match the original scalar code exactly.
template <bool UNIT_SPACING>
__device__ __forceinline__ double laplacianGlobal(const double* __restrict__ f,
                                                  const int x, const int y, const int z,
                                                  const int nx, const int ny, const int nz,
                                                  const double invdx2, const double invdy2,
                                                  const double invdz2) {
    const size_t stride_y = nx;
    const size_t stride_z = static_cast<size_t>(nx) * ny;
    const size_t idx = static_cast<size_t>(z) * stride_z
                     + static_cast<size_t>(y) * stride_y + x;
    const double center = f[idx];
    const double xp = (x < nx - 1) ? f[idx + 1] : center;
    const double xn = (x > 0) ? f[idx - 1] : center;
    const double yp = (y < ny - 1) ? f[idx + stride_y] : center;
    const double yn = (y > 0) ? f[idx - stride_y] : center;
    const double zp = (z < nz - 1) ? f[idx + stride_z] : center;
    const double zn = (z > 0) ? f[idx - stride_z] : center;
    if constexpr (UNIT_SPACING) {
        return xp + xn + yp + yn + zp + zn - 6.0 * center;
    } else {
        const double cxx = (xp + xn - 2.0 * center) * invdx2;
        const double cyy = (yp + yn - 2.0 * center) * invdy2;
        const double czz = (zp + zn - 2.0 * center) * invdz2;
        return cxx + cyy + czz;
    }
}

// Compute chemical potential
template <bool UNIT_SPACING>
__global__ void computeChemicalPotentialKernel(double* __restrict__ mu,
                                               const double* __restrict__ c,
                                               const int nx, const int ny, const int nz,
                                               const double invdx2, const double invdy2,
                                               const double invdz2,
                                               const double gamma, const double coef_c,
                                               const double coef_c0) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    const int z = blockIdx.z * BZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = static_cast<size_t>(z) * (static_cast<size_t>(ny) * nx)
                     + static_cast<size_t>(y) * nx + x;
    const double cv = c[idx];
    const double lap = laplacianGlobal<UNIT_SPACING>(c, x, y, z, nx, ny, nz,
                                                     invdx2, invdy2, invdz2);

    // Algebraically equivalent to the original expression, with the
    // parameter-dependent terms folded into two host-computed coefficients.
    mu[idx] = cv * (coef_c + cv * cv) + coef_c0 - gamma * lap;
}

// Cahn-Hilliard update step
template <bool UNIT_SPACING>
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const int nx, const int ny, const int nz,
                                         const double dtD,
                                         const double invdx2, const double invdy2,
                                         const double invdz2) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    const int z = blockIdx.z * BZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = static_cast<size_t>(z) * (static_cast<size_t>(ny) * nx)
                     + static_cast<size_t>(y) * nx + x;
    const double lap = laplacianGlobal<UNIT_SPACING>(mu, x, y, z, nx, ny, nz,
                                                     invdx2, invdy2, invdz2);
    cnew[idx] = cold[idx] + dtD * lap;
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
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

template <bool UNIT_SPACING>
void launchStep(double* d_mu, double* d_cnew, const double* d_cold,
                const dim3& grid, const dim3& block,
                const int nx, const int ny, const int nz,
                const double invdx2, const double invdy2, const double invdz2,
                const double gamma, const double coef_c, const double coef_c0,
                const double dtD) {
    computeChemicalPotentialKernel<UNIT_SPACING><<<grid, block>>>(
        d_mu, d_cold, nx, ny, nz, invdx2, invdy2, invdz2, gamma, coef_c, coef_c0);
    cahnHilliardUpdateKernel<UNIT_SPACING><<<grid, block>>>(
        d_cnew, d_cold, d_mu, nx, ny, nz, dtD, invdx2, invdy2, invdz2);
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

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;
    const double coef_c = 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 3.0;
    const double coef_c0 = 4.5 * (e_AA - e_BB);

    size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    // Allocate arrays
    std::vector<double> cold(gridSize);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    // Allocate device buffers and upload the initial state before timing so
    // that the measured region matches the original computation loop.
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(BX, BY, BZ);
    const dim3 grid((static_cast<unsigned>(nx) + BX - 1) / BX,
                    (static_cast<unsigned>(ny) + BY - 1) / BY,
                    (static_cast<unsigned>(nz) + BZ - 1) / BZ);

    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);
    const bool unitSpacing = (invdx2 == 1.0 && invdy2 == 1.0 && invdz2 == 1.0);

    // Warm up the device kernels so that module loading and lazy context
    // initialization are not charged to the timed simulation loop. The warm-up
    // writes only to the scratch buffer, leaving the initial state untouched.
    if (unitSpacing) {
        launchStep<true>(d_mu, d_cnew, d_cold, grid, block, inx, iny, inz,
                         invdx2, invdy2, invdz2, gamma, coef_c, coef_c0, dtD);
    } else {
        launchStep<false>(d_mu, d_cnew, d_cold, grid, block, inx, iny, inz,
                          invdx2, invdy2, invdz2, gamma, coef_c, coef_c0, dtD);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (unitSpacing) {
        for (int t = 0; t < iterations; ++t) {
            launchStep<true>(d_mu, d_cnew, d_cold, grid, block, inx, iny, inz,
                             invdx2, invdy2, invdz2, gamma, coef_c, coef_c0, dtD);
            std::swap(d_cold, d_cnew);
        }
    } else {
        for (int t = 0; t < iterations; ++t) {
            launchStep<false>(d_mu, d_cnew, d_cold, grid, block, inx, iny, inz,
                              invdx2, invdy2, invdz2, gamma, coef_c, coef_c0, dtD);
            std::swap(d_cold, d_cnew);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy the final field back for reporting/validation.
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    printf("Computation time: %ld ms\n", duration.count());

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
