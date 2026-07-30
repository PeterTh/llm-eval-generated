#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Device utilities
// ---------------------------------------------------------------------------
inline __device__ size_t idx3d(const size_t x, const size_t y, const size_t z,
                               const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Clamp helper (device)
inline __device__ size_t clamp_idx(size_t v, size_t lo, size_t hi) {
    return (v < lo) ? lo : (v >= hi) ? hi - 1 : v;
}

// ---------------------------------------------------------------------------
// Laplacian (device) – called from kernels
// ---------------------------------------------------------------------------
__device__ double computeLaplacianDev(const double* c, const size_t nx,
                                      const size_t ny, const size_t nz,
                                      const double dx, const double dy,
                                      const double dz,
                                      const size_t x, const size_t y,
                                      const size_t z) {
    const size_t xp = clamp_idx(x + 1, 0, nx);
    const size_t yp = clamp_idx(y + 1, 0, ny);
    const size_t zp = clamp_idx(z + 1, 0, nz);
    const size_t xn = clamp_idx(x > 0 ? x - 1 : 0, 0, nx);
    const size_t yn = clamp_idx(y > 0 ? y - 1 : 0, 0, ny);
    const size_t zn = clamp_idx(z > 0 ? z - 1 : 0, 0, nz);

    const double cx = c[idx3d(x, y, z, nx, ny)];
    const double cxx = (c[idx3d(xp, y, z, nx, ny)] +
                        c[idx3d(xn, y, z, nx, ny)] - 2.0 * cx) /
                       (dx * dx);
    const double cyy = (c[idx3d(x, yp, z, nx, ny)] +
                        c[idx3d(x, yn, z, nx, ny)] - 2.0 * cx) /
                       (dy * dy);
    const double czz = (c[idx3d(x, y, zp, nx, ny)] +
                        c[idx3d(x, y, zn, nx, ny)] - 2.0 * cx) /
                       (dz * dz);
    return cxx + cyy + czz;
}

// ---------------------------------------------------------------------------
// Kernel: initialize concentration
// ---------------------------------------------------------------------------
__global__ void initConcentrationKernel(double* c, const size_t nx,
                                        const size_t ny, const size_t nz,
                                        const size_t vol) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z);

    if (x < nx && y < ny && z < nz) {
        const size_t linear_id = idx3d(x, y, z, nx, ny);
        const double pseudo =
            ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[linear_id] = -1.0 + 2.0 * pseudo;
    }
}

// ---------------------------------------------------------------------------
// Kernel: compute chemical potential (shared-memory tiled stencil)
//
// Each 2D block loads an (bx+2)×(by+2) tile of `c` into shared memory
// (halo of 1 in XY).  Z-neighbors are read directly from global memory
// (they are coalesced across blocks).
// ---------------------------------------------------------------------------
template <unsigned int BX, unsigned int BY>
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB,
    const double e_AB) {

    constexpr unsigned int SH_BX = BX + 2;
    constexpr unsigned int SH_BY = BY + 2;

    __shared__ double sh[SH_BX][SH_BY];

    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z);

    // Shared-memory coordinates (with halo offset)
    const unsigned int sx = threadIdx.x + 1;
    const unsigned int sy = threadIdx.y + 1;

    // ---- Load tile with halo ----
    // Thread covers its own cell and halo neighbors in XY
    // We need to load SH_BX x SH_BY elements; each thread loads one.
    // Total threads = BX*BY, tile = (BX+2)*(BY+2).  Extra halo threads
    // are handled by a small epilogue below.
    {
        // Main cell + halo
        const size_t gx = (static_cast<size_t>(blockIdx.x) * BX + threadIdx.x);
        const size_t gy = (static_cast<size_t>(blockIdx.y) * BY + threadIdx.y);
        const size_t clx = clamp_idx(gx, 0, nx);
        const size_t cly = clamp_idx(gy, 0, ny);
        sh[sx][sy] = c[idx3d(clx, cly, z, nx, ny)];
    }

    // Load halo columns/rows that fall outside the main thread range
    // (only the threads on the edges need to do this)
    if (threadIdx.x == 0) {
        const size_t gx = static_cast<size_t>(blockIdx.x) * BX - 1;
        const size_t clx = clamp_idx(gx, 0, nx);
        for (unsigned int j = 0; j < BY; ++j) {
            const size_t gy = static_cast<size_t>(blockIdx.y) * BY + j;
            const size_t cly = clamp_idx(gy, 0, ny);
            sh[0][j + 1] = c[idx3d(clx, cly, z, nx, ny)];
        }
    }
    if (threadIdx.x == BX - 1) {
        const size_t gx = static_cast<size_t>(blockIdx.x) * BX + BX;
        const size_t clx = clamp_idx(gx, 0, nx);
        for (unsigned int j = 0; j < BY; ++j) {
            const size_t gy = static_cast<size_t>(blockIdx.y) * BY + j;
            const size_t cly = clamp_idx(gy, 0, ny);
            sh[SH_BX - 1][j + 1] = c[idx3d(clx, cly, z, nx, ny)];
        }
    }
    if (threadIdx.y == 0) {
        const size_t gy = static_cast<size_t>(blockIdx.y) * BY - 1;
        const size_t cly = clamp_idx(gy, 0, ny);
        for (unsigned int i = 0; i < BX; ++i) {
            const size_t gx = static_cast<size_t>(blockIdx.x) * BX + i;
            const size_t clx = clamp_idx(gx, 0, nx);
            sh[i + 1][0] = c[idx3d(clx, cly, z, nx, ny)];
        }
    }
    if (threadIdx.y == BY - 1) {
        const size_t gy = static_cast<size_t>(blockIdx.y) * BY + BY;
        const size_t cly = clamp_idx(gy, 0, ny);
        for (unsigned int i = 0; i < BX; ++i) {
            const size_t gx = static_cast<size_t>(blockIdx.x) * BX + i;
            const size_t clx = clamp_idx(gx, 0, nx);
            sh[i + 1][SH_BY - 1] = c[idx3d(clx, cly, z, nx, ny)];
        }
    }

    __syncthreads();

    // ---- Compute ----
    if (x < nx && y < ny && z < nz) {
        // Laplacian: XY from shared, Z from global
        const double cv = sh[sx][sy];
        const double lap_xy =
            (sh[sx + 1][sy] + sh[sx - 1][sy] - 2.0 * cv) / (dx * dx) +
            (sh[sx][sy + 1] + sh[sx][sy - 1] - 2.0 * cv) / (dy * dy);

        // Z neighbors from global memory
        const size_t zp = clamp_idx(z + 1, 0, nz);
        const size_t zn = (z > 0) ? z - 1 : 0;
        const double lap_z =
            (c[idx3d(x, y, zp, nx, ny)] + c[idx3d(x, y, zn, nx, ny)] -
             2.0 * cv) /
            (dz * dz);

        const double lap = lap_xy + lap_z;

        const size_t idx = idx3d(x, y, z, nx, ny);
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv - gamma * lap;
    }
}

// ---------------------------------------------------------------------------
// Kernel: Cahn-Hilliard update step (shared-memory tiled stencil)
// ---------------------------------------------------------------------------
template <unsigned int BX, unsigned int BY>
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t nz,
    const double D, const double dt, const double dx, const double dy,
    const double dz) {

    constexpr unsigned int SH_BX = BX + 2;
    constexpr unsigned int SH_BY = BY + 2;

    __shared__ double sh[SH_BX][SH_BY];

    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z);

    const unsigned int sx = threadIdx.x + 1;
    const unsigned int sy = threadIdx.y + 1;

    // ---- Load tile with halo ----
    {
        const size_t gx = static_cast<size_t>(blockIdx.x) * BX + threadIdx.x;
        const size_t gy = static_cast<size_t>(blockIdx.y) * BY + threadIdx.y;
        const size_t clx = clamp_idx(gx, 0, nx);
        const size_t cly = clamp_idx(gy, 0, ny);
        sh[sx][sy] = mu[idx3d(clx, cly, z, nx, ny)];
    }
    if (threadIdx.x == 0) {
        const size_t gx = static_cast<size_t>(blockIdx.x) * BX - 1;
        const size_t clx = clamp_idx(gx, 0, nx);
        for (unsigned int j = 0; j < BY; ++j) {
            const size_t gy = static_cast<size_t>(blockIdx.y) * BY + j;
            const size_t cly = clamp_idx(gy, 0, ny);
            sh[0][j + 1] = mu[idx3d(clx, cly, z, nx, ny)];
        }
    }
    if (threadIdx.x == BX - 1) {
        const size_t gx = static_cast<size_t>(blockIdx.x) * BX + BX;
        const size_t clx = clamp_idx(gx, 0, nx);
        for (unsigned int j = 0; j < BY; ++j) {
            const size_t gy = static_cast<size_t>(blockIdx.y) * BY + j;
            const size_t cly = clamp_idx(gy, 0, ny);
            sh[SH_BX - 1][j + 1] = mu[idx3d(clx, cly, z, nx, ny)];
        }
    }
    if (threadIdx.y == 0) {
        const size_t gy = static_cast<size_t>(blockIdx.y) * BY - 1;
        const size_t cly = clamp_idx(gy, 0, ny);
        for (unsigned int i = 0; i < BX; ++i) {
            const size_t gx = static_cast<size_t>(blockIdx.x) * BX + i;
            const size_t clx = clamp_idx(gx, 0, nx);
            sh[i + 1][0] = mu[idx3d(clx, cly, z, nx, ny)];
        }
    }
    if (threadIdx.y == BY - 1) {
        const size_t gy = static_cast<size_t>(blockIdx.y) * BY + BY;
        const size_t cly = clamp_idx(gy, 0, ny);
        for (unsigned int i = 0; i < BX; ++i) {
            const size_t gx = static_cast<size_t>(blockIdx.x) * BX + i;
            const size_t clx = clamp_idx(gx, 0, nx);
            sh[i + 1][SH_BY - 1] = mu[idx3d(clx, cly, z, nx, ny)];
        }
    }

    __syncthreads();

    // ---- Compute ----
    if (x < nx && y < ny && z < nz) {
        const double muv = sh[sx][sy];
        const double lap_xy =
            (sh[sx + 1][sy] + sh[sx - 1][sy] - 2.0 * muv) / (dx * dx) +
            (sh[sx][sy + 1] + sh[sx][sy - 1] - 2.0 * muv) / (dy * dy);

        const size_t zp = clamp_idx(z + 1, 0, nz);
        const size_t zn = (z > 0) ? z - 1 : 0;
        const double lap_z =
            (mu[idx3d(x, y, zp, nx, ny)] + mu[idx3d(x, y, zn, nx, ny)] -
             2.0 * muv) /
            (dz * dz);

        const double lap = lap_xy + lap_z;
        const size_t idx = idx3d(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * lap;
    }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
void launchInit(const size_t nx, const size_t ny, const size_t nz,
                double* d_c) {
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize((nx + 15) / 16, (ny + 15) / 16, nz);
    const size_t vol = nx * ny * nz;
    initConcentrationKernel<<<gridSize, blockSize>>>(d_c, nx, ny, nz, vol);
}

void launchComputeMu(const size_t nx, const size_t ny, const size_t nz,
                     const double* d_c, double* d_mu,
                     const double dx, const double dy, const double dz,
                     const double gamma, const double e_AA,
                     const double e_BB, const double e_AB) {
    constexpr unsigned int BX = 16;
    constexpr unsigned int BY = 16;
    const dim3 blockSize(BX, BY, 1);
    const dim3 gridSize((nx + BX - 1) / BX, (ny + BY - 1) / BY, nz);
    computeChemicalPotentialKernel<BX, BY><<<gridSize, blockSize>>>(
        d_c, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
}

void launchUpdate(const size_t nx, const size_t ny, const size_t nz,
                  double* d_cnew, const double* d_cold, const double* d_mu,
                  const double D, const double dt, const double dx,
                  const double dy, const double dz) {
    constexpr unsigned int BX = 16;
    constexpr unsigned int BY = 16;
    const dim3 blockSize(BX, BY, 1);
    const dim3 gridSize((nx + BX - 1) / BX, (ny + BY - 1) / BY, nz);
    cahnHilliardUpdateKernel<BX, BY><<<gridSize, blockSize>>>(
        d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
}

// ---------------------------------------------------------------------------
// Host-side helpers (unchanged semantics)
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
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
    const size_t memSize = gridSize * sizeof(double);

    // Allocate device memory
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_cold, memSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc cold failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_cnew, memSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc cnew failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_mu, memSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc mu failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    launchInit(nx, ny, nz, d_cold);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "init kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        launchComputeMu(nx, ny, nz, d_cold, d_mu, dx, dy, dz,
                        gamma, e_AA, e_BB, e_AB);
        launchUpdate(nx, ny, nz, d_cnew, d_cold, d_mu, D, dt, dx, dy, dz);

        // Swap device pointers (just swap the host-side pointers)
        std::swap(d_cold, d_cnew);
    }

    // Ensure all work is complete before timing
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Copy result back to host for validation / printing
    std::vector<double> h_cold(gridSize);
    cudaMemcpy(h_cold.data(), d_cold, memSize, cudaMemcpyDeviceToHost);

    // Calculate performance
    double cellUpdates = static_cast<double>(gridSize) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        print_results(h_cold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(h_cold, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    // Cleanup
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);

    return 0;
}
