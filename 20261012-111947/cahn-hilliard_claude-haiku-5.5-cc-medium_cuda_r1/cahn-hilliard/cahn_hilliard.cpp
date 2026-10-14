#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err_));                                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// Threads per block along the contiguous X dimension
constexpr unsigned int kBlockSizeX = 128;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
__device__ double computeLaplacian(const double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz,
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

// Map the 1D grid onto cells: blockIdx.x enumerates (row, x-block) pairs, where a row
// is one (y, z) line along X. Returns false for threads past the end of the row.
__device__ inline bool cellCoords(const size_t nx, const size_t ny, size_t& x, size_t& y, size_t& z) {
    const size_t blocksX = (nx + blockDim.x - 1) / blockDim.x;
    const size_t row = blockIdx.x / blocksX;
    x = (blockIdx.x % blocksX) * blockDim.x + threadIdx.x;
    y = row % ny;
    z = row / ny;
    return x < nx;
}

// Initialize concentration field: one thread per cell
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz) {
    size_t x, y, z;
    if (!cellCoords(nx, ny, x, y, z)) return;

    const size_t vol = nx * ny * nz;
    const size_t idx = idx3(x, y, z, nx, ny);
    // Generate pseudo-random value in [-1, 1]
    const size_t linear_id = z * (nx * ny) + y * nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[idx] = -1.0 + 2.0 * pseudo;
}

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    size_t x, y, z;
    if (!cellCoords(nx, ny, x, y, z)) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    const double cv = c[idx];

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt, const double dx, const double dy, const double dz) {
    size_t x, y, z;
    if (!cellCoords(nx, ny, x, y, z)) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + dt * D *
               computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
}

// Launch configuration: each (y, z) row is tiled into X blocks; see cellCoords()
inline unsigned int gridFor(const size_t nx, const size_t ny, const size_t nz) {
    const size_t blocksX = (nx + kBlockSizeX - 1) / kBlockSizeX;
    return static_cast<unsigned int>(blocksX * ny * nz);
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

    // Allocate device arrays
    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, gridSize * sizeof(double)));

    const dim3 block(kBlockSizeX, 1, 1);
    const dim3 grid(gridFor(nx, ny, nz), 1, 1);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    if (gridSize > 0) {
        initializeConcentrationKernel<<<grid, block>>>(cold, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations && gridSize > 0; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<grid, block>>>(cold, mu, nx, ny, nz, dx, dy, dz,
                                                        gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // Update concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        // Swap buffers
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy final concentration field back to the host
    std::vector<double> result(gridSize);
    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(result.data(), cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));

    // Print results for external validation
    if (printResults) {
        print_results(result, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, nx, ny, nz);

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
