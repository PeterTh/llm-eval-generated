#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static constexpr int BLOCK_SIZE = 256;

// 3D index calculation
__device__ __host__ inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Device Laplacian with clamped boundary conditions
__device__ double computeLaplacianDevice(const double* __restrict__ c,
                                          size_t nx, size_t ny, size_t nz,
                                          double inv_dx2, double inv_dy2, double inv_dz2,
                                          size_t x, size_t y, size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double center = c[idx3(x, y, z, nx, ny)];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * center) * inv_dx2;
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * center) * inv_dy2;
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * center) * inv_dz2;

    return cxx + cyy + czz;
}

// Initialize concentration field kernel
__global__ void initializeConcentrationKernel(double* c, size_t nx, size_t ny, size_t nz) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (tid >= vol) return;

    const double pseudo = ((((tid + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[tid] = -1.0 + 2.0 * pseudo;
}

// Chemical potential kernel
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                                size_t nx, size_t ny, size_t nz,
                                                double inv_dx2, double inv_dy2, double inv_dz2,
                                                double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (tid >= vol) return;

    const size_t x = tid % nx;
    const size_t y = (tid / nx) % ny;
    const size_t z = tid / (nx * ny);

    const double cv = c[tid];

    mu[tid] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * computeLaplacianDevice(c, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);
}

// Cahn-Hilliard update kernel
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                          const double* __restrict__ mu,
                                          size_t nx, size_t ny, size_t nz,
                                          double D_dt, double inv_dx2, double inv_dy2, double inv_dz2) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (tid >= vol) return;

    const size_t x = tid % nx;
    const size_t y = (tid / nx) % ny;
    const size_t z = tid / (nx * ny);

    cnew[tid] = cold[tid] + D_dt * computeLaplacianDevice(mu, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);
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

    // Precompute constants
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double D_dt = D * dt;

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    // Allocate device memory
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));

    const int numBlocks = (int)((gridSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Initialize concentration field on GPU
    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<numBlocks, BLOCK_SIZE>>>(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_cold, d_mu, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
            gamma, e_AA, e_BB, e_AB);

        // Update concentration
        cahnHilliardUpdateKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, D_dt, inv_dx2, inv_dy2, inv_dz2);

        // Swap device pointers
        double* tmp = d_cold;
        d_cold = d_cnew;
        d_cnew = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy result back to host if needed
    if (printResults || validate) {
        std::vector<double> cold(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));

        if (printResults) {
            print_results(cold, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(cold, nx, ny, nz);

            CUDA_CHECK(cudaFree(d_cold));
            CUDA_CHECK(cudaFree(d_cnew));
            CUDA_CHECK(cudaFree(d_mu));

            if (valid) {
                printf("Validation: PASSED\n");
                return 0;
            } else {
                printf("Validation: FAILED\n");
                return 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    return 0;
}
