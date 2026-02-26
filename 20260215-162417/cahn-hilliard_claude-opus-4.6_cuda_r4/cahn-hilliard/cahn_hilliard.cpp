#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static constexpr int BLOCK_SIZE = 256;

__host__ __device__ inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                              size_t nx, size_t ny, size_t nz) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t vol = nx * ny * nz;
    if (i >= vol) return;
    double pseudo = (((i + 1) * 1299709ULL) % vol) / (double)vol;
    c[i] = -1.0 + 2.0 * pseudo;
}

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double gamma, double e_AA, double e_BB, double e_AB)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny * nz) return;

    size_t x = i % nx;
    size_t y = (i / nx) % ny;
    size_t z = i / (nx * ny);

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zp = (z < nz - 1) ? z + 1 : z;
    size_t zn = (z > 0) ? z - 1 : 0;

    double cv = c[i];
    double lap = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * cv) * inv_dx2
               + (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * cv) * inv_dy2
               + (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * cv) * inv_dz2;

    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gamma * lap;
}

__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz,
    double D_dt, double inv_dx2, double inv_dy2, double inv_dz2)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny * nz) return;

    size_t x = i % nx;
    size_t y = (i / nx) % ny;
    size_t z = i / (nx * ny);

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zp = (z < nz - 1) ? z + 1 : z;
    size_t zn = (z > 0) ? z - 1 : 0;

    double mv = mu[i];
    double lap = (mu[idx3(xp, y, z, nx, ny)] + mu[idx3(xn, y, z, nx, ny)] - 2.0 * mv) * inv_dx2
               + (mu[idx3(x, yp, z, nx, ny)] + mu[idx3(x, yn, z, nx, ny)] - 2.0 * mv) * inv_dy2
               + (mu[idx3(x, y, zp, nx, ny)] + mu[idx3(x, y, zn, nx, ny)] - 2.0 * mv) * inv_dz2;

    cnew[i] = cold[i] + D_dt * lap;
}

bool validateResult(const std::vector<double>& c, size_t nx, size_t ny, size_t nz) {
    (void)nx; (void)ny; (void)nz;
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minVal = c[0], maxVal = c[0];
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
    
    // Precompute constants for kernels
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double D_dt = D * dt;
    
    size_t gridSize = nx * ny * nz;
    size_t bytes = gridSize * sizeof(double);
    
    // Allocate device memory
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    
    int numBlocks = (int)((gridSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
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
    
    // Copy result back to host
    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    
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
