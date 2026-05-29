#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation (host)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// ---- CUDA Kernels ----

__device__ inline double d_laplacian(
    const double* __restrict__ field,
    const int nx, const int ny, const int nz,
    const double dx, const double dy, const double dz,
    const int x, const int y, const int z)
{
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int zp = (z < nz - 1) ? z + 1 : z;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zn = (z > 0) ? z - 1 : z;

    const int nyx = ny * nx;
    const int idx = z * nyx + y * nx + x;

    const double cxx = (field[z * nyx + y * nx + xp] + field[z * nyx + y * nx + xn] -
                        2.0 * field[idx]) / (dx * dx);
    const double cyy = (field[z * nyx + yp * nx + x] + field[z * nyx + yn * nx + x] -
                        2.0 * field[idx]) / (dy * dy);
    const double czz = (field[zp * nyx + y * nx + x] + field[zn * nyx + y * nx + x] -
                        2.0 * field[idx]) / (dz * dz);

    return cxx + cyy + czz;
}

// Initialize concentration field kernel
__global__ void initializeConcentrationKernel(
    double* __restrict__ c,
    const int nx, const int ny, const int nz, const int vol)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        const int idx = z * (ny * nx) + y * nx + x;
        const double pseudo = ((((idx + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[idx] = -1.0 + 2.0 * pseudo;
    }
}

// Compute chemical potential kernel
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c,
    double* __restrict__ mu,
    const int nx, const int ny, const int nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        const double cv = c[z * (ny * nx) + y * nx + x];

        mu[z * (ny * nx) + y * nx + x] =
            4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
            + 3.0 * cv + cv * cv * cv
            - gamma * d_laplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// Cahn-Hilliard update kernel
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew,
    const double* __restrict__ cold,
    const double* __restrict__ mu,
    const int nx, const int ny, const int nz,
    const double D, const double dt, const double dx, const double dy, const double dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        const int idx = z * (ny * nx) + y * nx + x;
        cnew[idx] = cold[idx] + dt * D *
                    d_laplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// ---- Host-side wrappers ----

void initializeConcentration(double* d_c, const int nx, const int ny, const int nz) {
    const int vol = nx * ny * nz;
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z);
    initializeConcentrationKernel<<<grid, block>>>(d_c, nx, ny, nz, vol);
    CUDA_CHECK(cudaGetLastError());
}

void computeChemicalPotentialGPU(
    const double* d_c, double* d_mu,
    const int nx, const int ny, const int nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z);
    computeChemicalPotentialKernel<<<grid, block>>>(
        d_c, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

void cahnHilliardUpdateGPU(
    double* d_cnew, const double* d_cold, const double* d_mu,
    const int nx, const int ny, const int nz,
    const double D, const double dt, const double dx, const double dy, const double dz)
{
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z);
    cahnHilliardUpdateKernel<<<grid, block>>>(
        d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
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
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(d_cold, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialGPU(d_cold, d_mu,
            static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz),
            dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdateGPU(d_cnew, d_cold, d_mu,
            static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz),
            D, dt, dx, dy, dz);
        
        // Swap device buffers
        std::swap(d_cold, d_cnew);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy final result back to host for printing/validation
    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    int exitCode = 0;
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
    return exitCode;
}
