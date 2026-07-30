#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline __host__ __device__ size_t idx3(const size_t x, const size_t y, const size_t z,
                                       const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device function: compute Laplacian with clamped boundary conditions
__device__ double computeLaplacian(const double* c, const size_t nx, const size_t ny,
                                   const size_t nz, const double dx, const double dy,
                                   const double dz, const size_t x, const size_t y,
                                   const size_t z) {
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

// CUDA kernel: compute chemical potential at each grid point
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    const double cv = c[idx];

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

// CUDA kernel: Cahn-Hilliard update step at each grid point
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t nz,
    const double D, const double dt, const double dx, const double dy, const double dz)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + dt * D *
               computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
}

// CUDA kernel: initialize concentration field
__global__ void initializeConcentrationKernel(
    double* __restrict__ c,
    const size_t nx, const size_t ny, const size_t nz)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    const size_t vol = nx * ny * nz;
    const size_t linear_id = z * (nx * ny) + y * nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[idx] = -1.0 + 2.0 * pseudo;
}

// Host-side validation
bool validateResult(const std::vector<double>& c,
                    [[maybe_unused]] const size_t nx,
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

    // Allocate device memory
    double *d_cold, *d_cnew, *d_mu;
    if (cudaMalloc(&d_cold, gridSize * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_cnew, gridSize * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_mu,   gridSize * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return 1;
    }

    // Launch configuration: 8x8x8 threads per block (512 threads)
    const dim3 blockDim(8, 8, 8);
    const dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                       (ny + blockDim.y - 1) / blockDim.y,
                       (nz + blockDim.z - 1) / blockDim.z);

    // Initialize concentration field on device
    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<gridDim, blockDim>>>(d_cold, nx, ny, nz);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed (init)\n");
        return 1;
    }

    // Run simulation with CUDA event timing
    printf("Running Cahn-Hilliard simulation...\n");

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<gridDim, blockDim>>>(
            d_cold, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        cahnHilliardUpdateKernel<<<gridDim, blockDim>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);

        // Swap device buffers (pointer swap, equivalent to std::swap)
        double* tmp = d_cold;
        d_cold = d_cnew;
        d_cnew = tmp;
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    // Copy final result back to host
    std::vector<double> cold(gridSize);
    cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up device resources
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    printf("Computation time: %ld ms\n", static_cast<long>(milliseconds));

    // Calculate performance
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = cellUpdates / (milliseconds / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(cold, nx, ny, nz);

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
