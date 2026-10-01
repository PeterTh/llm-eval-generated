#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (false)

// One warp covers a contiguous X row; four warps cover adjacent Y rows.
constexpr unsigned BLOCK_X = 32;
constexpr unsigned BLOCK_Y = 4;

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             size_t at, size_t x, size_t y, size_t z,
                                             size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = field[at];
    const double cxx = (field[at + (x + 1 < nx)] + field[at - (x > 0)] - 2.0 * center);
    const double cyy = (field[at + (y + 1 < ny) * nx] + field[at - (y > 0) * nx] - 2.0 * center);
    const double czz = (field[at + (z + 1 < nz) * plane] + field[at - (z > 0) * plane] - 2.0 * center);
    return cxx + cyy + czz;
}

__global__ void initializeConcentration(double* c, size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y0 = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = y0; y < ny; y += size_t(gridDim.y) * blockDim.y) {
            if (x < nx) {
                const size_t at = z * (nx * ny) + y * nx + x;
                const size_t vol = nx * ny * nz;
                const double pseudo = (((at + 1) * size_t(1299709)) % vol) / static_cast<double>(vol);
                c[at] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

__global__ void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                          size_t nx, size_t ny, size_t nz,
                                          double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y0 = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = y0; y < ny; y += size_t(gridDim.y) * blockDim.y) {
            if (x < nx) {
                const size_t at = z * (nx * ny) + y * nx + x;
                const double cv = c[at];
                mu[at] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * laplacian(c, at, x, y, z, nx, ny, nz);
            }
        }
    }
}

__global__ void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                                    const double* __restrict__ mu, size_t nx, size_t ny, size_t nz,
                                    double D, double dt) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y0 = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        for (size_t y = y0; y < ny; y += size_t(gridDim.y) * blockDim.y) {
            if (x < nx) {
                const size_t at = z * (nx * ny) + y * nx + x;
                cnew[at] = cold[at] + dt * D * laplacian(mu, at, x, y, z, nx, ny, nz);
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
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Invalid grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;

    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(static_cast<unsigned>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned>(std::min((ny + BLOCK_Y - 1) / BLOCK_Y, size_t(65535))),
                    static_cast<unsigned>(std::min(nz, size_t(65535))));

    printf("Initializing concentration field...\n");
    initializeConcentration<<<grid, block>>>(cold, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential<<<grid, block>>>(cold, mu, nx, ny, nz,
                                                    gamma, e_AA, e_BB, e_AB);
        cahnHilliardUpdate<<<grid, block>>>(cnew, cold, mu, nx, ny, nz, D, dt);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / seconds / 1e6);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost));
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
