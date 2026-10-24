#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err_));                                           \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

struct Parameters {
    int nx;
    int ny;
    int nz;
    double invDx2;
    double invDy2;
    double invDz2;
    double gamma;
    double eAA;
    double eBB;
    double eAB;
    double dtD;
};

// Compute Laplacian with clamped boundary conditions
__device__ __forceinline__ double computeLaplacian(const double* __restrict__ a,
                                                   const size_t idx,
                                                   const int x, const int y, const int z,
                                                   const double center,
                                                   const Parameters& p) {
    const int xp = (x < p.nx - 1) ? x + 1 : x;
    const int yp = (y < p.ny - 1) ? y + 1 : y;
    const int zp = (z < p.nz - 1) ? z + 1 : z;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zn = (z > 0) ? z - 1 : 0;

    const size_t sy = static_cast<size_t>(p.nx);
    const size_t sz = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);

    const double cxx = (__ldg(&a[idx + (xp - x)]) +
                        __ldg(&a[idx - (x - xn)]) -
                        2.0 * center) * p.invDx2;
    const double cyy = (__ldg(&a[idx + static_cast<size_t>(yp - y) * sy]) +
                        __ldg(&a[idx - static_cast<size_t>(y - yn) * sy]) -
                        2.0 * center) * p.invDy2;
    const double czz = (__ldg(&a[idx + static_cast<size_t>(zp - z) * sz]) +
                        __ldg(&a[idx - static_cast<size_t>(z - zn) * sz]) -
                        2.0 * center) * p.invDz2;

    return cxx + cyy + czz;
}

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const Parameters p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= p.nx || y >= p.ny || z >= p.nz) {
        return;
    }

    const size_t idx = static_cast<size_t>(z) *
                           (static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny)) +
                       static_cast<size_t>(y) * static_cast<size_t>(p.nx) +
                       static_cast<size_t>(x);
    const double cv = __ldg(&c[idx]);

    mu[idx] = 4.5 * ((cv + 1.0) * p.eAA + (cv - 1.0) * p.eBB - 2.0 * cv * p.eAB)
            + 3.0 * cv + cv * cv * cv
            - p.gamma * computeLaplacian(c, idx, x, y, z, cv, p);
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const Parameters p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= p.nx || y >= p.ny || z >= p.nz) {
        return;
    }

    const size_t idx = static_cast<size_t>(z) *
                           (static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny)) +
                       static_cast<size_t>(y) * static_cast<size_t>(p.nx) +
                       static_cast<size_t>(x);
    const double muc = __ldg(&mu[idx]);

    cnew[idx] = __ldg(&cold[idx]) +
                p.dtD * computeLaplacian(mu, idx, x, y, z, muc, p);
}

// Initialize concentration field
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= vol) {
        return;
    }

    // Generate pseudo-random value in [-1, 1]
    const double pseudo = static_cast<double>((((idx + 1) * 1299709) % vol)) /
                          static_cast<double>(vol);
    c[idx] = -1.0 + 2.0 * pseudo;
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

    const size_t gridSize = nx * ny * nz;

    Parameters p;
    p.nx = static_cast<int>(nx);
    p.ny = static_cast<int>(ny);
    p.nz = static_cast<int>(nz);
    p.invDx2 = 1.0 / (dx * dx);
    p.invDy2 = 1.0 / (dy * dy);
    p.invDz2 = 1.0 / (dz * dz);
    p.gamma = gamma;
    p.eAA = e_AA;
    p.eBB = e_BB;
    p.eAB = e_AB;
    p.dtD = dt * D;

    // Allocate arrays on the GPU
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    {
        const int threads = 256;
        const int blocks = static_cast<int>((gridSize + threads - 1) / threads);
        initializeConcentrationKernel<<<blocks, threads>>>(d_cold, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Select a block shape that keeps memory accesses coalesced along x while
    // still exploiting reuse in y and z when those dimensions are large enough.
    dim3 block;
    if (nz >= 8) {
        block = dim3(16, 8, 8);
    } else if (nz >= 4) {
        block = dim3(16, 8, 4);
    } else if (nz >= 2) {
        block = dim3(16, 8, 2);
    } else {
        block = dim3(32, 8, 1);
    }
    const dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ny) + block.y - 1) / block.y,
                    (static_cast<unsigned>(nz) + block.z - 1) / block.z);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, p);

        // Update concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, p);

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_cold));

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
