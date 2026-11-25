#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < volume;
         i += size_t(gridDim.x) * blockDim.x) {
        const double pseudo = ((i + 1) * 1299709 % volume) / static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

__device__ double chemicalPotential(double c, double laplacian) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = e_AA, e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    const double pairEnergy = fma(c + 1.0, e_AA, (c - 1.0) * e_BB);
    const double interaction = fma(-2.0 * c, e_AB, pairEnergy);
    const double local = fma(c * c, c, fma(4.5, interaction, 3.0 * c));
    return fma(-gamma, laplacian, local);
}

// Each warp covers consecutive x coordinates. Stream a short column in z
// so successive stencils reuse their center and adjacent planes in registers.
template<bool potential, unsigned planes>
__global__ void stencil(const double* __restrict__ input, double* __restrict__ output,
                        unsigned nx, unsigned ny, unsigned nz) {
    constexpr double dt = 0.01, D = 1.0;
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t slice = size_t(nx) * ny;
    for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny;
         y += gridDim.y * blockDim.y) {
        for (size_t z = blockIdx.z * planes; z < nz; z += gridDim.z * planes) {
            size_t i = z * slice + y * nx + x;
            double center = input[i];
            double below = z ? input[i - slice] : center;
            #pragma unroll
            for (unsigned k = 0; k < planes && z + k < nz; ++k, i += slice) {
                const double above = z + k + 1 < nz ? input[i + slice] : center;
                const double twice = 2.0 * center;
                // The benchmark has unit spacing along all three axes.
                const double xx = (input[i + (x + 1 < nx)] + input[i - (x > 0)]) - twice;
                const double yy = (input[i + (y + 1 < ny ? nx : 0)] +
                                   input[i - (y > 0 ? nx : 0)]) - twice;
                const double zz = (above + below) - twice;
                const double laplacian = (xx + yy) + zz;
                if constexpr (potential)
                    output[i] = chemicalPotential(center, laplacian);
                else
                    output[i] = fma(dt * D, laplacian, output[i]);
                below = center;
                center = above;
            }
        }
    }
}

template<unsigned planes>
void simulate(double* c, double* mu, unsigned nx, unsigned ny, unsigned nz, int iterations) {
    if (nx == 0) return;
    const dim3 block(32, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
                    std::min((ny + block.y - 1) / block.y, 65535u),
                    std::min((nz + planes - 1) / planes, 65535u));
    for (int t = 0; t < iterations; ++t) {
        stencil<true, planes><<<grid, block>>>(c, mu, nx, ny, nz);
        // The stream orders both stages. The update reads only its own old c.
        stencil<false, planes><<<grid, block>>>(mu, c, nx, ny, nz);
    }
    checkCuda(cudaGetLastError(), "launching stencil kernels");
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    if (c.empty()) return true;
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
    
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
        (nx && (nx > maxCells / ny || nx * ny > maxCells / nz))) {
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    const size_t allocationBytes = std::max(bytes, sizeof(double));
    double* c;
    double* mu;
    checkCuda(cudaMalloc(&c, allocationBytes), "allocating concentration");
    checkCuda(cudaMalloc(&mu, allocationBytes), "allocating chemical potential");
    printf("Initializing concentration field...\n");
    const size_t initBlocks = std::clamp((gridSize + 255) / 256, size_t(1), size_t(65535));
    initializeConcentration<<<initBlocks, 256>>>(c, gridSize);
    checkCuda(cudaGetLastError(), "launching initialization");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration");

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (gridSize >= 1024 * 1024)
        simulate<4>(c, mu, nx, ny, nz, iterations);
    else
        simulate<1>(c, mu, nx, ny, nz, iterations);
    checkCuda(cudaDeviceSynchronize(), "completing simulation");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    std::vector<double> cold;
    if (gridSize && (printResults || validate)) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), c, bytes, cudaMemcpyDeviceToHost), "reading concentration");
    }
    checkCuda(cudaFree(mu), "freeing chemical potential");
    checkCuda(cudaFree(c), "freeing concentration");

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
