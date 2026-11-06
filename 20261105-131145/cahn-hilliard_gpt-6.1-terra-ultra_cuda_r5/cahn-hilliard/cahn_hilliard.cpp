#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA calls are checked so a failed launch or allocation cannot silently produce results.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * size_t(1299709)) % volume) / static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Each block processes one or more x rows, so neighboring threads access
// adjacent values. A flat layout keeps narrow grids well populated.
template <typename Index, bool Chemical, bool Flat>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ old,
                        double* __restrict__ output,
                        Index nx, Index ny, Index nz, Index volume) {
    const size_t rows = static_cast<size_t>(ny) * nz;
    const size_t row_start = Flat ? 0 : blockIdx.y;
    const size_t row_step = Flat ? rows : gridDim.y;
    for (size_t row = row_start; row < rows; row += row_step) {
        const size_t begin = Flat ? static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x
                                  : row * nx + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t end = Flat ? volume : (row + 1) * nx;
        const size_t stride = Flat ? static_cast<size_t>(gridDim.x) * blockDim.x : end;
        for (size_t pos = begin; pos < end; pos += stride) {
            const Index i = static_cast<Index>(pos);
            Index x, y, z;
            if constexpr (Flat) {
                x = i % nx;
                const Index yz = i / nx;
                y = yz % ny;
                z = yz / ny;
            } else {
                x = static_cast<Index>(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
                y = static_cast<Index>(row % ny);
                z = static_cast<Index>(row / ny);
            }
            const Index plane = nx * ny;
            const double center = input[i];
            const double cxx = input[x + 1 < nx ? i + 1 : i] + input[x > 0 ? i - 1 : i] - 2.0 * center;
            const double cyy = input[y + 1 < ny ? i + nx : i] + input[y > 0 ? i - nx : i] - 2.0 * center;
            const double czz = input[z + 1 < nz ? i + plane : i] + input[z > 0 ? i - plane : i] - 2.0 * center;
            const double lap = cxx + cyy + czz;
            if constexpr (Chemical) {
                constexpr double e_AA = -(2.0 / 9.0);
                constexpr double e_BB = -(2.0 / 9.0);
                constexpr double e_AB = (2.0 / 9.0);
                output[i] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB - 2.0 * center * e_AB)
                            + 3.0 * center + center * center * center - 0.5 * lap;
            } else {
                output[i] = old[i] + 0.01 * lap;
            }
        }
        if constexpr (Flat) break;
    }
}

template <typename Index>
void runSimulation(double*& cold, double*& cnew, double* mu,
                   size_t nx, size_t ny, size_t nz, size_t volume, int iterations) {
    constexpr unsigned threads = 128;
    const bool flat = nx < 64;
    const size_t blocks = flat ? (volume + threads - 1) / threads : (nx + threads - 1) / threads;
    const dim3 grid(static_cast<unsigned>(std::min<size_t>(blocks, 2147483647u)),
                    flat ? 1u : static_cast<unsigned>(std::min<size_t>(ny * nz, 65535u)));
    for (int t = 0; t < iterations; ++t) {
        if (flat) {
            stencil<Index, true, true><<<grid, threads>>>(cold, nullptr, mu, nx, ny, nz, volume);
            stencil<Index, false, true><<<grid, threads>>>(mu, cold, cnew, nx, ny, nz, volume);
        } else {
            stencil<Index, true, false><<<grid, threads>>>(cold, nullptr, mu, nx, ny, nz, volume);
            stencil<Index, false, false><<<grid, threads>>>(mu, cold, cnew, nx, ny, nz, volume);
        }
        std::swap(cold, cnew);
    }
    checkCuda(cudaGetLastError(), "stencil launch");
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
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or too large grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    checkCuda(cudaMalloc(&cold, bytes), "allocating concentration");
    checkCuda(cudaMalloc(&cnew, bytes), "allocating next concentration");
    checkCuda(cudaMalloc(&mu, bytes), "allocating chemical potential");

    printf("Initializing concentration field...\n");
    constexpr unsigned threads = 256;
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize + threads - 1) / threads, 65535));
    initializeConcentration<<<initBlocks, threads>>>(cold, gridSize);
    checkCuda(cudaGetLastError(), "initialization launch");
    checkCuda(cudaDeviceSynchronize(), "initialization");

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (gridSize <= std::numeric_limits<unsigned>::max()) {
        runSimulation<unsigned>(cold, cnew, mu, nx, ny, nz, gridSize, iterations);
    } else {
        runSimulation<size_t>(cold, cnew, mu, nx, ny, nz, gridSize, iterations);
    }
    checkCuda(cudaDeviceSynchronize(), "simulation");
    auto end = std::chrono::high_resolution_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Computation time: %.3f ms\n", milliseconds);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "copying result");
    }
    checkCuda(cudaFree(cold), "freeing concentration");
    checkCuda(cudaFree(cnew), "freeing next concentration");
    checkCuda(cudaFree(mu), "freeing chemical potential");

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (milliseconds / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
