#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Every CUDA thread updates one cell. The two kernels are separate because the
// second stencil needs chemical potentials from neighboring cells.
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* c, size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const double pseudo = (((i + 1) * size_t(1299709)) % count) / static_cast<double>(count);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

template <typename Index>
__device__ __forceinline__ double laplacian(const double* __restrict__ a, Index i,
                                             Index x, Index y, Index nx, Index ny,
                                             Index plane, Index count) {
    const double center = a[i];
    const double cxx = a[x + 1 < nx ? i + 1 : i] + a[x > 0 ? i - 1 : i] - 2.0 * center;
    const double cyy = a[y + 1 < ny ? i + nx : i] + a[y > 0 ? i - nx : i] - 2.0 * center;
    const double czz = a[i < count - plane ? i + plane : i] + a[i >= plane ? i - plane : i] - 2.0 * center;
    return cxx + cyy + czz;
}

template <typename Index>
__global__ void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                          Index nx, Index ny, Index plane, Index count) {
    Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(blockDim.x) * gridDim.x;
    while (i < count) {
        const Index x = i % nx;
        const Index y = (i / nx) % ny;
        const double cv = c[i];
        constexpr double e_AA = -(2.0 / 9.0);
        constexpr double e_BB = -(2.0 / 9.0);
        constexpr double e_AB = (2.0 / 9.0);
        mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c, i, x, y, nx, ny, plane, count);
        if (count - i <= stride) break;
        i += stride;
    }
}

template <typename Index>
__global__ void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                                    const double* __restrict__ mu, Index nx, Index ny,
                                    Index plane, Index count) {
    Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(blockDim.x) * gridDim.x;
    while (i < count) {
        const Index x = i % nx;
        const Index y = (i / nx) % ny;
        cnew[i] = cold[i] + 0.01 * laplacian(mu, i, x, y, nx, ny, plane, count);
        if (count - i <= stride) break;
        i += stride;
    }
}

// Map a warp across x and blocks across rows and planes, avoiding per-cell
// integer division. Grid-stride loops cover dimensions beyond CUDA grid limits.
template <typename Index, bool Chemical>
__global__ void stencil3d(const double* __restrict__ input, const double* __restrict__ cold,
                          double* __restrict__ output, Index nx, Index ny, Index nz,
                          Index plane, Index count) {
    const Index x0 = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index y0 = static_cast<Index>(blockIdx.y) * blockDim.y + threadIdx.y;
    const Index xstride = static_cast<Index>(gridDim.x) * blockDim.x;
    const Index ystride = static_cast<Index>(gridDim.y) * blockDim.y;
    for (Index z = blockIdx.z; z < nz;) {
        for (Index y = y0; y < ny;) {
            for (Index x = x0; x < nx;) {
                const Index i = z * plane + y * nx + x;
                const double lap = laplacian(input, i, x, y, nx, ny, plane, count);
                if constexpr (Chemical) {
                    const double cv = input[i];
                    constexpr double e_AA = -(2.0 / 9.0);
                    constexpr double e_BB = -(2.0 / 9.0);
                    constexpr double e_AB = (2.0 / 9.0);
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                } else {
                    output[i] = cold[i] + 0.01 * lap;
                }
                if (nx - x <= xstride) break;
                x += xstride;
            }
            if (ny - y <= ystride) break;
            y += ystride;
        }
        if (nz - z <= gridDim.z) break;
        z += gridDim.z;
    }
}

template <typename Index>
static void simulate(double*& cold, double*& cnew, double* mu, size_t nx, size_t ny,
                     size_t nz, size_t count, int iterations, unsigned int blocks) {
    const bool flat = nx < 16 || ny < 2;
    const dim3 threads(32, 4);
    const dim3 grid(static_cast<unsigned int>(std::min<size_t>((nx + 31) / 32, 65535)),
                    static_cast<unsigned int>(std::min<size_t>((ny + 3) / 4, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(nz, 65535)));
    for (int t = 0; t < iterations; ++t) {
        if (flat) {
            computeChemicalPotential<Index><<<blocks, 256>>>(cold, mu, static_cast<Index>(nx),
                static_cast<Index>(ny), static_cast<Index>(nx * ny), static_cast<Index>(count));
            cahnHilliardUpdate<Index><<<blocks, 256>>>(cnew, cold, mu, static_cast<Index>(nx),
                static_cast<Index>(ny), static_cast<Index>(nx * ny), static_cast<Index>(count));
        } else {
            stencil3d<Index, true><<<grid, threads>>>(cold, cold, mu, static_cast<Index>(nx),
                static_cast<Index>(ny), static_cast<Index>(nz), static_cast<Index>(nx * ny), static_cast<Index>(count));
            stencil3d<Index, false><<<grid, threads>>>(mu, cold, cnew, static_cast<Index>(nx),
                static_cast<Index>(ny), static_cast<Index>(nz), static_cast<Index>(nx * ny), static_cast<Index>(count));
        }
        std::swap(cold, cnew);
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
    
    if (nx == 0 || ny > std::numeric_limits<size_t>::max() / nx ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny) ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or excessively large grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>((gridSize + 255) / 256, 65535));
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    checkCuda(cudaMalloc(&cold, bytes), "allocating concentration");
    checkCuda(cudaMalloc(&cnew, bytes), "allocating updated concentration");
    checkCuda(cudaMalloc(&mu, bytes), "allocating chemical potential");

    printf("Initializing concentration field...\n");
    initializeConcentration<<<blocks, 256>>>(cold, gridSize);
    checkCuda(cudaGetLastError(), "launching initialization");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration");

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (gridSize <= std::numeric_limits<uint32_t>::max()) {
        simulate<uint32_t>(cold, cnew, mu, nx, ny, nz, gridSize, iterations, blocks);
    } else {
        simulate<size_t>(cold, cnew, mu, nx, ny, nz, gridSize, iterations, blocks);
    }
    checkCuda(cudaGetLastError(), "launching simulation");
    checkCuda(cudaDeviceSynchronize(), "running simulation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = (double)gridSize * iterations;
    double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "copying results");
    }
    checkCuda(cudaFree(cold), "freeing concentration");
    checkCuda(cudaFree(cnew), "freeing updated concentration");
    checkCuda(cudaFree(mu), "freeing chemical potential");

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
