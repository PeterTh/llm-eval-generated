#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(1); \
        } \
    } while (0)

constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_Z = 8;

__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__global__ void initialize_grid(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t idx = idx3(x, y, z, nx, ny);
    grid[idx] = static_cast<Real>(idx % 19);
}

__global__ void stencil_interior(const Real* __restrict__ input, Real* __restrict__ output,
                                 const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;

    const bool in_domain = (x < nx && y < ny && z < nz);
    const size_t plane = nx * ny;
    const size_t idx = idx3(x, y, z, nx, ny);

    extern __shared__ Real sh[];
    const int sNx = blockDim.x + 2;
    const int sNy = blockDim.y + 2;
    const int sPlane = sNx * sNy;

    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;
    const int tz = threadIdx.z + 1;
    const int sidx = (tz * sNy + ty) * sNx + tx;

    if (in_domain) {
        sh[sidx] = input[idx];
    }

    if (in_domain && threadIdx.x == 0) {
        sh[sidx - 1] = input[idx - 1];
    }
    if (in_domain && threadIdx.x == blockDim.x - 1 && x + 1 < nx) {
        sh[sidx + 1] = input[idx + 1];
    }
    if (in_domain && threadIdx.y == 0) {
        sh[sidx - sNx] = input[idx - nx];
    }
    if (in_domain && threadIdx.y == blockDim.y - 1 && y + 1 < ny) {
        sh[sidx + sNx] = input[idx + nx];
    }
    if (in_domain && threadIdx.z == 0) {
        sh[sidx - sPlane] = input[idx - plane];
    }
    if (in_domain && threadIdx.z == blockDim.z - 1 && z + 1 < nz) {
        sh[sidx + sPlane] = input[idx + plane];
    }

    __syncthreads();

    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) {
        return;
    }

    const Real center = sh[sidx];
    const Real left = sh[sidx - 1];
    const Real right = sh[sidx + 1];
    const Real front = sh[sidx - sNx];
    const Real back = sh[sidx + sNx];
    const Real bottom = sh[sidx - sPlane];
    const Real top = sh[sidx + sPlane];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

__global__ void copy_boundaries(const Real* __restrict__ input, Real* __restrict__ output,
                                const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    if (x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1) {
        const size_t idx = idx3(x, y, z, nx, ny);
        output[idx] = input[idx];
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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

    printf("3D Stencil Benchmark (CUDA)\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, bytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, bytes));

    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid_all((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y,
                  (nz + block.z - 1) / block.z);

    printf("Initializing grid...\n");
    initialize_grid<<<grid_all, block>>>(d_grid1, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running stencil computation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    const bool hasInterior = (nx > 2 && ny > 2 && nz > 2);
    dim3 grid_interior(0, 0, 0);
    size_t shared_bytes = 0;
    if (hasInterior) {
        grid_interior = dim3((nx - 2 + block.x - 1) / block.x,
                             (ny - 2 + block.y - 1) / block.y,
                             (nz - 2 + block.z - 1) / block.z);
        shared_bytes = static_cast<size_t>(block.x + 2) * (block.y + 2) * (block.z + 2) * sizeof(Real);
    }

    for (int iter = 0; iter < iterations; ++iter) {
        const Real* input = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* output = (iter % 2 == 0) ? d_grid2 : d_grid1;

        if (hasInterior) {
            stencil_interior<<<grid_interior, block, shared_bytes>>>(input, output, nx, ny, nz);
            CUDA_CHECK(cudaGetLastError());
        }
        copy_boundaries<<<grid_all, block>>>(input, output, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const bool needHostResults = validate || printResults;
    std::vector<Real> finalGrid;
    if (needHostResults) {
        finalGrid.resize(gridSize);
        const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, bytes, cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);

        CUDA_CHECK(cudaFree(d_grid1));
        CUDA_CHECK(cudaFree(d_grid2));

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        }
        printf("Validation: FAILED\n");
        return 1;
    }

    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    return 0;
}
