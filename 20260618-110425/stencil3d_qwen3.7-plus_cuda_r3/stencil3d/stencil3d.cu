#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Tile dimensions optimized for RTX 3090 (CC 8.6)
// TILE_X=32 matches warp size for coalesced global memory access
// 512 threads/block allows 3 blocks/SM for 100% occupancy
#define TILE_X 32
#define TILE_Y 8
#define TILE_Z 2

__global__ __launch_bounds__(TILE_X * TILE_Y * TILE_Z)
void initializeGridKernel(Real* __restrict__ grid,
                          const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        const size_t idx = z * (nx * ny) + y * nx + x;
        grid[idx] = static_cast<Real>(idx % 19);
    }
}

// 7-point stencil kernel with shared memory tiling.
// Each thread block loads a 3D tile plus 1-cell halo into shared memory,
// then computes the stencil from shared memory to reduce global memory traffic.
__global__ __launch_bounds__(TILE_X * TILE_Y * TILE_Z)
void stencilKernel(const Real* __restrict__ input,
                   Real* __restrict__ output,
                   const size_t nx, const size_t ny, const size_t nz) {
    // Shared memory: tile[z][y][x] with +1 halo in each dimension
    // Layout: [TILE_Z+2][TILE_Y+2][TILE_X+2]
    // Innermost dim TILE_X+2=34 ensures no shared memory bank conflicts
    // since consecutive threadIdx.x map to consecutive banks
    __shared__ Real tile[TILE_Z + 2][TILE_Y + 2][TILE_X + 2];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t tz = threadIdx.z;

    const size_t gx = blockIdx.x * TILE_X + tx;
    const size_t gy = blockIdx.y * TILE_Y + ty;
    const size_t gz = blockIdx.z * TILE_Z + tz;

    const size_t plane = nx * ny;
    const bool inBounds = (gx < nx && gy < ny && gz < nz);

    // Load tile interior
    if (inBounds) {
        tile[tz + 1][ty + 1][tx + 1] = input[gz * plane + gy * nx + gx];
    }

    // Load X-face halos (left and right)
    if (tx == 0 && gx >= 1 && gy < ny && gz < nz) {
        tile[tz + 1][ty + 1][0] = input[gz * plane + gy * nx + (gx - 1)];
    }
    if (tx == TILE_X - 1 && gx + 1 < nx && gy < ny && gz < nz) {
        tile[tz + 1][ty + 1][TILE_X + 1] = input[gz * plane + gy * nx + (gx + 1)];
    }

    // Load Y-face halos (front and back)
    if (ty == 0 && gx < nx && gy >= 1 && gz < nz) {
        tile[tz + 1][0][tx + 1] = input[gz * plane + (gy - 1) * nx + gx];
    }
    if (ty == TILE_Y - 1 && gy + 1 < ny && gx < nx && gz < nz) {
        tile[tz + 1][TILE_Y + 1][tx + 1] = input[gz * plane + (gy + 1) * nx + gx];
    }

    // Load Z-face halos (bottom and top)
    if (tz == 0 && gx < nx && gy < ny && gz >= 1) {
        tile[0][ty + 1][tx + 1] = input[(gz - 1) * plane + gy * nx + gx];
    }
    if (tz == TILE_Z - 1 && gz + 1 < nz && gx < nx && gy < ny) {
        tile[TILE_Z + 1][ty + 1][tx + 1] = input[(gz + 1) * plane + gy * nx + gx];
    }

    __syncthreads();

    if (!inBounds) return;

    // Boundary points: copy from input (already in shared memory)
    if (gx == 0 || gx == nx - 1 || gy == 0 || gy == ny - 1 || gz == 0 || gz == nz - 1) {
        output[gz * plane + gy * nx + gx] = tile[tz + 1][ty + 1][tx + 1];
        return;
    }

    // Interior points: 7-point stencil averaging
    const Real center = tile[tz + 1][ty + 1][tx + 1];
    const Real left   = tile[tz + 1][ty + 1][tx];
    const Real right  = tile[tz + 1][ty + 1][tx + 2];
    const Real front  = tile[tz + 1][ty][tx + 1];
    const Real back   = tile[tz + 1][ty + 2][tx + 1];
    const Real bottom = tile[tz][ty + 1][tx + 1];
    const Real top    = tile[tz + 2][ty + 1][tx + 1];

    output[gz * plane + gy * nx + gx] = (center + left + right + front + back + bottom + top) / 7.0;
}

bool validateResult(const std::vector<Real>& grid,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

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
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
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

    printf("3D Stencil Benchmark (CUDA)\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Print GPU info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const size_t gridSize = nx * ny * nz;
    const size_t gridBytes = gridSize * sizeof(Real);

    // Allocate device memory (double buffering)
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridBytes));

    // Initialize grid on device
    printf("Initializing grid...\n");
    {
        dim3 blk(TILE_X, TILE_Y, TILE_Z);
        dim3 grd(static_cast<unsigned int>((nx + TILE_X - 1) / TILE_X),
                 static_cast<unsigned int>((ny + TILE_Y - 1) / TILE_Y),
                 static_cast<unsigned int>((nz + TILE_Z - 1) / TILE_Z));
        initializeGridKernel<<<grd, blk>>>(d_grid1, nx, ny, nz);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run stencil iterations
    printf("Running stencil computation...\n");

    dim3 blk(TILE_X, TILE_Y, TILE_Z);
    dim3 grd(static_cast<unsigned int>((nx + TILE_X - 1) / TILE_X),
             static_cast<unsigned int>((ny + TILE_Y - 1) / TILE_Y),
             static_cast<unsigned int>((nz + TILE_Z - 1) / TILE_Z));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<grd, blk>>>(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilKernel<<<grd, blk>>>(d_grid2, d_grid1, nx, ny, nz);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy result back to host
    std::vector<Real> hostGrid(gridSize);
    const Real* d_finalGrid = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), d_finalGrid, gridBytes, cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        print_results(hostGrid, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(hostGrid, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    return 0;
}
