#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
__device__ __host__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel: initialize grid
__global__ void initGridKernel(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        grid[idx] = (idx % 19) * 1.0;
    }
}

// CUDA kernel: 7-point stencil with shared memory tiling and halo zones
// Block size: (TX+2) x (TY+2) x (TZ+2) threads
// Shared memory: (TZ+3) x (TY+3) x (TX+3) to hold tile + 1-cell halo
template <int TX, int TY, int TZ>
__global__ void stencilKernel(const Real* __restrict__ input, 
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    __shared__ Real s_tile[TZ + 3][TY + 3][TX + 3];
    
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const int lz = threadIdx.z + 1;
    
    const int gx = (int)blockIdx.x * TX + threadIdx.x;
    const int gy = (int)blockIdx.y * TY + threadIdx.y;
    const int gz = (int)blockIdx.z * TZ + threadIdx.z;
    
    // Each thread loads its center cell into shared memory
    // Only interior threads (0..TX, 0..TY, 0..TZ) load center cells
    if (threadIdx.x <= TX && threadIdx.y <= TY && threadIdx.z <= TZ) {
        if (gx >= 0 && gx < (int)nx && gy >= 0 && gy < (int)ny && gz >= 0 && gz < (int)nz) {
            s_tile[lz][ly][lx] = input[idx3((size_t)gx, (size_t)gy, (size_t)gz, nx, ny)];
        }
    }
    
    // Load halo cells - each thread loads at most one halo per dimension
    // Left halo: thread 0 loads its left neighbor (gx-1)
    if (threadIdx.x == 0 && threadIdx.y <= TY && threadIdx.z <= TZ) {
        const int hx = gx - 1;
        if (hx >= 0 && hx < (int)nx && gy >= 0 && gy < (int)ny && gz >= 0 && gz < (int)nz)
            s_tile[lz][ly][0] = input[idx3((size_t)hx, (size_t)gy, (size_t)gz, nx, ny)];
    }
    // Right halo: thread TX+1 loads its own cell (which IS the right neighbor)
    if (threadIdx.x == TX + 1 && threadIdx.y <= TY && threadIdx.z <= TZ) {
        if (gx >= 0 && gx < (int)nx && gy >= 0 && gy < (int)ny && gz >= 0 && gz < (int)nz)
            s_tile[lz][ly][TX + 2] = input[idx3((size_t)gx, (size_t)gy, (size_t)gz, nx, ny)];
    }
    // Front halo: thread 0 in y loads its front neighbor (gy-1)
    if (threadIdx.y == 0 && threadIdx.x <= TX && threadIdx.z <= TZ) {
        const int hy = gy - 1;
        if (gx >= 0 && gx < (int)nx && hy >= 0 && hy < (int)ny && gz >= 0 && gz < (int)nz)
            s_tile[lz][0][lx] = input[idx3((size_t)gx, (size_t)hy, (size_t)gz, nx, ny)];
    }
    // Back halo: thread TY+1 in y loads its own cell (which IS the back neighbor)
    if (threadIdx.y == TY + 1 && threadIdx.x <= TX && threadIdx.z <= TZ) {
        if (gx >= 0 && gx < (int)nx && gy >= 0 && gy < (int)ny && gz >= 0 && gz < (int)nz)
            s_tile[lz][TY + 2][lx] = input[idx3((size_t)gx, (size_t)gy, (size_t)gz, nx, ny)];
    }
    // Bottom halo: thread 0 in z loads its bottom neighbor (gz-1)
    if (threadIdx.z == 0 && threadIdx.x <= TX && threadIdx.y <= TY) {
        const int hz = gz - 1;
        if (gx >= 0 && gx < (int)nx && gy >= 0 && gy < (int)ny && hz >= 0 && hz < (int)nz)
            s_tile[0][ly][lx] = input[idx3((size_t)gx, (size_t)gy, (size_t)hz, nx, ny)];
    }
    // Top halo: thread TZ+1 in z loads its own cell (which IS the top neighbor)
    if (threadIdx.z == TZ + 1 && threadIdx.x <= TX && threadIdx.y <= TY) {
        if (gx >= 0 && gx < (int)nx && gy >= 0 && gy < (int)ny && gz >= 0 && gz < (int)nz)
            s_tile[TZ + 2][ly][lx] = input[idx3((size_t)gx, (size_t)gy, (size_t)gz, nx, ny)];
    }
    
    __syncthreads();
    
    // Compute stencil for interior threads only (threadIdx 1..TX)
    if (threadIdx.x >= 1 && threadIdx.x <= TX &&
        threadIdx.y >= 1 && threadIdx.y <= TY &&
        threadIdx.z >= 1 && threadIdx.z <= TZ) {
        if (gx >= 1 && gx < (int)nx - 1 && gy >= 1 && gy < (int)ny - 1 && gz >= 1 && gz < (int)nz - 1) {
            const Real val = (s_tile[lz][ly][lx] +
                              s_tile[lz][ly][lx - 1] +
                              s_tile[lz][ly][lx + 1] +
                              s_tile[lz][ly - 1][lx] +
                              s_tile[lz][ly + 1][lx] +
                              s_tile[lz - 1][ly][lx] +
                              s_tile[lz + 1][ly][lx]) / 7.0;
            output[idx3((size_t)gx, (size_t)gy, (size_t)gz, nx, ny)] = val;
        }
    }
}

// CUDA kernel: copy boundary values
__global__ void copyBoundaryKernel(const Real* __restrict__ input, 
                                    Real* __restrict__ output,
                                    const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
            const size_t idx = idx3(x, y, z, nx, ny);
            output[idx] = input[idx];
        }
    }
}

// Wrapper for stencil iteration on GPU
void stencilIterationGPU(const Real* d_input, Real* d_output,
                         const size_t nx, const size_t ny, const size_t nz) {
    // Tile size 8x8x8, block 10x10x10=1000 threads
    constexpr int TX = 8;
    constexpr int TY = 8;
    constexpr int TZ = 8;
    
    dim3 blockDim(TX + 2, TY + 2, TZ + 2);
    dim3 gridDim(
        (nx + TX - 1) / TX,
        (ny + TY - 1) / TY,
        (nz + TZ - 1) / TZ
    );
    
    stencilKernel<TX, TY, TZ><<<gridDim, blockDim>>>(d_input, d_output, nx, ny, nz);
    
    // Copy boundary values
    dim3 bdim(16, 16, 4);
    dim3 gdim(
        (nx + bdim.x - 1) / bdim.x,
        (ny + bdim.y - 1) / bdim.y,
        (nz + bdim.z - 1) / bdim.z
    );
    copyBoundaryKernel<<<gdim, bdim>>>(d_input, d_output, nx, ny, nz);
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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate host grids
    std::vector<Real> h_grid1(gridSize);
    std::vector<Real> h_grid2(gridSize);
    
    // Allocate device grids
    Real *d_grid1, *d_grid2;
    cudaMalloc(&d_grid1, gridSize * sizeof(Real));
    cudaMalloc(&d_grid2, gridSize * sizeof(Real));
    
    // Initialize on GPU
    printf("Initializing grid...\n");
    dim3 initBlock(16, 16, 4);
    dim3 initGrid(
        (nx + initBlock.x - 1) / initBlock.x,
        (ny + initBlock.y - 1) / initBlock.y,
        (nz + initBlock.z - 1) / initBlock.z
    );
    initGridKernel<<<initGrid, initBlock>>>(d_grid1, nx, ny, nz);
    cudaDeviceSynchronize();
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIterationGPU(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilIterationGPU(d_grid2, d_grid1, nx, ny, nz);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Computation time: %ld ms\n", (long)ms);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (ms / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy result back to host for validation/printing
    const Real* d_finalGrid = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    Real* h_finalGrid = (iterations % 2 == 0) ? h_grid1.data() : h_grid2.data();
    cudaMemcpy(h_finalGrid, d_finalGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? h_grid1 : h_grid2;
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    // Cleanup
    cudaFree(d_grid1);
    cudaFree(d_grid2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
