#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// 3D index calculation
__device__ __host__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for initializing grid
__global__ void initializeGridKernel(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        grid[idx] = (idx % 19) * 1.0;
    }
}

// CUDA kernel for 7-point stencil with shared memory optimization
#define TILE_X 8
#define TILE_Y 8
#define TILE_Z 8

__global__ void stencilKernel(const Real* __restrict__ input, 
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    // Shared memory with halo (1 cell on each side)
    __shared__ Real tile[TILE_Z + 2][TILE_Y + 2][TILE_X + 2];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const size_t gx = blockIdx.x * TILE_X + tx;
    const size_t gy = blockIdx.y * TILE_Y + ty;
    const size_t gz = blockIdx.z * TILE_Z + tz;
    
    const size_t tid_flat = tz * TILE_Y * TILE_X + ty * TILE_X + tx;
    const int total_threads = TILE_X * TILE_Y * TILE_Z;
    const int total_smem = (TILE_X + 2) * (TILE_Y + 2) * (TILE_Z + 2);
    
    // Cooperatively load all shared memory cells
    const size_t planeXY = nx * ny;
    for (int i = tid_flat; i < total_smem; i += total_threads) {
        int sz = i / ((TILE_Y + 2) * (TILE_X + 2));
        int rem = i % ((TILE_Y + 2) * (TILE_X + 2));
        int sy = rem / (TILE_X + 2);
        int sx = rem % (TILE_X + 2);
        
        int gxi = (int)blockIdx.x * TILE_X + sx - 1;
        int gyi = (int)blockIdx.y * TILE_Y + sy - 1;
        int gzi = (int)blockIdx.z * TILE_Z + sz - 1;
        
        if (gxi >= 0 && gxi < (int)nx && gyi >= 0 && gyi < (int)ny && gzi >= 0 && gzi < (int)nz) {
            tile[sz][sy][sx] = input[(size_t)gzi * planeXY + (size_t)gyi * nx + (size_t)gxi];
        } else {
            tile[sz][sy][sx] = 0.0;
        }
    }
    
    __syncthreads();
    
    // Check if this thread is within bounds
    if (gx < nx && gy < ny && gz < nz) {
        // Check if this is a boundary cell
        bool is_boundary = (gx == 0 || gx == nx - 1 || 
                           gy == 0 || gy == ny - 1 || 
                           gz == 0 || gz == nz - 1);
        
        if (is_boundary) {
            // Copy boundary values unchanged
            output[idx3(gx, gy, gz, nx, ny)] = input[idx3(gx, gy, gz, nx, ny)];
        } else {
            // Compute stencil for interior points
            const Real center = tile[tz + 1][ty + 1][tx + 1];
            const Real left   = tile[tz + 1][ty + 1][tx];
            const Real right  = tile[tz + 1][ty + 1][tx + 2];
            const Real front  = tile[tz + 1][ty][tx + 1];
            const Real back   = tile[tz + 1][ty + 2][tx + 1];
            const Real bottom = tile[tz][ty + 1][tx + 1];
            const Real top    = tile[tz + 2][ty + 1][tx + 1];
            
            output[idx3(gx, gy, gz, nx, ny)] = (center + left + right + front + back + bottom + top) * (1.0 / 7.0);
        }
    }
}

// CUDA kernel for copying boundary values
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

void initializeGrid(Real* d_grid, const size_t nx, const size_t ny, const size_t nz) {
    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x, 
              (ny + block.y - 1) / block.y, 
              (nz + block.z - 1) / block.z);
    initializeGridKernel<<<grid, block>>>(d_grid, nx, ny, nz);
    cudaCheckError(cudaDeviceSynchronize());
}

void stencilIteration(const Real* d_input, 
                      Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Launch stencil kernel with 8x8x8 tiles (includes boundary copy)
    dim3 block(TILE_X, TILE_Y, TILE_Z);
    dim3 grid((nx + TILE_X - 1) / TILE_X, 
              (ny + TILE_Y - 1) / TILE_Y, 
              (nz + TILE_Z - 1) / TILE_Z);
    stencilKernel<<<grid, block>>>(d_input, d_output, nx, ny, nz);
    cudaCheckError(cudaDeviceSynchronize());
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
    size_t gridBytes = gridSize * sizeof(Real);
    
    // Allocate device grids (double buffering)
    Real* d_grid1;
    Real* d_grid2;
    cudaCheckError(cudaMalloc(&d_grid1, gridBytes));
    cudaCheckError(cudaMalloc(&d_grid2, gridBytes));
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(d_grid1, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy final result back to host for printing/validation
    std::vector<Real> h_grid(gridSize);
    Real* d_finalGrid = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    cudaCheckError(cudaMemcpy(h_grid.data(), d_finalGrid, gridBytes, cudaMemcpyDeviceToHost));
    
    // Print results for external validation
    if (printResults) {
        print_results(h_grid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(h_grid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            cudaFree(d_grid1);
            cudaFree(d_grid2);
            return 1;
        }
    }
    
    cudaCheckError(cudaFree(d_grid1));
    cudaCheckError(cudaFree(d_grid2));
    
    return 0;
}
