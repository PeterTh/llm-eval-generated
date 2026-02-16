#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
// CUDA kernel for stencil
__global__ void stencilKernel(const Real* input, Real* output, size_t nx, size_t ny, size_t nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x < nx-1 && y < ny-1 && z < nz-1) {
        size_t idx = idx3(x, y, z, nx, ny);
        Real center = input[idx];
        Real left = input[idx3(x-1, y, z, nx, ny)];
        Real right = input[idx3(x+1, y, z, nx, ny)];
        Real front = input[idx3(x, y-1, z, nx, ny)];
        Real back = input[idx3(x, y+1, z, nx, ny)];
        Real bottom = input[idx3(x, y, z-1, nx, ny)];
        Real top = input[idx3(x, y, z+1, nx, ny)];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Offload to GPU
    Real *d_input, *d_output;
    size_t bytes = nx * ny * nz * sizeof(Real);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(8,8,8);
    dim3 grid((nx-2+block.x-1)/block.x, (ny-2+block.y-1)/block.y, (nz-2+block.z-1)/block.z);
    stencilKernel<<<grid, block>>>(d_input, d_output, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    // Copy boundary values (OpenMP parallelized)
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Domain decomposition along z
    size_t local_nz = nz / size;
    size_t z_start = rank * local_nz;
    size_t z_end = (rank == size-1) ? nz : z_start + local_nz;
    local_nz = z_end - z_start;

    printf("Rank %d: 3D Stencil Benchmark\n", rank);
    printf("Rank %d: Local grid size: %zu x %zu x %zu\n", rank, nx, ny, local_nz);
    printf("Rank %d: Iterations: %d\n", rank, iterations);
    printf("Rank %d: Validation: %s\n", rank, validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * local_nz;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Rank %d: Initializing grid...\n", rank);
    initializeGrid(grid1, nx, ny, local_nz);
    
    // Run stencil iterations
    printf("Rank %d: Running stencil computation...\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halos with neighbors (simplified, only z direction)
        if (size > 1) {
            MPI_Status status;
            if (rank > 0) {
                MPI_Sendrecv(grid1.data() + nx*ny, nx*ny, MPI_DOUBLE, rank-1, 0,
                             grid1.data(), nx*ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
            }
            if (rank < size-1) {
                MPI_Sendrecv(grid1.data() + (local_nz-2)*nx*ny, nx*ny, MPI_DOUBLE, rank+1, 0,
                             grid1.data() + (local_nz-1)*nx*ny, nx*ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &status);
            }
        }
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, local_nz);
        } else {
            stencilIteration(grid2, grid1, nx, ny, local_nz);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
    }
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (local_nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Rank %d: Performance: %.3f MCellUpdates/s\n", rank, mcups);
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, local_nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
