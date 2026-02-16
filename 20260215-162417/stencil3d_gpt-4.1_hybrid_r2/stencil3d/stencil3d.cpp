#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
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
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Process interior points (not on boundaries)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values
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

__global__ void stencil3d_kernel(const Real* input, Real* output, size_t nx, size_t ny, size_t nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x < nx-1 && y < ny-1 && z < nz-1) {
        size_t idx = z * (nx * ny) + y * nx + x;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size;)

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
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    
    // Domain decomposition (z-slab)
    size_t local_nz = nz / size;
    size_t z_start = rank * local_nz;
    size_t z_end = z_start + local_nz - 1;
    size_t local_gridSize = nx * ny * (local_nz + 2); // +2 for halos
    std::vector<Real> local_grid1(local_gridSize);
    std::vector<Real> local_grid2(local_gridSize);
    // Initialize local grid (excluding halos)
    initializeGrid(local_grid1, nx, ny, local_nz + 2);

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    Real *d_grid1, *d_grid2;
    cudaMalloc(&d_grid1, local_gridSize * sizeof(Real));
    cudaMalloc(&d_grid2, local_gridSize * sizeof(Real));
    cudaMemcpy(d_grid1, local_grid1.data(), local_gridSize * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid2, local_grid2.data(), local_gridSize * sizeof(Real), cudaMemcpyHostToDevice);

    dim3 block(8,8,8);
    dim3 grid((nx-2+block.x-1)/block.x, (ny-2+block.y-1)/block.y, (local_nz+block.z-1)/block.z);

    for (int iter = 0; iter < iterations; ++iter) {
        // Halo exchange (copy device to host for halos)
        cudaMemcpy(local_grid1.data(), d_grid1, local_gridSize * sizeof(Real), cudaMemcpyDeviceToHost);
        MPI_Request reqs[4];
        if (rank > 0) {
            MPI_Isend(&local_grid1[nx*ny], nx*ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(&local_grid1[0], nx*ny, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &reqs[1]);
        }
        if (rank < size-1) {
            MPI_Isend(&local_grid1[local_nz*nx*ny], nx*ny, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&local_grid1[(local_nz+1)*nx*ny], nx*ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[3]);
        }
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        cudaMemcpy(d_grid1, local_grid1.data(), local_gridSize * sizeof(Real), cudaMemcpyHostToDevice);

        // CUDA kernel launch
        if (iter % 2 == 0) {
            stencil3d_kernel<<<grid, block>>>(d_grid1, d_grid2, nx, ny, local_nz+2);
        } else {
            stencil3d_kernel<<<grid, block>>>(d_grid2, d_grid1, nx, ny, local_nz+2);
        }
        cudaDeviceSynchronize();
    }

    cudaMemcpy(local_grid1.data(), d_grid1, local_gridSize * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(local_grid2.data(), d_grid2, local_gridSize * sizeof(Real), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
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
    
    cudaFree(d_grid1);
    cudaFree(d_grid2);
    MPI_Finalize();
    return 0;
}
