#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for 7-point stencil computation
__global__ void stencilKernel(const Real* input, Real* output,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t z_start, const size_t z_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + z_start;
    
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z < z_end) {
        const size_t idx = z * (nx * ny) + y * nx + x;
        const Real center = input[idx];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];
        const Real bottom = input[idx - nx * ny];
        const Real top = input[idx + nx * ny];
        
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation with MPI domain decomposition and CUDA
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      int mpi_rank, int mpi_size,
                      Real* d_input, Real* d_output) {
    
    // Domain decomposition: divide Z dimension among MPI ranks
    size_t z_per_rank = nz / mpi_size;
    size_t z_start = mpi_rank * z_per_rank;
    size_t z_end = (mpi_rank == mpi_size - 1) ? nz : (mpi_rank + 1) * z_per_rank;
    
    // Include halo layers for communication
    size_t z_compute_start = (mpi_rank > 0) ? z_start : 1;
    size_t z_compute_end = (mpi_rank < mpi_size - 1) ? z_end : nz - 1;
    
    size_t local_size = (z_end - z_start) * nx * ny;
    
    // Copy input to device
    cudaMemcpy(d_input, input.data() + z_start * nx * ny, local_size * sizeof(Real), cudaMemcpyHostToDevice);
    
    // MPI halo exchange: send/receive boundary layers with neighboring ranks
    std::vector<Real> recv_bottom(nx * ny), recv_top(nx * ny);
    std::vector<Real> send_bottom(nx * ny), send_top(nx * ny);
    
    if (mpi_rank > 0) {
        // Send bottom layer to rank below, receive from rank below
        #pragma omp parallel for collapse(2)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                send_bottom[y * nx + x] = input[idx3(x, y, z_start, nx, ny)];
            }
        }
        MPI_Sendrecv(send_bottom.data(), nx * ny, MPI_DOUBLE, mpi_rank - 1, 0,
                     recv_bottom.data(), nx * ny, MPI_DOUBLE, mpi_rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if (mpi_rank < mpi_size - 1) {
        // Send top layer to rank above, receive from rank above
        #pragma omp parallel for collapse(2)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                send_top[y * nx + x] = input[idx3(x, y, z_end - 1, nx, ny)];
            }
        }
        MPI_Sendrecv(send_top.data(), nx * ny, MPI_DOUBLE, mpi_rank + 1, 0,
                     recv_top.data(), nx * ny, MPI_DOUBLE, mpi_rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Launch CUDA kernel for interior computation
    dim3 blockDim(16, 16, 4);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 ((z_compute_end - z_compute_start) + blockDim.z - 1) / blockDim.z);
    
    stencilKernel<<<gridDim, blockDim>>>(d_input, d_output, nx, ny, nz, z_compute_start, z_compute_end);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(output.data() + z_start * nx * ny, d_output + z_start * nx * ny, local_size * sizeof(Real), cudaMemcpyDeviceToHost);
    
    // Copy boundary values with OpenMP parallelization
    #pragma omp parallel for collapse(2)
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            // Copy bottom boundary (from received halo or input)
            if (mpi_rank > 0) {
                output[idx3(x, y, z_start, nx, ny)] = recv_bottom[y * nx + x];
            } else {
                output[idx3(x, y, 0, nx, ny)] = input[idx3(x, y, 0, nx, ny)];
            }
            
            // Copy top boundary (from received halo or input)
            if (mpi_rank < mpi_size - 1) {
                output[idx3(x, y, z_end - 1, nx, ny)] = recv_top[y * nx + x];
            } else {
                output[idx3(x, y, nz - 1, nx, ny)] = input[idx3(x, y, nz - 1, nx, ny)];
            }
        }
    }
    
    // Copy X and Y boundaries
    #pragma omp parallel for collapse(2)
    for (size_t z = z_start; z < z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            output[idx3(0, y, z, nx, ny)] = input[idx3(0, y, z, nx, ny)];
            output[idx3(nx - 1, y, z, nx, ny)] = input[idx3(nx - 1, y, z, nx, ny)];
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (size_t z = z_start; z < z_end; ++z) {
        for (size_t x = 0; x < nx; ++x) {
            output[idx3(x, 0, z, nx, ny)] = input[idx3(x, 0, z, nx, ny)];
            output[idx3(x, ny - 1, z, nx, ny)] = input[idx3(x, ny - 1, z, nx, ny)];
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz, int mpi_rank, int mpi_size) {
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            if (mpi_rank == 0) printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded) - use MPI reduction for global min/max
    Real local_min = grid[0];
    Real local_max = grid[0];
    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max)
    for (const auto& val : grid) {
        local_min = std::min(local_min, val);
        local_max = std::max(local_max, val);
    }
    
    Real global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
    }
    
    if (global_max > 1e6 || global_min < -1e6) {
        if (mpi_rank == 0) printf("Validation failed: values out of expected range\n");
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
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
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
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (mpi_rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d\n", mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t gridSize = nx * ny * nz;
    
    // Get CUDA device
    int device = mpi_rank % 1;  // In real setup, would be mpi_rank % gpus_per_node
    cudaSetDevice(device);
    
    // Allocate device memory
    Real* d_input = nullptr;
    Real* d_output = nullptr;
    cudaMalloc((void**)&d_input, gridSize * sizeof(Real));
    cudaMalloc((void**)&d_output, gridSize * sizeof(Real));
    
    // Allocate host grids
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    if (mpi_rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    
    // Copy initial grid to device
    cudaMemcpy(d_input, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice);
    
    // Run stencil iterations
    if (mpi_rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz, mpi_rank, mpi_size, d_input, d_output);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz, mpi_rank, mpi_size, d_output, d_input);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults && mpi_rank == 0) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResult(finalGrid, nx, ny, nz, mpi_rank, mpi_size);
        
        if (mpi_rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    MPI_Finalize();
    
    return 0;
}
