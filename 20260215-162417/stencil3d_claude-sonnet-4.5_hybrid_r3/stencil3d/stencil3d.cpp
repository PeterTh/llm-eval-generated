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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for initialization
__global__ void initializeGridKernel(Real* grid, const size_t nx, const size_t ny, const size_t nz, const size_t z_offset) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        size_t idx = idx3(x, y, z, nx, ny);
        size_t global_idx = idx3(x, y, z + z_offset, nx, ny);
        grid[idx] = (global_idx % 19) * 1.0;
    }
}

// CUDA kernel for stencil computation
__global__ void stencilKernel(const Real* __restrict__ input, 
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Process interior points
    if (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1 && z >= 1 && z < nz - 1) {
        const size_t idx = idx3(x, y, z, nx, ny);
        
        const Real center = input[idx];
        const Real left = input[idx3(x-1, y, z, nx, ny)];
        const Real right = input[idx3(x+1, y, z, nx, ny)];
        const Real front = input[idx3(x, y-1, z, nx, ny)];
        const Real back = input[idx3(x, y+1, z, nx, ny)];
        const Real bottom = input[idx3(x, y, z-1, nx, ny)];
        const Real top = input[idx3(x, y, z+1, nx, ny)];
        
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// CUDA kernel for boundary copying
__global__ void copyBoundaryKernel(const Real* __restrict__ input, 
                                   Real* __restrict__ output,
                                   const size_t nx, const size_t ny, const size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
            const size_t idx = idx3(x, y, z, nx, ny);
            output[idx] = input[idx];
        }
    }
}

void initializeGrid(Real* d_grid, const size_t nx, const size_t ny, const size_t nz, const size_t z_offset) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    initializeGridKernel<<<gridSize, blockSize>>>(d_grid, nx, ny, nz, z_offset);
    CUDA_CHECK(cudaGetLastError());
}

// 7-point stencil computation on GPU
void stencilIteration(Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    stencilKernel<<<gridSize, blockSize>>>(d_input, d_output, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    
    copyBoundaryKernel<<<gridSize, blockSize>>>(d_input, d_output, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    #pragma omp parallel for
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            printf("Validation failed: found NaN or Inf value\n");
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set CUDA device based on rank (for multi-GPU nodes)
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition in Z dimension
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    
    // Distribute remainder across first ranks
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) {
        local_nz++;
    }
    
    // Add halo zones (except at global boundaries)
    size_t local_nz_with_halo = local_nz;
    size_t halo_offset = 0;
    
    if (rank > 0) {
        local_nz_with_halo++;
        halo_offset = 1;
    }
    if (rank < size - 1) {
        local_nz_with_halo++;
    }
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t local_gridSize = nx * ny * local_nz_with_halo;
    
    // Allocate device memory (double buffering)
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_gridSize * sizeof(Real)));
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(d_grid1, nx, ny, local_nz_with_halo, z_start - halo_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Allocate host buffers for halo exchange
    std::vector<Real> send_buffer_top(nx * ny);
    std::vector<Real> send_buffer_bottom(nx * ny);
    std::vector<Real> recv_buffer_top(nx * ny);
    std::vector<Real> recv_buffer_bottom(nx * ny);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        Real *d_input = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real *d_output = (iter % 2 == 0) ? d_grid2 : d_grid1;
        
        // Halo exchange with neighbors
        MPI_Request requests[4];
        int req_count = 0;
        
        // Send to rank+1 (top), receive from rank-1 (bottom)
        if (rank < size - 1) {
            // Copy top boundary from device to host
            size_t top_plane_offset = (local_nz_with_halo - 2) * nx * ny;
            CUDA_CHECK(cudaMemcpy(send_buffer_top.data(), d_input + top_plane_offset,
                                 nx * ny * sizeof(Real), cudaMemcpyDeviceToHost));
            MPI_Isend(send_buffer_top.data(), nx * ny, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        if (rank > 0) {
            MPI_Irecv(recv_buffer_bottom.data(), nx * ny, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        // Send to rank-1 (bottom), receive from rank+1 (top)
        if (rank > 0) {
            // Copy bottom boundary from device to host
            size_t bottom_plane_offset = 1 * nx * ny;
            CUDA_CHECK(cudaMemcpy(send_buffer_bottom.data(), d_input + bottom_plane_offset,
                                 nx * ny * sizeof(Real), cudaMemcpyDeviceToHost));
            MPI_Isend(send_buffer_bottom.data(), nx * ny, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        if (rank < size - 1) {
            MPI_Irecv(recv_buffer_top.data(), nx * ny, MPI_DOUBLE, rank + 1, 1,
                     MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        // Wait for all communications to complete
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        
        // Copy received halos back to device
        if (rank > 0) {
            CUDA_CHECK(cudaMemcpy(d_input, recv_buffer_bottom.data(),
                                 nx * ny * sizeof(Real), cudaMemcpyHostToDevice));
        }
        
        if (rank < size - 1) {
            size_t top_halo_offset = (local_nz_with_halo - 1) * nx * ny;
            CUDA_CHECK(cudaMemcpy(d_input + top_halo_offset, recv_buffer_top.data(),
                                 nx * ny * sizeof(Real), cudaMemcpyHostToDevice));
        }
        
        // Perform stencil computation on GPU
        stencilIteration(d_input, d_output, nx, ny, local_nz_with_halo);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather timing information
    long local_time = duration.count();
    long max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_time);
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_time / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather final results to rank 0 for validation/output
    Real *d_finalGrid = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    
    std::vector<Real> local_grid(local_gridSize);
    CUDA_CHECK(cudaMemcpy(local_grid.data(), d_finalGrid, 
                         local_gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    
    // Extract interior data (without halos)
    std::vector<Real> local_interior(nx * ny * local_nz);
    #pragma omp parallel for
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t src_idx = idx3(x, y, z + halo_offset, nx, ny);
                size_t dst_idx = idx3(x, y, z, nx, ny);
                local_interior[dst_idx] = local_grid[src_idx];
            }
        }
    }
    
    std::vector<Real> globalGrid;
    if (rank == 0) {
        globalGrid.resize(nx * ny * nz);
    }
    
    // Gather all local grids to rank 0
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    int local_count = nx * ny * local_nz;
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(local_interior.data(), local_count, MPI_DOUBLE,
                globalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation (rank 0 only)
    if (rank == 0 && printResults) {
        print_results(globalGrid, "Grid");
    }
    
    // Validation (rank 0 only)
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(globalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    
    MPI_Finalize();
    return 0;
}
