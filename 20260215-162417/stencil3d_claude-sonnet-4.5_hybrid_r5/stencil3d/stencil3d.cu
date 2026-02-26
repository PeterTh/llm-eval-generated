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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz, const size_t z_offset) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z, nx, ny);
                const size_t global_idx = idx3(x, y, z + z_offset, nx, ny);
                grid[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// CUDA kernel for 7-point stencil computation
__global__ void stencilKernel(const Real* __restrict__ input, 
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Process interior points only
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1) {
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

// CUDA kernel for boundary copying
__global__ void copyBoundaryKernel(const Real* __restrict__ input,
                                   Real* __restrict__ output,
                                   const size_t nx, const size_t ny, const size_t nz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    
    if (idx < total) {
        const size_t x = idx % nx;
        const size_t y = (idx / nx) % ny;
        const size_t z = idx / (nx * ny);
        
        if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
            output[idx] = input[idx];
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Process interior points (not on boundaries)
    #pragma omp parallel for collapse(3)
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Set GPU device based on local rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device_id = world_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));
    
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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition along Z-axis
    size_t local_nz = nz / world_size;
    size_t remainder = nz % world_size;
    size_t z_start = world_rank * local_nz + std::min((size_t)world_rank, remainder);
    if (world_rank < (int)remainder) {
        local_nz++;
    }
    size_t z_end = z_start + local_nz;
    
    // Add halo zones (ghost cells for stencil)
    size_t local_nz_with_halo = local_nz + 2;
    size_t local_gridSize = nx * ny * local_nz_with_halo;
    
    if (world_rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", world_size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate host memory with halo zones
    std::vector<Real> grid1(local_gridSize);
    std::vector<Real> grid2(local_gridSize);
    
    // Initialize local domain (excluding halo zones)
    if (world_rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, local_nz_with_halo, z_start);
    
    // Allocate device memory
    Real *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, local_gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_output, local_gridSize * sizeof(Real)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_input, grid1.data(), local_gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    
    // Setup CUDA kernel configuration
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (local_nz_with_halo + blockDim.z - 1) / blockDim.z);
    
    const size_t boundaryThreads = 256;
    const size_t boundaryBlocks = (local_gridSize + boundaryThreads - 1) / boundaryThreads;
    
    // Prepare halo buffers
    size_t halo_size = nx * ny;
    std::vector<Real> send_top(halo_size), send_bottom(halo_size);
    std::vector<Real> recv_top(halo_size), recv_bottom(halo_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (world_rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        Real *d_in = (iter % 2 == 0) ? d_input : d_output;
        Real *d_out = (iter % 2 == 0) ? d_output : d_input;
        
        // Halo exchange
        // Copy data from device for halo exchange
        if (world_size > 1) {
            // Copy halo zones from device to host
            size_t top_offset = idx3(0, 0, local_nz, nx, ny);
            size_t bottom_offset = idx3(0, 0, 1, nx, ny);
            
            CUDA_CHECK(cudaMemcpy(send_top.data(), d_in + top_offset, 
                                 halo_size * sizeof(Real), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(send_bottom.data(), d_in + bottom_offset, 
                                 halo_size * sizeof(Real), cudaMemcpyDeviceToHost));
            
            // Exchange with neighbors
            MPI_Request requests[4];
            int req_count = 0;
            
            // Send to top neighbor, receive from bottom neighbor
            if (world_rank < world_size - 1) {
                MPI_Isend(send_top.data(), halo_size, MPI_DOUBLE, world_rank + 1, 0, 
                         MPI_COMM_WORLD, &requests[req_count++]);
            }
            if (world_rank > 0) {
                MPI_Irecv(recv_bottom.data(), halo_size, MPI_DOUBLE, world_rank - 1, 0, 
                         MPI_COMM_WORLD, &requests[req_count++]);
            }
            
            // Send to bottom neighbor, receive from top neighbor
            if (world_rank > 0) {
                MPI_Isend(send_bottom.data(), halo_size, MPI_DOUBLE, world_rank - 1, 1, 
                         MPI_COMM_WORLD, &requests[req_count++]);
            }
            if (world_rank < world_size - 1) {
                MPI_Irecv(recv_top.data(), halo_size, MPI_DOUBLE, world_rank + 1, 1, 
                         MPI_COMM_WORLD, &requests[req_count++]);
            }
            
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
            
            // Copy received halo data back to device
            if (world_rank > 0) {
                size_t recv_bottom_offset = idx3(0, 0, 0, nx, ny);
                CUDA_CHECK(cudaMemcpy(d_in + recv_bottom_offset, recv_bottom.data(),
                                     halo_size * sizeof(Real), cudaMemcpyHostToDevice));
            }
            if (world_rank < world_size - 1) {
                size_t recv_top_offset = idx3(0, 0, local_nz + 1, nx, ny);
                CUDA_CHECK(cudaMemcpy(d_in + recv_top_offset, recv_top.data(),
                                     halo_size * sizeof(Real), cudaMemcpyHostToDevice));
            }
        }
        
        // Launch CUDA kernel for interior computation
        stencilKernel<<<gridDim, blockDim>>>(d_in, d_out, nx, ny, local_nz_with_halo);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy boundary values
        copyBoundaryKernel<<<boundaryBlocks, boundaryThreads>>>(d_in, d_out, nx, ny, local_nz_with_halo);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy final result back to host
    Real *d_final = (iterations % 2 == 0) ? d_input : d_output;
    CUDA_CHECK(cudaMemcpy(grid1.data(), d_final, local_gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    
    // Gather results to rank 0 for validation and output
    std::vector<Real> global_grid;
    if (world_rank == 0) {
        global_grid.resize(nx * ny * nz);
    }
    
    // Prepare local data (excluding halo zones) for gathering
    std::vector<Real> local_data(nx * ny * local_nz);
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t local_idx_with_halo = idx3(x, y, z + 1, nx, ny);
                size_t local_idx = idx3(x, y, z, nx, ny);
                local_data[local_idx] = grid1[local_idx_with_halo];
            }
        }
    }
    
    // Gather sizes and displacements
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    
    int local_count = nx * ny * local_nz;
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(local_data.data(), local_count, MPI_DOUBLE,
                global_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        
        // Print results for external validation
        if (printResults) {
            print_results(global_grid, "Grid");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_grid, nx, ny, nz);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    MPI_Finalize();
    return 0;
}
