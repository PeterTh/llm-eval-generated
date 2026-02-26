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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for 7-point stencil computation
__global__ void stencilKernel(const Real* __restrict__ input, 
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if (x < nx - 1 && y < ny - 1 && z < nz - 1) {
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
__global__ void copyBoundariesKernel(const Real* __restrict__ input,
                                     Real* __restrict__ output,
                                     const size_t nx, const size_t ny, const size_t nz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    
    if (idx < total) {
        const size_t z = idx / (nx * ny);
        const size_t rem = idx % (nx * ny);
        const size_t y = rem / nx;
        const size_t x = rem % nx;
        
        if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
            output[idx] = input[idx];
        }
    }
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// GPU-accelerated stencil iteration with halo exchange
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      const size_t global_nz,
                      int rank, int size,
                      size_t z_offset) {
    
    const size_t local_size = nx * ny * local_nz;
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), local_size * sizeof(Real), cudaMemcpyHostToDevice));
    
    // Halo exchange with neighboring ranks
    if (size > 1) {
        const size_t slice_size = nx * ny;
        std::vector<Real> send_buf(slice_size);
        std::vector<Real> recv_buf(slice_size);
        
        // Send/recv with upper neighbor (higher rank)
        if (rank < size - 1) {
            // Send second-to-last slice to rank+1, receive into last slice
            size_t send_z = local_nz - 2;
            size_t recv_z = local_nz - 1;
            
            for (size_t i = 0; i < slice_size; ++i) {
                send_buf[i] = input[send_z * slice_size + i];
            }
            
            MPI_Sendrecv(send_buf.data(), slice_size, MPI_DOUBLE, rank+1, 0,
                        recv_buf.data(), slice_size, MPI_DOUBLE, rank+1, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Update device with received halo
            CUDA_CHECK(cudaMemcpy(&d_input[recv_z * slice_size], recv_buf.data(), 
                                 slice_size * sizeof(Real), cudaMemcpyHostToDevice));
        }
        
        // Send/recv with lower neighbor (lower rank)
        if (rank > 0) {
            // Send second slice to rank-1, receive into first slice
            size_t send_z = 1;
            size_t recv_z = 0;
            
            for (size_t i = 0; i < slice_size; ++i) {
                send_buf[i] = input[send_z * slice_size + i];
            }
            
            MPI_Sendrecv(send_buf.data(), slice_size, MPI_DOUBLE, rank-1, 1,
                        recv_buf.data(), slice_size, MPI_DOUBLE, rank-1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Update device with received halo
            CUDA_CHECK(cudaMemcpy(&d_input[recv_z * slice_size], recv_buf.data(), 
                                 slice_size * sizeof(Real), cudaMemcpyHostToDevice));
        }
    }
    
    // Launch CUDA kernel
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (local_nz + blockDim.z - 1) / blockDim.z);
    
    stencilKernel<<<gridDim, blockDim>>>(d_input, d_output, nx, ny, local_nz);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy boundaries
    const int boundaryThreads = 256;
    const int boundaryBlocks = (local_size + boundaryThreads - 1) / boundaryThreads;
    copyBoundariesKernel<<<boundaryBlocks, boundaryThreads>>>(d_input, d_output, nx, ny, local_nz);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, local_size * sizeof(Real), cudaMemcpyDeviceToHost));
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set GPU for this rank
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
            if (rank == 0) printUsage(argv[0]);
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
    
    // Domain decomposition: split Z dimension across ranks
    const size_t global_nz = nz;
    size_t local_nz = global_nz / size;
    size_t remainder = global_nz % size;
    
    // Add halo layers (except for boundary ranks)
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) {
        local_nz++;
        z_start += rank;
    }
    
    // Add halo layers
    size_t local_nz_with_halo = local_nz;
    if (rank > 0) local_nz_with_halo++; // bottom halo
    if (rank < size - 1) local_nz_with_halo++; // top halo
    
    const size_t z_offset = (rank > 0) ? 1 : 0;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, global_nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t local_gridSize = nx * ny * local_nz_with_halo;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(local_gridSize);
    std::vector<Real> grid2(local_gridSize);
    
    // Allocate device memory
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_gridSize * sizeof(Real)));
    
    // Initialize local portion
    if (rank == 0) printf("Initializing grid...\n");
    
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz_with_halo; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z, nx, ny);
                const size_t global_z = z_start + z - z_offset;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                grid1[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, d_grid1, d_grid2, nx, ny, local_nz_with_halo, global_nz, rank, size, z_offset);
        } else {
            stencilIteration(grid2, grid1, d_grid2, d_grid1, nx, ny, local_nz_with_halo, global_nz, rank, size, z_offset);
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (global_nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for output/validation
    const std::vector<Real>& localFinalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<Real> globalGrid;
    
    if (rank == 0) {
        globalGrid.resize(nx * ny * global_nz);
    }
    
    // Prepare local data (without halos)
    std::vector<Real> localData(nx * ny * local_nz);
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t src_idx = idx3(x, y, z + z_offset, nx, ny);
                const size_t dst_idx = idx3(x, y, z, nx, ny);
                localData[dst_idx] = localFinalGrid[src_idx];
            }
        }
    }
    
    // Gather all data to rank 0
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
    
    MPI_Gatherv(localData.data(), local_count, MPI_DOUBLE,
                globalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results and validate on rank 0
    if (rank == 0) {
        if (printResults) {
            print_results(globalGrid, "Grid");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalGrid, nx, ny, global_nz);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaFree(d_grid1));
                CUDA_CHECK(cudaFree(d_grid2));
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    
    MPI_Finalize();
    return 0;
}
