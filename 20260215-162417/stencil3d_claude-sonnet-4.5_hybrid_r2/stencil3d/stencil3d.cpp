#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

#ifdef USE_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)
#endif

// 3D index calculation
#ifdef USE_CUDA
__host__ __device__ 
#endif
inline size_t idx3(const size_t x, const size_t y, const size_t z, 
                    const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

#ifdef USE_CUDA
// CUDA kernel for grid initialization
__global__ void initializeGridKernel(Real* grid, const size_t nx, const size_t ny, 
                                     const size_t nz, const size_t z_offset) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        const size_t global_z = z + z_offset;
        const size_t global_idx = global_z * (nx * ny) + y * nx + x;
        grid[idx] = (global_idx % 19) * 1.0;
    }
}

// CUDA kernel for 7-point stencil computation
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Process interior points only
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1) {
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

// CUDA kernel for boundary copy
__global__ void copyBoundaryKernel(const Real* __restrict__ input, Real* __restrict__ output,
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

void initializeGrid(Real* d_grid, const size_t nx, const size_t ny, const size_t nz,
                    const size_t z_offset, cudaStream_t stream) {
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz + blockDim.z - 1) / blockDim.z);
    
    initializeGridKernel<<<gridDim, blockDim, 0, stream>>>(d_grid, nx, ny, nz, z_offset);
    CUDA_CHECK(cudaGetLastError());
}

// 7-point stencil computation with MPI halo exchange
void stencilIteration(Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz_local,
                      int rank, int size, cudaStream_t stream,
                      Real* h_send_bottom, Real* h_send_top,
                      Real* h_recv_bottom, Real* h_recv_top) {
    const size_t slice_size = nx * ny;
    
    // Halo exchange with neighbors
    MPI_Request requests[4];
    int req_count = 0;
    
    // Send bottom halo (z=1) to rank-1, receive from rank-1 into z=0
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_bottom, d_input + slice_size, 
                                   slice_size * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Isend(h_send_bottom, slice_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(h_recv_bottom, slice_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Send top halo (z=nz_local-2) to rank+1, receive from rank+1 into z=nz_local-1
    if (rank < size - 1) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_top, d_input + (nz_local - 2) * slice_size,
                                   slice_size * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Isend(h_send_top, slice_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(h_recv_top, slice_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Compute interior while communication happens
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz_local + blockDim.z - 1) / blockDim.z);
    
    stencilKernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, nx, ny, nz_local);
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for halo exchange to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        
        // Copy received halos back to device
        if (rank > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_input, h_recv_bottom, slice_size * sizeof(Real),
                                       cudaMemcpyHostToDevice, stream));
        }
        if (rank < size - 1) {
            CUDA_CHECK(cudaMemcpyAsync(d_input + (nz_local - 1) * slice_size, h_recv_top,
                                       slice_size * sizeof(Real), cudaMemcpyHostToDevice, stream));
        }
    }
    
    // Copy boundaries
    copyBoundaryKernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, nx, ny, nz_local);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool validateResult(const Real* d_grid, const size_t gridSize) {
    // Copy data from device to host for validation
    std::vector<Real> h_grid(gridSize);
    CUDA_CHECK(cudaMemcpy(h_grid.data(), d_grid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : h_grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = h_grid[0];
    Real maxVal = h_grid[0];
    for (const auto& val : h_grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}

#else
// CPU-only fallback implementation

void initializeGrid(Real* grid, const size_t nx, const size_t ny, const size_t nz,
                    const size_t z_offset) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t global_z = z + z_offset;
                const size_t global_idx = global_z * (nx * ny) + y * nx + x;
                grid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

void stencilIteration(Real* input, Real* output,
                      const size_t nx, const size_t ny, const size_t nz_local,
                      int rank, int size,
                      Real* send_bottom, Real* send_top,
                      Real* recv_bottom, Real* recv_top) {
    const size_t slice_size = nx * ny;
    
    // Halo exchange with neighbors
    MPI_Request requests[4];
    int req_count = 0;
    
    // Send bottom halo (z=1) to rank-1, receive from rank-1 into z=0
    if (rank > 0) {
        std::memcpy(send_bottom, input + slice_size, slice_size * sizeof(Real));
        MPI_Isend(send_bottom, slice_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(recv_bottom, slice_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Send top halo (z=nz_local-2) to rank+1, receive from rank+1 into z=nz_local-1
    if (rank < size - 1) {
        std::memcpy(send_top, input + (nz_local - 2) * slice_size, slice_size * sizeof(Real));
        MPI_Isend(send_top, slice_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(recv_top, slice_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Compute interior
    #pragma omp parallel for collapse(3)
    for (size_t z = 1; z < nz_local - 1; ++z) {
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
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Wait for halo exchange to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        
        // Copy received halos
        if (rank > 0) {
            std::memcpy(input, recv_bottom, slice_size * sizeof(Real));
        }
        if (rank < size - 1) {
            std::memcpy(input + (nz_local - 1) * slice_size, recv_top, slice_size * sizeof(Real));
        }
    }
    
    // Copy boundaries
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz_local-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

bool validateResult(const Real* grid, const size_t gridSize) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (size_t i = 0; i < gridSize; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (size_t i = 0; i < gridSize; ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}
#endif

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
    
#ifdef USE_CUDA
    // Set GPU device based on local rank (assumes one GPU per MPI rank)
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device));
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("CUDA devices: %d\n", num_devices);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
#else
    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
#endif
    
    // Domain decomposition in Z dimension
    size_t nz_local = nz / size;
    size_t remainder = nz % size;
    
    // Distribute remainder among first ranks
    size_t z_start = rank * nz_local + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) {
        nz_local++;
    }
    
    // Add halo layers (1 on each side for stencil)
    size_t nz_with_halo = nz_local + 2;
    size_t gridSize_local = nx * ny * nz_with_halo;
    size_t slice_size = nx * ny;
    
    if (rank == 0) {
        printf("Local Z-slices per rank: %zu (with halos: %zu)\n", nz_local, nz_with_halo);
    }
    
#ifdef USE_CUDA
    // Allocate device memory (double buffering)
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize_local * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize_local * sizeof(Real)));
    
    // Allocate pinned host memory for MPI halo exchange
    Real *h_send_bottom, *h_send_top, *h_recv_bottom, *h_recv_top;
    CUDA_CHECK(cudaMallocHost(&h_send_bottom, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_send_top, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, slice_size * sizeof(Real)));
    
    // Create CUDA streams for overlapping computation and communication
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Initialize grid
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    
    // Initialize interior (excluding halo layers)
    initializeGrid(d_grid1 + slice_size, nx, ny, nz_local, z_start, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Initialize halo layers
    CUDA_CHECK(cudaMemset(d_grid1, 0, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_grid1 + (nz_local + 1) * slice_size, 0, slice_size * sizeof(Real)));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(1)
    {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(d_grid1, d_grid2, nx, ny, nz_with_halo, 
                               rank, size, stream, 
                               h_send_bottom, h_send_top, h_recv_bottom, h_recv_top);
            } else {
                stencilIteration(d_grid2, d_grid1, nx, ny, nz_with_halo,
                               rank, size, stream,
                               h_send_bottom, h_send_top, h_recv_bottom, h_recv_top);
            }
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0 for printing
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    
    if (printResults) {
        // Allocate full grid on rank 0
        std::vector<Real> fullGrid;
        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
        }
        
        // Copy local data (without halos) from device to host
        std::vector<Real> localGrid(nx * ny * nz_local);
        CUDA_CHECK(cudaMemcpy(localGrid.data(), d_final + slice_size,
                             nx * ny * nz_local * sizeof(Real), cudaMemcpyDeviceToHost));
        
        // Gather all local grids to rank 0
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int local_size = nx * ny * nz_local;
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(localGrid.data(), local_size, MPI_DOUBLE,
                   fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(fullGrid, "Grid");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        
        bool local_valid = validateResult(d_final, gridSize_local);
        bool global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (global_valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_grid1));
        CUDA_CHECK(cudaFree(d_grid2));
        CUDA_CHECK(cudaFreeHost(h_send_bottom));
        CUDA_CHECK(cudaFreeHost(h_send_top));
        CUDA_CHECK(cudaFreeHost(h_recv_bottom));
        CUDA_CHECK(cudaFreeHost(h_recv_top));
        CUDA_CHECK(cudaStreamDestroy(stream));
        
        MPI_Finalize();
        return global_valid ? 0 : 1;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
#else
    // CPU-only version with MPI+OpenMP
    
    // Allocate host memory (double buffering)
    std::vector<Real> grid1(gridSize_local);
    std::vector<Real> grid2(gridSize_local);
    
    // Allocate buffers for MPI halo exchange
    std::vector<Real> send_bottom(slice_size);
    std::vector<Real> send_top(slice_size);
    std::vector<Real> recv_bottom(slice_size);
    std::vector<Real> recv_top(slice_size);
    
    // Initialize grid
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    
    // Initialize interior (excluding halo layers)
    initializeGrid(grid1.data() + slice_size, nx, ny, nz_local, z_start);
    
    // Initialize halo layers
    std::fill(grid1.begin(), grid1.begin() + slice_size, 0.0);
    std::fill(grid1.end() - slice_size, grid1.end(), 0.0);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1.data(), grid2.data(), nx, ny, nz_with_halo, 
                           rank, size,
                           send_bottom.data(), send_top.data(), 
                           recv_bottom.data(), recv_top.data());
        } else {
            stencilIteration(grid2.data(), grid1.data(), nx, ny, nz_with_halo,
                           rank, size,
                           send_bottom.data(), send_top.data(),
                           recv_bottom.data(), recv_top.data());
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0 for printing
    Real* final_grid = (iterations % 2 == 0) ? grid1.data() : grid2.data();
    
    if (printResults) {
        // Allocate full grid on rank 0
        std::vector<Real> fullGrid;
        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
        }
        
        // Gather all local grids to rank 0
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int local_size = nx * ny * nz_local;
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(final_grid + slice_size, local_size, MPI_DOUBLE,
                   fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(fullGrid, "Grid");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        
        bool local_valid = validateResult(final_grid, gridSize_local);
        bool global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (global_valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return global_valid ? 0 : 1;
    }
#endif
    
    MPI_Finalize();
    return 0;
}
