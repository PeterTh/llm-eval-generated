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

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// 7-point stencil computation kernel
__global__ void stencilIterationKernel(const Real* __restrict__ input, 
                                       Real* __restrict__ output,
                                       const size_t nx, const size_t ny, const size_t nz,
                                       const int rank, const int size) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    // We iterate over Z to improve cache/memory locality if block is 2D
    // Or we can use 3D blocks. Let's use 2D blocks (XY) and loop Z.
    // This reduces kernel launch overhead and allows reusing X/Y index.
    
    if (x >= nx || y >= ny) return;

    // Boundary conditions:
    // x=0, x=nx-1, y=0, y=ny-1 are physical boundaries -> copy
    // z=0, z=nz-1 are local boundaries (ghosts or physical) -> copy/skip
    
    // We process indices 0..nz-1.
    // However, we only compute stencils for z=1..nz-2.
    // z=0 and z=nz-1 are handled by copy or skipped (will be overwritten by halo exchange)
    
    for (size_t z = 0; z < nz; ++z) {
        const size_t idx = idx3(x, y, z, nx, ny);
        
        bool is_z_boundary = (z == 0 || z == nz - 1);
        bool is_xy_boundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);
        
        if (is_xy_boundary) {
            output[idx] = input[idx];
        } else if (is_z_boundary) {
            // Copy value. Correct for physical boundaries.
            // For ghost boundaries, this value will be overwritten by MPI exchange later.
            output[idx] = input[idx];
        } else {
            // Interior point
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

bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    // Simple sanity checks
    Real minVal = grid[0];
    Real maxVal = grid[0];
    bool valid = true;
    
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range [%f, %f]\n", minVal, maxVal);
        valid = false;
    }
    
    return valid;
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

    // Select GPU
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        CHECK_CUDA(cudaSetDevice(rank % num_devices));
    } else {
        if (rank == 0) printf("Error: No CUDA devices found!\n");
        MPI_Finalize();
        return 1;
    }
    
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
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
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI + CUDA + OpenMP)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
        printf("Iterations: %d\n", iterations);
    }
    
    // Domain Decomposition
    size_t local_nz_inner = nz / size;
    size_t remainder = nz % size;
    size_t z_start_global = rank * local_nz_inner + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) local_nz_inner++;
    
    size_t local_nz = local_nz_inner + 2; // +2 for ghost layers (top and bottom)
    
    size_t gridSize = nx * ny * local_nz;
    
    // Allocate host memory (pinned for faster transfer)
    Real *h_grid1, *h_grid2;
    CHECK_CUDA(cudaMallocHost(&h_grid1, gridSize * sizeof(Real)));
    CHECK_CUDA(cudaMallocHost(&h_grid2, gridSize * sizeof(Real)));
    
    // Initialize grid
    // Need to initialize with correct global indices
    // z index in h_grid maps to global index: z_start_global + (z - 1)
    // because h_grid[0] is ghost (global z_start_global - 1)
    // and h_grid[1] is start of real data (global z_start_global)
    
    // However, simplicity:
    // Let's iterate z from 0 to local_nz-1.
    // Calculate global_z for each.
    // global_z = z_start_global + (z - 1).
    // If rank == 0, z=0 corresponds to global -1 (boundary condition? no, physical boundary is global 0)
    // If rank == 0, local z=1 is global z=0.
    // Wait, let's adjust mapping.
    // Standard: 
    //   Rank 0 owns global [0, k-1].
    //   It needs ghost at -1 (doesn't exist) and k.
    //   So local buffer size k+2.
    //   Index 0: ghost (-1).
    //   Index 1..k: data (0..k-1).
    //   Index k+1: ghost (k).
    
    // Let's use this mapping.
    // global_z = z_start_global + (z - 1);
    
    #pragma omp parallel for collapse(2)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                long long gz = (long long)z_start_global + ((long long)z - 1);
                // Clamp to physical boundaries for initialization logic if needed,
                // but idx3 in initializeGrid logic used modulo, so it's fine to pass raw coordinate.
                // The original initialization:
                // grid[idx] = (idx3(x, y, z, nx, ny) % 19) * 1.0;
                // It used idx3 with GLOBAL dimensions and coordinates.
                // So we need:
                size_t global_idx = idx3(x, y, (size_t)std::max(0LL, std::min((long long)nz-1, gz)), nx, ny); 
                // Wait, simply passing global coordinates to logic?
                // original: grid[idx] = (idx % 19) * 1.0;
                // idx = z * (nx*ny) + ...
                
                // Construct global idx
                size_t true_global_idx = 0;
                if (gz >= 0 && gz < (long long)nz) {
                     true_global_idx = idx3(x, y, (size_t)gz, nx, ny);
                } else {
                     // For ghost cells outside domain, value doesn't strictly matter as they are overwritten
                     // but for consistency let's use nearest boundary
                     size_t clamped_z = (gz < 0) ? 0 : nz - 1;
                     true_global_idx = idx3(x, y, clamped_z, nx, ny);
                }
                
                h_grid1[idx3(x, y, z, nx, ny)] = (true_global_idx % 19) * 1.0;
            }
        }
    }
    
    // Allocate device memory
    Real *d_grid1, *d_grid2;
    CHECK_CUDA(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CHECK_CUDA(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
    
    CHECK_CUDA(cudaMemcpy(d_grid1, h_grid1, gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid2, h_grid1, gridSize * sizeof(Real), cudaMemcpyHostToDevice)); // Initialize grid2 too
    
    // Buffers for halo exchange
    size_t planeSize = nx * ny;
    Real *h_send_top, *h_recv_top, *h_send_bot, *h_recv_bot;
    CHECK_CUDA(cudaMallocHost(&h_send_top, planeSize * sizeof(Real)));
    CHECK_CUDA(cudaMallocHost(&h_recv_top, planeSize * sizeof(Real)));
    CHECK_CUDA(cudaMallocHost(&h_send_bot, planeSize * sizeof(Real)));
    CHECK_CUDA(cudaMallocHost(&h_recv_bot, planeSize * sizeof(Real)));
    
    // Real *d_top_halo_buf, *d_bot_halo_buf; // Removed unused variables
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 blockSize(32, 8);
    dim3 gridSizeKernel((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);
    
    for (int iter = 0; iter < iterations; ++iter) {
        Real *d_in = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real *d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;
        
        // 1. Compute Stencil (Interior + Boundary Copy)
        // Note: This updates 'd_out' based on 'd_in'.
        // 'd_in' has valid ghosts from previous step.
        stencilIterationKernel<<<gridSizeKernel, blockSize>>>(d_in, d_out, nx, ny, local_nz, rank, size);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize()); // Wait for compute before exchange
        
        // 2. Halo Exchange
        // We need to exchange data from 'd_out' to prepare for next iteration (where it becomes input).
        // Wait, standard double buffering:
        // Compute: Read T, Write T+1.
        // Exchange: Update ghosts of T+1 using T+1 data.
        // Next step: Read T+1 (now valid with ghosts), Write T+2.
        // Yes.
        
        // Prepare sends
        // Send UP: Send data from z = local_nz - 2 (top inner) to Rank+1
        // Recv UP: Recv data from Rank+1 into z = local_nz - 1 (top ghost)
        
        // Send DOWN: Send data from z = 1 (bottom inner) to Rank-1
        // Recv DOWN: Recv data from Rank-1 into z = 0 (bottom ghost)
        
        // Copy data to host buffers
        size_t top_inner_idx = idx3(0, 0, local_nz - 2, nx, ny);
        size_t bot_inner_idx = idx3(0, 0, 1, nx, ny);
        
        // Async copies? For simplicity, sync.
        CHECK_CUDA(cudaMemcpy(h_send_top, d_out + top_inner_idx, planeSize * sizeof(Real), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_send_bot, d_out + bot_inner_idx, planeSize * sizeof(Real), cudaMemcpyDeviceToHost));
        
        MPI_Request reqs[4];
        int nreqs = 0;
        
        // Exchange with Top (Rank + 1)
        if (rank < size - 1) {
            MPI_Isend(h_send_top, planeSize, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(h_recv_top, planeSize, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        
        // Exchange with Bottom (Rank - 1)
        if (rank > 0) {
            MPI_Isend(h_send_bot, planeSize, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(h_recv_bot, planeSize, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
        
        // Copy received data to device ghosts
        if (rank < size - 1) {
             size_t top_ghost_idx = idx3(0, 0, local_nz - 1, nx, ny);
             CHECK_CUDA(cudaMemcpy(d_out + top_ghost_idx, h_recv_top, planeSize * sizeof(Real), cudaMemcpyHostToDevice));
        }
        
        if (rank > 0) {
             size_t bot_ghost_idx = idx3(0, 0, 0, nx, ny);
             CHECK_CUDA(cudaMemcpy(d_out + bot_ghost_idx, h_recv_bot, planeSize * sizeof(Real), cudaMemcpyHostToDevice));
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results if validation or print required
    if (validate || printResults) {
        Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
        CHECK_CUDA(cudaMemcpy(h_grid1, d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
        
        // We need to strip ghost layers and gather to root.
        // Simplest: gather everything to root and reconstruct.
        // Be careful with memory on root. If grid is huge, this might fail, but for benchmark it's ok.
        
        // Local buffer size: local_nz * planeSize.
        // Valid data is at offset planeSize (z=1) of size (local_nz-2)*planeSize.
        // Wait, rank 0 valid data starts at z=1 (global z=0).
        // rank N-1 valid data ends at local_nz-2 (global nz-1).
        
        // Actually, let's just copy the "owned" portion.
        // Rank 0: owns global 0..local_nz_inner-1.
        //    Local indices: 1 .. local_nz_inner.
        // Rank k: owns global ...
        //    Local indices: 1 .. local_nz_inner.
        
        // So everyone sends 'local_nz_inner' planes starting from offset 'planeSize'.
        
        std::vector<Real> fullGrid;
        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
        }
        
        // This is tricky with MPI_Gatherv because counts might differ (remainder).
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_count = local_nz_inner * planeSize;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(h_grid1 + planeSize, my_count, MPI_DOUBLE, 
                   fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                   0, MPI_COMM_WORLD);
                   
        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                if (validateResult(fullGrid, nx, ny, nz)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }
    
    CHECK_CUDA(cudaFreeHost(h_grid1));
    CHECK_CUDA(cudaFreeHost(h_grid2));
    CHECK_CUDA(cudaFreeHost(h_send_top));
    CHECK_CUDA(cudaFreeHost(h_recv_top));
    CHECK_CUDA(cudaFreeHost(h_send_bot));
    CHECK_CUDA(cudaFreeHost(h_recv_bot));
    CHECK_CUDA(cudaFree(d_grid1));
    CHECK_CUDA(cudaFree(d_grid2));
    
    MPI_Finalize();
    return 0;
}
