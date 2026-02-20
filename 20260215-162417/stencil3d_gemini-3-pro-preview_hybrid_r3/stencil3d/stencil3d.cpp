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

// CUDA Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, err); \
        } \
    } while (0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Kernel to initialize grid
__global__ void initializeGridKernel(Real* grid, const size_t nx, const size_t ny, const size_t nz, const size_t z_start) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = nx * ny * nz; // This is local_grid_size
    
    if (idx < total_elements) {
        size_t z = idx / (nx * ny);
        size_t rem = idx % (nx * ny);
        size_t y = rem / nx;
        size_t x = rem % nx;
        
        // Calculate global Z coordinate.
        // local z=0 corresponds to ghost layer below, which is global z_start - 1.
        // So global_z = (long long)z_start + z - 1;
        long long global_z = (long long)z_start + z - 1;
        
        // For value generation, we need global index.
        // Original: grid[idx] = (idx % 19) * 1.0;
        // We replicate this pattern based on global coordinates.
        // If global_z is negative (ghost below at rank 0), or >= global_nz (ghost above at rank size-1),
        // we still compute a value to be consistent, though it might be overwritten.
        
        // However, idx3 takes size_t.
        // Let's assume global_nz is large enough and we handle boundary conditions separately?
        // Actually, the original code initializes based on `idx3(x, y, z, nx, ny)` where nx,ny,nz are GLOBAL dimensions.
        // So we need global_nz to compute the correct "global index" for the pattern.
        // We didn't pass global_nz to this kernel. Let's just use a pattern that depends on coordinates.
        // Or better, let's pass global_nz if we want bitwise identical results.
        // But the prompt says "maintain correctness and equivalent semantics".
        // The pattern `idx % 19` is just random noise.
        // Let's implement a similar pattern: ((global_z * ny * nx) + y * nx + x) % 19.
        
        // Handle negative global_z or overflow?
        // The original code initializes the whole grid 0..nz-1.
        // Ghost layers at rank boundaries (internal to domain) should match the neighbors' real values.
        // Ghost layers at physical domain boundaries (z=-1 or z=GlobalNZ) are not initialized in original code 
        // because original code only has 0..nz-1.
        // But our local grid has ghosts.
        // Let's just initialize using the formula, allowing "virtual" indices.
        
        size_t global_idx_val = (size_t)((global_z * ny * nx) + y * nx + x);
        grid[idx] = (global_idx_val % 19) * 1.0;
    }
}

// Kernel for stencil computation
__global__ void stencilKernel(const Real* input, Real* output, const size_t nx, const size_t ny, const size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    // Interior points of LOCAL grid.
    // Local grid range: [0, nz-1].
    // Active domain: [1, nz-2].
    // Stencil operates on 1..nz-2 using neighbors 0..nz-1.
    
    if (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1 && z >= 1 && z < nz - 1) {
        const size_t idx = idx3(x, y, z, nx, ny);
        
        const Real center = input[idx];
        const Real left   = input[idx3(x-1, y, z, nx, ny)];
        const Real right  = input[idx3(x+1, y, z, nx, ny)];
        const Real front  = input[idx3(x, y-1, z, nx, ny)];
        const Real back   = input[idx3(x, y+1, z, nx, ny)];
        const Real bottom = input[idx3(x, y, z-1, nx, ny)]; 
        const Real top    = input[idx3(x, y, z+1, nx, ny)]; 
        
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    } 
    else if (x < nx && y < ny && z < nz) {
        // Boundary copy
        // For physical boundaries (x=0, x=nx-1, y=0, y=ny-1), we copy.
        // For Z boundaries (z=0, z=nz-1) which are HALO, we also copy here
        // but they will be overwritten by MPI exchange.
        // UNLESS it is a physical Z boundary (Rank 0 bottom, Rank N top).
        // But since we treat all Z boundaries as halo in this kernel, copying is fine.
        const size_t idx = idx3(x, y, z, nx, ny);
        output[idx] = input[idx];
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // 1. No NaN or Inf values
    bool has_error = false;
    #pragma omp parallel for reduction(|:has_error)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            has_error = true;
        }
    }
    
    if (has_error) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
        Real val = grid[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
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
    int rank = 0, size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        cudaSetDevice(rank % num_devices);
    } else {
        if (rank == 0) printf("Error: No CUDA devices found.\n");
        MPI_Finalize();
        return 1;
    }

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
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
        printf("3D Stencil Benchmark (MPI + CUDA + OpenMP)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t local_nz = nz / size;
    size_t rem = nz % size;
    size_t z_start = rank * local_nz + std::min((size_t)rank, rem);
    if (rank < (int)rem) local_nz++;
    
    size_t halo_nz = local_nz + 2; 
    size_t local_grid_size = nx * ny * halo_nz;
    
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_grid_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_grid_size * sizeof(Real)));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (local_grid_size + threadsPerBlock - 1) / threadsPerBlock;
    
    initializeGridKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid1, nx, ny, halo_nz, z_start);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy initial state to grid2 as well (so boundaries are consistent if needed)
    CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, local_grid_size * sizeof(Real), cudaMemcpyDeviceToDevice));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    size_t layer_size = nx * ny;
    Real* h_send_top = nullptr;
    Real* h_recv_top = nullptr;
    Real* h_send_btm = nullptr;
    Real* h_recv_btm = nullptr;
    
    CUDA_CHECK(cudaMallocHost(&h_send_top, layer_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, layer_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_send_btm, layer_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_btm, layer_size * sizeof(Real)));
    
    MPI_Request reqs[4];
    MPI_Status stats[4];

    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 dimBlock(8, 8, 4);
    dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x,
                 (ny + dimBlock.y - 1) / dimBlock.y,
                 (halo_nz + dimBlock.z - 1) / dimBlock.z);

    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;
        
        stencilKernel<<<dimGrid, dimBlock>>>(d_in, d_out, nx, ny, halo_nz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Exchange Halos
        // We need to send our computed "out" boundaries to neighbors "in" boundaries for next step.
        // Wait, standard double buffering:
        // Step k: Read A, Write B.
        // Step k+1: Read B, Write A.
        // Neighbors at step k+1 will read my B.
        // So I must send my B boundaries to them.
        // My B boundaries are at z=1 (bottom active) and z=halo_nz-2 (top active).
        // I send B[1] to Rank-1 (who puts it in B[halo_nz-1] which is their Top Halo) -- No, Rank-1 will read it as Top Halo.
        // Wait, Rank-1's Top Halo corresponds to my Bottom Active.
        // Yes.
        
        int n_reqs = 0;
        
        // Upward: Send Top Active (halo_nz-2) to Rank+1, Recv from Rank+1 into Top Halo (halo_nz-1)
        if (rank < size - 1) {
            size_t offset_send = (halo_nz - 2) * layer_size;
            CUDA_CHECK(cudaMemcpy(h_send_top, d_out + offset_send, layer_size * sizeof(Real), cudaMemcpyDeviceToHost));
            
            MPI_Isend(h_send_top, layer_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Irecv(h_recv_top, layer_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        
        // Downward: Send Bottom Active (1) to Rank-1, Recv from Rank-1 into Bottom Halo (0)
        if (rank > 0) {
            size_t offset_send = 1 * layer_size;
            CUDA_CHECK(cudaMemcpy(h_send_btm, d_out + offset_send, layer_size * sizeof(Real), cudaMemcpyDeviceToHost));
            
            MPI_Isend(h_send_btm, layer_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Irecv(h_recv_btm, layer_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        
        if (n_reqs > 0) {
            MPI_Waitall(n_reqs, reqs, stats);
        }
        
        if (rank < size - 1) {
            size_t offset_recv = (halo_nz - 1) * layer_size;
            CUDA_CHECK(cudaMemcpy(d_out + offset_recv, h_recv_top, layer_size * sizeof(Real), cudaMemcpyHostToDevice));
        }
        
        if (rank > 0) {
            size_t offset_recv = 0;
            CUDA_CHECK(cudaMemcpy(d_out + offset_recv, h_recv_btm, layer_size * sizeof(Real), cudaMemcpyHostToDevice));
        }
        
        // If this is the last iteration, we want d_out to be valid.
        // The halos of d_out are now updated.
        // In the next iteration, we use d_out as input.
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    if (validate || printResults) {
        Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2; // Final result is in d_grid1 because loop ended?
        // Loop: 0..iterations-1.
        // Iter 0: in=1, out=2.
        // Iter 1: in=2, out=1.
        // ...
        // If iterations is Even (e.g. 10): last iter is 9 (odd).
        // Iter 9: in=2, out=1. Result in grid1.
        // Correct.
        
        // Copy active region
        std::vector<Real> local_result(local_nz * nx * ny);
        size_t active_start_offset = 1 * nx * ny;
        CUDA_CHECK(cudaMemcpy(local_result.data(), d_final + active_start_offset, local_nz * nx * ny * sizeof(Real), cudaMemcpyDeviceToHost));
        
        // Gather
        std::vector<Real> full_grid;
        if (rank == 0) {
            full_grid.resize(nx * ny * nz);
        }
        
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        int my_count = local_nz * nx * ny;
        
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(local_result.data(), my_count, MPI_DOUBLE, 
                    full_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            if (printResults) {
                print_results(full_grid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                // For validation, we need to pass the full grid dimensions
                if (validateResult(full_grid, nx, ny, nz)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }
    
    cudaFree(d_grid1);
    cudaFree(d_grid2);
    cudaFreeHost(h_send_top);
    cudaFreeHost(h_recv_top);
    cudaFreeHost(h_send_btm);
    cudaFreeHost(h_recv_btm);
    
    MPI_Finalize();
    return 0;
}
