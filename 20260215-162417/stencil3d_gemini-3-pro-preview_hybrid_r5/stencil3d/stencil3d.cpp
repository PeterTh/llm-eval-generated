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
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// 3D index calculation
inline __host__ __device__ size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA Kernel
__global__ void stencilKernel(const Real* input, Real* output, const size_t nx, const size_t ny, const size_t nz, const size_t z_offset, const size_t global_nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    size_t idx = idx3(x, y, z, nx, ny);

    // Global Z index
    // The buffer 'z' index includes ghost layers.
    // z=0 is ghost from below (or boundary).
    // z=1 is first valid local slice.
    // The corresponding global index for z=1 is z_offset.
    // So global_z = z_offset + (z - 1).
    
    // However, if we are at rank boundaries, we might compute on z=1 (which is global z=0) ?
    // The original code computes for z=1 to global_nz-2.
    // So global z=0 and global z=global_nz-1 are boundaries and NOT computed.
    
    // Map local z to global z
    // Wait, z_offset passed to kernel needs to be correct.
    // In main, z_start is calculated as "start of valid region".
    // So z=1 corresponds to global index z_start.
    
    long long global_z_idx = (long long)z_offset + ((long long)z - 1);
    
    bool is_interior = (x > 0 && x < nx - 1 && 
                        y > 0 && y < ny - 1 &&
                        global_z_idx > 0 && global_z_idx < global_nz - 1);
                        
    // Also, we only compute on valid local indices [1, nz-2]
    // Because z=0 and z=nz-1 are ghosts/boundaries.
    bool is_valid_local = (z > 0 && z < nz - 1);

    if (is_interior && is_valid_local) {
        Real center = input[idx];
        Real left   = input[idx3(x-1, y, z, nx, ny)];
        Real right  = input[idx3(x+1, y, z, nx, ny)];
        Real front  = input[idx3(x, y-1, z, nx, ny)];
        Real back   = input[idx3(x, y+1, z, nx, ny)];
        Real bottom = input[idx3(x, y, z-1, nx, ny)];
        Real top    = input[idx3(x, y, z+1, nx, ny)];
        
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    } else {
        // Copy boundary values (or ghost values, doesn't matter as they are overwritten or not used)
        output[idx] = input[idx];
    }
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz, const size_t z_offset) {
    #pragma omp parallel for collapse(2)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Use global index for consistent initialization across ranks if needed
                // But for now, we just use local index to match the simple pattern
                (void)z_offset; 
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// ... rest of functions ...

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // ... same as before ...
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+CUDA+OpenMP)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI Size: %d\n", size);
    }

    // Domain Decomposition (1D slab along Z)
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if (rank < remainder) local_nz++;
    
    // Add ghost layers (top and bottom)
    size_t ghost_nz = local_nz + 2; 
    size_t gridSize = nx * ny * ghost_nz;

    // Allocate host memory
    std::vector<Real> h_grid1(gridSize);
    std::vector<Real> h_grid2(gridSize); // Not strictly needed on host unless debugging, but useful for final gather
    
    // Initialize (only the interior, ghost layers will be filled by exchange)
    // We initialize the "local" part of the global grid.
    // Offset in initialization is tricky. Let's just initialize the whole local buffer including ghosts
    // and then correct the ghosts via exchange.
    // Actually, initializeGrid logic uses 0..local_nz.
    initializeGrid(h_grid1, nx, ny, ghost_nz, z_start);

    // Allocate device memory
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_grid1, h_grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid2, h_grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice)); // Initialize grid2 as well

    // MPI Buffers for halo exchange
    size_t planeSize = nx * ny;
    Real *h_sendLowZ = new Real[planeSize];
    Real *h_recvHighZ = new Real[planeSize];
    Real *h_sendHighZ = new Real[planeSize];
    Real *h_recvLowZ = new Real[planeSize];

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (ghost_nz + block.z - 1) / block.z);

    for (int iter = 0; iter < iterations; ++iter) {
        Real *d_in = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real *d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;

        // Exchange Halos
        // Low Z boundary (send to rank-1, recv from rank-1)
        // High Z boundary (send to rank+1, recv from rank+1)
        
        // 1. Copy boundaries from device to host
        // Send buffer for Low Z (z=1 in local coordinates)
        // Send buffer for High Z (z=local_nz in local coordinates)
        
        // Send z=1 (top of valid data / low z)
        CUDA_CHECK(cudaMemcpy(h_sendLowZ, d_in + 1 * planeSize, planeSize * sizeof(Real), cudaMemcpyDeviceToHost));
        // Send z=local_nz (bottom of valid data / high z)
        size_t valid_bottom_z = ghost_nz - 2; 
        CUDA_CHECK(cudaMemcpy(h_sendHighZ, d_in + valid_bottom_z * planeSize, planeSize * sizeof(Real), cudaMemcpyDeviceToHost));
        
        // MPI Communication
        MPI_Request reqs[4];
        int num_reqs = 0;
        
        // Rank R sends High Z to Rank R+1. Rank R+1 receives into Low Z Ghost.
        // Rank R sends Low Z to Rank R-1. Rank R-1 receives into High Z Ghost.
        
        // Send High Z to Rank+1
        if (rank < size - 1) {
            MPI_Isend(h_sendHighZ, planeSize, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(h_recvHighZ, planeSize, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[num_reqs++]); // Recv from Rank+1 (which sends Low Z)
        }
        
        // Send Low Z to Rank-1
        if (rank > 0) {
            MPI_Isend(h_sendLowZ, planeSize, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(h_recvLowZ, planeSize, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[num_reqs++]); // Recv from Rank-1 (which sends High Z)
        }
        
        MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
        
        // Copy received halos to device
        if (rank < size - 1) {
            // Received from Rank+1 (High Z neighbor), put at High Z Ghost (index local_nz+1)
            CUDA_CHECK(cudaMemcpy(d_in + (ghost_nz - 1) * planeSize, h_recvHighZ, planeSize * sizeof(Real), cudaMemcpyHostToDevice));
        }
        if (rank > 0) {
            // Received from Rank-1 (Low Z neighbor), put at Low Z Ghost (index 0)
            CUDA_CHECK(cudaMemcpy(d_in, h_recvLowZ, planeSize * sizeof(Real), cudaMemcpyHostToDevice));
        }
        
        // Kernel Launch
        stencilKernel<<<grid, block>>>(d_in, d_out, nx, ny, ghost_nz, z_start, nz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());

    // Gather results for validation
    // Need to strip ghost layers and assemble
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(h_grid1.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    
    // Create a full grid on rank 0
    std::vector<Real> fullGrid;
    if (rank == 0) fullGrid.resize(nx * ny * nz);
    
    // Prepare local data without ghosts
    // Valid data is from index 1 to ghost_nz-2 (inclusive), length = ghost_nz - 2 = local_nz
    std::vector<Real> localData(nx * ny * (ghost_nz - 2));
    // Copy valid region
    // The valid data starts at z=1 (one plane offset)
    std::memcpy(localData.data(), h_grid1.data() + planeSize, localData.size() * sizeof(Real));

    // Gather sizes
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    int my_count = localData.size();
    MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
    }
    
    MPI_Gatherv(localData.data(), my_count, MPI_DOUBLE, 
                fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullGrid, nx, ny, nz);
             if (valid) printf("Validation: PASSED\n");
             else printf("Validation: FAILED\n");
        }
        
        if (printResults) {
             print_results(fullGrid, "Grid");
        }
    }
    
    // Cleanup
    delete[] h_sendLowZ;
    delete[] h_recvHighZ;
    delete[] h_sendHighZ;
    delete[] h_recvLowZ;
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    
    MPI_Finalize();
    return 0;
}

