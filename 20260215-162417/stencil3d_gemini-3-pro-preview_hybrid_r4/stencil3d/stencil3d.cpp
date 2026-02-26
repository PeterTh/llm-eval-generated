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

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// 3D index calculation for flat array
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA Kernel
// nz is the local dimension including ghosts
// z_offset is the global Z index of the first real plane (local z=1)
// global_nz is the total Z dimension of the problem
__global__ void stencilKernel(const Real* input, Real* output, size_t nx, size_t ny, size_t nz, size_t z_offset, size_t global_nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    size_t idx = z * (nx * ny) + y * nx + x;

    bool is_boundary = false;
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        is_boundary = true;
    } else {
        // Check Z boundary
        // Local z=0 (ghost) and z=nz-1 (ghost) are "boundaries" in local sense -> copy
        if (z == 0 || z == nz - 1) {
            is_boundary = true;
        } else {
            // Check global Z boundary
            // global_z corresponding to local z
            // local z=1 -> global z_offset
            long long global_z = (long long)z_offset + ((long long)z - 1);
            if (global_z <= 0 || global_z >= (long long)global_nz - 1) {
                is_boundary = true;
            }
        }
    }

    if (is_boundary) {
        output[idx] = input[idx];
    } else {
        // Interior: 7-point stencil
        Real center = input[idx];
        Real left   = input[idx - 1];
        Real right  = input[idx + 1];
        Real front  = input[idx - nx];
        Real back   = input[idx + nx];
        Real bottom = input[idx - nx * ny];
        Real top    = input[idx + nx * ny];

        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// Host validation function
bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
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

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
            else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
            else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0) validate = true;
            else if (strcmp(argv[i], "-r") == 0) printResults = true;
            else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 0); 
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
        
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int flags[2];
    if (rank == 0) { flags[0] = validate ? 1 : 0; flags[1] = printResults ? 1 : 0; }
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    validate = flags[0]; printResults = flags[1];

    size_t nz_local = nz / size;
    size_t remainder = nz % size;
    size_t z_start_global;
    
    if (rank < (int)remainder) {
        nz_local++;
        z_start_global = rank * nz_local;
    } else {
        z_start_global = rank * nz_local + remainder;
    }

    size_t nz_ghost = nz_local + 2;
    size_t slice_size = nx * ny;
    size_t local_grid_size = slice_size * nz_ghost;

    std::vector<Real> h_grid1(local_grid_size);
    std::vector<Real> h_grid2(local_grid_size);

    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz_ghost; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                long long global_z = (long long)z_start_global + (long long)z - 1;
                size_t local_idx = z * slice_size + y * nx + x;
                
                if (global_z >= 0 && global_z < (long long)nz) {
                    size_t global_idx = (size_t)global_z * slice_size + y * nx + x;
                    h_grid1[local_idx] = (global_idx % 19) * 1.0;
                } else {
                    h_grid1[local_idx] = 0.0;
                }
                h_grid2[local_idx] = 0.0;
            }
        }
    }

    Real *d_grid1, *d_grid2;
    CHECK_CUDA(cudaMalloc(&d_grid1, local_grid_size * sizeof(Real)));
    CHECK_CUDA(cudaMalloc(&d_grid2, local_grid_size * sizeof(Real)));

    CHECK_CUDA(cudaMemcpy(d_grid1, h_grid1.data(), local_grid_size * sizeof(Real), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid2, h_grid2.data(), local_grid_size * sizeof(Real), cudaMemcpyHostToDevice));

    std::vector<Real> send_top(slice_size);
    std::vector<Real> recv_top(slice_size);
    std::vector<Real> send_bot(slice_size);
    std::vector<Real> recv_bot(slice_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, 
                  (ny + blockSize.y - 1) / blockSize.y, 
                  (nz_ghost + blockSize.z - 1) / blockSize.z);

    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* output = (iter % 2 == 0) ? d_grid2 : d_grid1;

        stencilKernel<<<gridSize, blockSize>>>(input, output, nx, ny, nz_ghost, z_start_global, nz);
        CHECK_CUDA(cudaGetLastError());
        
        // Wait for kernel before copying output
        CHECK_CUDA(cudaDeviceSynchronize());

        // We exchange the *OUTPUT* ghosts because output becomes input in next iter.
        // Wait.
        // iter 0: input=d_grid1, output=d_grid2.
        // Kernel reads d_grid1, writes d_grid2.
        // Kernel handles boundaries by copying d_grid1 -> d_grid2.
        // BUT ghost layers in d_grid1 were valid at start.
        // After kernel, d_grid2 has valid interior, but what about its ghosts?
        // My kernel: if z==0 || z==nz-1 (local ghosts), output[idx] = input[idx].
        // So d_grid2 ghosts are copies of d_grid1 ghosts.
        // This is OLD ghost data!
        // We need NEW ghost data from neighbors for the next step.
        // Neighbors computed their INTERIOR boundaries in d_grid2.
        // So we need to:
        // 1. Send my computed interior boundaries (from d_grid2) to neighbor's ghosts.
        // 2. Receive neighbor's computed interior boundaries into my ghosts (in d_grid2).
        
        // Correct.
        // Send buffer: d_grid2 slice z=1 (bottom real) -> to neighbor below.
        // Send buffer: d_grid2 slice z=nz_ghost-2 (top real) -> to neighbor above.
        // Recv buffer: from neighbor below -> into d_grid2 slice z=0.
        // Recv buffer: from neighbor above -> into d_grid2 slice z=nz_ghost-1.
        
        CHECK_CUDA(cudaMemcpy(send_bot.data(), output + 1 * slice_size, slice_size * sizeof(Real), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(send_top.data(), output + (nz_ghost - 2) * slice_size, slice_size * sizeof(Real), cudaMemcpyDeviceToHost));
        
        MPI_Request reqs[4];
        int nreq = 0;

        if (rank > 0) {
            MPI_Isend(send_bot.data(), slice_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(recv_bot.data(), slice_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (rank < size - 1) {
            MPI_Isend(send_top.data(), slice_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(recv_top.data(), slice_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        
        if (rank > 0) {
            CHECK_CUDA(cudaMemcpy(output, recv_bot.data(), slice_size * sizeof(Real), cudaMemcpyHostToDevice));
        }
        if (rank < size - 1) {
            CHECK_CUDA(cudaMemcpy(output + (nz_ghost - 1) * slice_size, recv_top.data(), slice_size * sizeof(Real), cudaMemcpyHostToDevice));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    Real* final_device = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CHECK_CUDA(cudaMemcpy(h_grid1.data(), final_device, local_grid_size * sizeof(Real), cudaMemcpyDeviceToHost));

    std::vector<Real> full_grid;
    std::vector<Real> local_real_data;
    
    if (rank == 0 && (validate || printResults)) {
        full_grid.resize(nx * ny * nz);
    }
    
    // Everyone prepares local real data
    if (validate || printResults) {
        local_real_data.resize(nz_local * slice_size);
        memcpy(local_real_data.data(), h_grid1.data() + slice_size, nz_local * slice_size * sizeof(Real));
        
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        int local_count = nz_local * slice_size;
        
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
        }
        
        MPI_Gatherv(local_real_data.data(), local_count, MPI_DOUBLE,
                    full_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        
        if (printResults) {
             print_results(full_grid, "Grid");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_grid, nx, ny, nz);
            if (valid) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
    }

    CHECK_CUDA(cudaFree(d_grid1));
    CHECK_CUDA(cudaFree(d_grid2));

    MPI_Finalize();
    return 0;
}
