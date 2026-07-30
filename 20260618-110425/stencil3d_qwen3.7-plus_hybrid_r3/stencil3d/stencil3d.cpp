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
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 3D index calculation (host and device compatible)
__host__ __device__ inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Combined CUDA kernel: applies 7-point stencil on interior points and copies boundary values.
// The local grid layout is: [bottom_halo | owned (local_nz planes) | top_halo]
// Owned planes are at local z indices [1, local_nz].
// Halos are at local z index 0 and local_nz+1.
__global__ void stencilKernel(
    const Real* __restrict__ input,
    Real* __restrict__ output,
    const size_t nx, const size_t ny,
    const size_t local_nz,
    const bool is_first_rank, const bool is_last_rank)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + 1; // owned region: [1, local_nz]

    if (x >= nx || y >= ny || lz > local_nz) return;

    const size_t plane = nx * ny;
    const size_t idx = lz * plane + y * nx + x;

    // Determine if this is a boundary point
    bool is_bdy = (x == 0) | (x == nx - 1) | (y == 0) | (y == ny - 1);
    if (is_first_rank && (lz == 1)) is_bdy = true;
    if (is_last_rank && (lz == local_nz)) is_bdy = true;

    if (is_bdy) {
        output[idx] = input[idx];
    } else {
        const Real center = input[idx];
        const Real left   = input[lz * plane + y * nx + (x - 1)];
        const Real right  = input[lz * plane + y * nx + (x + 1)];
        const Real front  = input[lz * plane + (y - 1) * nx + x];
        const Real back   = input[lz * plane + (y + 1) * nx + x];
        const Real bottom = input[(lz - 1) * plane + y * nx + x];
        const Real top    = input[(lz + 1) * plane + y * nx + x];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

void initializeGrid(Real* grid, size_t nx, size_t ny, size_t local_nz, size_t z_start) {
    size_t plane = nx * ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            size_t z_global = z_start + lz - 1;
            size_t base = z_global * plane + y * nx;
            size_t local_base = lz * plane + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                grid[local_base + x] = (Real)((base + x) % 19);
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    // 1. No NaN or Inf values
    bool has_nan_inf = false;
    #pragma omp parallel for reduction(|:has_nan_inf) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) has_nan_inf = true;
    }
    if (has_nan_inf) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0], maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal,max:maxVal) schedule(static)
    for (size_t i = 1; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
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
    // --- MPI initialization ---
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // --- Parse command line arguments (all ranks) ---
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = (size_t)atoi(argv[++i]);
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

    // --- Domain decomposition along Z axis ---
    size_t base_nz = nz / (size_t)num_procs;
    size_t remainder = nz % (size_t)num_procs;
    size_t local_nz, z_start;
    if ((size_t)rank < remainder) {
        local_nz = base_nz + 1;
        z_start = (size_t)rank * (base_nz + 1);
    } else {
        local_nz = base_nz;
        z_start = remainder * (base_nz + 1) + ((size_t)rank - remainder) * base_nz;
    }

    if (local_nz == 0) {
        fprintf(stderr, "Rank %d: local_nz is 0. Need nz >= num_procs.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    bool is_first_rank = (rank == 0);
    bool is_last_rank = (rank == num_procs - 1);

    // --- GPU setup ---
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices found.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % num_gpus));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, rank % num_gpus));

    int omp_threads = omp_get_max_threads();

    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d | OpenMP threads/rank: %d | GPUs: %d (%s)\n",
               num_procs, omp_threads, num_gpus, prop.name);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // --- Local grid dimensions (including 1 halo plane on each Z side) ---
    size_t plane_size = nx * ny;
    size_t local_nz_total = local_nz + 2; // halo + owned + halo
    size_t local_size = plane_size * local_nz_total;

    // --- Allocate device memory (double buffering) ---
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_size * sizeof(Real)));

    // --- Allocate host memory and initialize ---
    if (rank == 0) printf("Initializing grid...\n");

    std::vector<Real> h_grid(local_size, 0.0);
    initializeGrid(h_grid.data(), nx, ny, local_nz, z_start);

    // Zero out halo planes on host before uploading
    memset(h_grid.data(), 0, plane_size * sizeof(Real)); // bottom halo (lz=0)
    memset(h_grid.data() + (local_nz + 1) * plane_size, 0, plane_size * sizeof(Real)); // top halo

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_grid1, h_grid.data(), local_size * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grid2, 0, local_size * sizeof(Real)));

    // Free host grid (no longer needed until result gathering)
    h_grid.clear();
    h_grid.shrink_to_fit();

    // --- Halo exchange buffers (host, pinned for faster transfers) ---
    Real *h_send, *h_recv;
    CUDA_CHECK(cudaMallocHost(&h_send, plane_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv, plane_size * sizeof(Real)));

    int rank_down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_up   = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    // --- CUDA kernel launch configuration ---
    dim3 block(16, 4, 4); // 256 threads
    dim3 grid_dim((unsigned int)((nx + block.x - 1) / block.x),
                  (unsigned int)((ny + block.y - 1) / block.y),
                  (unsigned int)((local_nz + block.z - 1) / block.z));

    // --- Run stencil iterations ---
    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    Real* d_input = d_grid1;
    Real* d_output = d_grid2;

    for (int iter = 0; iter < iterations; ++iter) {
        // --- Halo exchange on d_input ---
        if (num_procs > 1) {
            // Exchange 1: send bottom owned (lz=1) down, receive top halo (lz=local_nz+1) from up
            CUDA_CHECK(cudaMemcpy(h_send, d_input + 1 * plane_size,
                                  plane_size * sizeof(Real), cudaMemcpyDeviceToHost));
            MPI_Sendrecv(h_send, (int)plane_size, MPI_DOUBLE, rank_down, 1,
                         h_recv, (int)plane_size, MPI_DOUBLE, rank_up, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank_up != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(d_input + (local_nz + 1) * plane_size, h_recv,
                                      plane_size * sizeof(Real), cudaMemcpyHostToDevice));
            }

            // Exchange 2: send top owned (lz=local_nz) up, receive bottom halo (lz=0) from down
            CUDA_CHECK(cudaMemcpy(h_send, d_input + local_nz * plane_size,
                                  plane_size * sizeof(Real), cudaMemcpyDeviceToHost));
            MPI_Sendrecv(h_send, (int)plane_size, MPI_DOUBLE, rank_up, 0,
                         h_recv, (int)plane_size, MPI_DOUBLE, rank_down, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank_down != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(d_input + 0 * plane_size, h_recv,
                                      plane_size * sizeof(Real), cudaMemcpyHostToDevice));
            }
        }

        // --- Launch combined stencil + boundary kernel ---
        stencilKernel<<<grid_dim, block>>>(d_input, d_output, nx, ny, local_nz,
                                           is_first_rank, is_last_rank);

        // Swap input/output
        Real* tmp = d_input;
        d_input = d_output;
        d_output = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // --- Timing and performance (rank 0) ---
    if (rank == 0) {
        long duration_ms = (long)(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * (double)iterations;
        double mcups = cellUpdates / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // --- Gather results on rank 0 for printing / validation ---
    // Copy owned region from device to host
    std::vector<Real> local_result(local_nz * plane_size);
    CUDA_CHECK(cudaMemcpy(local_result.data(), d_input + 1 * plane_size,
                           local_nz * plane_size * sizeof(Real), cudaMemcpyDeviceToHost));

    if (printResults || validate) {
        // Gather local_nz from all ranks on rank 0
        int my_local_nz = (int)local_nz;
        std::vector<int> all_local_nz(num_procs);
        MPI_Gather(&my_local_nz, 1, MPI_INT, all_local_nz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::vector<int> recvcounts(num_procs);
            std::vector<int> displs(num_procs);
            displs[0] = 0;
            for (int i = 0; i < num_procs; ++i) {
                recvcounts[i] = all_local_nz[i] * (int)plane_size;
                if (i > 0) displs[i] = displs[i - 1] + recvcounts[i - 1];
            }

            std::vector<Real> global_grid(nx * ny * nz);
            MPI_Gatherv(local_result.data(), (int)(local_nz * plane_size), MPI_DOUBLE,
                        global_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);

            if (printResults) {
                print_results(global_grid, "Grid");
            }

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
        } else {
            MPI_Gatherv(local_result.data(), (int)(local_nz * plane_size), MPI_DOUBLE,
                        nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // --- Cleanup ---
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    MPI_Finalize();
    return 0;
}
