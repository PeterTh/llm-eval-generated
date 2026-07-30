#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 7-point stencil CUDA kernel
// Local array layout: (local_nz+2) planes of (nx*ny), z-major order
//   Plane 0            = bottom ghost/halo
//   Planes [1,local_nz]= owned data
//   Plane (local_nz+1) = top ghost/halo
__global__ __launch_bounds__(512)
void stencil_kernel(const double* __restrict__ input,
                    double* __restrict__ output,
                    const size_t nx, const size_t ny,
                    const size_t local_nz,
                    const size_t z_start_global,
                    const size_t nz_global) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z_local = blockIdx.z * blockDim.z + threadIdx.z + 1; // +1 for bottom ghost

    if (x >= nx || y >= ny || z_local > local_nz) return;

    const size_t plane_size = nx * ny;
    const size_t idx = z_local * plane_size + y * nx + x;
    const size_t z_global = z_start_global + (z_local - 1);

    // Boundary points: copy from input
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
        z_global == 0 || z_global == nz_global - 1) {
        output[idx] = input[idx];
        return;
    }

    // Interior point: 7-point averaging stencil
    const double center = input[idx];
    const double left   = input[idx - 1];
    const double right  = input[idx + 1];
    const double front  = input[idx - nx];
    const double back   = input[idx + nx];
    const double bottom = input[idx - plane_size];
    const double top    = input[idx + plane_size];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

// Halo exchange: update ghost planes via MPI
// Uses pinned host buffers for D2H/H2D transfers
void halo_exchange(double* d_grid, size_t nx, size_t ny, size_t local_nz_padded,
                   double* h_send, double* h_recv,
                   int rank, int nprocs) {
    const size_t plane_size = nx * ny;
    const size_t local_nz = local_nz_padded - 2;

    const int below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int above = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
    const int count = (int)plane_size;

    // Exchange 1: send top owned plane upward, receive from below into bottom ghost
    CUDA_CHECK(cudaMemcpy(h_send, d_grid + local_nz * plane_size,
                          plane_size * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Sendrecv(h_send, count, MPI_DOUBLE, above, 0,
                 h_recv, count, MPI_DOUBLE, below, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (below != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_grid, h_recv,
                              plane_size * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Exchange 2: send bottom owned plane downward, receive from above into top ghost
    CUDA_CHECK(cudaMemcpy(h_send, d_grid + plane_size,
                          plane_size * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Sendrecv(h_send, count, MPI_DOUBLE, below, 1,
                 h_recv, count, MPI_DOUBLE, above, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (above != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_grid + (local_nz_padded - 1) * plane_size, h_recv,
                              plane_size * sizeof(double), cudaMemcpyHostToDevice));
    }
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
    printf("\nParallelization: MPI + OpenMP + CUDA (hybrid)\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // GPU selection: assign one GPU per MPI rank (round-robin)
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % num_gpus));

    // Parse command line arguments (all ranks parse identically)
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

    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d, OpenMP threads: %d, GPUs per node: %d\n",
               nprocs, omp_get_max_threads(), num_gpus);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain decomposition along Z axis (slab decomposition)
    if (nz < (size_t)nprocs) {
        if (rank == 0) {
            printf("Error: Z dimension (%zu) must be >= number of MPI ranks (%d)\n", nz, nprocs);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t base_nz = nz / nprocs;
    const size_t remainder = nz % nprocs;
    const size_t local_nz = base_nz + ((size_t)rank < remainder ? 1 : 0);
    const size_t z_start = (size_t)rank * base_nz + std::min((size_t)rank, remainder);

    const size_t local_nz_padded = local_nz + 2; // ghost planes at both ends
    const size_t plane_size = nx * ny;
    const size_t local_size = local_nz_padded * plane_size;

    if (rank == 0) {
        printf("Domain decomposition: Z split across %d ranks (~%zu planes/rank)\n",
               nprocs, local_nz);
        printf("Initializing grid...\n");
    }

    // Allocate GPU memory (double buffering)
    double *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_size * sizeof(double)));

    // Initialize grid on host with OpenMP parallelism, then copy to GPU
    std::vector<double> host_owned(local_nz * plane_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t p = 0; p < plane_size; ++p) {
            const size_t z_global = z_start + z;
            const size_t global_idx = z_global * plane_size + p;
            host_owned[z * plane_size + p] = (Real)(global_idx % 19) * 1.0;
        }
    }

    // Copy owned planes to GPU at offset plane_size (skip bottom ghost plane)
    CUDA_CHECK(cudaMemcpy(d_grid1 + plane_size, host_owned.data(),
                          local_nz * plane_size * sizeof(double), cudaMemcpyHostToDevice));
    host_owned.clear();
    host_owned.shrink_to_fit();

    // Allocate pinned host buffers for halo exchange
    double *h_send, *h_recv;
    CUDA_CHECK(cudaMallocHost(&h_send, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv, plane_size * sizeof(double)));

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // CUDA kernel launch configuration
    dim3 block(32, 4, 4); // 512 threads per block
    dim3 grid_dim((unsigned int)((nx + block.x - 1) / block.x),
                  (unsigned int)((ny + block.y - 1) / block.y),
                  (unsigned int)((local_nz + block.z - 1) / block.z));

    for (int iter = 0; iter < iterations; ++iter) {
        double* d_input  = (iter % 2 == 0) ? d_grid1 : d_grid2;
        double* d_output = (iter % 2 == 0) ? d_grid2 : d_grid1;

        // Halo exchange on input buffer (updates ghost planes)
        halo_exchange(d_input, nx, ny, local_nz_padded, h_send, h_recv, rank, nprocs);

        // Stencil computation on GPU
        stencil_kernel<<<grid_dim, block>>>(d_input, d_output, nx, ny, local_nz,
                                            z_start, nz);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start);
    long duration_ms = duration.count();

    // Copy result back to host (owned planes only, skip ghost plane 0)
    double* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<double> local_result(local_nz * plane_size);
    CUDA_CHECK(cudaMemcpy(local_result.data(), d_final + plane_size,
                          local_nz * plane_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather all results on rank 0 using MPI_Gatherv
    std::vector<int> recv_counts(nprocs), displacements(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t r_local_nz = base_nz + ((size_t)r < remainder ? 1 : 0);
        size_t r_z_start = (size_t)r * base_nz + std::min((size_t)r, remainder);
        recv_counts[r] = (int)(r_local_nz * plane_size);
        displacements[r] = (int)(r_z_start * plane_size);
    }

    std::vector<double> full_grid;
    if (rank == 0) {
        full_grid.resize(nx * ny * nz);
    }

    MPI_Gatherv(local_result.data(), (int)(local_nz * plane_size), MPI_DOUBLE,
                full_grid.data(), recv_counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Output results on rank 0
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = (duration_ms > 0) ? cellUpdates / (duration_ms / 1000.0) / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(full_grid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");

            bool valid = true;
            Real minVal = full_grid[0], maxVal = full_grid[0];

            #pragma omp parallel
            {
                bool local_valid = true;
                Real local_min = full_grid[0], local_max = full_grid[0];

                #pragma omp for schedule(static)
                for (size_t i = 0; i < full_grid.size(); ++i) {
                    if (std::isnan(full_grid[i]) || std::isinf(full_grid[i])) {
                        local_valid = false;
                    }
                    if (full_grid[i] < local_min) local_min = full_grid[i];
                    if (full_grid[i] > local_max) local_max = full_grid[i];
                }

                #pragma omp critical
                {
                    if (!local_valid) valid = false;
                    if (local_min < minVal) minVal = local_min;
                    if (local_max > maxVal) maxVal = local_max;
                }
            }

            printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

            if (maxVal > 1e6 || minVal < -1e6) {
                printf("Validation failed: values out of expected range\n");
                valid = false;
            }

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));

    MPI_Finalize();
    return 0;
}
