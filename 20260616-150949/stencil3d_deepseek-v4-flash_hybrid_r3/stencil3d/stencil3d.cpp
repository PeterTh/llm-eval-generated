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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                              const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(err));                                     \
        MPI_Abort(MPI_COMM_WORLD, 1);                                         \
    }                                                                         \
} while(0)

// ---------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------

// 7-point stencil: computes interior cells (x in [1, nx-2], y in [1, ny-2])
// z_range = [1 + z_offset, nz_local] (z_offset=1 for rank 0 to skip global Z=0 boundary)
__global__ void stencil_kernel(const Real* __restrict__ input, Real* __restrict__ output,
                                size_t nx, size_t ny, size_t nx_ny, size_t nz_local,
                                int z_offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z + 1 + z_offset;

    if (x >= nx - 1 || y >= ny - 1 || z > nz_local) return;

    size_t idx = z * nx_ny + y * nx + x;

    Real val = input[idx]
             + input[idx - 1]       // left
             + input[idx + 1]       // right
             + input[idx - nx]      // front
             + input[idx + nx]      // back
             + input[idx - nx_ny]   // bottom
             + input[idx + nx_ny];  // top

    output[idx] = val / Real(7.0);
}

// Copy boundary values from input to output for all faces
__global__ void boundary_kernel(Real* output, const Real* input,
                                 size_t nx, size_t ny, size_t nx_ny,
                                 size_t nz_local,
                                 bool bottom_boundary, bool top_boundary) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z + 1;

    if (x >= nx || y >= ny || z > nz_local) return;

    bool on_boundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);
    if (bottom_boundary && z == 1)          on_boundary = true;
    if (top_boundary    && z == nz_local)   on_boundary = true;

    if (on_boundary) {
        size_t idx = z * nx_ny + y * nx + x;
        output[idx] = input[idx];
    }
}

// Initialize local grid (including ghost layers) with (global_idx % 19) * 1.0
__global__ void init_kernel(Real* grid, size_t nx, size_t ny, size_t nx_ny,
                             size_t nz_local, size_t z_start_global) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= nx || y >= ny || z >= nz_local + 2) return;

    size_t global_z;
    if (z == 0) {
        global_z = (z_start_global > 0) ? z_start_global - 1 : 0;
    } else if (z == nz_local + 1) {
        global_z = z_start_global + nz_local;
    } else {
        global_z = z_start_global + z - 1;
    }

    size_t local_idx  = z * nx_ny + y * nx + x;
    size_t global_idx = global_z * nx_ny + y * nx + x;
    grid[local_idx] = Real(global_idx % 19);
}

// ---------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------

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

bool validateResult(const std::vector<Real>& grid, size_t /*nx*/, size_t /*ny*/, size_t /*nz*/) {
    // 1. No NaN or Inf values
    bool has_nan = false;
    #pragma omp parallel for reduction(||:has_nan)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            has_nan = true;
        }
    }
    if (has_nan) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse command line on rank 0
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) { nx = atoi(argv[++i]); }
            else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) { ny = atoi(argv[++i]); }
            else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) { nz = atoi(argv[++i]); }
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) { iterations = atoi(argv[++i]); }
            else if (strcmp(argv[i], "-v") == 0) { validate = true; }
            else if (strcmp(argv[i], "-r") == 0) { printResults = true; }
            else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]); MPI_Finalize(); return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]); MPI_Finalize(); return 1;
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    // Broadcast parameters
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // GPU device selection (round-robin)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int gpu_id = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    // Domain decomposition in Z
    size_t base = nz / num_procs;
    size_t rem  = nz % num_procs;
    size_t r = static_cast<size_t>(rank);
    size_t z_start = r * base + (r < rem ? r : rem);
    size_t z_count = base + (r < rem ? 1 : 0);
    size_t nz_local = z_count;

    size_t nx_ny = nx * ny;
    size_t local_alloc = nx * ny * (nz_local + 2);   // +2 ghost layers

    // Determine which global Z boundaries this rank owns
    bool bottom_boundary = (rank == 0);
    bool top_boundary    = (rank == num_procs - 1);

    // Stencil skip: global Z=0 / Z=nz-1 boundaries are copied, not averaged
    int z_skip_bottom = bottom_boundary ? 1 : 0;
    int z_skip_top    = top_boundary    ? 1 : 0;
    size_t z_layers = (nz_local > static_cast<size_t>(z_skip_bottom + z_skip_top))
                      ? nz_local - static_cast<size_t>(z_skip_bottom + z_skip_top) : 0;

    // Print configuration from rank 0
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", num_procs);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d (rank %d using GPU %d)\n", num_gpus, rank, gpu_id);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ------------------------------------------------------------------
    // Allocate device memory
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_alloc * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_alloc * sizeof(Real)));

    // Host pinned buffers for halo exchange
    Real *h_send_bottom, *h_send_top, *h_recv_bottom, *h_recv_top;
    CUDA_CHECK(cudaMallocHost(&h_send_bottom, nx_ny * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_send_top,    nx_ny * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom, nx_ny * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top,    nx_ny * sizeof(Real)));

    // ------------------------------------------------------------------
    // Initialize grid on device
    dim3 init_block(16, 16, 1);
    dim3 init_grid((nx + 15) / 16, (ny + 15) / 16, nz_local + 2);
    init_kernel<<<init_grid, init_block>>>(d_grid1, nx, ny, nx_ny, nz_local, z_start);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, local_alloc * sizeof(Real),
                          cudaMemcpyDeviceToDevice));

    // ------------------------------------------------------------------
    // CUDA launch configuration
    dim3 stencil_block(32, 4, 1);
    unsigned sz = static_cast<unsigned>(std::max(z_layers, static_cast<size_t>(1)));
    dim3 stencil_grid((nx - 2 + 31) / 32, (ny - 2 + 3) / 4, sz);

    dim3 boundary_block(32, 8, 1);
    dim3 boundary_grid((nx + 31) / 32, (ny + 7) / 8, nz_local);

    // ------------------------------------------------------------------
    // Timed iteration loop
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real *d_input  = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real *d_output = (iter % 2 == 0) ? d_grid2 : d_grid1;

        // ---- Halo exchange (ghost layer fill) ----
        if (num_procs > 1) {
            // Copy boundary planes from device → host (pinned)
            // bottom send plane = local_z=1, top send plane = local_z=nz_local
            #pragma omp parallel num_threads(2)
            {
                #pragma omp single nowait
                {
                    CUDA_CHECK(cudaMemcpy(h_send_bottom,
                                          d_input + nx_ny,
                                          nx_ny * sizeof(Real),
                                          cudaMemcpyDeviceToHost));
                }
                #pragma omp single nowait
                {
                    CUDA_CHECK(cudaMemcpy(h_send_top,
                                          d_input + nz_local * nx_ny,
                                          nx_ny * sizeof(Real),
                                          cudaMemcpyDeviceToHost));
                }
            }

            // Bidirectional exchange with neighbours
            if (rank > 0) {
                MPI_Sendrecv(h_send_bottom, nx_ny, MPI_DOUBLE, rank - 1, 0,
                             h_recv_bottom, nx_ny, MPI_DOUBLE, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank < num_procs - 1) {
                MPI_Sendrecv(h_send_top, nx_ny, MPI_DOUBLE, rank + 1, 0,
                             h_recv_top, nx_ny, MPI_DOUBLE, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // Copy received data to device ghost layers
            #pragma omp parallel num_threads(2)
            {
                #pragma omp single nowait
                {
                    if (rank > 0) {
                        CUDA_CHECK(cudaMemcpy(d_input,
                                              h_recv_bottom,
                                              nx_ny * sizeof(Real),
                                              cudaMemcpyHostToDevice));
                    }
                }
                #pragma omp single nowait
                {
                    if (rank < num_procs - 1) {
                        CUDA_CHECK(cudaMemcpy(d_input + (nz_local + 1) * nx_ny,
                                              h_recv_top,
                                              nx_ny * sizeof(Real),
                                              cudaMemcpyHostToDevice));
                    }
                }
            }
        }

        // ---- Stencil kernel ----
        if (z_layers > 0 && z_layers <= nz_local) {
            stencil_kernel<<<stencil_grid, stencil_block>>>(
                d_input, d_output, nx, ny, nx_ny, nz_local, z_skip_bottom);
            CUDA_CHECK(cudaGetLastError());
        }

        // ---- Boundary copy ----
        boundary_kernel<<<boundary_grid, boundary_block>>>(
            d_output, d_input, nx, ny, nx_ny, nz_local,
            bottom_boundary, top_boundary);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Determine the final buffer
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;

    // ------------------------------------------------------------------
    // Gather results to rank 0 for reporting / validation
    // Compute per-rank counts and displacements
    std::vector<int> recvcounts(num_procs), displs(num_procs);
    for (int p = 0; p < num_procs; ++p) {
        size_t pr = static_cast<size_t>(p);
        size_t r_base = nz / num_procs;
        size_t r_rem  = nz % num_procs;
        size_t r_cnt  = r_base + (pr < r_rem ? 1 : 0);
        recvcounts[p] = static_cast<int>(nx * ny * r_cnt);
        displs[p]     = static_cast<int>(nx * ny * (pr * base + (pr < rem ? pr : rem)));
    }

    // Copy local portion (actual data, not ghost) to host
    std::vector<Real> local_host(nx * ny * nz_local);
    CUDA_CHECK(cudaMemcpy(local_host.data(), d_final + nx_ny,
                          nx * ny * nz_local * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    std::vector<Real> full_grid;
    if (rank == 0) full_grid.resize(nx * ny * nz);

    MPI_Gatherv(local_host.data(), static_cast<int>(nx * ny * nz_local), MPI_DOUBLE,
                rank == 0 ? full_grid.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Output from rank 0
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);

        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1.0e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(full_grid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_grid, nx, ny, nz);
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
    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_top));

    MPI_Finalize();
    return 0;
}
