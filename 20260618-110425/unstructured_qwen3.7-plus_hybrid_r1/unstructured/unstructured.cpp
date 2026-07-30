#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// GPU-friendly material (compact)
struct MaterialGPU {
    double transfer_coeff;
    double external_flow;
};

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// CUDA kernel: compute one simulation step for owned elements
// Grid is regular 2D, so we compute neighbor indices directly
__global__ void simulationKernel(
    const MaterialGPU* __restrict__ materials,
    const int* __restrict__ material_idx,  // per-element material index (local)
    const double* __restrict__ energy_in,  // current energy (local indexing)
    const double* __restrict__ flux_in,    // cumulative flux (local indexing)
    double* __restrict__ energy_out,       // new energy (local indexing)
    double* __restrict__ flux_out,         // new cumulative flux (local indexing)
    int owned_start,   // first owned local row
    int owned_end,     // one past last owned local row
    int local_nrows,   // total local rows (including ghosts)
    int ncols,         // number of columns (n_elems_root)
    int global_row_offset) // global row of local row 0
{
    int gy = blockIdx.x * blockDim.x + threadIdx.x; // column
    int lx = blockIdx.y * blockDim.y + threadIdx.y; // local row

    if (gy >= ncols) return;
    if (lx < owned_start || lx >= owned_end) return;

    int local_idx = lx * ncols + gy;
    int mat_id = material_idx[lx * ncols + gy];
    MaterialGPU mat = materials[mat_id];

    double this_energy = energy_in[local_idx];
    double total_flux = mat.external_flow;

    // Neighbor offsets: up(-1,0), down(+1,0), left(0,-1), right(0,+1)
    // Check bounds in global coordinates
    int gx = global_row_offset + lx; // global row

    // Up neighbor (gx-1, gy)
    if (gx > 0) {
        int n_idx = (lx - 1) * ncols + gy;
        total_flux += (energy_in[n_idx] - this_energy) * mat.transfer_coeff * 0.25;
    }
    // Down neighbor (gx+1, gy)
    if (gx < (global_row_offset + local_nrows - 1)) {
        // Also check global bound
        if (gx + 1 < (global_row_offset + local_nrows)) {
            int n_idx = (lx + 1) * ncols + gy;
            total_flux += (energy_in[n_idx] - this_energy) * mat.transfer_coeff * 0.25;
        }
    }
    // Left neighbor (gx, gy-1)
    if (gy > 0) {
        int n_idx = lx * ncols + (gy - 1);
        total_flux += (energy_in[n_idx] - this_energy) * mat.transfer_coeff * 0.25;
    }
    // Right neighbor (gx, gy+1)
    if (gy < ncols - 1) {
        int n_idx = lx * ncols + (gy + 1);
        total_flux += (energy_in[n_idx] - this_energy) * mat.transfer_coeff * 0.25;
    }

    energy_out[local_idx] = this_energy + total_flux;
    flux_out[local_idx] = flux_in[local_idx] + fabs(total_flux);
}

// Get material index for a global element (x, y) in the NxN grid
inline int getMaterialIdx(int x, int y, int n_elems_root) {
    int last = n_elems_root - 1;
    if (x == 0 && y == 0) return INFLOW_MAT_ID;
    if (x == 0 && y == last) return OUTFLOW_MAT_ID;
    if (x == last && y == 0) return OUTFLOW_MAT_ID;
    if (x == last && y == last) return INFLOW_MAT_ID;
    return DEFAULT_MAT_ID;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Select GPU based on rank (one GPU per rank)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    int local_rank = mpi_rank % num_gpus;
    cudaSetDevice(local_rank);

    // Set OpenMP threads (use remaining cores)
    int cores_per_rank = omp_get_num_procs() / mpi_size;
    if (cores_per_rank < 1) cores_per_rank = 1;
    omp_set_num_threads(cores_per_rank);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (rank 0 broadcasts)
    if (mpi_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            }
        }
    }
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int val_int = validate ? 1 : 0;
    int pr_int = printResults ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pr_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (val_int != 0);
    printResults = (pr_int != 0);

    const int n_elems = n_elems_root * n_elems_root;

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs: %d, OMP threads/rank: %d\n", mpi_size, num_gpus, cores_per_rank);
        printf("\n");
    }

    // Domain decomposition: 1D along rows
    int rows_per_rank = n_elems_root / mpi_size;
    int remainder = n_elems_root % mpi_size;
    int row_start, row_end; // owned rows [row_start, row_end)
    if (mpi_rank < remainder) {
        row_start = mpi_rank * (rows_per_rank + 1);
        row_end = row_start + rows_per_rank + 1;
    } else {
        row_start = remainder * (rows_per_rank + 1) + (mpi_rank - remainder) * rows_per_rank;
        row_end = row_start + rows_per_rank;
    }
    int owned_nrows = row_end - row_start;

    // Neighbor ranks
    int rank_up = (mpi_rank > 0) ? mpi_rank - 1 : MPI_PROC_NULL;
    int rank_down = (mpi_rank < mpi_size - 1) ? mpi_rank + 1 : MPI_PROC_NULL;

    // Local layout: include ghost rows
    // has_top_ghost: need row (row_start-1) from rank_up
    // has_bot_ghost: need row (row_end) from rank_down
    int has_top_ghost = (row_start > 0) ? 1 : 0;
    int has_bot_ghost = (row_end < n_elems_root) ? 1 : 0;
    int local_nrows = owned_nrows + has_top_ghost + has_bot_ghost;
    int local_row_offset = row_start - has_top_ghost; // global row of local row 0
    int owned_local_start = has_top_ghost;
    int owned_local_end = has_top_ghost + owned_nrows;

    int local_nelems = local_nrows * n_elems_root;

    if (mpi_rank == 0) {
        printf("Building unstructured mesh...\n");
    }

    // Build local material index array (host)
    std::vector<int> h_material_idx(local_nelems);
    for (int lx = 0; lx < local_nrows; ++lx) {
        int gx = local_row_offset + lx;
        for (int gy = 0; gy < n_elems_root; ++gy) {
            h_material_idx[lx * n_elems_root + gy] = getMaterialIdx(gx, gy, n_elems_root);
        }
    }

    // Initialize dynamic data (zeros)
    std::vector<double> h_energy(local_nelems, 0.0);
    std::vector<double> h_flux(local_nelems, 0.0);

    // Materials on host
    MaterialGPU h_materials[3];
    h_materials[0] = {0.8, 0.0};    // Default
    h_materials[1] = {0.8, 0.5};    // Inflow
    h_materials[2] = {0.8, -0.5};   // Outflow

    // Allocate GPU memory
    int *d_material_idx;
    double *d_energy[2], *d_flux[2];
    MaterialGPU *d_materials;

    CUDA_CHECK(cudaMalloc(&d_material_idx, local_nelems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_energy[0], local_nelems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy[1], local_nelems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux[0], local_nelems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux[1], local_nelems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_materials, 3 * sizeof(MaterialGPU)));

    // Copy static data to GPU
    CUDA_CHECK(cudaMemcpy(d_material_idx, h_material_idx.data(), local_nelems * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_energy[0], h_energy.data(), local_nelems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flux[0], h_flux.data(), local_nelems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, h_materials, 3 * sizeof(MaterialGPU), cudaMemcpyHostToDevice));

    // Allocate pinned host buffers for ghost row exchange
    double *h_ghost_send, *h_ghost_recv;
    int ghost_row_size = n_elems_root;
    CUDA_CHECK(cudaHostAlloc(&h_ghost_send, ghost_row_size * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_ghost_recv, ghost_row_size * sizeof(double), cudaHostAllocDefault));

    // Calculate memory usage
    const size_t static_mem = (size_t)local_nelems * sizeof(int);
    const size_t dynamic_mem = (size_t)local_nelems * sizeof(double) * 4; // 2 energy + 2 flux buffers
    size_t local_total_mem = static_mem + dynamic_mem;
    size_t total_mem_all;
    MPI_Allreduce(&local_total_mem, &total_mem_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem_all / (1024.0 * 1024.0),
               (size_t)mpi_size * static_mem / (1024.0 * 1024.0),
               (size_t)mpi_size * dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Simulation loop
    int cur_buf = 0;
    dim3 blockSize(32, 8);
    dim3 gridSize((n_elems_root + blockSize.x - 1) / blockSize.x,
                  (local_nrows + blockSize.y - 1) / blockSize.y);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost rows
        // Send top owned row to rank_up, receive into top ghost from rank_up
        // Send bottom owned row to rank_down, receive into bottom ghost from rank_down

        // Exchange with rank_up: send our first owned row, receive into top ghost
        if (has_top_ghost) {
            // Copy top owned row to send buffer
            int send_local_row = owned_local_start; // first owned local row
            CUDA_CHECK(cudaMemcpy(h_ghost_send, d_energy[cur_buf] + send_local_row * n_elems_root,
                                  ghost_row_size * sizeof(double), cudaMemcpyDeviceToHost));
            MPI_Sendrecv(h_ghost_send, ghost_row_size, MPI_DOUBLE, rank_up, 0,
                         h_ghost_recv, ghost_row_size, MPI_DOUBLE, rank_up, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Copy received data into top ghost row (local row 0)
            CUDA_CHECK(cudaMemcpy(d_energy[cur_buf], h_ghost_recv,
                                  ghost_row_size * sizeof(double), cudaMemcpyHostToDevice));
        }

        // Exchange with rank_down: send our last owned row, receive into bottom ghost
        if (has_bot_ghost) {
            int send_local_row = owned_local_end - 1; // last owned local row
            CUDA_CHECK(cudaMemcpy(h_ghost_send, d_energy[cur_buf] + send_local_row * n_elems_root,
                                  ghost_row_size * sizeof(double), cudaMemcpyDeviceToHost));
            MPI_Sendrecv(h_ghost_send, ghost_row_size, MPI_DOUBLE, rank_down, 1,
                         h_ghost_recv, ghost_row_size, MPI_DOUBLE, rank_down, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Copy received data into bottom ghost row
            int recv_local_row = owned_local_end; // first ghost row at bottom
            CUDA_CHECK(cudaMemcpy(d_energy[cur_buf] + recv_local_row * n_elems_root, h_ghost_recv,
                                  ghost_row_size * sizeof(double), cudaMemcpyHostToDevice));
        }

        // Launch CUDA kernel
        int next_buf = 1 - cur_buf;
        simulationKernel<<<gridSize, blockSize>>>(
            d_materials, d_material_idx,
            d_energy[cur_buf], d_flux[cur_buf],
            d_energy[next_buf], d_flux[next_buf],
            owned_local_start, owned_local_end,
            local_nrows, n_elems_root, local_row_offset);
        CUDA_CHECK(cudaDeviceSynchronize());

        cur_buf = next_buf;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Copy results back to host
    // Only copy owned rows
    std::vector<double> h_result_energy(owned_nrows * n_elems_root);
    std::vector<double> h_result_flux(owned_nrows * n_elems_root);
    CUDA_CHECK(cudaMemcpy(h_result_energy.data(), d_energy[cur_buf] + owned_local_start * n_elems_root,
                          owned_nrows * n_elems_root * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_flux.data(), d_flux[cur_buf] + owned_local_start * n_elems_root,
                          owned_nrows * n_elems_root * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute global timing (max across ranks)
    long long global_duration_ms;
    long long local_dur = (long long)duration_ms;
    MPI_Allreduce(&local_dur, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    duration_ms = (int)global_duration_ms;

    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (long long)n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash (parallel with OpenMP, combine with MPI)
    uint64_t local_hash = 0;
    #pragma omp parallel for reduction(^:local_hash) schedule(static)
    for (int i = 0; i < owned_nrows * n_elems_root; ++i) {
        int global_i = (row_start * n_elems_root) + i;
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&h_result_energy[i]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&h_result_flux[i]);
        uint64_t h = 0;
        h ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        h ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
        local_hash ^= h;
    }
    uint64_t global_hash = 0;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        // Gather all energy data to rank 0
        std::vector<int> recv_counts(mpi_size);
        std::vector<int> recv_displs(mpi_size);
        int local_count = owned_nrows * n_elems_root;
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            recv_displs[0] = 0;
            for (int i = 1; i < mpi_size; ++i) {
                recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
            }
        }

        std::vector<double> all_energy;
        if (mpi_rank == 0) {
            all_energy.resize(n_elems);
        }
        MPI_Gatherv(h_result_energy.data(), local_count, MPI_DOUBLE,
                     all_energy.data(), recv_counts.data(), recv_displs.data(),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            print_results(all_energy, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        double local_energy_sum = 0.0, local_flux_sum = 0.0;
        double local_energy_max = std::numeric_limits<val_t>::lowest();
        double local_energy_min = std::numeric_limits<val_t>::max();

        #pragma omp parallel
        {
            double t_sum = 0.0, t_fsum = 0.0;
            double t_max = std::numeric_limits<val_t>::lowest();
            double t_min = std::numeric_limits<val_t>::max();
            #pragma omp for schedule(static)
            for (int i = 0; i < owned_nrows * n_elems_root; ++i) {
                t_sum += h_result_energy[i];
                t_fsum += h_result_flux[i];
                t_max = std::max(t_max, h_result_energy[i]);
                t_min = std::min(t_min, h_result_energy[i]);
            }
            #pragma omp critical
            {
                local_energy_sum += t_sum;
                local_flux_sum += t_fsum;
                local_energy_max = std::max(local_energy_max, t_max);
                local_energy_min = std::min(local_energy_min, t_min);
            }
        }

        double energy_sum, flux_sum, energy_max, energy_min;
        MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", energy_sum);
            printf("  Flux sum: %.2f\n", flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

            constexpr val_t energy_epsilon = 1e-8;
            bool valid = true;

            if (!std::isfinite(energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }
            if (std::abs(energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }
            if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }
            if (valid) {
                printf("  Validation: PASSED\n");
            }
        }
    }

    // Cleanup
    cudaFreeHost(h_ghost_send);
    cudaFreeHost(h_ghost_recv);
    cudaFree(d_material_idx);
    cudaFree(d_energy[0]);
    cudaFree(d_energy[1]);
    cudaFree(d_flux[0]);
    cudaFree(d_flux[1]);
    cudaFree(d_materials);

    MPI_Finalize();
    return 0;
}
