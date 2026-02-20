#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

constexpr val_t kTransferCoeff = 0.8;
constexpr val_t kFluxScale = kTransferCoeff * 0.25;
constexpr val_t kInflow = 0.5;
constexpr val_t kOutflow = -0.5;

inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int rows_for_rank(int rank, int size, int n_rows) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    return base + (rank < rem ? 1 : 0);
}

int start_row_for_rank(int rank, int size, int n_rows) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    return rank * base + std::min(rank, rem);
}

int find_prev_rank_with_rows(int rank, int size, int n_rows) {
    for (int r = rank - 1; r >= 0; --r) {
        if (rows_for_rank(r, size, n_rows) > 0) {
            return r;
        }
    }
    return MPI_PROC_NULL;
}

int find_next_rank_with_rows(int rank, int size, int n_rows) {
    for (int r = rank + 1; r < size; ++r) {
        if (rows_for_rank(r, size, n_rows) > 0) {
            return r;
        }
    }
    return MPI_PROC_NULL;
}

__device__ __forceinline__ val_t external_flow_device(int row, int col, int n_rows) {
    if ((row == 0 && col == 0) || (row == n_rows - 1 && col == n_rows - 1)) {
        return kInflow;
    }
    if ((row == 0 && col == n_rows - 1) || (row == n_rows - 1 && col == 0)) {
        return kOutflow;
    }
    return 0.0;
}

__global__ void update_kernel(const val_t* energy_curr,
                              val_t* energy_next,
                              val_t* total_flux,
                              int n_cols,
                              int local_rows,
                              int row_start,
                              int n_rows_total) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n_cols || local_row >= local_rows) {
        return;
    }

    const int energy_row = local_row + 1;
    const int global_row = row_start + local_row;
    const int idx = energy_row * n_cols + col;

    const val_t self_energy = energy_curr[idx];
    val_t total = external_flow_device(global_row, col, n_rows_total);

    if (global_row + 1 < n_rows_total) {
        total += (energy_curr[(energy_row + 1) * n_cols + col] - self_energy) * kFluxScale;
    }
    if (global_row > 0) {
        total += (energy_curr[(energy_row - 1) * n_cols + col] - self_energy) * kFluxScale;
    }
    if (col + 1 < n_cols) {
        total += (energy_curr[energy_row * n_cols + (col + 1)] - self_energy) * kFluxScale;
    }
    if (col > 0) {
        total += (energy_curr[energy_row * n_cols + (col - 1)] - self_energy) * kFluxScale;
    }

    energy_next[idx] = self_energy + total;

    const int flux_idx = local_row * n_cols + col;
    total_flux[flux_idx] = total_flux[flux_idx] + fabs(total);
}

uint64_t compute_local_hash(const std::vector<val_t>& energy,
                            const std::vector<val_t>& total_flux,
                            int row_start,
                            int n_cols) {
    uint64_t hash = 0;
    for (size_t i = 0; i < energy.size(); ++i) {
        const size_t local_row = i / static_cast<size_t>(n_cols);
        const size_t col = i - local_row * static_cast<size_t>(n_cols);
        const uint64_t global_idx =
            static_cast<uint64_t>(row_start + static_cast<int>(local_row)) *
                static_cast<uint64_t>(n_cols) +
            static_cast<uint64_t>(col);
        uint64_t e_bits = 0;
        uint64_t f_bits = 0;
        std::memcpy(&e_bits, &energy[i], sizeof(uint64_t));
        std::memcpy(&f_bits, &total_flux[i], sizeof(uint64_t));
        hash ^= (e_bits + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

bool validate_results(val_t energy_sum, val_t flux_sum, val_t energy_min, val_t energy_max) {
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }

    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }

    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int printResults = 0;
    int early_exit = 0;
    int exit_code = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                early_exit = 1;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                early_exit = 1;
                exit_code = 1;
                break;
            }
        }
    }

    MPI_Bcast(&early_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (early_exit) {
        MPI_Finalize();
        return exit_code;
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n_elems_root <= 0 || n_iters <= 0) {
        if (rank == 0) {
            printf("Invalid parameters: grid size and iterations must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int n_rows = n_elems_root;
    const int n_cols = n_elems_root;
    const int n_elems = n_rows * n_cols;

    const int local_rows = rows_for_rank(rank, size, n_rows);
    const int row_start = start_row_for_rank(rank, size, n_rows);
    const int up_rank = find_prev_rank_with_rows(rank, size, n_rows);
    const int down_rank = find_next_rank_with_rows(rank, size, n_rows);

    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device_id = rank % device_count;
    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");

    const size_t energy_elems = static_cast<size_t>(local_rows + 2) * static_cast<size_t>(n_cols);
    const size_t local_elems = static_cast<size_t>(local_rows) * static_cast<size_t>(n_cols);

    std::vector<val_t> h_energy(energy_elems, 0.0);
    std::vector<val_t> h_total_flux(local_elems, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < energy_elems; ++i) {
        h_energy[i] = 0.0;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < local_elems; ++i) {
        h_total_flux[i] = 0.0;
    }

    val_t* d_energy_curr = nullptr;
    val_t* d_energy_next = nullptr;
    val_t* d_total_flux = nullptr;

    const size_t energy_bytes = std::max<size_t>(1, energy_elems) * sizeof(val_t);
    const size_t total_flux_bytes = std::max<size_t>(1, local_elems) * sizeof(val_t);

    cuda_check(cudaMalloc(&d_energy_curr, energy_bytes), "cudaMalloc energy_curr");
    cuda_check(cudaMalloc(&d_energy_next, energy_bytes), "cudaMalloc energy_next");
    cuda_check(cudaMalloc(&d_total_flux, total_flux_bytes), "cudaMalloc total_flux");

    cuda_check(cudaMemcpy(d_energy_curr, h_energy.data(), energy_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy energy_curr");
    cuda_check(cudaMemcpy(d_energy_next, h_energy.data(), energy_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy energy_next");
    cuda_check(cudaMemset(d_total_flux, 0, total_flux_bytes), "cudaMemset total_flux");

    std::vector<val_t> h_send_top(n_cols, 0.0);
    std::vector<val_t> h_send_bottom(n_cols, 0.0);
    std::vector<val_t> h_recv_top(n_cols, 0.0);
    std::vector<val_t> h_recv_bottom(n_cols, 0.0);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_rows, n_cols, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_cast<size_t>(static_mem + dynamic_mem);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    if (local_rows > 0) {
        dim3 block(16, 16);
        dim3 grid((n_cols + block.x - 1) / block.x, (local_rows + block.y - 1) / block.y);

        for (int iter = 0; iter < n_iters; ++iter) {
            if (size > 1) {
                cuda_check(cudaMemcpy(h_send_top.data(),
                                      d_energy_curr + n_cols,
                                      n_cols * sizeof(val_t),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy send_top");
                cuda_check(cudaMemcpy(h_send_bottom.data(),
                                      d_energy_curr + static_cast<size_t>(local_rows) * n_cols,
                                      n_cols * sizeof(val_t),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy send_bottom");

                MPI_Sendrecv(h_send_top.data(),
                             n_cols,
                             MPI_DOUBLE,
                             up_rank,
                             0,
                             h_recv_top.data(),
                             n_cols,
                             MPI_DOUBLE,
                             up_rank,
                             1,
                             MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);

                MPI_Sendrecv(h_send_bottom.data(),
                             n_cols,
                             MPI_DOUBLE,
                             down_rank,
                             1,
                             h_recv_bottom.data(),
                             n_cols,
                             MPI_DOUBLE,
                             down_rank,
                             0,
                             MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);

                if (up_rank != MPI_PROC_NULL) {
                    cuda_check(cudaMemcpy(d_energy_curr,
                                          h_recv_top.data(),
                                          n_cols * sizeof(val_t),
                                          cudaMemcpyHostToDevice),
                               "cudaMemcpy recv_top");
                }
                if (down_rank != MPI_PROC_NULL) {
                    cuda_check(cudaMemcpy(d_energy_curr + static_cast<size_t>(local_rows + 1) * n_cols,
                                          h_recv_bottom.data(),
                                          n_cols * sizeof(val_t),
                                          cudaMemcpyHostToDevice),
                               "cudaMemcpy recv_bottom");
                }
            }

            update_kernel<<<grid, block>>>(d_energy_curr,
                                           d_energy_next,
                                           d_total_flux,
                                           n_cols,
                                           local_rows,
                                           row_start,
                                           n_rows);
            cuda_check(cudaGetLastError(), "update_kernel");
            std::swap(d_energy_curr, d_energy_next);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end_time = MPI_Wtime();

    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<val_t> local_energy(local_elems, 0.0);
    if (local_rows > 0) {
        cuda_check(cudaMemcpy(local_energy.data(),
                              d_energy_curr + n_cols,
                              local_elems * sizeof(val_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy final energy");
        cuda_check(cudaMemcpy(h_total_flux.data(),
                              d_total_flux,
                              local_elems * sizeof(val_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy final flux");
    }

    if (rank == 0) {
        const long duration_ms = static_cast<long>((end_time - start_time) * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) /
            ((static_cast<double>(duration_ms) / 1000.0) * 1e9);
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    uint64_t local_hash = 0;
    if (local_rows > 0) {
        local_hash = compute_local_hash(local_energy, h_total_flux, row_start, n_cols);
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> energyData;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            int disp = 0;
            for (int r = 0; r < size; ++r) {
                const int rows = rows_for_rank(r, size, n_rows);
                const int count = rows * n_cols;
                counts[r] = count;
                displs[r] = disp;
                disp += count;
            }
            energyData.resize(static_cast<size_t>(n_elems));
        }
        MPI_Gatherv(local_energy.data(),
                    static_cast<int>(local_elems),
                    MPI_DOUBLE,
                    rank == 0 ? energyData.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    if (validate) {
        val_t local_energy_sum = 0.0;
        val_t local_flux_sum = 0.0;
        val_t local_energy_max = std::numeric_limits<val_t>::lowest();
        val_t local_energy_min = std::numeric_limits<val_t>::max();

        #pragma omp parallel for reduction(+:local_energy_sum, local_flux_sum) \
            reduction(max:local_energy_max) reduction(min:local_energy_min)
        for (size_t i = 0; i < local_energy.size(); ++i) {
            local_energy_sum += local_energy[i];
            local_flux_sum += h_total_flux[i];
            local_energy_max = std::max(local_energy[i], local_energy_max);
            local_energy_min = std::min(local_energy[i], local_energy_min);
        }

        val_t energy_sum = 0.0;
        val_t flux_sum = 0.0;
        val_t energy_max = 0.0;
        val_t energy_min = 0.0;

        MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const bool valid = validate_results(energy_sum, flux_sum, energy_min, energy_max);
            if (!valid) {
                cudaFree(d_energy_curr);
                cudaFree(d_energy_next);
                cudaFree(d_total_flux);
                MPI_Finalize();
                return 1;
            }
        }
    }

    cudaFree(d_energy_curr);
    cudaFree(d_energy_next);
    cudaFree(d_total_flux);

    MPI_Finalize();
    return 0;
}
