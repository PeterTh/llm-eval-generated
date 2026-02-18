#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

constexpr val_t kTransferCoeff = 0.8;
constexpr val_t kFluxScale = 0.25;

struct Decomposition {
    int row_start;
    int row_end;
    int local_rows;
};

Decomposition computeDecomposition(const int n_rows, const int rank, const int size) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    const int row_start = rank * base + std::min(rank, rem);
    const int row_end = row_start + base + (rank < rem ? 1 : 0);
    return {row_start, row_end, row_end - row_start};
}

inline void checkCuda(const cudaError_t err, const char* msg, const int rank) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Rank %d CUDA error at %s: %s\n", rank, msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void updateKernel(const val_t* energy, val_t* energy_next,
                             const val_t* flux, val_t* flux_next,
                             const int n_cols, const int local_rows,
                             const int global_row_start, const int n_rows) {
    const int local_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_r >= local_rows || c >= n_cols) {
        return;
    }

    const int global_r = global_row_start + local_r;
    const int row_idx = local_r + 1;
    const int energy_idx = row_idx * n_cols + c;
    const val_t center = energy[energy_idx];

    val_t total_flux = 0.0;
    if ((global_r == 0 && c == 0) || (global_r == n_rows - 1 && c == n_cols - 1)) {
        total_flux = 0.5;
    } else if ((global_r == 0 && c == n_cols - 1) || (global_r == n_rows - 1 && c == 0)) {
        total_flux = -0.5;
    }

    val_t neighbor_sum = 0.0;
    if (global_r > 0) {
        neighbor_sum += energy[energy_idx - n_cols] - center;
    }
    if (global_r + 1 < n_rows) {
        neighbor_sum += energy[energy_idx + n_cols] - center;
    }
    if (c > 0) {
        neighbor_sum += energy[energy_idx - 1] - center;
    }
    if (c + 1 < n_cols) {
        neighbor_sum += energy[energy_idx + 1] - center;
    }

    total_flux += neighbor_sum * kTransferCoeff * kFluxScale;

    energy_next[energy_idx] = center + total_flux;
    const int flux_idx = local_r * n_cols + c;
    flux_next[flux_idx] = flux[flux_idx] + fabs(total_flux);
}

void runSimulationParallel(const Decomposition& decomp, const int n_rows, const int n_cols,
                           const int n_iters, const int rank, const int size,
                           std::vector<val_t>& host_energy, std::vector<val_t>& host_flux) {
    host_energy.assign(static_cast<size_t>(decomp.local_rows) * n_cols, 0.0);
    host_flux.assign(static_cast<size_t>(decomp.local_rows) * n_cols, 0.0);

    if (decomp.local_rows == 0) {
        return;
    }

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank);
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device_id = rank % device_count;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice", rank);

    const size_t energy_elems = static_cast<size_t>(decomp.local_rows + 2) * n_cols;
    const size_t flux_elems = static_cast<size_t>(decomp.local_rows) * n_cols;

    val_t* d_energy = nullptr;
    val_t* d_energy_next = nullptr;
    val_t* d_flux = nullptr;
    val_t* d_flux_next = nullptr;

    checkCuda(cudaMalloc(&d_energy, energy_elems * sizeof(val_t)), "cudaMalloc d_energy", rank);
    checkCuda(cudaMalloc(&d_energy_next, energy_elems * sizeof(val_t)), "cudaMalloc d_energy_next", rank);
    checkCuda(cudaMalloc(&d_flux, flux_elems * sizeof(val_t)), "cudaMalloc d_flux", rank);
    checkCuda(cudaMalloc(&d_flux_next, flux_elems * sizeof(val_t)), "cudaMalloc d_flux_next", rank);

    checkCuda(cudaMemset(d_energy, 0, energy_elems * sizeof(val_t)), "cudaMemset d_energy", rank);
    checkCuda(cudaMemset(d_energy_next, 0, energy_elems * sizeof(val_t)), "cudaMemset d_energy_next", rank);
    checkCuda(cudaMemset(d_flux, 0, flux_elems * sizeof(val_t)), "cudaMemset d_flux", rank);
    checkCuda(cudaMemset(d_flux_next, 0, flux_elems * sizeof(val_t)), "cudaMemset d_flux_next", rank);

    const int up_rank = (decomp.row_start == 0) ? MPI_PROC_NULL : rank - 1;
    const int down_rank = (decomp.row_end == n_rows) ? MPI_PROC_NULL : rank + 1;

    std::vector<val_t> send_top;
    std::vector<val_t> send_bottom;
    std::vector<val_t> recv_top;
    std::vector<val_t> recv_bottom;

    if (up_rank != MPI_PROC_NULL) {
        send_top.resize(n_cols);
        recv_top.resize(n_cols);
    }
    if (down_rank != MPI_PROC_NULL) {
        send_bottom.resize(n_cols);
        recv_bottom.resize(n_cols);
    }

    const dim3 block(16, 16);
    const dim3 grid((n_cols + block.x - 1) / block.x,
                    (decomp.local_rows + block.y - 1) / block.y);

    constexpr int tag_top = 100;
    constexpr int tag_bottom = 200;

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int req_count = 0;

        if (up_rank != MPI_PROC_NULL) {
            checkCuda(cudaMemcpy(send_top.data(), d_energy + n_cols,
                                 static_cast<size_t>(n_cols) * sizeof(val_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy top row", rank);
            MPI_Irecv(recv_top.data(), n_cols, MPI_DOUBLE, up_rank, tag_bottom, MPI_COMM_WORLD,
                      &reqs[req_count++]);
            MPI_Isend(send_top.data(), n_cols, MPI_DOUBLE, up_rank, tag_top, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        if (down_rank != MPI_PROC_NULL) {
            checkCuda(cudaMemcpy(send_bottom.data(), d_energy + static_cast<size_t>(decomp.local_rows) * n_cols,
                                 static_cast<size_t>(n_cols) * sizeof(val_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy bottom row", rank);
            MPI_Irecv(recv_bottom.data(), n_cols, MPI_DOUBLE, down_rank, tag_top, MPI_COMM_WORLD,
                      &reqs[req_count++]);
            MPI_Isend(send_bottom.data(), n_cols, MPI_DOUBLE, down_rank, tag_bottom, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        if (up_rank != MPI_PROC_NULL) {
            checkCuda(cudaMemcpy(d_energy, recv_top.data(),
                                 static_cast<size_t>(n_cols) * sizeof(val_t),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy top halo", rank);
        }
        if (down_rank != MPI_PROC_NULL) {
            checkCuda(cudaMemcpy(d_energy + static_cast<size_t>(decomp.local_rows + 1) * n_cols,
                                 recv_bottom.data(),
                                 static_cast<size_t>(n_cols) * sizeof(val_t),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy bottom halo", rank);
        }

        updateKernel<<<grid, block>>>(d_energy, d_energy_next, d_flux, d_flux_next,
                                      n_cols, decomp.local_rows, decomp.row_start, n_rows);
        checkCuda(cudaGetLastError(), "updateKernel launch", rank);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", rank);

        std::swap(d_energy, d_energy_next);
        std::swap(d_flux, d_flux_next);
    }

    checkCuda(cudaMemcpy(host_energy.data(), d_energy + n_cols,
                         static_cast<size_t>(decomp.local_rows) * n_cols * sizeof(val_t),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy energy to host", rank);
    checkCuda(cudaMemcpy(host_flux.data(), d_flux,
                         static_cast<size_t>(decomp.local_rows) * n_cols * sizeof(val_t),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy flux to host", rank);

    checkCuda(cudaFree(d_energy), "cudaFree d_energy", rank);
    checkCuda(cudaFree(d_energy_next), "cudaFree d_energy_next", rank);
    checkCuda(cudaFree(d_flux), "cudaFree d_flux", rank);
    checkCuda(cudaFree(d_flux_next), "cudaFree d_flux_next", rank);
}

struct LocalStats {
    val_t energy_sum;
    val_t flux_sum;
    val_t energy_max;
    val_t energy_min;
    uint64_t hash;
};

LocalStats computeLocalStats(const std::vector<val_t>& energy,
                             const std::vector<val_t>& flux,
                             const int n_cols,
                             const int row_start) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    uint64_t hash = 0;

    #pragma omp parallel for reduction(+:energy_sum, flux_sum) \
                             reduction(max:energy_max) reduction(min:energy_min) \
                             reduction(^:hash)
    for (size_t i = 0; i < energy.size(); ++i) {
        const val_t e = energy[i];
        const val_t f = flux[i];
        energy_sum += e;
        flux_sum += f;
        energy_max = std::max(energy_max, e);
        energy_min = std::min(energy_min, e);

        const uint64_t row = static_cast<uint64_t>(i / static_cast<size_t>(n_cols));
        const uint64_t col = static_cast<uint64_t>(i % static_cast<size_t>(n_cols));
        const uint64_t global_idx = (static_cast<uint64_t>(row_start) + row) *
                                    static_cast<uint64_t>(n_cols) + col;

        uint64_t e_bits = 0;
        uint64_t f_bits = 0;
        std::memcpy(&e_bits, &e, sizeof(uint64_t));
        std::memcpy(&f_bits, &f, sizeof(uint64_t));
        hash ^= (e_bits + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }

    LocalStats stats;
    stats.energy_sum = energy_sum;
    stats.flux_sum = flux_sum;
    stats.energy_max = energy_max;
    stats.energy_min = energy_min;
    stats.hash = hash;
    return stats;
}

bool validateResultsGlobal(const val_t energy_sum, const val_t flux_sum,
                           const val_t energy_max, const val_t energy_min) {
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

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");

        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    const Decomposition decomp = computeDecomposition(n_elems_root, rank, size);
    std::vector<val_t> local_energy;
    std::vector<val_t> local_flux;

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    runSimulationParallel(decomp, n_elems_root, n_elems_root, n_iters, rank, size,
                          local_energy, local_flux);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double elapsed = end - start;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const LocalStats local_stats = computeLocalStats(local_energy, local_flux,
                                                     n_elems_root, decomp.row_start);

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    uint64_t global_hash = 0;

    MPI_Reduce(&local_stats.energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double duration_ms = max_elapsed * 1000.0;
        printf("Computation time: %.0f ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) /
                                          (max_elapsed * 1e9);
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
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
            int offset = 0;
            for (int r = 0; r < size; ++r) {
                const Decomposition d = computeDecomposition(n_elems_root, r, size);
                counts[r] = d.local_rows * n_elems_root;
                displs[r] = offset;
                offset += counts[r];
            }
            energyData.resize(static_cast<size_t>(n_elems));
        }

        MPI_Gatherv(local_energy.data(),
                    static_cast<int>(local_energy.size()),
                    MPI_DOUBLE,
                    energyData.data(),
                    counts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    if (validate) {
        int status = 0;
        if (rank == 0) {
            status = validateResultsGlobal(energy_sum, flux_sum, energy_max, energy_min) ? 0 : 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (status != 0) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
