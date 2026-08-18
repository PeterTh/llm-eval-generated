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

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using val_t = double;

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t INFLOW = 0.5;
constexpr val_t OUTFLOW = -0.5;
constexpr int CUDA_BLOCK_SIZE = 256;

namespace {

int world_rank = 0;

void checkCuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d for %s: %s\n",
                 world_rank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
}

void checkMpi(int error, const char* expression, const char* file, int line) {
    if (error == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: MPI failure at %s:%d for %s: %.*s\n",
                 world_rank, file, line, expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

struct Partition {
    int first_row;
    int rows;
    int previous_rank;
    int next_rank;
};

Partition makePartition(int n, int rank, int ranks) {
    const int rows_per_rank = n / ranks;
    const int remainder = n % ranks;
    const int rows = rows_per_rank + (rank < remainder ? 1 : 0);
    const int first_row = rank * rows_per_rank + std::min(rank, remainder);
    return {first_row, rows,
            rank == 0 ? MPI_PROC_NULL : rank - 1,
            rank + 1 == ranks ? MPI_PROC_NULL : rank + 1};
}

bool cudaAwareMpiAvailable() {
#ifdef MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

__device__ __forceinline__ val_t externalFlow(int global_row, int column, int n) {
    if ((global_row == 0 && column == 0) ||
        (global_row == n - 1 && column == n - 1)) {
        return INFLOW;
    }
    if ((global_row == 0 && column == n - 1) ||
        (global_row == n - 1 && column == 0)) {
        return OUTFLOW;
    }
    return 0.0;
}

__device__ __forceinline__ void updateElement(
    const val_t* __restrict__ current_energy,
    const val_t* __restrict__ current_total_flux,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_total_flux,
    int n, int local_row, int global_first_row, int column) {
    const int global_row = global_first_row + local_row;
    const size_t index = static_cast<size_t>(local_row + 1) * n + column;
    const val_t energy = current_energy[index];
    val_t flux = externalFlow(global_row, column, n);

    // Retain the original connection order: +x, -x, +y, -y.
    if (global_row + 1 < n) {
        flux += (current_energy[index + n] - energy) * TRANSFER_COEFF * 1.0 * 0.25;
    }
    if (global_row > 0) {
        flux += (current_energy[index - n] - energy) * TRANSFER_COEFF * 1.0 * 0.25;
    }
    if (column + 1 < n) {
        flux += (current_energy[index + 1] - energy) * TRANSFER_COEFF * 1.0 * 0.25;
    }
    if (column > 0) {
        flux += (current_energy[index - 1] - energy) * TRANSFER_COEFF * 1.0 * 0.25;
    }

    next_energy[index] = energy + flux;
    next_total_flux[index] = current_total_flux[index] + fabs(flux);
}

__global__ void updateRowsKernel(
    const val_t* __restrict__ current_energy,
    const val_t* __restrict__ current_total_flux,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_total_flux,
    int n, int first_local_row, int row_count, int global_first_row) {
    const int column = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row_offset = static_cast<int>(blockIdx.y);
    if (column < n && row_offset < row_count) {
        updateElement(current_energy, current_total_flux, next_energy, next_total_flux,
                      n, first_local_row + row_offset, global_first_row, column);
    }
}

__global__ void updateBoundaryRowsKernel(
    const val_t* __restrict__ current_energy,
    const val_t* __restrict__ current_total_flux,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_total_flux,
    int n, int local_rows, int global_first_row) {
    const int column = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= n) {
        return;
    }
    const int local_row = blockIdx.y == 0 ? 0 : local_rows - 1;
    updateElement(current_energy, current_total_flux, next_energy, next_total_flux,
                  n, local_row, global_first_row, column);
}

struct DeviceState {
    val_t* energy[2] = {nullptr, nullptr};
    val_t* total_flux[2] = {nullptr, nullptr};
    size_t values = 0;

    explicit DeviceState(size_t value_count) : values(value_count) {
        const size_t bytes = values * sizeof(val_t);
        for (int buffer = 0; buffer < 2; ++buffer) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&energy[buffer]), bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&total_flux[buffer]), bytes));
            CUDA_CHECK(cudaMemset(energy[buffer], 0, bytes));
            CUDA_CHECK(cudaMemset(total_flux[buffer], 0, bytes));
        }
    }

    ~DeviceState() {
        for (int buffer = 0; buffer < 2; ++buffer) {
            if (energy[buffer] != nullptr) {
                cudaFree(energy[buffer]);
            }
            if (total_flux[buffer] != nullptr) {
                cudaFree(total_flux[buffer]);
            }
        }
    }

    DeviceState(const DeviceState&) = delete;
    DeviceState& operator=(const DeviceState&) = delete;
};

struct HostHalos {
    val_t* send_previous = nullptr;
    val_t* send_next = nullptr;
    val_t* receive_previous = nullptr;
    val_t* receive_next = nullptr;

    HostHalos(int n, bool needed) {
        if (!needed) {
            return;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(val_t);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&send_previous), bytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&send_next), bytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&receive_previous), bytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&receive_next), bytes));
    }

    ~HostHalos() {
        if (send_previous != nullptr) cudaFreeHost(send_previous);
        if (send_next != nullptr) cudaFreeHost(send_next);
        if (receive_previous != nullptr) cudaFreeHost(receive_previous);
        if (receive_next != nullptr) cudaFreeHost(receive_next);
    }

    HostHalos(const HostHalos&) = delete;
    HostHalos& operator=(const HostHalos&) = delete;
};

int runSimulation(DeviceState& state, const Partition& partition, int n, int n_iters,
                  bool cuda_aware_mpi) {
    cudaStream_t compute_stream = nullptr;
    cudaStream_t communication_stream = nullptr;
    cudaEvent_t update_complete = nullptr;
    cudaEvent_t halos_ready = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communication_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&update_complete, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(update_complete, compute_stream));

    const bool staged_halos = !cuda_aware_mpi &&
                              (partition.previous_rank != MPI_PROC_NULL ||
                               partition.next_rank != MPI_PROC_NULL);
    HostHalos host_halos(n, staged_halos);
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(val_t);
    const dim3 threads(CUDA_BLOCK_SIZE);
    const dim3 column_blocks((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    int current = 0;

    for (int iteration = 0; iteration < n_iters; ++iteration) {
        const int next = 1 - current;
        val_t* current_energy = state.energy[current];
        val_t* current_flux = state.total_flux[current];
        val_t* next_energy = state.energy[next];
        val_t* next_flux = state.total_flux[next];

        if (partition.previous_rank == MPI_PROC_NULL &&
            partition.next_rank == MPI_PROC_NULL) {
            const dim3 grid(column_blocks.x, partition.rows);
            updateRowsKernel<<<grid, threads, 0, compute_stream>>>(
                current_energy, current_flux, next_energy, next_flux,
                n, 0, partition.rows, partition.first_row);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(update_complete, compute_stream));
            current = next;
            continue;
        }

        MPI_Request requests[4];
        int request_count = 0;

        if (cuda_aware_mpi) {
            // MPI may not observe work in a non-default CUDA stream, so make only
            // the preceding boundary update visible before handing it pointers.
            CUDA_CHECK(cudaEventSynchronize(update_complete));
            if (partition.previous_rank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(current_energy, n, MPI_DOUBLE,
                                    partition.previous_rank, 17, MPI_COMM_WORLD,
                                    &requests[request_count++]));
            }
            if (partition.next_rank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(current_energy +
                                        static_cast<size_t>(partition.rows + 1) * n,
                                    n, MPI_DOUBLE, partition.next_rank, 18,
                                    MPI_COMM_WORLD, &requests[request_count++]));
            }
            if (partition.previous_rank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Isend(current_energy + n, n, MPI_DOUBLE,
                                    partition.previous_rank, 18, MPI_COMM_WORLD,
                                    &requests[request_count++]));
            }
            if (partition.next_rank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Isend(current_energy +
                                        static_cast<size_t>(partition.rows) * n,
                                    n, MPI_DOUBLE, partition.next_rank, 17,
                                    MPI_COMM_WORLD, &requests[request_count++]));
            }
        } else {
            if (partition.previous_rank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(host_halos.receive_previous, n, MPI_DOUBLE,
                                    partition.previous_rank, 17, MPI_COMM_WORLD,
                                    &requests[request_count++]));
            }
            if (partition.next_rank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(host_halos.receive_next, n, MPI_DOUBLE,
                                    partition.next_rank, 18, MPI_COMM_WORLD,
                                    &requests[request_count++]));
            }

            CUDA_CHECK(cudaStreamWaitEvent(communication_stream, update_complete, 0));
            if (partition.previous_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(host_halos.send_previous,
                                           current_energy + n, row_bytes,
                                           cudaMemcpyDeviceToHost,
                                           communication_stream));
            }
            if (partition.next_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(host_halos.send_next,
                                           current_energy +
                                               static_cast<size_t>(partition.rows) * n,
                                           row_bytes, cudaMemcpyDeviceToHost,
                                           communication_stream));
            }
        }

        // Interior rows do not depend on incoming halos and overlap communication.
        if (partition.rows > 2) {
            const dim3 grid(column_blocks.x, partition.rows - 2);
            updateRowsKernel<<<grid, threads, 0, compute_stream>>>(
                current_energy, current_flux, next_energy, next_flux,
                n, 1, partition.rows - 2, partition.first_row);
            CUDA_CHECK(cudaGetLastError());
        }

        if (!cuda_aware_mpi) {
            CUDA_CHECK(cudaStreamSynchronize(communication_stream));
            if (partition.previous_rank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Isend(host_halos.send_previous, n, MPI_DOUBLE,
                                    partition.previous_rank, 18, MPI_COMM_WORLD,
                                    &requests[request_count++]));
            }
            if (partition.next_rank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Isend(host_halos.send_next, n, MPI_DOUBLE,
                                    partition.next_rank, 17, MPI_COMM_WORLD,
                                    &requests[request_count++]));
            }
        }

        MPI_CHECK(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));

        if (!cuda_aware_mpi) {
            if (partition.previous_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(current_energy,
                                           host_halos.receive_previous, row_bytes,
                                           cudaMemcpyHostToDevice,
                                           communication_stream));
            }
            if (partition.next_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(
                    current_energy + static_cast<size_t>(partition.rows + 1) * n,
                    host_halos.receive_next, row_bytes, cudaMemcpyHostToDevice,
                    communication_stream));
            }
            CUDA_CHECK(cudaEventRecord(halos_ready, communication_stream));
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, halos_ready, 0));
        }

        const unsigned int boundary_rows = partition.rows == 1 ? 1U : 2U;
        const dim3 boundary_grid(column_blocks.x, boundary_rows);
        updateBoundaryRowsKernel<<<boundary_grid, threads, 0, compute_stream>>>(
            current_energy, current_flux, next_energy, next_flux,
            n, partition.rows, partition.first_row);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(update_complete, compute_stream));
        current = next;
    }

    CUDA_CHECK(cudaEventSynchronize(update_complete));
    CUDA_CHECK(cudaEventDestroy(halos_ready));
    CUDA_CHECK(cudaEventDestroy(update_complete));
    CUDA_CHECK(cudaStreamDestroy(communication_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    return current;
}

uint64_t doubleBits(val_t value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t computeDistributedHash(const std::vector<val_t>& energy,
                                const std::vector<val_t>& total_flux,
                                size_t global_offset) {
    uint64_t local_hash = 0;
    const int64_t count = static_cast<int64_t>(energy.size());
#pragma omp parallel for schedule(static) reduction(^ : local_hash)
    for (int64_t i = 0; i < count; ++i) {
        const uint64_t global_index = global_offset + static_cast<uint64_t>(i);
        local_hash ^= (doubleBits(energy[i]) + global_index) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (doubleBits(total_flux[i]) + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    return local_hash;
}

bool validateDistributed(const std::vector<val_t>& energy,
                         const std::vector<val_t>& total_flux, int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    const int64_t count = static_cast<int64_t>(energy.size());

#pragma omp parallel for schedule(static) reduction(+ : local_energy_sum, local_flux_sum) \
    reduction(max : local_energy_max) reduction(min : local_energy_min)
    for (int64_t i = 0; i < count; ++i) {
        local_energy_sum += energy[i];
        local_flux_sum += total_flux[i];
        local_energy_max = std::max(local_energy_max, energy[i]);
        local_energy_min = std::min(local_energy_min, energy[i]);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = 0.0;
    val_t energy_min = 0.0;
    MPI_CHECK(MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM,
                         0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM,
                         0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX,
                         0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN,
                         0, MPI_COMM_WORLD));

    int valid = 1;
    if (rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", energy_sum);
        std::printf("  Flux sum: %.2f\n", flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(energy_sum)) {
            std::printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        }
        if (std::abs(energy_sum) > 1e-8) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            std::printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            std::printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) {
            std::printf("  Validation: PASSED\n");
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return valid != 0;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level));

    int rank_count = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &rank_count));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return 1;
    }

    int n = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool show_help = false;
    bool arguments_valid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (world_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            arguments_valid = false;
        }
    }

    constexpr int max_grid_root = 46340;
    if (n <= 0 || n > max_grid_root || n_iters < 0 || rank_count > n) {
        if (world_rank == 0) {
            std::fprintf(stderr,
                         "Grid size must be in [1, %d], iterations must be nonnegative, "
                         "and MPI ranks cannot exceed grid rows.\n",
                         max_grid_root);
        }
        arguments_valid = false;
    }
    if (show_help || !arguments_valid) {
        if (world_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return arguments_valid ? 0 : 1;
    }

    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                  world_rank, MPI_INFO_NULL, &local_communicator));
    int local_rank = 0;
    int local_rank_count = 1;
    MPI_CHECK(MPI_Comm_rank(local_communicator, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_communicator, &local_rank_count));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (world_rank == 0) {
            std::fprintf(stderr, "No CUDA devices are available; CUDA execution is required.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device));
    const bool cuda_aware_mpi = cudaAwareMpiAvailable();
    const Partition partition = makePartition(n, world_rank, rank_count);
    const size_t global_elements = static_cast<size_t>(n) * n;
    const size_t local_elements = static_cast<size_t>(partition.rows) * n;
    const size_t device_values = static_cast<size_t>(partition.rows + 2) * n;

    if (world_rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n, n, global_elements);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", rank_count);
        std::printf("OpenMP threads/rank: %d\n", omp_get_max_threads());
        std::printf("CUDA device 0: %s\n", device_properties.name);
        std::printf("CUDA-aware MPI: %s (pinned host staging %s)\n",
                    cuda_aware_mpi ? "enabled" : "unavailable",
                    cuda_aware_mpi ? "disabled" : "enabled");
        if (local_rank_count > device_count) {
            std::printf("WARNING: %d local MPI ranks share %d CUDA device(s).\n",
                        local_rank_count, device_count);
        }
        std::printf("\nBuilding distributed unstructured mesh...\n");
    }

    DeviceState state(device_values);
    CUDA_CHECK(cudaDeviceSynchronize());

    const uint64_t local_device_bytes = 4ULL * device_values * sizeof(val_t);
    uint64_t total_device_bytes = 0;
    uint64_t maximum_device_bytes = 0;
    MPI_CHECK(MPI_Reduce(&local_device_bytes, &total_device_bytes, 1, MPI_UINT64_T,
                         MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_device_bytes, &maximum_device_bytes, 1, MPI_UINT64_T,
                         MPI_MAX, 0, MPI_COMM_WORLD));
    if (world_rank == 0) {
        std::printf("Device memory: %.2f MB aggregate (%.2f MB maximum/rank)\n\n",
                    total_device_bytes / (1024.0 * 1024.0),
                    maximum_device_bytes / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start_time = MPI_Wtime();
    const int final_buffer = runSimulation(state, partition, n, n_iters, cuda_aware_mpi);
    const double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
                         0, MPI_COMM_WORLD));

    std::vector<val_t> local_energy(local_elements);
    std::vector<val_t> local_total_flux(local_elements);
    const size_t local_bytes = local_elements * sizeof(val_t);
    CUDA_CHECK(cudaMemcpy(local_energy.data(), state.energy[final_buffer] + n,
                          local_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_total_flux.data(), state.total_flux[final_buffer] + n,
                          local_bytes, cudaMemcpyDeviceToHost));

    const size_t global_offset = static_cast<size_t>(partition.first_row) * n;
    const uint64_t local_hash =
        computeDistributedHash(local_energy, local_total_flux, global_offset);
    uint64_t result_hash = 0;
    MPI_CHECK(MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR,
                         0, MPI_COMM_WORLD));

    if (world_rank == 0) {
        const double duration_ms = elapsed * 1000.0;
        const double time_per_iteration = n_iters > 0 ? duration_ms / n_iters : 0.0;
        const double giga_elements_per_second =
            elapsed > 0.0 ? (static_cast<double>(n_iters) * global_elements) /
                                elapsed / 1.0e9
                          : 0.0;
        const double gflops = giga_elements_per_second * 22.0;
        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(result_hash));
    }

    if (print_results_requested) {
        std::vector<int> receive_counts;
        std::vector<int> displacements;
        std::vector<val_t> global_energy;
        if (world_rank == 0) {
            receive_counts.resize(rank_count);
            displacements.resize(rank_count);
#pragma omp parallel for schedule(static)
            for (int rank = 0; rank < rank_count; ++rank) {
                const Partition rank_partition = makePartition(n, rank, rank_count);
                receive_counts[rank] = rank_partition.rows * n;
                displacements[rank] = rank_partition.first_row * n;
            }
            global_energy.resize(global_elements);
        }
        MPI_CHECK(MPI_Gatherv(local_energy.data(), static_cast<int>(local_elements),
                              MPI_DOUBLE, global_energy.data(), receive_counts.data(),
                              displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (world_rank == 0) {
            print_results(global_energy, "ElementEnergy");
        }
    }

    bool valid = true;
    if (validate) {
        valid = validateDistributed(local_energy, local_total_flux, world_rank);
    }

    MPI_CHECK(MPI_Comm_free(&local_communicator));
    MPI_Finalize();
    return valid ? 0 : 1;
}
