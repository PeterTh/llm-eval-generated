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

using idx_t = uint64_t;
using val_t = double;

// This is retained as the externally visible dynamic state of one mesh element.
// The accelerator representation below is structure-of-arrays because only the
// two values, rather than the connectivity records, are updated every step.
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t CONNECTION_FLUX = 1.0;
constexpr val_t FLUX_SCALE = 0.25;

void mpiCheck(const int status, const char* call, const char* file, const int line) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING]{};
    int error_length = 0;
    MPI_Error_string(status, error, &error_length);
    std::fprintf(stderr, "MPI failure at %s:%d in %s: %.*s\n", file, line, call,
                 error_length, error);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}

void cudaCheck(const cudaError_t status, const char* call, const char* file, const int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "CUDA failure at %s:%d in %s: %s\n", file, line, call,
                 cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

#define MPI_CHECK(call) mpiCheck((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

// The original unstructured connectivity is a square grid with exactly these
// four possible edges (in down, up, right, left order).  Keeping that operation
// order preserves the numerical update while avoiding indirection-heavy loads.
__global__ void updateRows(const val_t* __restrict__ current_energy,
                           const val_t* __restrict__ current_flux,
                           val_t* __restrict__ next_energy,
                           val_t* __restrict__ next_flux,
                           const int n_cols,
                           const int global_first_row,
                           const int first_local_row,
                           const int rows_to_update) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_row = first_local_row + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (col >= n_cols || local_row >= first_local_row + rows_to_update) {
        return;
    }

    // Device row zero is the upper halo.  Physical local row r is r + 1.
    const int global_row = global_first_row + local_row;
    const size_t element = (static_cast<size_t>(local_row) + 1U) * n_cols + col;
    const val_t energy = current_energy[element];

    // These are exactly the four corner material assignments used by the
    // original mesh construction.
    val_t total_flux = 0.0;
    if ((global_row == 0 && col == 0) ||
        (global_row == n_cols - 1 && col == n_cols - 1)) {
        total_flux = 0.5;
    } else if ((global_row == 0 && col == n_cols - 1) ||
               (global_row == n_cols - 1 && col == 0)) {
        total_flux = -0.5;
    }

    if (global_row + 1 < n_cols) {
        total_flux += (current_energy[element + n_cols] - energy) *
                      TRANSFER_COEFF * CONNECTION_FLUX * FLUX_SCALE;
    }
    if (global_row > 0) {
        total_flux += (current_energy[element - n_cols] - energy) *
                      TRANSFER_COEFF * CONNECTION_FLUX * FLUX_SCALE;
    }
    if (col + 1 < n_cols) {
        total_flux += (current_energy[element + 1] - energy) *
                      TRANSFER_COEFF * CONNECTION_FLUX * FLUX_SCALE;
    }
    if (col > 0) {
        total_flux += (current_energy[element - 1] - energy) *
                      TRANSFER_COEFF * CONNECTION_FLUX * FLUX_SCALE;
    }

    next_energy[element] = energy + total_flux;
    next_flux[element] = current_flux[element] + fabs(total_flux);
}

void launchRows(const val_t* current_energy, const val_t* current_flux,
                val_t* next_energy, val_t* next_flux, const int n_cols,
                const int global_first_row, const int first_local_row,
                const int rows_to_update, cudaStream_t stream) {
    if (rows_to_update == 0) {
        return;
    }

    constexpr dim3 block{32, 8, 1};
    const dim3 grid{static_cast<unsigned int>((n_cols + block.x - 1) / block.x),
                    static_cast<unsigned int>((rows_to_update + block.y - 1) / block.y), 1};
    updateRows<<<grid, block, 0, stream>>>(current_energy, current_flux, next_energy, next_flux,
                                           n_cols, global_first_row, first_local_row, rows_to_update);
    CUDA_CHECK(cudaGetLastError());
}

struct DeviceBuffers {
    val_t* energy[2]{};
    val_t* flux[2]{};

    DeviceBuffers(const int local_rows, const int n_cols)
    {
        const size_t bytes = (static_cast<size_t>(local_rows) + 2U) * n_cols * sizeof(val_t);
        CUDA_CHECK(cudaMalloc(&energy[0], bytes));
        CUDA_CHECK(cudaMalloc(&energy[1], bytes));
        CUDA_CHECK(cudaMalloc(&flux[0], bytes));
        CUDA_CHECK(cudaMalloc(&flux[1], bytes));
    }

    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    ~DeviceBuffers() {
        // Destructors run only after the final device synchronization.
        if (energy[0] != nullptr) cudaFree(energy[0]);
        if (energy[1] != nullptr) cudaFree(energy[1]);
        if (flux[0] != nullptr) cudaFree(flux[0]);
        if (flux[1] != nullptr) cudaFree(flux[1]);
    }
};

struct PinnedHalos {
    val_t* send_top{};
    val_t* send_bottom{};
    val_t* receive_top{};
    val_t* receive_bottom{};

    explicit PinnedHalos(const int n_cols) {
        const size_t bytes = static_cast<size_t>(n_cols) * sizeof(val_t);
        CUDA_CHECK(cudaHostAlloc(&send_top, bytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&send_bottom, bytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&receive_top, bytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&receive_bottom, bytes, cudaHostAllocPortable));

        // OpenMP performs the host-local initialization and first touch.  Its
        // use is unconditional; MPI calls remain on the master thread.
#pragma omp parallel for schedule(static)
        for (int col = 0; col < n_cols; ++col) {
            send_top[col] = 0.0;
            send_bottom[col] = 0.0;
            receive_top[col] = 0.0;
            receive_bottom[col] = 0.0;
        }
    }

    PinnedHalos(const PinnedHalos&) = delete;
    PinnedHalos& operator=(const PinnedHalos&) = delete;

    ~PinnedHalos() {
        if (send_top != nullptr) cudaFreeHost(send_top);
        if (send_bottom != nullptr) cudaFreeHost(send_bottom);
        if (receive_top != nullptr) cudaFreeHost(receive_top);
        if (receive_bottom != nullptr) cudaFreeHost(receive_bottom);
    }
};

bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) {
        std::printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > energy_epsilon) {
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(flux_sum)) {
        std::printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    std::printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t* energy = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* flux = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*energy + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*flux + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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

int main(int argc, char** argv) {
    int mpi_thread_level = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_level));
    if (mpi_thread_level < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool argument_error = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            argument_error = true;
        }
    }

    if (n_elems_root <= 0 || n_iters < 0) {
        argument_error = true;
    }
    int any_argument_error = argument_error ? 1 : 0;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &any_argument_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (any_argument_error != 0) {
        if (world_rank == 0) {
            std::printf("Invalid or unknown option\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    // Ranks with no rows cannot participate in a halo exchange.  They still
    // initialize and finalize MPI, while the active subcommunicator runs the
    // accelerator algorithm.
    const int active_size = std::min(world_size, n_elems_root);
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 1 : MPI_UNDEFINED,
                             world_rank, &active_comm));
    if (active_comm == MPI_COMM_NULL) {
        MPI_CHECK(MPI_Finalize());
        return 0;
    }

    int rank = 0;
    int rank_count = 1;
    MPI_CHECK(MPI_Comm_rank(active_comm, &rank));
    MPI_CHECK(MPI_Comm_size(active_comm, &rank_count));

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(active_comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(active_comm, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    const int base_rows = n_elems_root / rank_count;
    const int extra_rows = n_elems_root % rank_count;
    const int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int global_first_row = rank * base_rows + std::min(rank, extra_rows);
    const size_t local_elements = static_cast<size_t>(local_rows) * n_elems_root;
    const int local_element_count = static_cast<int>(local_elements);

    if (rank == 0) {
        const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
        const size_t device_bytes = 4U * (static_cast<size_t>(local_rows) + 2U) * n_elems_root * sizeof(val_t);
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d active of %d launched; OpenMP max threads/rank: %d\n",
                    rank_count, world_size, omp_get_max_threads());
        std::printf("\nBuilding distributed unstructured mesh...\n");
        std::printf("Device state per rank: %.2f MB (four SoA buffers plus halos)\n\n",
                    device_bytes / (1024.0 * 1024.0));
    }

    DeviceBuffers device(local_rows, n_elems_root);
    PinnedHalos halos(n_elems_root);
    cudaStream_t compute_stream{};
    cudaStream_t copy_stream{};
    cudaEvent_t halos_ready{};
    cudaEvent_t computation_done{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&computation_done, cudaEventDisableTiming));

    // Initialize local physical rows with OpenMP, then upload them once.  The
    // GPU owns all state during the timed stencil iterations.
    std::vector<val_t> local_energy(local_elements);
    std::vector<val_t> local_flux(local_elements);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_elements; ++i) {
        local_energy[i] = 0.0;
        local_flux[i] = 0.0;
    }
    const size_t local_bytes = local_elements * sizeof(val_t);
    CUDA_CHECK(cudaMemcpyAsync(device.energy[0] + n_elems_root, local_energy.data(), local_bytes,
                               cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(device.flux[0] + n_elems_root, local_flux.data(), local_bytes,
                               cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));

    val_t* current_energy = device.energy[0];
    val_t* next_energy = device.energy[1];
    val_t* current_flux = device.flux[0];
    val_t* next_flux = device.flux[1];
    const int previous_rank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next_rank = rank + 1 == rank_count ? MPI_PROC_NULL : rank + 1;
    const bool has_previous = previous_rank != MPI_PROC_NULL;
    const bool has_next = next_rank != MPI_PROC_NULL;
    bool have_completed_step = false;

    if (rank == 0) {
        std::printf("Running simulation...\n");
    }
    MPI_CHECK(MPI_Barrier(active_comm));
    const auto start = std::chrono::steady_clock::now();

    for (int iteration = 0; iteration < n_iters; ++iteration) {
        // Stage outgoing rows through pinned host memory.  This is portable to
        // both CUDA-aware and conventional MPI implementations and leaves the
        // interior GPU work free to overlap the MPI transfer.
        if (have_completed_step) {
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, computation_done, 0));
        }
        if (has_previous) {
            CUDA_CHECK(cudaMemcpyAsync(halos.send_top, current_energy + n_elems_root,
                                       static_cast<size_t>(n_elems_root) * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, copy_stream));
        }
        if (has_next) {
            CUDA_CHECK(cudaMemcpyAsync(halos.send_bottom,
                                       current_energy + static_cast<size_t>(local_rows) * n_elems_root,
                                       static_cast<size_t>(n_elems_root) * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, copy_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));

        MPI_Request requests[4];
        int request_count = 0;
        if (has_previous) {
            MPI_CHECK(MPI_Irecv(halos.receive_top, n_elems_root, MPI_DOUBLE, previous_rank, 101,
                                active_comm, &requests[request_count++]));
            MPI_CHECK(MPI_Isend(halos.send_top, n_elems_root, MPI_DOUBLE, previous_rank, 102,
                                active_comm, &requests[request_count++]));
        }
        if (has_next) {
            MPI_CHECK(MPI_Irecv(halos.receive_bottom, n_elems_root, MPI_DOUBLE, next_rank, 102,
                                active_comm, &requests[request_count++]));
            MPI_CHECK(MPI_Isend(halos.send_bottom, n_elems_root, MPI_DOUBLE, next_rank, 101,
                                active_comm, &requests[request_count++]));
        }

        // Rows not adjacent to an MPI boundary only reference local data, so
        // they execute while the halo messages are in flight.
        const int inner_first = has_previous ? 1 : 0;
        const int inner_last = local_rows - (has_next ? 1 : 0);
        launchRows(current_energy, current_flux, next_energy, next_flux, n_elems_root,
                   global_first_row, inner_first, inner_last - inner_first, compute_stream);

        if (request_count != 0) {
            MPI_CHECK(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
        }
        if (has_previous) {
            CUDA_CHECK(cudaMemcpyAsync(current_energy, halos.receive_top,
                                       static_cast<size_t>(n_elems_root) * sizeof(val_t),
                                       cudaMemcpyHostToDevice, copy_stream));
        }
        if (has_next) {
            CUDA_CHECK(cudaMemcpyAsync(current_energy + static_cast<size_t>(local_rows + 1) * n_elems_root,
                                       halos.receive_bottom,
                                       static_cast<size_t>(n_elems_root) * sizeof(val_t),
                                       cudaMemcpyHostToDevice, copy_stream));
        }
        CUDA_CHECK(cudaEventRecord(halos_ready, copy_stream));
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, halos_ready, 0));

        if (has_previous) {
            launchRows(current_energy, current_flux, next_energy, next_flux, n_elems_root,
                       global_first_row, 0, 1, compute_stream);
        }
        if (has_next) {
            launchRows(current_energy, current_flux, next_energy, next_flux, n_elems_root,
                       global_first_row, local_rows - 1, 1, compute_stream);
        }
        CUDA_CHECK(cudaEventRecord(computation_done, compute_stream));
        have_completed_step = true;
        std::swap(current_energy, next_energy);
        std::swap(current_flux, next_flux);
    }

    if (have_completed_step) {
        CUDA_CHECK(cudaEventSynchronize(computation_done));
    }
    MPI_CHECK(MPI_Barrier(active_comm));
    const auto end = std::chrono::steady_clock::now();
    const auto local_duration_ms = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long long duration_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, active_comm));

    // Collect the final distributed state in global row order.  This is only
    // needed for the existing root-only validation/result protocol.
    if (have_completed_step) {
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, computation_done, 0));
    }
    CUDA_CHECK(cudaMemcpyAsync(local_energy.data(), current_energy + n_elems_root, local_bytes,
                               cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(local_flux.data(), current_flux + n_elems_root, local_bytes,
                               cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));

    std::vector<int> receive_counts;
    std::vector<int> displacements;
    std::vector<val_t> global_energy;
    std::vector<val_t> global_flux;
    if (rank == 0) {
        receive_counts.resize(rank_count);
        displacements.resize(rank_count);
        for (int r = 0; r < rank_count; ++r) {
            const int rows = base_rows + (r < extra_rows ? 1 : 0);
            receive_counts[r] = rows * n_elems_root;
            displacements[r] = (r * base_rows + std::min(r, extra_rows)) * n_elems_root;
        }
        const size_t global_elements = static_cast<size_t>(n_elems_root) * n_elems_root;
        global_energy.resize(global_elements);
        global_flux.resize(global_elements);
    }
    MPI_CHECK(MPI_Gatherv(local_energy.data(), local_element_count, MPI_DOUBLE,
                          rank == 0 ? global_energy.data() : nullptr,
                          rank == 0 ? receive_counts.data() : nullptr,
                          rank == 0 ? displacements.data() : nullptr,
                          MPI_DOUBLE, 0, active_comm));
    MPI_CHECK(MPI_Gatherv(local_flux.data(), local_element_count, MPI_DOUBLE,
                          rank == 0 ? global_flux.data() : nullptr,
                          rank == 0 ? receive_counts.data() : nullptr,
                          rank == 0 ? displacements.data() : nullptr,
                          MPI_DOUBLE, 0, active_comm));

    int result_is_valid = 1;
    if (rank == 0) {
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) *
                                           static_cast<double>(n_elems_root) * n_elems_root) /
                                          (static_cast<double>(duration_ms) / 1000.0) / 1e9;
        std::printf("Computation time: %lld ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);

        std::vector<ElementDynamic> global_state(global_energy.size());
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < global_state.size(); ++i) {
            global_state[i].current_energy = global_energy[i];
            global_state[i].total_flux = global_flux[i];
        }
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(computeHash(global_state)));

        if (print_results_requested) {
            print_results(global_energy, "ElementEnergy");
        }
        if (validate && !validateResults(global_state)) {
            result_is_valid = 0;
        }
    }
    MPI_CHECK(MPI_Bcast(&result_is_valid, 1, MPI_INT, 0, active_comm));

    CUDA_CHECK(cudaEventDestroy(computation_done));
    CUDA_CHECK(cudaEventDestroy(halos_ready));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    MPI_CHECK(MPI_Comm_free(&active_comm));
    MPI_CHECK(MPI_Finalize());
    return result_is_valid == 0 ? 1 : 0;
}
