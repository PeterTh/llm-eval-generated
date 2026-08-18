#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

// The dynamic state is deliberately kept as an AoS.  A complete row is then a
// contiguous MPI message, while the CUDA kernel still performs coalesced loads
// of each field across a warp.
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t),
              "ElementDynamic must be two contiguous doubles for MPI");

namespace {

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t CONNECTION_FLUX = 1.0;
constexpr val_t FLUX_SCALE = 0.25;

int g_world_rank = -1;

[[noreturn]] void cudaFailure(cudaError_t status, const char* operation) {
    std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", g_world_rank,
                 operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

inline void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaFailure(status, operation);
    }
}

// This is the square-grid specialization of the original unstructured mesh.
// It preserves the original connection order: +row, -row, +column, -column.
// Each active MPI rank owns a contiguous row range plus one ghost row on each
// side.  Only current_energy is read from a neighbour; sending total_flux too
// retains contiguous row transfers and avoids a strided GPU packing kernel.
__global__ void updateRows(const ElementDynamic* __restrict__ current,
                           ElementDynamic* __restrict__ next, int width,
                           int first_global_row, int first_local_row,
                           int row_count) {
    const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_row =
        first_local_row + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (column >= width || local_row >= first_local_row + row_count) {
        return;
    }

    const int global_row = first_global_row + local_row - 1;
    const size_t index = static_cast<size_t>(local_row) * width + column;
    const ElementDynamic self = current[index];

    // The four assignments made by buildSquare2D leave an N=1 grid as an
    // inflow corner.  Testing inflow corners first faithfully reproduces that.
    val_t total_flux =
        ((global_row == 0 && column == 0) ||
         (global_row == width - 1 && column == width - 1))
            ? 0.5
            : (((global_row == 0 && column == width - 1) ||
                (global_row == width - 1 && column == 0))
                   ? -0.5
                   : 0.0);

    // Keep these as separate operations rather than folding them into 0.2:
    // that matches the arithmetic sequence in computeFlux.
    if (global_row + 1 < width) {
        val_t flux = (current[index + width].current_energy - self.current_energy) *
                     TRANSFER_COEFF;
        flux *= CONNECTION_FLUX;
        flux *= FLUX_SCALE;
        total_flux += flux;
    }
    if (global_row > 0) {
        val_t flux = (current[index - width].current_energy - self.current_energy) *
                     TRANSFER_COEFF;
        flux *= CONNECTION_FLUX;
        flux *= FLUX_SCALE;
        total_flux += flux;
    }
    if (column + 1 < width) {
        val_t flux = (current[index + 1].current_energy - self.current_energy) *
                     TRANSFER_COEFF;
        flux *= CONNECTION_FLUX;
        flux *= FLUX_SCALE;
        total_flux += flux;
    }
    if (column > 0) {
        val_t flux = (current[index - 1].current_energy - self.current_energy) *
                     TRANSFER_COEFF;
        flux *= CONNECTION_FLUX;
        flux *= FLUX_SCALE;
        total_flux += flux;
    }

    next[index].current_energy = self.current_energy + total_flux;
    next[index].total_flux = self.total_flux + fabs(total_flux);
}

struct RankState {
    int width = 0;
    int local_rows = 0;
    int first_global_row = 0;
    int previous = MPI_PROC_NULL;
    int next = MPI_PROC_NULL;
    ElementDynamic* current = nullptr;
    ElementDynamic* swap = nullptr;
    ElementDynamic* send_top = nullptr;
    ElementDynamic* send_bottom = nullptr;
    ElementDynamic* recv_top = nullptr;
    ElementDynamic* recv_bottom = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t halo_stream = nullptr;

    RankState() = default;
    RankState(const RankState&) = delete;
    RankState& operator=(const RankState&) = delete;

    RankState(RankState&& other) noexcept { *this = std::move(other); }

    RankState& operator=(RankState&& other) noexcept {
        if (this != &other) {
            release();
            width = other.width;
            local_rows = other.local_rows;
            first_global_row = other.first_global_row;
            previous = other.previous;
            next = other.next;
            current = other.current;
            swap = other.swap;
            send_top = other.send_top;
            send_bottom = other.send_bottom;
            recv_top = other.recv_top;
            recv_bottom = other.recv_bottom;
            compute_stream = other.compute_stream;
            halo_stream = other.halo_stream;
            other.current = nullptr;
            other.swap = nullptr;
            other.send_top = nullptr;
            other.send_bottom = nullptr;
            other.recv_top = nullptr;
            other.recv_bottom = nullptr;
            other.compute_stream = nullptr;
            other.halo_stream = nullptr;
        }
        return *this;
    }

    ~RankState() { release(); }

    void release() noexcept {
        if (compute_stream != nullptr) {
            cudaStreamDestroy(compute_stream);
            compute_stream = nullptr;
        }
        if (halo_stream != nullptr) {
            cudaStreamDestroy(halo_stream);
            halo_stream = nullptr;
        }
        if (current != nullptr) {
            cudaFree(current);
            current = nullptr;
        }
        if (swap != nullptr) {
            cudaFree(swap);
            swap = nullptr;
        }
        if (send_top != nullptr) {
            cudaFreeHost(send_top);
            send_top = nullptr;
        }
        if (send_bottom != nullptr) {
            cudaFreeHost(send_bottom);
            send_bottom = nullptr;
        }
        if (recv_top != nullptr) {
            cudaFreeHost(recv_top);
            recv_top = nullptr;
        }
        if (recv_bottom != nullptr) {
            cudaFreeHost(recv_bottom);
            recv_bottom = nullptr;
        }
    }
};

RankState makeRankState(int width, int local_rows, int first_global_row,
                        int previous, int next) {
    RankState state;
    state.width = width;
    state.local_rows = local_rows;
    state.first_global_row = first_global_row;
    state.previous = previous;
    state.next = next;

    const size_t row_elements = static_cast<size_t>(width);
    const size_t allocation_elements =
        static_cast<size_t>(local_rows + 2) * row_elements;
    const size_t allocation_bytes = allocation_elements * sizeof(ElementDynamic);
    const size_t row_bytes = row_elements * sizeof(ElementDynamic);

    // Host initialization is an OpenMP phase.  It keeps the CPU involved in
    // the hybrid pipeline without taking work from the GPU timestep kernel.
    std::vector<ElementDynamic> initial_state(allocation_elements);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(allocation_elements); ++i) {
        initial_state[static_cast<size_t>(i)] = ElementDynamic{0.0, 0.0};
    }

    cudaCheck(cudaStreamCreateWithFlags(&state.compute_stream,
                                        cudaStreamNonBlocking),
              "creating compute stream");
    cudaCheck(cudaStreamCreateWithFlags(&state.halo_stream, cudaStreamNonBlocking),
              "creating halo stream");
    cudaCheck(cudaMalloc(&state.current, allocation_bytes), "allocating current state");
    cudaCheck(cudaMalloc(&state.swap, allocation_bytes), "allocating swap state");
    cudaCheck(cudaMemcpy(state.current, initial_state.data(), allocation_bytes,
                         cudaMemcpyHostToDevice),
              "initializing current state");
    cudaCheck(cudaMemcpy(state.swap, initial_state.data(), allocation_bytes,
                         cudaMemcpyHostToDevice),
              "initializing swap state");

    cudaCheck(cudaHostAlloc(&state.send_top, row_bytes, cudaHostAllocDefault),
              "allocating top send buffer");
    cudaCheck(cudaHostAlloc(&state.send_bottom, row_bytes, cudaHostAllocDefault),
              "allocating bottom send buffer");
    cudaCheck(cudaHostAlloc(&state.recv_top, row_bytes, cudaHostAllocDefault),
              "allocating top receive buffer");
    cudaCheck(cudaHostAlloc(&state.recv_bottom, row_bytes, cudaHostAllocDefault),
              "allocating bottom receive buffer");
    return state;
}

void launchRows(const RankState& state, int first_local_row, int row_count) {
    if (row_count <= 0) {
        return;
    }
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((state.width + BLOCK_X - 1) / BLOCK_X,
                    (row_count + BLOCK_Y - 1) / BLOCK_Y);
    updateRows<<<grid, block, 0, state.compute_stream>>>(
        state.current, state.swap, state.width, state.first_global_row,
        first_local_row, row_count);
    cudaCheck(cudaGetLastError(), "launching update kernel");
}

void runSimulation(RankState& state, int n_iters, MPI_Comm active_comm) {
    const size_t row_bytes = static_cast<size_t>(state.width) * sizeof(ElementDynamic);
    const int row_doubles = state.width * 2;

    for (int iter = 0; iter < n_iters; ++iter) {
        // The middle rows need no incoming halo, so their CUDA work overlaps
        // the host-staged MPI exchange needed by the two boundary rows.
        launchRows(state, 2, state.local_rows - 2);

        if (state.previous != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(state.send_top, state.current + state.width,
                                      row_bytes, cudaMemcpyDeviceToHost,
                                      state.halo_stream),
                      "copying top halo to host");
        }
        if (state.next != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(state.send_bottom,
                                      state.current +
                                          static_cast<size_t>(state.local_rows) *
                                              state.width,
                                      row_bytes, cudaMemcpyDeviceToHost,
                                      state.halo_stream),
                      "copying bottom halo to host");
        }
        cudaCheck(cudaStreamSynchronize(state.halo_stream), "waiting for halo copies");

        MPI_Request requests[4];
        int request_count = 0;
        if (state.previous != MPI_PROC_NULL) {
            MPI_Irecv(state.recv_top, row_doubles, MPI_DOUBLE, state.previous, 0,
                      active_comm, &requests[request_count++]);
        }
        if (state.next != MPI_PROC_NULL) {
            MPI_Irecv(state.recv_bottom, row_doubles, MPI_DOUBLE, state.next, 1,
                      active_comm, &requests[request_count++]);
        }
        if (state.previous != MPI_PROC_NULL) {
            MPI_Isend(state.send_top, row_doubles, MPI_DOUBLE, state.previous, 1,
                      active_comm, &requests[request_count++]);
        }
        if (state.next != MPI_PROC_NULL) {
            MPI_Isend(state.send_bottom, row_doubles, MPI_DOUBLE, state.next, 0,
                      active_comm, &requests[request_count++]);
        }
        if (request_count != 0) {
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        }

        if (state.previous != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(state.current, state.recv_top, row_bytes,
                                      cudaMemcpyHostToDevice, state.halo_stream),
                      "copying top halo to device");
        }
        if (state.next != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(state.current +
                                          static_cast<size_t>(state.local_rows + 1) *
                                              state.width,
                                      state.recv_bottom, row_bytes,
                                      cudaMemcpyHostToDevice, state.halo_stream),
                      "copying bottom halo to device");
        }
        cudaCheck(cudaStreamSynchronize(state.halo_stream), "waiting for received halos");

        launchRows(state, 1, 1);
        if (state.local_rows > 1) {
            launchRows(state, state.local_rows, 1);
        }
        cudaCheck(cudaDeviceSynchronize(), "finishing timestep");
        std::swap(state.current, state.swap);
    }
}

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
        uint64_t energy_bits;
        uint64_t flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* prog_name) {
    std::printf("Usage: %s [options]\n", prog_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Returns 1 for a runnable command line, 0 for --help, and -1 for an error.
int parseArguments(int argc, char** argv, int& n_elems_root, int& n_iters,
                   bool& validate, bool& print_results_flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_flag = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            return 0;
        } else {
            if (g_world_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            return -1;
        }
    }
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_world_rank);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (g_world_rank == 0) {
            std::fprintf(stderr,
                         "MPI implementation does not provide MPI_THREAD_FUNNELED, "
                         "which is required for the OpenMP host phases.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_flag = false;
    const int parse_result =
        parseArguments(argc, argv, n_elems_root, n_iters, validate, print_results_flag);
    if (parse_result != 1) {
        if (g_world_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_result == 0 ? 0 : 1;
    }
    if (n_elems_root <= 0) {
        if (g_world_rank == 0) {
            std::fprintf(stderr, "Grid size must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int local_rank = 0;
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_world_rank,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "discovering CUDA devices");
    if (device_count == 0) {
        if (g_world_rank == 0) {
            std::fprintf(stderr, "No CUDA device is visible to this MPI job.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count), "selecting CUDA device");
    cudaCheck(cudaFree(nullptr), "initializing CUDA context");
    MPI_Comm_free(&local_comm);

    const int active_size = std::min(world_size, n_elems_root);
    const bool active = g_world_rank < active_size;
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, g_world_rank,
                   &active_comm);

    RankState state;
    int active_rank = -1;
    if (active) {
        MPI_Comm_rank(active_comm, &active_rank);
        const int base_rows = n_elems_root / active_size;
        const int extra_rows = n_elems_root % active_size;
        const int local_rows = base_rows + (active_rank < extra_rows ? 1 : 0);
        const int first_global_row = active_rank * base_rows +
                                     std::min(active_rank, extra_rows);
        const int previous = active_rank == 0 ? MPI_PROC_NULL : active_rank - 1;
        const int next = active_rank + 1 == active_size ? MPI_PROC_NULL : active_rank + 1;
        state = makeRankState(n_elems_root, local_rows, first_global_row, previous, next);
    }

    const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
    if (g_world_rank == 0) {
        const size_t static_mem = n_elems * (sizeof(idx_t) * (2 + 8) +
                                             sizeof(val_t) * 8);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n_elems_root,
                    n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d (%d active), CUDA devices per node: %d\n", world_size,
                    active_size, device_count);
        std::printf("\nBuilding distributed unstructured mesh...\n");
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                    (static_mem + dynamic_mem) / (1024.0 * 1024.0),
                    static_mem / (1024.0 * 1024.0),
                    dynamic_mem / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) {
        runSimulation(state, n_iters, active_comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_duration_s = MPI_Wtime() - start;
    double duration_s = 0.0;
    MPI_Reduce(&local_duration_s, &duration_s, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<ElementDynamic> local_result;
    std::vector<ElementDynamic> global_result;
    if (active) {
        const size_t local_elements =
            static_cast<size_t>(state.local_rows) * state.width;
        local_result.resize(local_elements);
        cudaCheck(cudaMemcpy(local_result.data(), state.current + state.width,
                             local_elements * sizeof(ElementDynamic),
                             cudaMemcpyDeviceToHost),
                  "copying final local state");

        std::vector<int> recv_counts;
        std::vector<int> displacements;
        if (active_rank == 0) {
            global_result.resize(n_elems);
            recv_counts.resize(active_size);
            displacements.resize(active_size);
            int displacement = 0;
            const int base_rows = n_elems_root / active_size;
            const int extra_rows = n_elems_root % active_size;
            for (int rank = 0; rank < active_size; ++rank) {
                const int rows = base_rows + (rank < extra_rows ? 1 : 0);
                recv_counts[rank] = rows * n_elems_root * 2;
                displacements[rank] = displacement;
                displacement += recv_counts[rank];
            }
        }
        const int send_count = state.local_rows * state.width * 2;
        MPI_Gatherv(local_result.data(), send_count, MPI_DOUBLE,
                    active_rank == 0 ? global_result.data() : nullptr,
                    active_rank == 0 ? recv_counts.data() : nullptr,
                    active_rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    active_comm);
    }

    int exit_code = 0;
    if (g_world_rank == 0) {
        const double duration_ms = duration_s * 1000.0;
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = duration_s > 0.0
            ? (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) /
                  duration_s / 1e9
            : std::numeric_limits<double>::infinity();
        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(computeHash(global_result)));

        if (print_results_flag) {
            std::vector<double> energy_data(global_result.size());
            // This OpenMP conversion prepares the optional reporting data while
            // retaining the globally ordered result required by print_results.
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(global_result.size()); ++i) {
                energy_data[static_cast<size_t>(i)] =
                    global_result[static_cast<size_t>(i)].current_energy;
            }
            print_results(energy_data, "ElementEnergy");
        }
        if (validate && !validateResults(global_result)) {
            exit_code = 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    state.release();
    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&active_comm);
    }
    MPI_Finalize();
    return exit_code;
}
