#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

namespace {

constexpr val_t TRANSFER = 0.8 * 0.25;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", rank, operation,
                 cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 2);
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__device__ __forceinline__ val_t externalFlow(int global_x, int y, int n) {
    // These tests reproduce the assignment order in buildSquare2D, including n=1.
    if ((global_x == 0 && y == 0) ||
        (global_x == n - 1 && y == n - 1)) return 0.5;
    if ((global_x == 0 && y == n - 1) ||
        (global_x == n - 1 && y == 0)) return -0.5;
    return 0.0;
}

__device__ __forceinline__ void updateElement(const val_t* __restrict__ energy,
                                               const val_t* __restrict__ accumulated_flux,
                                               val_t* __restrict__ next_energy,
                                               val_t* __restrict__ next_flux,
                                               int local_x, int global_x, int y,
                                               int n) {
    const size_t pitch = static_cast<size_t>(n);
    const size_t p = static_cast<size_t>(local_x) * pitch + y;
    const val_t center = energy[p];
    val_t flux = externalFlow(global_x, y, n);

    // Keep the reference neighbor order: +x, -x, +y, -y.
    // Explicit round-to-nearest operations retain the reference's non-contracted
    // arithmetic while allowing contraction elsewhere in the CUDA translation unit.
    if (global_x + 1 < n)
        flux = __dadd_rn(flux, __dmul_rn(energy[p + pitch] - center, TRANSFER));
    if (global_x > 0)
        flux = __dadd_rn(flux, __dmul_rn(energy[p - pitch] - center, TRANSFER));
    if (y + 1 < n)
        flux = __dadd_rn(flux, __dmul_rn(energy[p + 1] - center, TRANSFER));
    if (y > 0)
        flux = __dadd_rn(flux, __dmul_rn(energy[p - 1] - center, TRANSFER));

    next_energy[p] = __dadd_rn(center, flux);
    next_flux[p] = __dadd_rn(accumulated_flux[p], fabs(flux));
}

__global__ void updateRows(const val_t* __restrict__ energy,
                           const val_t* __restrict__ accumulated_flux,
                           val_t* __restrict__ next_energy,
                           val_t* __restrict__ next_flux,
                           int global_row_begin, int first_local_row,
                           int row_count, int n) {
    const int y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row_offset = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int local_x = first_local_row + row_offset;
    if (y < n && row_offset < row_count)
        updateElement(energy, accumulated_flux, next_energy, next_flux,
                      local_x, global_row_begin + local_x - 1, y, n);
}

__global__ void updateBoundaries(const val_t* __restrict__ energy,
                                 const val_t* __restrict__ accumulated_flux,
                                 val_t* __restrict__ next_energy,
                                 val_t* __restrict__ next_flux,
                                 int global_row_begin, int local_rows, int n) {
    const int y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (y >= n || local_rows == 0) return;
    const int local_x = (blockIdx.y == 0) ? 1 : local_rows;
    updateElement(energy, accumulated_flux, next_energy, next_flux,
                  local_x, global_row_begin + local_x - 1, y, n);
}

struct DistributedState {
    int rank = 0;
    int ranks = 1;
    int global_row_begin = 0;
    int local_rows = 0;
    size_t owned_count = 0;
    size_t padded_count = 0;
    val_t* energy[2]{nullptr, nullptr};
    val_t* flux[2]{nullptr, nullptr};
    val_t* send_top = nullptr;
    val_t* send_bottom = nullptr;
    val_t* recv_top = nullptr;
    val_t* recv_bottom = nullptr;
    cudaStream_t compute_stream{};
    cudaStream_t communication_stream{};
    cudaEvent_t interior_done{};
    cudaEvent_t iteration_done{};
};

void partitionRows(int n, int rank, int ranks, int& begin, int& rows) {
    const int base = n / ranks;
    const int extra = n % ranks;
    rows = base + (rank < extra ? 1 : 0);
    begin = rank * base + std::min(rank, extra);
}

void initializeState(DistributedState& state, int n) {
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.ranks);
    partitionRows(n, state.rank, state.ranks, state.global_row_begin, state.local_rows);
    state.owned_count = static_cast<size_t>(state.local_rows) * n;
    state.padded_count = static_cast<size_t>(state.local_rows + 2) * n;

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (state.rank == 0) std::fprintf(stderr, "No CUDA devices are visible.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&state.energy[b], state.padded_count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&state.flux[b], state.padded_count * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(state.energy[b], 0, state.padded_count * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(state.flux[b], 0, state.padded_count * sizeof(val_t)));
    }
    CUDA_CHECK(cudaHostAlloc(&state.send_top, static_cast<size_t>(n) * sizeof(val_t),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&state.send_bottom, static_cast<size_t>(n) * sizeof(val_t),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&state.recv_top, static_cast<size_t>(n) * sizeof(val_t),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&state.recv_bottom, static_cast<size_t>(n) * sizeof(val_t),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaStreamCreateWithFlags(&state.compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&state.communication_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&state.interior_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&state.iteration_done, cudaEventDisableTiming));

    // OpenMP initializes the rank-local mesh state; CUDA owns it after this copy.
    std::vector<val_t> initial_energy(state.owned_count);
    std::vector<val_t> initial_flux(state.owned_count);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.owned_count); ++i) {
        initial_energy[static_cast<size_t>(i)] = 0.0;
        initial_flux[static_cast<size_t>(i)] = 0.0;
    }
    if (state.owned_count != 0) {
        CUDA_CHECK(cudaMemcpy(state.energy[0] + n, initial_energy.data(),
                              state.owned_count * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.flux[0] + n, initial_flux.data(),
                              state.owned_count * sizeof(val_t), cudaMemcpyHostToDevice));
    }
}

void destroyState(DistributedState& state) {
    CUDA_CHECK(cudaEventDestroy(state.iteration_done));
    CUDA_CHECK(cudaEventDestroy(state.interior_done));
    CUDA_CHECK(cudaStreamDestroy(state.communication_stream));
    CUDA_CHECK(cudaStreamDestroy(state.compute_stream));
    CUDA_CHECK(cudaFreeHost(state.recv_bottom));
    CUDA_CHECK(cudaFreeHost(state.recv_top));
    CUDA_CHECK(cudaFreeHost(state.send_bottom));
    CUDA_CHECK(cudaFreeHost(state.send_top));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(state.flux[b]));
        CUDA_CHECK(cudaFree(state.energy[b]));
    }
}

void runSimulation(DistributedState& state, int n, int iterations) {
    if (state.local_rows == 0 || iterations == 0) return;
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(val_t);
    const int active_ranks = std::min(n, state.ranks);
    const dim3 interior_block(32, 8);

    // The one-rank case needs neither halo staging nor a separate boundary kernel.
    if (active_ranks == 1) {
        const dim3 grid((n + interior_block.x - 1) / interior_block.x,
                        (state.local_rows + interior_block.y - 1) / interior_block.y);
        int current = 0;
        for (int iter = 0; iter < iterations; ++iter) {
            const int next = 1 - current;
            updateRows<<<grid, interior_block, 0, state.compute_stream>>>(
                state.energy[current], state.flux[current], state.energy[next], state.flux[next],
                state.global_row_begin, 1, state.local_rows, n);
            CUDA_CHECK(cudaGetLastError());
            current = next;
        }
        CUDA_CHECK(cudaStreamSynchronize(state.compute_stream));
        if (current != 0) {
            std::swap(state.energy[0], state.energy[1]);
            std::swap(state.flux[0], state.flux[1]);
        }
        return;
    }

    const int up = state.global_row_begin > 0 ? state.rank - 1 : MPI_PROC_NULL;
    const int down = (state.rank + 1 < active_ranks) ? state.rank + 1 : MPI_PROC_NULL;
    const dim3 interior_grid((n + interior_block.x - 1) / interior_block.x,
                             state.local_rows > 2
                                 ? (state.local_rows - 2 + interior_block.y - 1) / interior_block.y
                                 : 0);
    const dim3 boundary_block(256, 1);
    const dim3 boundary_grid((n + boundary_block.x - 1) / boundary_block.x,
                             state.local_rows == 1 ? 1 : 2);

    int current = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        const int next = 1 - current;
        if (iter != 0)
            CUDA_CHECK(cudaStreamWaitEvent(state.compute_stream, state.iteration_done, 0));

        // Stage only the two halo rows. Interior GPU work proceeds concurrently.
        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(state.send_top, state.energy[current] + n,
                                       row_bytes, cudaMemcpyDeviceToHost,
                                       state.communication_stream));
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(state.send_bottom,
                                       state.energy[current] +
                                           static_cast<size_t>(state.local_rows) * n,
                                       row_bytes, cudaMemcpyDeviceToHost,
                                       state.communication_stream));

        if (state.local_rows > 2) {
            updateRows<<<interior_grid, interior_block, 0, state.compute_stream>>>(
                state.energy[current], state.flux[current], state.energy[next], state.flux[next],
                state.global_row_begin, 2, state.local_rows - 2, n);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(state.interior_done, state.compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(state.communication_stream));

        MPI_Request requests[4];
        int request_count = 0;
        if (up != MPI_PROC_NULL) {
            MPI_Irecv(state.recv_top, n, MPI_DOUBLE, up, 101, MPI_COMM_WORLD,
                      &requests[request_count++]);
            MPI_Isend(state.send_top, n, MPI_DOUBLE, up, 102, MPI_COMM_WORLD,
                      &requests[request_count++]);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Irecv(state.recv_bottom, n, MPI_DOUBLE, down, 102, MPI_COMM_WORLD,
                      &requests[request_count++]);
            MPI_Isend(state.send_bottom, n, MPI_DOUBLE, down, 101, MPI_COMM_WORLD,
                      &requests[request_count++]);
        }
        if (request_count != 0)
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(state.energy[current], state.recv_top, row_bytes,
                                       cudaMemcpyHostToDevice, state.communication_stream));
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(
                state.energy[current] + static_cast<size_t>(state.local_rows + 1) * n,
                state.recv_bottom, row_bytes, cudaMemcpyHostToDevice,
                state.communication_stream));

        updateBoundaries<<<boundary_grid, boundary_block, 0, state.communication_stream>>>(
            state.energy[current], state.flux[current], state.energy[next], state.flux[next],
            state.global_row_begin, state.local_rows, n);
        CUDA_CHECK(cudaGetLastError());
        // Make one event represent completion of both streams without serializing the kernels.
        CUDA_CHECK(cudaStreamWaitEvent(state.communication_stream, state.interior_done, 0));
        CUDA_CHECK(cudaEventRecord(state.iteration_done, state.communication_stream));
        current = next;
    }
    CUDA_CHECK(cudaEventSynchronize(state.iteration_done));

    if (current != 0) {
        std::swap(state.energy[0], state.energy[1]);
        std::swap(state.flux[0], state.flux[1]);
    }
}

void gatherResults(const DistributedState& state, int n,
                   std::vector<val_t>& global_energy, std::vector<val_t>& global_flux) {
    std::vector<val_t> local_energy(state.owned_count);
    std::vector<val_t> local_flux(state.owned_count);
    if (state.owned_count != 0) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), state.energy[0] + n,
                              state.owned_count * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), state.flux[0] + n,
                              state.owned_count * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts(state.ranks), displacements(state.ranks);
    for (int r = 0; r < state.ranks; ++r) {
        int begin = 0, rows = 0;
        partitionRows(n, r, state.ranks, begin, rows);
        counts[r] = rows * n;
        displacements[r] = begin * n;
    }
    if (state.rank == 0) {
        global_energy.resize(static_cast<size_t>(n) * n);
        global_flux.resize(static_cast<size_t>(n) * n);
    }
    MPI_Gatherv(local_energy.data(), static_cast<int>(state.owned_count), MPI_DOUBLE,
                global_energy.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), static_cast<int>(state.owned_count), MPI_DOUBLE,
                global_flux.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    uint64_t result = 0;
#pragma omp parallel for reduction(^ : result) schedule(static)
    for (long long i = 0; i < static_cast<long long>(energy.size()); ++i) {
        uint64_t energy_bits = 0, flux_bits = 0;
        std::memcpy(&energy_bits, &energy[static_cast<size_t>(i)], sizeof(energy_bits));
        std::memcpy(&flux_bits, &flux[static_cast<size_t>(i)], sizeof(flux_bits));
        const uint64_t index = static_cast<uint64_t>(i);
        result ^= (energy_bits + index) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + index) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }
    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    bool valid = true;
    if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) ||
        !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: simulation produced a non-finite result\n");
        valid = false;
    }
    if (std::abs(energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    std::printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int options[5] = {512, 10, 0, 0, 1}; // n, iterations, validate, results, parse status
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) options[0] = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) options[1] = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-v") == 0) options[2] = 1;
            else if (std::strcmp(argv[i], "-r") == 0) options[3] = 1;
            else if (std::strcmp(argv[i], "-h") == 0) options[4] = 2;
            else {
                std::printf("Unknown option: %s\n", argv[i]);
                options[4] = 0;
            }
        }
        if (options[0] <= 0 || options[0] > 46340 || options[1] < 0) {
            std::fprintf(stderr, "Grid size must be in [1, 46340] and iterations must be nonnegative.\n");
            options[4] = 0;
        }
        if (options[4] != 1) printUsage(argv[0]);
    }
    MPI_Bcast(options, 5, MPI_INT, 0, MPI_COMM_WORLD);
    if (options[4] != 1) {
        MPI_Finalize();
        return options[4] == 2 ? 0 : 1;
    }

    const int n = options[0], iterations = options[1];
    const size_t element_count = static_cast<size_t>(n) * n;
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n, n, element_count);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n\n", options[2] ? "enabled" : "disabled");
        std::printf("Building distributed unstructured mesh...\n");
    }

    DistributedState state;
    initializeState(state, n);
    if (rank == 0) {
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        const size_t global_gpu_bytes = 4 * element_count * sizeof(val_t);
        std::printf("Hybrid execution: %d MPI rank(s), %d OpenMP thread(s)/rank, %d visible GPU(s)/node\n",
                    ranks, omp_get_max_threads(), devices);
        std::printf("Core GPU state: %.2f MB globally (plus halo buffers)\n\n",
                    global_gpu_bytes / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(state, n, iterations);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> energy, flux;
    gatherResults(state, n, energy, flux);
    int valid = 1;
    if (rank == 0) {
        const double milliseconds = seconds * 1000.0;
        const int measured_iterations = std::max(iterations - 1, 1);
        const double time_per_iteration = milliseconds / measured_iterations;
        const double giga_elements_per_second = seconds > 0.0
            ? (static_cast<double>(measured_iterations) * element_count) / seconds / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016" PRIX64 "\n\n", computeHash(energy, flux));
        if (options[3]) print_results(energy, "ElementEnergy");
        if (options[2]) valid = validateResults(energy, flux) ? 1 : 0;
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    destroyState(state);
    MPI_Finalize();
    return valid ? 0 : 1;
}
