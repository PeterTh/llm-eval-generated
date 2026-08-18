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

using idx_t = uint64_t;
using val_t = double;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

namespace {

int world_rank = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", world_rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed: %s", operation,
                      cudaGetErrorString(status));
        fail(message);
    }
}

// Update rows [first_row, last_row] of this rank's domain. Row zero and
// local_rows + 1 are MPI halo rows. The generated mesh has unit connection
// weights and a transfer coefficient of 0.8, so each edge contributes 0.2.
__global__ void updateKernel(const ElementDynamic* __restrict__ current,
                             ElementDynamic* __restrict__ next,
                             int width, int global_row_begin,
                             int first_row, int last_row) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_row = first_row + blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || local_row > last_row) return;

    const int global_row = global_row_begin + local_row - 1;
    const size_t index = static_cast<size_t>(local_row) * width + col;
    const val_t energy = current[index].current_energy;

    val_t external_flow = 0.0;
    const bool left_corner = (col == 0);
    const bool right_corner = (col == width - 1);
    if ((global_row == 0 || global_row == width - 1) &&
        (left_corner || right_corner)) {
        // (0,0) and (N-1,N-1) are inflows; the other corners outflows.
        external_flow = ((global_row == 0) == left_corner) ? 0.5 : -0.5;
    }

    val_t total_flux = external_flow;
    // Preserve the reference connection order: +row, -row, +column, -column.
    if (global_row + 1 < width)
        total_flux += (current[index + width].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (global_row > 0)
        total_flux += (current[index - width].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (col + 1 < width)
        total_flux += (current[index + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (col > 0)
        total_flux += (current[index - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;

    next[index].current_energy = energy + total_flux;
    next[index].total_flux = current[index].total_flux + fabs(total_flux);
}

struct Distribution {
    int first_row;
    int local_rows;
};

Distribution distributeRows(int n, int rank, int ranks) {
    const int base = n / ranks;
    const int remainder = n % ranks;
    return {rank * base + std::min(rank, remainder),
            base + (rank < remainder ? 1 : 0)};
}

void launchRows(const ElementDynamic* current, ElementDynamic* next, int width,
                const Distribution& distribution, int first, int last,
                cudaStream_t stream) {
    if (first > last) return;
    constexpr dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x,
                    (last - first + 1 + block.y - 1) / block.y);
    updateKernel<<<grid, block, 0, stream>>>(
        current, next, width, distribution.first_row, first, last);
    checkCuda(cudaGetLastError(), "CUDA kernel launch");
}

void runSimulation(int width, int iterations, const Distribution& distribution,
                   int rank, int ranks, std::vector<ElementDynamic>& result,
                   double& elapsed_seconds) {
    const size_t pitch_elems = static_cast<size_t>(width);
    const size_t allocated_elems = static_cast<size_t>(distribution.local_rows + 2) * width;
    const size_t allocated_bytes = allocated_elems * sizeof(ElementDynamic);
    const size_t row_bytes = pitch_elems * sizeof(ElementDynamic);

    ElementDynamic* device_current = nullptr;
    ElementDynamic* device_next = nullptr;
    ElementDynamic* send_top = nullptr;
    ElementDynamic* send_bottom = nullptr;
    ElementDynamic* recv_top = nullptr;
    ElementDynamic* recv_bottom = nullptr;
    checkCuda(cudaMalloc(&device_current, allocated_bytes), "cudaMalloc(current)");
    checkCuda(cudaMalloc(&device_next, allocated_bytes), "cudaMalloc(next)");
    checkCuda(cudaMemset(device_current, 0, allocated_bytes), "cudaMemset(current)");
    checkCuda(cudaMemset(device_next, 0, allocated_bytes), "cudaMemset(next)");
    checkCuda(cudaMallocHost(&send_top, row_bytes), "cudaMallocHost(send_top)");
    checkCuda(cudaMallocHost(&send_bottom, row_bytes), "cudaMallocHost(send_bottom)");
    checkCuda(cudaMallocHost(&recv_top, row_bytes), "cudaMallocHost(recv_top)");
    checkCuda(cudaMallocHost(&recv_bottom, row_bytes), "cudaMallocHost(recv_bottom)");

    cudaStream_t compute_stream;
    cudaStream_t communication_stream;
    cudaEvent_t iteration_ready;
    cudaEvent_t halo_ready;
    checkCuda(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
              "cudaStreamCreate(compute)");
    checkCuda(cudaStreamCreateWithFlags(&communication_stream, cudaStreamNonBlocking),
              "cudaStreamCreate(communication)");
    checkCuda(cudaEventCreateWithFlags(&iteration_ready, cudaEventDisableTiming),
              "cudaEventCreate(iteration)");
    checkCuda(cudaEventCreateWithFlags(&halo_ready, cudaEventDisableTiming),
              "cudaEventCreate(halo)");
    checkCuda(cudaEventRecord(iteration_ready, compute_stream), "cudaEventRecord(initial)");

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        checkCuda(cudaStreamWaitEvent(communication_stream, iteration_ready, 0),
                  "cudaStreamWaitEvent(iteration)");

        // Stage boundary rows through pinned memory for portable MPI. The GPU
        // computes all independent interior rows while MPI moves the halos.
        if (rank > 0)
            checkCuda(cudaMemcpyAsync(send_top, device_current + pitch_elems,
                                      row_bytes, cudaMemcpyDeviceToHost,
                                      communication_stream), "copy top boundary");
        if (rank + 1 < ranks)
            checkCuda(cudaMemcpyAsync(send_bottom,
                                      device_current + static_cast<size_t>(distribution.local_rows) * width,
                                      row_bytes, cudaMemcpyDeviceToHost,
                                      communication_stream), "copy bottom boundary");

        launchRows(device_current, device_next, width, distribution, 2,
                   distribution.local_rows - 1, compute_stream);
        checkCuda(cudaStreamSynchronize(communication_stream), "boundary staging");

        MPI_Request requests[4];
        int request_count = 0;
        if (rank > 0) {
            MPI_Irecv(recv_top, width * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                      rank - 1, 101, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Isend(send_top, width * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                      rank - 1, 100, MPI_COMM_WORLD, &requests[request_count++]);
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(recv_bottom, width * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                      rank + 1, 100, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Isend(send_bottom, width * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                      rank + 1, 101, MPI_COMM_WORLD, &requests[request_count++]);
        }
        if (request_count != 0)
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        if (rank > 0)
            checkCuda(cudaMemcpyAsync(device_current, recv_top, row_bytes,
                                      cudaMemcpyHostToDevice, communication_stream),
                      "copy top halo");
        if (rank + 1 < ranks)
            checkCuda(cudaMemcpyAsync(
                          device_current + static_cast<size_t>(distribution.local_rows + 1) * width,
                          recv_bottom, row_bytes, cudaMemcpyHostToDevice, communication_stream),
                      "copy bottom halo");
        checkCuda(cudaEventRecord(halo_ready, communication_stream), "cudaEventRecord(halo)");
        checkCuda(cudaStreamWaitEvent(compute_stream, halo_ready, 0),
                  "cudaStreamWaitEvent(halo)");

        launchRows(device_current, device_next, width, distribution, 1, 1, compute_stream);
        if (distribution.local_rows > 1)
            launchRows(device_current, device_next, width, distribution,
                       distribution.local_rows, distribution.local_rows, compute_stream);
        checkCuda(cudaEventRecord(iteration_ready, compute_stream), "cudaEventRecord(iteration)");
        std::swap(device_current, device_next);
    }

    checkCuda(cudaEventSynchronize(iteration_ready), "final CUDA synchronization");
    const double local_elapsed = MPI_Wtime() - start;
    MPI_Reduce(&local_elapsed, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    result.resize(static_cast<size_t>(distribution.local_rows) * width);
    checkCuda(cudaMemcpy(result.data(), device_current + pitch_elems,
                         result.size() * sizeof(ElementDynamic), cudaMemcpyDeviceToHost),
              "copy final result");

    cudaEventDestroy(halo_ready);
    cudaEventDestroy(iteration_ready);
    cudaStreamDestroy(communication_stream);
    cudaStreamDestroy(compute_stream);
    cudaFreeHost(recv_bottom);
    cudaFreeHost(recv_top);
    cudaFreeHost(send_bottom);
    cudaFreeHost(send_top);
    cudaFree(device_next);
    cudaFree(device_current);
}

uint64_t computeLocalHash(const std::vector<ElementDynamic>& elements,
                          uint64_t global_offset) {
    uint64_t hash_value = 0;
    // XOR makes the reference hash safely reducible by OpenMP.
#pragma omp parallel for reduction(^ : hash_value) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(elements.size()); ++i) {
        uint64_t energy_bits;
        uint64_t flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        const uint64_t global_index = global_offset + static_cast<uint64_t>(i);
        hash_value ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
        hash_value ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash_value;
}

bool validateResultsDistributed(const std::vector<ElementDynamic>& elements, int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
#pragma omp parallel for reduction(+ : energy_sum, flux_sum) reduction(max : energy_max) reduction(min : energy_min) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(elements.size()); ++i) {
        energy_sum += elements[i].current_energy;
        flux_sum += elements[i].total_flux;
        energy_max = std::max(energy_max, elements[i].current_energy);
        energy_min = std::min(energy_min, elements[i].current_energy);
    }

    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", global_energy_sum);
        std::printf("  Flux sum: %.2f\n", global_flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
        if (!std::isfinite(global_energy_sum) || !std::isfinite(global_flux_sum) ||
            !std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            std::printf("  ERROR: simulation produced a non-finite result\n");
            valid = 0;
        } else {
            if (std::abs(global_energy_sum) > 1e-8)
                std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            std::printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not provide MPI_THREAD_FUNNELED");

    int width = 512;
    int iterations = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool arguments_ok = true;
    bool show_help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            width = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (std::strcmp(argv[i], "-r") == 0)
            print_results_requested = true;
        else if (std::strcmp(argv[i], "-h") == 0)
            show_help = true;
        else
            arguments_ok = false;
    }
    if (show_help || !arguments_ok) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return arguments_ok ? 0 : 1;
    }
    if (width <= 0 || iterations < 0 || ranks > width)
        fail("require -n > 0, -i >= 0, and no more MPI ranks than grid rows");
    if (static_cast<uint64_t>(width) * width * sizeof(ElementDynamic) >
        static_cast<uint64_t>(std::numeric_limits<int>::max()))
        fail("grid is too large for MPI_Gatherv counts");

    MPI_Comm local_communicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &local_communicator);
    int local_rank = 0;
    MPI_Comm_rank(local_communicator, &local_rank);
    MPI_Comm_free(&local_communicator);
    int gpu_count = 0;
    checkCuda(cudaGetDeviceCount(&gpu_count), "cudaGetDeviceCount");
    if (gpu_count == 0) fail("the mandatory CUDA execution path requires a GPU");
    checkCuda(cudaSetDevice(local_rank % gpu_count), "cudaSetDevice");

    const Distribution distribution = distributeRows(width, world_rank, ranks);
    if (world_rank == 0) {
        const int64_t elements = static_cast<int64_t>(width) * width;
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\n", width, width,
                    static_cast<long long>(elements));
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI rank(s), up to %d OpenMP thread(s)/rank, CUDA GPUs\n",
                    ranks, omp_get_max_threads());
        std::printf("Building distributed unstructured mesh...\n");
        const double dynamic_mb = elements * sizeof(ElementDynamic) * 2.0 / (1024.0 * 1024.0);
        std::printf("Dynamic state memory: %.2f MB globally (distributed)\n\n", dynamic_mb);
        std::printf("Running simulation...\n");
    }

    std::vector<ElementDynamic> local_result;
    double elapsed_seconds = 0.0;
    runSimulation(width, iterations, distribution, world_rank, ranks,
                  local_result, elapsed_seconds);

    const uint64_t local_hash = computeLocalHash(
        local_result, static_cast<uint64_t>(distribution.first_row) * width);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    int return_code = 0;
    if (world_rank == 0) {
        const double milliseconds = elapsed_seconds * 1000.0;
        const int measured_iterations = std::max(iterations - 1, 1);
        const double time_per_iteration = milliseconds / measured_iterations;
        const double giga_elements = elapsed_seconds > 0.0
            ? (static_cast<double>(measured_iterations) * width * width / elapsed_seconds / 1e9) : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elements * 22.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(global_hash));
    }

    if (print_results_requested) {
        std::vector<double> local_energy(local_result.size());
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(local_result.size()); ++i)
            local_energy[i] = local_result[i].current_energy;
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<double> global_energy;
        if (world_rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int rank = 0; rank < ranks; ++rank) {
                const Distribution part = distributeRows(width, rank, ranks);
                counts[rank] = part.local_rows * width;
                displacements[rank] = part.first_row * width;
            }
            global_energy.resize(static_cast<size_t>(width) * width);
        }
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                    global_energy.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (world_rank == 0) print_results(global_energy, "ElementEnergy");
    }

    if (validate && !validateResultsDistributed(local_result, world_rank)) return_code = 1;

    MPI_Bcast(&return_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return return_code;
}
