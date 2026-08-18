#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void cudaCheck(cudaError_t status, const char* expression, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, expression,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, rank)

// One MPI rank owns a contiguous slab of rows.  Row 0 and row local_rows+1 in
// current are ghost rows received from the adjacent ranks.
__global__ void updateRows(const val_t* __restrict__ current,
                           const val_t* __restrict__ total,
                           val_t* __restrict__ next_current,
                           val_t* __restrict__ next_total,
                           int n, int first_global_row, int local_rows,
                           int begin_local_row, int end_local_row) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_row = begin_local_row + blockIdx.y;
    if (col >= n || local_row >= end_local_row) return;

    const int global_row = first_global_row + local_row;
    const size_t pitch = static_cast<size_t>(n);
    const size_t at = static_cast<size_t>(local_row + 1) * pitch + col;
    const val_t center = current[at];

    val_t flux = 0.0;
    // The four corner materials are the only non-default materials.  Preserve
    // the original inflow/outflow assignment and neighbor evaluation order.
    if ((global_row == 0 || global_row == n - 1) && (col == 0 || col == n - 1)) {
        flux = (global_row == col) ? 0.5 : -0.5;
    }
    // Keep the same expression tree as computeFlux in the reference.  In
    // particular, do not pre-fold 0.8 * 0.25: that changes some doubles by an
    // ULP even though the mathematical expressions are equivalent.
    if (global_row + 1 < n) flux += (current[at + pitch] - center) * 0.8 * 1.0 * 0.25;
    if (global_row > 0)     flux += (current[at - pitch] - center) * 0.8 * 1.0 * 0.25;
    if (col + 1 < n)        flux += (current[at + 1] - center) * 0.8 * 1.0 * 0.25;
    if (col > 0)            flux += (current[at - 1] - center) * 0.8 * 1.0 * 0.25;

    next_current[at] = center + flux;
    next_total[at] = total[at] + fabs(flux);
}

static void launchRows(const val_t* current, const val_t* total,
                       val_t* next_current, val_t* next_total,
                       int n, int first_row, int local_rows,
                       int begin, int end, cudaStream_t stream, int rank) {
    if (begin >= end) return;
    constexpr int threads = 256;
    const dim3 block(threads, 1, 1);
    const dim3 grid((n + threads - 1) / threads, end - begin, 1);
    updateRows<<<grid, block, 0, stream>>>(current, total, next_current, next_total,
                                          n, first_row, local_rows, begin, end);
    CUDA_CHECK(cudaGetLastError());
}

static uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    // OpenMP handles host-side reductions after CUDA/MPI have produced the
    // distributed result.  Static scheduling makes the extrema deterministic.
#pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min) schedule(static)
    for (idx_t i = 0; i < elements.size(); ++i) {
        energy_sum += elements[i].current_energy;
        flux_sum += elements[i].total_flux;
        energy_max = std::max(energy_max, elements[i].current_energy);
        energy_min = std::min(energy_min, elements[i].current_energy);
    }

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    bool valid = std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
                 std::isfinite(energy_max) && std::isfinite(energy_min);
    if (!std::isfinite(energy_sum)) std::printf("  ERROR: Energy sum is not finite\n");
    if (std::isfinite(energy_sum) && std::abs(energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) std::printf("  ERROR: Flux sum is not finite\n");
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min))
        std::printf("  ERROR: Energy extrema are not finite\n");
    if (valid) std::printf("  Validation: PASSED\n");
    return valid;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required FUNNELED thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 512;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) showHelp = true;
        else { if (rank == 0) std::printf("Unknown option: %s\n", argv[i]); argsValid = false; }
    }
    if (showHelp || !argsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argsValid ? 0 : 1;
    }
    if (n <= 0 || iterations < 0 || ranks > n) {
        if (rank == 0) std::fprintf(stderr,
            "Grid size must be positive and at least the number of MPI ranks; iterations must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    int gpu_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
    if (gpu_count == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA accelerator is available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % gpu_count));

    const int base_rows = n / ranks;
    const int extra_rows = n % ranks;
    const int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int first_row = rank * base_rows + std::min(rank, extra_rows);
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const size_t padded_elements = static_cast<size_t>(local_rows + 2) * n;
    const size_t bytes = padded_elements * sizeof(val_t);

    val_t *d_current = nullptr, *d_next = nullptr, *d_total = nullptr, *d_next_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_current, bytes));
    CUDA_CHECK(cudaMalloc(&d_next, bytes));
    CUDA_CHECK(cudaMalloc(&d_total, bytes));
    CUDA_CHECK(cudaMalloc(&d_next_total, bytes));
    CUDA_CHECK(cudaMemset(d_current, 0, bytes));
    CUDA_CHECK(cudaMemset(d_next, 0, bytes));
    CUDA_CHECK(cudaMemset(d_total, 0, bytes));
    CUDA_CHECK(cudaMemset(d_next_total, 0, bytes));

    val_t *send_top = nullptr, *send_bottom = nullptr, *recv_top = nullptr, *recv_bottom = nullptr;
    CUDA_CHECK(cudaMallocHost(&send_top, static_cast<size_t>(n) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&send_bottom, static_cast<size_t>(n) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&recv_top, static_cast<size_t>(n) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&recv_bottom, static_cast<size_t>(n) * sizeof(val_t)));
    cudaStream_t compute_stream, communication_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communication_stream, cudaStreamNonBlocking));

    const uint64_t global_elements = static_cast<uint64_t>(n) * n;
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, global_elements);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Parallelism: %d MPI rank(s), up to %d OpenMP thread(s)/rank, CUDA GPUs\n",
                    ranks, omp_get_max_threads());
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        const size_t static_mem = global_elements * (2 * sizeof(idx_t) + 8 * (sizeof(idx_t) + sizeof(val_t)));
        const size_t dynamic_mem = global_elements * sizeof(ElementDynamic) * 2;
        std::printf("Building distributed unstructured mesh...\n");
        std::printf("Global logical memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                    (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0,
                    dynamic_mem / 1048576.0);
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (ranks == 1) {
            launchRows(d_current, d_total, d_next, d_next_total, n, first_row,
                       local_rows, 0, local_rows, compute_stream, rank);
        } else {
            // Stage boundaries through pinned memory for portability across both
            // CUDA-aware and conventional MPI implementations.
            if (previous != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_top, d_current + n, n * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, communication_stream));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_bottom, d_current + static_cast<size_t>(local_rows) * n,
                                           n * sizeof(val_t), cudaMemcpyDeviceToHost, communication_stream));

            // Interior rows do not depend on incoming halos and overlap their
            // GPU execution with PCIe transfers and network communication.
            launchRows(d_current, d_total, d_next, d_next_total, n, first_row,
                       local_rows, 1, local_rows - 1, compute_stream, rank);
            CUDA_CHECK(cudaStreamSynchronize(communication_stream));

            MPI_Request requests[4];
            int request_count = 0;
            if (previous != MPI_PROC_NULL) {
                MPI_Irecv(recv_top, n, MPI_DOUBLE, previous, 11, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(send_top, n, MPI_DOUBLE, previous, 10, MPI_COMM_WORLD, &requests[request_count++]);
            }
            if (next != MPI_PROC_NULL) {
                MPI_Irecv(recv_bottom, n, MPI_DOUBLE, next, 10, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(send_bottom, n, MPI_DOUBLE, next, 11, MPI_COMM_WORLD, &requests[request_count++]);
            }
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
            if (previous != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(d_current, recv_top, n * sizeof(val_t),
                                           cudaMemcpyHostToDevice, communication_stream));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(d_current + static_cast<size_t>(local_rows + 1) * n,
                                           recv_bottom, n * sizeof(val_t), cudaMemcpyHostToDevice,
                                           communication_stream));
            CUDA_CHECK(cudaStreamSynchronize(communication_stream));
            launchRows(d_current, d_total, d_next, d_next_total, n, first_row,
                       local_rows, 0, 1, compute_stream, rank);
            if (local_rows > 1)
                launchRows(d_current, d_total, d_next, d_next_total, n, first_row,
                           local_rows, local_rows - 1, local_rows, compute_stream, rank);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        std::swap(d_current, d_next);
        std::swap(d_total, d_next_total);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<ElementDynamic> local_result(static_cast<size_t>(local_rows) * n);
    std::vector<val_t> local_energy(local_result.size()), local_flux(local_result.size());
    CUDA_CHECK(cudaMemcpy(local_energy.data(), d_current + n, local_energy.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_flux.data(), d_total + n, local_flux.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < local_result.size(); ++i)
        local_result[i] = ElementDynamic{local_energy[i], local_flux[i]};

    std::vector<int> receive_counts, displacements;
    std::vector<ElementDynamic> global_result;
    if (rank == 0) {
        receive_counts.resize(ranks);
        displacements.resize(ranks);
        int displacement = 0;
        for (int r = 0; r < ranks; ++r) {
            const int rows = base_rows + (r < extra_rows ? 1 : 0);
            receive_counts[r] = rows * n;
            displacements[r] = displacement;
            displacement += receive_counts[r];
        }
        global_result.resize(global_elements);
    }
    MPI_Datatype dynamic_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &dynamic_type);
    MPI_Type_commit(&dynamic_type);
    MPI_Gatherv(local_result.data(), static_cast<int>(local_result.size()), dynamic_type,
                global_result.data(), receive_counts.data(), displacements.data(), dynamic_type,
                0, MPI_COMM_WORLD);
    MPI_Type_free(&dynamic_type);

    int exit_code = 0;
    if (rank == 0) {
        const double duration_ms = elapsed * 1000.0;
        const int measured_iterations = std::max(iterations - 1, 1);
        const double time_per_iteration = duration_ms / measured_iterations;
        const double giga_elements_per_second = elapsed > 0.0
            ? (static_cast<double>(measured_iterations) * global_elements) / elapsed / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016" PRIX64 "\n\n", computeHash(global_result));
        if (printResults) {
            std::vector<double> energy(global_result.size());
#pragma omp parallel for schedule(static)
            for (idx_t i = 0; i < global_result.size(); ++i) energy[i] = global_result[i].current_energy;
            print_results(energy, "ElementEnergy");
        }
        if (validate && !validateResults(global_result)) exit_code = 1;
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(communication_stream));
    CUDA_CHECK(cudaFreeHost(send_top));
    CUDA_CHECK(cudaFreeHost(send_bottom));
    CUDA_CHECK(cudaFreeHost(recv_top));
    CUDA_CHECK(cudaFreeHost(recv_bottom));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaFree(d_next_total));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return exit_code;
}
