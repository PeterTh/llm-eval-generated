#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t cuda_status_ = (call);                                    \
    if (cuda_status_ != cudaSuccess) {                                          \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(cuda_status_));                         \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

struct Domain {
    int first_row;
    int rows;
};

static Domain decompose(const int n, const int rank, const int ranks) {
    const int q = n / ranks;
    const int r = n % ranks;
    return {rank * q + std::min(rank, r), q + (rank < r ? 1 : 0)};
}

// local_x is padded by one halo row on either side.  The neighbor operations
// deliberately retain the reference implementation's (+x,-x,+y,-y) order.
__global__ void update_rows(const val_t* __restrict__ energy,
                            const val_t* __restrict__ accumulated_flux,
                            val_t* __restrict__ next_energy,
                            val_t* __restrict__ next_accumulated_flux,
                            const int n, const int first_global_row,
                            const int local_row_begin, const int local_row_end) {
    const size_t work = static_cast<size_t>(local_row_end - local_row_begin) * n;
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < work;
         linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int local_x = local_row_begin + static_cast<int>(linear / n);
        const int y = static_cast<int>(linear % n);
        const int global_x = first_global_row + local_x - 1;
        const size_t i = static_cast<size_t>(local_x) * n + y;
        const val_t here = energy[i];

        val_t total = 0.0;
        if ((global_x == 0 || global_x == n - 1) && (y == 0 || y == n - 1))
            total = (global_x == y) ? 0.5 : -0.5;

        if (global_x + 1 < n) total += ((energy[i + n] - here) * 0.8) * 0.25;
        if (global_x > 0)     total += ((energy[i - n] - here) * 0.8) * 0.25;
        if (y + 1 < n)        total += ((energy[i + 1] - here) * 0.8) * 0.25;
        if (y > 0)            total += ((energy[i - 1] - here) * 0.8) * 0.25;

        next_energy[i] = here + total;
        next_accumulated_flux[i] = accumulated_flux[i] + fabs(total);
    }
}

static void launch_rows(cudaStream_t stream, const val_t* energy, const val_t* flux,
                        val_t* next_energy, val_t* next_flux, const int n,
                        const int first_row, const int begin, const int end,
                        const int multiprocessors) {
    if (begin >= end) return;
    constexpr int threads = 256;
    const size_t count = static_cast<size_t>(end - begin) * n;
    const int blocks = std::max(1, std::min(static_cast<int>((count + threads - 1) / threads),
                                            multiprocessors * 8));
    update_rows<<<blocks, threads, 0, stream>>>(energy, flux, next_energy, next_flux,
                                                n, first_row, begin, end);
    CUDA_CHECK(cudaGetLastError());
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

static uint64_t bits(const val_t value) {
    uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

int main(int argc, char** argv) {
    int supplied = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &supplied);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (supplied < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) std::fprintf(stderr, "MPI lacks required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int n = 512;
    int iterations = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool help = false;
    bool args_valid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_requested = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else { if (world_rank == 0) std::printf("Unknown option: %s\n", argv[i]); args_valid = false; }
    }
    if (help || !args_valid) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return args_valid ? 0 : 1;
    }
    if (n <= 0 || iterations < 0 || static_cast<uint64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (world_rank == 0) std::fprintf(stderr, "Grid size must be positive, iterations nonnegative, and N*N must fit in an int\n");
        MPI_Finalize();
        return 1;
    }
    if (world_size > n) {
        if (world_rank == 0) std::fprintf(stderr, "MPI rank count (%d) must not exceed grid rows (%d)\n", world_size, n);
        MPI_Finalize();
        return 1;
    }

    // Assign ranks sharing a node round-robin to its visible accelerators.
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (world_rank == 0) std::fprintf(stderr, "The hybrid benchmark requires CUDA accelerators\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(node_rank % device_count));
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, node_rank % device_count));

    const Domain domain = decompose(n, world_rank, world_size);
    const size_t local_count = static_cast<size_t>(domain.rows) * n;
    const size_t padded_count = static_cast<size_t>(domain.rows + 2) * n;
    const size_t bytes = padded_count * sizeof(val_t);
    val_t *energy = nullptr, *flux = nullptr, *next_energy = nullptr, *next_flux = nullptr;
    CUDA_CHECK(cudaMalloc(&energy, bytes));
    CUDA_CHECK(cudaMalloc(&flux, bytes));
    CUDA_CHECK(cudaMalloc(&next_energy, bytes));
    CUDA_CHECK(cudaMalloc(&next_flux, bytes));
    CUDA_CHECK(cudaMemset(energy, 0, bytes));
    CUDA_CHECK(cudaMemset(flux, 0, bytes));
    CUDA_CHECK(cudaMemset(next_energy, 0, bytes));
    CUDA_CHECK(cudaMemset(next_flux, 0, bytes));

    val_t *send_top = nullptr, *send_bottom = nullptr, *recv_top = nullptr, *recv_bottom = nullptr;
    CUDA_CHECK(cudaMallocHost(&send_top, static_cast<size_t>(4) * n * sizeof(val_t)));
    send_bottom = send_top + n;
    recv_top = send_bottom + n;
    recv_bottom = recv_top + n;
    cudaStream_t compute_stream, communication_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communication_stream, cudaStreamNonBlocking));

    const int global_elements = n * n;
    if (world_rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n",
                    n, n, global_elements, iterations, validate ? "enabled" : "disabled");
        std::printf("Building unstructured mesh...\n");
        const size_t device_total = static_cast<size_t>(global_elements) * sizeof(val_t) * 4;
        std::printf("Memory usage: %.2f MB distributed device state (%d MPI ranks)\n\n",
                    device_total / (1024.0 * 1024.0), world_size);
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (world_size == 1) {
            launch_rows(compute_stream, energy, flux, next_energy, next_flux, n,
                        domain.first_row, 1, domain.rows + 1, device_properties.multiProcessorCount);
        } else {
            // Stage only boundary energy.  Interior GPU work overlaps these copies and MPI traffic.
            CUDA_CHECK(cudaMemcpyAsync(send_top, energy + n, n * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, communication_stream));
            CUDA_CHECK(cudaMemcpyAsync(send_bottom, energy + static_cast<size_t>(domain.rows) * n,
                                       n * sizeof(val_t), cudaMemcpyDeviceToHost, communication_stream));
            launch_rows(compute_stream, energy, flux, next_energy, next_flux, n,
                        domain.first_row, 2, domain.rows, device_properties.multiProcessorCount);
            CUDA_CHECK(cudaStreamSynchronize(communication_stream));

            const int previous = world_rank == 0 ? MPI_PROC_NULL : world_rank - 1;
            const int next = world_rank + 1 == world_size ? MPI_PROC_NULL : world_rank + 1;
            MPI_Sendrecv(send_top, n, MPI_DOUBLE, previous, 100,
                         recv_bottom, n, MPI_DOUBLE, next, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_bottom, n, MPI_DOUBLE, next, 101,
                         recv_top, n, MPI_DOUBLE, previous, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (previous != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(energy, recv_top, n * sizeof(val_t),
                                           cudaMemcpyHostToDevice, communication_stream));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(energy + static_cast<size_t>(domain.rows + 1) * n,
                                           recv_bottom, n * sizeof(val_t), cudaMemcpyHostToDevice,
                                           communication_stream));
            CUDA_CHECK(cudaStreamSynchronize(communication_stream));
            launch_rows(communication_stream, energy, flux, next_energy, next_flux, n,
                        domain.first_row, 1, 2, device_properties.multiProcessorCount);
            if (domain.rows > 1)
                launch_rows(communication_stream, energy, flux, next_energy, next_flux, n,
                            domain.first_row, domain.rows, domain.rows + 1,
                            device_properties.multiProcessorCount);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(communication_stream));
        std::swap(energy, next_energy);
        std::swap(flux, next_flux);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> local_energy(local_count), local_flux(local_count);
    CUDA_CHECK(cudaMemcpy(local_energy.data(), energy + n, local_count * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_flux.data(), flux + n, local_count * sizeof(val_t), cudaMemcpyDeviceToHost));

    val_t local_energy_sum = 0.0, local_flux_sum = 0.0;
    val_t local_min = std::numeric_limits<val_t>::max();
    val_t local_max = std::numeric_limits<val_t>::lowest();
    uint64_t local_hash = 0;
    #pragma omp parallel for reduction(+:local_energy_sum,local_flux_sum) reduction(min:local_min) reduction(max:local_max) reduction(^:local_hash) schedule(static)
    for (size_t i = 0; i < local_count; ++i) {
        local_energy_sum += local_energy[i];
        local_flux_sum += local_flux[i];
        local_min = std::min(local_min, local_energy[i]);
        local_max = std::max(local_max, local_energy[i]);
        const uint64_t global_i = static_cast<uint64_t>(domain.first_row) * n + i;
        local_hash ^= (bits(local_energy[i]) + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (bits(local_flux[i]) + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    val_t energy_sum = 0.0, flux_sum = 0.0, energy_min = 0.0, energy_max = 0.0;
    uint64_t hash = 0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const int measured_iterations = std::max(iterations - 1, 1);
        const double giga_elements_per_second = elapsed > 0.0
            ? (static_cast<double>(measured_iterations) * global_elements / elapsed / 1e9) : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance:\n  Time per iteration: %.4f ms\n", milliseconds / measured_iterations);
        std::printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
                    giga_elements_per_second, giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }

    if (print_results_requested) {
        std::vector<int> counts(world_size), displacements(world_size);
        for (int rank = 0; rank < world_size; ++rank) {
            const Domain d = decompose(n, rank, world_size);
            counts[rank] = d.rows * n;
            displacements[rank] = d.first_row * n;
        }
        std::vector<double> global_energy;
        if (world_rank == 0) global_energy.resize(global_elements);
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_count), MPI_DOUBLE,
                    world_rank == 0 ? global_energy.data() : nullptr, counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (world_rank == 0) print_results(global_energy, "ElementEnergy");
    }

    int valid = 1;
    if (validate && world_rank == 0) {
        std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n", energy_sum, flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) ||
            !std::isfinite(energy_min) || !std::isfinite(energy_max)) {
            std::printf("  ERROR: non-finite simulation result\n");
            valid = 0;
        } else {
            if (std::abs(energy_sum) > 1e-8)
                std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            std::printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(communication_stream));
    CUDA_CHECK(cudaFreeHost(send_top));
    CUDA_CHECK(cudaFree(energy));
    CUDA_CHECK(cudaFree(flux));
    CUDA_CHECK(cudaFree(next_energy));
    CUDA_CHECK(cudaFree(next_flux));
    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
