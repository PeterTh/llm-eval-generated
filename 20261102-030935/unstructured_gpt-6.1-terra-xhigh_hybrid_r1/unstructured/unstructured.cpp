#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

static void cuda_check(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cuda_check((call), __FILE__, __LINE__)

// The original mesh is a square grid, with connections ordered +row, -row,
// +column, -column. Preserve that order and the original arithmetic here.
__global__ void update(const double* energy, const double* flux, double* next_energy,
                       double* next_flux, int n, int rows, int first_row,
                       int row_begin, int row_count, bool boundary) {
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(row_count) * n;
    if (k >= count) return;
    int row = row_begin + static_cast<int>(k / n);
    if (boundary && row == 1) row = rows - 1;
    int col = static_cast<int>(k % n);
    int global_row = first_row + row;
    size_t p = static_cast<size_t>(row + 1) * n + col;
    size_t q = static_cast<size_t>(row) * n + col;
    double current = energy[p];
    double total = 0.0;
    if ((global_row == 0 || global_row == n - 1) && (col == 0 || col == n - 1)) {
        // For n == 1 the final material assignment in the original is inflow.
        total = (global_row == col || n == 1) ? 0.5 : -0.5;
    }
    if (global_row + 1 < n) total += ((energy[p + n] - current) * 0.8) * 0.25;
    if (global_row > 0)     total += ((energy[p - n] - current) * 0.8) * 0.25;
    if (col + 1 < n)        total += ((energy[p + 1] - current) * 0.8) * 0.25;
    if (col > 0)            total += ((energy[p - 1] - current) * 0.8) * 0.25;
    next_energy[p] = current + total;
    next_flux[q] = flux[q] + fabs(total);
}

static void launch_update(const double* energy, const double* flux, double* next_energy,
                          double* next_flux, int n, int rows, int first_row,
                          int begin, int count, cudaStream_t stream, bool boundary = false) {
    if (count == 0) return;
    constexpr int threads = 256;
    size_t cells = static_cast<size_t>(count) * n;
    update<<<static_cast<unsigned>((cells + threads - 1) / threads), threads, 0, stream>>>(
        energy, flux, next_energy, next_flux, n, rows, first_row, begin, count, boundary);
    CUDA(cudaGetLastError());
}

static bool validateResults(const std::vector<double>& energies, const std::vector<double>& fluxes) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (size_t i = 0; i < energies.size(); ++i) {
        energy_sum += energies[i];
        flux_sum += fluxes[i];
        energy_max = std::max(energies[i], energy_max);
        energy_min = std::min(energies[i], energy_min);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512, iterations = 10;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || static_cast<long long>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and contain at most INT_MAX elements\n");
        MPI_Finalize(); return 1;
    }
    const int active_size = std::min(size, n);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active_size ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank >= active_size) { MPI_Finalize(); return 0; }

    MPI_Comm local;
    MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_free(&local);
    int devices = 0;
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices));

    const int rows = n / active_size + (rank < n % active_size);
    const int first_row = rank * (n / active_size) + std::min(rank, n % active_size);
    const int local_cells = rows * n;
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(double);
    const size_t energy_bytes = static_cast<size_t>(rows + 2) * row_bytes;
    const size_t flux_bytes = static_cast<size_t>(local_cells) * sizeof(double);
    double *energy[2], *flux[2], *send_buffer = nullptr, *recv_buffer = nullptr;
    for (int b = 0; b < 2; ++b) {
        CUDA(cudaMalloc(&energy[b], energy_bytes));
        CUDA(cudaMalloc(&flux[b], flux_bytes));
        CUDA(cudaMemset(energy[b], 0, energy_bytes));
        CUDA(cudaMemset(flux[b], 0, flux_bytes));
    }
    if (active_size > 1) {
        CUDA(cudaMallocHost(&send_buffer, 2 * row_bytes));
        CUDA(cudaMallocHost(&recv_buffer, 2 * row_bytes));
    }
    cudaStream_t compute, exchange;
    cudaEvent_t ready;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
    CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA(cudaEventRecord(ready, compute));
    CUDA(cudaDeviceSynchronize());

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n * n);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        printf("MPI ranks: %d; grid connectivity computed implicitly\n\n", active_size);
        printf("Running simulation...\n");
    }
    MPI_Barrier(active);
    double start = MPI_Wtime();
    int current = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        int next = 1 - current;
        if (active_size == 1) {
            launch_update(energy[current], flux[current], energy[next], flux[next],
                          n, rows, first_row, 0, rows, compute);
        } else {
            // Copy the outgoing rows while the GPU computes the independent interior.
            CUDA(cudaStreamWaitEvent(exchange, ready, 0));
            if (rank > 0) CUDA(cudaMemcpyAsync(send_buffer, energy[current] + n,
                                                row_bytes, cudaMemcpyDeviceToHost, exchange));
            if (rank + 1 < active_size)
                CUDA(cudaMemcpyAsync(send_buffer + n, energy[current] + static_cast<size_t>(rows) * n,
                                     row_bytes, cudaMemcpyDeviceToHost, exchange));
            launch_update(energy[current], flux[current], energy[next], flux[next],
                          n, rows, first_row, 1, std::max(rows - 2, 0), compute);
            // The previous halo upload must finish before MPI reuses recv_buffer.
            CUDA(cudaStreamSynchronize(exchange));
            MPI_Request req[4]; int requests = 0;
            if (rank > 0) MPI_Irecv(recv_buffer, n, MPI_DOUBLE, rank - 1, 1, active, &req[requests++]);
            if (rank + 1 < active_size) MPI_Irecv(recv_buffer + n, n, MPI_DOUBLE, rank + 1, 0, active, &req[requests++]);
            if (rank > 0) MPI_Isend(send_buffer, n, MPI_DOUBLE, rank - 1, 0, active, &req[requests++]);
            if (rank + 1 < active_size) MPI_Isend(send_buffer + n, n, MPI_DOUBLE, rank + 1, 1, active, &req[requests++]);
            MPI_Waitall(requests, req, MPI_STATUSES_IGNORE);
            if (rank > 0) CUDA(cudaMemcpyAsync(energy[current], recv_buffer,
                                                row_bytes, cudaMemcpyHostToDevice, compute));
            if (rank + 1 < active_size)
                CUDA(cudaMemcpyAsync(energy[current] + static_cast<size_t>(rows + 1) * n, recv_buffer + n,
                                     row_bytes, cudaMemcpyHostToDevice, compute));
            launch_update(energy[current], flux[current], energy[next], flux[next],
                          n, rows, first_row, 0, std::min(rows, 2), compute, true);
        }
        CUDA(cudaEventRecord(ready, compute));
        current = next;
    }
    CUDA(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active);

    std::vector<double> local_energy(local_cells), local_flux(local_cells);
    CUDA(cudaMemcpy(local_energy.data(), energy[current] + n, flux_bytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(local_flux.data(), flux[current], flux_bytes, cudaMemcpyDeviceToHost));
    uint64_t local_hash = 0;
    #pragma omp parallel for reduction(^:local_hash) schedule(static) if(local_cells >= 4096)
    for (int i = 0; i < local_cells; ++i) {
        uint64_t e, f;
        memcpy(&e, &local_energy[i], sizeof(e));
        memcpy(&f, &local_flux[i], sizeof(f));
        uint64_t global_index = static_cast<uint64_t>(first_row) * n + i;
        local_hash ^= (e + global_index) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, active);
    std::vector<int> counts, displacements;
    std::vector<double> all_energy, all_flux;
    if (rank == 0 && (validate || print_results_flag)) {
        counts.resize(active_size); displacements.resize(active_size);
        for (int r = 0; r < active_size; ++r) {
            counts[r] = (n / active_size + (r < n % active_size)) * n;
            displacements[r] = (r * (n / active_size) + std::min(r, n % active_size)) * n;
        }
        all_energy.resize(static_cast<size_t>(n) * n);
        if (validate) all_flux.resize(static_cast<size_t>(n) * n);
    }
    if (validate || print_results_flag)
        MPI_Gatherv(local_energy.data(), local_cells, MPI_DOUBLE, all_energy.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, active);
    if (validate)
        MPI_Gatherv(local_flux.data(), local_cells, MPI_DOUBLE, all_flux.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, active);
    int valid = 1;
    if (rank == 0) {
        double duration_ms = max_elapsed * 1000.0;
        int measured = std::max(iterations, 0);
        double rate = measured * static_cast<double>(n) * n / max_elapsed / 1e9;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration_ms / std::max(measured, 1));
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(hash));
        if (print_results_flag) print_results(all_energy, "ElementEnergy");
        if (validate) valid = validateResults(all_energy, all_flux);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, active);
    for (int b = 0; b < 2; ++b) { CUDA(cudaFree(energy[b])); CUDA(cudaFree(flux[b])); }
    if (send_buffer) CUDA(cudaFreeHost(send_buffer));
    if (recv_buffer) CUDA(cudaFreeHost(recv_buffer));
    CUDA(cudaEventDestroy(ready)); CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(exchange));
    MPI_Comm_free(&active);
    MPI_Finalize();
    return valid ? 0 : 1;
}
