#include <algorithm>
#include <climits>
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

using val_t = double;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Rows 1..rows are owned; rows 0 and rows+1 contain the neighboring ranks' halos.
// The four additions are deliberately in the same order as the original mesh.
__global__ void update(const val_t* energy, val_t* next, val_t* accumulated,
                       int n, int rows, int first_row, int mode) {
    size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = mode == 1 ? static_cast<size_t>(rows > 2 ? rows - 2 : 0) * n :
                   mode == 2 ? static_cast<size_t>(rows > 1 ? 2 : 1) * n :
                               static_cast<size_t>(rows) * n;
    if (t >= count) return;
    int row = static_cast<int>(t / n) + 1;
    int col = static_cast<int>(t % n);
    if (mode == 1) ++row;
    if (mode == 2 && row == 2) row = rows;
    int global_row = first_row + row - 1;
    size_t p = static_cast<size_t>(row) * n + col;
    val_t e = energy[p];
    val_t flux = 0.0;
    if ((global_row == 0 || global_row == n - 1) && (col == 0 || col == n - 1)) {
        // The last corner assignment wins when n == 1.
        flux = (global_row == col || n == 1) ? 0.5 : -0.5;
    }
    if (global_row + 1 < n) flux += ((energy[p + n] - e) * 0.8) * 0.25;
    if (global_row > 0)     flux += ((energy[p - n] - e) * 0.8) * 0.25;
    if (col + 1 < n)        flux += ((energy[p + 1] - e) * 0.8) * 0.25;
    if (col > 0)            flux += ((energy[p - 1] - e) * 0.8) * 0.25;
    next[p] = e + flux;
    accumulated[p] += fabs(flux);
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512, iterations = 10;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!world_rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || static_cast<long long>(n) * n > INT_MAX) {
        if (!world_rank) fprintf(stderr, "Grid size must be positive and contain at most INT_MAX elements\n");
        MPI_Finalize(); return 1;
    }

    // Exclude empty partitions, but keep every process in MPI until the work is done.
    int active_size = std::min(n, world_size);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED, world_rank, &active);
    if (world_rank >= active_size) { MPI_Finalize(); return 0; }
    int rank, size;
    MPI_Comm_rank(active, &rank);
    MPI_Comm_size(active, &size);
    int rows = n / size + (rank < n % size);
    int first = rank * (n / size) + std::min(rank, n % size);
    int prev = rank ? rank - 1 : MPI_PROC_NULL;
    int next_rank = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    MPI_Comm local;
    MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local);
    int local_rank;
    MPI_Comm_rank(local, &local_rank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local);

    size_t padded = static_cast<size_t>(rows + 2) * n;
    size_t owned = static_cast<size_t>(rows) * n;
    size_t row_bytes = static_cast<size_t>(n) * sizeof(val_t);
    val_t *energy, *other, *flux, *send_top, *send_bottom, *recv_top, *recv_bottom;
    CUDA_CHECK(cudaMalloc(&energy, padded * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&other, padded * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&flux, padded * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&send_top, row_bytes));
    CUDA_CHECK(cudaMallocHost(&send_bottom, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_top, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_bottom, row_bytes));
    cudaStream_t compute, comm;
    cudaEvent_t ready, halos_ready;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemsetAsync(energy, 0, padded * sizeof(val_t), compute));
    CUDA_CHECK(cudaMemsetAsync(other, 0, padded * sizeof(val_t), compute));
    CUDA_CHECK(cudaMemsetAsync(flux, 0, padded * sizeof(val_t), compute));
    CUDA_CHECK(cudaEventRecord(ready, compute));
    CUDA_CHECK(cudaStreamSynchronize(compute));

    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\n\n", n, n,
               static_cast<long long>(n) * n, iterations, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        printf("MPI ranks: %d, implicit grid connectivity\n\nRunning simulation...\n", size);
    }
    MPI_Barrier(active);
    double start = MPI_Wtime();
    constexpr int block = 256;
    for (int iter = 0; iter < iterations; ++iter) {
        if (size == 1) {
            update<<<static_cast<unsigned>((owned + block - 1) / block), block, 0, compute>>>(energy, other, flux, n, rows, first, 0);
            CUDA_CHECK(cudaGetLastError());
        } else {
            MPI_Request requests[4];
            int nr = 0;
            // The previous asynchronous host-to-device copies must finish
            // before MPI is allowed to reuse their pinned receive buffers.
            if (iter > 0) CUDA_CHECK(cudaEventSynchronize(halos_ready));
            if (prev != MPI_PROC_NULL) MPI_Irecv(recv_top, n, MPI_DOUBLE, prev, 1, active, &requests[nr++]);
            if (next_rank != MPI_PROC_NULL) MPI_Irecv(recv_bottom, n, MPI_DOUBLE, next_rank, 0, active, &requests[nr++]);
            CUDA_CHECK(cudaStreamWaitEvent(comm, ready, 0));
            if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(send_top, energy + n, row_bytes, cudaMemcpyDeviceToHost, comm));
            if (next_rank != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(send_bottom, energy + static_cast<size_t>(rows) * n, row_bytes, cudaMemcpyDeviceToHost, comm));
            size_t interior = static_cast<size_t>(rows > 2 ? rows - 2 : 0) * n;
            if (interior) {
                update<<<static_cast<unsigned>((interior + block - 1) / block), block, 0, compute>>>(energy, other, flux, n, rows, first, 1);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaStreamSynchronize(comm));
            if (prev != MPI_PROC_NULL) MPI_Isend(send_top, n, MPI_DOUBLE, prev, 0, active, &requests[nr++]);
            if (next_rank != MPI_PROC_NULL) MPI_Isend(send_bottom, n, MPI_DOUBLE, next_rank, 1, active, &requests[nr++]);
            MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
            if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(energy, recv_top, row_bytes, cudaMemcpyHostToDevice, comm));
            if (next_rank != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(energy + static_cast<size_t>(rows + 1) * n, recv_bottom, row_bytes, cudaMemcpyHostToDevice, comm));
            CUDA_CHECK(cudaEventRecord(halos_ready, comm));
            CUDA_CHECK(cudaStreamWaitEvent(compute, halos_ready, 0));
            size_t boundary = static_cast<size_t>(rows > 1 ? 2 : 1) * n;
            update<<<static_cast<unsigned>((boundary + block - 1) / block), block, 0, compute>>>(energy, other, flux, n, rows, first, 2);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ready, compute));
        }
        std::swap(energy, other);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active);

    std::vector<val_t> local_energy(owned), local_flux(owned);
    CUDA_CHECK(cudaMemcpy(local_energy.data(), energy + n, owned * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_flux.data(), flux + n, owned * sizeof(val_t), cudaMemcpyDeviceToHost));
    unsigned long long local_hash = 0;
    #pragma omp parallel for reduction(^:local_hash) schedule(static)
    for (size_t i = 0; i < owned; ++i) {
        uint64_t e, f;
        memcpy(&e, &local_energy[i], sizeof(e));
        memcpy(&f, &local_flux[i], sizeof(f));
        uint64_t global = static_cast<uint64_t>(first) * n + i;
        local_hash ^= (e + global) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f + global) * 0xbf58476d1ce4e5b9ULL;
    }
    unsigned long long hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, active);
    std::vector<int> counts, displacements;
    std::vector<val_t> all_energy, all_flux;
    if ((validate || results) && !rank) {
        counts.resize(size); displacements.resize(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = (n / size + (r < n % size)) * n;
            displacements[r] = (r * (n / size) + std::min(r, n % size)) * n;
        }
        all_energy.resize(static_cast<size_t>(n) * n);
        if (validate) all_flux.resize(all_energy.size());
    }
    if (validate || results)
        MPI_Gatherv(local_energy.data(), static_cast<int>(owned), MPI_DOUBLE, all_energy.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, active);
    if (validate)
        MPI_Gatherv(local_flux.data(), static_cast<int>(owned), MPI_DOUBLE, all_flux.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, active);
    int valid = 1;
    if (!rank) {
        long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        int measured = std::max(iterations, 0);
        double time_per_iter = max_elapsed * 1000.0 / std::max(measured, 1);
        double rate = max_elapsed > 0.0 ? (static_cast<double>(measured) * n * n) / max_elapsed / 1e9 : 0.0;
        printf("Computation time: %ld ms\nPerformance:\n", duration_ms);
        printf("  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n", time_per_iter, rate);
        printf("  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", rate * 22.0, hash);
        if (results) print_results(all_energy, "ElementEnergy");
        if (validate) {
            val_t energy_sum = 0.0, flux_sum = 0.0;
            val_t energy_max = std::numeric_limits<val_t>::lowest(), energy_min = std::numeric_limits<val_t>::max();
            for (size_t i = 0; i < all_energy.size(); ++i) {
                energy_sum += all_energy[i]; flux_sum += all_flux[i];
                energy_max = std::max(all_energy[i], energy_max); energy_min = std::min(all_energy[i], energy_min);
            }
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum, energy_min, energy_max);
            if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); valid = 0; }
            else if (std::abs(energy_sum) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); valid = 0; }
            if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) { printf("  ERROR: Energy extrema are not finite\n"); valid = 0; }
            if (valid) printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, active);
    CUDA_CHECK(cudaEventDestroy(ready)); CUDA_CHECK(cudaEventDestroy(halos_ready));
    CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(comm));
    CUDA_CHECK(cudaFree(energy)); CUDA_CHECK(cudaFree(other)); CUDA_CHECK(cudaFree(flux));
    CUDA_CHECK(cudaFreeHost(send_top)); CUDA_CHECK(cudaFreeHost(send_bottom));
    CUDA_CHECK(cudaFreeHost(recv_top)); CUDA_CHECK(cudaFreeHost(recv_bottom));
    MPI_Comm_free(&active);
    MPI_Finalize();
    return valid ? 0 : 1;
}
