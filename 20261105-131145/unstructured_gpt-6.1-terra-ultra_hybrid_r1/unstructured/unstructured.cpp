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

static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// Each rank owns complete rows. Energy has one ghost row at each end;
// accumulated flux is local and never needs to be communicated.
__global__ void update(const double* energy, const double* flux,
                       double* next_energy, double* next_flux,
                       int n, int first_row, int rows, int boundary) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(boundary ? (rows == 1 ? 1 : 2) : rows - 2) * n;
    if (tid >= count) return;
    int local_row = boundary ? (tid < static_cast<size_t>(n) ? 0 : rows - 1)
                             : static_cast<int>(tid / n) + 1;
    int y = static_cast<int>(tid % n);
    int x = first_row + local_row;
    size_t local = static_cast<size_t>(local_row) * n + y;
    size_t pos = local + n;
    double e = energy[pos];
    // The original corner assignments end with the bottom-right inflow,
    // including the degenerate one-element mesh.
    double total = 0.0;
    if ((x == 0 && y == 0) || (x == n - 1 && y == n - 1)) total = 0.5;
    else if ((x == 0 && y == n - 1) || (x == n - 1 && y == 0)) total = -0.5;
    // Preserve the original neighbor order and arithmetic for bitwise hashes.
    if (x + 1 < n) total += ((energy[pos + n] - e) * 0.8) * 0.25;
    if (x > 0)     total += ((energy[pos - n] - e) * 0.8) * 0.25;
    if (y + 1 < n) total += ((energy[pos + 1] - e) * 0.8) * 0.25;
    if (y > 0)     total += ((energy[pos - 1] - e) * 0.8) * 0.25;
    next_energy[pos] = e + total;
    next_flux[local] = flux[local] + fabs(total);
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

static bool validateResults(const std::vector<double>& energy, const std::vector<double>& flux) {
    double energy_sum = 0.0, flux_sum = 0.0;
    double energy_max = std::numeric_limits<double>::lowest();
    double energy_min = std::numeric_limits<double>::max();
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    // MPI_Gatherv uses int counts and displacements, as did the original
    // benchmark's integer element count.
    if (n <= 0 || static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and N*N must fit in an int\n");
        MPI_Finalize(); return 1;
    }
    const int n_elems = n * n;
    const int rows = n / size + (rank < n % size);
    const int first = rank * (n / size) + std::min(rank, n % size);
    const int local_count = rows * n;
    const int active = std::min(n, size);
    const int prev = rank > 0 && rows ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        const size_t device_mem = (4 * static_cast<size_t>(n_elems) +
                                   4 * static_cast<size_t>(active) * n) * sizeof(double);
        printf("GPU memory usage: %.2f MB (all ranks, including halo rows)\n\n",
               device_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    double *e[2] = {nullptr, nullptr}, *f[2] = {nullptr, nullptr};
    double *send_top = nullptr, *send_bottom = nullptr, *recv_top = nullptr, *recv_bottom = nullptr;
    cudaStream_t compute_stream = nullptr, transfer_stream = nullptr;
    cudaEvent_t ready = nullptr;
    if (rows) {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
        int local_rank, devices;
        MPI_Comm_rank(local_comm, &local_rank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA(cudaSetDevice(local_rank % devices));
        MPI_Comm_free(&local_comm);
        CUDA(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        for (int b = 0; b < 2; ++b) {
            CUDA(cudaMalloc(&e[b], static_cast<size_t>(rows + 2) * n * sizeof(double)));
            CUDA(cudaMalloc(&f[b], static_cast<size_t>(local_count) * sizeof(double)));
            CUDA(cudaMemsetAsync(e[b], 0, static_cast<size_t>(rows + 2) * n * sizeof(double), compute_stream));
            CUDA(cudaMemsetAsync(f[b], 0, static_cast<size_t>(local_count) * sizeof(double), compute_stream));
        }
        if (prev != MPI_PROC_NULL || next != MPI_PROC_NULL) {
            CUDA(cudaMallocHost(&send_top, static_cast<size_t>(4) * n * sizeof(double)));
            send_bottom = send_top + n; recv_top = send_bottom + n; recv_bottom = recv_top + n;
        }
        CUDA(cudaEventRecord(ready, compute_stream));
    } else {
        // split_type is collective, including ranks without rows.
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
        MPI_Comm_free(&local_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int cur = 0;
    constexpr int block = 256;
    for (int iter = 0; iter < n_iters; ++iter) {
        if (!rows) continue;
        int out = 1 - cur;
        if (rows > 2) {
            size_t count = static_cast<size_t>(rows - 2) * n;
            update<<<static_cast<unsigned>((count + block - 1) / block), block, 0, compute_stream>>>(
                e[cur], f[cur], e[out], f[out], n, first, rows, 0);
            CUDA(cudaGetLastError());
        }
        if (prev != MPI_PROC_NULL || next != MPI_PROC_NULL) {
            CUDA(cudaStreamWaitEvent(transfer_stream, ready, 0));
            size_t bytes = static_cast<size_t>(n) * sizeof(double);
            if (prev != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(send_top, e[cur] + n, bytes, cudaMemcpyDeviceToHost, transfer_stream));
            if (next != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(send_bottom, e[cur] + static_cast<size_t>(rows) * n, bytes, cudaMemcpyDeviceToHost, transfer_stream));
            MPI_Request requests[4]; int nr = 0;
            if (prev != MPI_PROC_NULL) MPI_Irecv(recv_top, n, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &requests[nr++]);
            if (next != MPI_PROC_NULL) MPI_Irecv(recv_bottom, n, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &requests[nr++]);
            CUDA(cudaStreamSynchronize(transfer_stream));
            if (prev != MPI_PROC_NULL) MPI_Isend(send_top, n, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &requests[nr++]);
            if (next != MPI_PROC_NULL) MPI_Isend(send_bottom, n, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &requests[nr++]);
            MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
            if (prev != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(e[cur], recv_top, bytes, cudaMemcpyHostToDevice, transfer_stream));
            if (next != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(e[cur] + static_cast<size_t>(rows + 1) * n, recv_bottom, bytes, cudaMemcpyHostToDevice, transfer_stream));
            CUDA(cudaEventRecord(ready, transfer_stream));
            CUDA(cudaStreamWaitEvent(compute_stream, ready, 0));
        }
        size_t count = static_cast<size_t>(rows == 1 ? 1 : 2) * n;
        update<<<static_cast<unsigned>((count + block - 1) / block), block, 0, compute_stream>>>(
            e[cur], f[cur], e[out], f[out], n, first, rows, 1);
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(ready, compute_stream));
        cur = out;
    }
    if (rows) CUDA(cudaStreamSynchronize(compute_stream));
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> local_energy(local_count), local_flux(local_count);
    if (rows) {
        CUDA(cudaMemcpy(local_energy.data(), e[cur] + n, static_cast<size_t>(local_count) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(local_flux.data(), f[cur], static_cast<size_t>(local_count) * sizeof(double), cudaMemcpyDeviceToHost));
    }
    uint64_t local_hash = 0;
    #pragma omp parallel for reduction(^:local_hash) schedule(static)
    for (int i = 0; i < local_count; ++i) {
        uint64_t ebits, fbits;
        memcpy(&ebits, &local_energy[i], sizeof(ebits));
        memcpy(&fbits, &local_flux[i], sizeof(fbits));
        uint64_t index = static_cast<uint64_t>(first) * n + i;
        local_hash ^= (ebits + index) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (fbits + index) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    std::vector<double> energy, flux;
    std::vector<int> counts, displs;
    if (printResults || validate) {
        if (rank == 0) {
            counts.resize(size); displs.resize(size);
            for (int r = 0; r < size; ++r) {
                counts[r] = (n / size + (r < n % size)) * n;
                displs[r] = (r * (n / size) + std::min(r, n % size)) * n;
            }
            energy.resize(n_elems);
            if (validate) flux.resize(n_elems);
        }
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE, flux.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int valid = 1;
    if (rank == 0) {
        auto duration_ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const int measured = std::max(n_iters - 1, 1);
        const double time_per_iter = max_elapsed * 1000.0 / measured;
        const double giga = (static_cast<double>(measured) * n_elems) / max_elapsed / 1e9;
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", time_per_iter, giga, giga * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(result_hash));
        if (printResults) print_results(energy, "ElementEnergy");
        if (validate) valid = validateResults(energy, flux);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rows) {
        if (send_top) CUDA(cudaFreeHost(send_top));
        for (int b = 0; b < 2; ++b) { CUDA(cudaFree(e[b])); CUDA(cudaFree(f[b])); }
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaStreamDestroy(compute_stream)); CUDA(cudaStreamDestroy(transfer_stream));
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
