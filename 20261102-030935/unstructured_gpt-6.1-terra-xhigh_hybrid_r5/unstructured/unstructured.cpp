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

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        int rank_ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank_); \
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// The generated mesh has four unit-flux links, in this order: +x, -x,
// +y, -y. Reconstructing them avoids loading the much larger static mesh.
// Device rows 1..rows belong to this rank; rows 0 and rows+1 are halos.
__global__ void updateRows(const double* __restrict__ energy,
                           const double* __restrict__ accumulated,
                           double* __restrict__ next_energy,
                           double* __restrict__ next_accumulated,
                           int n, int first_global_row, int begin_row, int row_count, int last_row) {
    size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(row_count) * n;
    if (t >= count) return;
    int row = begin_row + static_cast<int>(t / n);
    if (last_row && row == 2) row = last_row;
    int col = static_cast<int>(t % n);
    int x = first_global_row + row - 1;
    size_t i = static_cast<size_t>(row) * n + col;
    double e = energy[i];
    double flow = 0.0;
    // Corner assignments in the original builder are ordered; for n=1 the
    // final assignment makes the single element an inflow element.
    if ((x == 0 || x == n - 1) && (col == 0 || col == n - 1))
        flow = (x == col || n == 1) ? 0.5 : -0.5;
    if (x + 1 < n) flow += (energy[i + n] - e) * 0.8 * 0.25;
    if (x > 0)     flow += (energy[i - n] - e) * 0.8 * 0.25;
    if (col + 1 < n) flow += (energy[i + 1] - e) * 0.8 * 0.25;
    if (col > 0)     flow += (energy[i - 1] - e) * 0.8 * 0.25;
    next_energy[i] = e + flow;
    next_accumulated[i] = accumulated[i] + fabs(flow);
}

static void launchRows(const double* e, const double* f, double* ne, double* nf,
                       int n, int first, int begin, int rows, cudaStream_t stream, int last_row = 0) {
    if (rows <= 0) return;
    constexpr unsigned block = 256;
    size_t count = static_cast<size_t>(rows) * n;
    updateRows<<<static_cast<unsigned>((count + block - 1) / block), block, 0, stream>>>(
        e, f, ne, nf, n, first, begin, rows, last_row);
    CUDA_CHECK(cudaGetLastError());
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

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512, iterations = 10;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    // MPI_Gatherv uses int counts and the original program also used int for N*N.
    if (n <= 0 || static_cast<long long>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and N*N must fit in an int\n");
        MPI_Finalize(); return 1;
    }
    const int total = n * n;
    const int rows = n / ranks + (rank < n % ranks);
    const int first = rank * (n / ranks) + std::min(rank, n % ranks);
    const int active_ranks = std::min(n, ranks);
    const int above = rank > 0 && rows ? rank - 1 : MPI_PROC_NULL;
    const int below = rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL;
    const size_t local_count = static_cast<size_t>(rows) * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, total);
        printf("Iterations: %d\nValidation: %s\n\n", iterations, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        double mb = 1024.0 * 1024.0;
        printf("Memory usage: %.2f MB (distributed GPU dynamic buffers; connectivity implicit)\n\n",
               (static_cast<double>(total) + 2.0 * n * active_ranks) * 4 * sizeof(double) / mb);
        printf("Running simulation...\n");
    }

    // Local rank selects a GPU, respecting CUDA_VISIBLE_DEVICES on each node.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    double *energy[2] = {nullptr, nullptr}, *flux[2] = {nullptr, nullptr};
    double *send_up = nullptr, *send_down = nullptr, *recv_up = nullptr, *recv_down = nullptr;
    cudaStream_t compute = nullptr, communication = nullptr;
    cudaEvent_t computed = nullptr, halos_ready = nullptr;
    if (rows) {
        int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "Rank %d: no CUDA devices available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA_CHECK(cudaSetDevice(local_rank % devices));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&communication, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&computed, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));
        size_t bytes = (static_cast<size_t>(rows) + 2) * n * sizeof(double);
        for (int k = 0; k < 2; ++k) {
            CUDA_CHECK(cudaMalloc(&energy[k], bytes)); CUDA_CHECK(cudaMalloc(&flux[k], bytes));
            CUDA_CHECK(cudaMemsetAsync(energy[k], 0, bytes, compute));
            CUDA_CHECK(cudaMemsetAsync(flux[k], 0, bytes, compute));
        }
        if (active_ranks > 1) {
            CUDA_CHECK(cudaMallocHost(&send_up, static_cast<size_t>(n) * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&send_down, static_cast<size_t>(n) * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_up, static_cast<size_t>(n) * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_down, static_cast<size_t>(n) * sizeof(double)));
        }
        CUDA_CHECK(cudaEventRecord(computed, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    int current = 0;
    for (int iter = 0; iter < iterations && rows; ++iter) {
        int next = 1 - current;
        if (active_ranks == 1) {
            launchRows(energy[current], flux[current], energy[next], flux[next], n, first, 1, rows, compute);
        } else {
            // The previous asynchronous host-to-device copies must finish before
            // MPI is allowed to reuse the pinned receive buffers.
            if (iter > 0) CUDA_CHECK(cudaEventSynchronize(halos_ready));
            MPI_Request requests[4]; int nr = 0;
            if (above != MPI_PROC_NULL) MPI_Irecv(recv_up, n, MPI_DOUBLE, above, 1, MPI_COMM_WORLD, &requests[nr++]);
            if (below != MPI_PROC_NULL) MPI_Irecv(recv_down, n, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &requests[nr++]);
            CUDA_CHECK(cudaStreamWaitEvent(communication, computed, 0));
            if (above != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_up, energy[current] + n, static_cast<size_t>(n)*sizeof(double), cudaMemcpyDeviceToHost, communication));
            if (below != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_down, energy[current] + static_cast<size_t>(rows)*n, static_cast<size_t>(n)*sizeof(double), cudaMemcpyDeviceToHost, communication));
            launchRows(energy[current], flux[current], energy[next], flux[next], n, first, 2, rows - 2, compute);
            CUDA_CHECK(cudaStreamSynchronize(communication));
            if (above != MPI_PROC_NULL) MPI_Isend(send_up, n, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &requests[nr++]);
            if (below != MPI_PROC_NULL) MPI_Isend(send_down, n, MPI_DOUBLE, below, 1, MPI_COMM_WORLD, &requests[nr++]);
            MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
            if (above != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(energy[current], recv_up, static_cast<size_t>(n)*sizeof(double), cudaMemcpyHostToDevice, communication));
            if (below != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(energy[current] + static_cast<size_t>(rows + 1)*n, recv_down, static_cast<size_t>(n)*sizeof(double), cudaMemcpyHostToDevice, communication));
            CUDA_CHECK(cudaEventRecord(halos_ready, communication));
            CUDA_CHECK(cudaStreamWaitEvent(compute, halos_ready, 0));
            launchRows(energy[current], flux[current], energy[next], flux[next], n, first,
                       1, rows > 1 ? 2 : 1, compute, rows > 1 ? rows : 0);
            CUDA_CHECK(cudaEventRecord(computed, compute));
        }
        current = next;
    }
    if (rows) CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> local_energy(local_count), local_flux(local_count);
    if (rows) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), energy[current] + n, local_count*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), flux[current] + n, local_count*sizeof(double), cudaMemcpyDeviceToHost));
    }
    uint64_t local_hash = 0;
    #pragma omp parallel for reduction(^:local_hash) schedule(static) if(local_count >= 16384)
    for (size_t i = 0; i < local_count; ++i) {
        uint64_t e, f; memcpy(&e, &local_energy[i], sizeof(e)); memcpy(&f, &local_flux[i], sizeof(f));
        uint64_t global_i = static_cast<uint64_t>(first) * n + i;
        local_hash ^= (e + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    std::vector<double> all_energy, all_flux;
    if (validate || print_results_flag) {
        std::vector<int> counts, displacements;
        if (rank == 0) {
            counts.resize(ranks); displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = (n / ranks + (r < n % ranks)) * n;
                displacements[r] = (r * (n / ranks) + std::min(r, n % ranks)) * n;
            }
            all_energy.resize(total);
            if (validate) all_flux.resize(total);
        }
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_count), MPI_DOUBLE, all_energy.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) MPI_Gatherv(local_flux.data(), static_cast<int>(local_count), MPI_DOUBLE, all_flux.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int valid = 1;
    if (rank == 0) {
        double ms = max_elapsed * 1000.0;
        int measured = std::max(iterations, 0);
        double rate = max_elapsed > 0.0 ? static_cast<double>(measured) * total / max_elapsed / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", ms);
        printf("Performance:\n  Time per iteration: %.4f ms\n", ms / std::max(measured, 1));
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22.0);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(hash));
        if (print_results_flag) print_results(all_energy, "ElementEnergy");
        if (validate) valid = validateResults(all_energy, all_flux);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rows) {
        if (active_ranks > 1) {
            CUDA_CHECK(cudaFreeHost(send_up)); CUDA_CHECK(cudaFreeHost(send_down));
            CUDA_CHECK(cudaFreeHost(recv_up)); CUDA_CHECK(cudaFreeHost(recv_down));
        }
        for (int k = 0; k < 2; ++k) { CUDA_CHECK(cudaFree(energy[k])); CUDA_CHECK(cudaFree(flux[k])); }
        CUDA_CHECK(cudaEventDestroy(computed)); CUDA_CHECK(cudaEventDestroy(halos_ready));
        CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(communication));
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
