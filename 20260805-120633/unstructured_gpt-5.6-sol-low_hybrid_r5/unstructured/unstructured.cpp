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
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(e_));                                  \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

struct Partition {
    int first_row;
    int rows;
};

static Partition partition_rows(int n, int rank, int nranks) {
    const int base = n / nranks;
    const int extra = n % nranks;
    return {rank * base + std::min(rank, extra), base + (rank < extra)};
}

// The two ghost rows make the local stencil independent of rank boundaries.
// Separate arrays give coalesced GPU accesses and avoid transferring total_flux.
__global__ void update_rows(const val_t* __restrict__ energy,
                            const val_t* __restrict__ flux,
                            val_t* __restrict__ next_energy,
                            val_t* __restrict__ next_flux,
                            int n, int first_global_row, int local_rows,
                            int begin_row, int end_row) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = begin_row + blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n || row >= end_row) return;

    const int p = row * n + col;
    const int global_row = first_global_row + row - 1;
    const val_t center = energy[p];
    val_t delta = 0.0;
    if (global_row + 1 < n) delta += energy[p + n] - center;
    if (global_row > 0)     delta += energy[p - n] - center;
    if (col + 1 < n)        delta += energy[p + 1] - center;
    if (col > 0)            delta += energy[p - 1] - center;

    val_t external = 0.0;
    if ((global_row == 0 || global_row == n - 1) &&
        (col == 0 || col == n - 1)) {
        external = (global_row == col) ? 0.5 : -0.5;
    }
    const val_t total = external + 0.2 * delta;
    next_energy[p] = center + total;
    next_flux[p] = flux[p] + fabs(total);
}

static void launch_rows(const val_t* energy, const val_t* flux,
                        val_t* next_energy, val_t* next_flux, int n,
                        const Partition& part, int begin, int end,
                        cudaStream_t stream) {
    if (begin >= end) return;
    const dim3 block(32, 8);
    const dim3 grid((n + block.x - 1) / block.x,
                    (end - begin + block.y - 1) / block.y);
    update_rows<<<grid, block, 0, stream>>>(energy, flux, next_energy, next_flux,
                                            n, part.first_row, part.rows,
                                            begin, end);
    CUDA_CHECK(cudaGetLastError());
}

static void printUsage(const char* name, int rank) {
    if (rank != 0) return;
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -n <num>  Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>  Number of iterations (default: 10)\n");
    std::printf("  -v        Enable validation\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show this help message\n");
}

static uint64_t computeHash(const std::vector<val_t>& energy,
                            const std::vector<val_t>& flux, size_t global_offset) {
    uint64_t result = 0;
    #pragma omp parallel for reduction(^:result) schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(energy.size()); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        uint64_t e, f;
        std::memcpy(&e, &energy[i], sizeof(e));
        std::memcpy(&f, &flux[i], sizeof(f));
        const size_t global_i = global_offset + i;
        result ^= (e + global_i) * 0x9e3779b97f4a7c15ULL;
        result ^= (f + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false;
    bool args_ok = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else args_ok = false;
    }
    if (help || !args_ok) {
        printUsage(argv[0], rank);
        MPI_Finalize();
        return args_ok ? 0 : 1;
    }
    if (n <= 0 || iterations < 0 || nranks > n) {
        if (rank == 0) std::fprintf(stderr, "Grid must be positive, iterations nonnegative, and MPI ranks <= grid rows.\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) std::fprintf(stderr, "This benchmark unconditionally requires CUDA GPUs.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    const Partition part = partition_rows(n, rank, nranks);
    const size_t pitch_elems = static_cast<size_t>(part.rows + 2) * n;
    const size_t local_elems = static_cast<size_t>(part.rows) * n;
    val_t *d_energy[2], *d_flux[2];
    CUDA_CHECK(cudaMalloc(&d_energy[0], pitch_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy[1], pitch_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux[0], pitch_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux[1], pitch_elems * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMemset(d_energy[b], 0, pitch_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d_flux[b], 0, pitch_elems * sizeof(val_t)));
    }

    val_t *send_top, *send_bottom, *recv_top, *recv_bottom;
    CUDA_CHECK(cudaMallocHost(&send_top, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&send_bottom, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&recv_top, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&recv_bottom, n * sizeof(val_t)));
    cudaStream_t copy_stream, compute_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\n",
                    n, n, static_cast<long long>(n) * n, iterations,
                    validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI ranks, up to %d OpenMP threads/rank, CUDA\n\n",
                    nranks, omp_get_max_threads());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    int current = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        const int next = 1 - current;
        MPI_Request req[4];
        int count = 0;
        if (rank > 0) {
            CUDA_CHECK(cudaMemcpyAsync(send_top, d_energy[current] + n,
                                       n * sizeof(val_t), cudaMemcpyDeviceToHost, copy_stream));
        }
        if (rank + 1 < nranks) {
            CUDA_CHECK(cudaMemcpyAsync(send_bottom, d_energy[current] + static_cast<size_t>(part.rows) * n,
                                       n * sizeof(val_t), cudaMemcpyDeviceToHost, copy_stream));
        }
        launch_rows(d_energy[current], d_flux[current], d_energy[next], d_flux[next],
                    n, part, 2, part.rows, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));
        if (rank > 0) {
            MPI_Irecv(recv_top, n, MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &req[count++]);
            MPI_Isend(send_top, n, MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &req[count++]);
        }
        if (rank + 1 < nranks) {
            MPI_Irecv(recv_bottom, n, MPI_DOUBLE, rank + 1, 12, MPI_COMM_WORLD, &req[count++]);
            MPI_Isend(send_bottom, n, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &req[count++]);
        }
        if (count) MPI_Waitall(count, req, MPI_STATUSES_IGNORE);
        if (rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_energy[current], recv_top, n * sizeof(val_t),
                                       cudaMemcpyHostToDevice, copy_stream));
        if (rank + 1 < nranks)
            CUDA_CHECK(cudaMemcpyAsync(d_energy[current] + static_cast<size_t>(part.rows + 1) * n,
                                       recv_bottom, n * sizeof(val_t), cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));
        launch_rows(d_energy[current], d_flux[current], d_energy[next], d_flux[next],
                    n, part, 1, std::min(2, part.rows + 1), compute_stream);
        if (part.rows > 1)
            launch_rows(d_energy[current], d_flux[current], d_energy[next], d_flux[next],
                        n, part, part.rows, part.rows + 1, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        current = next;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> local_energy(local_elems), local_flux(local_elems);
    CUDA_CHECK(cudaMemcpy(local_energy.data(), d_energy[current] + n,
                          local_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_flux.data(), d_flux[current] + n,
                          local_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const Partition p = partition_rows(n, r, nranks);
        counts[r] = p.rows * n;
        displs[r] = p.first_row * n;
    }
    const uint64_t local_hash = computeHash(local_energy, local_flux,
                                             static_cast<size_t>(part.first_row) * n);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    std::vector<val_t> all_energy;
    if (printResults) {
        if (rank == 0) all_energy.resize(static_cast<size_t>(n) * n);
        MPI_Gatherv(local_energy.data(), counts[rank], MPI_DOUBLE, all_energy.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    val_t local_energy_sum = 0.0, local_flux_sum = 0.0;
    val_t local_min = std::numeric_limits<val_t>::max();
    val_t local_max = std::numeric_limits<val_t>::lowest();
    #pragma omp parallel for reduction(+:local_energy_sum,local_flux_sum) reduction(min:local_min) reduction(max:local_max) schedule(static)
    for (long long i = 0; i < static_cast<long long>(local_energy.size()); ++i) {
        local_energy_sum += local_energy[i];
        local_flux_sum += local_flux[i];
        local_min = std::min(local_min, local_energy[i]);
        local_max = std::max(local_max, local_energy[i]);
    }
    val_t energy_sum = 0.0, flux_sum = 0.0, energy_min = 0.0, energy_max = 0.0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int status = 0;
    if (rank == 0) {
        const double ms = seconds * 1000.0;
        const double updates = static_cast<double>(iterations) * n * n;
        const double geps = seconds > 0.0 ? updates / seconds / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\nPerformance:\n", ms);
        std::printf("  Time per iteration: %.4f ms\n", iterations ? ms / iterations : 0.0);
        std::printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
                    geps, geps * 22.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(global_hash));
        if (printResults) print_results(all_energy, "ElementEnergy");
        if (validate) {
            std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n", energy_sum, flux_sum);
            std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
            status = !(std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
                       std::isfinite(energy_min) && std::isfinite(energy_max));
            std::printf("  Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(copy_stream)); CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaFreeHost(send_top)); CUDA_CHECK(cudaFreeHost(send_bottom));
    CUDA_CHECK(cudaFreeHost(recv_top)); CUDA_CHECK(cudaFreeHost(recv_bottom));
    for (int b = 0; b < 2; ++b) { CUDA_CHECK(cudaFree(d_energy[b])); CUDA_CHECK(cudaFree(d_flux[b])); }
    MPI_Finalize();
    return status;
}
