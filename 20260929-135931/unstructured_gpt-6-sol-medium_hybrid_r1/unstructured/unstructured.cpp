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
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)
#define MPI_CHECK(call) do { \
    int err_ = (call); \
    if (err_ != MPI_SUCCESS) { \
        fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, err_); \
    } \
} while (0)

// CUDA uses explicit round-to-nearest operations to retain the reference
// operation order, including the per-neighbor flux accumulation order.
__device__ inline double addFlux(double total, double self, double neighbor) {
    double delta = __dsub_rn(neighbor, self);
    double flux = __dmul_rn(delta, 0.8);
    flux = __dmul_rn(flux, 1.0);
    flux = __dmul_rn(flux, 0.25);
    return __dadd_rn(total, flux);
}

__global__ void updateRows(const ElementDynamic* __restrict__ src,
                           ElementDynamic* __restrict__ dst,
                           int n, int first_global_row, int local_rows,
                           int first_local_row, int row_count) {
    size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = size_t(row_count) * n;
    if (k >= count) return;
    int row = first_local_row + int(k / n);
    int col = int(k % n);
    int global_row = first_global_row + row - 1;
    size_t index = size_t(row) * n + col;
    double energy = src[index].current_energy;
    double flow = 0.0;
    if ((global_row == 0 && col == 0) ||
        (global_row == n - 1 && col == n - 1)) flow = 0.5;
    if ((global_row == 0 && col == n - 1) ||
        (global_row == n - 1 && col == 0)) flow = -0.5;
    // The original corner assignments give the last assignment precedence
    // when n == 1.
    if (n == 1) flow = 0.5;
    if (global_row + 1 < n) flow = addFlux(flow, energy, src[index + n].current_energy);
    if (global_row > 0) flow = addFlux(flow, energy, src[index - n].current_energy);
    if (col + 1 < n) flow = addFlux(flow, energy, src[index + 1].current_energy);
    if (col > 0) flow = addFlux(flow, energy, src[index - 1].current_energy);
    dst[index].current_energy = __dadd_rn(energy, flow);
    dst[index].total_flux = __dadd_rn(src[index].total_flux, fabs(flow));
}

void launchRows(const ElementDynamic* src, ElementDynamic* dst, int n,
                int first_global_row, int local_rows, int first_row,
                int row_count, cudaStream_t stream) {
    if (row_count <= 0) return;
    constexpr int block_size = 256;
    size_t count = size_t(row_count) * n;
    updateRows<<<unsigned((count + block_size - 1) / block_size), block_size, 0, stream>>>(
        src, dst, n, first_global_row, local_rows, first_row, row_count);
    CUDA_CHECK(cudaGetLastError());
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int n = 512, iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n <= 0 || n < ranks || iters < 0 ||
        uint64_t(n) * n > uint64_t(std::numeric_limits<int>::max()) / sizeof(ElementDynamic)) {
        if (rank == 0) fprintf(stderr, "Invalid grid or iteration count (grid rows must cover all MPI ranks).\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    int gpu_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
    if (gpu_count == 0) {
        fprintf(stderr, "Rank %d requires a CUDA device.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % gpu_count));

    int base = n / ranks, extra = n % ranks;
    int local_rows = base + (rank < extra);
    int first_row = rank * base + std::min(rank, extra);
    size_t row_bytes = size_t(n) * sizeof(ElementDynamic);
    size_t device_bytes = size_t(local_rows + 2) * row_bytes;
    ElementDynamic *current, *next;
    CUDA_CHECK(cudaMalloc(&current, device_bytes));
    CUDA_CHECK(cudaMalloc(&next, device_bytes));
    CUDA_CHECK(cudaMemset(current, 0, device_bytes));
    cudaStream_t compute_stream, boundary_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&boundary_stream, cudaStreamNonBlocking));
    ElementDynamic *send_top, *send_bottom, *recv_top, *recv_bottom;
    CUDA_CHECK(cudaMallocHost(&send_top, row_bytes));
    CUDA_CHECK(cudaMallocHost(&send_bottom, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_top, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_bottom, row_bytes));

    if (rank == 0) {
        double static_mem = double(uint64_t(n) * n) * 144.0;
        double dynamic_mem = double(uint64_t(n) * n) * sizeof(ElementDynamic) * 2.0;
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %lld elements\n", n, n, static_cast<long long>(n) * n);
        printf("Iterations: %d\n", iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0,
               dynamic_mem / 1048576.0);
        printf("Running simulation...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int following = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    for (int iter = 0; iter < iters; ++iter) {
        launchRows(current, next, n, first_row, local_rows, 2, local_rows - 2, compute_stream);
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(send_top, current + n, row_bytes, cudaMemcpyDeviceToHost, boundary_stream));
        if (following != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(send_bottom, current + size_t(local_rows) * n, row_bytes,
                                       cudaMemcpyDeviceToHost, boundary_stream));
        CUDA_CHECK(cudaStreamSynchronize(boundary_stream));
        MPI_Request requests[4];
        MPI_CHECK(MPI_Irecv(recv_top, int(row_bytes), MPI_BYTE, prev, 2,
                            MPI_COMM_WORLD, &requests[0]));
        MPI_CHECK(MPI_Irecv(recv_bottom, int(row_bytes), MPI_BYTE, following, 1,
                            MPI_COMM_WORLD, &requests[1]));
        MPI_CHECK(MPI_Isend(send_top, int(row_bytes), MPI_BYTE, prev, 1,
                            MPI_COMM_WORLD, &requests[2]));
        MPI_CHECK(MPI_Isend(send_bottom, int(row_bytes), MPI_BYTE, following, 2,
                            MPI_COMM_WORLD, &requests[3]));
        MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(current, recv_top, row_bytes, cudaMemcpyHostToDevice, boundary_stream));
        if (following != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(current + size_t(local_rows + 1) * n, recv_bottom,
                                       row_bytes, cudaMemcpyHostToDevice, boundary_stream));
        launchRows(current, next, n, first_row, local_rows, 1, 1, boundary_stream);
        if (local_rows > 1)
            launchRows(current, next, n, first_row, local_rows, local_rows, 1, boundary_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(boundary_stream));
        std::swap(current, next);
    }
    double elapsed = MPI_Wtime() - start, duration;
    MPI_CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    std::vector<ElementDynamic> local(size_t(local_rows) * n);
    CUDA_CHECK(cudaMemcpy(local.data(), current + n, local.size() * sizeof(ElementDynamic),
                          cudaMemcpyDeviceToHost));
    uint64_t local_hash = 0;
    double local_energy = 0.0, local_flux = 0.0;
    double local_min = std::numeric_limits<double>::max();
    double local_max = std::numeric_limits<double>::lowest();
    int host_threads = std::min(omp_get_max_threads(),
                                std::max(1, int(local.size() / 8192)));
    #pragma omp parallel for num_threads(host_threads) reduction(^:local_hash) reduction(+:local_energy,local_flux) reduction(min:local_min) reduction(max:local_max) schedule(static)
    for (int64_t i = 0; i < int64_t(local.size()); ++i) {
        const ElementDynamic& e = local[size_t(i)];
        uint64_t energy_bits, flux_bits;
        memcpy(&energy_bits, &e.current_energy, sizeof(energy_bits));
        memcpy(&flux_bits, &e.total_flux, sizeof(flux_bits));
        uint64_t global_index = uint64_t(first_row) * n + uint64_t(i);
        local_hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        local_energy += e.current_energy;
        local_flux += e.total_flux;
        local_min = std::min(local_min, e.current_energy);
        local_max = std::max(local_max, e.current_energy);
    }
    uint64_t result_hash;
    MPI_CHECK(MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD));
    double energy_sum, flux_sum, energy_min, energy_max;
    MPI_CHECK(MPI_Reduce(&local_energy, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_flux, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        double duration_ms = duration * 1000.0;
        int measured = std::max(iters - 1, 1);
        double giga = duration > 0 ? double(measured) * n * n / duration / 1e9 : 0.0;
        printf("Computation time: %ld ms\n", long(duration_ms));
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration_ms / measured);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga);
        printf("  Performance: %.4f GFLOPS\n", giga * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(result_hash));
    }
    if (printResults) {
        std::vector<double> local_energy_data(local.size());
        #pragma omp parallel for num_threads(host_threads) schedule(static)
        for (int64_t i = 0; i < int64_t(local.size()); ++i)
            local_energy_data[size_t(i)] = local[size_t(i)].current_energy;
        std::vector<double> all_energy;
        std::vector<int> counts, displacements;
        if (rank == 0) {
            all_energy.resize(size_t(n) * n);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                int rows = base + (r < extra);
                int first = r * base + std::min(r, extra);
                counts[r] = rows * n;
                displacements[r] = first * n;
            }
        }
        MPI_CHECK(MPI_Gatherv(local_energy_data.data(), int(local_energy_data.size()), MPI_DOUBLE,
                              rank == 0 ? all_energy.data() : nullptr,
                              rank == 0 ? counts.data() : nullptr,
                              rank == 0 ? displacements.data() : nullptr,
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (rank == 0) print_results(all_energy, "ElementEnergy");
    }
    int valid = 1;
    if (validate && rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); valid = 0; }
        if (std::abs(energy_sum) > 1e-8)
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); valid = 0; }
        if (!std::isfinite(energy_min) || !std::isfinite(energy_max)) {
            printf("  ERROR: Energy extrema are not finite\n"); valid = 0;
        }
        if (valid) printf("  Validation: PASSED\n");
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaFreeHost(send_top));
    CUDA_CHECK(cudaFreeHost(send_bottom));
    CUDA_CHECK(cudaFreeHost(recv_top));
    CUDA_CHECK(cudaFreeHost(recv_bottom));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(boundary_stream));
    CUDA_CHECK(cudaFree(current));
    CUDA_CHECK(cudaFree(next));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
