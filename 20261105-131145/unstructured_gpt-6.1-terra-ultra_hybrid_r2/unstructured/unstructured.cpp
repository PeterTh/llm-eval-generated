#include <mpi.h>
#include <cuda_runtime.h>

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

static void cudaCheck(cudaError_t result, const char* expression, int line) {
    if (result != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: CUDA error at line %d (%s): %s\n",
                     rank, line, expression, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call, __LINE__)

// A rank owns complete rows, with one ghost row at each end. Connectivity
// is generated from the square mesh rather than stored for every element.
__global__ void updateRows(const val_t* energy, val_t* next, val_t* flux,
                           int n, int first_row, int begin, int end) {
    for (int row = begin + static_cast<int>(blockIdx.y); row < end;
         row += static_cast<int>(gridDim.y)) {
      for (size_t y0 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           y0 < static_cast<size_t>(n); y0 += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int y = static_cast<int>(y0);
        const int x = first_row + row;
        const size_t i = static_cast<size_t>(row + 1) * n + y;
        const val_t self = energy[i];
        val_t total = 0.0;
        // The last assignment in mesh construction wins when n == 1.
        if ((x == 0 && y == 0) || (x == n - 1 && y == n - 1))
            total = 0.5;
        else if ((x == 0 && y == n - 1) || (x == n - 1 && y == 0))
            total = -0.5;
        // Match the original +x, -x, +y, -y order and arithmetic.
        if (x + 1 < n) total += ((energy[i + n] - self) * 0.8) * 0.25;
        if (x > 0)     total += ((energy[i - n] - self) * 0.8) * 0.25;
        if (y + 1 < n) total += ((energy[i + 1] - self) * 0.8) * 0.25;
        if (y > 0)     total += ((energy[i - 1] - self) * 0.8) * 0.25;
        next[i] = self + total;
        flux[static_cast<size_t>(row) * n + y] += fabs(total);
      }
    }
}

static void launchRows(const val_t* energy, val_t* next, val_t* flux,
                       int n, int first_row, int begin, int end, cudaStream_t stream) {
    if (begin >= end) return;
    const dim3 blocks(static_cast<unsigned>(std::min<size_t>((static_cast<size_t>(n) + 255) / 256, 65535)),
                      static_cast<unsigned>(std::min(end - begin, 65535)));
    updateRows<<<blocks, 256, 0, stream>>>(energy, next, flux, n, first_row, begin, end);
    CUDA(cudaGetLastError());
}

// Reduce the XOR hash on the device, then combine the small block array on
// the host. Full state is transferred only when validation or output needs it.
__global__ void hashBlocks(const val_t* energy, const val_t* flux, size_t count,
                           uint64_t global_offset, uint64_t* hashes) {
    __shared__ uint64_t partial[256];
    uint64_t h = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const uint64_t index = global_offset + i;
        const uint64_t e = static_cast<uint64_t>(__double_as_longlong(energy[i]));
        const uint64_t f = static_cast<uint64_t>(__double_as_longlong(flux[i]));
        h ^= (e + index) * 0x9e3779b97f4a7c15ULL;
        h ^= (f + index) * 0xbf58476d1ce4e5b9ULL;
    }
    partial[threadIdx.x] = h;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] ^= partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) hashes[blockIdx.x] = partial[0];
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
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
    if (!std::isfinite(energy_sum)) {
        std::printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) {
        std::printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    std::printf("  Validation: PASSED\n");
    return true;
}

// Chunk transfers so result collection also works when a slab exceeds INT_MAX.
static void transferResults(const std::vector<val_t>& local, std::vector<val_t>& global,
                            int n, int rows, int first, int rank, int ranks, int tag,
                            MPI_Comm comm) {
    if (rank == 0) {
        std::copy(local.begin(), local.end(), global.begin());
        for (int r = 1; r < ranks; ++r) {
            const int rrows = n / ranks + (r < n % ranks);
            const int rfirst = r * (n / ranks) + std::min(r, n % ranks);
            size_t left = static_cast<size_t>(rrows) * n;
            size_t pos = static_cast<size_t>(rfirst) * n;
            while (left) {
                const int count = static_cast<int>(std::min<size_t>(left, 1u << 26));
                MPI_Recv(global.data() + pos, count, MPI_DOUBLE, r, tag, comm, MPI_STATUS_IGNORE);
                pos += count; left -= count;
            }
        }
    } else {
        size_t pos = 0, left = static_cast<size_t>(rows) * n;
        while (left) {
            const int count = static_cast<int>(std::min<size_t>(left, 1u << 26));
            MPI_Send(local.data() + pos, count, MPI_DOUBLE, 0, tag, comm);
            pos += count; left -= count;
        }
    }
    (void)first;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512, iters = 10;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (world_rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0) {
        if (world_rank == 0) std::fprintf(stderr, "Grid size must be positive\n");
        MPI_Finalize(); return 1;
    }

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);
    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    const int active_ranks = std::min(n, world_size);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_ranks) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    int device_count = 0;
    CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(local_rank % device_count));
    const int rows = n / ranks + (rank < n % ranks);
    const int first = rank * (n / ranks) + std::min(rank, n % ranks);
    const size_t local_size = static_cast<size_t>(rows) * n;
    const size_t grid_size = static_cast<size_t>(n) * n;
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(val_t);
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n, n, grid_size);
        std::printf("Iterations: %d\n", iters);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Building unstructured mesh...\n");
        std::printf("Running simulation...\n");
    }

    val_t *energy, *next, *flux, *halo = nullptr;
    CUDA(cudaMalloc(&energy, (static_cast<size_t>(rows) + 2) * row_bytes));
    CUDA(cudaMalloc(&next, (static_cast<size_t>(rows) + 2) * row_bytes));
    CUDA(cudaMalloc(&flux, local_size * sizeof(val_t)));
    CUDA(cudaMemset(energy, 0, (static_cast<size_t>(rows) + 2) * row_bytes));
    CUDA(cudaMemset(next, 0, (static_cast<size_t>(rows) + 2) * row_bytes));
    CUDA(cudaMemset(flux, 0, local_size * sizeof(val_t)));
    cudaStream_t compute, copy;
    cudaEvent_t ready, halos_ready;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));
    if (ranks > 1) CUDA(cudaMallocHost(&halo, 4 * row_bytes));
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iters; ++iter) {
        if (ranks == 1) {
            launchRows(energy, next, flux, n, first, 0, rows, compute);
        } else {
            // Send current boundary rows through pinned host buffers, allowing
            // the GPU to compute rows that do not depend on remote neighbors.
            CUDA(cudaEventRecord(ready, compute));
            CUDA(cudaStreamWaitEvent(copy, ready, 0));
            if (rank > 0) CUDA(cudaMemcpyAsync(halo, energy + n, row_bytes, cudaMemcpyDeviceToHost, copy));
            if (rank + 1 < ranks) CUDA(cudaMemcpyAsync(halo + n, energy + static_cast<size_t>(rows) * n, row_bytes, cudaMemcpyDeviceToHost, copy));
            MPI_Request requests[4];
            int count = 0;
            if (rank > 0)
                MPI_Irecv(halo + 2 * static_cast<size_t>(n), n, MPI_DOUBLE, rank - 1, 1, comm, &requests[count++]);
            if (rank + 1 < ranks)
                MPI_Irecv(halo + 3 * static_cast<size_t>(n), n, MPI_DOUBLE, rank + 1, 0, comm, &requests[count++]);
            launchRows(energy, next, flux, n, first, 1, rows - 1, compute);
            CUDA(cudaStreamSynchronize(copy));
            if (rank > 0) MPI_Isend(halo, n, MPI_DOUBLE, rank - 1, 0, comm, &requests[count++]);
            if (rank + 1 < ranks) MPI_Isend(halo + n, n, MPI_DOUBLE, rank + 1, 1, comm, &requests[count++]);
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
            if (rank > 0) CUDA(cudaMemcpyAsync(energy, halo + 2 * static_cast<size_t>(n), row_bytes, cudaMemcpyHostToDevice, copy));
            if (rank + 1 < ranks) CUDA(cudaMemcpyAsync(energy + static_cast<size_t>(rows + 1) * n, halo + 3 * static_cast<size_t>(n), row_bytes, cudaMemcpyHostToDevice, copy));
            CUDA(cudaEventRecord(halos_ready, copy));
            CUDA(cudaStreamWaitEvent(compute, halos_ready, 0));
            launchRows(energy, next, flux, n, first, 0, 1, compute);
            if (rows > 1) launchRows(energy, next, flux, n, first, rows - 1, rows, compute);
        }
        std::swap(energy, next);
    }
    CUDA(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const int hash_blocks = static_cast<int>(std::min<size_t>((local_size + 255) / 256, 4096));
    uint64_t* device_hashes;
    CUDA(cudaMalloc(&device_hashes, static_cast<size_t>(hash_blocks) * sizeof(uint64_t)));
    hashBlocks<<<hash_blocks, 256>>>(energy + n, flux, local_size,
                                    static_cast<uint64_t>(first) * n, device_hashes);
    CUDA(cudaGetLastError());
    std::vector<uint64_t> hashes(hash_blocks);
    CUDA(cudaMemcpy(hashes.data(), device_hashes, hashes.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA(cudaFree(device_hashes));
    uint64_t local_hash = 0;
    #pragma omp parallel for reduction(^:local_hash) schedule(static)
    for (int i = 0; i < hash_blocks; ++i) local_hash ^= hashes[i];
    std::vector<val_t> local_energy, local_flux;
    if (validate || print_results_flag) {
        local_energy.resize(local_size);
        CUDA(cudaMemcpy(local_energy.data(), energy + n, local_size * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    if (validate) {
        local_flux.resize(local_size);
        CUDA(cudaMemcpy(local_flux.data(), flux, local_size * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    std::vector<val_t> global_energy, global_flux;
    if (rank == 0 && (validate || print_results_flag)) global_energy.resize(grid_size);
    if (rank == 0 && validate) global_flux.resize(grid_size);
    if (validate || print_results_flag)
        transferResults(local_energy, global_energy, n, rows, first, rank, ranks, 10, comm);
    if (validate) transferResults(local_flux, global_flux, n, rows, first, rank, ranks, 11, comm);
    int valid = 1;
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        const int measured = std::max(iters - 1, 1);
        const double rate = measured * static_cast<double>(grid_size) / max_elapsed / 1e9;
        std::printf("Computation time: %ld ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", max_elapsed * 1000.0 / measured);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        std::printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
        std::printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
        if (print_results_flag) print_results(global_energy, "ElementEnergy");
        if (validate) valid = validateResults(global_energy, global_flux) ? 1 : 0;
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    if (halo) CUDA(cudaFreeHost(halo));
    CUDA(cudaEventDestroy(ready)); CUDA(cudaEventDestroy(halos_ready));
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(copy));
    CUDA(cudaFree(energy)); CUDA(cudaFree(next)); CUDA(cudaFree(flux));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
