#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using val_t = double;

// Abort all ranks on a CUDA error, so that no rank is left waiting in MPI.
static void cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank, file, line,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

// The mesh builder creates a square grid, with neighbors in this order:
// x+1, x-1, y+1, y-1. Rows 0 and rows+1 are the halo rows.
__global__ void update(const val_t* energy, const val_t* flux, val_t* next_energy,
                       val_t* next_flux, int n, int global_first_row,
                       int first_row, int row_count) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(row_count) * n;
    if (k >= count) return;
    const int row = first_row + k / n;
    const int col = k % n;
    const int global_row = global_first_row + row - 1;
    const size_t pos = static_cast<size_t>(row) * n + col;
    const val_t current = energy[pos];
    val_t total = 0.0;
    // Assignment order matches buildSquare2D, including the n=1 case.
    if ((global_row == 0 && col == 0) || (global_row == n - 1 && col == n - 1))
        total = 0.5;
    else if ((global_row == 0 && col == n - 1) || (global_row == n - 1 && col == 0))
        total = -0.5;
    if (global_row + 1 < n) total += (energy[pos + n] - current) * 0.8 * 1.0 * 0.25;
    if (global_row > 0)     total += (energy[pos - n] - current) * 0.8 * 1.0 * 0.25;
    if (col + 1 < n)        total += (energy[pos + 1] - current) * 0.8 * 1.0 * 0.25;
    if (col > 0)            total += (energy[pos - 1] - current) * 0.8 * 1.0 * 0.25;
    next_energy[pos] = current + total;
    next_flux[pos] = flux[pos] + fabs(total);
}

// Reduce the exact bitwise hash on the GPU, then finish the small reduction on
// the host. Global indices are used so the hash is independent of partitioning.
__global__ void hashBlocks(const val_t* energy, const val_t* flux, size_t count,
                           size_t global_offset, unsigned long long* partial) {
    __shared__ unsigned long long sums[256];
    unsigned long long h = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        unsigned long long index = global_offset + i;
        h ^= (static_cast<unsigned long long>(__double_as_longlong(energy[i])) + index)
             * 0x9e3779b97f4a7c15ULL;
        h ^= (static_cast<unsigned long long>(__double_as_longlong(flux[i])) + index)
             * 0xbf58476d1ce4e5b9ULL;
    }
    sums[threadIdx.x] = h;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
        if (threadIdx.x < step) sums[threadIdx.x] ^= sums[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}

static bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    // Keep the reference's serial accumulation order for its printed statistics.
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
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
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
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int n = 512, iterations = 10;
    bool validate = false, print_results_flag = false;
    int exit_code = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n <= 0 || static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (world_rank == 0) fprintf(stderr, "Grid size must be positive and N*N must fit in an int.\n");
        MPI_Finalize();
        return 1;
    }

    // Assign devices by node-local rank, even when ranks are spread over nodes.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    const int active_size = std::min(world_size, n);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_size) {
        MPI_Finalize();
        return 0;
    }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    const int rows = n / ranks + (rank < n % ranks);
    const int first = rank * (n / ranks) + std::min(rank, n % ranks);
    const size_t local_count = static_cast<size_t>(rows) * n;
    const size_t alloc_count = static_cast<size_t>(rows + 2) * n;
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(val_t);
    const int total_elements = n * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, total_elements);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        // The grid connectivity is implicit. Count device state, including halos.
        double static_mem = 0.0;
        double dynamic_mem = (static_cast<double>(total_elements) + 2.0 * n * ranks) * 4 * sizeof(val_t);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0, dynamic_mem / 1048576.0);
        printf("Running simulation...\n");
    }

    val_t *energy, *next_energy, *flux, *next_flux;
    CUDA_CHECK(cudaMalloc(&energy, alloc_count * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&next_energy, alloc_count * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&flux, alloc_count * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&next_flux, alloc_count * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(energy, 0, alloc_count * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(next_energy, 0, alloc_count * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(flux, 0, alloc_count * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(next_flux, 0, alloc_count * sizeof(val_t)));
    cudaStream_t compute_stream, halo_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&halo_stream, cudaStreamNonBlocking));
    val_t* halo = nullptr;
    if (ranks > 1) CUDA_CHECK(cudaMallocHost(&halo, 4 * row_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    constexpr int threads = 256;
    auto launch = [&](int begin, int count, cudaStream_t stream) {
        if (count > 0) {
            const size_t elements = static_cast<size_t>(count) * n;
            update<<<static_cast<unsigned>((elements + threads - 1) / threads), threads, 0, stream>>>(
                energy, flux, next_energy, next_flux, n, first, begin, count);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launch(1, rows, compute_stream);
        } else {
            MPI_Request requests[4];
            int request_count = 0;
            if (rank > 0) {
                MPI_Irecv(halo, n, MPI_DOUBLE, rank - 1, 1, comm, &requests[request_count++]);
                CUDA_CHECK(cudaMemcpyAsync(halo + 2 * n, energy + n, row_bytes, cudaMemcpyDeviceToHost, halo_stream));
            }
            if (rank + 1 < ranks) {
                MPI_Irecv(halo + n, n, MPI_DOUBLE, rank + 1, 0, comm, &requests[request_count++]);
                CUDA_CHECK(cudaMemcpyAsync(halo + 3 * n, energy + static_cast<size_t>(rows) * n,
                                           row_bytes, cudaMemcpyDeviceToHost, halo_stream));
            }
            launch(2, rows - 2, compute_stream);
            CUDA_CHECK(cudaStreamSynchronize(halo_stream));
            if (rank > 0) MPI_Isend(halo + 2 * n, n, MPI_DOUBLE, rank - 1, 0, comm, &requests[request_count++]);
            if (rank + 1 < ranks) MPI_Isend(halo + 3 * n, n, MPI_DOUBLE, rank + 1, 1, comm, &requests[request_count++]);
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
            if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(energy, halo, row_bytes, cudaMemcpyHostToDevice, halo_stream));
            if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(energy + static_cast<size_t>(rows + 1) * n,
                                                              halo + n, row_bytes, cudaMemcpyHostToDevice, halo_stream));
            launch(1, 1, halo_stream);
            if (rows > 1) launch(rows, 1, halo_stream);
            CUDA_CHECK(cudaStreamSynchronize(halo_stream));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        }
        std::swap(energy, next_energy);
        std::swap(flux, next_flux);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    int blocks = static_cast<int>(std::min<size_t>((local_count + threads - 1) / threads, 4096));
    unsigned long long* device_hash;
    CUDA_CHECK(cudaMalloc(&device_hash, blocks * sizeof(*device_hash)));
    hashBlocks<<<blocks, threads>>>(energy + n, flux + n, local_count,
                                    static_cast<size_t>(first) * n, device_hash);
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned long long> partial(blocks);
    CUDA_CHECK(cudaMemcpy(partial.data(), device_hash, blocks * sizeof(*device_hash), cudaMemcpyDeviceToHost));
    unsigned long long local_hash = 0, hash = 0;
    const int hash_threads = std::min(omp_get_max_threads(), std::max(1, blocks / 64));
    #pragma omp parallel for reduction(^:local_hash) num_threads(hash_threads)
    for (int i = 0; i < blocks; ++i) local_hash ^= partial[i];
    MPI_Reduce(&local_hash, &hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, comm);

    std::vector<val_t> all_energy, all_flux;
    if (validate || print_results_flag) {
        std::vector<val_t> local_energy(local_count);
        CUDA_CHECK(cudaMemcpy(local_energy.data(), energy + n, local_count * sizeof(val_t), cudaMemcpyDeviceToHost));
        std::vector<int> counts(ranks), displacements(ranks);
        #pragma omp parallel for if(ranks > 512)
        for (int r = 0; r < ranks; ++r) {
            counts[r] = (n / ranks + (r < n % ranks)) * n;
            displacements[r] = (r * (n / ranks) + std::min(r, n % ranks)) * n;
        }
        if (rank == 0) all_energy.resize(total_elements);
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_count), MPI_DOUBLE,
                    all_energy.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, comm);
        if (validate) {
            std::vector<val_t> local_flux(local_count);
            CUDA_CHECK(cudaMemcpy(local_flux.data(), flux + n, local_count * sizeof(val_t), cudaMemcpyDeviceToHost));
            if (rank == 0) all_flux.resize(total_elements);
            MPI_Gatherv(local_flux.data(), static_cast<int>(local_count), MPI_DOUBLE,
                        all_flux.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, comm);
        }
    }
    if (rank == 0) {
        double duration_ms = max_elapsed * 1000.0;
        int measured_iters = std::max(iterations - 1, 1);
        double giga_elems_per_sec = measured_iters * static_cast<double>(total_elements) / max_elapsed / 1e9;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration_ms / measured_iters);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016llX\n\n", hash);
        if (print_results_flag) print_results(all_energy, "ElementEnergy");
        if (validate && !validateResults(all_energy, all_flux)) exit_code = 1;
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, comm);
    CUDA_CHECK(cudaFree(device_hash));
    if (halo) CUDA_CHECK(cudaFreeHost(halo));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(halo_stream));
    CUDA_CHECK(cudaFree(energy));
    CUDA_CHECK(cudaFree(next_energy));
    CUDA_CHECK(cudaFree(flux));
    CUDA_CHECK(cudaFree(next_flux));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return exit_code;
}
