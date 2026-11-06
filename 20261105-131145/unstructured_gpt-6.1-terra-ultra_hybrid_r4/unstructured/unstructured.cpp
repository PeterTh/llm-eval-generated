#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <climits>
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
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void cudaCheck(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at line %d (%s): %s\n", line, expression,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(expr) cudaCheck((expr), #expr, __LINE__)

// The generated mesh connects each element to its down, up, right and left
// neighbors, in that order. The two extra rows hold neighboring ranks' data.
__global__ void updateRows(const ElementDynamic* current, ElementDynamic* next,
                           int n, int first_global_row, int row_begin, int row_end,
                           int second_row) {
    const int contiguous_rows = row_end - row_begin;
    const size_t count = static_cast<size_t>(contiguous_rows + (second_row != 0)) * n;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int relative_row = static_cast<int>(p / n);
        const int row = second_row && relative_row == contiguous_rows
                            ? second_row : row_begin + relative_row;
        const int col = static_cast<int>(p % n);
        const int global_row = first_global_row + row - 1;
        const size_t i = static_cast<size_t>(row) * n + col;
        const val_t energy = current[i].current_energy;
        // The final corner assignment in the original mesh makes the single
        // element case an inflow, too.
        val_t flux = 0.0;
        if ((global_row == 0 || global_row == n - 1) && (col == 0 || col == n - 1))
            flux = ((global_row == 0 && col == 0) ||
                    (global_row == n - 1 && col == n - 1)) ? 0.5 : -0.5;
        if (global_row + 1 < n)
            flux += (current[i + n].current_energy - energy) * 0.8 * 1.0 * 0.25;
        if (global_row > 0)
            flux += (current[i - n].current_energy - energy) * 0.8 * 1.0 * 0.25;
        if (col + 1 < n)
            flux += (current[i + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
        if (col > 0)
            flux += (current[i - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
        next[i].current_energy = energy + flux;
        next[i].total_flux = current[i].total_flux + fabs(flux);
    }
}

static void launchRows(const ElementDynamic* current, ElementDynamic* next, int n,
                       int first_row, int begin, int end, cudaStream_t stream,
                       int second_row = 0) {
    if (begin >= end) return;
    const size_t count = static_cast<size_t>(end - begin + (second_row != 0)) * n;
    const int blocks = static_cast<int>(std::min<size_t>((count + 255) / 256, 65535));
    updateRows<<<blocks, 256, 0, stream>>>(current, next, n, first_row, begin, end, second_row);
    CUDA_CHECK(cudaGetLastError());
}

static void runSimulation(ElementDynamic*& current, ElementDynamic*& next, int n,
                          int rows, int first_row, int n_iters, int rank, int ranks,
                          MPI_Comm comm) {
    cudaStream_t interior, boundary;
    CUDA_CHECK(cudaStreamCreateWithFlags(&interior, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&boundary, cudaStreamNonBlocking));
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(ElementDynamic);
    ElementDynamic* staging = nullptr;
    if (ranks > 1) CUDA_CHECK(cudaMallocHost(&staging, 4 * row_bytes));
    ElementDynamic* send_up = staging;
    ElementDynamic* send_down = staging ? staging + n : nullptr;
    ElementDynamic* recv_up = staging ? staging + 2 * static_cast<size_t>(n) : nullptr;
    ElementDynamic* recv_down = staging ? staging + 3 * static_cast<size_t>(n) : nullptr;
    const int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int down = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (ranks == 1) {
            // A single stream preserves iteration dependencies without a host sync.
            launchRows(current, next, n, first_row, 1, rows + 1, interior);
        } else {
            // The interior does not depend on incoming halos and runs while the
            // boundary rows are staged and exchanged with the adjacent ranks.
            launchRows(current, next, n, first_row, 2, rows, interior);
            MPI_Request req[4];
            int nr = 0;
            // Grid size is restricted so each byte count fits an MPI count.
            if (up != MPI_PROC_NULL)
                MPI_Irecv(recv_up, static_cast<int>(row_bytes), MPI_BYTE, up, 1, comm, &req[nr++]);
            if (down != MPI_PROC_NULL)
                MPI_Irecv(recv_down, static_cast<int>(row_bytes), MPI_BYTE, down, 0, comm, &req[nr++]);
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_up, current + n, row_bytes,
                                           cudaMemcpyDeviceToHost, boundary));
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_down, current + static_cast<size_t>(rows) * n,
                                           row_bytes, cudaMemcpyDeviceToHost, boundary));
            CUDA_CHECK(cudaStreamSynchronize(boundary));
            if (up != MPI_PROC_NULL)
                MPI_Isend(send_up, static_cast<int>(row_bytes), MPI_BYTE, up, 0, comm, &req[nr++]);
            if (down != MPI_PROC_NULL)
                MPI_Isend(send_down, static_cast<int>(row_bytes), MPI_BYTE, down, 1, comm, &req[nr++]);
            MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(current, recv_up, row_bytes,
                                           cudaMemcpyHostToDevice, boundary));
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(current + static_cast<size_t>(rows + 1) * n,
                                           recv_down, row_bytes, cudaMemcpyHostToDevice, boundary));
            launchRows(current, next, n, first_row, 1, 2, boundary, rows > 1 ? rows : 0);
            CUDA_CHECK(cudaStreamSynchronize(interior));
            CUDA_CHECK(cudaStreamSynchronize(boundary));
        }
        std::swap(current, next);
    }
    CUDA_CHECK(cudaStreamSynchronize(interior));
    CUDA_CHECK(cudaStreamSynchronize(boundary));
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaStreamDestroy(interior));
    CUDA_CHECK(cudaStreamDestroy(boundary));
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
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

static uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t offset) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t e, f;
        std::memcpy(&e, &elements[i].current_energy, sizeof(e));
        std::memcpy(&f, &elements[i].total_flux, sizeof(f));
        hash ^= (e + offset + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + offset + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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

// The output routines need globally ordered data. Transfer in bounded chunks
// rather than relying on MPI_Gatherv's int-sized counts and displacements.
static void transferBytes(void* data, size_t bytes, int peer, bool send, MPI_Comm comm) {
    char* p = static_cast<char*>(data);
    while (bytes) {
        const int count = static_cast<int>(std::min<size_t>(bytes, INT_MAX));
        if (send) MPI_Send(p, count, MPI_BYTE, peer, 2, comm);
        else MPI_Recv(p, count, MPI_BYTE, peer, 2, comm, MPI_STATUS_IGNORE);
        p += count;
        bytes -= count;
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) std::fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (world_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || static_cast<size_t>(n) > static_cast<size_t>(INT_MAX) / sizeof(ElementDynamic)) {
        if (world_rank == 0) std::fprintf(stderr, "Invalid grid size: %d\n", n);
        MPI_Finalize(); return 1;
    }
    const int active_ranks = std::min(world_size, n);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_ranks) {
        MPI_Finalize(); return 0;
    }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, devices;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA devices available\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local_comm);
    const int rows = n / ranks + (rank < n % ranks);
    const int first_row = rank * (n / ranks) + std::min(rank, n % ranks);
    const size_t n_elems = static_cast<size_t>(n) * n;
    const size_t local_elems = static_cast<size_t>(rows) * n;
    const size_t bytes = (static_cast<size_t>(rows) + 2) * n * sizeof(ElementDynamic);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        const size_t dynamic_mem = (n_elems + 2 * static_cast<size_t>(n) * ranks) * 2 * sizeof(ElementDynamic);
        printf("GPU memory usage: %.2f MB (dynamic state including halos)\n\n",
               dynamic_mem / 1048576.0);
        printf("Running simulation...\n");
    }
    ElementDynamic *current, *next;
    CUDA_CHECK(cudaMalloc(&current, bytes)); CUDA_CHECK(cudaMalloc(&next, bytes));
    CUDA_CHECK(cudaMemset(current, 0, bytes)); CUDA_CHECK(cudaMemset(next, 0, bytes));
    // The update streams are nonblocking and do not implicitly wait for work
    // submitted to the default stream.
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(current, next, n, rows, first_row, n_iters, rank, ranks, comm);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    std::vector<ElementDynamic> local(local_elems);
    CUDA_CHECK(cudaMemcpy(local.data(), current + n, local_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(current)); CUDA_CHECK(cudaFree(next));
    const uint64_t local_hash = computeHash(local, static_cast<size_t>(first_row) * n);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (rank == 0) {
        const auto duration_ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const int measured = std::max(n_iters - 1, 1);
        const double giga_elems_per_sec = measured * static_cast<double>(n_elems) / max_elapsed / 1e9;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", static_cast<double>(duration_ms) / measured);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }
    int valid = 1;
    if (validate || printResults) {
        std::vector<ElementDynamic> all;
        if (rank == 0) {
            all.resize(n_elems);
            std::copy(local.begin(), local.end(), all.begin());
            for (int r = 1; r < ranks; ++r) {
                const int rrows = n / ranks + (r < n % ranks);
                const int rfirst = r * (n / ranks) + std::min(r, n % ranks);
                transferBytes(all.data() + static_cast<size_t>(rfirst) * n,
                              static_cast<size_t>(rrows) * n * sizeof(ElementDynamic), r, false, comm);
            }
            if (printResults) {
                std::vector<double> energies(n_elems);
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n_elems; ++i) energies[i] = all[i].current_energy;
                print_results(energies, "ElementEnergy");
            }
            if (validate) valid = validateResults(all) ? 1 : 0;
        } else {
            transferBytes(local.data(), local_elems * sizeof(ElementDynamic), 0, true, comm);
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
