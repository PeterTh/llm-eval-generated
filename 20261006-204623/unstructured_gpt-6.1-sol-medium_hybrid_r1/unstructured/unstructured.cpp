#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

static int world_rank = 0;

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", world_rank, operation,
                cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

static void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "Rank %d: %s: %.*s\n", world_rank, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define MPI_CHECK(operation) mpiCheck((operation), #operation)

// The input mesh is always the square grid built by the original buildSquare2D.
// Implicit connectivity avoids fetching eight 64-bit indices and coefficients
// for every cell. The order below is the original +x, -x, +y, -y order.
__global__ void updateEnergy(const val_t* __restrict__ energy,
                             val_t* __restrict__ next,
                             val_t* __restrict__ accumulated_flux,
                             int n, int first_global_row,
                             int first_local_row, int num_rows, int row_stride) {
    const size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= size_t(n)) return;
    // A two-dimensional launch avoids expensive per-element integer division.
    for (int r = int(blockIdx.y); r < num_rows; r += int(gridDim.y)) {
        const int local_row = first_local_row + r * row_stride;
        const int row = first_global_row + local_row;
        const size_t i = size_t(local_row) * n + col;
        const size_t p = i + n; // one ghost row on either side
        const val_t e = energy[p];
        val_t flux = 0.0;
        // Preserve corner assignment precedence, including the 1x1 grid.
        if (row == 0 && col == 0) flux = 0.5;
        if (row == 0 && col == size_t(n - 1)) flux = -0.5;
        if (row == n - 1 && col == 0) flux = -0.5;
        if (row == n - 1 && col == size_t(n - 1)) flux = 0.5;
        if (row + 1 < n) flux += (energy[p + n] - e) * 0.8 * 0.25;
        if (row > 0)     flux += (energy[p - n] - e) * 0.8 * 0.25;
        if (col + 1 < size_t(n)) flux += (energy[p + 1] - e) * 0.8 * 0.25;
        if (col > 0)     flux += (energy[p - 1] - e) * 0.8 * 0.25;
        next[p] = e + flux;
        accumulated_flux[i] += fabs(flux);
    }
}

struct World {
    int n, rows, first_row, rank, ranks;
    MPI_Comm comm;
    size_t count;
    val_t *energy = nullptr, *next = nullptr, *flux = nullptr;
    val_t *halo = nullptr; // pinned send-low, send-high, receive-low, receive-high
    cudaStream_t compute, exchange;
    cudaEvent_t interior_done;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int graph_iters = 0;
    std::vector<val_t> host_energy, host_flux;

    World(int grid_size, MPI_Comm active_comm) : n(grid_size), comm(active_comm) {
        MPI_CHECK(MPI_Comm_rank(comm, &rank));
        MPI_CHECK(MPI_Comm_size(comm, &ranks));
        rows = n / ranks + (rank < n % ranks);
        first_row = rank * (n / ranks) + std::min(rank, n % ranks);
        count = size_t(rows) * n;
        MPI_Comm local;
        MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_rank, devices;
        MPI_CHECK(MPI_Comm_rank(local, &local_rank));
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: CUDA GPU required\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % devices));
        MPI_CHECK(MPI_Comm_free(&local));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&interior_done, cudaEventDisableTiming));
        const size_t bytes = (count + size_t(2) * n) * sizeof(val_t);
        CUDA_CHECK(cudaMalloc(&energy, bytes));
        CUDA_CHECK(cudaMalloc(&next, bytes));
        CUDA_CHECK(cudaMalloc(&flux, count * sizeof(val_t)));
        CUDA_CHECK(cudaMemsetAsync(energy, 0, bytes, compute));
        CUDA_CHECK(cudaMemsetAsync(next, 0, bytes, compute));
        CUDA_CHECK(cudaMemsetAsync(flux, 0, count * sizeof(val_t), compute));
        if (ranks > 1) {
            CUDA_CHECK(cudaMallocHost(&halo, size_t(4) * n * sizeof(val_t)));
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < int64_t(4) * n; ++i) halo[i] = 0.0;
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    void launch(int first_local_row, int num_rows, cudaStream_t stream, int row_stride = 1) {
        if (num_rows <= 0) return;
        const dim3 blocks(unsigned((size_t(n) + 255) / 256), unsigned(std::min(num_rows, 65535)));
        updateEnergy<<<blocks, 256, 0, stream>>>(energy, next, flux, n, first_row,
                                               first_local_row, num_rows, row_stride);
        CUDA_CHECK(cudaGetLastError());
    }

    // Repeated single-rank timesteps are captured once to minimize host launch
    // overhead. An even batch returns to the same pair of ping-pong pointers.
    void prepareGraph(int iterations) {
        if (ranks != 1 || iterations < 2) return;
        graph_iters = std::min(iterations, 64) & ~1;
        CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < graph_iters; ++i) {
            launch(0, rows, compute);
            std::swap(energy, next);
        }
        CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(graph_exec, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    void simulate(int iterations) {
        if (ranks == 1) {
            int iter = 0;
            if (graph_iters) {
                for (; iter <= iterations - graph_iters; iter += graph_iters)
                    CUDA_CHECK(cudaGraphLaunch(graph_exec, compute));
            }
            for (; iter < iterations; ++iter) {
                launch(0, rows, compute);
                std::swap(energy, next);
            }
            CUDA_CHECK(cudaStreamSynchronize(compute));
            return;
        }
        const int low = rank ? rank - 1 : MPI_PROC_NULL;
        const int high = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        const size_t row_bytes = size_t(n) * sizeof(val_t);
        for (int iter = 0; iter < iterations; ++iter) {
            MPI_Request requests[4];
            MPI_CHECK(MPI_Irecv(halo + 2 * size_t(n), n, MPI_DOUBLE, low, 1, comm, &requests[0]));
            MPI_CHECK(MPI_Irecv(halo + 3 * size_t(n), n, MPI_DOUBLE, high, 0, comm, &requests[1]));
            if (low != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(halo, energy + n, row_bytes, cudaMemcpyDeviceToHost, exchange));
            if (high != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(halo + n, energy + count, row_bytes, cudaMemcpyDeviceToHost, exchange));
            // Interior work proceeds while MPI progresses the halo transfers.
            launch(1, rows - 2, compute);
            CUDA_CHECK(cudaEventRecord(interior_done, compute));
            CUDA_CHECK(cudaStreamSynchronize(exchange));
            MPI_CHECK(MPI_Isend(halo, n, MPI_DOUBLE, low, 0, comm, &requests[2]));
            MPI_CHECK(MPI_Isend(halo + n, n, MPI_DOUBLE, high, 1, comm, &requests[3]));
            MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
            if (low != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(energy, halo + 2 * size_t(n), row_bytes, cudaMemcpyHostToDevice, exchange));
            if (high != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(energy + count + n, halo + 3 * size_t(n), row_bytes, cudaMemcpyHostToDevice, exchange));
            launch(0, std::min(rows, 2), exchange, rows - 1);
            // Both disjoint writes must finish before next iteration's halo reads.
            CUDA_CHECK(cudaStreamWaitEvent(exchange, interior_done, 0));
            CUDA_CHECK(cudaStreamSynchronize(exchange));
            std::swap(energy, next);
        }
    }

    void download() {
        host_energy.resize(count);
        host_flux.resize(count);
        CUDA_CHECK(cudaMemcpyAsync(host_energy.data(), energy + n, count * sizeof(val_t), cudaMemcpyDeviceToHost, compute));
        CUDA_CHECK(cudaMemcpyAsync(host_flux.data(), flux, count * sizeof(val_t), cudaMemcpyDeviceToHost, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    ~World() {
        if (graph_exec) CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
        if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
        if (halo) CUDA_CHECK(cudaFreeHost(halo));
        CUDA_CHECK(cudaFree(energy));
        CUDA_CHECK(cudaFree(next));
        CUDA_CHECK(cudaFree(flux));
        CUDA_CHECK(cudaEventDestroy(interior_done));
        CUDA_CHECK(cudaStreamDestroy(compute));
        CUDA_CHECK(cudaStreamDestroy(exchange));
    }
};

static uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0, global_hash = 0;
    const uint64_t offset = size_t(world.first_row) * world.n;
    #pragma omp parallel for reduction(^:local_hash) schedule(static)
    for (size_t i = 0; i < world.count; ++i) {
        uint64_t e, f;
        // memcpy avoids the original hash's strict-aliasing violation.
        memcpy(&e, &world.host_energy[i], sizeof(e));
        memcpy(&f, &world.host_flux[i], sizeof(f));
        local_hash ^= (e + offset + i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f + offset + i) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_CHECK(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm));
    return global_hash;
}

static bool validateResults(const World& world) {
    val_t energy_sum = 0, flux_sum = 0;
    val_t energy_min = std::numeric_limits<val_t>::max();
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    int invalid = 0;
    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(min:energy_min) reduction(max:energy_max) reduction(|:invalid) schedule(static)
    for (size_t i = 0; i < world.count; ++i) {
        const val_t e = world.host_energy[i], f = world.host_flux[i];
        energy_sum += e;
        flux_sum += f;
        energy_min = std::min(energy_min, e);
        energy_max = std::max(energy_max, e);
        invalid |= !std::isfinite(e) || !std::isfinite(f);
    }
    val_t sums[2] = {energy_sum, flux_sum}, global_sums[2], global_min, global_max;
    int global_invalid;
    MPI_CHECK(MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, world.comm));
    MPI_CHECK(MPI_Allreduce(&energy_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, world.comm));
    MPI_CHECK(MPI_Allreduce(&energy_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, world.comm));
    MPI_CHECK(MPI_Allreduce(&invalid, &global_invalid, 1, MPI_INT, MPI_BOR, world.comm));
    const bool valid = !global_invalid && std::isfinite(global_sums[0]) && std::isfinite(global_sums[1]);
    if (world.rank == 0) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",
               global_sums[0], global_sums[1], global_min, global_max);
        if (std::abs(global_sums[0]) > 1e-8)
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

// Collect only for requested external validation. Chunking avoids MPI's int
// count limit; ranks and rows are received in original global element order.
static void printResults(const World& world) {
    std::vector<val_t> all;
    if (world.rank == 0) {
        all.resize(size_t(world.n) * world.n);
        std::copy(world.host_energy.begin(), world.host_energy.end(), all.begin());
    }
    constexpr size_t chunk = 1 << 26;
    for (int rank = 1; rank < world.ranks; ++rank) {
        const size_t rows = world.n / world.ranks + (rank < world.n % world.ranks);
        const size_t first = size_t(rank) * (world.n / world.ranks) + std::min(rank, world.n % world.ranks);
        const size_t count = rows * world.n;
        for (size_t i = 0; i < count; i += chunk) {
            const int length = int(std::min(chunk, count - i));
            if (world.rank == rank)
                MPI_CHECK(MPI_Send(world.host_energy.data() + i, length, MPI_DOUBLE, 0, 2, world.comm));
            if (world.rank == 0)
                MPI_CHECK(MPI_Recv(all.data() + first * world.n + i, length, MPI_DOUBLE, rank, 2, world.comm, MPI_STATUS_IGNORE));
        }
    }
    if (world.rank == 0) print_results(all, "ElementEnergy");
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseInteger(const char* text, int minimum, int& result) {
    char* end = nullptr;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < minimum || value > std::numeric_limits<int>::max()) return false;
    result = int(value);
    return true;
}

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512, iterations = 10;
    bool validate = false, results = false, help = false, arguments_valid = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            if (!parseInteger(argv[++i], 1, n)) arguments_valid = false;
        } else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            if (!parseInteger(argv[++i], 0, iterations)) arguments_valid = false;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else arguments_valid = false;
    }
    // Validate allocation arithmetic before constructing device/host arrays.
    const size_t max_values = std::numeric_limits<size_t>::max() / sizeof(val_t);
    if (size_t(n) > (max_values - size_t(2) * n) / size_t(n)) arguments_valid = false;
    if (!arguments_valid || help) {
        if (world_rank == 0) {
            if (!arguments_valid) fprintf(stderr, "Invalid argument: require positive grid size and nonnegative iteration count\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return arguments_valid ? 0 : 1;
    }
    int world_size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_Comm active;
    // More ranks than grid rows have no cells and need no GPU or halo buffers.
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < n ? 0 : MPI_UNDEFINED, world_rank, &active));
    int exit_code = 0;
    if (active != MPI_COMM_NULL) {
        try {
            World world(n, active);
            world.prepareGraph(iterations);
            if (world_rank == 0) {
                printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
                printf("Grid size: %d x %d = %zu elements\nIterations: %d\nValidation: %s\n\n",
                       n, n, size_t(n) * n, iterations, validate ? "enabled" : "disabled");
                printf("MPI ranks: %d (of %d), OpenMP threads/rank: %d; CUDA row-block decomposition\n",
                       world.ranks, world_size, omp_get_max_threads());
                printf("Building unstructured mesh (implicit square-grid connectivity)...\n");
                printf("Device state: %.2f MB per rank on rank 0\n\nRunning simulation...\n",
                       (3.0 * world.count + 4.0 * n) * sizeof(val_t) / (1024.0 * 1024.0));
            }
            MPI_CHECK(MPI_Barrier(active));
            const double start = MPI_Wtime();
            world.simulate(iterations);
            const double local_elapsed = MPI_Wtime() - start;
            double elapsed;
            MPI_CHECK(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active));
            world.download();
            const uint64_t digest = computeHash(world);
            if (world_rank == 0) {
                const double rate = elapsed > 0 ? double(iterations) * n * n / elapsed / 1e9 : 0;
                printf("Computation time: %.3f ms\nPerformance:\n", elapsed * 1000);
                printf("  Time per iteration: %.4f ms\n", iterations ? elapsed * 1000 / iterations : 0);
                printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
                printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(digest));
            }
            if (results) printResults(world);
            if (validate && !validateResults(world)) exit_code = 1;
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", world_rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_CHECK(MPI_Comm_free(&active));
    }
    int global_exit;
    MPI_CHECK(MPI_Allreduce(&exit_code, &global_exit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return global_exit;
}
