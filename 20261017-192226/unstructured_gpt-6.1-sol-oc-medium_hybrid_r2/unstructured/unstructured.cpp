#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using val_t = double;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

struct World {
    int n, rows, first_row, rank, ranks;
    MPI_Comm comm;
    std::vector<val_t> energy, flux;
};

// The only mesh constructed by the benchmark is a square grid. Its adjacency
// can be reconstructed exactly, avoiding eight indices and eight coefficients
// per cell. Energies are SoA, with one halo row at either end of each slab.
// Both interior and boundary kernels preserve the original down/up/right/left
// accumulation order; in particular, do not replace this with a Laplacian sum.
__global__ void update(const val_t* __restrict__ current,
                       val_t* __restrict__ next, val_t* __restrict__ flux,
                       int n, int rows, int first_row, int row_begin,
                       size_t count, bool boundary) {
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         t < count; t += size_t(blockDim.x) * gridDim.x) {
        const int row = boundary ? (t < size_t(n) ? 0 : rows - 1)
                                 : row_begin + int(t / n);
        const int col = int(t % n);
        const int x = first_row + row;
        const size_t i = size_t(row + 1) * n + col;
        const val_t e = current[i];
        val_t total = 0.0;
        if ((x == 0 || x == n - 1) && (col == 0 || col == n - 1))
            total = ((x == 0 && col == 0) || (x == n - 1 && col == n - 1)) ? 0.5 : -0.5;
        if (x + 1 < n) total += (current[i + n] - e) * 0.8 * 1.0 * 0.25;
        if (x > 0)     total += (current[i - n] - e) * 0.8 * 1.0 * 0.25;
        if (col + 1 < n) total += (current[i + 1] - e) * 0.8 * 1.0 * 0.25;
        if (col > 0)     total += (current[i - 1] - e) * 0.8 * 1.0 * 0.25;
        next[i] = e + total;
        // No neighbor reads cumulative flux, so it is safe to update in place.
        flux[size_t(row) * n + col] += fabs(total);
    }
}

static void launchUpdate(const World& w, const val_t* current, val_t* next,
                         val_t* flux, int row_begin, size_t count,
                         bool boundary, cudaStream_t stream) {
    if (!count) return;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    update<<<blocks, 256, 0, stream>>>(current, next, flux, w.n, w.rows,
                                      w.first_row, row_begin, count, boundary);
    CUDA_CHECK(cudaGetLastError());
}

// Only energies cross rank boundaries. Pinned staging works with ordinary MPI;
// no CUDA-aware MPI extension or host copy of the mesh is required. Communication
// and interior computation are overlapped on separate CUDA streams.
static double runSimulation(World& w, int iterations) {
    const size_t cells = size_t(w.rows) * w.n;
    const size_t row_bytes = size_t(w.n) * sizeof(val_t);
    const size_t energy_bytes = (cells + 2 * size_t(w.n)) * sizeof(val_t);
    val_t *current = nullptr, *next = nullptr, *flux = nullptr, *halo = nullptr;
    CUDA_CHECK(cudaMalloc(&current, energy_bytes));
    CUDA_CHECK(cudaMalloc(&next, energy_bytes));
    CUDA_CHECK(cudaMalloc(&flux, cells * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(current, 0, energy_bytes));
    CUDA_CHECK(cudaMemset(next, 0, energy_bytes));
    CUDA_CHECK(cudaMemset(flux, 0, cells * sizeof(val_t)));
    cudaStream_t compute, exchange;
    cudaEvent_t updated, received;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&updated, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
    if (w.ranks > 1) CUDA_CHECK(cudaMallocHost(&halo, 4 * row_bytes));
    // Resolve lazy module loading/PTX JIT before timing, without changing state.
    update<<<1, 256, 0, compute>>>(current, next, flux, w.n, w.rows,
                                  w.first_row, 0, 0, false);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(w.comm);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (w.ranks == 1) {
            launchUpdate(w, current, next, flux, 0, cells, false, compute);
        } else {
            MPI_Request requests[4];
            int nr = 0;
            const bool up = w.rank > 0, down = w.rank + 1 < w.ranks;
            if (up) MPI_Irecv(halo + 2 * size_t(w.n), w.n, MPI_DOUBLE,
                              w.rank - 1, 1, w.comm, &requests[nr++]);
            if (down) MPI_Irecv(halo + 3 * size_t(w.n), w.n, MPI_DOUBLE,
                                w.rank + 1, 0, w.comm, &requests[nr++]);
            if (iter) CUDA_CHECK(cudaStreamWaitEvent(exchange, updated, 0));
            if (up) CUDA_CHECK(cudaMemcpyAsync(halo, current + w.n, row_bytes,
                                               cudaMemcpyDeviceToHost, exchange));
            if (down) CUDA_CHECK(cudaMemcpyAsync(halo + w.n, current + cells,
                                                 row_bytes, cudaMemcpyDeviceToHost, exchange));
            if (w.rows > 2)
                launchUpdate(w, current, next, flux, 1, size_t(w.rows - 2) * w.n, false, compute);
            CUDA_CHECK(cudaStreamSynchronize(exchange));
            if (up) MPI_Isend(halo, w.n, MPI_DOUBLE, w.rank - 1, 0, w.comm, &requests[nr++]);
            if (down) MPI_Isend(halo + w.n, w.n, MPI_DOUBLE, w.rank + 1, 1, w.comm, &requests[nr++]);
            MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
            if (up) CUDA_CHECK(cudaMemcpyAsync(current, halo + 2 * size_t(w.n), row_bytes,
                                               cudaMemcpyHostToDevice, exchange));
            if (down) CUDA_CHECK(cudaMemcpyAsync(current + cells + w.n,
                                                 halo + 3 * size_t(w.n), row_bytes,
                                                 cudaMemcpyHostToDevice, exchange));
            CUDA_CHECK(cudaEventRecord(received, exchange));
            CUDA_CHECK(cudaStreamWaitEvent(compute, received, 0));
            launchUpdate(w, current, next, flux, 0, size_t(w.rows == 1 ? 1 : 2) * w.n, true, compute);
            CUDA_CHECK(cudaEventRecord(updated, compute));
        }
        std::swap(current, next);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double maximum = 0;
    MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, w.comm);

    w.energy.resize(cells);
    w.flux.resize(cells);
    CUDA_CHECK(cudaMemcpy(w.energy.data(), current + w.n, cells * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w.flux.data(), flux, cells * sizeof(val_t), cudaMemcpyDeviceToHost));
    if (halo) CUDA_CHECK(cudaFreeHost(halo));
    CUDA_CHECK(cudaFree(current));
    CUDA_CHECK(cudaFree(next));
    CUDA_CHECK(cudaFree(flux));
    CUDA_CHECK(cudaEventDestroy(updated));
    CUDA_CHECK(cudaEventDestroy(received));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(exchange));
    return maximum;
}

static uint64_t computeHash(const World& w) {
    uint64_t local = 0, global = 0;
    const uint64_t offset = uint64_t(w.first_row) * w.n;
    #pragma omp parallel for reduction(^:local) schedule(static)
    for (size_t i = 0; i < w.energy.size(); ++i) {
        uint64_t e, f;
        // memcpy preserves the exact bits without violating strict aliasing.
        memcpy(&e, &w.energy[i], sizeof(e));
        memcpy(&f, &w.flux[i], sizeof(f));
        local ^= (e + offset + i) * 0x9e3779b97f4a7c15ULL;
        local ^= (f + offset + i) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_Reduce(&local, &global, 1, MPI_UINT64_T, MPI_BXOR, 0, w.comm);
    return global;
}

static bool validateResults(const World& w) {
    val_t energy_sum = 0, flux_sum = 0;
    val_t energy_min = std::numeric_limits<val_t>::max();
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    int nonfinite = 0;
    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(min:energy_min) reduction(max:energy_max) reduction(|:nonfinite) schedule(static)
    for (size_t i = 0; i < w.energy.size(); ++i) {
        energy_sum += w.energy[i];
        flux_sum += w.flux[i];
        energy_min = std::min(energy_min, w.energy[i]);
        energy_max = std::max(energy_max, w.energy[i]);
        nonfinite |= !std::isfinite(w.energy[i]) || !std::isfinite(w.flux[i]);
    }
    val_t sums[2] = {energy_sum, flux_sum}, totals[2], minimum, maximum;
    int invalid;
    MPI_Allreduce(sums, totals, 2, MPI_DOUBLE, MPI_SUM, w.comm);
    MPI_Allreduce(&energy_min, &minimum, 1, MPI_DOUBLE, MPI_MIN, w.comm);
    MPI_Allreduce(&energy_max, &maximum, 1, MPI_DOUBLE, MPI_MAX, w.comm);
    MPI_Allreduce(&nonfinite, &invalid, 1, MPI_INT, MPI_MAX, w.comm);
    const bool valid = !invalid && std::isfinite(totals[0]) && std::isfinite(totals[1]);
    if (w.rank == 0) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
               "  Energy range: [%.6f, %.6f]\n", totals[0], totals[1], minimum, maximum);
        if (std::abs(totals[0]) > 1e-8)
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

// Gather only when the user requests the original order-sensitive output hash.
// Chunked messages avoid MPI's int-count limit for very large global meshes.
static void printResults(const World& w) {
    constexpr size_t chunk = 1u << 26;
    if (w.rank == 0) {
        std::vector<val_t> all(size_t(w.n) * w.n);
        std::copy(w.energy.begin(), w.energy.end(), all.begin());
        for (int rank = 1; rank < w.ranks; ++rank) {
            const int rows = w.n / w.ranks + (rank < w.n % w.ranks);
            const int first = rank * (w.n / w.ranks) + std::min(rank, w.n % w.ranks);
            const size_t count = size_t(rows) * w.n, offset = size_t(first) * w.n;
            for (size_t i = 0; i < count; i += chunk)
                MPI_Recv(all.data() + offset + i, int(std::min(chunk, count - i)),
                         MPI_DOUBLE, rank, 2, w.comm, MPI_STATUS_IGNORE);
        }
        print_results(all, "ElementEnergy");
    } else {
        for (size_t i = 0; i < w.energy.size(); i += chunk)
            MPI_Send(w.energy.data() + i, int(std::min(chunk, w.energy.size() - i)),
                     MPI_DOUBLE, 0, 2, w.comm);
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseInt(const char* text, int& value) {
    char* end = nullptr;
    const long long parsed = strtoll(text, &end, 10);
    if (!*text || *end || parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
    value = int(parsed);
    return true;
}

static int benchmark(int argc, char** argv, int world_rank, int world_size) {
    int n = 512, iterations = 10;
    bool validate = false, output = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const bool grid = !strcmp(argv[i], "-n");
            if (!parseInt(argv[++i], grid ? n : iterations)) {
                if (!world_rank) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) output = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!world_rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (n < 1 || size_t(n) > std::numeric_limits<size_t>::max() / size_t(n) / (3 * sizeof(val_t))) {
        if (!world_rank) fprintf(stderr, "Grid size must be positive and fit in addressable memory.\n");
        return 1;
    }

    // Assign devices by node-local rank, respecting CUDA_VISIBLE_DEVICES. Empty
    // slabs (when ranks > rows) do not allocate devices or enter halo exchanges.
    MPI_Comm node, comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node);
    int local_rank;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
    const int ranks = std::min(n, world_size);
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < ranks ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (comm == MPI_COMM_NULL) return 0;
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    World w{n, n / ranks + (world_rank < n % ranks),
            world_rank * (n / ranks) + std::min(world_rank, n % ranks),
            world_rank, ranks, comm, {}, {}};
    if (!world_rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n, n, size_t(n) * n);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n\n",
               ranks, omp_get_max_threads(), devices);
        printf("Building unstructured mesh (implicit square connectivity)...\n");
        printf("Running simulation...\n");
    }
    const double seconds = runSimulation(w, iterations);
    const uint64_t result_hash = computeHash(w);
    if (!world_rank) {
        const double ms = seconds * 1000;
        const double rate = seconds > 0 ? double(iterations) * n * n / seconds / 1e9 : 0;
        printf("Computation time: %.3f ms\nPerformance:\n", ms);
        printf("  Time per iteration: %.4f ms\n", iterations ? ms / iterations : 0);
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
    }
    if (output) printResults(w);
    const bool valid = !validate || validateResults(w);
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 1;
    try {
        status = benchmark(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
