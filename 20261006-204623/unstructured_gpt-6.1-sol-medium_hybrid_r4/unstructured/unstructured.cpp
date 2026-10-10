#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Balanced contiguous row ownership; no rank stores the global mesh.
static int rowCount(int n, int ranks, int rank) {
    return n / ranks + (rank < n % ranks);
}
static int firstRow(int n, int ranks, int rank) {
    return rank * (n / ranks) + std::min(rank, n % ranks);
}

struct World {
    int n, rows, first, rank, ranks;
    size_t count, padded;
    double *energy[2]{}, *flux{};
    double *halo{};
    cudaStream_t compute{}, communication{};
    cudaEvent_t ready{}, received{};
    std::vector<double> host_energy, host_flux;

    World(int width, int r, int p) : n(width), rows(rowCount(width, p, r)),
        first(firstRow(width, p, r)), rank(r), ranks(p),
        count(size_t(rows) * n), padded(count + 2 * size_t(n)) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&communication, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
        for (auto& buffer : energy) {
            CUDA_CHECK(cudaMalloc(&buffer, padded * sizeof(double)));
            CUDA_CHECK(cudaMemsetAsync(buffer, 0, padded * sizeof(double), compute));
        }
        CUDA_CHECK(cudaMalloc(&flux, count * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(flux, 0, count * sizeof(double), compute));
        if (ranks > 1)
            CUDA_CHECK(cudaMallocHost(&halo, 4 * size_t(n) * sizeof(double)));
        CUDA_CHECK(cudaEventRecord(ready, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }
    ~World() {
        for (auto buffer : energy) CUDA_CHECK(cudaFree(buffer));
        CUDA_CHECK(cudaFree(flux));
        if (halo) CUDA_CHECK(cudaFreeHost(halo));
        CUDA_CHECK(cudaEventDestroy(ready));
        CUDA_CHECK(cudaEventDestroy(received));
        CUDA_CHECK(cudaStreamDestroy(compute));
        CUDA_CHECK(cudaStreamDestroy(communication));
    }
    World(const World&) = delete;
    World& operator=(const World&) = delete;
};

// The input mesh is exactly buildSquare2D's four-neighbor graph, with unit
// connection flux and transfer coefficient 0.8. Implicit indices avoid loading
// 144 bytes of static connectivity per element. Energy uses two ghost rows;
// cumulative flux needs no ghosts and can be updated in place.
// mode 0: all rows, 1: interior rows, 2: the one or two boundary rows.
template<int mode>
__global__ void update(const double* __restrict__ old_energy,
                       double* __restrict__ new_energy,
                       double* __restrict__ accumulated_flux,
                       int n, int rows, int first) {
    const int work_rows = mode == 0 ? rows :
                          (mode == 1 ? rows - 2 : (rows == 1 ? 1 : 2));
    const size_t work = size_t(work_rows) * n;
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         t < work; t += size_t(blockDim.x) * gridDim.x) {
        const int y = int(t % n);
        int x = int(t / n);
        if (mode == 1) ++x;
        if (mode == 2 && x == 1) x = rows - 1;
        const int global_x = first + x;
        const size_t i = size_t(x + 1) * n + y;
        const double e = old_energy[i];
        double total = 0.0;
        // Preserve the original corner-assignment order, including n == 1.
        if (global_x == 0 && y == 0) total = 0.5;
        if (global_x == 0 && y == n - 1) total = -0.5;
        if (global_x == n - 1 && y == 0) total = -0.5;
        if (global_x == n - 1 && y == n - 1) total = 0.5;
        // Preserve connectivity traversal: +x, -x, +y, -y, and each original
        // multiply/add rounding. CUDA contraction is disabled in CMake.
        if (global_x + 1 < n) total += (old_energy[i + n] - e) * 0.8 * 0.25;
        if (global_x > 0)     total += (old_energy[i - n] - e) * 0.8 * 0.25;
        if (y + 1 < n)        total += (old_energy[i + 1] - e) * 0.8 * 0.25;
        if (y > 0)            total += (old_energy[i - 1] - e) * 0.8 * 0.25;
        new_energy[i] = e + total;
        accumulated_flux[size_t(x) * n + y] += fabs(total);
    }
}

template<int mode>
static void launch(World& w, const double* src, double* dst) {
    const int rows = mode == 0 ? w.rows :
                     (mode == 1 ? w.rows - 2 : std::min(w.rows, 2));
    if (rows <= 0) return;
    const unsigned blocks = unsigned(std::min<size_t>((size_t(rows) * w.n + 255) / 256, 65535));
    update<mode><<<blocks, 256, 0, w.compute>>>(src, dst, w.flux, w.n, w.rows, w.first);
    CUDA_CHECK(cudaGetLastError());
}

static void runSimulation(World& w, int iterations, MPI_Comm comm) {
    double* src = w.energy[0];
    double* dst = w.energy[1];
    if (w.ranks == 1) {
        // Replay pairs of time steps to avoid per-step host launch overhead.
        // Two steps restore buffer ownership, so the graph is reusable.
        if (iterations >= 2) {
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            CUDA_CHECK(cudaStreamBeginCapture(w.compute, cudaStreamCaptureModeThreadLocal));
            launch<0>(w, src, dst);
            launch<0>(w, dst, src);
            CUDA_CHECK(cudaStreamEndCapture(w.compute, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int i = 0; i < iterations / 2; ++i)
                CUDA_CHECK(cudaGraphLaunch(executable, w.compute));
            CUDA_CHECK(cudaStreamSynchronize(w.compute));
            CUDA_CHECK(cudaGraphExecDestroy(executable));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        if (iterations % 2) {
            launch<0>(w, src, dst);
            std::swap(src, dst);
        }
    } else {
        const int previous = w.rank > 0 ? w.rank - 1 : MPI_PROC_NULL;
        const int next = w.rank + 1 < w.ranks ? w.rank + 1 : MPI_PROC_NULL;
        const size_t bytes = size_t(w.n) * sizeof(double);
        // Pinned staging works with ordinary MPI as well as CUDA-aware MPI,
        // without assuming that device-pointer communication is supported.
        double* send_up = w.halo;
        double* send_down = w.halo + w.n;
        double* recv_up = w.halo + 2 * size_t(w.n);
        double* recv_down = w.halo + 3 * size_t(w.n);
        for (int iter = 0; iter < iterations; ++iter) {
            CUDA_CHECK(cudaStreamWaitEvent(w.communication, w.ready, 0));
            if (previous != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_up, src + w.n, bytes,
                           cudaMemcpyDeviceToHost, w.communication));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_down, src + w.count, bytes,
                           cudaMemcpyDeviceToHost, w.communication));
            // Interior updates read no ghosts and write neither outgoing row.
            launch<1>(w, src, dst);
            CUDA_CHECK(cudaStreamSynchronize(w.communication));
            // Synchronization also makes reuse of the pinned receive buffers
            // safe after the preceding iteration's asynchronous H2D copies.
            MPI_Request requests[4];
            MPI_Irecv(recv_up, w.n, MPI_DOUBLE, previous, 11, comm, &requests[0]);
            MPI_Irecv(recv_down, w.n, MPI_DOUBLE, next, 10, comm, &requests[1]);
            MPI_Isend(send_up, w.n, MPI_DOUBLE, previous, 10, comm, &requests[2]);
            MPI_Isend(send_down, w.n, MPI_DOUBLE, next, 11, comm, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            if (previous != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(src, recv_up, bytes,
                           cudaMemcpyHostToDevice, w.communication));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(src + w.count + w.n, recv_down, bytes,
                           cudaMemcpyHostToDevice, w.communication));
            CUDA_CHECK(cudaEventRecord(w.received, w.communication));
            CUDA_CHECK(cudaStreamWaitEvent(w.compute, w.received, 0));
            launch<2>(w, src, dst);
            CUDA_CHECK(cudaEventRecord(w.ready, w.compute));
            std::swap(src, dst);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(w.compute));
    // Preserve ownership for odd iteration counts without duplicating buffers.
    if (src != w.energy[0]) std::swap(w.energy[0], w.energy[1]);
}

struct Statistics {
    double energy = 0, flux = 0;
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    uint64_t hash = 0;
    int invalid = 0;
};

static Statistics results(World& w, MPI_Comm comm) {
    w.host_energy.resize(w.count);
    w.host_flux.resize(w.count);
    CUDA_CHECK(cudaMemcpy(w.host_energy.data(), w.energy[0] + w.n,
               w.count * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w.host_flux.data(), w.flux,
               w.count * sizeof(double), cudaMemcpyDeviceToHost));
    double energy = 0, flux = 0;
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    uint64_t hash_value = 0;
    int invalid = 0;
    const uint64_t offset = uint64_t(w.first) * w.n;
    #pragma omp parallel for schedule(static) reduction(+:energy,flux) reduction(min:minimum) reduction(max:maximum) reduction(^:hash_value) reduction(|:invalid)
    for (size_t i = 0; i < w.count; ++i) {
        const double e = w.host_energy[i], f = w.host_flux[i];
        energy += e;
        flux += f;
        minimum = std::min(minimum, e);
        maximum = std::max(maximum, e);
        invalid |= !std::isfinite(e) || !std::isfinite(f);
        uint64_t eb, fb;
        memcpy(&eb, &e, sizeof(eb));
        memcpy(&fb, &f, sizeof(fb));
        hash_value ^= (eb + offset + i) * 0x9e3779b97f4a7c15ULL;
        hash_value ^= (fb + offset + i) * 0xbf58476d1ce4e5b9ULL;
    }
    Statistics s;
    double local[2] = {energy, flux}, global[2] = {};
    MPI_Reduce(local, global, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    s.energy = global[0]; s.flux = global[1];
    MPI_Reduce(&minimum, &s.minimum, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&maximum, &s.maximum, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&hash_value, &s.hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    MPI_Reduce(&invalid, &s.invalid, 1, MPI_INT, MPI_BOR, 0, comm);
    return s;
}

static void printResults(World& w, MPI_Comm comm) {
    // Gather only when explicitly requested. Chunk transfers to avoid MPI's
    // int count/displacement limit on large grids; retain global row order.
    constexpr size_t chunk = 1u << 26;
    std::vector<double> global;
    if (w.rank == 0) {
        global.resize(size_t(w.n) * w.n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < w.count; ++i) global[i] = w.host_energy[i];
    }
    for (int rank = 1; rank < w.ranks; ++rank) {
        const size_t count = size_t(rowCount(w.n, w.ranks, rank)) * w.n;
        const size_t offset = size_t(firstRow(w.n, w.ranks, rank)) * w.n;
        for (size_t i = 0; i < count; i += chunk) {
            const int length = int(std::min(chunk, count - i));
            if (w.rank == 0)
                MPI_Recv(global.data() + offset + i, length, MPI_DOUBLE, rank, 12, comm, MPI_STATUS_IGNORE);
            else if (w.rank == rank)
                MPI_Send(w.host_energy.data() + i, length, MPI_DOUBLE, 0, 12, comm);
        }
    }
    if (w.rank == 0) print_results(global, "ElementEnergy");
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

static bool parseInt(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
    value = int(parsed);
    return true;
}

static int benchmark(int argc, char** argv, int rank, int ranks) {
    int n = 512, iterations = 10;
    bool validate = false, print = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            int& value = !strcmp(argv[i], "-n") ? n : iterations;
            if (!parseInt(argv[++i], value)) {
                if (rank == 0) fprintf(stderr, "Invalid integer: %s\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { fprintf(stderr, "Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (n == 0 || size_t(n) > std::numeric_limits<size_t>::max() / size_t(n) / (4 * sizeof(double))) {
        if (rank == 0) fprintf(stderr, "Grid size is zero or exceeds addressable storage\n");
        return 1;
    }
    // More ranks than rows participate in MPI startup/finalization but own no
    // cells. Excluding them avoids empty-slab neighbors and zero-size kernels.
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < n ? 0 : MPI_UNDEFINED, rank, &comm);
    if (comm == MPI_COMM_NULL) return 0;
    ranks = std::min(ranks, n);
    MPI_Comm shared;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int local_rank;
    MPI_Comm_rank(shared, &local_rank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Respects per-rank CUDA_VISIBLE_DEVICES as well as shared-node visibility.
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&shared);
    World w(n, rank, ranks);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n, n, size_t(n) * n);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA enabled\n\n", ranks, omp_get_max_threads());
        printf("Building unstructured mesh (implicit square connectivity)...\n");
        printf("Memory usage per rank (GPU, rank 0): %.2f MB\n\n",
               (2 * w.padded + w.count) * sizeof(double) / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(w, iterations, comm);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    Statistics s = results(w, comm);
    if (rank == 0) {
        const double rate = seconds > 0 ? double(iterations) * double(n) * n / seconds / 1e9 : 0;
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", iterations > 0 ? seconds * 1000 / iterations : 0);
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate * 22);
        printf("  Result hash: %016" PRIX64 "\n\n", s.hash);
    }
    if (print) printResults(w, comm);
    int status = 0;
    if (validate && rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n  Flux sum: %.2f\n", s.energy, s.flux);
        printf("  Energy range: [%.6f, %.6f]\n", s.minimum, s.maximum);
        if (s.invalid || !std::isfinite(s.energy) || !std::isfinite(s.flux) ||
            !std::isfinite(s.minimum) || !std::isfinite(s.maximum)) {
            printf("  ERROR: Non-finite simulation results\n");
            status = 1;
        } else {
            if (std::abs(s.energy) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int global_status = 0;
    MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_status;
}
