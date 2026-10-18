#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// The builder always creates a square, unit-flux, four-neighbor mesh.
// Implicit connectivity eliminates the original per-cell index/flux lists.
// Each rank owns a row slab with two ghost rows. Only energy needs halos;
// accumulated absolute flux can be updated in place.
struct World {
    int n, rows, first, rank, ranks;
    MPI_Comm comm;
    double *energy[2]{}, *flux{}, *halos{};
    cudaStream_t compute{}, exchange{};
    cudaEvent_t ready{};
    int current = 0;

    World(int side, int localRows, int firstRow, int id, int count, MPI_Comm c)
        : n(side), rows(localRows), first(firstRow), rank(id), ranks(count), comm(c) {
        const size_t cells = size_t(rows) * n;
        const size_t bytes = (cells + size_t(2) * n) * sizeof(double);
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        for (auto& e : energy) {
            CUDA(cudaMalloc(&e, bytes));
            CUDA(cudaMemsetAsync(e, 0, bytes, compute));
        }
        CUDA(cudaMalloc(&flux, cells * sizeof(double)));
        CUDA(cudaMemsetAsync(flux, 0, cells * sizeof(double), compute));
        if (ranks > 1) CUDA(cudaMallocHost(&halos, size_t(4) * n * sizeof(double)));
        CUDA(cudaEventRecord(ready, compute));
        CUDA(cudaStreamSynchronize(compute));
    }
    ~World() {
        for (auto e : energy) CUDA(cudaFree(e));
        CUDA(cudaFree(flux));
        if (halos) CUDA(cudaFreeHost(halos));
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaStreamDestroy(compute));
        CUDA(cudaStreamDestroy(exchange));
    }
};

__device__ inline double computeFlux(double other, double self) {
    return (other - self) * 0.8 * 1.0 * 0.25;
}

__global__ void update(const double* __restrict__ input,
                       double* __restrict__ output, double* __restrict__ flux,
                       int n, int first, int beginRow, int endRow) {
    const size_t count = size_t(endRow - beginRow) * n;
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < count; k += size_t(blockDim.x) * gridDim.x) {
        const int row = beginRow + k / n;
        const int y = k % n;
        const int x = first + row;
        const size_t i = size_t(row + 1) * n + y;
        const double e = input[i];
        double total = 0.0;
        // Preserve material assignment order even when all corners coincide.
        if (x == 0 && y == 0) total = 0.5;
        if (x == 0 && y == n - 1) total = -0.5;
        if (x == n - 1 && y == 0) total = -0.5;
        if (x == n - 1 && y == n - 1) total = 0.5;
        // Retain the original +x, -x, +y, -y accumulation order.
        if (x + 1 < n) total += computeFlux(input[i + n], e);
        if (x > 0) total += computeFlux(input[i - n], e);
        if (y + 1 < n) total += computeFlux(input[i + 1], e);
        if (y > 0) total += computeFlux(input[i - 1], e);
        output[i] = e + total;
        flux[size_t(row) * n + y] += fabs(total);
    }
}

static void launch(World& w, int begin, int end) {
    if (end <= begin) return;
    const size_t count = size_t(end - begin) * w.n;
    const int blocks = int(std::min<size_t>((count + 255) / 256, 65535));
    update<<<blocks, 256, 0, w.compute>>>(w.energy[w.current],
        w.energy[1 - w.current], w.flux, w.n, w.first, begin, end);
    CUDA(cudaGetLastError());
}

static void runSimulation(World& w, int iterations) {
    const int up = w.rank ? w.rank - 1 : MPI_PROC_NULL;
    const int down = w.rank + 1 < w.ranks ? w.rank + 1 : MPI_PROC_NULL;
    const size_t rowBytes = size_t(w.n) * sizeof(double);
    for (int iter = 0; iter < iterations; ++iter) {
        if (w.ranks == 1) {
            launch(w, 0, w.rows);
        } else {
            double* e = w.energy[w.current];
            MPI_Request requests[4];
            MPI_Irecv(w.halos + 2 * size_t(w.n), w.n, MPI_DOUBLE, up, 1, w.comm, &requests[0]);
            MPI_Irecv(w.halos + 3 * size_t(w.n), w.n, MPI_DOUBLE, down, 0, w.comm, &requests[1]);
            CUDA(cudaStreamWaitEvent(w.exchange, w.ready, 0));
            if (up != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(w.halos, e + w.n, rowBytes, cudaMemcpyDeviceToHost, w.exchange));
            if (down != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(w.halos + w.n, e + size_t(w.rows) * w.n,
                                     rowBytes, cudaMemcpyDeviceToHost, w.exchange));
            // Interior work overlaps pinned-memory transfers and MPI progress.
            launch(w, 1, w.rows - 1);
            CUDA(cudaStreamSynchronize(w.exchange));
            MPI_Isend(w.halos, w.n, MPI_DOUBLE, up, 0, w.comm, &requests[2]);
            MPI_Isend(w.halos + w.n, w.n, MPI_DOUBLE, down, 1, w.comm, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            if (up != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(e, w.halos + 2 * size_t(w.n), rowBytes, cudaMemcpyHostToDevice, w.exchange));
            if (down != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(e + size_t(w.rows + 1) * w.n,
                                     w.halos + 3 * size_t(w.n), rowBytes,
                                     cudaMemcpyHostToDevice, w.exchange));
            CUDA(cudaStreamSynchronize(w.exchange));
            launch(w, 0, 1);
            if (w.rows > 1) launch(w, w.rows - 1, w.rows);
        }
        w.current = 1 - w.current;
        CUDA(cudaEventRecord(w.ready, w.compute));
    }
    CUDA(cudaStreamSynchronize(w.compute));
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseInteger(const char* text, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < 0 || parsed > INT_MAX) return false;
    value = int(parsed);
    return true;
}

static int benchmark(int argc, char** argv, int worldRank, int worldSize) {
    int n = 512, iterations = 10;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            int& value = !strcmp(argv[i], "-n") ? n : iterations;
            if (!parseInteger(argv[++i], value)) {
                if (!worldRank) fprintf(stderr, "Invalid integer: %s\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) printUsage(argv[0]);
            return 1;
        }
    }
    if (n < 1 || size_t(n) > (SIZE_MAX / sizeof(double) - 2 * size_t(n)) / size_t(n)) {
        if (!worldRank) fprintf(stderr, "Grid size is invalid or too large\n");
        return 1;
    }
    // Respect scheduler device visibility and distribute ranks over local GPUs.
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &shared);
    int localRank;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_free(&shared);
    const int ranks = std::min(worldSize, n);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) return 0;
    int devices = 0;
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "A CUDA device is required on every participating rank\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    const int rows = n / ranks + (worldRank < n % ranks);
    const int first = worldRank * (n / ranks) + std::min(worldRank, n % ranks);
    const size_t cells = size_t(rows) * n;
    const size_t globalCells = size_t(n) * n;
    World w(n, rows, first, worldRank, ranks, comm);
    if (!worldRank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n, n, globalCells);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Parallelism: %d MPI ranks, CUDA, OpenMP (%d host threads/rank)\n", ranks, omp_get_max_threads());
        printf("Building unstructured mesh (implicit square connectivity)...\n");
        printf("Device state per rank (rank 0): %.2f MB\n", (3 * cells + 4 * size_t(n)) * sizeof(double) / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(w, iterations);
    const double localSeconds = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    std::vector<double> energy(cells), flux(cells);
    CUDA(cudaMemcpy(energy.data(), w.energy[w.current] + n, cells * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(flux.data(), w.flux, cells * sizeof(double), cudaMemcpyDeviceToHost));
    uint64_t localHash = 0;
    double energySum = 0, fluxSum = 0;
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    int bad = 0;
    #pragma omp parallel for schedule(static) reduction(^:localHash) reduction(+:energySum,fluxSum) reduction(min:minimum) reduction(max:maximum) reduction(|:bad)
    for (size_t i = 0; i < cells; ++i) {
        uint64_t e, f;
        memcpy(&e, &energy[i], sizeof(e));
        memcpy(&f, &flux[i], sizeof(f));
        const uint64_t global = size_t(first) * n + i;
        localHash ^= (e + global) * 0x9e3779b97f4a7c15ULL;
        localHash ^= (f + global) * 0xbf58476d1ce4e5b9ULL;
        if (validate) {
            energySum += energy[i];
            fluxSum += flux[i];
            minimum = std::min(minimum, energy[i]);
            maximum = std::max(maximum, energy[i]);
            bad |= !std::isfinite(energy[i]) || !std::isfinite(flux[i]);
        }
    }
    uint64_t hash;
    MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (!worldRank) {
        const double perIter = iterations ? seconds * 1000.0 / iterations : 0.0;
        const double rate = seconds > 0 ? double(iterations) * globalCells / seconds / 1e9 : 0.0;
        printf("Computation time: %.3f ms\nPerformance:\n", seconds * 1000.0);
        printf("  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n", perIter, rate);
        printf("  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", rate * 22.0, (unsigned long long)hash);
    }
    if (results) {
        // Gather in global row order only on request, using int-sized chunks
        // for portability to MPI implementations without large-count calls.
        std::vector<double> all;
        if (!worldRank) {
            all.resize(globalCells);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cells; ++i) all[i] = energy[i];
        }
        for (int r = 1; r < ranks; ++r) {
            const size_t offset = size_t(r * (n / ranks) + std::min(r, n % ranks)) * n;
            const size_t count = size_t(n / ranks + (r < n % ranks)) * n;
            for (size_t pos = 0; pos < count;) {
                const int chunk = int(std::min<size_t>(count - pos, INT_MAX));
                if (!worldRank) MPI_Recv(all.data() + offset + pos, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                else if (worldRank == r) MPI_Send(energy.data() + pos, chunk, MPI_DOUBLE, 0, 2, comm);
                pos += chunk;
            }
        }
        if (!worldRank) print_results(all, "ElementEnergy");
    }
    int valid = 1;
    if (validate) {
        double sums[2] = {energySum, fluxSum}, globalSums[2];
        double lo, hi;
        int globalBad;
        MPI_Reduce(sums, globalSums, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&minimum, &lo, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&maximum, &hi, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&bad, &globalBad, 1, MPI_INT, MPI_BOR, 0, comm);
        if (!worldRank) {
            valid = !globalBad && std::isfinite(globalSums[0]) && std::isfinite(globalSums[1]);
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", globalSums[0], globalSums[1], lo, hi);
            if (std::abs(globalSums[0]) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
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
    int globalStatus;
    MPI_Allreduce(&status, &globalStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return globalStatus;
}
