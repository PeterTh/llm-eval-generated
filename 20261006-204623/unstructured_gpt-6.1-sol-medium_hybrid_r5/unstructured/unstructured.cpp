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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// The generated mesh has unit-weight edges in the order +x, -x, +y, -y.
// Deriving those edges saves the original 144-byte adjacency record per cell.
__global__ void update(const double* __restrict__ energy,
                       double* __restrict__ next, double* __restrict__ accumulated,
                       int n, int rows, int firstRow, size_t begin, size_t count,
                       bool boundaryOnly) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (; t < count; t += stride) {
        size_t local = begin + t;
        if (boundaryOnly) {
            local = t < size_t(n) ? t : size_t(rows - 1) * n + t - n;
        }
        const int row = int(local / n);
        const int col = int(local % n);
        const int globalRow = firstRow + row;
        const size_t p = local + n; // Leading and trailing energy halo rows.
        const double current = energy[p];
        double flux = 0.0;
        // Assignment order in buildSquare2D leaves the one-cell grid as inflow.
        if ((globalRow == 0 || globalRow == n - 1) && (col == 0 || col == n - 1)) {
            flux = ((globalRow == 0 && col == 0) ||
                    (globalRow == n - 1 && col == n - 1)) ? 0.5 : -0.5;
        }
        if (globalRow + 1 < n) flux += (energy[p + n] - current) * 0.8 * 1.0 * 0.25;
        if (globalRow > 0) flux += (energy[p - n] - current) * 0.8 * 1.0 * 0.25;
        if (col + 1 < n) flux += (energy[p + 1] - current) * 0.8 * 1.0 * 0.25;
        if (col > 0) flux += (energy[p - 1] - current) * 0.8 * 1.0 * 0.25;
        next[p] = current + flux;
        accumulated[local] += fabs(flux);
    }
}

static void launch(const double* energy, double* next, double* flux,
                   int n, int rows, int first, size_t begin, size_t count,
                   bool boundary, cudaStream_t stream) {
    if (!count) return;
    const unsigned blocks = unsigned(std::min<size_t>((count + 255) / 256, 65535));
    update<<<blocks, 256, 0, stream>>>(energy, next, flux, n, rows, first, begin, count, boundary);
    CUDA(cudaGetLastError());
}

struct DeviceState {
    double *energy = nullptr, *next = nullptr, *flux = nullptr, *halo = nullptr;
    cudaStream_t compute, exchange;
    cudaEvent_t ready, boundaries;
    int n, rows, first, rank, ranks;
    MPI_Comm comm;

    DeviceState(int width, int height, int start, MPI_Comm communicator)
        : n(width), rows(height), first(start), comm(communicator) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        const size_t cells = size_t(n) * rows;
        CUDA(cudaMalloc(&energy, (cells + 2 * size_t(n)) * sizeof(double)));
        CUDA(cudaMalloc(&next, (cells + 2 * size_t(n)) * sizeof(double)));
        CUDA(cudaMalloc(&flux, cells * sizeof(double)));
        CUDA(cudaMemset(energy, 0, (cells + 2 * size_t(n)) * sizeof(double)));
        CUDA(cudaMemset(next, 0, (cells + 2 * size_t(n)) * sizeof(double)));
        CUDA(cudaMemset(flux, 0, cells * sizeof(double)));
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&boundaries, cudaEventDisableTiming));
        if (ranks > 1) CUDA(cudaMallocHost(&halo, 4 * size_t(n) * sizeof(double)));
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaEventRecord(ready, compute));
    }

    void run(int iterations) {
        const size_t cells = size_t(n) * rows;
        const size_t bytes = size_t(n) * sizeof(double);
        const int previous = rank ? rank - 1 : MPI_PROC_NULL;
        const int following = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        for (int iter = 0; iter < iterations; ++iter) {
            if (ranks == 1) {
                launch(energy, next, flux, n, rows, first, 0, cells, false, compute);
            } else {
                MPI_Request requests[4];
                MPI_Irecv(halo + 2 * size_t(n), n, MPI_DOUBLE, previous, 1, comm, &requests[0]);
                MPI_Irecv(halo + 3 * size_t(n), n, MPI_DOUBLE, following, 0, comm, &requests[1]);
                // Copy just the outgoing energy rows, overlapping interior GPU work.
                // Pinned staging works with MPI implementations without CUDA support.
                CUDA(cudaStreamWaitEvent(exchange, ready, 0));
                if (previous != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(halo, energy + n, bytes, cudaMemcpyDeviceToHost, exchange));
                if (following != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(halo + n, energy + cells, bytes, cudaMemcpyDeviceToHost, exchange));
                if (rows > 2)
                    launch(energy, next, flux, n, rows, first, n, size_t(rows - 2) * n, false, compute);
                CUDA(cudaStreamSynchronize(exchange));
                MPI_Isend(halo, n, MPI_DOUBLE, previous, 0, comm, &requests[2]);
                MPI_Isend(halo + n, n, MPI_DOUBLE, following, 1, comm, &requests[3]);
                MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
                if (previous != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(energy, halo + 2 * size_t(n), bytes, cudaMemcpyHostToDevice, exchange));
                if (following != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(energy + cells + n, halo + 3 * size_t(n), bytes,
                                         cudaMemcpyHostToDevice, exchange));
                launch(energy, next, flux, n, rows, first, 0,
                       size_t(n) * (rows == 1 ? 1 : 2), true, exchange);
                CUDA(cudaEventRecord(boundaries, exchange));
                CUDA(cudaStreamWaitEvent(compute, boundaries, 0));
                CUDA(cudaEventRecord(ready, compute));
            }
            std::swap(energy, next);
        }
        CUDA(cudaStreamSynchronize(compute));
    }

    ~DeviceState() {
        CUDA(cudaFree(energy)); CUDA(cudaFree(next)); CUDA(cudaFree(flux));
        if (halo) CUDA(cudaFreeHost(halo));
        CUDA(cudaEventDestroy(ready)); CUDA(cudaEventDestroy(boundaries));
        CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(exchange));
    }
};

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iterations = 10;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || iterations < 0 || size_t(n) > std::numeric_limits<size_t>::max() / size_t(n) / (4 * sizeof(double))) {
        if (!rank) fprintf(stderr, "Grid size must be positive and iterations nonnegative; grid must fit in address space.\n");
        MPI_Finalize(); return 1;
    }
    // Ranks beyond the number of rows stay in world collectives, but own no cells.
    const int activeRanks = std::min(n, ranks);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
    const int rows = rank < activeRanks ? n / activeRanks + (rank < n % activeRanks) : 0;
    const int first = rank < activeRanks ? rank * (n / activeRanks) + std::min(rank, n % activeRanks) : n;
    const size_t cells = size_t(n) * rows, globalCells = size_t(n) * n;
    std::vector<double> energy(cells), flux(cells);
    // Parallel first-touch of the host output buffers.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells; ++i) { energy[i] = 0.0; flux[i] = 0.0; }

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    DeviceState* device = nullptr;
    if (rows) {
        int devices;
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "A CUDA GPU is required on every active rank.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA(cudaSetDevice(localRank % devices));
        device = new DeviceState(n, rows, first, active);
    }
    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %zu elements\nIterations: %d\nValidation: %s\n\n",
               n, n, globalCells, iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d active), OpenMP threads/rank: %d\n", ranks, activeRanks, omp_get_max_threads());
        printf("Building unstructured mesh...\nConnectivity: implicit unit-weight square mesh\n");
        printf("Memory usage (global GPU state): %.2f MB\n\nRunning simulation...\n",
               (3.0 * globalCells + 4.0 * n * activeRanks) * sizeof(double) / (1024 * 1024));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (device) device->run(iterations);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (device) {
        CUDA(cudaMemcpy(energy.data(), device->energy + n, cells * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(flux.data(), device->flux, cells * sizeof(double), cudaMemcpyDeviceToHost));
        delete device;
        MPI_Comm_free(&active);
    }

    uint64_t hashValue = 0;
    double energySum = 0, fluxSum = 0;
    double minimum = std::numeric_limits<double>::max(), maximum = std::numeric_limits<double>::lowest();
    int invalid = 0;
    #pragma omp parallel for schedule(static) reduction(^:hashValue) reduction(+:energySum,fluxSum) reduction(min:minimum) reduction(max:maximum) reduction(|:invalid)
    for (size_t i = 0; i < cells; ++i) {
        uint64_t e, f;
        memcpy(&e, &energy[i], sizeof(e)); memcpy(&f, &flux[i], sizeof(f));
        const uint64_t globalIndex = size_t(first) * n + i;
        hashValue ^= (e + globalIndex) * 0x9e3779b97f4a7c15ULL;
        hashValue ^= (f + globalIndex) * 0xbf58476d1ce4e5b9ULL;
        energySum += energy[i]; fluxSum += flux[i];
        minimum = std::min(minimum, energy[i]); maximum = std::max(maximum, energy[i]);
        invalid |= !std::isfinite(energy[i]) || !std::isfinite(flux[i]);
    }
    uint64_t globalHash;
    MPI_Reduce(&hashValue, &globalHash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    double sums[2] = {energySum, fluxSum}, globalSums[2], globalMin, globalMax;
    MPI_Reduce(sums, globalSums, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int globalInvalid;
    MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
    if (!rank) {
        const double ms = seconds * 1000;
        const double rate = seconds > 0 ? double(iterations) * globalCells / seconds / 1e9 : 0;
        printf("Computation time: %.3f ms\nPerformance:\n", ms);
        printf("  Time per iteration: %.4f ms\n", iterations ? ms / iterations : 0);
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
        printf("  Result hash: %016" PRIX64 "\n\n", globalHash);
    }
    if (results) {
        // Gather only when the original ordered external-results hash is requested.
        // Chunking avoids MPI's int count limit without replicating the mesh.
        std::vector<double> all;
        if (!rank) { all.resize(globalCells); std::copy(energy.begin(), energy.end(), all.begin()); }
        constexpr size_t chunk = 1 << 26;
        for (int source = 1; source < activeRanks; ++source) {
            const int sourceRows = n / activeRanks + (source < n % activeRanks);
            const size_t offset = size_t(source * (n / activeRanks) + std::min(source, n % activeRanks)) * n;
            const size_t count = size_t(sourceRows) * n;
            for (size_t begin = 0; begin < count; begin += chunk) {
                const int length = int(std::min(chunk, count - begin));
                if (rank == source) MPI_Send(energy.data() + begin, length, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                if (!rank) MPI_Recv(all.data() + offset + begin, length, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (!rank) print_results(all, "ElementEnergy");
    }
    int failed = 0;
    if (!rank && validate) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",
               globalSums[0], globalSums[1], globalMin, globalMax);
        failed = globalInvalid || !std::isfinite(globalSums[0]) || !std::isfinite(globalSums[1]) ||
                 !std::isfinite(globalMin) || !std::isfinite(globalMax);
        if (std::abs(globalSums[0]) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        printf("  Validation: %s\n", failed ? "FAILED" : "PASSED");
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failed;
}
