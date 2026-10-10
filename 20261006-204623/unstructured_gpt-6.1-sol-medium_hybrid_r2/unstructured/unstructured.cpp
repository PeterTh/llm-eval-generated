#include <algorithm>
#include <cerrno>
#include <cinttypes>
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

// All MPI calls are made by the main thread (MPI_THREAD_FUNNELED).
static void mpiCheck(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    fprintf(stderr, "%s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, status);
}
static void cudaCheck(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
}
#define MPI_CHECK(call) mpiCheck((call), #call)
#define CUDA_CHECK(call) cudaCheck((call), #call)

// buildSquare2D's connectivity and material data are implicit here: every edge
// has coefficient 1 and every material has transfer coefficient 0.8. This
// removes the large per-element connectivity table and its indirect loads.
// The neighbor traversal and arithmetic order match the original exactly.
__device__ __forceinline__ void updateElement(
    const double* __restrict__ energy, double* __restrict__ next,
    double* __restrict__ flux, int n, int firstRow, int row, int col) {
    const size_t i = size_t(row + 1) * n + col; // first/last rows are halos
    const int x = firstRow + row;
    const double e = energy[i];
    double total = 0.0;
    // Assignment order in buildSquare2D makes the single-element grid inflow.
    if ((x == 0 || x == n - 1) && (col == 0 || col == n - 1))
        total = (x == col) ? 0.5 : -0.5;
    if (x + 1 < n) total += ((energy[i + n] - e) * 0.8) * 0.25;
    if (x > 0)     total += ((energy[i - n] - e) * 0.8) * 0.25;
    if (col + 1 < n) total += ((energy[i + 1] - e) * 0.8) * 0.25;
    if (col > 0)     total += ((energy[i - 1] - e) * 0.8) * 0.25;
    next[i] = e + total;
    // Flux is only read by its owner; no second buffer or halo is needed.
    flux[size_t(row) * n + col] += fabs(total);
}

__global__ void updateRows(const double* __restrict__ energy,
                           double* __restrict__ next, double* __restrict__ flux,
                           int n, int firstRow, int beginRow, int endRow) {
    const size_t count = size_t(endRow - beginRow) * n;
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += size_t(blockDim.x) * gridDim.x) {
        const int row = beginRow + int(j / n);
        updateElement(energy, next, flux, n, firstRow, row, int(j % n));
    }
}

__global__ void updateBoundary(const double* __restrict__ energy,
                               double* __restrict__ next, double* __restrict__ flux,
                               int n, int firstRow, int rows) {
    const size_t count = size_t(rows == 1 ? 1 : 2) * n;
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += size_t(blockDim.x) * gridDim.x) {
        const int row = j < size_t(n) ? 0 : rows - 1;
        updateElement(energy, next, flux, n, firstRow, row, int(j % n));
    }
}

static int blocksFor(size_t count) {
    return int(std::min<size_t>((count + 255) / 256, 65535));
}

struct World {
    int n, rows, firstRow, rank, ranks;
    MPI_Comm comm;
    double *energy = nullptr, *next = nullptr, *flux = nullptr;
    double* halo = nullptr; // pinned send/receive buffers; portable to any MPI
    cudaStream_t compute{}, exchange{};
    cudaEvent_t ready{}, interiorDone{};

    World(int grid, MPI_Comm communicator) : n(grid), comm(communicator) {
        MPI_CHECK(MPI_Comm_rank(comm, &rank));
        MPI_CHECK(MPI_Comm_size(comm, &ranks));
        rows = n / ranks + (rank < n % ranks);
        firstRow = rank * (n / ranks) + std::min(rank, n % ranks);
        // Select GPUs by node-local rank, respecting CUDA_VISIBLE_DEVICES.
        MPI_Comm node;
        MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
        int localRank, devices;
        MPI_CHECK(MPI_Comm_rank(node, &localRank));
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        MPI_CHECK(MPI_Comm_free(&node));
        const size_t energyBytes = size_t(rows + 2) * n * sizeof(double);
        const size_t fluxBytes = size_t(rows) * n * sizeof(double);
        CUDA_CHECK(cudaMalloc(&energy, energyBytes));
        CUDA_CHECK(cudaMalloc(&next, energyBytes));
        CUDA_CHECK(cudaMalloc(&flux, fluxBytes));
        CUDA_CHECK(cudaMemset(energy, 0, energyBytes));
        CUDA_CHECK(cudaMemset(next, 0, energyBytes));
        CUDA_CHECK(cudaMemset(flux, 0, fluxBytes));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&interiorDone, cudaEventDisableTiming));
        if (ranks > 1)
            CUDA_CHECK(cudaMallocHost(&halo, size_t(4) * n * sizeof(double)));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ready, exchange));
    }

    void run(int iterations) {
        const size_t rowBytes = size_t(n) * sizeof(double);
        const int previous = rank > 0 ? rank - 1 : MPI_PROC_NULL;
        const int following = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        for (int iter = 0; iter < iterations; ++iter) {
            if (ranks == 1) {
                updateRows<<<blocksFor(size_t(rows) * n), 256, 0, compute>>>(
                    energy, next, flux, n, firstRow, 0, rows);
                CUDA_CHECK(cudaGetLastError());
            } else {
                MPI_Request requests[4];
                MPI_CHECK(MPI_Irecv(halo + 2 * n, n, MPI_DOUBLE, previous, 11, comm, &requests[0]));
                MPI_CHECK(MPI_Irecv(halo + 3 * n, n, MPI_DOUBLE, following, 10, comm, &requests[1]));
                // The event joins both kernels from the preceding iteration.
                CUDA_CHECK(cudaStreamWaitEvent(compute, ready, 0));
                if (rows > 2) {
                    updateRows<<<blocksFor(size_t(rows - 2) * n), 256, 0, compute>>>(
                        energy, next, flux, n, firstRow, 1, rows - 1);
                    CUDA_CHECK(cudaGetLastError());
                }
                CUDA_CHECK(cudaEventRecord(interiorDone, compute));
                if (previous != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(halo, energy + n, rowBytes,
                                               cudaMemcpyDeviceToHost, exchange));
                if (following != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(halo + n, energy + size_t(rows) * n, rowBytes,
                                               cudaMemcpyDeviceToHost, exchange));
                // Interior kernel continues while the host progresses MPI.
                CUDA_CHECK(cudaStreamSynchronize(exchange));
                MPI_CHECK(MPI_Isend(halo, n, MPI_DOUBLE, previous, 10, comm, &requests[2]));
                MPI_CHECK(MPI_Isend(halo + n, n, MPI_DOUBLE, following, 11, comm, &requests[3]));
                MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
                if (previous != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(energy, halo + 2 * n, rowBytes,
                                               cudaMemcpyHostToDevice, exchange));
                if (following != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(energy + size_t(rows + 1) * n, halo + 3 * n,
                                               rowBytes, cudaMemcpyHostToDevice, exchange));
                updateBoundary<<<blocksFor(size_t(rows == 1 ? 1 : 2) * n), 256, 0, exchange>>>(
                    energy, next, flux, n, firstRow, rows);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamWaitEvent(exchange, interiorDone, 0));
                CUDA_CHECK(cudaEventRecord(ready, exchange));
            }
            std::swap(energy, next);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaStreamSynchronize(exchange));
    }

    ~World() {
        CUDA_CHECK(cudaFree(energy));
        CUDA_CHECK(cudaFree(next));
        CUDA_CHECK(cudaFree(flux));
        if (halo) CUDA_CHECK(cudaFreeHost(halo));
        CUDA_CHECK(cudaEventDestroy(ready));
        CUDA_CHECK(cudaEventDestroy(interiorDone));
        CUDA_CHECK(cudaStreamDestroy(compute));
        CUDA_CHECK(cudaStreamDestroy(exchange));
    }
    World(const World&) = delete;
    World& operator=(const World&) = delete;
};

static bool reportResults(const World& world, bool validate, bool printResults) {
    const size_t count = size_t(world.rows) * world.n;
    std::vector<double> energy(count), flux(count);
    CUDA_CHECK(cudaMemcpy(energy.data(), world.energy + world.n, count * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), world.flux, count * sizeof(double), cudaMemcpyDeviceToHost));
    uint64_t localHash = 0, globalHash = 0;
    double energySum = 0.0, fluxSum = 0.0;
    double energyMin = std::numeric_limits<double>::max();
    double energyMax = std::numeric_limits<double>::lowest();
    int finite = 1;
    const uint64_t offset = uint64_t(world.firstRow) * world.n;
    // Each rank uses OpenMP for local verification. XOR plus global indices
    // makes the original bitwise hash independent of MPI decomposition.
    #pragma omp parallel for schedule(static) reduction(^:localHash) \
        reduction(+:energySum,fluxSum) reduction(min:energyMin) \
        reduction(max:energyMax) reduction(&:finite)
    for (size_t i = 0; i < count; ++i) {
        uint64_t e, f;
        std::memcpy(&e, &energy[i], sizeof(e));
        std::memcpy(&f, &flux[i], sizeof(f));
        localHash ^= (e + offset + i) * 0x9e3779b97f4a7c15ULL;
        localHash ^= (f + offset + i) * 0xbf58476d1ce4e5b9ULL;
        if (validate) {
            energySum += energy[i];
            fluxSum += flux[i];
            energyMin = std::min(energyMin, energy[i]);
            energyMax = std::max(energyMax, energy[i]);
            finite &= std::isfinite(energy[i]) && std::isfinite(flux[i]);
        }
    }
    MPI_CHECK(MPI_Reduce(&localHash, &globalHash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm));
    if (world.rank == 0) printf("  Result hash: %016" PRIX64 "\n\n", globalHash);
    if (printResults) {
        // Gather only when explicitly requested: print_results requires the
        // global array in the original element order (including its hash).
        std::vector<double> gathered;
        std::vector<int> counts(world.ranks), displacements(world.ranks);
        for (int r = 0; r < world.ranks; ++r) {
            counts[r] = (world.n / world.ranks + (r < world.n % world.ranks)) * world.n;
            displacements[r] = (r * (world.n / world.ranks) + std::min(r, world.n % world.ranks)) * world.n;
        }
        if (world.rank == 0) gathered.resize(size_t(world.n) * world.n);
        MPI_CHECK(MPI_Gatherv(energy.data(), int(count), MPI_DOUBLE, gathered.data(),
                             counts.data(), displacements.data(), MPI_DOUBLE, 0, world.comm));
        if (world.rank == 0) print_results(gathered, "ElementEnergy");
    }
    bool valid = true;
    if (validate) {
        double localSums[2] = {energySum, fluxSum}, sums[2];
        double low, high;
        int allFinite;
        MPI_CHECK(MPI_Allreduce(localSums, sums, 2, MPI_DOUBLE, MPI_SUM, world.comm));
        MPI_CHECK(MPI_Allreduce(&energyMin, &low, 1, MPI_DOUBLE, MPI_MIN, world.comm));
        MPI_CHECK(MPI_Allreduce(&energyMax, &high, 1, MPI_DOUBLE, MPI_MAX, world.comm));
        MPI_CHECK(MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, world.comm));
        valid = allFinite && std::isfinite(sums[0]) && std::isfinite(sums[1]) &&
                std::isfinite(low) && std::isfinite(high);
        if (world.rank == 0) {
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
                   "  Energy range: [%.6f, %.6f]\n", sums[0], sums[1], low, high);
            if (std::abs(sums[0]) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            printf("  Validation: %s\n", valid ? "PASSED" : "FAILED (non-finite result)");
        }
    }
    return valid;
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

static bool parseInteger(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < 0 || parsed > std::numeric_limits<int>::max())
        return false;
    value = int(parsed);
    return true;
}

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            bad |= !parseInteger(argv[++i], n);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            bad |= !parseInteger(argv[++i], iterations);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            bad = true;
        }
    }
    // The original mesh and MPI_Gatherv use signed-int element counts.
    if (n < 1 || uint64_t(n) * n > uint64_t(std::numeric_limits<int>::max())) bad = true;
    if (help || bad) {
        if (rank == 0) {
            if (bad) fprintf(stderr, "Require 1 <= n <= 46340 and nonnegative iterations.\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return bad ? 1 : 0;
    }

    // Exclude empty partitions when there are more ranks than grid rows.
    MPI_Comm active;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < n ? 0 : MPI_UNDEFINED, rank, &active));
    int result = 0;
    if (active != MPI_COMM_NULL) {
        try {
            World world(n, active);
            if (rank == 0) {
                printf("Unstructured Mesh Energy Transfer Benchmark\n");
                printf("============================================\n");
                printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, uint64_t(n) * n);
                printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
                printf("MPI ranks: %d active (%d launched); OpenMP threads/rank: %d; CUDA enabled\n\n",
                       world.ranks, ranks, omp_get_max_threads());
                const double dynamicBytes = (2.0 * (n + 2.0 * world.ranks) * n + double(n) * n) * sizeof(double);
                printf("Building unstructured mesh...\n");
                printf("Memory usage: %.2f MB (device; implicit connectivity)\n\n", dynamicBytes / (1024.0 * 1024.0));
                printf("Running simulation...\n");
            }
            MPI_CHECK(MPI_Barrier(active));
            const double start = MPI_Wtime();
            world.run(iterations);
            const double localSeconds = MPI_Wtime() - start;
            double seconds;
            MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active));
            if (rank == 0) {
                const double milliseconds = seconds * 1000.0;
                const double rate = seconds > 0 ? (double(iterations) * n * n) / seconds / 1e9 : 0;
                printf("Computation time: %.3f ms\nPerformance:\n", milliseconds);
                printf("  Time per iteration: %.4f ms\n", iterations ? milliseconds / iterations : 0);
                printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22.0);
            }
            result = reportResults(world, validate, printResults) ? 0 : 1;
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_CHECK(MPI_Comm_free(&active));
    }
    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return result;
}
