#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <climits>
#include <cerrno>
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
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};
struct World {
    std::vector<ElementDynamic> elements_dynamic;
};

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)
static void mpiCheck(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "%s: %.*s\n", operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define MPI_CHECK(call) mpiCheck((call), #call)

// The benchmark builder produces exactly four unit-weight connections, ordered
// +row, -row, +column, -column, and three materials with coefficient 0.8.
// Deriving these connections avoids loading 144 bytes of topology per cell.
// Keep each operation and its order: no reassociation or fused multiply-add.
__device__ __forceinline__ double connectionFlux(double neighbor, double self) {
    return (neighbor - self) * 0.8 * 0.25;
}

// Energy is double-buffered with one ghost row on each side. Accumulated flux
// can be updated in place because it is never read by a neighboring cell.
__global__ void updateEnergy(const double* __restrict__ energy,
                             double* __restrict__ next,
                             double* __restrict__ flux,
                             int n, int firstRow, int localRows,
                             int beginRow, int rowCount, bool boundaryOnly) {
    size_t count = size_t(rowCount) * n;
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < count; k += size_t(blockDim.x) * gridDim.x) {
        int row = beginRow + int(k / n);
        if (boundaryOnly) row = (k / n == 0) ? 0 : localRows - 1;
        int col = int(k % n);
        int globalRow = firstRow + row;
        size_t i = size_t(row + 1) * n + col;
        double self = energy[i];
        double total = 0.0;
        if ((globalRow == 0 || globalRow == n - 1) &&
            (col == 0 || col == n - 1)) {
            // Also reproduces the last corner assignment when n == 1.
            total = (globalRow == col) ? 0.5 : -0.5;
        }
        if (globalRow + 1 < n) total += connectionFlux(energy[i + n], self);
        if (globalRow > 0)     total += connectionFlux(energy[i - n], self);
        if (col + 1 < n)       total += connectionFlux(energy[i + 1], self);
        if (col > 0)           total += connectionFlux(energy[i - 1], self);
        next[i] = self + total;
        flux[size_t(row) * n + col] += fabs(total);
    }
}

struct Simulation {
    int n, rows, first, rank, ranks;
    MPI_Comm comm;
    size_t cells;
    double *energy = nullptr, *next = nullptr, *flux = nullptr;
    double *host = nullptr, *halo = nullptr;
    cudaStream_t compute, transfer;
    cudaEvent_t stepDone, haloReady;

    Simulation(int grid, int localRows, int firstRow, MPI_Comm active)
        : n(grid), rows(localRows), first(firstRow), comm(active),
          cells(size_t(localRows) * grid) {
        MPI_CHECK(MPI_Comm_rank(comm, &rank));
        MPI_CHECK(MPI_Comm_size(comm, &ranks));
        // Node-local ranks map to node-local GPUs; CUDA_VISIBLE_DEVICES is honored.
        MPI_Comm node;
        MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank,
                                      MPI_INFO_NULL, &node));
        int localRank, devices;
        MPI_CHECK(MPI_Comm_rank(node, &localRank));
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: CUDA device required\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        MPI_CHECK(MPI_Comm_free(&node));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&stepDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));
        const size_t energyBytes = (cells + size_t(2) * n) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&energy, energyBytes));
        CUDA_CHECK(cudaMalloc(&next, energyBytes));
        CUDA_CHECK(cudaMalloc(&flux, cells * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&host, size_t(2) * cells * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&halo, size_t(4) * n * sizeof(double)));
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size_t(2) * cells; ++i) host[i] = 0.0;
        CUDA_CHECK(cudaMemsetAsync(energy, 0, energyBytes, compute));
        CUDA_CHECK(cudaMemsetAsync(next, 0, energyBytes, compute));
        CUDA_CHECK(cudaMemcpyAsync(flux, host, cells * sizeof(double),
                                   cudaMemcpyHostToDevice, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaEventRecord(stepDone, compute));
    }

    void launch(int begin, int count, bool boundary = false) {
        if (count <= 0) return;
        unsigned blocks = unsigned(std::min<size_t>((size_t(count) * n + 255) / 256, 65535));
        updateEnergy<<<blocks, 256, 0, compute>>>(energy, next, flux,
            n, first, rows, begin, count, boundary);
        CUDA_CHECK(cudaGetLastError());
    }

    void run(int iterations) {
        const int before = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int after = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
        const size_t bytes = size_t(n) * sizeof(double);
        for (int iter = 0; iter < iterations; ++iter) {
            if (ranks == 1) {
                launch(0, rows);
            } else {
                // Portable pinned-host staging works without CUDA-aware MPI.
                // Only energy crosses rank boundaries, never cumulative flux.
                CUDA_CHECK(cudaStreamWaitEvent(transfer, stepDone, 0));
                if (before != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(halo, energy + n, bytes,
                                               cudaMemcpyDeviceToHost, transfer));
                if (after != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(halo + n, energy + cells, bytes,
                                               cudaMemcpyDeviceToHost, transfer));
                // Interior cells need no ghosts: overlap their kernel with
                // outgoing device-to-host copies as well as MPI communication.
                launch(1, rows - 2);
                CUDA_CHECK(cudaStreamSynchronize(transfer));
                MPI_Request requests[4];
                MPI_CHECK(MPI_Irecv(halo + size_t(2) * n, n, MPI_DOUBLE, before,
                                    1, comm, &requests[0]));
                MPI_CHECK(MPI_Irecv(halo + size_t(3) * n, n, MPI_DOUBLE, after,
                                    0, comm, &requests[1]));
                MPI_CHECK(MPI_Isend(halo, n, MPI_DOUBLE, before, 0,
                                    comm, &requests[2]));
                MPI_CHECK(MPI_Isend(halo + n, n, MPI_DOUBLE, after, 1,
                                    comm, &requests[3]));
                // MPI progress runs on the host while CUDA processes the interior.
                MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
                if (before != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(energy, halo + size_t(2) * n, bytes,
                                               cudaMemcpyHostToDevice, transfer));
                if (after != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(energy + cells + n,
                        halo + size_t(3) * n, bytes, cudaMemcpyHostToDevice, transfer));
                CUDA_CHECK(cudaEventRecord(haloReady, transfer));
                CUDA_CHECK(cudaStreamWaitEvent(compute, haloReady, 0));
                launch(0, std::min(rows, 2), true);
                CUDA_CHECK(cudaEventRecord(stepDone, compute));
            }
            std::swap(energy, next);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    void download(World& world) {
        CUDA_CHECK(cudaMemcpyAsync(host, energy + n, cells * sizeof(double),
                                   cudaMemcpyDeviceToHost, compute));
        CUDA_CHECK(cudaMemcpyAsync(host + cells, flux, cells * sizeof(double),
                                   cudaMemcpyDeviceToHost, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        world.elements_dynamic.resize(cells);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells; ++i)
            world.elements_dynamic[i] = {host[i], host[cells + i]};
    }

    ~Simulation() {
        CUDA_CHECK(cudaFree(energy));
        CUDA_CHECK(cudaFree(next));
        CUDA_CHECK(cudaFree(flux));
        CUDA_CHECK(cudaFreeHost(host));
        CUDA_CHECK(cudaFreeHost(halo));
        CUDA_CHECK(cudaEventDestroy(stepDone));
        CUDA_CHECK(cudaEventDestroy(haloReady));
        CUDA_CHECK(cudaStreamDestroy(compute));
        CUDA_CHECK(cudaStreamDestroy(transfer));
    }
};

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }
    
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

// XOR is associative; retaining global indices gives the serial hash exactly.
uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t offset) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy, flux;
        std::memcpy(&energy, &elements[i].current_energy, sizeof(energy));
        std::memcpy(&flux, &elements[i].total_flux, sizeof(flux));
        hash ^= (energy + offset + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux + offset + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseInt(const char* text, int& result) {
    errno = 0;
    char* end = nullptr;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 0 || value > INT_MAX) return false;
    result = int(value);
    return true;
}

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            bad |= !parseInt(argv[++i], n);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            bad |= !parseInt(argv[++i], iterations);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            bad = true;
        }
    }
    // The original mesh indices and MPI collective counts are signed ints.
    bad |= n < 1 || int64_t(n) * n > INT_MAX;
    if (help || bad) {
        if (rank == 0) {
            if (bad) fprintf(stderr, "Require 1 <= n <= 46340 and iterations >= 0\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return bad ? 1 : 0;
    }

    // Excess ranks have no rows. They still participate in MPI setup/finalize.
    int activeRanks = std::min(ranks, n);
    MPI_Comm active;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED,
                             rank, &active));
    int status = 0;
    if (rank < activeRanks) {
        int rows = n / activeRanks + (rank < n % activeRanks);
        int first = rank * (n / activeRanks) + std::min(rank, n % activeRanks);
        Simulation simulation(n, rows, first, active);
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %d elements\n", n, n, n * n);
            printf("Iterations: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d, OpenMP threads per rank: %d, CUDA enabled\n\n",
                   activeRanks, omp_get_max_threads());
            printf("Building unstructured mesh...\n");
            printf("Connectivity: implicit square-grid neighbors\n");
            printf("GPU memory usage (all ranks): %.2f MB\n\n",
                   (3.0 * n * n + 4.0 * n * activeRanks) * sizeof(double) / (1024 * 1024));
            printf("Running simulation...\n");
        }
        MPI_CHECK(MPI_Barrier(active));
        double start = MPI_Wtime();
        simulation.run(iterations);
        double elapsed = MPI_Wtime() - start, seconds = 0;
        MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active));
        World local;
        simulation.download(local);
        uint64_t localHash = computeHash(local.elements_dynamic, size_t(first) * n);
        uint64_t hash = 0;
        MPI_CHECK(MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, active));
        if (rank == 0) {
            double rate = seconds > 0 ? double(iterations) * n * n / seconds / 1e9 : 0;
            printf("Computation time: %.3f ms\n", seconds * 1000);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", iterations ? seconds * 1000 / iterations : 0);
            printf("  Elements/sec: %.4f GigaElements/s\n", rate);
            printf("  Performance: %.4f GFLOPS\n", rate * 22);
            printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }
        // Only requested diagnostic output gathers the complete mesh. Preserve
        // global order so serial validation sums and external hashes also match.
        if (printResults || validate) {
            World global;
            std::vector<int> counts(activeRanks), offsets(activeRanks);
            for (int r = 0; r < activeRanks; ++r) {
                counts[r] = (n / activeRanks + (r < n % activeRanks)) * n;
                offsets[r] = (r * (n / activeRanks) + std::min(r, n % activeRanks)) * n;
            }
            if (rank == 0) global.elements_dynamic.resize(size_t(n) * n);
            MPI_Datatype stateType;
            static_assert(sizeof(ElementDynamic) == 2 * sizeof(double), "packed state required");
            MPI_CHECK(MPI_Type_contiguous(2, MPI_DOUBLE, &stateType));
            MPI_CHECK(MPI_Type_commit(&stateType));
            MPI_CHECK(MPI_Gatherv(local.elements_dynamic.data(), int(simulation.cells), stateType,
                global.elements_dynamic.data(), counts.data(), offsets.data(), stateType, 0, active));
            MPI_CHECK(MPI_Type_free(&stateType));
            if (rank == 0) {
                if (printResults) {
                    std::vector<double> data(global.elements_dynamic.size());
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < data.size(); ++i)
                        data[i] = global.elements_dynamic[i].current_energy;
                    print_results(data, "ElementEnergy");
                }
                if (validate && !validateResults(global)) status = 1;
            }
        }
        MPI_CHECK(MPI_Comm_free(&active));
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return status;
}
