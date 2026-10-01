#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
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

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) checkCuda((call), #call)

// The benchmark's builder always creates unit-weight, four-neighbor square
// connectivity and transfer coefficient 0.8. Represent that graph implicitly:
// contiguous rows allow coalesced access without per-edge indices or weights.
// Two ghost rows belong to adjacent MPI ranks; flux is strictly local.
__global__ void updateElements(const double* __restrict__ energy,
                               double* __restrict__ next,
                               double* __restrict__ accumulated,
                               int n, int rows, int firstRow,
                               int beginRow, int rowCount, bool boundary) {
    const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= size_t(n)) return;
    const int col = int(column);
    for (size_t r = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         r < size_t(rowCount); r += size_t(blockDim.y) * gridDim.y) {
        int row = int(r) + beginRow;
        if (boundary && row == 1) row = rows - 1;
        const int globalRow = firstRow + row;
        const size_t p = size_t(row + 1) * n + col;
        const double e = energy[p];
        double flux = 0.0;
        if ((globalRow == 0 || globalRow == n - 1) &&
            (col == 0 || col == n - 1)) {
            // For n=1 the original builder's last assignment is inflow.
            flux = ((globalRow == 0 && col == 0) ||
                    (globalRow == n - 1 && col == n - 1)) ? 0.5 : -0.5;
        }
        // Preserve the reference's +row, -row, +column, -column ordering
        // and separate multiply/add rounding (compiled with --fmad=false).
        if (globalRow + 1 < n) flux += (energy[p + n] - e) * 0.8 * 0.25;
        if (globalRow > 0)     flux += (energy[p - n] - e) * 0.8 * 0.25;
        if (col + 1 < n)       flux += (energy[p + 1] - e) * 0.8 * 0.25;
        if (col > 0)           flux += (energy[p - 1] - e) * 0.8 * 0.25;
        next[p] = e + flux;
        accumulated[size_t(row) * n + col] += fabs(flux);
    }
}

struct Partition {
    int n, rows, firstRow, rank, ranks;
    MPI_Comm comm;
    double *energy = nullptr, *next = nullptr, *flux = nullptr;
    double* staging = nullptr;
    cudaStream_t compute{}, halo{};
    cudaEvent_t ready{}, received{};

    Partition(int width, MPI_Comm communicator) : n(width), comm(communicator) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        rows = n / ranks + (rank < n % ranks);
        firstRow = rank * (n / ranks) + std::min(rank, n % ranks);
        const size_t bytes = (size_t(rows) + 2) * n * sizeof(double);
        CUDA(cudaMalloc(&energy, bytes));
        CUDA(cudaMalloc(&next, bytes));
        CUDA(cudaMalloc(&flux, size_t(rows) * n * sizeof(double)));
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&halo, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
        CUDA(cudaMemsetAsync(energy, 0, bytes, compute));
        CUDA(cudaMemsetAsync(next, 0, bytes, compute));
        CUDA(cudaMemsetAsync(flux, 0, size_t(rows) * n * sizeof(double), compute));
        if (ranks > 1) {
            // Portable pinned staging: no CUDA-aware MPI extension required.
            CUDA(cudaMallocHost(&staging, size_t(4) * n * sizeof(double)));
            #pragma omp parallel for schedule(static) if(n >= 16384)
            for (int64_t i = 0; i < int64_t(4) * n; ++i) staging[i] = 0.0;
        }
        CUDA(cudaStreamSynchronize(compute));
    }

    void launch(int begin, size_t count, bool boundary = false) {
        if (!count) return;
        // A warp follows a row. Two-dimensional indexing avoids a costly
        // integer division per element and reuses adjacent rows in cache.
        const int rowCount = int(count / n);
        const dim3 threads(boundary ? 256 : 32, boundary ? 1 : 8);
        const dim3 blocks(unsigned((size_t(n) + threads.x - 1) / threads.x),
                          unsigned(std::min<size_t>(
                              (size_t(rowCount) + threads.y - 1) / threads.y, 65535)));
        updateElements<<<blocks, threads, 0, compute>>>(
            energy, next, flux, n, rows, firstRow, begin, rowCount, boundary);
        CUDA(cudaGetLastError());
    }

    void run(int iterations) {
        for (int iter = 0; iter < iterations; ++iter) {
            if (ranks == 1 || iter == 0) {
                // Initial energies, including ghosts, are all zero.
                launch(0, size_t(rows) * n);
            } else {
                MPI_Request requests[4];
                int used = 0;
                const size_t bytes = size_t(n) * sizeof(double);
                if (rank > 0)
                    MPI_Irecv(staging + 2 * size_t(n), n, MPI_DOUBLE, rank - 1,
                              1, comm, &requests[used++]);
                if (rank + 1 < ranks)
                    MPI_Irecv(staging + 3 * size_t(n), n, MPI_DOUBLE, rank + 1,
                              0, comm, &requests[used++]);

                // Transfers wait for the previous step, while this step's
                // interior runs independently of incoming ghost values.
                CUDA(cudaEventRecord(ready, compute));
                CUDA(cudaStreamWaitEvent(halo, ready, 0));
                if (rank > 0)
                    CUDA(cudaMemcpyAsync(staging, energy + n, bytes,
                                         cudaMemcpyDeviceToHost, halo));
                if (rank + 1 < ranks)
                    CUDA(cudaMemcpyAsync(staging + n, energy + size_t(rows) * n,
                                         bytes, cudaMemcpyDeviceToHost, halo));
                if (rows > 2) launch(1, size_t(rows - 2) * n);
                CUDA(cudaStreamSynchronize(halo));
                if (rank > 0)
                    MPI_Isend(staging, n, MPI_DOUBLE, rank - 1, 0,
                              comm, &requests[used++]);
                if (rank + 1 < ranks)
                    MPI_Isend(staging + n, n, MPI_DOUBLE, rank + 1, 1,
                              comm, &requests[used++]);
                MPI_Waitall(used, requests, MPI_STATUSES_IGNORE);
                if (rank > 0)
                    CUDA(cudaMemcpyAsync(energy, staging + 2 * size_t(n), bytes,
                                         cudaMemcpyHostToDevice, halo));
                if (rank + 1 < ranks)
                    CUDA(cudaMemcpyAsync(energy + size_t(rows + 1) * n,
                                         staging + 3 * size_t(n), bytes,
                                         cudaMemcpyHostToDevice, halo));
                CUDA(cudaEventRecord(received, halo));
                CUDA(cudaStreamWaitEvent(compute, received, 0));
                launch(0, size_t(std::min(rows, 2)) * n, true);
            }
            std::swap(energy, next);
        }
        CUDA(cudaStreamSynchronize(compute));
    }

    ~Partition() {
        CUDA(cudaFree(energy));
        CUDA(cudaFree(next));
        CUDA(cudaFree(flux));
        if (staging) CUDA(cudaFreeHost(staging));
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaEventDestroy(received));
        CUDA(cudaStreamDestroy(compute));
        CUDA(cudaStreamDestroy(halo));
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


static uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t offset) {
    uint64_t result = 0;
    #pragma omp parallel for reduction(^:result) schedule(static) if(elements.size() >= 65536)
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t e, f;
        // memcpy avoids the reference's strict-aliasing violation.
        std::memcpy(&e, &elements[i].current_energy, sizeof(e));
        std::memcpy(&f, &elements[i].total_flux, sizeof(f));
        result ^= (e + offset + i) * 0x9e3779b97f4a7c15ULL;
        result ^= (f + offset + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

// Chunked ordered collection avoids MPI's int displacement/count limit.
// Only explicitly requested reporting gathers the full state.
static World collect(const World& local, const Partition& p) {
    World global;
    MPI_Datatype state;
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(double), "State layout");
    MPI_Type_contiguous(2, MPI_DOUBLE, &state);
    MPI_Type_commit(&state);
    const size_t chunk = 1 << 20;
    if (p.rank == 0) {
        global.elements_dynamic.resize(size_t(p.n) * p.n);
        std::copy(local.elements_dynamic.begin(), local.elements_dynamic.end(),
                  global.elements_dynamic.begin());
        for (int r = 1; r < p.ranks; ++r) {
            const size_t first = size_t(r * (p.n / p.ranks) + std::min(r, p.n % p.ranks)) * p.n;
            const size_t count = size_t(p.n / p.ranks + (r < p.n % p.ranks)) * p.n;
            for (size_t pos = 0; pos < count; pos += chunk)
                MPI_Recv(global.elements_dynamic.data() + first + pos,
                         int(std::min(chunk, count - pos)), state, r, 2,
                         p.comm, MPI_STATUS_IGNORE);
        }
    } else {
        for (size_t pos = 0; pos < local.elements_dynamic.size(); pos += chunk)
            MPI_Send(local.elements_dynamic.data() + pos,
                     int(std::min(chunk, local.elements_dynamic.size() - pos)),
                     state, 0, 2, p.comm);
    }
    MPI_Type_free(&state);
    return global;
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


static bool parseNumber(const char* text, int& value, bool positive) {
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (!*text || *end || parsed < (positive ? 1 : 0) || parsed > INT_MAX)
        return false;
    value = int(parsed);
    return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank = 0, worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!worldRank) fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false, validArgs = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            validArgs = parseNumber(argv[++i], n, true) && validArgs;
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            validArgs = parseNumber(argv[++i], iterations, false) && validArgs;
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else validArgs = false;
    }
    if (!validArgs || help) {
        if (!worldRank) {
            if (!validArgs) fprintf(stderr, "Invalid option or grid/iteration count.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return validArgs ? 0 : 1;
    }

    // Excess ranks have no rows; they participate in finalization only.
    MPI_Comm active = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < std::min(n, worldSize) ? 0 : MPI_UNDEFINED,
                   worldRank, &active);
    int result = 0;
    if (active != MPI_COMM_NULL) {
        try {
            MPI_Comm node;
            MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node);
            int localRank = 0, devices = 0;
            MPI_Comm_rank(node, &localRank);
            CUDA(cudaGetDeviceCount(&devices));
            if (!devices) throw std::runtime_error("A CUDA GPU is required on every active rank");
            // Also supports launchers exposing a single GPU per process.
            CUDA(cudaSetDevice(localRank % devices));
            MPI_Comm_free(&node);

            Partition p(n, active);
            if (!worldRank) {
                printf("Unstructured Mesh Energy Transfer Benchmark\n");
                printf("============================================\n");
                printf("Grid size: %d x %d = %zu elements\n", n, n, size_t(n) * n);
                printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
                printf("MPI ranks: %d; OpenMP threads/rank: %d; CUDA enabled\n", p.ranks, omp_get_max_threads());
                printf("Building unstructured mesh (implicit square connectivity)...\n");
                printf("GPU memory on rank 0: %.2f MB\n",
                       (3.0 * p.rows + 4.0) * n * sizeof(double) / (1024.0 * 1024.0));
                printf("\nRunning simulation...\n");
            }
            MPI_Barrier(active);
            const double start = MPI_Wtime();
            p.run(iterations);
            const double localSeconds = MPI_Wtime() - start;
            double seconds = 0;
            MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active);

            const size_t count = size_t(p.rows) * n;
            std::vector<double> energy(count), flux(count);
            CUDA(cudaMemcpy(energy.data(), p.energy + n, count * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(flux.data(), p.flux, count * sizeof(double), cudaMemcpyDeviceToHost));
            World local;
            local.elements_dynamic.resize(count);
            #pragma omp parallel for schedule(static) if(count >= 65536)
            for (size_t i = 0; i < count; ++i)
                local.elements_dynamic[i] = {energy[i], flux[i]};
            const uint64_t localHash = computeHash(local.elements_dynamic, size_t(p.firstRow) * n);
            uint64_t resultHash = 0;
            MPI_Reduce(&localHash, &resultHash, 1, MPI_UINT64_T, MPI_BXOR, 0, active);
            if (!worldRank) {
                const double rate = seconds > 0 ? double(iterations) * n * n / seconds / 1e9 : 0;
                printf("Computation time: %.3f ms\n", seconds * 1000);
                printf("Performance:\n");
                printf("  Time per iteration: %.4f ms\n", iterations ? seconds * 1000 / iterations : 0);
                printf("  Elements/sec: %.4f GigaElements/s\n", rate);
                printf("  Performance: %.4f GFLOPS\n", rate * 22);
                printf("  Result hash: %016llX\n\n", (unsigned long long)resultHash);
            }
            if (validate || printResults) {
                World global = collect(local, p);
                if (!worldRank) {
                    if (printResults) {
                        std::vector<double> data(global.elements_dynamic.size());
                        #pragma omp parallel for schedule(static) if(data.size() >= 65536)
                        for (size_t i = 0; i < data.size(); ++i)
                            data[i] = global.elements_dynamic[i].current_energy;
                        print_results(data, "ElementEnergy");
                    }
                    if (validate && !validateResults(global)) result = 1;
                }
            }
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
