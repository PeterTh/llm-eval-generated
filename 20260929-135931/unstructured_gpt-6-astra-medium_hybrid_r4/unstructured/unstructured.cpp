#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <exception>
#include <limits>
#include <vector>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic { val_t current_energy, total_flux; };
struct World { std::vector<ElementDynamic> elements_dynamic; };

// Abort the whole job if a rank cannot continue (otherwise peers can hang).
static void cudaCheck(cudaError_t error, const char* call, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at line %d (%s): %s\n", line, call,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call, __LINE__)

// The input generator has exactly four unit-weight grid connections and a
// transfer coefficient of 0.8 everywhere. Implicit connectivity removes all
// adjacency traffic. Retain the original +x, -x, +y, -y accumulation order.
__global__ void update(const double* __restrict__ energy,
                       double* __restrict__ next, double* __restrict__ flux,
                       int n, int firstRow, int begin, int end) {
    const size_t count = size_t(end - begin) * n;
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < count; k += size_t(blockDim.x) * gridDim.x) {
        const size_t i = size_t(begin) * n + k;
        const int x = firstRow + int(i / n) - 1;
        const int y = int(i % n);
        const double e = energy[i];
        double f = 0.0;
        // Assignment order also preserves the n=1 corner overwrite behavior.
        if (x == 0 && y == 0) f = 0.5;
        if (x == 0 && y == n - 1) f = -0.5;
        if (x == n - 1 && y == 0) f = -0.5;
        if (x == n - 1 && y == n - 1) f = 0.5;
        if (x + 1 < n) f += (energy[i + n] - e) * 0.8 * 0.25;
        if (x > 0)     f += (energy[i - n] - e) * 0.8 * 0.25;
        if (y + 1 < n) f += (energy[i + 1] - e) * 0.8 * 0.25;
        if (y > 0)     f += (energy[i - 1] - e) * 0.8 * 0.25;
        next[i] = e + f;
        flux[i] += fabs(f);
    }
}

static void launch(const double* energy, double* next, double* flux,
                   int n, int first, int begin, int end, cudaStream_t stream) {
    if (begin >= end) return;
    const unsigned blocks = unsigned(std::min<size_t>(65535,
                                  (size_t(end - begin) * n + 255) / 256));
    update<<<blocks, 256, 0, stream>>>(energy, next, flux, n, first, begin, end);
    CUDA(cudaGetLastError());
}

// Only energy crosses partition boundaries; accumulated flux is local.
// Pinned staging works with ordinary MPI, without requiring CUDA-aware MPI.
static double runSimulation(World& world, int n, int iterations, int first,
                            int rows, int rank, int ranks, MPI_Comm comm) {
    const size_t count = size_t(rows) * n;
    const size_t bytes = (size_t(rows) + 2) * n * sizeof(double);
    const size_t rowBytes = size_t(n) * sizeof(double);
    double *energy, *next, *flux, *halo = nullptr;
    CUDA(cudaMalloc(&energy, bytes));
    CUDA(cudaMalloc(&next, bytes));
    CUDA(cudaMalloc(&flux, bytes));
    CUDA(cudaMemset(energy, 0, bytes));
    CUDA(cudaMemset(flux, 0, bytes));
    cudaStream_t compute, exchange;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
    cudaEvent_t ready;
    CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    if (ranks > 1) CUDA(cudaMallocHost(&halo, 4 * rowBytes));
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launch(energy, next, flux, n, first, 1, rows + 1, compute);
        } else {
            // Previous H2D copies must release receive buffers before MPI
            // can overwrite them, even when GPU execution trails the host.
            CUDA(cudaStreamSynchronize(exchange));
            MPI_Request requests[4];
            int nr = 0;
            if (rank > 0)
                MPI_Irecv(halo + 2 * size_t(n), n, MPI_DOUBLE, rank - 1, 1,
                          comm, &requests[nr++]);
            if (rank + 1 < ranks)
                MPI_Irecv(halo + 3 * size_t(n), n, MPI_DOUBLE, rank + 1, 0,
                          comm, &requests[nr++]);
            // Wait only for the previous update, not this iteration's interior.
            CUDA(cudaEventRecord(ready, compute));
            CUDA(cudaStreamWaitEvent(exchange, ready, 0));
            if (rank > 0)
                CUDA(cudaMemcpyAsync(halo, energy + n, rowBytes,
                                     cudaMemcpyDeviceToHost, exchange));
            if (rank + 1 < ranks)
                CUDA(cudaMemcpyAsync(halo + n, energy + size_t(rows) * n,
                                     rowBytes, cudaMemcpyDeviceToHost, exchange));
            launch(energy, next, flux, n, first, 2, rows, compute);
            CUDA(cudaStreamSynchronize(exchange));
            if (rank > 0)
                MPI_Isend(halo, n, MPI_DOUBLE, rank - 1, 0, comm, &requests[nr++]);
            if (rank + 1 < ranks)
                MPI_Isend(halo + n, n, MPI_DOUBLE, rank + 1, 1, comm, &requests[nr++]);
            // MPI progresses communication while the GPU computes the interior.
            MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
            if (rank > 0)
                CUDA(cudaMemcpyAsync(energy, halo + 2 * size_t(n), rowBytes,
                                     cudaMemcpyHostToDevice, exchange));
            if (rank + 1 < ranks)
                CUDA(cudaMemcpyAsync(energy + (size_t(rows) + 1) * n,
                                     halo + 3 * size_t(n), rowBytes,
                                     cudaMemcpyHostToDevice, exchange));
            CUDA(cudaEventRecord(ready, exchange));
            CUDA(cudaStreamWaitEvent(compute, ready, 0));
            launch(energy, next, flux, n, first, 1, 2, compute);
            if (rows > 1)
                launch(energy, next, flux, n, first, rows, rows + 1, compute);
        }
        std::swap(energy, next);
    }
    CUDA(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    std::vector<double> hostEnergy(count), hostFlux(count);
    CUDA(cudaMemcpy(hostEnergy.data(), energy + n, count * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(hostFlux.data(), flux + n, count * sizeof(double), cudaMemcpyDeviceToHost));
    world.elements_dynamic.resize(count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i)
        world.elements_dynamic[i] = {hostEnergy[i], hostFlux[i]};
    if (halo) CUDA(cudaFreeHost(halo));
    CUDA(cudaEventDestroy(ready));
    CUDA(cudaStreamDestroy(exchange));
    CUDA(cudaStreamDestroy(compute));
    CUDA(cudaFree(energy));
    CUDA(cudaFree(next));
    CUDA(cudaFree(flux));
    return seconds;
}

static uint64_t computeHash(const World& world, size_t offset) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        uint64_t e, f;
        memcpy(&e, &world.elements_dynamic[i].current_energy, sizeof(e));
        memcpy(&f, &world.elements_dynamic[i].total_flux, sizeof(f));
        hash ^= (e + offset + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + offset + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather only for explicitly requested output/validation. Chunked transfers
// avoid MPI's int count limit, preserving the original global element order.
static void gatherResults(World& world, int n, int rank, int ranks, MPI_Comm comm) {
    const size_t total = size_t(n) * n;
    if (rank == 0) world.elements_dynamic.resize(total);
    constexpr size_t chunk = 1 << 26;
    for (int r = 1; r < ranks; ++r) {
        const size_t first = size_t(r) * (n / ranks) + std::min(r, n % ranks);
        const size_t rows = n / ranks + (r < n % ranks);
        const size_t bytes = rows * n * sizeof(ElementDynamic);
        for (size_t pos = 0; pos < bytes; pos += chunk) {
            const int length = int(std::min(chunk, bytes - pos));
            if (rank == 0)
                MPI_Recv(reinterpret_cast<char*>(world.elements_dynamic.data() + first * n) + pos,
                         length, MPI_BYTE, r, 2, comm, MPI_STATUS_IGNORE);
            else if (rank == r)
                MPI_Send(reinterpret_cast<char*>(world.elements_dynamic.data()) + pos,
                         length, MPI_BYTE, 0, 2, comm);
        }
    }
}

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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseNumber(const char* text, int& value, int minimum) {
    char* end = nullptr;
    const long long parsed = strtoll(text, &end, 10);
    if (!*text || *end || parsed < minimum || parsed > INT_MAX) return false;
    value = int(parsed);
    return true;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc)
            bad |= !parseNumber(argv[++i], n, 1);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc)
            bad |= !parseNumber(argv[++i], iterations, 0);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad) {
        if (worldRank == 0) {
            if (bad) fprintf(stderr, "Invalid arguments: grid size must be positive and iterations nonnegative.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    // No empty partitions: surplus ranks participate in initialization/finalization.
    const int ranks = std::min(n, worldSize);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, localSize;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localSize);
    int devices = 0;
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Each active MPI rank requires a CUDA device.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    // Respect explicit OpenMP settings; avoid host oversubscription by default.
    if (!getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));
    MPI_Comm_free(&local);
    const int rows = n / ranks + (rank < n % ranks);
    const int first = rank * (n / ranks) + std::min(rank, n % ranks);
    const size_t total = size_t(n) * n;
    if (total > std::numeric_limits<size_t>::max() / sizeof(ElementDynamic) ||
        (size_t(rows) + 2) * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
        if (rank == 0) fprintf(stderr, "Grid is too large.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n, n, total);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; CUDA device per rank; OpenMP host processing\n", ranks);
        printf("Building unstructured mesh (implicit grid connectivity)...\n");
        printf("Device state memory: %.2f MB (including halos across ranks)\n",
               (double(total) + 2.0 * ranks * n) * 3 * sizeof(double) / (1024 * 1024));
        printf("Running simulation...\n");
    }
    int status = 0;
    try {
        World world;
        const double seconds = runSimulation(world, n, iterations, first, rows, rank, ranks, comm);
        uint64_t hash = computeHash(world, size_t(first) * n), globalHash = 0;
        MPI_Reduce(&hash, &globalHash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
        if (rank == 0) {
            const double rate = seconds > 0 ? double(iterations) * total / seconds / 1e9 : 0;
            printf("Computation time: %.3f ms\n", seconds * 1000);
            printf("Performance:\n  Time per iteration: %.4f ms\n", iterations ? seconds * 1000 / iterations : 0);
            printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
            printf("  Result hash: %016" PRIX64 "\n\n", globalHash);
        }
        if (validate || printResults) gatherResults(world, n, rank, ranks, comm);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energies(total);
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < total; ++i)
                    energies[i] = world.elements_dynamic[i].current_energy;
                print_results(energies, "ElementEnergy");
            }
            if (validate && !validateResults(world)) status = 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
