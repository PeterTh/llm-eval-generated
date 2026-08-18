#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

static void mpiCheck(int error, const char* what) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "%s: %.*s\n", what, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
}

static void cudaCheck(cudaError_t error, const char* what) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t firstRow,
                          size_t localRows, size_t n) {
    #pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(localRows); ++local) {
        const unsigned int source = static_cast<unsigned int>(firstRow + local);
        unsigned int* row = path.data() + static_cast<size_t>(local) * n;
        // Preserve the original predecessor/path initialization semantics.
        for (size_t j = 0; j < n; ++j) row[j] = source;
    }
}

__global__ void relaxRows(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const unsigned int* __restrict__ pivot,
                          size_t n, size_t localRows, size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= localRows || j >= n) return;
    const size_t offset = i * n;
    const unsigned int candidate = dist[offset + k] + pivot[j];
    if (candidate < dist[offset + j]) {
        dist[offset + j] = candidate;
        path[offset + j] = static_cast<unsigned int>(k);
    }
}

static int ownerOfRow(size_t row, size_t n, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t largeRows = (base + 1) * extra;
    if (row < largeRows) return static_cast<int>(row / (base + 1));
    return static_cast<int>(extra + (row - largeRows) / base);
}

static size_t firstRowOf(int rank, size_t n, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
}

static size_t rowsOf(int rank, size_t n, int ranks) {
    return n / static_cast<size_t>(ranks) + (static_cast<size_t>(rank) < n % ranks);
}

void floydWarshall(unsigned int* dDist, unsigned int* dPath, unsigned int* dPivot,
                   unsigned int* hostPivot,
                   size_t n, size_t firstRow, size_t localRows, int ranks) {
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x),
                    static_cast<unsigned int>((localRows + block.y - 1) / block.y));
    for (size_t k = 0; k < n; ++k) {
        const int owner = ownerOfRow(k, n, ranks);
        if (k >= firstRow && k < firstRow + localRows)
            cudaCheck(cudaMemcpy(dPivot, dDist + (k - firstRow) * n, n * sizeof(unsigned int),
                                 cudaMemcpyDeviceToDevice), "copy pivot row");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        // A CUDA-aware MPI sends the row directly between device memories.
        mpiCheck(MPI_Bcast(dPivot, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD),
                 "broadcast pivot row");
#else
        if (k >= firstRow && k < firstRow + localRows)
            cudaCheck(cudaMemcpy(hostPivot, dPivot, n * sizeof(unsigned int), cudaMemcpyDeviceToHost),
                      "stage pivot row to host");
        mpiCheck(MPI_Bcast(hostPivot, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD),
                 "broadcast pivot row");
        cudaCheck(cudaMemcpy(dPivot, hostPivot, n * sizeof(unsigned int), cudaMemcpyHostToDevice),
                  "stage pivot row to device");
#endif
        relaxRows<<<grid, block>>>(dDist, dPath, dPivot, n, localRows, k);
        cudaCheck(cudaGetLastError(), "launch Floyd-Warshall kernel");
        // MPI_Bcast in the next iteration also orders the preceding kernel on CUDA-aware MPI.
        cudaCheck(cudaDeviceSynchronize(), "execute Floyd-Warshall kernel");
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        valid &= (dist[idx2(i, i, n)] == 0);
    const size_t sample = std::min(n, size_t{10});
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(sample); ++i)
        for (long long j = 0; j < static_cast<long long>(sample); ++j)
            for (size_t k = 0; k < n; ++k) {
                const unsigned int dik = dist[idx2(k, i, n)];
                const unsigned int dkj = dist[idx2(j, k, n)];
                if (dik < INF && dkj < INF) valid &= (dik + dkj >= dist[idx2(j, i, n)]);
            }
    return valid != 0;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of nodes (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "initialize MPI");
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false, help = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else badArgs = true;
    }
    const bool mpiCountTooLarge = n != 0 && n > static_cast<size_t>(INT_MAX) / n;
    if (help || badArgs || n == 0 || n > static_cast<size_t>(INT_MAX) || mpiCountTooLarge ||
        ranks > static_cast<int>(n)) {
        if (rank == 0) {
            if (ranks > static_cast<int>(n)) std::fprintf(stderr, "Number of MPI ranks must not exceed nodes.\n");
            if (mpiCountTooLarge) std::fprintf(stderr, "Matrix exceeds the MPI Scatterv/Gatherv count limit.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (help && !badArgs) ? 0 : 1;
    }

    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm),
             "create node-local communicator");
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "query CUDA devices");
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "select CUDA device");
    MPI_Comm_free(&localComm);

    const size_t firstRow = firstRowOf(rank, n, ranks);
    const size_t localRows = rowsOf(rank, n, ranks);
    const size_t localElements = localRows * n;
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t count = rowsOf(r, n, ranks) * n;
        if (count > static_cast<size_t>(INT_MAX)) { if (rank == 0) std::fprintf(stderr, "Per-rank matrix portion exceeds MPI count limit.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(firstRowOf(r, n, ranks) * n);
    }

    std::vector<unsigned int> fullDist(rank == 0 ? n * n : 0);
    std::vector<unsigned int> localDist(localElements), localPath(localElements);
    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n"
                    "MPI ranks: %d, OpenMP threads/rank: %d\nValidation: %s\nInitializing graph...\n",
                    n, ranks, omp_get_max_threads(), validate ? "enabled" : "disabled");
        initializeDistanceMatrix(fullDist, n, 1, MAX_DISTANCE);
    }
    initializePathMatrix(localPath, firstRow, localRows, n);
    mpiCheck(MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                          localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD), "scatter matrix");

    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr, *hostPivot = nullptr;
    cudaCheck(cudaMalloc(&dDist, localElements * sizeof(unsigned int)), "allocate device distances");
    cudaCheck(cudaMalloc(&dPath, localElements * sizeof(unsigned int)), "allocate device paths");
    cudaCheck(cudaMalloc(&dPivot, n * sizeof(unsigned int)), "allocate device pivot");
    cudaCheck(cudaMallocHost(&hostPivot, n * sizeof(unsigned int)), "allocate pinned pivot staging buffer");
    cudaCheck(cudaMemcpy(dDist, localDist.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice), "upload distances");
    cudaCheck(cudaMemcpy(dPath, localPath.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice), "upload paths");

    if (rank == 0) std::printf("Computing shortest paths...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "start barrier");
    const double start = MPI_Wtime();
    floydWarshall(dDist, dPath, dPivot, hostPivot, n, firstRow, localRows, ranks);
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "finish barrier");
    const double elapsed = MPI_Wtime() - start;
    cudaCheck(cudaMemcpy(localDist.data(), dDist, localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost), "download distances");
    mpiCheck(MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? fullDist.data() : nullptr,
                         counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD), "gather matrix");
    cudaFreeHost(hostPivot); cudaFree(dPivot); cudaFree(dPath); cudaFree(dDist);

    int status = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", elapsed * 1000.0,
                    static_cast<double>(n) * n * n / elapsed / 1e9);
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            status = validateResult(fullDist, n) ? 0 : 1;
            std::printf("Validation: %s\n", status == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
