#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static void failCuda(cudaError_t error, const char* operation, MPI_Comm comm) {
    if (error == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", rank, operation,
                 cudaGetErrorString(error));
    MPI_Abort(comm, EXIT_FAILURE);
}

#define CUDA_CHECK(call, comm) failCuda((call), #call, (comm))

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    // Keep the original rand_r stream exactly, so single- and multi-rank runs use
    // the same graph as the sequential benchmark.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i)
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] = 0;
}

static size_t firstRow(int rank, size_t n, int ranks) {
    return n * static_cast<size_t>(rank) / static_cast<size_t>(ranks);
}

__global__ void floydWarshallStep(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ pivot,
                                  size_t n, size_t localRows, unsigned int k) {
    const size_t row = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    if (row >= localRows) return;

    __shared__ unsigned int dik;
    if (threadIdx.x == 0) dik = dist[row * n + k];
    __syncthreads();

    unsigned int* const rowDist = dist + row * n;
    unsigned int* const rowPath = path + row * n;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        const unsigned int candidate = dik + pivot[j];
        if (candidate < rowDist[j]) {
            rowDist[j] = candidate;
            rowPath[j] = k;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath, size_t n, size_t localRows,
                   const std::vector<int>& owner, MPI_Comm comm) {
    unsigned int *deviceDist = nullptr, *devicePath = nullptr, *devicePivot = nullptr;
    unsigned int* hostPivot = nullptr;
    const size_t localBytes = localRows * n * sizeof(unsigned int);
    const size_t rowBytes = n * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&deviceDist, std::max<size_t>(localBytes, 1)), comm);
    CUDA_CHECK(cudaMalloc(&devicePath, std::max<size_t>(localBytes, 1)), comm);
    CUDA_CHECK(cudaMalloc(&devicePivot, std::max<size_t>(rowBytes, 1)), comm);
    CUDA_CHECK(cudaMallocHost(&hostPivot, std::max<size_t>(rowBytes, 1)), comm);
    if (localBytes) {
        CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data(), localBytes, cudaMemcpyHostToDevice), comm);
        CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(), localBytes, cudaMemcpyHostToDevice), comm);
    }

    int rank = 0, ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const size_t localFirst = firstRow(rank, n, ranks);
    constexpr unsigned int threads = 256;
    const unsigned int gx = static_cast<unsigned int>(std::min<size_t>(localRows, 65535));
    const unsigned int gy = gx ? static_cast<unsigned int>((localRows + gx - 1) / gx) : 1;

    for (size_t k = 0; k < n; ++k) {
        if (rank == owner[k]) {
            const size_t localK = k - localFirst;
            CUDA_CHECK(cudaMemcpy(hostPivot, deviceDist + localK * n, rowBytes,
                                  cudaMemcpyDeviceToHost), comm);
        }
        MPI_Bcast(hostPivot, static_cast<int>(n), MPI_UNSIGNED, owner[k], comm);
        CUDA_CHECK(cudaMemcpy(devicePivot, hostPivot, rowBytes, cudaMemcpyHostToDevice), comm);
        if (localRows) {
            floydWarshallStep<<<dim3(gx, gy), threads>>>(
                deviceDist, devicePath, devicePivot, n, localRows, static_cast<unsigned int>(k));
            CUDA_CHECK(cudaGetLastError(), comm);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize(), comm);
    if (localBytes)
        CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist, localBytes, cudaMemcpyDeviceToHost), comm);

    CUDA_CHECK(cudaFreeHost(hostPivot), comm);
    CUDA_CHECK(cudaFree(devicePivot), comm);
    CUDA_CHECK(cudaFree(devicePath), comm);
    CUDA_CHECK(cudaFree(deviceDist), comm);
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        valid &= (dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), n)] == 0);

    const size_t sample = std::min(n, static_cast<size_t>(10));
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(sample); ++i) {
        for (long long j = 0; j < static_cast<long long>(sample); ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int dij = dist[idx2(static_cast<size_t>(j), static_cast<size_t>(i), n)];
                const unsigned int dik = dist[idx2(k, static_cast<size_t>(i), n)];
                const unsigned int dkj = dist[idx2(static_cast<size_t>(j), k, n)];
                if (dik < INF && dkj < INF) valid &= (dik + dkj >= dij);
            }
        }
    }
    if (!valid) std::printf("Validation failed: shortest-path invariants violated\n");
    return valid != 0;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false, badArguments = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long long parsed = std::atoll(argv[++i]);
            if (parsed <= 0) badArguments = true; else n = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else badArguments = true;
    }
    if (help || badArguments) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badArguments ? 1 : 0;
    }
    if (n > static_cast<size_t>(INT_MAX) || n > static_cast<size_t>(UINT_MAX)) {
        if (rank == 0) std::fprintf(stderr, "Matrix dimension exceeds MPI/CUDA index limits\n");
        MPI_Finalize();
        return 1;
    }

    // Bind ranks round-robin to GPUs on their node.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), MPI_COMM_WORLD);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "The hybrid benchmark requires at least one CUDA GPU per node\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount), MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), displacements(ranks), owner(n);
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = firstRow(r, n, ranks), end = firstRow(r + 1, n, ranks);
        const size_t elements = (end - begin) * n;
        const size_t displacement = begin * n;
        if (elements > static_cast<size_t>(INT_MAX) || displacement > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI collectives\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        counts[r] = static_cast<int>(elements);
        displacements[r] = static_cast<int>(displacement);
        for (size_t row = begin; row < end; ++row) owner[row] = r;
    }
    const size_t localFirst = firstRow(rank, n, ranks);
    const size_t localRows = firstRow(rank + 1, n, ranks) - localFirst;
    std::vector<unsigned int> fullDist;
    if (rank == 0) {
        fullDist.resize(n * n);
        initializeDistanceMatrix(fullDist, n, 1, MAX_DISTANCE);
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n", n);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\nValidation: %s\n", ranks,
                    omp_get_max_threads(), validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\nComputing shortest paths...\n");
    }
    std::vector<unsigned int> localDist(localRows * n), localPath(localRows * n);
    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, counts.data(), displacements.data(),
                 MPI_UNSIGNED, localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(localRows); ++row) {
        const size_t globalRow = localFirst + static_cast<size_t>(row);
        // Both assignments in the original initializer resolve to the source
        // (matrix row) index for every off-diagonal entry.
        for (size_t j = 0; j < n; ++j)
            localPath[static_cast<size_t>(row) * n + j] = static_cast<unsigned int>(globalRow);
    }
    if (rank == 0) std::vector<unsigned int>().swap(fullDist);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(localDist, localPath, n, localRows, owner, MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0 && (validate || printResults)) fullDist.resize(n * n);
    if (validate || printResults)
        MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr, counts.data(), displacements.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        const double gops = static_cast<double>(n) * n * n / std::max(seconds, 1.0e-12) / 1.0e9;
        std::printf("Computation time: %lld ms\nPerformance: %.3f GOPS\n", milliseconds, gops);
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(fullDist, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
