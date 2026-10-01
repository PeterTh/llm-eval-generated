#include <algorithm>
#include <climits>
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

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

static void checkCuda(cudaError_t status, int rank, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void checkMpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI operation failed: %s\n", operation);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    // The generator is sequential so that the graph is identical for every rank count.
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        dist[idx2(i, i, n)] = 0;
}

// Each thread owns one output element. Positive edge weights and a zero
// diagonal keep row k and column k unchanged during iteration k.
__global__ void updateRows(unsigned int* __restrict__ dist,
                           unsigned int* __restrict__ path,
                           const unsigned int* __restrict__ pivot,
                           size_t n, size_t localRows, size_t k) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= localRows || j >= n) return;
    const size_t offset = i * n + j;
    const unsigned int old = dist[offset];
    const unsigned int candidate = dist[i * n + k] + pivot[j];
    if (candidate < old) {
        dist[offset] = candidate;
        path[offset] = static_cast<unsigned int>(k);
    }
}

static int rowsForRank(size_t n, int size, int rank) {
    return static_cast<int>(n / size + (static_cast<size_t>(rank) < n % size));
}

static size_t firstRow(size_t n, int size, int rank) {
    return static_cast<size_t>(rank) * (n / size) +
           std::min(static_cast<size_t>(rank), n % size);
}

void floydWarshall(std::vector<unsigned int>& dist,
                   size_t n, int rank, int size, bool gatherResult) {
    const size_t localRows = rowsForRank(n, size, rank);
    const size_t localStart = firstRow(n, size, rank);
    const size_t localElements = localRows * n;
    std::vector<int> counts(size), offsets(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = rowsForRank(n, size, r) * static_cast<int>(n);
        offsets[r] = static_cast<int>(firstRow(n, size, r) * n);
    }

    std::vector<unsigned int> hostRows(localElements);
    std::vector<unsigned int> hostPath(localElements);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows; ++i)
        for (size_t j = 0; j < n; ++j)
            hostPath[i * n + j] = static_cast<unsigned int>(j);

    checkMpi(MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(),
                          offsets.data(), MPI_UNSIGNED, hostRows.data(),
                          counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD), "Scatterv");

    unsigned int *deviceRows = nullptr, *devicePath = nullptr, *devicePivot = nullptr;
    const size_t allocatedElements = std::max<size_t>(localElements, 1);
    checkCuda(cudaMalloc(&deviceRows, allocatedElements * sizeof(unsigned int)), rank, "allocate distances");
    checkCuda(cudaMalloc(&devicePath, allocatedElements * sizeof(unsigned int)), rank, "allocate paths");
    checkCuda(cudaMalloc(&devicePivot, n * sizeof(unsigned int)), rank, "allocate pivot");
    if (localElements) {
        checkCuda(cudaMemcpy(deviceRows, hostRows.data(), localElements * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), rank, "upload distances");
        checkCuda(cudaMemcpy(devicePath, hostPath.data(), localElements * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), rank, "upload paths");
    }
    hostPath.clear();
    hostPath.shrink_to_fit();

    std::vector<unsigned int> pivot(n);
    const dim3 threads(32, 8);
    const dim3 blocks((n + threads.x - 1) / threads.x,
                      (localRows + threads.y - 1) / threads.y);
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(
            std::upper_bound(offsets.begin(), offsets.end(), static_cast<int>(k * n)) - offsets.begin() - 1);
        if (rank == owner)
            checkCuda(cudaMemcpy(pivot.data(), deviceRows + (k - localStart) * n,
                                 n * sizeof(unsigned int), cudaMemcpyDeviceToHost),
                      rank, "download pivot row");
        checkMpi(MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED,
                           owner, MPI_COMM_WORLD), "Bcast pivot row");
        if (localRows) {
            checkCuda(cudaMemcpy(devicePivot, pivot.data(), n * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice), rank, "upload pivot row");
            updateRows<<<blocks, threads>>>(deviceRows, devicePath, devicePivot, n, localRows, k);
            checkCuda(cudaGetLastError(), rank, "update rows kernel");
        }
    }
    if (localElements)
        checkCuda(cudaMemcpy(hostRows.data(), deviceRows, localElements * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), rank, "download distances");
    if (gatherResult)
        checkMpi(MPI_Gatherv(hostRows.data(), counts[rank], MPI_UNSIGNED,
                             rank == 0 ? dist.data() : nullptr, counts.data(),
                             offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD), "Gatherv");
    checkCuda(cudaFree(devicePivot), rank, "free pivot");
    checkCuda(cudaFree(devicePath), rank, "free paths");
    checkCuda(cudaFree(deviceRows), rank, "free distances");
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int invalid = 0;
#pragma omp parallel for reduction(|:invalid) schedule(static)
    for (size_t i = 0; i < n; ++i)
        invalid |= (dist[idx2(i, i, n)] != 0);
    const size_t sample = std::min(n, static_cast<size_t>(10));
#pragma omp parallel for collapse(2) reduction(|:invalid) schedule(static)
    for (size_t i = 0; i < sample; ++i)
        for (size_t j = 0; j < sample; ++j)
            for (size_t k = 0; k < n; ++k) {
                const unsigned int a = dist[idx2(k, i, n)];
                const unsigned int b = dist[idx2(j, k, n)];
                if (a < INF && b < INF)
                    invalid |= (a + b < dist[idx2(j, i, n)]);
            }
    if (invalid) printf("Validation failed: invalid diagonal or triangle inequality\n");
    return invalid == 0;
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    checkMpi(MPI_Init(&argc, &argv), "Init");
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = strtoull(argv[++i], &end, 10);
            if (*end != '\0' || parsed == 0 || parsed > static_cast<unsigned long long>(INT_MAX) / parsed) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes\n");
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (n > INT_MAX || n > static_cast<size_t>(INT_MAX) / n || n > UINT_MAX) {
        if (rank == 0) fprintf(stderr, "Number of nodes exceeds MPI or path index limits\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm shared;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &shared), "local communicator");
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(shared, &localRank);
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "query CUDA devices");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), rank, "select CUDA device");
    MPI_Comm_free(&shared);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(rank == 0 ? n * n : 0);
    if (rank == 0) initializeDistanceMatrix(dist, n, 1, MAX_DISTANCE);
    if (rank == 0) printf("Computing shortest paths...\n");
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "start barrier");
    const double start = MPI_Wtime();
    floydWarshall(dist, n, rank, size, validate || printResults);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    checkMpi(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD), "timing reduction");
    int result = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", ops / maxElapsed / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            result = validateResult(dist, n) ? 0 : 1;
            printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    checkMpi(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD), "result broadcast");
    MPI_Finalize();
    return result;
}
