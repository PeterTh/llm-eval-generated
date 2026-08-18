#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static void mpiCheck(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    std::fprintf(stderr, "MPI error in %s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, status);
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d CUDA error in %s: %s\n", rank, operation,
                 cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
}

#define MPI_CHECK(call) mpiCheck((call), #call)
#define CUDA_CHECK(call) cudaCheck((call), #call)

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
            range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    // Preserve the original random graph exactly; only the independent diagonal
    // stores are parallelized on the host.
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] = 0;
    }
}

__global__ void initializePathKernel(unsigned int* path, size_t localRows,
                                     size_t rowOffset, size_t n) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = localRows * n;
    if (element < elements) path[element] = static_cast<unsigned int>(rowOffset + element / n);
}

// A block updates one 256-column tile of one local row.  The pivot tile is
// staged once per block, and d[i,k] is broadcast from shared memory.
__global__ void floydWarshallStep(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ pivot,
                                  size_t n, size_t localRows, size_t k) {
    const size_t localRow = blockIdx.y;
    const size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localRow >= localRows) return;

    __shared__ unsigned int pivotTile[256];
    __shared__ unsigned int distIK;
    if (threadIdx.x == 0) distIK = dist[localRow * n + k];
    if (column < n) pivotTile[threadIdx.x] = pivot[column];
    __syncthreads();

    if (column < n) {
        const size_t index = localRow * n + column;
        const unsigned int candidate = distIK + pivotTile[threadIdx.x];
        if (candidate < dist[index]) {
            dist[index] = candidate;
            path[index] = static_cast<unsigned int>(k);
        }
    }
}

static int ownerOfRow(size_t row, size_t n, int ranks) {
    const size_t wideRanks = n % static_cast<size_t>(ranks);
    const size_t wideRows = n / static_cast<size_t>(ranks) + 1;
    const size_t wideEnd = wideRanks * wideRows;
    if (row < wideEnd) return static_cast<int>(row / wideRows);
    const size_t narrowRows = n / static_cast<size_t>(ranks);
    return static_cast<int>(wideRanks + (row - wideEnd) / narrowRows);
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    int valid = 1;
#pragma omp parallel for reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        valid &= (dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] == 0);
    }
    if (!valid) {
        std::printf("Validation failed: a diagonal element is not zero\n");
        return false;
    }

    const size_t sample = std::min(numNodes, static_cast<size_t>(10));
#pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(sample); ++i) {
        for (long long j = 0; j < static_cast<long long>(sample); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(static_cast<size_t>(j), static_cast<size_t>(i), numNodes)];
                const unsigned int distIK = dist[idx2(k, static_cast<size_t>(i), numNodes)];
                const unsigned int distKJ = dist[idx2(static_cast<size_t>(j), k, numNodes)];
                if (distIK < INF && distKJ < INF) valid &= (distIK + distKJ >= distIJ);
            }
        }
    }
    if (!valid) std::printf("Validation failed: triangle inequality violated\n");
    return valid != 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    size_t numNodes = 512;
    bool validate = false, printResults = false, showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            argumentsValid &= end != argv[i] && *end == '\0' && value > 0 &&
                              value <= std::numeric_limits<unsigned int>::max();
            numNodes = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) showHelp = true;
        else argumentsValid = false;
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            if (!argumentsValid) std::printf("Invalid command line\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }
    if (numNodes > static_cast<size_t>(std::sqrt(static_cast<double>(INT_MAX)))) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI collective counts\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s) per rank\n",
                    ranks, omp_get_max_threads());
        std::printf("Initializing graph...\n");
    }

    std::vector<int> counts(ranks), displacements(ranks);
    std::vector<size_t> rowOffsets(ranks);
    size_t nextRow = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = numNodes / ranks + (static_cast<size_t>(r) < numNodes % ranks);
        rowOffsets[r] = nextRow;
        counts[r] = static_cast<int>(rows * numNodes);
        displacements[r] = static_cast<int>(nextRow * numNodes);
        nextRow += rows;
    }
    const size_t localRows = static_cast<size_t>(counts[rank]) / numNodes;
    const size_t localElements = localRows * numNodes;

    std::vector<unsigned int> dist;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    std::vector<unsigned int> localDist(localElements);
    MPI_CHECK(MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), displacements.data(),
                           MPI_UNSIGNED, localDist.data(), counts[rank], MPI_UNSIGNED, 0,
                           MPI_COMM_WORLD));

    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "CUDA-capable GPU required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    unsigned int *deviceDist = nullptr, *devicePath = nullptr, *devicePivot = nullptr;
    const size_t allocationElements = std::max<size_t>(localElements, 1);
    CUDA_CHECK(cudaMalloc(&deviceDist, allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePath, allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePivot, numNodes * sizeof(unsigned int)));
    if (localElements) {
        CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data(), localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        const int threads = 256;
        const int blocks = static_cast<int>((localElements + threads - 1) / threads);
        initializePathKernel<<<blocks, threads>>>(devicePath, localRows, rowOffsets[rank], numNodes);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned int* pivot = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pivot, numNodes * sizeof(unsigned int), cudaHostAllocPortable));

    if (rank == 0) std::printf("Computing shortest paths...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const dim3 threads(256);
    const dim3 blocks(static_cast<unsigned int>((numNodes + threads.x - 1) / threads.x),
                      static_cast<unsigned int>(localRows));
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, numNodes, ranks);
        if (rank == owner) {
            const size_t localK = k - rowOffsets[rank];
            CUDA_CHECK(cudaMemcpy(pivot, deviceDist + localK * numNodes,
                                  numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Bcast(pivot, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpyAsync(devicePivot, pivot, numNodes * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice));
        if (localRows) {
            floydWarshallStep<<<blocks, threads>>>(deviceDist, devicePath, devicePivot,
                                                   numNodes, localRows, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (printResults || validate) {
        if (localElements) {
            CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist,
                                  localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (rank == 0 && dist.empty()) dist.resize(numNodes * numNodes);
        MPI_CHECK(MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED,
                              rank == 0 ? dist.data() : nullptr, counts.data(), displacements.data(),
                              MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    }

    int exitCode = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        std::printf("Performance: %.3f GOPS\n", ops / seconds / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaFreeHost(pivot));
    CUDA_CHECK(cudaFree(devicePivot));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));
    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
