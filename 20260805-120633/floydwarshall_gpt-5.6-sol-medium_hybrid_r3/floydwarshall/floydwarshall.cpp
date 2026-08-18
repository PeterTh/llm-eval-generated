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

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] static void abortCuda(cudaError_t error, const char* expression,
                                   const char* file, int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA error at %s:%d for %s: %s\n", rank, file,
                 line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        const cudaError_t cuda_check_error = (expr);                            \
        if (cuda_check_error != cudaSuccess)                                    \
            abortCuda(cuda_check_error, #expr, __FILE__, __LINE__);             \
    } while (false)

static void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                                     const size_t numNodes,
                                     const unsigned int rangeMin,
                                     const unsigned int rangeMax) {
    // Keep the benchmark's original deterministic rand_r sequence exactly.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(
                                 range * rand_r(&seed) / static_cast<double>(RAND_MAX));

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i)
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] = 0;
}

__global__ static void floydWarshallStep(unsigned int* __restrict__ dist,
                                         unsigned int* __restrict__ path,
                                         const unsigned int* __restrict__ pivot,
                                         size_t numNodes, size_t localRows,
                                         size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t localI = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (j >= numNodes || localI >= localRows)
        return;

    const size_t offset = localI * numNodes + j;
    const unsigned int candidate = dist[localI * numNodes + k] + pivot[j];
    if (candidate < dist[offset]) {
        dist[offset] = candidate;
        path[offset] = static_cast<unsigned int>(k);
    }
}

static bool validateResult(const std::vector<unsigned int>& dist,
                           const size_t numNodes) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i)
        valid &= dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] == 0;
    if (!valid) {
        std::printf("Validation failed: a diagonal element is not zero\n");
        return false;
    }

    const size_t sample = std::min(numNodes, static_cast<size_t>(10));
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(sample); ++ii) {
        for (long long jj = 0; jj < static_cast<long long>(sample); ++jj) {
            const size_t i = static_cast<size_t>(ii);
            const size_t j = static_cast<size_t>(jj);
            const unsigned int distIJ = dist[idx2(j, i, numNodes)];
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF)
                    valid &= distIK + distKJ >= distIJ;
            }
        }
    }
    if (!valid)
        std::printf("Validation failed: triangle inequality violated\n");
    return valid != 0;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0' || value == 0 || value > UINT_MAX)
                parseStatus = 1;
            else
                numNodes = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            parseStatus = 2;
        } else {
            if (rank == 0)
                std::printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1;
        }
    }
    if (parseStatus != 0) {
        if (rank == 0)
            printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (numNodes > static_cast<size_t>(INT_MAX) ||
        numNodes > static_cast<size_t>(INT_MAX) / numNodes) {
        if (rank == 0)
            std::fprintf(stderr, "Matrix is too large for MPI collective counts\n");
        MPI_Finalize();
        return 1;
    }

    // Assign one rank per GPU within each shared-memory node.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0)
            std::fprintf(stderr, "The hybrid benchmark requires CUDA GPUs\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t extraRows = numNodes % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localElements = localRows * numNodes;

    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows);
        const size_t first = static_cast<size_t>(r) * baseRows +
                             std::min(static_cast<size_t>(r), extraRows);
        counts[r] = static_cast<int>(rows * numNodes);
        displacements[r] = static_cast<int>(first * numNodes);
    }

    std::vector<unsigned int> fullDist;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize,
                    omp_get_max_threads());
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> localDist(localElements);
    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, counts.data(),
                 displacements.data(), MPI_UNSIGNED, localDist.data(),
                 counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    fullDist.clear();
    fullDist.shrink_to_fit();

    std::vector<unsigned int> localPath(localElements);
    #pragma omp parallel for schedule(static)
    for (long long localI = 0; localI < static_cast<long long>(localRows); ++localI) {
        const unsigned int source = static_cast<unsigned int>(
            firstRow + static_cast<size_t>(localI));
        std::fill_n(localPath.data() + static_cast<size_t>(localI) * numNodes,
                    numNodes, source);
    }

    unsigned int *deviceDist = nullptr, *devicePath = nullptr, *devicePivot = nullptr;
    const size_t localBytes = std::max<size_t>(localElements, 1) * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&deviceDist, localBytes));
    CUDA_CHECK(cudaMalloc(&devicePath, localBytes));
    CUDA_CHECK(cudaMalloc(&devicePivot, numNodes * sizeof(unsigned int)));
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data(), localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(), localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }
    localPath.clear();
    localPath.shrink_to_fit();

    unsigned int* pivot = nullptr;
    CUDA_CHECK(cudaMallocHost(&pivot, numNodes * sizeof(unsigned int)));
    if (rank == 0)
        std::printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((numNodes + block.x - 1) / block.x),
                    static_cast<unsigned int>((localRows + block.y - 1) / block.y));

    int owner = 0;
    for (size_t k = 0; k < numNodes; ++k) {
        while (owner + 1 < worldSize &&
               k >= static_cast<size_t>(displacements[owner] + counts[owner]) / numNodes)
            ++owner;
        if (rank == owner) {
            const size_t localK = k - firstRow;
            CUDA_CHECK(cudaMemcpy(pivot, deviceDist + localK * numNodes,
                                  numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(pivot, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(devicePivot, pivot, numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        if (localRows != 0) {
            floydWarshallStep<<<grid, block>>>(deviceDist, devicePath, devicePivot,
                                               numNodes, localRows, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (localElements != 0)
        CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist,
                              localElements * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFreeHost(pivot));
    CUDA_CHECK(cudaFree(devicePivot));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));

    if (validate || printResults) {
        if (rank == 0)
            fullDist.resize(numNodes * numNodes);
        MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr, counts.data(),
                    displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    int exitStatus = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsedSeconds * 1000.0);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = elapsedSeconds > 0.0 ? operations / elapsedSeconds / 1.0e9 : 0.0;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f GOPS\n", gops);
        if (printResults)
            print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitStatus = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitStatus;
}
