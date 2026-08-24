#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int CUDA_BLOCK_X = 32;
constexpr int CUDA_BLOCK_Y = 8;

// The distributed representation is row-major by source node.  The original
// benchmark is column-major, with idx2(source, destination, n) =
// destination * n + source.  Keeping complete source rows local makes every
// GPU update contiguous and only requires one n-element MPI broadcast per k.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

inline constexpr size_t rowIndex(const size_t source, const size_t destination,
                                 const size_t n) noexcept {
    return source * n + destination;
}

struct Partition {
    int rank = 0;
    int ranks = 1;
    size_t firstRow = 0;
    size_t localRows = 0;
    std::vector<int> elementCounts;
    std::vector<int> elementDisplacements;
};

Partition makePartition(const size_t numNodes, const int rank, const int ranks) {
    Partition partition;
    partition.rank = rank;
    partition.ranks = ranks;
    partition.firstRow = (numNodes * static_cast<size_t>(rank)) /
                         static_cast<size_t>(ranks);
    const size_t endRow = (numNodes * static_cast<size_t>(rank + 1)) /
                          static_cast<size_t>(ranks);
    partition.localRows = endRow - partition.firstRow;
    partition.elementCounts.resize(static_cast<size_t>(ranks));
    partition.elementDisplacements.resize(static_cast<size_t>(ranks));

    for (int r = 0; r < ranks; ++r) {
        const size_t begin = (numNodes * static_cast<size_t>(r)) /
                             static_cast<size_t>(ranks);
        const size_t end = (numNodes * static_cast<size_t>(r + 1)) /
                           static_cast<size_t>(ranks);
        const size_t count = (end - begin) * numNodes;
        const size_t displacement = begin * numNodes;
        if (count > static_cast<size_t>(INT_MAX) ||
            displacement > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) {
                std::fprintf(stderr,
                             "Matrix is too large for the MPI implementation's "
                             "int-count collectives\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        partition.elementCounts[static_cast<size_t>(r)] = static_cast<int>(count);
        partition.elementDisplacements[static_cast<size_t>(r)] =
            static_cast<int>(displacement);
    }
    return partition;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    // This intentionally remains serial: rand_r's sequence and the original
    // column-major initialization order are part of the benchmark's result.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    }

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                               const size_t localRows, const size_t numNodes) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long localSource = 0;
         localSource < static_cast<long long>(localRows); ++localSource) {
        for (long long destination = 0;
             destination < static_cast<long long>(numNodes); ++destination) {
            path[rowIndex(static_cast<size_t>(localSource),
                          static_cast<size_t>(destination), numNodes)] =
                static_cast<unsigned int>(destination);
        }
    }
}

// Each block handles a 32-column tile and eight source rows.  The current row
// of the Floyd-Warshall recurrence is cached in shared memory, as are the
// eight dist[source][k] values, avoiding a repeated global load for every
// destination in a row.
__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int* __restrict__ rowK,
                                    const size_t localRows, const size_t numNodes,
                                    const size_t k) {
    __shared__ unsigned int sharedRowK[CUDA_BLOCK_X];
    __shared__ unsigned int sharedDistIK[CUDA_BLOCK_Y];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const size_t destination = static_cast<size_t>(blockIdx.x) * CUDA_BLOCK_X + tx;
    const size_t localSource = static_cast<size_t>(blockIdx.y) * CUDA_BLOCK_Y + ty;

    const size_t rowStart = static_cast<size_t>(blockIdx.x) * CUDA_BLOCK_X;
    if (rowStart + tx < numNodes) {
        sharedRowK[tx] = rowK[rowStart + tx];
    } else {
        sharedRowK[tx] = 0;
    }
    if (tx == 0) {
        sharedDistIK[ty] = localSource < localRows
                               ? dist[localSource * numNodes + k]
                               : 0;
    }
    __syncthreads();

    if (localSource < localRows && destination < numNodes) {
        const size_t cell = localSource * numNodes + destination;
        const unsigned int newDist = sharedDistIK[ty] + sharedRowK[tx];
        const unsigned int oldDist = dist[cell];
        if (newDist < oldDist) {
            dist[cell] = newDist;
            path[cell] = static_cast<unsigned int>(k);
        }
    }
}

void cudaCheck(const cudaError_t error, const char* operation, const int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA failure in %s: %s\n", rank,
                     operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
}

int selectCudaDevice(const int rank) {
    int localRank = 0;
    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localCommunicator);
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_free(&localCommunicator);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        std::fprintf(stderr, "MPI rank %d: no CUDA device is available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    const int device = localRank % deviceCount;
    cudaCheck(cudaSetDevice(device), "cudaSetDevice", rank);
    cudaCheck(cudaFree(nullptr), "CUDA context initialization", rank);
    return device;
}

void packColumnMajorToRows(const std::vector<unsigned int>& columnMajor,
                           std::vector<unsigned int>& rows,
                           const size_t numNodes) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long source = 0; source < static_cast<long long>(numNodes); ++source) {
        for (long long destination = 0;
             destination < static_cast<long long>(numNodes); ++destination) {
            rows[rowIndex(static_cast<size_t>(source),
                          static_cast<size_t>(destination), numNodes)] =
                columnMajor[idx2(static_cast<size_t>(source),
                                 static_cast<size_t>(destination), numNodes)];
        }
    }
}

void unpackRowsToColumnMajor(const std::vector<unsigned int>& rows,
                             std::vector<unsigned int>& columnMajor,
                             const size_t numNodes) {
    columnMajor.resize(numNodes * numNodes);
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long source = 0; source < static_cast<long long>(numNodes); ++source) {
        for (long long destination = 0;
             destination < static_cast<long long>(numNodes); ++destination) {
            columnMajor[idx2(static_cast<size_t>(source),
                             static_cast<size_t>(destination), numNodes)] =
                rows[rowIndex(static_cast<size_t>(source),
                              static_cast<size_t>(destination), numNodes)];
        }
    }
}

bool validateRows(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[rowIndex(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                        i, i);
            return false;
        }
    }

    const size_t sample = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[rowIndex(i, j, numNodes)];
                const unsigned int distIK = dist[rowIndex(i, k, numNodes)];
                const unsigned int distKJ = dist[rowIndex(k, j, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf(
                        "Validation failed: triangle inequality violated at "
                        "[%zu,%zu,%zu]\n",
                        i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' ||
                parsed > static_cast<unsigned long long>(
                             std::numeric_limits<size_t>::max())) {
                if (rank == 0) {
                    std::fprintf(stderr, "Invalid node count: %s\n", argv[i]);
                }
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const bool matrixCountTooLarge =
        numNodes != 0 && numNodes > static_cast<size_t>(INT_MAX) / numNodes;
    if (numNodes > static_cast<size_t>(INT_MAX) || matrixCountTooLarge ||
        (numNodes != 0 && static_cast<size_t>(ranks) > numNodes)) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "This benchmark requires ranks <= nodes and node/matrix "
                         "dimensions within MPI's int-count limit\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    const int device = selectCudaDevice(rank);
    const Partition partition = makePartition(numNodes, rank, ranks);
    const size_t localElements = partition.localRows * numNodes;
    std::vector<int> rowOwners(numNodes);
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = (numNodes * static_cast<size_t>(r)) /
                             static_cast<size_t>(ranks);
        const size_t end = (numNodes * static_cast<size_t>(r + 1)) /
                           static_cast<size_t>(ranks);
        std::fill(rowOwners.begin() + begin, rowOwners.begin() + end, r);
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("CUDA device per rank: selected with local-rank affinity "
                    "(rank 0 device %d)\n",
                    device);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::fflush(stdout);
    }

    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    initializeLocalPathMatrix(localPath, partition.localRows, numNodes);

    // Rank zero creates the exact original graph.  It is packed once into
    // contiguous source rows, after which the O(n^2) state is distributed.
    std::vector<unsigned int> packedRows;
    if (rank == 0) {
        std::vector<unsigned int> originalColumnMajor(numNodes * numNodes);
        initializeDistanceMatrix(originalColumnMajor, numNodes, 1, MAX_DISTANCE);
        packedRows.resize(numNodes * numNodes);
        packColumnMajorToRows(originalColumnMajor, packedRows, numNodes);
    }

    if (rank == 0) {
        std::printf("Initializing graph...\n");
        std::fflush(stdout);
    }
    MPI_Scatterv(rank == 0 ? packedRows.data() : nullptr,
                 partition.elementCounts.data(),
                 partition.elementDisplacements.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localElements), MPI_UNSIGNED, 0,
                 MPI_COMM_WORLD);
    if (rank == 0) {
        std::vector<unsigned int>().swap(packedRows);
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* deviceRowK = nullptr;
    const size_t deviceElements = std::max<size_t>(1, localElements);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceDist),
                         deviceElements * sizeof(unsigned int)),
              "cudaMalloc(deviceDist)", rank);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&devicePath),
                         deviceElements * sizeof(unsigned int)),
              "cudaMalloc(devicePath)", rank);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceRowK),
                         std::max<size_t>(1, numNodes) * sizeof(unsigned int)),
              "cudaMalloc(deviceRowK)", rank);
    if (localElements != 0) {
        cudaCheck(cudaMemcpy(deviceDist, localDist.data(),
                             localElements * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(localDist)", rank);
        cudaCheck(cudaMemcpy(devicePath, localPath.data(),
                             localElements * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(localPath)", rank);
    }

    std::vector<unsigned int> rowK[2] = {
        std::vector<unsigned int>(numNodes), std::vector<unsigned int>(numNodes)};
    cudaStream_t stream = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags", rank);

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
        std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    const dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y, 1);
    const dim3 grid(static_cast<unsigned int>(
                        (numNodes + CUDA_BLOCK_X - 1) / CUDA_BLOCK_X),
                    static_cast<unsigned int>((partition.localRows + CUDA_BLOCK_Y - 1) /
                                               CUDA_BLOCK_Y),
                    1);

    if (rank == 0) {
        std::copy(localDist.begin(), localDist.begin() + numNodes,
                  rowK[0].begin());
    }

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwners[k];
        const int currentBuffer = static_cast<int>(k & 1U);
        MPI_Bcast(rowK[currentBuffer].data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner,
                  MPI_COMM_WORLD);
        cudaCheck(cudaMemcpyAsync(deviceRowK, rowK[currentBuffer].data(),
                                  numNodes * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(rowK)", rank);
        if (partition.localRows != 0) {
            floydWarshallKernel<<<grid, block, 0, stream>>>(
                deviceDist, devicePath, deviceRowK, partition.localRows, numNodes, k);
            cudaCheck(cudaGetLastError(), "floydWarshallKernel launch", rank);
        }

        // Only the next row owner needs a device-to-host transfer.  This
        // keeps the host copy synchronized for the next MPI broadcast while
        // avoiding an O(n^2) transfer after every iteration.
        if (k + 1 < numNodes) {
            const int nextOwner = rowOwners[k + 1];
            if (rank == nextOwner) {
                const size_t localNext = (k + 1) - partition.firstRow;
                const int nextBuffer = static_cast<int>((k + 1) & 1U);
                cudaCheck(cudaMemcpyAsync(rowK[nextBuffer].data(),
                                          deviceDist + localNext * numNodes,
                                          numNodes * sizeof(unsigned int),
                                          cudaMemcpyDeviceToHost, stream),
                          "cudaMemcpyAsync(next row)", rank);
                cudaCheck(cudaStreamSynchronize(stream),
                          "cudaStreamSynchronize(next row)", rank);
            }
        }
    }

    cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize(final)", rank);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long long localMilliseconds = std::max<long long>(1, duration.count());
    long long globalMilliseconds = 0;
    MPI_Reduce(&localMilliseconds, &globalMilliseconds, 1, MPI_LONG_LONG_INT,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %lld ms\n", globalMilliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / (static_cast<double>(globalMilliseconds) / 1000.0) / 1e9;
        std::printf("Performance: %.3f GOPS\n", gflops);
    }

    const bool needResultsOnRoot = printResults || validate;
    if (needResultsOnRoot && localElements != 0) {
        cudaCheck(cudaMemcpy(localDist.data(), deviceDist,
                             localElements * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(final distance matrix)", rank);
    }

    if (needResultsOnRoot) {
        if (rank == 0) {
            packedRows.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                    rank == 0 ? packedRows.data() : nullptr,
                    partition.elementCounts.data(),
                    partition.elementDisplacements.data(), MPI_UNSIGNED, 0,
                    MPI_COMM_WORLD);
    }

    bool valid = true;
    if (rank == 0 && needResultsOnRoot) {
        if (printResults) {
            std::vector<unsigned int> columnMajor;
            unpackRowsToColumnMajor(packedRows, columnMajor, numNodes);
            print_results_int(columnMajor, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            valid = validateRows(packedRows, numNodes);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy", rank);
    cudaCheck(cudaFree(deviceRowK), "cudaFree(deviceRowK)", rank);
    cudaCheck(cudaFree(devicePath), "cudaFree(devicePath)", rank);
    cudaCheck(cudaFree(deviceDist), "cudaFree(deviceDist)", rank);
    MPI_Finalize();
    return (rank == 0 && validate && !valid) ? 1 : 0;
}
