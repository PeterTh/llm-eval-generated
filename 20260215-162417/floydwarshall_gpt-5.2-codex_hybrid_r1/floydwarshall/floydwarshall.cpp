#include <algorithm>
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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t err = (call);                                                                \
        if (err != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__,       \
                    __LINE__);                                                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                        \
        }                                                                                        \
    } while (0)

struct Partition {
    size_t row_start = 0;
    size_t local_rows = 0;
    size_t base = 0;
    size_t rem = 0;
};

Partition computePartition(const size_t numNodes, const int rank, const int size) {
    Partition part;
    part.base = numNodes / static_cast<size_t>(size);
    part.rem = numNodes % static_cast<size_t>(size);
    part.local_rows = part.base + (rank < static_cast<int>(part.rem) ? 1 : 0);
    part.row_start = static_cast<size_t>(rank) * part.base + std::min(part.rem, static_cast<size_t>(rank));
    return part;
}

int ownerOfRow(const size_t row, const size_t base, const size_t rem) {
    const size_t cutoff = (base + 1) * rem;
    if (row < cutoff) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - cutoff) / base);
}

void buildCounts(const size_t numNodes,
                 const int worldSize,
                 std::vector<size_t>& rowCounts,
                 std::vector<size_t>& rowStarts,
                 std::vector<int>& counts,
                 std::vector<int>& displs) {
    rowCounts.resize(worldSize);
    rowStarts.resize(worldSize);
    counts.resize(worldSize);
    displs.resize(worldSize);

    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t rem = numNodes % static_cast<size_t>(worldSize);
    size_t running = 0;

    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = base + (r < static_cast<int>(rem) ? 1 : 0);
        rowCounts[r] = rows;
        rowStarts[r] = static_cast<size_t>(r) * base + std::min(rem, static_cast<size_t>(r));
        counts[r] = static_cast<int>(rows * numNodes);
        displs[r] = static_cast<int>(running);
        running += rows * numNodes;
    }
}

void packRows(const std::vector<unsigned int>& full,
              std::vector<unsigned int>& packed,
              const size_t numNodes,
              const std::vector<size_t>& rowStarts,
              const std::vector<size_t>& rowCounts,
              const std::vector<int>& displs) {
    const size_t worldSize = rowCounts.size();
    for (size_t r = 0; r < worldSize; ++r) {
        const size_t rows = rowCounts[r];
        if (rows == 0) {
            continue;
        }
        const size_t start = rowStarts[r];
        const size_t offset = static_cast<size_t>(displs[r]);
        #pragma omp parallel for
        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int* src = &full[idx2(start, j, numNodes)];
            unsigned int* dst = &packed[offset + j * rows];
            std::memcpy(dst, src, rows * sizeof(unsigned int));
        }
    }
}

void unpackRows(const std::vector<unsigned int>& packed,
                std::vector<unsigned int>& full,
                const size_t numNodes,
                const std::vector<size_t>& rowStarts,
                const std::vector<size_t>& rowCounts,
                const std::vector<int>& displs) {
    const size_t worldSize = rowCounts.size();
    for (size_t r = 0; r < worldSize; ++r) {
        const size_t rows = rowCounts[r];
        if (rows == 0) {
            continue;
        }
        const size_t start = rowStarts[r];
        const size_t offset = static_cast<size_t>(displs[r]);
        #pragma omp parallel for
        for (size_t j = 0; j < numNodes; ++j) {
            unsigned int* dst = &full[idx2(start, j, numNodes)];
            const unsigned int* src = &packed[offset + j * rows];
            std::memcpy(dst, src, rows * sizeof(unsigned int));
        }
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
        }
    }
}

__global__ void gatherRowKernel(const unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ row_k,
                                const unsigned int numNodes,
                                const unsigned int localRows,
                                const unsigned int local_k) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < numNodes) {
        row_k[j] = dist[j * localRows + local_k];
    }
}

__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int* __restrict__ row_k,
                                    const unsigned int numNodes,
                                    const unsigned int localRows,
                                    const unsigned int k) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < numNodes && i < localRows) {
        const unsigned int idx = j * localRows + i;
        const unsigned int distIJ = dist[idx];
        const unsigned int distIK = dist[k * localRows + i];
        const unsigned int distKJ = row_k[j];
        const unsigned int newDist = distIK + distKJ;
        if (newDist < distIJ) {
            dist[idx] = newDist;
            path[idx] = k;
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks

    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            break;
        }
    }

    if (showHelp || parseError) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    const Partition part = computePartition(numNodes, worldRank, worldSize);
    const size_t localRows = part.local_rows;
    const size_t localCount = localRows * numNodes;
    const size_t totalCount = numNodes * numNodes;

    int localSizeOk = (numNodes <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                       localCount <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                       totalCount <= static_cast<size_t>(std::numeric_limits<int>::max()))
                          ? 0
                          : 1;
    int anySizeBad = 0;
    MPI_Allreduce(&localSizeOk, &anySizeBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anySizeBad) {
        if (worldRank == 0) {
            printf("Problem size too large for MPI counts.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<size_t> rowCounts;
    std::vector<size_t> rowStarts;
    std::vector<int> counts;
    std::vector<int> displs;
    if (worldRank == 0) {
        buildCounts(numNodes, worldSize, rowCounts, rowStarts, counts, displs);
    }

    std::vector<unsigned int> distLocal(localCount);
    std::vector<unsigned int> pathLocal(localCount);

    std::vector<unsigned int> distFull;
    std::vector<unsigned int> pathFull;
    std::vector<unsigned int> packedDist;
    std::vector<unsigned int> packedPath;

    if (worldRank == 0) {
        distFull.resize(numNodes * numNodes);
        pathFull.resize(numNodes * numNodes);

        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", worldSize);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");

        initializeDistanceMatrix(distFull, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(pathFull, numNodes);

        packedDist.resize(numNodes * numNodes);
        packedPath.resize(numNodes * numNodes);
        packRows(distFull, packedDist, numNodes, rowStarts, rowCounts, displs);
        packRows(pathFull, packedPath, numNodes, rowStarts, rowCounts, displs);
    }

    MPI_Scatterv(worldRank == 0 ? packedDist.data() : nullptr,
                 worldRank == 0 ? counts.data() : nullptr,
                 worldRank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 distLocal.data(),
                 static_cast<int>(localCount),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(worldRank == 0 ? packedPath.data() : nullptr,
                 worldRank == 0 ? counts.data() : nullptr,
                 worldRank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 pathLocal.data(),
                 static_cast<int>(localCount),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    if (worldRank == 0) {
        packedDist.clear();
        packedPath.clear();
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (worldRank == 0) {
            printf("No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(worldRank % deviceCount));

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dRowK = nullptr;

    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(&dDist, localCount * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dPath, localCount * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(dDist, distLocal.data(), localCount * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, pathLocal.data(), localCount * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&dRowK, numNodes * sizeof(unsigned int)));
    std::vector<unsigned int> rowK(numNodes);

    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const dim3 gatherBlock(256);
    const dim3 gatherGrid((numNodes + gatherBlock.x - 1) / gatherBlock.x);
    const dim3 updateBlock(16, 16);
    const dim3 updateGrid((numNodes + updateBlock.x - 1) / updateBlock.x,
                          (localRows + updateBlock.y - 1) / updateBlock.y);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, part.base, part.rem);
        if (worldRank == owner) {
            const unsigned int localK = static_cast<unsigned int>(k - part.row_start);
            if (localCount > 0) {
                gatherRowKernel<<<gatherGrid, gatherBlock>>>(dDist, dRowK,
                                                             static_cast<unsigned int>(numNodes),
                                                             static_cast<unsigned int>(localRows),
                                                             localK);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(rowK.data(), dRowK, numNodes * sizeof(unsigned int),
                                      cudaMemcpyDeviceToHost));
            }
        }

        MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dRowK, rowK.data(), numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        if (localCount > 0) {
            floydWarshallKernel<<<updateGrid, updateBlock>>>(dDist, dPath, dRowK,
                                                             static_cast<unsigned int>(numNodes),
                                                             static_cast<unsigned int>(localRows),
                                                             static_cast<unsigned int>(k));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpy(distLocal.data(), dDist, localCount * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pathLocal.data(), dPath, localCount * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    if (dDist) {
        CUDA_CHECK(cudaFree(dDist));
    }
    if (dPath) {
        CUDA_CHECK(cudaFree(dPath));
    }
    if (dRowK) {
        CUDA_CHECK(cudaFree(dRowK));
    }

    std::vector<unsigned int> packedResult;
    if (printResults || validate) {
        if (worldRank == 0) {
            if (rowCounts.empty()) {
                buildCounts(numNodes, worldSize, rowCounts, rowStarts, counts, displs);
            }
            distFull.assign(numNodes * numNodes, 0);
            packedResult.resize(numNodes * numNodes);
        }

        MPI_Gatherv(distLocal.data(),
                    static_cast<int>(localCount),
                    MPI_UNSIGNED,
                    worldRank == 0 ? packedResult.data() : nullptr,
                    worldRank == 0 ? counts.data() : nullptr,
                    worldRank == 0 ? displs.data() : nullptr,
                    MPI_UNSIGNED,
                    0,
                    MPI_COMM_WORLD);

        if (worldRank == 0) {
            unpackRows(packedResult, distFull, numNodes, rowStarts, rowCounts, displs);
        }
    }

    if (worldRank == 0) {
        const double timeMs = maxTime * 1000.0;
        printf("Computation time: %.3f ms\n", timeMs);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = maxTime > 0.0 ? ops / maxTime / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults && worldRank == 0) {
        print_results_int(distFull, "DistanceMatrix");
    }

    int exitCode = 0;
    if (validate && worldRank == 0) {
        printf("Validating result...\n");
        const bool valid = validateResult(distFull, numNodes);
        if (valid) {
            printf("Validation: PASSED\n");
            exitCode = 0;
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
