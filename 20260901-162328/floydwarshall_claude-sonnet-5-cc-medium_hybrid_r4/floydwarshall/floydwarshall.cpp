#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t cudaCheckErr_ = (call);                                          \
        if (cudaCheckErr_ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(cudaCheckErr_));                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                            \
        }                                                                            \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Relaxes a contiguous block of rows [0, rowCount) of the local distance/path
// matrices against the broadcast pivot row k. Each thread covers one row*col
// work item via a grid-stride loop so arbitrarily large blocks are supported.
__global__ void relaxRowsKernel(unsigned int* dist, unsigned int* path,
                                 const unsigned int* pivotRow,
                                 unsigned long long totalElems,
                                 size_t n, size_t k) {
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (; idx < totalElems; idx += stride) {
        const size_t li = static_cast<size_t>(idx / n);
        const size_t j = static_cast<size_t>(idx % n);

        unsigned int* row = dist + li * n;
        const unsigned int distIJ = row[j];
        const unsigned int distIK = row[k];
        const unsigned int distKJ = pivotRow[j];

        const unsigned int newDist = distIK + distKJ;
        if (newDist < distIJ) {
            row[j] = newDist;
            path[li * n + j] = static_cast<unsigned int>(k);
        }
    }
}

// Describes a contiguous block assigned to one worker in a 1D distribution.
struct Block {
    size_t count = 0;
    size_t offset = 0;
};

// Splits `total` contiguous items as evenly as possible across `workers`.
std::vector<Block> distribute(size_t total, int workers) {
    std::vector<Block> blocks(std::max(workers, 0));
    if (workers <= 0) return blocks;

    const size_t base = total / static_cast<size_t>(workers);
    const size_t rem = total % static_cast<size_t>(workers);
    size_t offset = 0;
    for (int w = 0; w < workers; ++w) {
        const size_t count = base + (static_cast<size_t>(w) < rem ? 1 : 0);
        blocks[w] = {count, offset};
        offset += count;
    }
    return blocks;
}

// Returns the index of the block owning global index `idx`.
int ownerOf(size_t idx, const std::vector<Block>& blocks) {
    for (int w = static_cast<int>(blocks.size()) - 1; w >= 0; --w) {
        if (idx >= blocks[w].offset) return w;
    }
    return 0;
}

void floydWarshallHybrid(std::vector<unsigned int>& localDist,
                          std::vector<unsigned int>& localPath,
                          const size_t numNodes,
                          const int rank,
                          const std::vector<Block>& rankBlocks,
                          const std::vector<int>& deviceIds) {
    const size_t rowStart = rankBlocks[rank].offset;
    const size_t localRows = rankBlocks[rank].count;
    const int numDevices = static_cast<int>(deviceIds.size());

    // Further split this rank's row block across the GPUs assigned to it;
    // one OpenMP thread drives each GPU.
    const std::vector<Block> gpuBlocks = distribute(localRows, numDevices);

    std::vector<unsigned int*> devDist(numDevices, nullptr);
    std::vector<unsigned int*> devPath(numDevices, nullptr);
    std::vector<unsigned int*> devPivot(numDevices, nullptr);
    std::vector<cudaStream_t> streams(numDevices);
    std::vector<unsigned int> hostPivot(numNodes);

    constexpr int blockSize = 256;
    constexpr unsigned long long maxBlocks = 1ULL << 20;

    #pragma omp parallel num_threads(numDevices)
    {
        const int g = omp_get_thread_num();
        const size_t rowsG = gpuBlocks[g].count;
        const size_t offG = gpuBlocks[g].offset;

        CUDA_CHECK(cudaSetDevice(deviceIds[g]));
        CUDA_CHECK(cudaStreamCreate(&streams[g]));
        CUDA_CHECK(cudaMalloc(&devPivot[g], numNodes * sizeof(unsigned int)));

        if (rowsG > 0) {
            CUDA_CHECK(cudaMalloc(&devDist[g], rowsG * numNodes * sizeof(unsigned int)));
            CUDA_CHECK(cudaMalloc(&devPath[g], rowsG * numNodes * sizeof(unsigned int)));
            CUDA_CHECK(cudaMemcpyAsync(devDist[g], localDist.data() + offG * numNodes,
                                        rowsG * numNodes * sizeof(unsigned int),
                                        cudaMemcpyHostToDevice, streams[g]));
            CUDA_CHECK(cudaMemcpyAsync(devPath[g], localPath.data() + offG * numNodes,
                                        rowsG * numNodes * sizeof(unsigned int),
                                        cudaMemcpyHostToDevice, streams[g]));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));
        }

        // Wait for every thread's device buffers to be ready before the
        // pivot-extraction logic below starts reading from them.
        #pragma omp barrier

        for (size_t k = 0; k < numNodes; ++k) {
            #pragma omp single
            {
                const int ownerRank = ownerOf(k, rankBlocks);
                if (ownerRank == rank) {
                    const size_t localK = k - rowStart;
                    const int g0 = ownerOf(localK, gpuBlocks);
                    const size_t subK = localK - gpuBlocks[g0].offset;
                    CUDA_CHECK(cudaSetDevice(deviceIds[g0]));
                    CUDA_CHECK(cudaMemcpy(hostPivot.data(), devDist[g0] + subK * numNodes,
                                          numNodes * sizeof(unsigned int),
                                          cudaMemcpyDeviceToHost));
                }
                MPI_Bcast(hostPivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                          ownerRank, MPI_COMM_WORLD);
            }

            // The single-executing thread may have changed the CUDA context
            // to another device to extract the pivot row; restore our own.
            CUDA_CHECK(cudaSetDevice(deviceIds[g]));

            if (rowsG > 0) {
                CUDA_CHECK(cudaMemcpyAsync(devPivot[g], hostPivot.data(),
                                            numNodes * sizeof(unsigned int),
                                            cudaMemcpyHostToDevice, streams[g]));

                const unsigned long long totalElems =
                    static_cast<unsigned long long>(rowsG) * numNodes;
                const unsigned long long gridSize64 =
                    std::min((totalElems + blockSize - 1) / blockSize, maxBlocks);
                const int gridSize = static_cast<int>(std::max(gridSize64, 1ULL));

                relaxRowsKernel<<<gridSize, blockSize, 0, streams[g]>>>(
                    devDist[g], devPath[g], devPivot[g], totalElems, numNodes, k);
                CUDA_CHECK(cudaStreamSynchronize(streams[g]));
            }

            // Ensure every thread's kernel for this k has completed before
            // any thread proceeds to extract the next pivot row from device
            // memory that may still be in flight otherwise.
            #pragma omp barrier
        }

        if (rowsG > 0) {
            CUDA_CHECK(cudaMemcpyAsync(localDist.data() + offG * numNodes, devDist[g],
                                        rowsG * numNodes * sizeof(unsigned int),
                                        cudaMemcpyDeviceToHost, streams[g]));
            CUDA_CHECK(cudaMemcpyAsync(localPath.data() + offG * numNodes, devPath[g],
                                        rowsG * numNodes * sizeof(unsigned int),
                                        cudaMemcpyDeviceToHost, streams[g]));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));
            CUDA_CHECK(cudaFree(devDist[g]));
            CUDA_CHECK(cudaFree(devPath[g]));
        }
        CUDA_CHECK(cudaFree(devPivot[g]));
        CUDA_CHECK(cudaStreamDestroy(streams[g]));
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
    bool failed = false;
    const size_t iLimit = std::min(numNodes, static_cast<size_t>(10));
    const size_t jLimit = std::min(numNodes, static_cast<size_t>(10));

    #pragma omp parallel for collapse(2) schedule(dynamic) shared(failed)
    for (size_t i = 0; i < iLimit; ++i) {
        for (size_t j = 0; j < jLimit; ++j) {
            if (failed) continue;
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        #pragma omp critical
                        {
                            printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                   i, j, k);
                        }
                        failed = true;
                    }
                }
            }
        }
    }

    return !failed;
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

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    // Determine this rank's position within its node so GPUs on a node are
    // shared out across the ranks running on it.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int devicesForThisRank;
    int deviceStart;
    if (nodeSize <= deviceCount) {
        // Enough GPUs for every local rank to get at least one; hand out any
        // remainder so a rank may drive several GPUs via OpenMP threads.
        const int base = deviceCount / nodeSize;
        const int rem = deviceCount % nodeSize;
        devicesForThisRank = base + (nodeRank < rem ? 1 : 0);
        deviceStart = nodeRank * base + std::min(nodeRank, rem);
    } else {
        // More ranks than GPUs on this node: fall back to round-robin sharing.
        devicesForThisRank = 1;
        deviceStart = nodeRank % deviceCount;
    }
    std::vector<int> deviceIds(devicesForThisRank);
    for (int i = 0; i < devicesForThisRank; ++i) deviceIds[i] = deviceStart + i;

    // Distribute rows of the matrices across MPI ranks.
    const std::vector<Block> rankBlocks = distribute(numNodes, worldSize);
    const size_t localRows = rankBlocks[rank].count;

    std::vector<int> counts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = static_cast<int>(rankBlocks[r].count * numNodes);
        displs[r] = static_cast<int>(rankBlocks[r].offset * numNodes);
    }

    std::vector<unsigned int> dist, path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);

    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED, 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallHybrid(localDist, localPath, numNodes, rank, rankBlocks, deviceIds);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
    }
    MPI_Gatherv(localDist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? path.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxMs));

        // Calculate operations per second
        // Floyd-Warshall has O(n^3) complexity
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
