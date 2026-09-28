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

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t cudaCheckErr__ = (call);                                        \
        if (cudaCheckErr__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(cudaCheckErr__));                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
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

// Block decomposition of `total` items over `parts` participants, as balanced
// as possible (first `total % parts` participants receive one extra item).
struct BlockDecomp {
    std::vector<size_t> counts;
    std::vector<size_t> displs;
};

BlockDecomp computeBlockDecomp(const size_t total, const int parts) {
    BlockDecomp bd;
    bd.counts.resize(parts);
    bd.displs.resize(parts);
    const size_t base = total / static_cast<size_t>(parts);
    const size_t rem = total % static_cast<size_t>(parts);
    size_t offset = 0;
    for (int p = 0; p < parts; ++p) {
        const size_t count = base + (static_cast<size_t>(p) < rem ? 1 : 0);
        bd.counts[p] = count;
        bd.displs[p] = offset;
        offset += count;
    }
    return bd;
}

int findOwner(const BlockDecomp& bd, const size_t index, const int parts) {
    for (int p = 0; p < parts; ++p) {
        if (index >= bd.displs[p] && index < bd.displs[p] + bd.counts[p]) {
            return p;
        }
    }
    return parts - 1;
}

// One Floyd-Warshall relaxation step for a contiguous block of local rows,
// using the (already up to date) broadcast row `k`.
__global__ void fwUpdateKernel(unsigned int* __restrict__ localDist,
                                unsigned int* __restrict__ localPath,
                                const unsigned int* __restrict__ rowK,
                                const size_t numNodes,
                                const size_t localRows,
                                const unsigned int k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= localRows || j >= numNodes) {
        return;
    }

    const size_t base = i * numNodes;
    const unsigned int distIK = localDist[base + k];
    const unsigned int distKJ = rowK[j];
    const unsigned int newDist = distIK + distKJ;

    if (newDist < localDist[base + j]) {
        localDist[base + j] = newDist;
        localPath[base + j] = k;
    }
}

// Distributed Floyd-Warshall: rows are block-partitioned across MPI ranks,
// then further block-partitioned across the GPUs local to each rank. Each
// GPU is driven by its own OpenMP host thread so that all local GPUs stay
// concurrently busy. Only the broadcast row `k` (owned by whichever
// rank/GPU currently holds row k) needs to travel over MPI each iteration.
double floydWarshallDistributed(std::vector<unsigned int>& fullDist,
                                 std::vector<unsigned int>& fullPath,
                                 const size_t numNodes,
                                 const int rank,
                                 const int numRanks) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const BlockDecomp mpiBlocks = computeBlockDecomp(numNodes, numRanks);
    const size_t myRows = mpiBlocks.counts[rank];
    const size_t myRowStart = mpiBlocks.displs[rank];

    std::vector<int> sendCounts(numRanks), sendDispls(numRanks);
    for (int p = 0; p < numRanks; ++p) {
        sendCounts[p] = static_cast<int>(mpiBlocks.counts[p] * numNodes);
        sendDispls[p] = static_cast<int>(mpiBlocks.displs[p] * numNodes);
    }

    std::vector<unsigned int> localDistHost(myRows * numNodes);
    std::vector<unsigned int> localPathHost(myRows * numNodes);

    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_UNSIGNED, localDistHost.data(), static_cast<int>(myRows * numNodes),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullPath.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_UNSIGNED, localPathHost.data(), static_cast<int>(myRows * numNodes),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Sub-partition this rank's row block across its local GPUs.
    const BlockDecomp gpuBlocks = computeBlockDecomp(myRows, deviceCount);

    std::vector<unsigned int*> dDist(deviceCount, nullptr);
    std::vector<unsigned int*> dPath(deviceCount, nullptr);
    std::vector<unsigned int*> dRowK(deviceCount, nullptr);
    std::vector<cudaStream_t> streams(deviceCount);

    for (int g = 0; g < deviceCount; ++g) {
        const size_t rows = gpuBlocks.counts[g];
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreate(&streams[g]));
        CUDA_CHECK(cudaMalloc(&dRowK[g], numNodes * sizeof(unsigned int)));
        if (rows > 0) {
            CUDA_CHECK(cudaMalloc(&dDist[g], rows * numNodes * sizeof(unsigned int)));
            CUDA_CHECK(cudaMalloc(&dPath[g], rows * numNodes * sizeof(unsigned int)));
            CUDA_CHECK(cudaMemcpyAsync(dDist[g], localDistHost.data() + gpuBlocks.displs[g] * numNodes,
                                        rows * numNodes * sizeof(unsigned int),
                                        cudaMemcpyHostToDevice, streams[g]));
            CUDA_CHECK(cudaMemcpyAsync(dPath[g], localPathHost.data() + gpuBlocks.displs[g] * numNodes,
                                        rows * numNodes * sizeof(unsigned int),
                                        cudaMemcpyHostToDevice, streams[g]));
        }
    }
    for (int g = 0; g < deviceCount; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }

    std::vector<unsigned int> hostRowK(numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (size_t k = 0; k < numNodes; ++k) {
        const int ownerRank = findOwner(mpiBlocks, k, numRanks);

        if (rank == ownerRank) {
            const size_t localK = k - myRowStart;
            const int ownerGpu = findOwner(gpuBlocks, localK, deviceCount);
            const size_t gpuLocalK = localK - gpuBlocks.displs[ownerGpu];
            CUDA_CHECK(cudaSetDevice(ownerGpu));
            CUDA_CHECK(cudaMemcpy(hostRowK.data(), dDist[ownerGpu] + gpuLocalK * numNodes,
                                   numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(hostRowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, ownerRank, MPI_COMM_WORLD);

        #pragma omp parallel num_threads(deviceCount)
        {
            const int g = omp_get_thread_num();
            const size_t rows = gpuBlocks.counts[g];
            if (rows > 0) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMemcpyAsync(dRowK[g], hostRowK.data(), numNodes * sizeof(unsigned int),
                                            cudaMemcpyHostToDevice, streams[g]));

                const dim3 block(32, 8);
                const dim3 grid(static_cast<unsigned int>((numNodes + block.x - 1) / block.x),
                                 static_cast<unsigned int>((rows + block.y - 1) / block.y));
                fwUpdateKernel<<<grid, block, 0, streams[g]>>>(dDist[g], dPath[g], dRowK[g],
                                                                 numNodes, rows,
                                                                 static_cast<unsigned int>(k));
                CUDA_CHECK(cudaStreamSynchronize(streams[g]));
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();

    for (int g = 0; g < deviceCount; ++g) {
        const size_t rows = gpuBlocks.counts[g];
        if (rows > 0) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(localDistHost.data() + gpuBlocks.displs[g] * numNodes, dDist[g],
                                   rows * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
    }

    MPI_Gatherv(localDistHost.data(), static_cast<int>(myRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? fullDist.data() : nullptr, sendCounts.data(), sendDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    for (int g = 0; g < deviceCount; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        if (dDist[g]) CUDA_CHECK(cudaFree(dDist[g]));
        if (dPath[g]) CUDA_CHECK(cudaFree(dPath[g]));
        CUDA_CHECK(cudaFree(dRowK[g]));
        CUDA_CHECK(cudaStreamDestroy(streams[g]));
    }

    return tEnd - tStart;
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
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < iLimit; ++i) {
        for (size_t j = 0; j < jLimit; ++j) {
            if (failed) {
                continue;
            }
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        #pragma omp atomic write
                        failed = true;
                    }
                }
            }
        }
    }
    if (failed) {
        return false;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int showHelp = 0;
    int parseError = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseError = 1;
            }
        }
        if (showHelp || parseError) {
            printUsage(argv[0]);
        }
    }

    struct {
        size_t numNodes;
        int validate;
        int printResults;
        int showHelp;
        int parseError;
    } cfg{numNodes, validate ? 1 : 0, printResults ? 1 : 0, showHelp, parseError};

    MPI_Bcast(&cfg, sizeof(cfg), MPI_BYTE, 0, MPI_COMM_WORLD);
    numNodes = cfg.numNodes;
    validate = cfg.validate != 0;
    printResults = cfg.printResults != 0;

    if (cfg.showHelp) {
        MPI_Finalize();
        return 0;
    }
    if (cfg.parseError) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        int deviceCountReport = 0;
        cudaGetDeviceCount(&deviceCountReport);
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Full matrices are only materialized on rank 0 (used for initialization,
    // scatter source, gather destination, printing, and validation).
    std::vector<unsigned int> fullDist;
    std::vector<unsigned int> fullPath;

    if (rank == 0) {
        printf("Initializing graph...\n");
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    const double elapsedSeconds = floydWarshallDistributed(fullDist, fullPath, numNodes, rank, numRanks);

    int exitCode = 0;

    if (rank == 0) {
        const long ms = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate operations per second
        // Floyd-Warshall has O(n^3) complexity
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / elapsedSeconds / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
