#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t cudaCheckErr_ = (call);                                       \
        if (cudaCheckErr_ != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(cudaCheckErr_));                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Row-block distribution of the global N x N matrix (row-major, row i occupies
// the contiguous range [i*n, (i+1)*n) ) across MPI ranks. Rows are split as
// evenly as possible, with the first `remainder` ranks getting one extra row.
struct RowRange {
    size_t start;
    size_t count;
};

RowRange rowRangeForRank(const size_t numNodes, const int rank, const int numRanks) {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    const size_t bigBlockRows = rem * (base + 1);

    if (static_cast<size_t>(rank) < rem) {
        const size_t count = base + 1;
        return {static_cast<size_t>(rank) * count, count};
    }
    return {bigBlockRows + (static_cast<size_t>(rank) - rem) * base, base};
}

int ownerOfRow(const size_t k, const size_t numNodes, const int numRanks) {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    const size_t bigBlockRows = rem * (base + 1);

    if (k < bigBlockRows) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(rem + (k - bigBlockRows) / base);
}

// Generates this rank's row-block of the distance matrix so that the values
// are bit-for-bit identical to the values the original sequential
// rand_r()-based generator would have produced at the same flat indices.
void initializeDistanceMatrixLocal(std::vector<unsigned int>& localDist, const size_t numNodes,
                                    const size_t rowStart, const size_t localRows,
                                    const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t skip = rowStart * numNodes;
    for (size_t p = 0; p < skip; ++p) {
        rand_r(&seed);
    }

    const size_t total = localRows * numNodes;
    for (size_t p = 0; p < total; ++p) {
        localDist[p] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    for (size_t r = 0; r < localRows; ++r) {
        const size_t globalRow = rowStart + r;
        localDist[r * numNodes + globalRow] = 0;
    }
}

void initializePathMatrixLocal(std::vector<unsigned int>& localPath, const size_t numNodes,
                                const size_t rowStart, const size_t localRows) {
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < localRows; ++r) {
        const unsigned int globalRow = static_cast<unsigned int>(rowStart + r);
        for (size_t c = 0; c < numNodes; ++c) {
            localPath[r * numNodes + c] = globalRow;
        }
    }
}

__global__ void fwUpdateKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ rowK, const size_t localRows,
                                const size_t n, const size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= localRows || j >= n) return;

    const unsigned int distIK = dist[i * n + k];
    const unsigned int distKJ = rowK[j];
    const unsigned int newDist = distIK + distKJ;

    if (newDist < dist[i * n + j]) {
        dist[i * n + j] = newDist;
        path[i * n + j] = static_cast<unsigned int>(k);
    }
}

// Hybrid MPI + OpenMP + CUDA Floyd-Warshall.
//
// The distance/path matrices are row-block distributed across MPI ranks.
// For each intermediate node k, the owning rank stages its row k out of GPU
// memory, that row is broadcast to every rank via MPI, and every rank then
// launches a CUDA kernel that updates its local row-block in parallel across
// GPU threads. OpenMP is used for the host-side setup work that surrounds
// the GPU-resident inner loop.
void floydWarshallHybrid(std::vector<unsigned int>& localDist, std::vector<unsigned int>& localPath,
                          const size_t numNodes, const size_t rowStart, const size_t localRows,
                          const int rank, const int numRanks) {
    const size_t localElems = localRows * numNodes;

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_rowK = nullptr;

    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_dist, localDist.data(), localElems * sizeof(unsigned int),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, localPath.data(), localElems * sizeof(unsigned int),
                               cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_rowK, numNodes * sizeof(unsigned int)));

    unsigned int* hostRowK = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostRowK, numNodes * sizeof(unsigned int)));

    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((numNodes + block.x - 1) / block.x),
                     static_cast<unsigned int>((localRows + block.y - 1) / block.y));

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, numNodes, numRanks);

        if (rank == owner) {
            const size_t localK = k - rowStart;
            CUDA_CHECK(cudaMemcpy(hostRowK, d_dist + localK * numNodes,
                                   numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(hostRowK, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_rowK, hostRowK, numNodes * sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

        if (localRows > 0) {
            fwUpdateKernel<<<grid, block>>>(d_dist, d_path, d_rowK, localRows, numNodes, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), d_dist, localElems * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data(), d_path, localElems * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_path));
    }
    CUDA_CHECK(cudaFree(d_rowK));
    CUDA_CHECK(cudaFreeHost(hostRowK));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks

    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[i * numNodes + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    bool valid = true;
    const size_t lim = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < lim; ++i) {
        for (size_t j = 0; j < lim; ++j) {
            const unsigned int distIJ = dist[i * numNodes + j];
            int failCount = 0;
            size_t failK = 0;
            #pragma omp parallel for reduction(+:failCount) schedule(static)
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = dist[k * numNodes + j];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        failCount++;
                        failK = k;
                    }
                }
            }
            if (failCount > 0) {
                printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, failK);
                valid = false;
            }
        }
    }

    return valid;
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    // Map this rank to a local GPU.
    int localRank = rank;
    const char* localRankEnv = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!localRankEnv) localRankEnv = getenv("MPI_LOCALRANKID");
    if (!localRankEnv) localRankEnv = getenv("SLURM_LOCALID");
    if (!localRankEnv) localRankEnv = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (localRankEnv) localRank = atoi(localRankEnv);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", numRanks);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const RowRange myRows = rowRangeForRank(numNodes, rank, numRanks);

    // Allocate this rank's row-block of the matrices
    std::vector<unsigned int> localDist(myRows.count * numNodes);
    std::vector<unsigned int> localPath(myRows.count * numNodes);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrixLocal(localDist, numNodes, myRows.start, myRows.count, 1, MAX_DISTANCE);
    initializePathMatrixLocal(localPath, numNodes, myRows.start, myRows.count);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallHybrid(localDist, localPath, numNodes, myRows.start, myRows.count, rank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Gather the full distance matrix on rank 0 for validation / printing.
        std::vector<int> recvCounts;
        std::vector<int> displs;
        if (rank == 0) {
            recvCounts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                const RowRange rr = rowRangeForRank(numNodes, r, numRanks);
                recvCounts[r] = static_cast<int>(rr.count * numNodes);
                displs[r] = static_cast<int>(rr.start * numNodes);
            }
        }

        std::vector<unsigned int> fullDist;
        if (rank == 0) fullDist.resize(numNodes * numNodes);

        MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        if (validate) {
            int validInt = 1;
            if (rank == 0) {
                printf("Validating result...\n");
                const bool valid = validateResult(fullDist, numNodes);
                validInt = valid ? 1 : 0;
                printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            }
            MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
            exitCode = validInt ? 0 : 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
