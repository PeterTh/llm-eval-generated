#include <algorithm>
#include <chrono>
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

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t cudaCheckErr__ = (call);                                         \
        if (cudaCheckErr__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(cudaCheckErr__));                             \
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
    // Split into two disjoint (per-outer-index) parallel sweeps so that each
    // OpenMP iteration only ever touches memory no other iteration touches:
    // sweep 1 owns entire rows, sweep 2 owns entire columns. This reproduces
    // the original sequential result exactly while staying race-free.
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// CUDA kernel: for a fixed intermediate node k, relax all (localRow, j) pairs
// owned by this rank's row-block against the broadcast pivot row k.
__global__ void fwKernel(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const unsigned int* __restrict__ pivotRow,
                          const size_t k, const size_t localRows, const size_t n) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t li = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (li >= localRows || j >= n) {
        return;
    }

    const size_t base = li * n;
    const unsigned int distIK = dist[base + k];
    const unsigned int distKJ = pivotRow[j];
    const unsigned int newDist = distIK + distKJ;

    if (newDist < dist[base + j]) {
        dist[base + j] = newDist;
        path[base + j] = static_cast<unsigned int>(k);
    }
}

// Computes a balanced contiguous row partition of numNodes rows across `size`
// MPI ranks. starts[r] is the first global row owned by rank r; starts[size]
// == numNodes.
void computeRowPartition(const size_t numNodes, const int size, std::vector<size_t>& starts) {
    starts.resize(static_cast<size_t>(size) + 1);
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        starts[r] = offset;
        offset += base + (static_cast<size_t>(r) < rem ? 1 : 0);
    }
    starts[size] = numNodes;
}

int ownerOfRow(const std::vector<size_t>& starts, const size_t row, const int size) {
    int lo = 0;
    int hi = size - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (starts[mid] <= row) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// Hybrid MPI + OpenMP + CUDA Floyd-Warshall.
//
// Row-blocks of the distance/path matrices are distributed across MPI ranks
// (one rank per accelerator, node-local rank selects the local GPU). Each
// iteration k broadcasts the pivot row (owned by whichever rank holds global
// row k) to every rank, which then relaxes its local row-block against that
// pivot row entirely on its GPU. This is the classic row-broadcast
// parallelization of Floyd-Warshall, with the O(n) work per row and O(n^2)
// per-rank relaxation offloaded to CUDA.
void floydWarshallDistributed(std::vector<unsigned int>& dist,
                              std::vector<unsigned int>& path,
                              const size_t numNodes,
                              const int rank,
                              const int size) {
    // Bind this rank to a local GPU based on its node-local rank, so that
    // multiple ranks sharing a node spread across that node's accelerators.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    const int device = deviceCount > 0 ? (localRank % deviceCount) : 0;
    CUDA_CHECK(cudaSetDevice(device));

    std::vector<size_t> starts;
    computeRowPartition(numNodes, size, starts);
    const size_t rowStart = starts[rank];
    const size_t rowEnd = starts[static_cast<size_t>(rank) + 1];
    const size_t localRows = rowEnd - rowStart;
    const size_t localElems = localRows * numNodes;

    unsigned int* distDev = nullptr;
    unsigned int* pathDev = nullptr;
    unsigned int* pivotDev = nullptr;
    unsigned int* pivotHost = nullptr;

    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&distDev, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&pathDev, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(distDev, dist.data() + rowStart * numNodes,
                               localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pathDev, path.data() + rowStart * numNodes,
                               localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&pivotDev, numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(&pivotHost, numNodes * sizeof(unsigned int), cudaHostAllocDefault));

    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((numNodes + block.x - 1) / block.x),
                     static_cast<unsigned int>((localRows + block.y - 1) / block.y));

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(starts, k, size);

        if (rank == owner) {
            const size_t localK = k - rowStart;
            CUDA_CHECK(cudaMemcpy(pivotHost, distDev + localK * numNodes,
                                   numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(pivotHost, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (localRows > 0) {
            CUDA_CHECK(cudaMemcpy(pivotDev, pivotHost, numNodes * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice));
            fwKernel<<<grid, block>>>(distDev, pathDev, pivotDev, k, localRows, numNodes);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data() + rowStart * numNodes, distDev,
                               localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    std::vector<int> recvCounts(static_cast<size_t>(size));
    std::vector<int> displs(static_cast<size_t>(size));
    for (int r = 0; r < size; ++r) {
        recvCounts[r] = static_cast<int>((starts[r + 1] - starts[r]) * numNodes);
        displs[r] = static_cast<int>(starts[r] * numNodes);
    }

    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_UNSIGNED, dist.data(), recvCounts.data(), displs.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(dist.data() + rowStart * numNodes, static_cast<int>(localElems), MPI_UNSIGNED,
                    nullptr, nullptr, nullptr, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFreeHost(pivotHost));
    CUDA_CHECK(cudaFree(pivotDev));
    if (distDev) CUDA_CHECK(cudaFree(distDev));
    if (pathDev) CUDA_CHECK(cudaFree(pathDev));
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
    const size_t iLimit = std::min(numNodes, static_cast<size_t>(10));
    const size_t jLimit = std::min(numNodes, static_cast<size_t>(10));
    bool valid = true;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < iLimit; ++i) {
        for (size_t j = 0; j < jLimit; ++j) {
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
                            valid = false;
                        }
                    }
                }
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("MPI ranks: %d\n", size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (every rank builds the full matrices identically and
    // deterministically so that no initial-data communication is required;
    // each rank then only computes and keeps its own row-block on its GPU)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallDistributed(dist, path, numNodes, rank, size);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int result = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
                result = 0;
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
