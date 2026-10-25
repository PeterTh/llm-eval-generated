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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t cudaCheckErr_ = (call);                                              \
        if (cudaCheckErr_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(cudaCheckErr_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
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

// GPU kernel performing the Floyd-Warshall relaxation for one pivot `k` across
// a contiguous block of locally-owned rows. Row/column `k` are invariant
// during this pass (D[k][k] == 0), so all (i,j) updates for a fixed k are
// independent and safe to parallelize arbitrarily.
__global__ void fwUpdateKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ pivotRow, size_t localRows,
                                size_t n, size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (i < localRows && j < n) {
        const size_t idx = i * n + j;
        const unsigned int distIK = dist[i * n + k];
        const unsigned int distKJ = pivotRow[j];
        const unsigned int newDist = distIK + distKJ;

        if (newDist < dist[idx]) {
            dist[idx] = newDist;
            path[idx] = static_cast<unsigned int>(k);
        }
    }
}

// Split `total` contiguous items as evenly as possible across `parts` parts.
void computeBlockDistribution(size_t total, size_t parts, std::vector<size_t>& start,
                               std::vector<size_t>& count) {
    start.assign(parts, 0);
    count.assign(parts, 0);
    const size_t base = total / parts;
    const size_t rem = total % parts;
    size_t offset = 0;
    for (size_t p = 0; p < parts; ++p) {
        count[p] = base + (p < rem ? 1 : 0);
        start[p] = offset;
        offset += count[p];
    }
}

// Locate the owner of a given global index within a contiguous block distribution.
int findOwner(const std::vector<size_t>& start, const std::vector<size_t>& count, size_t index) {
    for (size_t p = 0; p < start.size(); ++p) {
        if (count[p] > 0 && index >= start[p] && index < start[p] + count[p]) {
            return static_cast<int>(p);
        }
    }
    return static_cast<int>(start.size() - 1);
}

// Hybrid MPI + OpenMP + CUDA Floyd-Warshall.
//
// Rows of the distance/path matrices are block-distributed across MPI ranks.
// Within a rank, the local row block is further split across the CUDA
// devices owned by that rank (assigned via node-local MPI topology), with an
// OpenMP thread per device driving its independent CUDA stream. For each
// pivot k, the owning rank/device extracts row k, it is broadcast to every
// rank via MPI, and every device updates its local rows in parallel on the
// GPU.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const size_t n = numNodes;
    if (n == 0) {
        return;
    }

    // Row distribution across MPI ranks.
    std::vector<size_t> rankRowStart, rankRowCount;
    computeBlockDistribution(n, static_cast<size_t>(worldSize), rankRowStart, rankRowCount);
    const size_t myRowStart = rankRowStart[worldRank];
    const size_t myRowCount = rankRowCount[worldRank];

    std::vector<int> sendCounts(worldSize), sendDispls(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        sendCounts[r] = static_cast<int>(rankRowCount[r] * n);
        sendDispls[r] = static_cast<int>(rankRowStart[r] * n);
    }

    std::vector<unsigned int> localDist(myRowCount * n);
    std::vector<unsigned int> localPath(myRowCount * n);

    MPI_Scatterv(worldRank == 0 ? dist.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_UNSIGNED, localDist.data(), static_cast<int>(myRowCount * n), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(worldRank == 0 ? path.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_UNSIGNED, localPath.data(), static_cast<int>(myRowCount * n), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Discover the CUDA devices visible on this node and partition them
    // across the MPI ranks that share the node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    int deviceCountOnNode = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCountOnNode));
    if (deviceCountOnNode <= 0) {
        fprintf(stderr, "Rank %d: no CUDA-capable devices found\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> myDevices;
    if (static_cast<size_t>(nodeSize) <= static_cast<size_t>(deviceCountOnNode)) {
        std::vector<size_t> devStart, devCount;
        computeBlockDistribution(static_cast<size_t>(deviceCountOnNode),
                                  static_cast<size_t>(nodeSize), devStart, devCount);
        for (size_t d = 0; d < devCount[nodeRank]; ++d) {
            myDevices.push_back(static_cast<int>(devStart[nodeRank] + d));
        }
    } else {
        myDevices.push_back(nodeRank % deviceCountOnNode);
    }
    MPI_Comm_free(&nodeComm);

    const int numMyDevices = static_cast<int>(myDevices.size());

    // Partition this rank's row block across its assigned GPUs.
    std::vector<size_t> devRowStart, devRowCount;
    computeBlockDistribution(myRowCount, static_cast<size_t>(numMyDevices), devRowStart,
                              devRowCount);

    std::vector<unsigned int*> d_dist(numMyDevices, nullptr);
    std::vector<unsigned int*> d_path(numMyDevices, nullptr);
    std::vector<unsigned int*> d_pivotRow(numMyDevices, nullptr);
    std::vector<cudaStream_t> streams(numMyDevices);

    #pragma omp parallel for num_threads(numMyDevices) schedule(static)
    for (int d = 0; d < numMyDevices; ++d) {
        CUDA_CHECK(cudaSetDevice(myDevices[d]));
        CUDA_CHECK(cudaStreamCreate(&streams[d]));
        CUDA_CHECK(cudaMalloc(&d_pivotRow[d], n * sizeof(unsigned int)));

        const size_t rows = devRowCount[d];
        if (rows > 0) {
            CUDA_CHECK(cudaMalloc(&d_dist[d], rows * n * sizeof(unsigned int)));
            CUDA_CHECK(cudaMalloc(&d_path[d], rows * n * sizeof(unsigned int)));
            CUDA_CHECK(cudaMemcpyAsync(d_dist[d], localDist.data() + devRowStart[d] * n,
                                       rows * n * sizeof(unsigned int), cudaMemcpyHostToDevice,
                                       streams[d]));
            CUDA_CHECK(cudaMemcpyAsync(d_path[d], localPath.data() + devRowStart[d] * n,
                                       rows * n * sizeof(unsigned int), cudaMemcpyHostToDevice,
                                       streams[d]));
            CUDA_CHECK(cudaStreamSynchronize(streams[d]));
        }
    }

    std::vector<unsigned int> pivotRowHost(n);
    const dim3 blockDim(32, 8);

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t k = 0; k < n; ++k) {
        const int ownerRank = findOwner(rankRowStart, rankRowCount, k);

        if (worldRank == ownerRank) {
            const size_t localRowIdx = k - myRowStart;
            const int ownerDevice = findOwner(devRowStart, devRowCount, localRowIdx);
            const size_t deviceLocalRow = localRowIdx - devRowStart[ownerDevice];
            CUDA_CHECK(cudaSetDevice(myDevices[ownerDevice]));
            CUDA_CHECK(cudaMemcpy(pivotRowHost.data(), d_dist[ownerDevice] + deviceLocalRow * n,
                                   n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(pivotRowHost.data(), static_cast<int>(n), MPI_UNSIGNED, ownerRank,
                  MPI_COMM_WORLD);

        #pragma omp parallel for num_threads(numMyDevices) schedule(static)
        for (int d = 0; d < numMyDevices; ++d) {
            const size_t rows = devRowCount[d];
            if (rows == 0) continue;

            CUDA_CHECK(cudaSetDevice(myDevices[d]));
            CUDA_CHECK(cudaMemcpyAsync(d_pivotRow[d], pivotRowHost.data(),
                                       n * sizeof(unsigned int), cudaMemcpyHostToDevice,
                                       streams[d]));

            const dim3 gridDim(static_cast<unsigned int>((n + blockDim.x - 1) / blockDim.x),
                               static_cast<unsigned int>((rows + blockDim.y - 1) / blockDim.y));
            fwUpdateKernel<<<gridDim, blockDim, 0, streams[d]>>>(d_dist[d], d_path[d],
                                                                  d_pivotRow[d], rows, n, k);
            CUDA_CHECK(cudaStreamSynchronize(streams[d]));
        }
    }

    #pragma omp parallel for num_threads(numMyDevices) schedule(static)
    for (int d = 0; d < numMyDevices; ++d) {
        const size_t rows = devRowCount[d];
        if (rows == 0) continue;

        CUDA_CHECK(cudaSetDevice(myDevices[d]));
        CUDA_CHECK(cudaMemcpy(localDist.data() + devRowStart[d] * n, d_dist[d],
                              rows * n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data() + devRowStart[d] * n, d_path[d],
                              rows * n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    for (int d = 0; d < numMyDevices; ++d) {
        CUDA_CHECK(cudaSetDevice(myDevices[d]));
        if (d_dist[d]) CUDA_CHECK(cudaFree(d_dist[d]));
        if (d_path[d]) CUDA_CHECK(cudaFree(d_path[d]));
        CUDA_CHECK(cudaFree(d_pivotRow[d]));
        CUDA_CHECK(cudaStreamDestroy(streams[d]));
    }

    MPI_Gatherv(localDist.data(), static_cast<int>(myRowCount * n), MPI_UNSIGNED,
                worldRank == 0 ? dist.data() : nullptr, sendCounts.data(), sendDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), static_cast<int>(myRowCount * n), MPI_UNSIGNED,
                worldRank == 0 ? path.data() : nullptr, sendCounts.data(), sendDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (worldRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (only rank 0 holds the full matrices; other ranks
    // only ever operate on their scattered row blocks).
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (worldRank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);

        // Initialize
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Run Floyd-Warshall
    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes);

    auto end = std::chrono::high_resolution_clock::now();

    int exitCode = 0;

    if (worldRank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

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
