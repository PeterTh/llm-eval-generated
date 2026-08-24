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
constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;

// Index calculation for the original flattened, column-major matrix.
// A row owned by an MPI rank is contiguous because idx2(j, i, n) == i*n+j.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    // Keep the original deterministic rand_r sequence.  The root rank creates the
    // matrix once and MPI distributes its rows to the participating ranks.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0).
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t rowBegin,
                          const size_t localRows, const size_t numNodes) {
    // The original two symmetric writes produce path[row][column] == row.
    // A direct row-wise initialization avoids duplicate writes and exposes the
    // independent host work to OpenMP.
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < localRows; ++row) {
        for (size_t column = 0; column < numNodes; ++column) {
            path[idx2(column, row, numNodes)] =
                static_cast<unsigned int>(rowBegin + row);
        }
    }
}

// One CUDA block updates TILE_Y rows by TILE_X columns.  The pivot row and the
// pivot-column values are reused from shared memory by all threads in a tile.
// This is the GPU equivalent of one k step of Floyd-Warshall:
//     dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
// where each MPI rank stores a contiguous block of rows i.
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                    const unsigned int* pivotRow,
                                    const size_t localRows, const size_t numNodes,
                                    const size_t k) {
    __shared__ unsigned int pivotRowTile[TILE_X];
    __shared__ unsigned int pivotColumnTile[TILE_Y];

    const size_t row = static_cast<size_t>(blockIdx.y) * TILE_Y + threadIdx.y;
    const size_t column = static_cast<size_t>(blockIdx.x) * TILE_X + threadIdx.x;

    if (threadIdx.y == 0 && column < numNodes) {
        pivotRowTile[threadIdx.x] = pivotRow[column];
    }
    if (threadIdx.x == 0 && row < localRows) {
        pivotColumnTile[threadIdx.y] = dist[row * numNodes + k];
    }
    __syncthreads();

    if (row < localRows && column < numNodes) {
        const size_t element = row * numNodes + column;
        const unsigned int candidate = pivotColumnTile[threadIdx.y] +
                                       pivotRowTile[threadIdx.x];
        if (candidate < dist[element]) {
            dist[element] = candidate;
            path[element] = static_cast<unsigned int>(k);
        }
    }
}

void checkCuda(const cudaError_t error, const char* expression, const int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA failure in %s: %s\n", rank, expression,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, rank)

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks.

    // 1. Diagonal should be zero.
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j].
    // Check a sample of paths to avoid O(n^3) validation time.
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition.
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on every rank so that all ranks follow the
    // same control flow through MPI collectives.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
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

    if (numNodes > 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        if (rank == 0) {
            fprintf(stderr, "The matrix size overflows size_t\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const size_t matrixElements = numNodes * numNodes;
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            fprintf(stderr, "The number of nodes is too large for MPI's count interface\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::vector<size_t> rowStarts(static_cast<size_t>(worldSize) + 1);
    for (int process = 0; process <= worldSize; ++process) {
        rowStarts[static_cast<size_t>(process)] =
            (numNodes * static_cast<size_t>(process)) / static_cast<size_t>(worldSize);
    }
    const size_t rowBegin = rowStarts[static_cast<size_t>(rank)];
    const size_t rowEnd = rowStarts[static_cast<size_t>(rank) + 1];
    const size_t localRows = rowEnd - rowBegin;
    const size_t localElements = localRows * numNodes;

    std::vector<int> mpiCounts(static_cast<size_t>(worldSize));
    std::vector<int> mpiDisplacements(static_cast<size_t>(worldSize));
    for (int process = 0; process < worldSize; ++process) {
        const size_t begin = rowStarts[static_cast<size_t>(process)] * numNodes;
        const size_t end = rowStarts[static_cast<size_t>(process) + 1] * numNodes;
        if (end - begin > static_cast<size_t>(INT_MAX) || begin > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) {
                fprintf(stderr, "The matrix is too large for MPI's 32-bit count interface\n");
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        mpiCounts[static_cast<size_t>(process)] = static_cast<int>(end - begin);
        mpiDisplacements[static_cast<size_t>(process)] = static_cast<int>(begin);
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "MPI rank %d: no CUDA device is available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", worldSize);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA devices visible on this node: %d\n", deviceCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Only rank zero materializes the full input, retaining the exact original
    // random stream.  MPI then distributes contiguous source rows.
    std::vector<unsigned int> dist;
    if (rank == 0) {
        dist.resize(matrixElements);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    std::vector<unsigned int> localDist(localElements);
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, mpiCounts.data(), mpiDisplacements.data(),
                 MPI_UNSIGNED, localDist.data(), mpiCounts[static_cast<size_t>(rank)],
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::vector<unsigned int> localPath(localElements);
    initializePathMatrix(localPath, rowBegin, localRows, numNodes);

    // Keep one device-resident copy of the distributed matrix and path matrix.
    // A pinned pivot buffer makes the per-k host broadcast and host-to-device
    // transfer inexpensive while avoiding a CUDA-aware MPI dependency.
    const size_t deviceElements = std::max<size_t>(localElements, 1);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivotRow = nullptr;
    unsigned int* hostPivotRow = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist),
                          deviceElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath),
                          deviceElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePivotRow),
                          std::max<size_t>(numNodes, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostPivotRow),
                             std::max<size_t>(numNodes, 1) * sizeof(unsigned int),
                             cudaHostAllocPortable));

    if (localElements > 0) {
        CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data(),
                              localElements * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(),
                              localElements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    const dim3 block(TILE_X, TILE_Y, 1);
    const dim3 grid(static_cast<unsigned int>(std::max<size_t>(1, (numNodes + TILE_X - 1) / TILE_X)),
                    static_cast<unsigned int>(std::max<size_t>(1, (localRows + TILE_Y - 1) / TILE_Y)),
                    1);

    for (size_t k = 0; k < numNodes; ++k) {
        const auto ownerIterator = std::upper_bound(rowStarts.begin(), rowStarts.end() - 1, k);
        const int pivotOwner = static_cast<int>(ownerIterator - rowStarts.begin()) - 1;

        if (rank == pivotOwner) {
            const size_t localPivotRow = k - rowBegin;
            CUDA_CHECK(cudaMemcpy(hostPivotRow, deviceDist + localPivotRow * numNodes,
                                  numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hostPivotRow, static_cast<int>(numNodes), MPI_UNSIGNED, pivotOwner,
                  MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(devicePivotRow, hostPivotRow,
                              numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));

        floydWarshallKernel<<<grid, block>>>(deviceDist, devicePath, devicePivotRow,
                                             localRows, numNodes, k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDuration = static_cast<long>(duration.count());
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localElements > 0) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist,
                              localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data(), devicePath,
                              localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    MPI_Gatherv(localDist.data(), mpiCounts[static_cast<size_t>(rank)], MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, mpiCounts.data(), mpiDisplacements.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Floyd-Warshall performs one min-plus operation per matrix element per k.
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double seconds = maxDuration / 1000.0;
        const double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

            CUDA_CHECK(cudaFree(deviceDist));
            CUDA_CHECK(cudaFree(devicePath));
            CUDA_CHECK(cudaFree(devicePivotRow));
            CUDA_CHECK(cudaFreeHost(hostPivotRow));
            MPI_Comm_free(&localCommunicator);
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(devicePivotRow));
    CUDA_CHECK(cudaFreeHost(hostPivotRow));
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return 0;
}

#undef CUDA_CHECK
