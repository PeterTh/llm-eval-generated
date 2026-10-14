#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

// Thread block shape for the GPU relaxation kernel
constexpr unsigned int BLOCK_X = 32;
constexpr unsigned int BLOCK_Y = 8;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Fills the rows [rowBegin, rowBegin + rowCount) of the distance matrix. The
// generator is advanced over the preceding rows, so every rank obtains exactly
// the values the sequential generator produces for its rows.
void initializeDistanceRows(std::vector<unsigned int>& dist, const size_t numNodes,
                            const size_t rowBegin, const size_t rowCount,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < rowBegin * numNodes; ++i) {
        rand_r(&seed);
    }
    for (size_t i = 0; i < rowCount * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t r = 0; r < rowCount; ++r) {
        dist[r * numNodes + rowBegin + r] = 0;
    }
}

// Path entries of global row i are initialized to i
void initializePathRows(std::vector<unsigned int>& path, const size_t numNodes,
                        const size_t rowBegin, const size_t rowCount) {
    for (size_t r = 0; r < rowCount; ++r) {
        std::fill_n(path.begin() + r * numNodes, numNodes, static_cast<unsigned int>(rowBegin + r));
    }
}

// Relaxes the local rows on the GPU for intermediate node k. The pivot row
// (row k of the distance matrix) is passed in separately.
__global__ void relaxRowsKernel(unsigned int* dist, unsigned int* path,
                                const unsigned int* pivotRow,
                                const size_t rows, const size_t numNodes,
                                const unsigned int k) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= numNodes) {
        return;
    }

    // dist[i][k] is not changed during iteration k, since dist[k][k] == 0
    unsigned int* rowI = dist + i * numNodes;
    const unsigned int newDist = rowI[k] + pivotRow[j];
    if (newDist < rowI[j]) {
        rowI[j] = newDist;
        path[i * numNodes + j] = k;
    }
}

// Classic Floyd-Warshall algorithm on the rows owned by this rank. The rows are
// split into contiguous sub-blocks, one per GPU, and each OpenMP thread drives
// one GPU. In iteration k the owner of row k copies it to the host, MPI
// broadcasts it between ranks, and every GPU relaxes its rows against it.
// The driver threads need cores of their own, so launch with core binding
// relaxed (e.g. mpirun --bind-to none) when a rank drives several GPUs.
// Returns the computation time of this rank in milliseconds.
long long floydWarshall(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path,
                        const std::vector<size_t>& rowStarts,
                        const int rank, const size_t numNodes,
                        const int numThreads, const int firstDevice, const int deviceCount) {
    const size_t rowBegin = rowStarts[rank];
    const size_t rowCount = rowStarts[rank + 1] - rowBegin;

    // Local rows [threadStarts[t], threadStarts[t+1]) belong to thread t
    std::vector<size_t> threadStarts(numThreads + 1);
    for (int t = 0; t <= numThreads; ++t) {
        threadStarts[t] = rowCount * t / numThreads;
    }

    // Pivot row buffers, alternated by parity of k so that the copy for
    // iteration k+1 cannot overwrite the row still read by iteration k
    std::vector<unsigned int> pivotRows(2 * numNodes);

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    #pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();
        const size_t firstRow = threadStarts[tid];
        const size_t subRows = threadStarts[tid + 1] - firstRow;
        const size_t bytes = subRows * numNodes * sizeof(unsigned int);

        // Non-blocking stream, so that the host only waits where the owner needs a row
        CUDA_CHECK(cudaSetDevice((firstDevice + tid) % deviceCount));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        unsigned int* dDist = nullptr;
        unsigned int* dPath = nullptr;
        unsigned int* dPivot = nullptr;
        CUDA_CHECK(cudaMalloc(&dPivot, numNodes * sizeof(unsigned int)));
        if (subRows > 0) {
            CUDA_CHECK(cudaMalloc(&dDist, bytes));
            CUDA_CHECK(cudaMalloc(&dPath, bytes));
            CUDA_CHECK(cudaMemcpyAsync(dDist, dist.data() + firstRow * numNodes, bytes,
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dPath, path.data() + firstRow * numNodes, bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        #pragma omp barrier
        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            start = std::chrono::high_resolution_clock::now();
        }
        #pragma omp barrier

        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid(static_cast<unsigned int>((numNodes + BLOCK_X - 1) / BLOCK_X),
                        static_cast<unsigned int>((subRows + BLOCK_Y - 1) / BLOCK_Y));

        for (size_t k = 0; k < numNodes; ++k) {
            const int owner = static_cast<int>(std::upper_bound(rowStarts.begin(), rowStarts.end(), k)
                                               - rowStarts.begin()) - 1;
            const size_t localK = k - rowBegin;
            unsigned int* pivotRow = pivotRows.data() + (k & 1) * numNodes;

            // The thread whose GPU holds row k on the owner rank copies it to the host
            if (rank == owner && localK >= firstRow && localK < firstRow + subRows) {
                CUDA_CHECK(cudaMemcpyAsync(pivotRow, dDist + (localK - firstRow) * numNodes,
                                           numNodes * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            #pragma omp barrier
            #pragma omp master
            {
                MPI_Bcast(pivotRow, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            }
            #pragma omp barrier

            // Pageable source: the copy is staged before the call returns, so the
            // host buffer can be reused by the broadcast of a later iteration
            if (subRows > 0) {
                CUDA_CHECK(cudaMemcpyAsync(dPivot, pivotRow, numNodes * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice, stream));
                relaxRowsKernel<<<grid, block, 0, stream>>>(dDist, dPath, dPivot, subRows, numNodes,
                                                            static_cast<unsigned int>(k));
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // Bring the result back to the host
        if (subRows > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dist.data() + firstRow * numNodes, dDist, bytes,
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (subRows > 0) {
            CUDA_CHECK(cudaFree(dDist));
            CUDA_CHECK(cudaFree(dPath));
        }
        CUDA_CHECK(cudaFree(dPivot));
        CUDA_CHECK(cudaStreamDestroy(stream));

        #pragma omp barrier
        #pragma omp master
        {
            end = std::chrono::high_resolution_clock::now();
        }
    }

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
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
    // Only the thread that drives OpenMP's master role makes MPI calls
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = rank == 0;

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
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each rank drives the GPUs of its node, sharing them with the other ranks
    // on the node when there are more ranks than devices
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int devicesPerRank = std::max(1, deviceCount / localSize);
    const int firstDevice = (localRank * devicesPerRank) % deviceCount;

    // Balanced contiguous row blocks: rank r owns rows [rowStarts[r], rowStarts[r+1])
    std::vector<size_t> rowStarts(size + 1);
    for (int r = 0; r <= size; ++r) {
        rowStarts[r] = numNodes * static_cast<size_t>(r) / static_cast<size_t>(size);
    }
    const size_t rowBegin = rowStarts[rank];
    const size_t rowCount = rowStarts[rank + 1] - rowBegin;

    // Allocate matrices (local rows only)
    std::vector<unsigned int> dist(rowCount * numNodes);
    std::vector<unsigned int> path(rowCount * numNodes);

    // Initialize
    if (root) {
        printf("Initializing graph...\n");
    }
    initializeDistanceRows(dist, numNodes, rowBegin, rowCount, 1, MAX_DISTANCE);
    initializePathRows(path, numNodes, rowBegin, rowCount);

    // Run Floyd-Warshall
    if (root) {
        printf("Computing shortest paths...\n");
    }
    const long long localMs = floydWarshall(dist, path, rowStarts, rank, numNodes,
                                            devicesPerRank, firstDevice, deviceCount);
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the full distance matrix on the root for output and validation
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<unsigned int> fullDist(root ? numNodes * numNodes : 0);
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>((rowStarts[r + 1] - rowStarts[r]) * numNodes);
            displs[r] = static_cast<int>(rowStarts[r] * numNodes);
        }
        MPI_Gatherv(dist.data(), static_cast<int>(rowCount * numNodes), MPI_UNSIGNED,
                    fullDist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (root) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(fullDist, "DistanceMatrix");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullDist, numNodes);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
