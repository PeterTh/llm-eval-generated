#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Fraction of each rank's rows relaxed by its OpenMP host threads. The remaining rows
// are relaxed on the rank's GPU, and both run concurrently for every pivot k.
constexpr double CPU_ROW_SHARE = 0.1;

// Host threads per rank beyond this only add barrier cost, since the host share is
// memory-bandwidth bound and is synchronized once per pivot.
constexpr int MAX_HOST_THREADS_PER_RANK = 32;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Fills the local rows [rowBegin, rowEnd) of the distance matrix. Every rank walks the
// full random sequence so the values are identical to the sequential program; a rank
// simply keeps the entries that belong to its rows.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t rowBegin, const size_t rowEnd,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t localBegin = rowBegin * numNodes;
    const size_t localEnd = rowEnd * numNodes;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= localBegin && i < localEnd) {
            dist[i - localBegin] = value;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        dist[(i - rowBegin) * numNodes + i] = 0;
    }
}

// Path(src, dst) starts as src, so every entry of local row r holds the global row index.
void initializePathMatrix(std::vector<unsigned int>& path, const size_t rowBegin,
                          const size_t rowEnd, const size_t numNodes) {
    const long long localRows = static_cast<long long>(rowEnd - rowBegin);
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < localRows; ++r) {
        std::fill_n(path.begin() + r * numNodes, numNodes,
                    static_cast<unsigned int>(rowBegin + static_cast<size_t>(r)));
    }
}

// Relaxes the GPU-resident rows for pivot k. Row k and column k never change during
// iteration k (the diagonal is zero and weights are non-negative), so reading them
// concurrently with these writes is race-free and matches the sequential order.
__global__ void relaxRowsKernel(unsigned int* dist, unsigned int* path,
                                const unsigned int* rowK, const size_t k,
                                const size_t rows, const size_t n) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= n) {
        return;
    }

    const size_t ij = i * n + j;
    const unsigned int newDist = dist[i * n + k] + rowK[j];
    if (newDist < dist[ij]) {
        dist[ij] = newDist;
        path[ij] = static_cast<unsigned int>(k);
    }
}

// Distributed hybrid Floyd-Warshall. Rank r owns the global rows [rowStarts[r], rowStarts[r+1]).
// Inside a rank, the first cpuRows local rows are relaxed by OpenMP threads on the host and
// the rest live on the rank's GPU. For each pivot k, the owner of row k broadcasts it, and
// every rank relaxes its rows against it. dist and path hold the local rows on entry and exit.
void floydWarshallHybrid(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                         const std::vector<size_t>& rowStarts, const int rank,
                         MPI_Comm comm, const size_t numNodes) {
    const size_t rowBegin = rowStarts[rank];
    const size_t localRows = rowStarts[rank + 1] - rowBegin;
    const size_t cpuRows = static_cast<size_t>(localRows * CPU_ROW_SHARE);
    const size_t gpuRows = localRows - cpuRows;
    const size_t gpuBytes = gpuRows * numNodes * sizeof(unsigned int);
    const size_t rowBytes = numNodes * sizeof(unsigned int);
    const int count = static_cast<int>(numNodes);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dRow = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (gpuRows > 0) {
        CUDA_CHECK(cudaMalloc(&dDist, gpuBytes));
        CUDA_CHECK(cudaMalloc(&dPath, gpuBytes));
        CUDA_CHECK(cudaMemcpy(dDist, dist.data() + cpuRows * numNodes, gpuBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, path.data() + cpuRows * numNodes, gpuBytes, cudaMemcpyHostToDevice));
    }
    if (numNodes > 0) {
        CUDA_CHECK(cudaMalloc(&dRow, rowBytes));
    }

    // Broadcast buffer for the current pivot row (host side)
    std::vector<unsigned int> rowK(numNodes);
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((numNodes + block.x - 1) / block.x),
                    static_cast<unsigned int>((gpuRows + block.y - 1) / block.y));

#pragma omp parallel
    for (size_t k = 0; k < numNodes; ++k) {
#pragma omp master
        {
            const int owner = static_cast<int>(
                std::upper_bound(rowStarts.begin(), rowStarts.end(), k) - rowStarts.begin()) - 1;
            if (rank == owner) {
                const size_t li = k - rowBegin;
                if (li < cpuRows) {
                    std::memcpy(rowK.data(), dist.data() + li * numNodes, rowBytes);
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(rowK.data(), dDist + (li - cpuRows) * numNodes,
                                               rowBytes, cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }

            MPI_Bcast(rowK.data(), count, MPI_UNSIGNED, owner, comm);

            if (gpuRows > 0) {
                CUDA_CHECK(cudaMemcpyAsync(dRow, rowK.data(), rowBytes, cudaMemcpyHostToDevice, stream));
                relaxRowsKernel<<<grid, block, 0, stream>>>(dDist, dPath, dRow, k, gpuRows, numNodes);
                CUDA_CHECK(cudaGetLastError());
            }
        }
#pragma omp barrier

        // CPU share of the rows, overlapping with the GPU kernel launched above
        const unsigned int* rk = rowK.data();
#pragma omp for schedule(static)
        for (long long li = 0; li < static_cast<long long>(cpuRows); ++li) {
            unsigned int* d = dist.data() + static_cast<size_t>(li) * numNodes;
            unsigned int* p = path.data() + static_cast<size_t>(li) * numNodes;
            const unsigned int distIK = d[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rk[j];
                const bool improved = newDist < d[j];
                d[j] = improved ? newDist : d[j];
                p[j] = improved ? static_cast<unsigned int>(k) : p[j];
            }
        }

#pragma omp master
        {
            if (gpuRows > 0) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
    }

    // Bring the GPU share back into the host block
    if (gpuRows > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data() + cpuRows * numNodes, dDist, gpuBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data() + cpuRows * numNodes, dPath, gpuBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dDist));
        CUDA_CHECK(cudaFree(dPath));
    }
    if (dRow != nullptr) {
        CUDA_CHECK(cudaFree(dRow));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
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
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int exitCode = -1;
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
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode != -1) {
        MPI_Finalize();
        return exitCode;
    }

    // Bind each rank to a GPU and give each rank a share of the host cores
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    omp_set_num_threads(std::min(MAX_HOST_THREADS_PER_RANK,
                                 std::max(1, omp_get_num_procs() / localSize)));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row partition across MPI ranks
    std::vector<size_t> rowStarts(size + 1);
    for (int r = 0; r <= size; ++r) {
        rowStarts[r] = static_cast<size_t>(r) * numNodes / static_cast<size_t>(size);
    }
    const size_t rowBegin = rowStarts[rank];
    const size_t rowEnd = rowStarts[rank + 1];

    // Allocate and initialize the local rows of the matrices
    std::vector<unsigned int> dist((rowEnd - rowBegin) * numNodes);
    std::vector<unsigned int> path((rowEnd - rowBegin) * numNodes);

    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, rowBegin, rowEnd, 1, MAX_DISTANCE);
    initializePathMatrix(path, rowBegin, rowEnd, numNodes);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallHybrid(dist, path, rowStarts, rank, MPI_COMM_WORLD, numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    const double seconds = MPI_Wtime() - start;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the full matrix on rank 0 only when it is needed for output or validation
    if (printResults || validate) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>((rowStarts[r + 1] - rowStarts[r]) * numNodes);
            displs[r] = static_cast<int>(rowStarts[r] * numNodes);
        }
        std::vector<unsigned int> fullDist(rank == 0 ? numNodes * numNodes : 0);
        MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                    fullDist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(fullDist, "DistanceMatrix");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(fullDist, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                exitCode = valid ? 0 : 1;
            }
        }
    }

    // Every rank returns the same status as rank 0
    if (validate) {
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        exitCode = 0;
    }

    MPI_Finalize();
    return exitCode;
}
