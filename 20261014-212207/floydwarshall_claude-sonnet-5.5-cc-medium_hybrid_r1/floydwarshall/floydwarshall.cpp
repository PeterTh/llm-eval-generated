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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Block-row distribution of the matrix rows over MPI ranks
struct Partition {
    size_t n;
    int P;
    size_t base, rem;
    Partition(size_t n_, int P_) : n(n_), P(P_), base(n_ / P_), rem(n_ % P_) {}
    size_t rowStart(int r) const { return r * base + std::min<size_t>(r, rem); }
    size_t rowCount(int r) const { return base + (static_cast<size_t>(r) < rem ? 1 : 0); }
    int owner(size_t row) const {
        const size_t big = rem * (base + 1);
        if (row < big) return static_cast<int>(row / (base + 1));
        return static_cast<int>(rem + (row - big) / base);
    }
};

// Generates only the local rows [rowBegin, rowBegin+rows) of the distance matrix,
// reproducing the exact sequential rand_r stream of the original code.
void initializeDistanceRows(std::vector<unsigned int>& dist, const size_t numNodes,
                            const size_t rowBegin, const size_t rows,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t first = rowBegin * numNodes;
    const size_t last = (rowBegin + rows) * numNodes;

    for (size_t i = 0; i < first; ++i) rand_r(&seed);
    for (size_t i = first; i < last; ++i) {
        dist[i - first] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero
    for (size_t i = rowBegin; i < rowBegin + rows; ++i) {
        dist[(i - rowBegin) * numNodes + i] = 0;
    }
}

// path[idx2(i, j)] == j in the original initialisation (row index of the matrix)
__global__ void initPathKernel(unsigned int* __restrict__ path, size_t n, size_t rows, size_t rowBegin) {
    const size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t r = blockIdx.y;
    if (j < n && r < rows) path[r * n + j] = static_cast<unsigned int>(rowBegin + r);
}

// Relax rows [r0, r0+cnt) of the local block using row k (rowK) of the matrix
__global__ void relaxKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ rowK, size_t n, size_t r0,
                            size_t cnt, unsigned int k) {
    const size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t r = r0 + blockIdx.y;
    if (j >= n || blockIdx.y >= cnt) return;
    const size_t o = r * n;
    const unsigned int distIK = dist[o + k];
    const unsigned int newDist = distIK + __ldg(&rowK[j]);
    if (newDist < dist[o + j]) {
        dist[o + j] = newDist;
        path[o + j] = k;
    }
}

// Distributed Floyd-Warshall. Returns local result in 'dist' (host).
void floydWarshallHybrid(std::vector<unsigned int>& dist, const size_t n, const Partition& part,
                         const int rank, double& elapsedMs) {
    const size_t rows = part.rowCount(rank);
    const size_t rowBegin = part.rowStart(rank);
    const size_t localElems = rows * n;

    unsigned int *dDist = nullptr, *dPath = nullptr, *dRow[2] = {nullptr, nullptr};
    unsigned int* hRow[2] = {nullptr, nullptr};
    if (localElems) {
        CUDA_CHECK(cudaMalloc(&dDist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dPath, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(dDist, dist.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dRow[b], n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&hRow[b], n * sizeof(unsigned int)));
    }

    cudaStream_t sBulk, sRow;
    cudaEvent_t evEarly;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sBulk, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sRow, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&evEarly, cudaEventDisableTiming));

    const dim3 blk(256);
    const unsigned gx = static_cast<unsigned>((n + blk.x - 1) / blk.x);

    if (localElems) {
        initPathKernel<<<dim3(gx, (unsigned)rows), blk, 0, sBulk>>>(dPath, n, rows, rowBegin);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(sBulk));
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    // Row 0 bootstrap
    {
        const int o = part.owner(0);
        if (rank == o) {
            CUDA_CHECK(cudaMemcpy(hRow[0], dDist + (0 - rowBegin) * n, n * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hRow[0], static_cast<int>(n), MPI_UNSIGNED, o, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpyAsync(dRow[0], hRow[0], n * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, sBulk));
    }

    for (size_t k = 0; k < n; ++k) {
        const int cur = k & 1, nxt = cur ^ 1;
        const bool hasNext = (k + 1 < n);
        const int nextOwner = hasNext ? part.owner(k + 1) : -1;

        // Owner of row k+1 relaxes it first so it can be broadcast while the bulk update runs.
        // sRow must see the row buffer that was uploaded on sBulk.
        if (hasNext && rank == nextOwner) {
            CUDA_CHECK(cudaEventRecord(evEarly, sBulk));
            CUDA_CHECK(cudaStreamWaitEvent(sRow, evEarly, 0));
            relaxKernel<<<dim3(gx, 1), blk, 0, sRow>>>(dDist, dPath, dRow[cur], n, k + 1 - rowBegin, 1,
                                                       static_cast<unsigned>(k));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hRow[nxt], dDist + (k + 1 - rowBegin) * n, n * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, sRow));
            CUDA_CHECK(cudaEventRecord(evEarly, sRow));
            CUDA_CHECK(cudaStreamWaitEvent(sBulk, evEarly, 0));  // bulk waits for the early row update
        }

        // Bulk update of all local rows with row k
        if (rows) {
            // Process in y-chunks to respect the grid.y limit (65535)
            for (size_t r0 = 0; r0 < rows; r0 += 65535) {
                const size_t cnt = std::min<size_t>(65535, rows - r0);
                relaxKernel<<<dim3(gx, (unsigned)cnt), blk, 0, sBulk>>>(dDist, dPath, dRow[cur], n, r0, cnt,
                                                                      static_cast<unsigned>(k));
            }
            CUDA_CHECK(cudaGetLastError());
        }

        if (hasNext) {
            if (rank == nextOwner) CUDA_CHECK(cudaStreamSynchronize(sRow));
            MPI_Bcast(hRow[nxt], static_cast<int>(n), MPI_UNSIGNED, nextOwner, MPI_COMM_WORLD);
            // Ordered after bulk kernel k on sBulk (which is the last reader of dRow[nxt] from step k-1)
            CUDA_CHECK(cudaMemcpyAsync(dRow[nxt], hRow[nxt], n * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, sBulk));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(sBulk));
    CUDA_CHECK(cudaStreamSynchronize(sRow));
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    if (localElems) {
        CUDA_CHECK(cudaMemcpy(dist.data(), dDist, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dDist));
        CUDA_CHECK(cudaFree(dPath));
    }
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(dRow[b]));
        CUDA_CHECK(cudaFreeHost(hRow[b]));
    }
    CUDA_CHECK(cudaEventDestroy(evEarly));
    CUDA_CHECK(cudaStreamDestroy(sBulk));
    CUDA_CHECK(cudaStreamDestroy(sRow));
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
    const size_t lim = std::min(numNodes, static_cast<size_t>(10));
    long firstBad = -1;
    #pragma omp parallel for collapse(2) schedule(static) reduction(max : firstBad)
    for (size_t i = 0; i < lim; ++i) {
        for (size_t j = 0; j < lim; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    firstBad = std::max(firstBad, static_cast<long>((i * lim + j) * numNodes + k));
                    break;
                }
            }
        }
    }
    if (firstBad >= 0) {
        const size_t v = static_cast<size_t>(firstBad);
        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
               v / numNodes / lim, (v / numNodes) % lim, v % numNodes);
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // One GPU per rank (round-robin over node-local ranks)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

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
        printf("Initializing graph...\n");
    }

    const Partition part(numNodes, nprocs);
    const size_t rows = part.rowCount(rank);
    std::vector<unsigned int> localDist(rows * numNodes);
    initializeDistanceRows(localDist, numNodes, part.rowStart(rank), rows, 1, MAX_DISTANCE);

    if (rank == 0) printf("Computing shortest paths...\n");
    double ms = 0.0;
    floydWarshallHybrid(localDist, numNodes, part, rank, ms);

    // Gather full distance matrix on rank 0
    std::vector<unsigned int> dist;
    if (printResults || validate) {
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            counts[r] = static_cast<int>(part.rowCount(r) * numNodes);
            displs[r] = static_cast<int>(part.rowStart(r) * numNodes);
        }
        if (rank == 0) dist.resize(numNodes * numNodes);
        MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, dist.data(), counts.data(),
                    displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    int rc = 0;
    if (rank == 0) {
        const long msCount = static_cast<long>(ms);
        printf("Computation time: %ld ms\n", msCount);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (msCount / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            if (validateResult(dist, numNodes)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }
    MPI_Finalize();
    return rc;
}
