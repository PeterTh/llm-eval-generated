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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

constexpr int TX = 32;
constexpr int TY = 8;
constexpr int ROWS_PER_THREAD = 2;

// Path matrix: path[i][j] = j for the initial state (same as the original
// initializePathMatrix, which sets path[idx2(i,j)] = j and path[idx2(j,i)] = i,
// i.e. row-major path[r][c] = c). Only local rows are initialized here.
__global__ void initPathKernel(unsigned int* path, const size_t localRows, const size_t n) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (i < localRows && j < n) path[i * n + j] = (unsigned int)j;
}

// One Floyd-Warshall step for intermediate node k on a block of local rows.
// rowK is the (final) global row k; it is not modified during step k.
__global__ void fwStepKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                             const unsigned int* __restrict__ rowK, const size_t localRows,
                             const size_t n, const unsigned int k) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const unsigned int distKJ = rowK[j];
    const size_t i0 = (size_t)blockIdx.y * (TY * ROWS_PER_THREAD) + threadIdx.y;
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const size_t i = i0 + (size_t)r * TY;
        if (i < localRows) {
            const unsigned int distIK = dist[i * n + k];
            const unsigned int newDist = distIK + distKJ;
            if (newDist < dist[i * n + j]) {
                dist[i * n + j] = newDist;
                path[i * n + j] = k;
            }
        }
    }
}

// Distributed GPU Floyd-Warshall. Rows are block-distributed over MPI ranks;
// each rank keeps its rows (dist and path) on its GPU. For each k the owner of
// row k broadcasts it. Host staging buffers are double-buffered so that ranks
// can run ahead of their GPUs.
void floydWarshallDistributed(unsigned int* dDist, unsigned int* dPath, const size_t localRows,
                              const size_t blk, const size_t n, MPI_Comm comm, int rank) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    unsigned int* hRow[2];
    unsigned int* dRow[2];
    cudaEvent_t ev[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&hRow[b], n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dRow[b], n * sizeof(unsigned int)));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ev[b], stream));
    }

    const dim3 block(TX, TY);
    const dim3 grid((unsigned)((n + TX - 1) / TX),
                    (unsigned)((localRows + TY * ROWS_PER_THREAD - 1) / (TY * ROWS_PER_THREAD)));

    for (size_t k = 0; k < n; ++k) {
        const int b = (int)(k & 1);
        const int owner = (int)(k / blk);
        const unsigned int* rowPtr;
        // host buffer b / device buffer b may still be in use by step k-2
        CUDA_CHECK(cudaEventSynchronize(ev[b]));
        if (owner == rank) {
            const size_t lk = k - (size_t)rank * blk;
            rowPtr = dDist + lk * n;
            CUDA_CHECK(cudaMemcpyAsync(hRow[b], rowPtr, n * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Bcast(hRow[b], (int)n, MPI_UNSIGNED, owner, comm);
        } else {
            MPI_Bcast(hRow[b], (int)n, MPI_UNSIGNED, owner, comm);
            CUDA_CHECK(cudaMemcpyAsync(dRow[b], hRow[b], n * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, stream));
            rowPtr = dRow[b];
        }
        if (localRows > 0)
            fwStepKernel<<<grid, block, 0, stream>>>(dDist, dPath, rowPtr, localRows, n,
                                                     (unsigned int)k);
        CUDA_CHECK(cudaEventRecord(ev[b], stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFreeHost(hRow[b]));
        CUDA_CHECK(cudaFree(dRow[b]));
        CUDA_CHECK(cudaEventDestroy(ev[b]));
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
    const size_t lim = std::min(numNodes, static_cast<size_t>(10));
    bool ok = true;
    #pragma omp parallel for collapse(2) reduction(&&:ok)
    for (size_t i = 0; i < lim; ++i) {
        for (size_t j = 0; j < lim; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        #pragma omp critical
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        ok = false;
                        break;
                    }
                }
            }
        }
    }
    
    return ok;
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
        printf("MPI ranks: %d, OpenMP threads: %d\n", size, omp_get_max_threads());
    }

    // Bind each rank to a GPU (round-robin over the GPUs of its node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Row-block distribution
    const size_t blk = std::max<size_t>(1, (numNodes + size - 1) / size);
    const size_t rowBegin = std::min(numNodes, (size_t)rank * blk);
    const size_t rowEnd = std::min(numNodes, rowBegin + blk);
    const size_t localRows = rowEnd - rowBegin;

    std::vector<unsigned int> dist;  // full matrix, rank 0 only
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t b = std::min(numNodes, (size_t)r * blk);
        const size_t e = std::min(numNodes, b + blk);
        counts[r] = (int)((e - b) * numNodes);
        displs[r] = (int)(b * numNodes);
    }

    if (rank == 0) {
        printf("Initializing graph...\n");
        dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }

    unsigned int *dDist = nullptr, *dPath = nullptr;
    const size_t localElems = localRows * numNodes;
    CUDA_CHECK(cudaMalloc(&dDist, std::max<size_t>(1, localElems) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, std::max<size_t>(1, localElems) * sizeof(unsigned int)));
    {
        std::vector<unsigned int> local(localElems);
        MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                     local.data(), (int)localElems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (localElems > 0) {
            CUDA_CHECK(cudaMemcpy(dDist, local.data(), localElems * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            dim3 blk2(TX, TY);
            dim3 grd((unsigned)((numNodes + TX - 1) / TX), (unsigned)((localRows + TY - 1) / TY));
            initPathKernel<<<grd, blk2>>>(dPath, localRows, numNodes);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallDistributed(dDist, dPath, localRows, blk, numNodes, MPI_COMM_WORLD, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather the distance matrix on rank 0
    {
        std::vector<unsigned int> local(localElems);
        if (localElems > 0)
            CUDA_CHECK(cudaMemcpy(local.data(), dDist, localElems * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        MPI_Gatherv(local.data(), (int)localElems, MPI_UNSIGNED, rank == 0 ? dist.data() : nullptr,
                    counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));

    int rc = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            if (valid) {
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
