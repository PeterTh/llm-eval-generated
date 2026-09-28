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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // rand_r consumes a sequential seed state, so this loop must stay serial
    // to reproduce the original input graph exactly.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// One Floyd-Warshall relaxation step for a block of rows resident on the GPU.
// For iteration k, row k and column k are invariant (a strict improvement of
// dist[k][j] or dist[i][k] via node k is impossible since dist[k][k] == 0), so
// all (i, j) updates within one k are independent and the result is identical
// to the serial algorithm.
__global__ void fwRelaxKernel(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const unsigned int* __restrict__ krow,
                              const size_t n, const size_t localRows,
                              const unsigned int k) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t li = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (li >= localRows || j >= n) return;

    const unsigned int distIK = dist[li * n + k];
    const unsigned int newDist = distIK + __ldg(&krow[j]);
    const size_t off = li * n + j;
    if (newDist < dist[off]) {
        dist[off] = newDist;
        path[off] = k;
    }
}

// Distributed hybrid Floyd-Warshall: rows are block-distributed over MPI
// ranks, each rank relaxes its block on its GPU, and the owner of row k
// broadcasts that row every iteration.
void floydWarshall(unsigned int* d_dist, unsigned int* d_path,
                   unsigned int* h_krow, unsigned int* d_krow,
                   const size_t numNodes, const size_t rowBegin,
                   const size_t localRows, MPI_Comm comm,
                   const int* rowOwner, cudaStream_t stream) {
    const dim3 block(128, 4);
    const dim3 grid((unsigned)((numNodes + block.x - 1) / block.x),
                    (unsigned)((localRows + block.y - 1) / block.y));

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[k];
        int rank;
        MPI_Comm_rank(comm, &rank);

        // Owner pulls the current row k from its GPU block.
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpyAsync(h_krow, d_dist + (k - rowBegin) * numNodes,
                                       numNodes * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        MPI_Bcast(h_krow, (int)numNodes, MPI_UNSIGNED, owner, comm);

        CUDA_CHECK(cudaMemcpyAsync(d_krow, h_krow, numNodes * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));

        if (localRows > 0) {
            fwRelaxKernel<<<grid, block, 0, stream>>>(d_dist, d_path, d_krow,
                                                      numNodes, localRows,
                                                      (unsigned int)k);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
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
    bool ok = true;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
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
                            ok = false;
                        }
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI ranks: %d, OpenMP threads: %d\n", nprocs, omp_get_max_threads());
    }

    // Bind each rank to a GPU (round-robin across the node's devices).
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // Block row distribution: first (numNodes % nprocs) ranks get one extra row.
    std::vector<int> rowCounts(nprocs), rowStarts(nprocs);
    {
        const size_t base = numNodes / nprocs;
        const size_t rem = numNodes % nprocs;
        size_t start = 0;
        for (int r = 0; r < nprocs; ++r) {
            rowCounts[r] = (int)(base + ((size_t)r < rem ? 1 : 0));
            rowStarts[r] = (int)start;
            start += rowCounts[r];
        }
    }
    const size_t rowBegin = rowStarts[rank];
    const size_t localRows = rowCounts[rank];

    // Precompute the owning rank of every row k.
    std::vector<int> rowOwner(numNodes);
    for (int r = 0; r < nprocs; ++r) {
        for (int i = 0; i < rowCounts[r]; ++i) rowOwner[rowStarts[r] + i] = r;
    }

    // Element counts/displacements for Scatterv/Gatherv (one row = numNodes elements).
    std::vector<int> elemCounts(nprocs), elemDispls(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        elemCounts[r] = rowCounts[r] * (int)numNodes;
        elemDispls[r] = rowStarts[r] * (int)numNodes;
    }

    // Rank 0 holds the full matrices; every rank holds its row block.
    std::vector<unsigned int> dist, path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);

        // Initialize
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Pinned host staging buffers for fast H2D/D2H transfers.
    unsigned int* h_block = nullptr;
    unsigned int* h_pathBlock = nullptr;
    unsigned int* h_krow = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_block, std::max<size_t>(1, localRows * numNodes) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&h_pathBlock, std::max<size_t>(1, localRows * numNodes) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&h_krow, std::max<size_t>(1, numNodes) * sizeof(unsigned int)));

    // Device buffers for the local row blocks and the broadcast row.
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_krow = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, std::max<size_t>(1, localRows * numNodes) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_path, std::max<size_t>(1, localRows * numNodes) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_krow, std::max<size_t>(1, numNodes) * sizeof(unsigned int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Distribute row blocks and upload to the GPUs.
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, elemCounts.data(), elemDispls.data(),
                 MPI_UNSIGNED, h_block, elemCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, elemCounts.data(), elemDispls.data(),
                 MPI_UNSIGNED, h_pathBlock, elemCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(d_dist, h_block, localRows * numNodes * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, h_pathBlock, localRows * numNodes * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(d_dist, d_path, h_krow, d_krow, numNodes, rowBegin, localRows,
                  MPI_COMM_WORLD, rowOwner.data(), stream);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Collect the result blocks back on rank 0.
    CUDA_CHECK(cudaMemcpy(h_block, d_dist, localRows * numNodes * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pathBlock, d_path, localRows * numNodes * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    MPI_Gatherv(h_block, elemCounts[rank], MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, elemCounts.data(), elemDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(h_pathBlock, elemCounts[rank], MPI_UNSIGNED,
                rank == 0 ? path.data() : nullptr, elemCounts.data(), elemDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_krow));
    CUDA_CHECK(cudaFreeHost(h_block));
    CUDA_CHECK(cudaFreeHost(h_pathBlock));
    CUDA_CHECK(cudaFreeHost(h_krow));

    int ret = 0;
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
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
