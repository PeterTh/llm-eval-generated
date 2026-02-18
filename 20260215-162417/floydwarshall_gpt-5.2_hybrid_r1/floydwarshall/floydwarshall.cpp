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

#define CUDA_CHECK(expr) do { \
    const cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

// Index calculation for flattened 2D array (row-major)
__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
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
    // Deterministic initialization; safe to parallelize.
#pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx2(j, i, numNodes)] = static_cast<unsigned int>(i);
        }
    }
}

static inline void computeRowPartition(const size_t n, const int size, const int rank,
                                       size_t& rowStart, size_t& rowCount) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    rowCount = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    rowStart = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
}

static inline int ownerRankForRow(const size_t n, const int size, const size_t row) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    // Ranks [0, rem) own (base+1) rows, others own base rows.
    if (row < (base + 1) * rem) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - (base + 1) * rem) / base);
}

__global__ void fw_update_kernel(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ rowK,
                                const int n, const int k, const int localRows) {
    const int i = static_cast<int>(blockIdx.y);
    const int j = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (i >= localRows || j >= n) return;

    __shared__ unsigned int distIK;
    if (threadIdx.x == 0) {
        distIK = dist[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(k)];
    }
    __syncthreads();

    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(n);
    const size_t idx = base + static_cast<size_t>(j);

    const unsigned int distIJ = dist[idx];
    const unsigned int newDist = distIK + rowK[j];
    if (newDist < distIJ) {
        dist[idx] = newDist;
        path[idx] = static_cast<unsigned int>(k);
    }
}

static double floydWarshallHybridMPI_OMP_CUDA(std::vector<unsigned int>& distGlobal,
                                            std::vector<unsigned int>& pathGlobal,
                                            const size_t numNodes,
                                            const bool needFullGather) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t rowStart = 0, localRows = 0;
    computeRowPartition(numNodes, size, rank, rowStart, localRows);

    const size_t localElems = localRows * numNodes;
    std::vector<unsigned int> distLocal(localElems);
    std::vector<unsigned int> pathLocal(localElems);

    // Scatter distance matrix rows from rank 0 to all ranks (preserves original initialization semantics).
    std::vector<int> sendcounts, displs;
    if (rank == 0) {
        sendcounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            size_t rs = 0, rc = 0;
            computeRowPartition(numNodes, size, r, rs, rc);
            sendcounts[r] = static_cast<int>(rc * numNodes);
            displs[r] = static_cast<int>(rs * numNodes);
        }
    }

    MPI_Scatterv(rank == 0 ? distGlobal.data() : nullptr,
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 distLocal.data(), static_cast<int>(localElems), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Initialize local path deterministically (matches original path init).
#pragma omp parallel for
    for (size_t li = 0; li < localRows; ++li) {
        const unsigned int gi = static_cast<unsigned int>(rowStart + li);
        const size_t base = li * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            pathLocal[base + j] = gi;
        }
    }

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    CUDA_CHECK(cudaSetDevice(rank % devCount));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_rowK = nullptr;

    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpyAsync(d_dist, distLocal.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_path, pathLocal.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMalloc(&d_rowK, numNodes * sizeof(unsigned int)));

    unsigned int* h_rowK = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_rowK, numNodes * sizeof(unsigned int)));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int n = static_cast<int>(numNodes);
    const int lrows = static_cast<int>(localRows);
    const dim3 block(256, 1, 1);
    const dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x),
                    static_cast<unsigned int>(std::max(1, lrows)), 1);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto computeStart = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; ++k) {
        const int owner = ownerRankForRow(numNodes, size, static_cast<size_t>(k));

        if (rank == owner) {
            const size_t lk = static_cast<size_t>(k) - rowStart;
            if (lk < localRows && localElems > 0) {
                CUDA_CHECK(cudaMemcpyAsync(h_rowK,
                                           d_dist + lk * numNodes,
                                           numNodes * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost,
                                           stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                // Should not happen; but keep MPI collective well-defined.
                std::fill(h_rowK, h_rowK + numNodes, 0u);
            }
        }

        MPI_Bcast(h_rowK, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpyAsync(d_rowK, h_rowK, numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

        if (localElems > 0) {
            fw_update_kernel<<<grid, block, 0, stream>>>(d_dist, d_path, d_rowK, n, k, lrows);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto computeEnd = std::chrono::high_resolution_clock::now();
    const double computeSec = std::chrono::duration<double>(computeEnd - computeStart).count();

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpyAsync(distLocal.data(), d_dist, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (needFullGather) {
        if (rank == 0) {
            distGlobal.resize(numNodes * numNodes);
        }
        MPI_Gatherv(distLocal.data(), static_cast<int>(localElems), MPI_UNSIGNED,
                    rank == 0 ? distGlobal.data() : nullptr,
                    rank == 0 ? sendcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            // pathGlobal is unused for output but keep semantics (size) consistent.
            pathGlobal.resize(numNodes * numNodes);
        }
    }

    CUDA_CHECK(cudaFreeHost(h_rowK));
    if (d_rowK) CUDA_CHECK(cudaFree(d_rowK));
    if (d_path) CUDA_CHECK(cudaFree(d_path));
    if (d_dist) CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return computeSec;
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
            numNodes = static_cast<size_t>(atoi(argv[++i]));
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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0 allocates & initializes full distance matrix to preserve original semantics.
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Gather is only needed when printing/validating (all ranks must participate).
    const bool needGather = (validate || printResults);
    const double localSec = floydWarshallHybridMPI_OMP_CUDA(dist, path, numNodes, needGather);
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxSec * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Floyd-Warshall has O(n^3) operations.
        const double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) * static_cast<double>(numNodes);
        const double gops = ops / maxSec / 1e9;
        printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
