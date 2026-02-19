#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static inline void mpi_abort_with_cuda_error(const char* msg, cudaError_t err) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            mpi_abort_with_cuda_error(#call, _e); \
        } \
    } while (0)

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
    // Equivalent to the original initialization (path[i][j] = j) but without redundant writes.
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx2(j, i, numNodes)] = static_cast<unsigned int>(j);
        }
    }
}

__global__ void fw_update_kernel(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ rowk,
                                int n,
                                int localRows,
                                int k) {
    const int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= localRows || j >= n) return;

    const int base = i * n;
    const unsigned int distIJ = dist[base + j];
    const unsigned int distIK = dist[base + k];
    const unsigned int distKJ = rowk[j];

    const unsigned int newDist = distIK + distKJ;
    if (newDist < distIJ) {
        dist[base + j] = newDist;
        path[base + j] = (unsigned int)k;
    }
}

static inline void compute_row_partition(int n, int worldSize,
                                        std::vector<int>& rowsPerRank,
                                        std::vector<int>& rowDispls) {
    rowsPerRank.assign(worldSize, 0);
    rowDispls.assign(worldSize, 0);

    const int base = (worldSize > 0) ? (n / worldSize) : 0;
    const int rem = (worldSize > 0) ? (n % worldSize) : 0;

    int disp = 0;
    for (int r = 0; r < worldSize; ++r) {
        rowsPerRank[r] = base + (r < rem ? 1 : 0);
        rowDispls[r] = disp;
        disp += rowsPerRank[r];
    }
}

static inline int owner_of_row(int k, int n, int worldSize) {
    const int base = (worldSize > 0) ? (n / worldSize) : 0;
    const int rem = (worldSize > 0) ? (n % worldSize) : 0;

    // First 'rem' ranks have (base+1) rows, remaining have base rows.
    const int cutoff = (base + 1) * rem;
    if (k < cutoff) {
        return (base + 1) ? (k / (base + 1)) : 0;
    }
    // If base==0 then cutoff==n and we never reach here.
    return rem + (k - cutoff) / base;
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

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    uint64_t numNodesU64 = 512;
    int validate = 0;
    int printResults = 0;

    int earlyExit = 0;
    int earlyCode = 0;

    if (worldRank == 0) {
        // Parse command line arguments (rank 0 only)
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodesU64 = (uint64_t)atoll(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                earlyExit = 1;
                earlyCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                earlyExit = 1;
                earlyCode = 1;
                break;
            }
        }

        if (numNodesU64 == 0) {
            printf("Number of nodes must be > 0\n");
            earlyExit = 1;
            earlyCode = 1;
        }
    }

    MPI_Bcast(&earlyExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&earlyCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (earlyExit) {
        MPI_Finalize();
        return earlyCode;
    }

    MPI_Bcast(&numNodesU64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t numNodes = (size_t)numNodesU64;
    if (numNodes > (size_t)INT32_MAX) {
        if (worldRank == 0) {
            printf("numNodes too large for this implementation\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int n = (int)numNodes;

    if (worldRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", worldSize);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row partitioning for all ranks.
    std::vector<int> rowsPerRank;
    std::vector<int> rowDispls;
    compute_row_partition(n, worldSize, rowsPerRank, rowDispls);

    const int localRows = rowsPerRank[worldRank];
    const int localRow0 = rowDispls[worldRank];
    const int localElems = localRows * n;

    std::vector<int> sendcounts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        sendcounts[r] = rowsPerRank[r] * n;
        displs[r] = rowDispls[r] * n;
    }

    std::vector<unsigned int> distGlobal;
    std::vector<unsigned int> pathGlobal;

    if (worldRank == 0) {
        // Allocate and initialize full matrices on rank 0 to preserve the original RNG semantics.
        distGlobal.resize(numNodes * numNodes);
        pathGlobal.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(distGlobal, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(pathGlobal, numNodes);
    }

    std::vector<unsigned int> distLocal((size_t)localElems);
    std::vector<unsigned int> pathLocal((size_t)localElems);

    MPI_Scatterv(worldRank == 0 ? distGlobal.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 distLocal.data(), localElems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(worldRank == 0 ? pathGlobal.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 pathLocal.data(), localElems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // CUDA setup (unconditional)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (worldRank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = worldRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_rowk = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_rowk, (size_t)n * sizeof(unsigned int)));

    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_dist, (size_t)localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc((void**)&d_path, (size_t)localElems * sizeof(unsigned int)));

        CUDA_CHECK(cudaMemcpyAsync(d_dist, distLocal.data(), (size_t)localElems * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_path, pathLocal.data(), (size_t)localElems * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<unsigned int> rowkHost((size_t)n);

    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    dim3 block(32, 8);
    dim3 grid((n + (int)block.x - 1) / (int)block.x,
              (localRows + (int)block.y - 1) / (int)block.y);

    for (int k = 0; k < n; ++k) {
        const int owner = owner_of_row(k, n, worldSize);

        if (worldRank == owner) {
            const int localK = k - localRow0;
            // Copy the current pivot row from device to host for broadcast.
            CUDA_CHECK(cudaMemcpyAsync(rowkHost.data(), d_dist + (size_t)localK * (size_t)n,
                                       (size_t)n * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        MPI_Bcast(rowkHost.data(), n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpyAsync(d_rowk, rowkHost.data(), (size_t)n * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

        if (localElems > 0) {
            fw_update_kernel<<<grid, block, 0, stream>>>(d_dist, d_path, d_rowk, n, localRows, k);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    const double t1 = MPI_Wtime();
    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy results back
    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpyAsync(distLocal.data(), d_dist, (size_t)localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(pathLocal.data(), d_path, (size_t)localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (worldRank == 0) {
        distGlobal.assign(numNodes * numNodes, 0u);
        pathGlobal.assign(numNodes * numNodes, 0u);
    }

    MPI_Gatherv(distLocal.data(), localElems, MPI_UNSIGNED,
                worldRank == 0 ? distGlobal.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(pathLocal.data(), localElems, MPI_UNSIGNED,
                worldRank == 0 ? pathGlobal.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        const long ms = (long)llround(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        double ops = (double)numNodes * (double)numNodes * (double)numNodes;
        double gops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(distGlobal, "DistanceMatrix");
        }
    }

    int rc = 0;
    if (validate && worldRank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(distGlobal, numNodes);
        if (valid) {
            printf("Validation: PASSED\n");
            rc = 0;
        } else {
            printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup
    if (d_dist) CUDA_CHECK(cudaFree(d_dist));
    if (d_path) CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_rowk));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return rc;
}
