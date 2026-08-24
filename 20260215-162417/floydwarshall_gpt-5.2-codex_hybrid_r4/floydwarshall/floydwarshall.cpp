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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

struct RowPartition {
    size_t local_rows;
    size_t row_offset;
};

RowPartition getPartition(const size_t numNodes, const int rank, const int size) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t local_rows = base + ((static_cast<size_t>(rank) < rem) ? 1 : 0);
    const size_t row_offset = base * static_cast<size_t>(rank) + std::min(rem, static_cast<size_t>(rank));
    return {local_rows, row_offset};
}

int ownerOfRow(const size_t k, const size_t numNodes, const int size) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t threshold = (base + 1) * rem;
    if (k < threshold) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(rem + (k - threshold) / base);
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

void initializePathMatrixLocal(std::vector<unsigned int>& path, const size_t numNodes, const size_t localRows) {
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < localRows; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[j * numNodes + i] = static_cast<unsigned int>(i);
        }
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

__global__ void floydWarshallKernel(unsigned int* dist,
                                    unsigned int* path,
                                    const unsigned int* row_k,
                                    const size_t numNodes,
                                    const size_t localRows,
                                    const size_t k) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t j = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (i < numNodes && j < localRows) {
        const size_t idx = j * numNodes + i;
        const unsigned int distIJ = dist[idx];
        const unsigned int distIK = row_k[i];
        const unsigned int distKJ = dist[j * numNodes + k];
        const unsigned int newDist = distIK + distKJ;

        if (newDist < distIJ) {
            dist[idx] = newDist;
            path[idx] = static_cast<unsigned int>(k);
        }
    }
}

void floydWarshallHybrid(std::vector<unsigned int>& dist,
                         std::vector<unsigned int>& path,
                         const size_t numNodes,
                         const size_t localRows,
                         const size_t rowOffset,
                         const int rank,
                         const int size) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    std::vector<unsigned int> row_k(numNodes);

    if (localRows == 0) {
        for (size_t k = 0; k < numNodes; ++k) {
            const int owner = ownerOfRow(k, numNodes, size);
            MPI_Bcast(row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        }
        return;
    }

    const size_t bytes = localRows * numNodes * sizeof(unsigned int);
    const size_t row_bytes = numNodes * sizeof(unsigned int);

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_row_k = nullptr;

    CUDA_CHECK(cudaMalloc(&d_dist, bytes));
    CUDA_CHECK(cudaMalloc(&d_path, bytes));
    CUDA_CHECK(cudaMalloc(&d_row_k, row_bytes));

    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((numNodes + block.x - 1) / block.x, (localRows + block.y - 1) / block.y);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, numNodes, size);
        if (rank == owner) {
            const size_t local_k = k - rowOffset;
            CUDA_CHECK(cudaMemcpy(row_k.data(), d_dist + local_k * numNodes, row_bytes, cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_row_k, row_k.data(), row_bytes, cudaMemcpyHostToDevice));

        floydWarshallKernel<<<grid, block>>>(d_dist, d_path, d_row_k, numNodes, localRows, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_row_k));
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

    int rank = 0;
    int size = 1;
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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const RowPartition partition = getPartition(numNodes, rank, size);

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    for (int r = 0; r < size; ++r) {
        const RowPartition part = getPartition(numNodes, r, size);
        counts[r] = static_cast<int>(part.local_rows * numNodes);
        displs[r] = static_cast<int>(part.row_offset * numNodes);
    }

    std::vector<unsigned int> dist(partition.local_rows * numNodes);
    std::vector<unsigned int> path(partition.local_rows * numNodes);

    std::vector<unsigned int> dist_full;
    if (rank == 0) {
        printf("Initializing graph...\n");
        dist_full.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist_full, numNodes, 1, MAX_DISTANCE);
    }

    MPI_Scatterv(rank == 0 ? dist_full.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_UNSIGNED,
                 dist.data(),
                 counts[rank],
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    initializePathMatrixLocal(path, numNodes, partition.local_rows);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallHybrid(dist, path, numNodes, partition.local_rows, partition.row_offset, rank, size);

    const double end = MPI_Wtime();
    const double local_duration = end - start;
    double global_duration = 0.0;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double duration_ms = global_duration * 1000.0;
        printf("Computation time: %.3f ms\n", duration_ms);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / global_duration / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    const bool needGather = validate || printResults;
    if (needGather) {
        if (rank == 0 && dist_full.empty()) {
            dist_full.resize(numNodes * numNodes);
        }

        MPI_Gatherv(dist.data(),
                    counts[rank],
                    MPI_UNSIGNED,
                    rank == 0 ? dist_full.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_UNSIGNED,
                    0,
                    MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        print_results_int(dist_full, "DistanceMatrix");
    }

    int final_status = 0;
    if (validate) {
        int valid_flag = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            valid_flag = validateResult(dist_full, numNodes) ? 1 : 0;
            printf("Validation: %s\n", valid_flag ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        final_status = valid_flag ? 0 : 1;
    }

    MPI_Finalize();
    return final_status;
}
