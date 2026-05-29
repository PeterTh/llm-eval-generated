#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array (column-major: element (i,j) at j*n+i)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// CUDA kernel: parallel Floyd-Warshall inner loop
// Each thread (local_i, j) updates dist[local_i][j] using row k:
//   dist[local_i][j] = min(dist[local_i][j], dist[local_i][k] + dist[k][j])
// Data is stored in row-major order on the GPU.
__global__ void fw_kernel(unsigned int* dist, unsigned int* path,
                          const unsigned int* row_k,
                          size_t N, size_t num_local_rows, unsigned int k) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_i >= num_local_rows || j >= N) return;

    size_t idx = local_i * N + j;
    unsigned int newDist = dist[local_i * N + k] + row_k[j];

    if (newDist < dist[idx]) {
        dist[idx] = newDist;
        path[idx] = k;
    }
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

// Hybrid MPI + CUDA Floyd-Warshall
// - MPI: row-wise block distribution across ranks
// - CUDA: GPU kernel for the inner i,j update loop
// - OpenMP: parallel data layout conversion
void floydWarshallParallel(unsigned int* local_dist, unsigned int* local_path,
                           size_t numNodes, int rank, int numRanks,
                           size_t localStart, size_t localRows) {
    // All ranks must participate in the collective Bcast below, even if they own no rows.
    std::vector<unsigned int> row_k_buf(numNodes);

    unsigned int *d_dist = nullptr, *d_path = nullptr, *d_row_k = nullptr;
    bool has_gpu_work = (localRows > 0);

    if (has_gpu_work) {
        cudaMalloc(&d_dist, localRows * numNodes * sizeof(unsigned int));
        cudaMalloc(&d_path, localRows * numNodes * sizeof(unsigned int));
        cudaMalloc(&d_row_k, numNodes * sizeof(unsigned int));

        // Upload initial data to GPU
        cudaMemcpy(d_dist, local_dist, localRows * numNodes * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_path, local_path, localRows * numNodes * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
    }

    dim3 block(16, 16);
    dim3 grid((numNodes + block.x - 1) / block.x,
              has_gpu_work ? (localRows + block.y - 1) / block.y : 1);

    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns row k
        int owner_rank = static_cast<int>(k * numRanks / numNodes);

        if (owner_rank == rank) {
            if (has_gpu_work) {
                size_t local_k = k - localStart;
                // Read row k from GPU
                cudaMemcpy(row_k_buf.data(), d_dist + local_k * numNodes,
                           numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            } else {
                // This rank has no GPU rows but is the owner — shouldn't happen
                // with the distribution scheme, but handle gracefully.
            }
        }

        // Broadcast row k to all ranks (blocking collective)
        MPI_Bcast(row_k_buf.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner_rank, MPI_COMM_WORLD);

        if (has_gpu_work) {
            // Upload row_k to GPU
            cudaMemcpy(d_row_k, row_k_buf.data(), numNodes * sizeof(unsigned int),
                       cudaMemcpyHostToDevice);

            // Launch CUDA kernel for this k-iteration
            fw_kernel<<<grid, block>>>(d_dist, d_path, d_row_k,
                                       numNodes, localRows, static_cast<unsigned int>(k));
            cudaDeviceSynchronize();
        }
    }

    if (has_gpu_work) {
        // Download final results from GPU
        cudaMemcpy(local_dist, d_dist, localRows * numNodes * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(local_path, d_path, localRows * numNodes * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_dist);
        cudaFree(d_path);
        cudaFree(d_row_k);
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

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
    int rank, numRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    // Compute row-wise block distribution across MPI ranks
    size_t baseRows = numNodes / numRanks;
    size_t extraRows = numNodes % numRanks;
    size_t localStart = baseRows * rank + std::min(static_cast<size_t>(rank), extraRows);
    size_t localRows = baseRows + (rank < static_cast<int>(extraRows) ? 1 : 0);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize full matrices (column-major, same as original)
    std::vector<unsigned int> fullDist(numNodes * numNodes);
    std::vector<unsigned int> fullPath(numNodes * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(fullPath, numNodes);

    // Extract local rows to row-major layout for GPU (OpenMP parallelized)
    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);

    #pragma omp parallel for collapse(2)
    for (size_t local_i = 0; local_i < localRows; ++local_i) {
        for (size_t j = 0; j < numNodes; ++j) {
            localDist[local_i * numNodes + j] =
                fullDist[idx2(localStart + local_i, j, numNodes)];
            localPath[local_i * numNodes + j] =
                fullPath[idx2(localStart + local_i, j, numNodes)];
        }
    }

    // Run hybrid MPI + CUDA Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallParallel(localDist.data(), localPath.data(),
                          numNodes, rank, numRanks, localStart, localRows);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather results back to full column-major matrix
    // Build send/recv counts and displacements for MPI_Allgatherv
    std::vector<int> recvcounts(numRanks), displs(numRanks);
    size_t disp = 0;
    for (int r = 0; r < numRanks; ++r) {
        size_t rBase = numNodes / numRanks;
        size_t rExtra = numNodes % numRanks;
        size_t rRows = rBase + (r < static_cast<int>(rExtra) ? 1 : 0);
        recvcounts[r] = static_cast<int>(rRows * numNodes);
        displs[r] = static_cast<int>(disp);
        disp += rRows * numNodes;
    }

    // Gather into a temporary row-major buffer
    std::vector<unsigned int> gatheredDist(numNodes * numNodes);
    std::vector<unsigned int> gatheredPath(numNodes * numNodes);

    MPI_Allgatherv(localDist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                   gatheredDist.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(localPath.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                   gatheredPath.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                   MPI_COMM_WORLD);

    // Convert from gathered row-major back to column-major (OpenMP parallelized)
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            fullDist[idx2(i, j, numNodes)] = gatheredDist[i * numNodes + j];
            fullPath[idx2(i, j, numNodes)] = gatheredPath[i * numNodes + j];
        }
    }

    // Performance reporting
    double ops = static_cast<double>(numNodes) * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    // Validation (rank 0 only)
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
