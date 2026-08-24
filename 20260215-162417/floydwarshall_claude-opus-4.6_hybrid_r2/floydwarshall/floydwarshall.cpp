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

// Column-major index: idx2(i,j,n) = j*n + i
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: for a given k, update all (i,j) pairs in the local row block.
// d_dist is the local block (numCols columns, localRows rows) in column-major order.
// d_krow is row k (length numCols): krow[j] = dist[idx2(j, k, n)] for global matrix.
// d_kcol is the local portion of column k (length localRows): kcol[li] = dist[idx2(k, globalRowStart+li, n)].
// d_path is the local path block.
__global__ void floydWarshallKernel(unsigned int* __restrict__ d_dist,
                                    unsigned int* __restrict__ d_path,
                                    const unsigned int* __restrict__ d_krow,
                                    const unsigned int* __restrict__ d_kcol,
                                    const int localRows, const int numCols,
                                    const unsigned int k) {
    // 2D grid: x = local row index (i), y = column index (j)
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    int j  = blockIdx.y * blockDim.y + threadIdx.y;
    if (li >= localRows || j >= numCols) return;

    unsigned int distIK = d_kcol[li];     // dist[k][globalRow_i]
    unsigned int distKJ = d_krow[j];      // dist[j][k]
    unsigned int newDist = distIK + distKJ;

    int idx = j * localRows + li;  // column-major in local block
    unsigned int distIJ = d_dist[idx];

    if (newDist < distIJ) {
        d_dist[idx] = newDist;
        d_path[idx] = k;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

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

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

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

    const int N = static_cast<int>(numNodes);

    // Assign GPU: each rank uses rank % numDevices
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per rank: 1, OpenMP threads: %d\n",
               nprocs, omp_get_max_threads());
    }

    // Distribute rows across MPI ranks (row = i dimension)
    // Each rank gets a contiguous block of rows
    int baseRows = N / nprocs;
    int extraRows = N % nprocs;
    int localRows = baseRows + (rank < extraRows ? 1 : 0);
    int rowStart = rank * baseRows + std::min(rank, extraRows);

    // Compute sendcounts and displacements for Scatterv/Gatherv
    // The matrix is column-major: column j is dist[j*N .. j*N + N-1]
    // We distribute rows, so for each column we send a slice of localRows elements.
    // We'll pack as local column-major: localDist[j*localRows + li] = dist[j*N + rowStart + li]

    // Full matrix on rank 0
    std::vector<unsigned int> dist, path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Local matrices in column-major with localRows rows
    std::vector<unsigned int> localDist(static_cast<size_t>(localRows) * N);
    std::vector<unsigned int> localPath(static_cast<size_t>(localRows) * N);

    // Scatter: distribute rows for each column
    // We need per-rank row counts and displacements
    std::vector<int> rowCounts(nprocs), rowDispls(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        rowCounts[r] = baseRows + (r < extraRows ? 1 : 0);
        rowDispls[r] = r * baseRows + std::min(r, extraRows);
    }

    // For column-major layout, scatter each column's row slice
    // Create an MPI type for sending a column slice from the full matrix
    // Full column in global matrix: N contiguous unsigned ints starting at col*N
    // Each rank gets rowCounts[r] elements starting at col*N + rowDispls[r]
    // We'll do N scatters (one per column) or use a derived type.
    // More efficient: use MPI_Type_vector for the full local block.

    // Simple approach: scatter column by column (N scatters)
    // This is fine since N is typically moderate (512-4096)
    if (rank == 0) printf("Computing shortest paths...\n");

    for (int j = 0; j < N; ++j) {
        unsigned int* colBase = (rank == 0) ? dist.data() + (size_t)j * N : nullptr;
        MPI_Scatterv(colBase, rowCounts.data(), rowDispls.data(), MPI_UNSIGNED,
                     localDist.data() + (size_t)j * localRows, localRows, MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }
    for (int j = 0; j < N; ++j) {
        unsigned int* colBase = (rank == 0) ? path.data() + (size_t)j * N : nullptr;
        MPI_Scatterv(colBase, rowCounts.data(), rowDispls.data(), MPI_UNSIGNED,
                     localPath.data() + (size_t)j * localRows, localRows, MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }

    // Allocate GPU memory for local block
    unsigned int *d_dist, *d_path, *d_krow, *d_kcol;
    size_t localSize = (size_t)localRows * N * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_dist, localSize));
    CUDA_CHECK(cudaMalloc(&d_path, localSize));
    CUDA_CHECK(cudaMalloc(&d_krow, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_kcol, localRows * sizeof(unsigned int)));

    // Copy local data to GPU
    CUDA_CHECK(cudaMemcpy(d_dist, localDist.data(), localSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, localPath.data(), localSize, cudaMemcpyHostToDevice));

    // Prepare row k buffer (full row k of dist, length N)
    std::vector<unsigned int> krow(N);
    // Local column k buffer (localRows entries from column k of local dist)
    std::vector<unsigned int> kcol(localRows);

    // CUDA kernel launch config
    dim3 blockDim(16, 16);
    dim3 gridDim((localRows + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < N; ++k) {
        // Determine which rank owns row k
        int kOwner, kLocalIdx;
        if (k < extraRows * (baseRows + 1)) {
            kOwner = k / (baseRows + 1);
            kLocalIdx = k % (baseRows + 1);
        } else {
            int adjusted = k - extraRows * (baseRows + 1);
            kOwner = extraRows + adjusted / baseRows;
            kLocalIdx = adjusted % baseRows;
        }

        // Row k of the dist matrix: dist[idx2(j, k, N)] for j=0..N-1
        // In column-major: element (k, j) = dist[j*N + k]
        // In local storage on kOwner: localDist[j*localRows_owner + kLocalIdx]

        if (rank == kOwner) {
            // Extract row k from GPU local data
            // Row k is at local index kLocalIdx: for each column j,
            // element is at d_dist[j * localRows + kLocalIdx]
            // Use cudaMemcpy2D to gather the row
            CUDA_CHECK(cudaMemcpy2D(krow.data(), sizeof(unsigned int),
                                     d_dist + kLocalIdx, localRows * sizeof(unsigned int),
                                     sizeof(unsigned int), N,
                                     cudaMemcpyDeviceToHost));
        }

        // Broadcast row k to all ranks
        MPI_Bcast(krow.data(), N, MPI_UNSIGNED, kOwner, MPI_COMM_WORLD);

        // Extract local column k (column k in local block = d_dist[k_col_in_local * localRows .. +localRows])
        // Column k of the global matrix: all (k, i) for i in local rows
        // In local storage: column index = k, so localDist[k * localRows .. k*localRows + localRows - 1]
        CUDA_CHECK(cudaMemcpy(kcol.data(), d_dist + (size_t)k * localRows,
                              localRows * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Upload krow and kcol to GPU
        CUDA_CHECK(cudaMemcpy(d_krow, krow.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kcol, kcol.data(), localRows * sizeof(unsigned int), cudaMemcpyHostToDevice));

        // Launch kernel
        floydWarshallKernel<<<gridDim, blockDim>>>(d_dist, d_path, d_krow, d_kcol,
                                                    localRows, N, k);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double localDurationMs = std::chrono::duration<double, std::milli>(end - start).count();
    double durationMs = 0.0;
    MPI_Reduce(&localDurationMs, &durationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy results back from GPU
    CUDA_CHECK(cudaMemcpy(localDist.data(), d_dist, localSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(localPath.data(), d_path, localSize, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_krow));
    CUDA_CHECK(cudaFree(d_kcol));

    // Gather results back to rank 0
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
    }
    for (int j = 0; j < N; ++j) {
        unsigned int* colBase = (rank == 0) ? dist.data() + (size_t)j * N : nullptr;
        MPI_Gatherv(localDist.data() + (size_t)j * localRows, localRows, MPI_UNSIGNED,
                    colBase, rowCounts.data(), rowDispls.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }
    for (int j = 0; j < N; ++j) {
        unsigned int* colBase = (rank == 0) ? path.data() + (size_t)j * N : nullptr;
        MPI_Gatherv(localPath.data() + (size_t)j * localRows, localRows, MPI_UNSIGNED,
                    colBase, rowCounts.data(), rowDispls.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", durationMs);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
