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

// Index calculation for flattened 2D array (preserved for API compat)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Extract column k from the local (row-major) dist sub-matrix on GPU.
// Each thread gathers one element:  colOut[i] = dist[i * n + k]
__global__ void extractColumnKernel(
    const unsigned int* __restrict__ dist,
    unsigned int* __restrict__ colOut,
    const size_t n,
    const size_t localRowCount,
    const size_t k)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < localRowCount) {
        colOut[i] = dist[i * n + k];
    }
}

// Floyd-Warshall inner i×j update for a fixed intermediate node k.
//   dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
//   path[i][j] = k  (when updated)
// rowK[j]  == dist[k][j]   (full row k, broadcast to every rank)
// colK[i]  == dist[i][k]   (full column k, all-gathered from every rank)
// dist / path are the local sub-matrices (rows localRowStart..localRowStart+localRowCount-1).
__global__ void floydWarshallKernel(
    const unsigned int* __restrict__ rowK,
    const unsigned int* __restrict__ colK,
    unsigned int* __restrict__ dist,
    unsigned int* __restrict__ path,
    const size_t n,
    const size_t k,
    const size_t localRowCount,
    const size_t localRowStart)
{
    const size_t j         = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i_local   = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_local >= localRowCount || j >= n) return;

    const size_t i_global  = localRowStart + i_local;
    const size_t idx       = i_local * n + j;  // local offset

    const unsigned int distIJ = dist[idx];
    const unsigned int distIK = colK[i_global];  // dist[i_global][k]
    const unsigned int distKJ = rowK[j];          // dist[k][j]

    const unsigned int newDist = distIK + distKJ;

    if (newDist < distIJ) {
        dist[idx] = newDist;
        path[idx] = static_cast<unsigned int>(k);
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main – hybrid MPI / OpenMP / CUDA Floyd-Warshall
// ---------------------------------------------------------------------------
// Decomposition: 1-D block partition by rows across MPI ranks.
//   Each rank owns a contiguous set of rows of the N×N matrix.
//   The outermost k-loop is sequential (data dependency).
//   Within each k-step:
//     1. Extract column-k from GPU → host (small transfer)
//     2. MPI_Allgatherv  → every rank has the full column-k
//     3. Extract row-k   from GPU → host (contiguous cudaMemcpy)
//     4. MPI_Bcast       → every rank has the full row-k
//     5. Copy row-k / column-k to GPU
//     6. CUDA kernel     → update local sub-matrix
//   The matrix stays on GPU throughout the k-loop; only thin slices
//   cross the PCIe bus each iteration.
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int mpiRank, mpiSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // ---- argument parsing (all ranks) ------------------------------------
    size_t numNodes    = 512;
    bool   validate    = false;
    bool   printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (mpiRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", mpiSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- row decomposition -----------------------------------------------
    const size_t rowsPerRank = numNodes / static_cast<size_t>(mpiSize);
    const size_t extraRows   = numNodes % static_cast<size_t>(mpiSize);

    std::vector<size_t> rowStarts(mpiSize);
    std::vector<int>    rowCounts(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        rowCounts[r] = static_cast<int>(rowsPerRank + (r < static_cast<int>(extraRows) ? 1 : 0));
        rowStarts[r] = (r == 0) ? 0 : rowStarts[r - 1] + rowCounts[r - 1];
    }

    const size_t localRowCount = rowCounts[mpiRank];
    const size_t localRowStart = rowStarts[mpiRank];
    const size_t localSize     = localRowCount * numNodes;

    // Pre-compute which rank owns each row k.
    std::vector<int> rankOfK(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        for (int r = 0; r < mpiSize; ++r) {
            if (k >= rowStarts[r] && k < rowStarts[r] + rowCounts[r]) {
                rankOfK[k] = r;
                break;
            }
        }
    }

    // Scatter / gather parameters (counts in elements).
    std::vector<int> sendCounts(mpiSize);
    std::vector<int> displs(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        sendCounts[r] = rowCounts[r] * static_cast<int>(numNodes);
        displs[r]     = static_cast<int>(rowStarts[r] * numNodes);
    }

    // Column-Allgatherv parameters (one element per row).
    std::vector<int> colSendCounts(mpiSize);
    std::vector<int> colDispls(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        colSendCounts[r] = rowCounts[r];
        colDispls[r]     = static_cast<int>(rowStarts[r]);
    }

    // ---- initialise on rank 0 --------------------------------------------
    std::vector<unsigned int> fullDist;
    std::vector<unsigned int> fullPath;

    if (mpiRank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);

        printf("Initializing graph...\n");

        // Distance matrix – sequential PRNG for reproducibility.
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE - 1) + 1.0;
        for (size_t i = 0; i < numNodes * numNodes; ++i) {
            fullDist[i] = 1 + static_cast<unsigned int>(
                range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        }
        for (size_t i = 0; i < numNodes; ++i) {
            fullDist[i * numNodes + i] = 0;
        }

        // Path matrix – parallelised with OpenMP.
        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < numNodes; ++j) {
            for (size_t i = 0; i < numNodes; ++i) {
                fullPath[j * numNodes + i] = static_cast<unsigned int>(j);
                fullPath[i * numNodes + j] = static_cast<unsigned int>(i);
            }
        }
    }

    // ---- scatter to all ranks --------------------------------------------
    std::vector<unsigned int> localDist(localSize);
    std::vector<unsigned int> localPath(localSize);

    MPI_Scatterv(fullDist.data(), sendCounts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localSize), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullPath.data(), sendCounts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localSize), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // ---- GPU allocation --------------------------------------------------
    unsigned int* d_dist     = nullptr;
    unsigned int* d_path     = nullptr;
    unsigned int* d_rowK     = nullptr;
    unsigned int* d_colK     = nullptr;
    unsigned int* d_colLocal = nullptr;

    cudaMalloc(&d_dist,     localSize * sizeof(unsigned int));
    cudaMalloc(&d_path,     localSize * sizeof(unsigned int));
    cudaMalloc(&d_rowK,     numNodes * sizeof(unsigned int));
    cudaMalloc(&d_colK,     numNodes * sizeof(unsigned int));
    cudaMalloc(&d_colLocal, std::max(localRowCount, static_cast<size_t>(1)) *
                            sizeof(unsigned int));

    cudaMemcpy(d_dist, localDist.data(), localSize * sizeof(unsigned int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, localPath.data(), localSize * sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    // Host buffers for communication.
    std::vector<unsigned int> rowK(numNodes);
    std::vector<unsigned int> colK(numNodes);
    std::vector<unsigned int> colKLocal(localRowCount);

    // CUDA kernel configuration.
    const dim3 blockDim(16, 16);
    const dim3 gridDim(
        (numNodes + blockDim.x - 1) / blockDim.x,
        (localRowCount + blockDim.y - 1) / blockDim.y
    );

    // ---- main k-loop -----------------------------------------------------
    if (mpiRank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < numNodes; ++k) {
        // 1. Extract column k from GPU → small host buffer.
        if (localRowCount > 0) {
            const int colBS = 256;
            const int colGS = static_cast<int>(
                (localRowCount + static_cast<size_t>(colBS) - 1) / static_cast<size_t>(colBS));
            extractColumnKernel<<<colGS, colBS>>>(
                d_dist, d_colLocal, numNodes, localRowCount, k);
            cudaMemcpy(colKLocal.data(), d_colLocal,
                       localRowCount * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost);
        }

        // 2. All-gather column k across MPI ranks.
        MPI_Allgatherv(colKLocal.data(),
                       static_cast<int>(localRowCount), MPI_UNSIGNED,
                       colK.data(), colSendCounts.data(), colDispls.data(),
                       MPI_UNSIGNED, MPI_COMM_WORLD);

        // 3. Upload full column k to GPU.
        cudaMemcpy(d_colK, colK.data(),
                   numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);

        // 4. Extract row k from GPU (only on the owning rank).
        const int myRankOfK = rankOfK[k];
        if (mpiRank == myRankOfK) {
            const size_t localK = k - localRowStart;
            cudaMemcpy(rowK.data(),
                       d_dist + localK * numNodes,
                       numNodes * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost);
        }

        // 5. Broadcast row k to all ranks.
        MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  myRankOfK, MPI_COMM_WORLD);

        // 6. Upload full row k to GPU.
        cudaMemcpy(d_rowK, rowK.data(),
                   numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);

        // 7. Launch Floyd-Warshall kernel on local sub-matrix.
        if (localRowCount > 0) {
            floydWarshallKernel<<<gridDim, blockDim>>>(
                d_rowK, d_colK, d_dist, d_path,
                numNodes, k, localRowCount, localRowStart);
        }
    }

    // Ensure all GPU work is complete before timing stop.
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ---- copy results back from GPU --------------------------------------
    cudaMemcpy(localDist.data(), d_dist, localSize * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(localPath.data(), d_path, localSize * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    // ---- gather to rank 0 ------------------------------------------------
    if (mpiRank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
    }

    MPI_Gatherv(localDist.data(), static_cast<int>(localSize), MPI_UNSIGNED,
                fullDist.data(), sendCounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), static_cast<int>(localSize), MPI_UNSIGNED,
                fullPath.data(), sendCounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // ---- cleanup GPU -----------------------------------------------------
    cudaFree(d_dist);
    cudaFree(d_path);
    cudaFree(d_rowK);
    cudaFree(d_colK);
    cudaFree(d_colLocal);

    // ---- rank-0 output / validation --------------------------------------
    if (mpiRank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = true;

            // 1. Diagonal must be zero.
            for (size_t i = 0; i < numNodes; ++i) {
                if (fullDist[i * numNodes + i] != 0) {
                    printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                    valid = false;
                }
            }

            // 2. Triangle inequality (sampled).
            for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
                for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
                    for (size_t kk = 0; kk < numNodes; ++kk) {
                        const unsigned int distIJ = fullDist[i * numNodes + j];
                        const unsigned int distIK = fullDist[i * numNodes + kk];
                        const unsigned int distKJ = fullDist[kk * numNodes + j];

                        if (distIK < INF && distKJ < INF) {
                            if (distIK + distKJ < distIJ) {
                                printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                       i, j, kk);
                                valid = false;
                            }
                        }
                    }
                }
            }

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
