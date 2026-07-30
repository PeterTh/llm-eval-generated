#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// ─── CUDA error checking macro ───────────────────────────────────────────────
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[MPI rank %d] CUDA error at %s:%d: %s\n", \
                myRank, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

static int myRank; // used by CUDA_CHECK

// ─── Index helper: element at (row, col) in column-major layout ──────────────
// Original: idx2(i, j, n) = j * n + i  (col-major: column j, row i)
// Local storage for this rank: columns [colStart, colStart+numLocalCols)
// local_col = 0 corresponds to global column colStart
inline __host__ __device__ constexpr size_t localIdx(size_t row, size_t localCol, size_t n) noexcept {
    return localCol * n + row;
}

// ─── Column distribution helpers ─────────────────────────────────────────────
// Each rank gets either baseCols or baseCols+1 columns.
inline void columnDistribution(size_t n, int numRanks, int rank,
                               size_t& colStart, size_t& numLocalCols) {
    size_t base = n / numRanks;
    size_t rem = n % numRanks;
    numLocalCols = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    colStart = rank * base + std::min(static_cast<size_t>(rank), rem);
}

inline int columnOwner(size_t col, size_t n, int numRanks) {
    size_t base = n / numRanks;
    size_t rem = n % numRanks;
    if (col < (base + 1) * rem) {
        return static_cast<int>(col / (base + 1));
    } else {
        return static_cast<int>(rem + (col - rem * (base + 1)) / base);
    }
}

// ─── CUDA kernel: one Floyd-Warshall step for local columns ─────────────────
// For a fixed intermediate node k, each thread processes one element (row, col)
// where col is a local column.  Column k is treated as read‑only.
__global__ void floydStepKernel(unsigned int* __restrict__ d_dist,
                                unsigned int* __restrict__ d_path,
                                const unsigned int* __restrict__ d_colK,
                                size_t n, unsigned int k,
                                size_t colStart, size_t numLocalCols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t localCol = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || localCol >= numLocalCols) return;

    size_t col = colStart + localCol;
    if (col == k) return; // Column k is read-only during this iteration

    // dist[row][k] from the broadcast column-k buffer
    unsigned int distIK = d_colK[row];
    if (distIK >= INF) return;

    // dist[k][col] – strided read from the local column data
    unsigned int distKJ = d_dist[localIdx(k, localCol, n)];
    if (distKJ >= INF) return;

    unsigned int newDist = distIK + distKJ;

    // dist[row][col] – coalesced read/write inside the local column
    unsigned int* ptr = &d_dist[localIdx(row, localCol, n)];
    if (newDist < *ptr) {
        *ptr = newDist;
        d_path[localIdx(row, localCol, n)] = k;
    }
}

// ─── Initialisation: distance matrix (local columns only) ────────────────────
void initDistanceMatrix(std::vector<unsigned int>& dist,
                        size_t n, size_t colStart, size_t numLocalCols,
                        unsigned int rangeMin, unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    #pragma omp parallel
    {
        unsigned int seed = 424242 + myRank * 7919 + omp_get_thread_num() * 104729;
        #pragma omp for collapse(2)
        for (size_t lc = 0; lc < numLocalCols; ++lc) {
            for (size_t row = 0; row < n; ++row) {
                dist[localIdx(row, lc, n)] =
                    rangeMin + static_cast<unsigned int>(
                        range * rand_r(&seed) / static_cast<double>(RAND_MAX));
            }
        }
    }

    // Zero the diagonal elements that belong to this rank's columns
    for (size_t lc = 0; lc < numLocalCols; ++lc) {
        size_t col = colStart + lc;
        dist[localIdx(col, lc, n)] = 0;
    }
}

// ─── Initialisation: path matrix (local columns only) ────────────────────────
// path[i][j] = j  (the direct neighbour toward destination j)
void initPathMatrix(std::vector<unsigned int>& path,
                    size_t n, size_t colStart, size_t numLocalCols) {
    #pragma omp parallel for collapse(2)
    for (size_t lc = 0; lc < numLocalCols; ++lc) {
        for (size_t row = 0; row < n; ++row) {
            size_t col = colStart + lc;
            path[localIdx(row, lc, n)] = static_cast<unsigned int>(col);
        }
    }
}

// ─── Validation (on rank 0 with full gathered matrix) ────────────────────────
bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    // 1. Diagonal must be zero
    for (size_t i = 0; i < n; ++i) {
        if (dist[localIdx(i, i, n)] != 0) { // col == row == i
            printf("Validation FAILED: diagonal [%zu,%zu] = %u (expected 0)\n",
                   i, i, dist[localIdx(i, i, n)]);
            return false;
        }
    }

    // 2. Sample triangle-inequality check (O(n) per sample pair)
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(n, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < n; ++k) {
                unsigned int distIJ = dist[localIdx(i, j, n)];
                unsigned int distIK = dist[localIdx(i, k, n)];
                unsigned int distKJ = dist[localIdx(k, j, n)];
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation FAILED: triangle inequality at [%zu,%zu,%zu]\n",
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// ─── Usage ───────────────────────────────────────────────────────────────────
void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // ── Parse arguments on rank 0 and broadcast ──────────────────────────
    size_t numNodes = 512;
    bool   validate = false;
    bool   printResults = false;

    if (myRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    unsigned long long n_ull = static_cast<unsigned long long>(numNodes);
    MPI_Bcast(&n_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(n_ull);

    int val_int = validate ? 1 : 0;
    int print_int = printResults ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = val_int != 0;
    printResults = print_int != 0;

    // ── Column distribution ──────────────────────────────────────────────
    size_t colStart, numLocalCols;
    columnDistribution(numNodes, numRanks, myRank, colStart, numLocalCols);

    // ── CUDA device setup (round-robin across GPUs) ──────────────────────
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        fprintf(stderr, "[rank %d] No CUDA-capable device found\n", myRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev = myRank % numDevices;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    if (myRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes:   %zu\n", numNodes);
        printf("MPI ranks:         %d\n", numRanks);
        printf("CUDA device:       %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
        printf("Validation:        %s\n", validate ? "enabled" : "disabled");
        printf("Results output:    %s\n", printResults ? "enabled" : "disabled");
        fflush(stdout);
    }

    // ── Host memory allocation ───────────────────────────────────────────
    size_t localSize = numLocalCols * numNodes;
    std::vector<unsigned int> h_dist(localSize);
    std::vector<unsigned int> h_path(localSize);
    std::vector<unsigned int> h_colK(numNodes);

    // ── Initialise local columns ─────────────────────────────────────────
    if (myRank == 0) printf("Initialising graph (distributed)...\n");
    initDistanceMatrix(h_dist, numNodes, colStart, numLocalCols, 1, MAX_DISTANCE);
    initPathMatrix(h_path, numNodes, colStart, numLocalCols);

    // ── Device memory allocation ─────────────────────────────────────────
    unsigned int *d_dist  = nullptr;
    unsigned int *d_path  = nullptr;
    unsigned int *d_colK  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist,  localSize * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_path,  localSize * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_colK,  numNodes   * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), localSize * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, h_path.data(), localSize * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // ── Kernel launch configuration ──────────────────────────────────────
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;
    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((numNodes + BLOCK_X - 1) / BLOCK_X,
                 (numLocalCols + BLOCK_Y - 1) / BLOCK_Y);

    // ── Synchronise before timing ────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    if (myRank == 0) printf("Computing shortest paths (hybrid MPI+OpenMP+CUDA)...\n");
    fflush(stdout);

    auto tStart = std::chrono::high_resolution_clock::now();

    // ── Main Floyd-Warshall loop ─────────────────────────────────────────
    // Implicit CUDA stream ordering guarantees safety:
    //   cudaMemcpy on default stream is serialised after the previous kernel,
    //   and the next kernel launch is serialised after cudaMemcpy.
    for (size_t k = 0; k < numNodes; ++k) {
        int rankK = columnOwner(k, numNodes, numRanks);

        // The rank that owns column k extracts it from its GPU device buffer.
        // cudaMemcpy D2H on default stream – waits for prior kernels implicitly.
        if (myRank == rankK) {
            size_t localK = k - colStart;
            CUDA_CHECK(cudaMemcpy(h_colK.data(),
                                  d_dist + localK * numNodes,
                                  numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        // Broadcast column k to every rank
        MPI_Bcast(h_colK.data(), static_cast<int>(numNodes),
                  MPI_UNSIGNED, rankK, MPI_COMM_WORLD);

        // Copy column k to GPU (waits for prior kernel on default stream)
        CUDA_CHECK(cudaMemcpy(d_colK, h_colK.data(),
                              numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // Launch kernel – ordered after H2D on default stream
        if (numLocalCols > 0) {
            unsigned int k_u = static_cast<unsigned int>(k);
            floydStepKernel<<<gridDim, blockDim>>>(
                d_dist, d_path, d_colK,
                numNodes, k_u,
                colStart, numLocalCols);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    double seconds = durationMs.count() / 1000.0;

    // ── Copy results back to host ────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, localSize * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_path.data(), d_path, localSize * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // ── Gather to rank 0 ─────────────────────────────────────────────────
    std::vector<int> elemCounts(numRanks);
    std::vector<int> elemDispl(numRanks);
    {
        size_t base = numNodes / numRanks;
        size_t rem  = numNodes % numRanks;
        int offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t nlc = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            elemCounts[r] = static_cast<int>(nlc * numNodes);
            elemDispl[r]  = offset;
            offset += static_cast<int>(nlc * numNodes);
        }
    }

    std::vector<unsigned int> full_dist;
    std::vector<unsigned int> full_path;
    if (myRank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
    }

    MPI_Gatherv(h_dist.data(), elemCounts[myRank], MPI_UNSIGNED,
                full_dist.data(), elemCounts.data(), elemDispl.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    MPI_Gatherv(h_path.data(), elemCounts[myRank], MPI_UNSIGNED,
                full_path.data(), elemCounts.data(), elemDispl.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // ── Output and validation on rank 0 ──────────────────────────────────
    if (myRank == 0) {
        printf("Computation time:  %ld ms\n", durationMs.count());

        double ops = static_cast<double>(numNodes)
                   * static_cast<double>(numNodes)
                   * static_cast<double>(numNodes);
        double gops = ops / seconds / 1e9;
        printf("Performance:       %.3f GOPS\n", gops);
        fflush(stdout);

        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            fflush(stdout);
            bool valid = validateResult(full_dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            fflush(stdout);
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_colK));

    MPI_Finalize();
    return 0;
}
