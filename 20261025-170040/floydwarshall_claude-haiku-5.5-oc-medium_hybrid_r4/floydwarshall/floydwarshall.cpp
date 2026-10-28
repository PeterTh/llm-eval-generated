#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Contiguous row-block distribution of numRows rows over parts ranks.
// Ranks [0, rem) own (base+1) rows, the rest own base rows.
struct RowPartition {
    size_t base;
    size_t rem;

    RowPartition(size_t numRows, int parts)
        : base(numRows / static_cast<size_t>(parts)),
          rem(numRows % static_cast<size_t>(parts)) {}

    size_t begin(int r) const {
        const size_t rr = static_cast<size_t>(r);
        return rr < rem ? rr * (base + 1) : rem * (base + 1) + (rr - rem) * base;
    }
    size_t count(int r) const { return begin(r + 1) - begin(r); }
    int owner(size_t row) const {
        const size_t boundary = rem * (base + 1);
        return row < boundary ? static_cast<int>(row / (base + 1))
                              : static_cast<int>(rem + (row - boundary) / base);
    }
};

// Fills rows [rowBegin, rowEnd) of the distance matrix. The random sequence is
// generated exactly as in the sequential version (same seed, same order), so
// every rank sees identical values for its rows.
void initializeDistanceMatrixBlock(std::vector<unsigned int>& block, const size_t numNodes,
                                   const size_t rowBegin, const size_t rowEnd,
                                   const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t lo = rowBegin * numNodes;
    const size_t hi = rowEnd * numNodes;

    for (size_t g = 0; g < numNodes * numNodes; ++g) {
        const unsigned int v =
            rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (g >= lo && g < hi) {
            block[g - lo] = v;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        block[(i - rowBegin) * numNodes + i] = 0;
    }
}

// Path initialization for the local row block. Equivalent to the sequential
// initializePathMatrix: every entry (row r, column c) holds r.
void initializePathMatrixBlock(std::vector<unsigned int>& block, const size_t numNodes,
                               const size_t rowBegin, const size_t numLocalRows) {
#pragma omp parallel for schedule(static)
    for (long long li = 0; li < static_cast<long long>(numLocalRows); ++li) {
        const unsigned int r = static_cast<unsigned int>(rowBegin + static_cast<size_t>(li));
        unsigned int* row = block.data() + static_cast<size_t>(li) * numNodes;
        for (size_t c = 0; c < numNodes; ++c) {
            row[c] = r;
        }
    }
}

// Relaxes rows [rowBegin, rowEnd) (local indices) using pivot row k.
// Row k and column k are invariant during iteration k (diagonal is zero), so
// reading them from the pivot copy and in-place matches the sequential order.
__global__ void fwUpdateKernel(unsigned int* dist, unsigned int* path,
                               const unsigned int* pivot, const size_t n,
                               const size_t rowBegin, const size_t rowEnd,
                               const unsigned int k) {
    const size_t i = rowBegin + blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowEnd || j >= n) return;

    unsigned int* rowI = dist + i * n;
    const unsigned int distIK = rowI[k];
    const unsigned int distKJ = pivot[j];
    const unsigned int newDist = distIK + distKJ;
    if (newDist < rowI[j]) {
        rowI[j] = newDist;
        path[i * n + j] = k;
    }
}

static void launchFwRows(unsigned int* d_dist, unsigned int* d_path, const unsigned int* d_pivot,
                         const size_t n, const size_t rowBegin, const size_t rowEnd,
                         const unsigned int k) {
    if (n == 0 || rowBegin >= rowEnd) return;
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((n + 31) / 32),
                    static_cast<unsigned int>((rowEnd - rowBegin + 7) / 8));
    fwUpdateKernel<<<grid, block>>>(d_dist, d_path, d_pivot, n, rowBegin, rowEnd, k);
    CUDA_CHECK(cudaGetLastError());
}

// Distributed Floyd-Warshall. Each rank owns a contiguous block of rows that
// resides on its GPU. For iteration k the owner of pivot row k+1 updates that
// row first and broadcasts it (non-blocking), so the broadcast overlaps the
// remaining GPU work of iteration k.
void floydWarshallDistributed(std::vector<unsigned int>& distLocal,
                              std::vector<unsigned int>& pathLocal,
                              const size_t numNodes, const RowPartition& part,
                              const int rank, MPI_Comm comm) {
    const size_t n = numNodes;
    const size_t r0 = part.begin(rank);
    const size_t nloc = part.count(rank);
    const size_t localElems = std::max<size_t>(nloc * n, 1);

    unsigned int *d_dist = nullptr, *d_path = nullptr;
    unsigned int* d_piv[2] = {nullptr, nullptr};
    unsigned int* h_piv[2] = {nullptr, nullptr};

    CUDA_CHECK(cudaMalloc(&d_dist, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_path, localElems * sizeof(unsigned int)));
    const size_t pivBytes = std::max<size_t>(n, 1) * sizeof(unsigned int);
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_piv[b], pivBytes));
        CUDA_CHECK(cudaMallocHost(&h_piv[b], pivBytes));
    }

    if (nloc > 0) {
        CUDA_CHECK(cudaMemcpy(d_dist, distLocal.data(), nloc * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, pathLocal.data(), nloc * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    if (n > 0) {
        // Prime pivot row 0 (taken from the host copy, no kernel has run yet).
        const int root0 = part.owner(0);
        if (root0 == rank) {
            std::memcpy(h_piv[0], distLocal.data(), n * sizeof(unsigned int));
        }
        MPI_Bcast(h_piv[0], static_cast<int>(n), MPI_UNSIGNED, root0, comm);
        CUDA_CHECK(cudaMemcpy(d_piv[0], h_piv[0], n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        for (size_t k = 0; k < n; ++k) {
            const bool hasNext = (k + 1 < n);
            const unsigned int kk = static_cast<unsigned int>(k);

            bool ownsNext = false;
            size_t nextLocal = 0;
            MPI_Request nextReq = MPI_REQUEST_NULL;
            unsigned int* nextBuf = nullptr;
            int nextRoot = -1;

            if (hasNext) {
                nextRoot = part.owner(k + 1);
                nextBuf = h_piv[(k + 1) & 1];
                if (nextRoot == rank) {
                    ownsNext = true;
                    nextLocal = (k + 1) - r0;
                    // Update the next pivot row first, then publish it.
                    launchFwRows(d_dist, d_path, d_piv[k & 1], n, nextLocal, nextLocal + 1, kk);
                    CUDA_CHECK(cudaMemcpy(nextBuf, d_dist + nextLocal * n,
                                          n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
                }
                MPI_Ibcast(nextBuf, static_cast<int>(n), MPI_UNSIGNED, nextRoot, comm, &nextReq);
            }

            // Remaining local rows for iteration k.
            if (ownsNext) {
                launchFwRows(d_dist, d_path, d_piv[k & 1], n, 0, nextLocal, kk);
                launchFwRows(d_dist, d_path, d_piv[k & 1], n, nextLocal + 1, nloc, kk);
            } else {
                launchFwRows(d_dist, d_path, d_piv[k & 1], n, 0, nloc, kk);
            }

            if (!hasNext) break;

            // Receive pivot row k+1 and stage it on the device for iteration k+1.
            MPI_Wait(&nextReq, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(d_piv[(k + 1) & 1], nextBuf, n * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (nloc > 0) {
        CUDA_CHECK(cudaMemcpy(distLocal.data(), d_dist, nloc * n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(d_piv[b]));
        CUDA_CHECK(cudaFreeHost(h_piv[b]));
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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

    // Bind each rank to one GPU on its node
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const RowPartition part(numNodes, numRanks);
    const size_t r0 = part.begin(rank);
    const size_t nloc = part.count(rank);

    // Initialize local row block
    std::vector<unsigned int> distLocal(nloc * numNodes);
    std::vector<unsigned int> pathLocal(nloc * numNodes);
    initializeDistanceMatrixBlock(distLocal, numNodes, r0, r0 + nloc, 1, MAX_DISTANCE);
    initializePathMatrixBlock(pathLocal, numNodes, r0, nloc);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallDistributed(distLocal, pathLocal, numNodes, part, rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the full distance matrix on rank 0 only when it is needed
    int status = 0;
    if (printResults || validate) {
        if (numNodes > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) fprintf(stderr, "Number of nodes too large for gather\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(numRanks), displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = static_cast<int>(part.count(r));
            displs[r] = static_cast<int>(part.begin(r));
        }

        std::vector<unsigned int> dist;
        if (rank == 0) dist.resize(numNodes * numNodes);
        MPI_Gatherv(distLocal.data(), static_cast<int>(nloc), rowType,
                    rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), rowType,
                    0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);

        if (rank == 0) {
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
                    status = 1;
                }
            }
        }
        // Share the validation status so every rank exits consistently
        if (validate) MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return status;
}
