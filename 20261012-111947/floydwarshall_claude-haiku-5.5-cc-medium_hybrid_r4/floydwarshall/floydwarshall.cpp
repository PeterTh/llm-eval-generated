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

// Percentage of each rank's rows updated by OpenMP host threads; the remaining rows are updated on the GPU
constexpr size_t CPU_ROW_PERCENT = 10;
constexpr unsigned int GPU_BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

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

// Fills the path rows firstRow .. firstRow+numRows-1 (same values as the full-matrix initialization:
// every entry of row i holds i)
void initializePathRows(unsigned int* path, const size_t firstRow, const size_t numRows, const size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < numRows; ++r) {
        std::fill(path + r * numNodes, path + (r + 1) * numNodes, static_cast<unsigned int>(firstRow + r));
    }
}

// Applies intermediate node k to one local source row (in-place, branch-free form of the
// "if (newDist < distIJ)" update so the inner loop vectorizes)
inline void relaxRow(unsigned int* distRow, unsigned int* pathRow, const unsigned int* pivotRow,
                     const size_t numNodes, const unsigned int k) noexcept {
    const unsigned int distIK = distRow[k];
    #pragma omp simd
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int distIJ = distRow[j];
        const unsigned int newDist = distIK + pivotRow[j];
        const bool improved = newDist < distIJ;
        distRow[j] = improved ? newDist : distIJ;
        pathRow[j] = improved ? k : pathRow[j];
    }
}

// Applies intermediate node k to the GPU-resident rows of this rank
__global__ void floydWarshallStepGpu(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                     const unsigned int* __restrict__ pivotRow,
                                     const size_t numRows, const size_t numNodes, const unsigned int k) {
    const size_t j = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (j >= numNodes) {
        return;
    }
    for (size_t i = blockIdx.y; i < numRows; i += gridDim.y) {
        unsigned int* distRow = dist + i * numNodes;
        unsigned int* pathRow = path + i * numNodes;
        const unsigned int newDist = distRow[k] + pivotRow[j];
        if (newDist < distRow[j]) {
            distRow[j] = newDist;
            pathRow[j] = k;
        }
    }
}

// Row-block partition of the distance matrix across MPI ranks
struct RowPartition {
    std::vector<int> counts;  // rows owned by each rank
    std::vector<int> starts;  // first global row owned by each rank

    int owner(const size_t row) const {
        int r = 0;
        while (static_cast<size_t>(starts[r] + counts[r]) <= row) {
            ++r;
        }
        return r;
    }
};

// Floyd-Warshall over the distributed matrix. Rank-local rows [0, gpuRows) live on the GPU
// (dDist/dPath), the remaining rows live in localDist/cpuPath on the host.
// Intermediate node k must be processed in order; within one step, rows are independent
// (row k and column k do not change), so each step is parallel across all rows.
void floydWarshall(const RowPartition& part, const int rank, const size_t numNodes, const size_t gpuRows,
                   unsigned int* dDist, unsigned int* dPath, unsigned int* dPivot,
                   unsigned int* localDist, unsigned int* cpuPath, unsigned int* pivotRow) {
    const size_t localStart = part.starts[rank];
    const size_t localRows = part.counts[rank];
    const size_t cpuRows = localRows - gpuRows;
    unsigned int* cpuDist = localDist + gpuRows * numNodes;
    const size_t rowBytes = numNodes * sizeof(unsigned int);

    const dim3 block(GPU_BLOCK_SIZE);
    const dim3 grid(static_cast<unsigned int>((numNodes + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE),
                    static_cast<unsigned int>(std::min<size_t>(gpuRows, 65535)));

    for (size_t k = 0; k < numNodes; ++k) {
        // The owner of row k supplies the (up-to-date) pivot row
        const int owner = part.owner(k);
        if (owner == rank) {
            const size_t localK = k - localStart;
            if (localK < gpuRows) {
                CUDA_CHECK(cudaMemcpy(pivotRow, dDist + localK * numNodes, rowBytes, cudaMemcpyDeviceToHost));
            } else {
                std::copy_n(cpuDist + (localK - gpuRows) * numNodes, numNodes, pivotRow);
            }
        }
        MPI_Bcast(pivotRow, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // GPU rows: the kernel runs asynchronously while the host threads update the CPU rows
        if (gpuRows > 0) {
            CUDA_CHECK(cudaMemcpy(dPivot, pivotRow, rowBytes, cudaMemcpyHostToDevice));
            floydWarshallStepGpu<<<grid, block>>>(dDist, dPath, dPivot, gpuRows, numNodes,
                                                  static_cast<unsigned int>(k));
            CUDA_CHECK(cudaGetLastError());
        }

        // Host rows
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < cpuRows; ++r) {
            relaxRow(cpuDist + r * numNodes, cpuPath + r * numNodes, pivotRow, numNodes,
                     static_cast<unsigned int>(k));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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

    // Parse command line arguments (every rank parses the same arguments, so all ranks exit alike)
    int exitCode = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // One rank per GPU: ranks on a node are mapped round-robin onto its devices,
    // and each rank gets an equal share of the node's host cores for its OpenMP threads
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    // Distribute contiguous row blocks across ranks
    RowPartition part;
    part.counts.resize(numRanks);
    part.starts.resize(numRanks);
    {
        const size_t base = numNodes / numRanks;
        const size_t extra = numNodes % numRanks;
        size_t start = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t count = base + (static_cast<size_t>(r) < extra ? 1 : 0);
            part.counts[r] = static_cast<int>(count);
            part.starts[r] = static_cast<int>(start);
            start += count;
        }
    }
    const size_t localStart = part.starts[rank];
    const size_t localRows = part.counts[rank];
    // Rows [0, gpuRows) of this rank are GPU-resident, the rest are updated by OpenMP threads
    const size_t cpuRows = localRows * CPU_ROW_PERCENT / 100;
    const size_t gpuRows = localRows - cpuRows;

    // Allocate and initialize matrices. Rank 0 generates the full graph (keeping the original
    // random sequence) and scatters the row blocks.
    std::vector<unsigned int> fullDist;  // rank 0 only: initialization, gather and output
    if (rank == 0) {
        printf("Initializing graph...\n");
        fullDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<int> scatterCounts(numRanks);
    std::vector<int> scatterDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        scatterCounts[r] = static_cast<int>(part.counts[r] * numNodes);
        scatterDispls[r] = static_cast<int>(part.starts[r] * numNodes);
    }
    std::vector<unsigned int> localDist(localRows * numNodes);
    MPI_Scatterv(fullDist.data(), scatterCounts.data(), scatterDispls.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::vector<unsigned int> cpuPath(cpuRows * numNodes);
    initializePathRows(cpuPath.data(), localStart + gpuRows, cpuRows, numNodes);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dPivot = nullptr;
    const size_t gpuBytes = gpuRows * numNodes * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&dPivot, numNodes * sizeof(unsigned int)));
    if (gpuRows > 0) {
        CUDA_CHECK(cudaMalloc(&dDist, gpuBytes));
        CUDA_CHECK(cudaMalloc(&dPath, gpuBytes));
        CUDA_CHECK(cudaMemcpy(dDist, localDist.data(), gpuBytes, cudaMemcpyHostToDevice));
        std::vector<unsigned int> gpuPath(gpuRows * numNodes);
        initializePathRows(gpuPath.data(), localStart, gpuRows, numNodes);
        CUDA_CHECK(cudaMemcpy(dPath, gpuPath.data(), gpuBytes, cudaMemcpyHostToDevice));
    }
    std::vector<unsigned int> pivotRow(numNodes);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(part, rank, numNodes, gpuRows, dDist, dPath, dPivot,
                  localDist.data(), cpuPath.data(), pivotRow.data());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather the distance matrix on rank 0 only when it is needed for output or validation
    if (printResults || validate) {
        if (gpuRows > 0) {
            CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, gpuBytes, cudaMemcpyDeviceToHost));
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                    fullDist.data(), scatterCounts.data(), scatterDispls.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            exitCode = valid ? 0 : 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        exitCode = 0;
    }

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dPivot));
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
