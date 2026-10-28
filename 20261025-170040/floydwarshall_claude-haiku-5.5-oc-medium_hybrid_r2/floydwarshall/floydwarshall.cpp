// Hybrid MPI + OpenMP + CUDA Floyd-Warshall all-pairs shortest path.
//
// Decomposition:
//   * MPI: the n columns of the (column-major) distance and path matrices are
//     split into contiguous blocks, one block per rank. Each iteration k, the
//     owner of column k broadcasts it; every rank then updates its own columns.
//     Row k of the local block is always local, so no other communication is
//     needed.
//   * CUDA: each rank drives one GPU (local rank modulo device count) and
//     updates the leading part of its local block with a 2D kernel.
//   * OpenMP: the trailing part of the local block (a small share of the
//     columns) is updated by host threads concurrently with the GPU kernel, and
//     the path matrix is initialized in parallel.
//
// Semantics match the sequential algorithm exactly: for each k, column k and
// row k are invariant (the diagonal is zero), so every element update reads the
// same operands as the in-place sequential loop, and k is processed in order,
// so path tie-breaking (strict '<') is identical.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <sched.h>

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

// Fraction of each rank's local columns updated by OpenMP host threads, per host
// thread. The rest of the local columns are updated on the rank's GPU. The share
// scales with the thread count so that the host work does not become the bottleneck
// when fewer CPUs are available to the rank.
constexpr double CPU_COLUMN_SHARE_PER_THREAD = 0.00125;

// Host OpenMP team size per rank. The host share is small, so larger teams only add
// fork/join and synchronization overhead.
constexpr unsigned int MAX_HOST_THREADS = 8;

constexpr unsigned int CUDA_BLOCK_SIZE = 256;
constexpr unsigned int CUDA_MAX_GRID_Y = 65535;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
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
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// Initializes the path entries for the local columns [firstCol, firstCol + numCols).
// The path value of element (a, b) is b in the original initialization.
void initializeLocalPathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                               const size_t firstCol, const size_t numCols) {
    #pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(numCols); ++j) {
        unsigned int* column = path.data() + static_cast<size_t>(j) * numNodes;
        const unsigned int value = static_cast<unsigned int>(firstCol + static_cast<size_t>(j));
        std::fill(column, column + numNodes, value);
    }
}

// Device kernel: one step k of Floyd-Warshall on the GPU-resident columns.
// Element (a, b) of the local block lives at b * n + a.
__global__ void floydWarshallStepGpu(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int* __restrict__ colK,
                                     const unsigned int* __restrict__ rowK,
                                     const size_t n, const size_t gpuCols,
                                     const unsigned int k) {
    const size_t a = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (a >= n) {
        return;
    }
    for (size_t b = blockIdx.y; b < gpuCols; b += gridDim.y) {
        const size_t idx = b * n + a;
        const unsigned int distIJ = dist[idx];
        const unsigned int newDist = colK[a] + rowK[b];
        if (newDist < distIJ) {
            dist[idx] = newDist;
            path[idx] = k;
        }
    }
}

// Device kernel: gathers row k of the GPU-resident columns, i.e. element (k, b) at b * n + k.
__global__ void gatherRowGpu(const unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ rowK,
                             const size_t n, const size_t gpuCols, const size_t k) {
    const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b < gpuCols) {
        rowK[b] = dist[b * n + k];
    }
}

// Host update of one Floyd-Warshall step on the CPU-resident columns.
// `dist`/`path` point to the first CPU column; `rowK[b]` holds M(k, b).
void floydWarshallStepCpu(unsigned int* dist, unsigned int* path,
                          const unsigned int* colK, const unsigned int* rowK,
                          const size_t n, const size_t numCols, const unsigned int k) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long b = 0; b < static_cast<long long>(numCols); ++b) {
        for (long long a = 0; a < static_cast<long long>(n); ++a) {
            const size_t idx = static_cast<size_t>(b) * n + static_cast<size_t>(a);
            const unsigned int distIJ = dist[idx];
            const unsigned int newDist = colK[a] + rowK[b];
            if (newDist < distIJ) {
                dist[idx] = newDist;
                path[idx] = k;
            }
        }
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
    const bool root = (rank == 0);
    const MPI_Comm comm = MPI_COMM_WORLD;

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Bind each rank to a GPU, round-robin over the GPUs on its node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    // Share the cores this process may run on among the ranks of the node, so that
    // the host OpenMP teams do not oversubscribe the CPUs.
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        int allowedCpus = 1;
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
            allowedCpus = CPU_COUNT(&mask);
        }
        omp_set_num_threads(std::clamp(allowedCpus / localSize, 1, static_cast<int>(MAX_HOST_THREADS)));
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(comm, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Column-block distribution: rank r owns columns [colStart(r), colStart(r) + colCount(r)).
    const size_t n = numNodes;
    const size_t baseCols = n / static_cast<size_t>(numRanks);
    const size_t extraCols = n % static_cast<size_t>(numRanks);
    auto colCount = [&](int r) -> size_t {
        return baseCols + (static_cast<size_t>(r) < extraCols ? 1 : 0);
    };
    auto colStart = [&](int r) -> size_t {
        return static_cast<size_t>(r) * baseCols + std::min(static_cast<size_t>(r), extraCols);
    };
    auto ownerOf = [&](size_t k) -> int {
        for (int r = 0; r < numRanks; ++r) {
            if (k < colStart(r) + colCount(r)) return r;
        }
        return numRanks - 1;
    };

    const size_t myCols = colCount(rank);
    const size_t myStart = colStart(rank);
    const size_t cpuCols = std::min(myCols, static_cast<size_t>(
        static_cast<double>(myCols) * CPU_COLUMN_SHARE_PER_THREAD * omp_get_max_threads()));
    const size_t gpuCols = myCols - cpuCols;  // local columns [0, gpuCols) live on the GPU

    // Column datatype: one matrix column of n elements (columns are contiguous in memory).
    MPI_Datatype colType;
    MPI_Type_contiguous(static_cast<int>(n), MPI_UNSIGNED, &colType);
    MPI_Type_commit(&colType);

    std::vector<int> counts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = static_cast<int>(colCount(r));
        displs[r] = static_cast<int>(colStart(r));
    }

    // Host-side local blocks (authoritative for CPU columns; staging for GPU columns).
    std::vector<unsigned int> hDist(myCols * n);
    std::vector<unsigned int> hPath(myCols * n);

    // Full distance matrix is generated on the root (sequential RNG, identical to original)
    // and scattered by columns.
    std::vector<unsigned int> dist;
    if (root) {
        dist.resize(n * n);
        initializeDistanceMatrix(dist, n, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(root ? dist.data() : nullptr, counts.data(), displs.data(), colType,
                 hDist.data(), static_cast<int>(myCols), colType, 0, comm);

    initializeLocalPathMatrix(hPath, n, myStart, myCols);

    // Device buffers.
    unsigned int *dDist = nullptr, *dPath = nullptr, *dCol = nullptr, *dRow = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (gpuCols > 0) {
        CUDA_CHECK(cudaMalloc(&dDist, gpuCols * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dPath, gpuCols * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dCol, n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dRow, gpuCols * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(dDist, hDist.data(), gpuCols * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, hPath.data(), gpuCols * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    // Pinned broadcast buffer for column k (faster host-device transfers each step).
    unsigned int* hCol = nullptr;
    CUDA_CHECK(cudaMallocHost(&hCol, std::max<size_t>(n, 1) * sizeof(unsigned int)));
    std::vector<unsigned int> hRow(cpuCols);  // row k entries for CPU columns

    const dim3 block(CUDA_BLOCK_SIZE);
    const dim3 grid(static_cast<unsigned int>((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE),
                    static_cast<unsigned int>(std::min<size_t>(gpuCols, CUDA_MAX_GRID_Y)));

    if (root) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    unsigned int* hDistCpu = hDist.data() + gpuCols * n;
    unsigned int* hPathCpu = hPath.data() + gpuCols * n;

    for (size_t k = 0; k < n; ++k) {
        const int owner = ownerOf(k);

        // The owner provides column k. Its GPU part is up to date since the previous
        // step was synchronized.
        if (rank == owner) {
            const size_t jk = k - myStart;
            if (jk < gpuCols) {
                CUDA_CHECK(cudaMemcpyAsync(hCol, dDist + jk * n,
                                           n * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                std::memcpy(hCol, hDist.data() + jk * n, n * sizeof(unsigned int));
            }
        }
        MPI_Bcast(hCol, static_cast<int>(n), MPI_UNSIGNED, owner, comm);

        // GPU part: enqueue column/row staging and the update kernel.
        if (gpuCols > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dCol, hCol, n * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, stream));
            // Row k of the GPU columns: element (k, b) is at b * n + k.
            gatherRowGpu<<<static_cast<unsigned int>((gpuCols + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE),
                           CUDA_BLOCK_SIZE, 0, stream>>>(dDist, dRow, n, gpuCols, k);
            CUDA_CHECK(cudaGetLastError());
            floydWarshallStepGpu<<<grid, block, 0, stream>>>(dDist, dPath, dCol, dRow,
                                                             n, gpuCols,
                                                             static_cast<unsigned int>(k));
            CUDA_CHECK(cudaGetLastError());
        }

        // CPU part: runs concurrently with the GPU kernel.
        if (cpuCols > 0) {
            for (size_t b = 0; b < cpuCols; ++b) {
                hRow[b] = hDistCpu[b * n + k];
            }
            floydWarshallStepCpu(hDistCpu, hPathCpu, hCol, hRow.data(),
                                 n, cpuCols, static_cast<unsigned int>(k));
        }

        if (gpuCols > 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    if (gpuCols > 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    MPI_Barrier(comm);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Collect results on the root (outside the timed region).
    if (gpuCols > 0) {
        CUDA_CHECK(cudaMemcpy(hDist.data(), dDist, gpuCols * n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hPath.data(), dPath, gpuCols * n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }
    std::vector<unsigned int> path;
    if (root) {
        path.resize(n * n);
    }
    MPI_Gatherv(hDist.data(), static_cast<int>(myCols), colType,
                root ? dist.data() : nullptr, counts.data(), displs.data(), colType, 0, comm);
    MPI_Gatherv(hPath.data(), static_cast<int>(myCols), colType,
                root ? path.data() : nullptr, counts.data(), displs.data(), colType, 0, comm);

    if (dDist) CUDA_CHECK(cudaFree(dDist));
    if (dPath) CUDA_CHECK(cudaFree(dPath));
    if (dCol) CUDA_CHECK(cudaFree(dCol));
    if (dRow) CUDA_CHECK(cudaFree(dRow));
    CUDA_CHECK(cudaFreeHost(hCol));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Type_free(&colType);

    int exitCode = 0;
    if (root) {
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
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
