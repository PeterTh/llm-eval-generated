// Hybrid MPI + OpenMP + CUDA Floyd-Warshall all-pairs shortest path benchmark.
//
// Decomposition:
//   MPI    : matrix rows (source nodes) are split into contiguous blocks, one per rank.
//            For every pivot k, the rank owning row k broadcasts it to all ranks.
//   OpenMP : within a rank, one thread per visible GPU (node-local share). Each thread
//            drives its own device and owns a contiguous sub-block of the rank's rows.
//            Threads synchronize per pivot and share the broadcast pivot row.
//   CUDA   : each device updates its sub-block of rows for pivot k with one kernel launch.
//
// Semantics are identical to the serial algorithm: for a fixed k, row k and column k are
// invariant (dist[k][k] == 0), so every (i, j) update within a pivot step reads the same
// values regardless of execution order. The pivot-step loop over k remains sequential.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
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

static int gRank = 0;
static int gSize = 1;

[[noreturn]] static void fatal(const char* msg) {
    fprintf(stderr, "[rank %d] fatal: %s\n", gRank, msg);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "[rank %d] CUDA error at %s:%d: %s\n", gRank, __FILE__,     \
                    __LINE__, cudaGetErrorString(err_));                                 \
            fflush(stderr);                                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// Row-block partition of [0, numNodes) over gSize ranks: rank r owns [start[r], start[r+1]).
static std::vector<size_t> makeRowPartition(const size_t numNodes, const int parts) {
    std::vector<size_t> start(parts + 1);
    for (int r = 0; r <= parts; ++r) {
        start[r] = numNodes * static_cast<size_t>(r) / static_cast<size_t>(parts);
    }
    return start;
}

// Generates the same distance values as the serial reference (same RNG stream, same
// linear order) but only stores the rows [rowBegin, rowEnd) into `local`.
void initializeDistanceRows(std::vector<unsigned int>& local, const size_t numNodes,
                            const size_t rowBegin, const size_t rowEnd,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t lo = rowBegin * numNodes;
    const size_t hi = rowEnd * numNodes;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        const unsigned int value =
            rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= lo && i < hi) {
            local[i - lo] = value;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        local[(i - rowBegin) * numNodes + i] = 0;
    }
}

// Same values as the serial path initialization, restricted to rows [rowBegin, rowEnd).
void initializePathRows(std::vector<unsigned int>& local, const size_t numNodes,
                        const size_t rowBegin, const size_t rowEnd) {
    for (size_t r = rowBegin; r < rowEnd; ++r) {
        unsigned int* row = local.data() + (r - rowBegin) * numNodes;
        for (size_t c = 0; c < numNodes; ++c) {
            row[c] = static_cast<unsigned int>(r);
        }
    }
}

// One pivot step k for a block of local rows [0, rows):
//   dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]); path[i][j] = k on improvement.
// rowK holds the (unchanged during step k) row k of the full matrix.
__global__ void fwPivotKernel(unsigned int* dist, unsigned int* path,
                              const unsigned int* rowK,
                              const size_t rows, const size_t n, const size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= n) {
        return;
    }
    const size_t stride = static_cast<size_t>(gridDim.y) * blockDim.y;
    for (size_t li = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y; li < rows;
         li += stride) {
        const size_t base = li * n;
        const unsigned int distIK = dist[base + k];
        const unsigned int newDist = distIK + rowK[j];
        if (newDist < dist[base + j]) {
            dist[base + j] = newDist;
            path[base + j] = static_cast<unsigned int>(k);
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

struct DeviceSlice {
    int device;
    size_t begin;  // local row range [begin, end) within this rank's block
    size_t end;
};

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fatal("MPI implementation does not support MPI_THREAD_FUNNELED");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);
    MPI_Comm_size(MPI_COMM_WORLD, &gSize);
    const bool root = (gRank == 0);

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
    }

    // Node-local rank and GPU assignment
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, gRank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fatal("no CUDA devices available");
    }
    const int numThreads = std::max(1, numDevices / localSize);
    omp_set_dynamic(0);

    // Row partition across ranks and local partition across this rank's threads/devices
    const std::vector<size_t> rowStart = makeRowPartition(numNodes, gSize);
    const size_t rowBegin = rowStart[gRank];
    const size_t rowEnd = rowStart[gRank + 1];
    const size_t localRows = rowEnd - rowBegin;

    std::vector<DeviceSlice> slices(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        slices[t].device = (localRank * numThreads + t) % numDevices;
        slices[t].begin = localRows * t / numThreads;
        slices[t].end = localRows * (t + 1) / numThreads;
    }

    // Allocate and initialize (not timed, as in the original)
    if (root) {
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> hDist(localRows * numNodes);
    std::vector<unsigned int> hPath(localRows * numNodes);
    initializeDistanceRows(hDist, numNodes, rowBegin, rowEnd, 1, MAX_DISTANCE);
    initializePathRows(hPath, numNodes, rowBegin, rowEnd);

    // Full result matrices exist only on the root, for output and validation
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (root) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
    }

    // Pivot row double buffers in host memory (shared by all threads of this rank)
    std::vector<unsigned int> rowBuf[2];
    rowBuf[0].resize(numNodes);
    rowBuf[1].resize(numNodes);

    if (root) {
        printf("Computing shortest paths...\n");
    }

    double tStart = 0.0, tEnd = 0.0;

#pragma omp parallel num_threads(numThreads)
    {
        if (omp_get_num_threads() != numThreads) {
            if (omp_get_thread_num() == 0) fatal("OpenMP did not provide requested thread count");
        }
        const int t = omp_get_thread_num();
        const DeviceSlice s = slices[t];
        const size_t rows = s.end - s.begin;

        CUDA_CHECK(cudaSetDevice(s.device));
        unsigned int* dDist = nullptr;
        unsigned int* dPath = nullptr;
        unsigned int* dRow[2] = {nullptr, nullptr};
        if (rows > 0) {
            CUDA_CHECK(cudaMalloc(&dDist, rows * numNodes * sizeof(unsigned int)));
            CUDA_CHECK(cudaMalloc(&dPath, rows * numNodes * sizeof(unsigned int)));
            CUDA_CHECK(cudaMalloc(&dRow[0], numNodes * sizeof(unsigned int)));
            CUDA_CHECK(cudaMalloc(&dRow[1], numNodes * sizeof(unsigned int)));
        }

        // Context/allocation setup is excluded from timing
#pragma omp barrier
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            tStart = MPI_Wtime();
        }
#pragma omp barrier

        if (rows > 0) {
            CUDA_CHECK(cudaMemcpy(dDist, hDist.data() + s.begin * numNodes,
                                  rows * numNodes * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dPath, hPath.data() + s.begin * numNodes,
                                  rows * numNodes * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        }

        const dim3 block(32, 8);
        for (size_t k = 0; k < numNodes; ++k) {
            // Owner rank of pivot row k
            int owner = 0;
            while (owner + 1 < gSize && rowStart[owner + 1] <= k) ++owner;

            unsigned int* buf = rowBuf[k & 1].data();

            // The thread holding row k copies it from its device into the host buffer
            if (owner == gRank && rows > 0) {
                const size_t kLocal = k - rowBegin;
                if (kLocal >= s.begin && kLocal < s.end) {
                    CUDA_CHECK(cudaMemcpy(buf, dDist + (kLocal - s.begin) * numNodes,
                                          numNodes * sizeof(unsigned int),
                                          cudaMemcpyDeviceToHost));
                }
            }
#pragma omp barrier
#pragma omp master
            {
                MPI_Bcast(buf, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            }
#pragma omp barrier

            if (rows > 0) {
                CUDA_CHECK(cudaMemcpy(dRow[k & 1], buf, numNodes * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
                const size_t gx = (numNodes + block.x - 1) / block.x;
                const size_t gy = std::min<size_t>((rows + block.y - 1) / block.y, 65535);
                fwPivotKernel<<<dim3(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy)),
                                block>>>(dDist, dPath, dRow[k & 1], rows, numNodes, k);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (rows > 0) {
            CUDA_CHECK(cudaMemcpy(hDist.data() + s.begin * numNodes, dDist,
                                  rows * numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hPath.data() + s.begin * numNodes, dPath,
                                  rows * numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

#pragma omp barrier
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            tEnd = MPI_Wtime();
        }

        if (rows > 0) {
            CUDA_CHECK(cudaFree(dDist));
            CUDA_CHECK(cudaFree(dPath));
            CUDA_CHECK(cudaFree(dRow[0]));
            CUDA_CHECK(cudaFree(dRow[1]));
        }
    }

    // Gather the distributed result onto the root (row blocks are contiguous)
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);
    std::vector<int> counts(gSize), displs(gSize);
    for (int r = 0; r < gSize; ++r) {
        counts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        displs[r] = static_cast<int>(rowStart[r]);
    }
    MPI_Gatherv(hDist.data(), static_cast<int>(localRows), rowType,
                dist.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
    MPI_Gatherv(hPath.data(), static_cast<int>(localRows), rowType,
                path.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&rowType);

    int ret = 0;
    if (root) {
        const double seconds = tEnd - tStart;
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        const double ops = (double)numNodes * numNodes * numNodes;
        const double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Finalize();
    return ret;
}
