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

#define CUDA_CHECK(call)                                                              \
    do {                                                                             \
        cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err__));                                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                            \
        }                                                                             \
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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Independent, order-agnostic writes: safe to parallelize with OpenMP.
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Device kernel: relax one pivot node k against a contiguous block of local rows.
// Row layout: element (i, j) for local row-block lives at localBase[li * n + j],
// matching dist[idx2(j, i, n)] == dist[i * n + j] in the original flattened scheme.
__global__ void fwRelaxKernel(unsigned int* __restrict__ d_dist,
                               unsigned int* __restrict__ d_path,
                               const unsigned int* __restrict__ d_pivotRow,
                               size_t localRows, size_t n, size_t k) {
    size_t li = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (li >= localRows || j >= n) return;

    const size_t idx = li * n + j;
    const unsigned int distIK = d_dist[li * n + k];
    const unsigned int distKJ = d_pivotRow[j];
    const unsigned int newDist = distIK + distKJ;

    if (newDist < d_dist[idx]) {
        d_dist[idx] = newDist;
        d_path[idx] = static_cast<unsigned int>(k);
    }
}

// Per-rank state for one local GPU handling a sub-range of this rank's row block.
struct GpuSlice {
    int deviceId = 0;
    size_t rowStart = 0;   // offset within this rank's local row block
    size_t rowCount = 0;
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_pivot = nullptr;
    cudaStream_t stream = nullptr;
};

// Hybrid MPI + OpenMP + CUDA Floyd-Warshall.
//
// Rows of the (row-major) dist/path matrices are block-distributed across MPI
// ranks. Within a rank, the local row block is further split across the
// visible GPUs, each driven by its own OpenMP host thread. For each pivot k,
// the owning rank/GPU extracts the pivot row, MPI_Bcast distributes it to all
// ranks, and every GPU relaxes its local rows against the pivot row in parallel.
void floydWarshallHybrid(std::vector<unsigned int>& dist,
                          std::vector<unsigned int>& path,
                          const size_t numNodes,
                          int mpiRank, int mpiSize) {
    // ---- Row block distribution across MPI ranks ----
    std::vector<size_t> rowStart(mpiSize + 1, 0);
    {
        const size_t base = numNodes / static_cast<size_t>(mpiSize);
        const size_t rem = numNodes % static_cast<size_t>(mpiSize);
        size_t acc = 0;
        for (int r = 0; r < mpiSize; ++r) {
            rowStart[r] = acc;
            acc += base + (static_cast<size_t>(r) < rem ? 1 : 0);
        }
        rowStart[mpiSize] = numNodes;
    }
    const size_t myRowStart = rowStart[mpiRank];
    const size_t myRowCount = rowStart[mpiRank + 1] - myRowStart;

    // Precompute owning rank for every row (cheap, O(n)).
    std::vector<int> ownerOfRow(numNodes, 0);
    for (int r = 0; r < mpiSize; ++r) {
        for (size_t i = rowStart[r]; i < rowStart[r + 1]; ++i) {
            ownerOfRow[i] = r;
        }
    }

    // ---- Scatter row blocks to every rank ----
    std::vector<int> sendCountsDist(mpiSize), sendDisplsDist(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        sendCountsDist[r] = static_cast<int>((rowStart[r + 1] - rowStart[r]) * numNodes);
        sendDisplsDist[r] = static_cast<int>(rowStart[r] * numNodes);
    }

    std::vector<unsigned int> distLocal(myRowCount * numNodes);
    std::vector<unsigned int> pathLocal(myRowCount * numNodes);

    MPI_Scatterv(dist.data(), sendCountsDist.data(), sendDisplsDist.data(), MPI_UNSIGNED,
                 distLocal.data(), static_cast<int>(myRowCount * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(path.data(), sendCountsDist.data(), sendDisplsDist.data(), MPI_UNSIGNED,
                 pathLocal.data(), static_cast<int>(myRowCount * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // ---- Partition the node's physical GPUs across the ranks that share it ----
    // (MPI ranks are not necessarily one-per-node, so every rank must only claim
    // a disjoint slice of the local devices to avoid oversubscribing the GPUs.)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "Rank %d: no CUDA devices visible\n", mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int localRank = 0, localSize = 1;
    {
        MPI_Comm shmComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpiRank, MPI_INFO_NULL, &shmComm);
        MPI_Comm_rank(shmComm, &localRank);
        MPI_Comm_size(shmComm, &localSize);
        MPI_Comm_free(&shmComm);
    }

    int myDeviceStart, myDeviceCount;
    if (localSize <= deviceCount) {
        const int base = deviceCount / localSize;
        const int rem = deviceCount % localSize;
        myDeviceStart = localRank * base + std::min(localRank, rem);
        myDeviceCount = base + (localRank < rem ? 1 : 0);
    } else {
        // More local ranks than devices: share devices round-robin, one each.
        myDeviceStart = localRank % deviceCount;
        myDeviceCount = 1;
    }

    const int numGpus = (myRowCount > 0)
                            ? static_cast<int>(std::min<size_t>(static_cast<size_t>(myDeviceCount), myRowCount))
                            : 0;

    std::vector<GpuSlice> slices(numGpus);
    if (numGpus > 0) {
        const size_t base = myRowCount / static_cast<size_t>(numGpus);
        const size_t rem = myRowCount % static_cast<size_t>(numGpus);
        size_t acc = 0;
        for (int t = 0; t < numGpus; ++t) {
            slices[t].deviceId = myDeviceStart + t;
            slices[t].rowStart = acc;
            slices[t].rowCount = base + (static_cast<size_t>(t) < rem ? 1 : 0);
            acc += slices[t].rowCount;
        }
    }

    // Row owning-GPU (local index) lookup, used to locate the pivot row within a rank.
    std::vector<int> ownerOfLocalRow(myRowCount, 0);
    for (int t = 0; t < numGpus; ++t) {
        for (size_t li = slices[t].rowStart; li < slices[t].rowStart + slices[t].rowCount; ++li) {
            ownerOfLocalRow[li] = t;
        }
    }

    // ---- Allocate device buffers and upload each GPU's row sub-block ----
    #pragma omp parallel for num_threads(numGpus > 0 ? numGpus : 1) schedule(static)
    for (int t = 0; t < numGpus; ++t) {
        GpuSlice& s = slices[t];
        CUDA_CHECK(cudaSetDevice(s.deviceId));
        CUDA_CHECK(cudaStreamCreate(&s.stream));
        const size_t bytes = s.rowCount * numNodes * sizeof(unsigned int);
        CUDA_CHECK(cudaMalloc(&s.d_dist, bytes));
        CUDA_CHECK(cudaMalloc(&s.d_path, bytes));
        CUDA_CHECK(cudaMalloc(&s.d_pivot, numNodes * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpyAsync(s.d_dist, &distLocal[s.rowStart * numNodes], bytes,
                                    cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_path, &pathLocal[s.rowStart * numNodes], bytes,
                                    cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    std::vector<unsigned int> pivotRow(numNodes);

    const dim3 blockDim(32, 8);

    // ---- Main Floyd-Warshall loop over pivot nodes ----
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow[k];

        if (mpiRank == owner) {
            const size_t localK = k - myRowStart;
            const int t = ownerOfLocalRow[localK];
            GpuSlice& s = slices[t];
            CUDA_CHECK(cudaSetDevice(s.deviceId));
            const size_t rowOffsetElems = (localK - s.rowStart) * numNodes;
            CUDA_CHECK(cudaMemcpy(pivotRow.data(), s.d_dist + rowOffsetElems,
                                   numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        #pragma omp parallel for num_threads(numGpus > 0 ? numGpus : 1) schedule(static)
        for (int t = 0; t < numGpus; ++t) {
            GpuSlice& s = slices[t];
            CUDA_CHECK(cudaSetDevice(s.deviceId));
            CUDA_CHECK(cudaMemcpyAsync(s.d_pivot, pivotRow.data(), numNodes * sizeof(unsigned int),
                                        cudaMemcpyHostToDevice, s.stream));

            const dim3 gridDim(static_cast<unsigned int>((numNodes + blockDim.x - 1) / blockDim.x),
                                static_cast<unsigned int>((s.rowCount + blockDim.y - 1) / blockDim.y));
            fwRelaxKernel<<<gridDim, blockDim, 0, s.stream>>>(s.d_dist, s.d_path, s.d_pivot,
                                                               s.rowCount, numNodes, k);
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
    }

    // ---- Download results and free device buffers ----
    #pragma omp parallel for num_threads(numGpus > 0 ? numGpus : 1) schedule(static)
    for (int t = 0; t < numGpus; ++t) {
        GpuSlice& s = slices[t];
        CUDA_CHECK(cudaSetDevice(s.deviceId));
        const size_t bytes = s.rowCount * numNodes * sizeof(unsigned int);
        CUDA_CHECK(cudaMemcpyAsync(&distLocal[s.rowStart * numNodes], s.d_dist, bytes,
                                    cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(&pathLocal[s.rowStart * numNodes], s.d_path, bytes,
                                    cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
        CUDA_CHECK(cudaFree(s.d_dist));
        CUDA_CHECK(cudaFree(s.d_path));
        CUDA_CHECK(cudaFree(s.d_pivot));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
    }

    // ---- Gather local blocks back into the full matrices on rank 0 ----
    MPI_Gatherv(distLocal.data(), static_cast<int>(myRowCount * numNodes), MPI_UNSIGNED,
                dist.data(), sendCountsDist.data(), sendDisplsDist.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(pathLocal.data(), static_cast<int>(myRowCount * numNodes), MPI_UNSIGNED,
                path.data(), sendCountsDist.data(), sendDisplsDist.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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

    std::vector<unsigned int> dist, path;

    if (mpiRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", mpiSize);

        // Allocate matrices
        dist.assign(numNodes * numNodes, 0);
        path.assign(numNodes * numNodes, 0);

        // Initialize
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);

        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallHybrid(dist, path, numNodes, mpiRank, mpiSize);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (mpiRank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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
