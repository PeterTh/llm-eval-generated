#include <algorithm>
#include <cmath>
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

// Index calculation for flattened 2D array (row-major: dist[i][j] = j * n + i)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = call;                                            \
    if (err_ != cudaSuccess) {                                          \
        fprintf(stderr, "[rank %d] CUDA error at %s:%d: %s\n",         \
                mpiRank, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1);                                   \
    }                                                                   \
} while (0)

static int mpiRank = 0;   // forward declaration for CUDA_CHECK macro

// ---------------------------------------------------------------------------
// CUDA kernel: one iteration (k) of Floyd-Warshall on a local row block
// ---------------------------------------------------------------------------
__global__ void floydKernel(unsigned int* __restrict__ localDist,
                            unsigned int* __restrict__ localPath,
                            const unsigned int* __restrict__ rowK,
                            int localN, int N, int k) {
    int local_i = blockIdx.y * blockDim.y + threadIdx.y;
    int j       = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_i < localN && j < N) {
        unsigned int* distRow = localDist + local_i * N;
        unsigned int* pathRow = localPath + local_i * N;

        unsigned int distIK = distRow[k];          // dist[global_i][k]
        unsigned int newDist = distIK + rowK[j];   // dist[global_i][k] + dist[k][j]

        if (newDist < distRow[j]) {
            distRow[j] = newDist;
            pathRow[j] = k;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers (called by rank 0 only, outside the hot path)
// ---------------------------------------------------------------------------
void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    #pragma omp parallel
    {
        unsigned int seed = 42 + omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < numNodes * numNodes; ++i) {
            dist[i] = rangeMin +
                static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path,
                          const size_t numNodes) {
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// ---------------------------------------------------------------------------
// Validation (called on rank 0 after gather, parallelised with OpenMP)
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes) {
    bool valid = true;

    // 1. Diagonal should be zero
    #pragma omp parallel for reduction(&& : valid)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            valid = false;
        }
    }
    if (!valid) {
        printf("Validation failed: diagonal element is not zero\n");
        return false;
    }

    // 2. Triangle inequality (sample a small subset)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            #pragma omp parallel for reduction(&& : valid)
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        valid = false;
                    }
                }
            }
            if (!valid) {
                printf("Validation failed: triangle inequality violated "
                       "at [%zu,%zu]\n", i, j);
                return false;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Usage
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
// Main – hybrid MPI + OpenMP + CUDA Floyd-Warshall
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Initialise MPI with thread support for OpenMP overlap
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    int nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    // ---- argument parsing (all ranks parse identically) -----------------
    size_t numNodes = 512;
    bool   validate = false;
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

    // ---- CUDA device selection ------------------------------------------
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        fprintf(stderr, "[rank %d] No CUDA-capable device found\n", mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int devID = mpiRank % numDevices;
    CUDA_CHECK(cudaSetDevice(devID));

    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, devID));

    // ---- banner ---------------------------------------------------------
    if (mpiRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (Hybrid)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks:       %d\n", nRanks);
        printf("GPUs per node:   %d  (rank %d using %s)\n",
               numDevices, devID, devProp.name);
        printf("Validation:      %s\n", validate ? "enabled" : "disabled");
    }

    // ---- row distribution -----------------------------------------------
    size_t rowsPerRank = (numNodes + nRanks - 1) / nRanks;   // ceil
    size_t localStart  = mpiRank * rowsPerRank;
    size_t localEnd    = std::min(localStart + rowsPerRank, numNodes);
    size_t localN      = localEnd - localStart;

    // Allocate host-side local buffers
    std::vector<unsigned int> localDist(localN * numNodes);
    std::vector<unsigned int> localPath(localN * numNodes);

    // ---- rank 0 initialises the full matrices --------------------------
    std::vector<unsigned int> fullDist;
    std::vector<unsigned int> fullPath;
    if (mpiRank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }

    // ---- scatter --------------------------------------------------------
    std::vector<int> sendCounts(nRanks);
    std::vector<int> displacements(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        size_t rStart = static_cast<size_t>(r) * rowsPerRank;
        size_t rEnd   = std::min(rStart + rowsPerRank, numNodes);
        size_t rN     = rEnd - rStart;
        sendCounts[r]     = static_cast<int>(rN * numNodes);
        displacements[r]  = static_cast<int>(rStart * numNodes);
    }

    MPI_Scatterv(mpiRank == 0 ? fullDist.data() : nullptr,
                 sendCounts.data(), displacements.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localN * numNodes),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    MPI_Scatterv(mpiRank == 0 ? fullPath.data() : nullptr,
                 sendCounts.data(), displacements.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localN * numNodes),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Free full matrices on rank 0 (no longer needed on rank 0 either)
    if (mpiRank == 0) {
        std::vector<unsigned int>().swap(fullDist);
        std::vector<unsigned int>().swap(fullPath);
    }

    // ---- GPU allocations ------------------------------------------------
    unsigned int *d_localDist = nullptr, *d_localPath = nullptr;
    unsigned int *d_rowK      = nullptr;

    size_t localBytes = localN * numNodes * sizeof(unsigned int);
    size_t rowBytes   = numNodes * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&d_localDist, localBytes));
    CUDA_CHECK(cudaMalloc(&d_localPath, localBytes));
    CUDA_CHECK(cudaMalloc(&d_rowK,      rowBytes));

    CUDA_CHECK(cudaMemcpy(d_localDist, localDist.data(), localBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_localPath, localPath.data(), localBytes,
                          cudaMemcpyHostToDevice));

    // Host buffer for MPI broadcasts of the pivot row
    std::vector<unsigned int> hostRowK(numNodes);

    // ---- OpenMP thread/stream setup ------------------------------------
    int ompThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        ompThreads = omp_get_num_threads();
    }
    // Use at most as many streams as local rows (no benefit in more)
    int nStreams = std::min(ompThreads, static_cast<int>(localN));
    if (nStreams < 1) nStreams = 1;

    std::vector<cudaStream_t> streams(nStreams);
    for (int t = 0; t < nStreams; ++t) {
        CUDA_CHECK(cudaStreamCreate(&streams[t]));
    }

    // Pre-compute per-stream row chunks
    std::vector<int> chunkStart(nStreams), chunkN(nStreams);
    for (int t = 0; t < nStreams; ++t) {
        chunkStart[t] = static_cast<int>(
            (static_cast<size_t>(t) * localN) / nStreams);
        int end = static_cast<int>(
            (static_cast<size_t>(t + 1) * localN) / nStreams);
        chunkN[t] = end - chunkStart[t];
    }

    // ---- compute: hybrid parallel Floyd-Warshall -----------------------
    if (mpiRank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;

    for (size_t k = 0; k < numNodes; ++k) {
        int owner = static_cast<int>(k / rowsPerRank);

        // --- owner extracts row k from GPU and broadcasts ---------------
        if (mpiRank == owner) {
            size_t localK = k - localStart;
            CUDA_CHECK(cudaMemcpy(hostRowK.data(),
                                  d_localDist + localK * numNodes,
                                  rowBytes,
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hostRowK.data(), static_cast<int>(numNodes),
                  MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // --- copy row k to GPU ------------------------------------------
        CUDA_CHECK(cudaMemcpy(d_rowK, hostRowK.data(), rowBytes,
                              cudaMemcpyHostToDevice));

        // --- launch CUDA kernels from OpenMP threads (one per stream) ---
        #pragma omp parallel for schedule(static, 1) if (nStreams > 1)
        for (int t = 0; t < nStreams; ++t) {
            if (chunkN[t] == 0) continue;

            dim3 blockDim(BLOCK_X, BLOCK_Y);
            dim3 gridDim(
                (static_cast<int>(numNodes) + BLOCK_X - 1) / BLOCK_X,
                (chunkN[t] + BLOCK_Y - 1) / BLOCK_Y);

            unsigned int* chunkDist = d_localDist +
                static_cast<size_t>(chunkStart[t]) * numNodes;
            unsigned int* chunkPath = d_localPath +
                static_cast<size_t>(chunkStart[t]) * numNodes;

            floydKernel<<<gridDim, blockDim, 0, streams[t]>>>(
                chunkDist, chunkPath, d_rowK, chunkN[t],
                static_cast<int>(numNodes), static_cast<int>(k));
        }
    }

    // Wait for all streams to finish
    for (int t = 0; t < nStreams; ++t) {
        CUDA_CHECK(cudaStreamSynchronize(streams[t]));
    }

    double endTime = MPI_Wtime();
    double durationMs = (endTime - startTime) * 1000.0;
    MPI_Allreduce(MPI_IN_PLACE, &durationMs, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    // ---- copy results back to host -------------------------------------
    CUDA_CHECK(cudaMemcpy(localDist.data(), d_localDist, localBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(localPath.data(), d_localPath, localBytes,
                          cudaMemcpyDeviceToHost));

    // ---- cleanup GPU resources -----------------------------------------
    for (int t = 0; t < nStreams; ++t) {
        CUDA_CHECK(cudaStreamDestroy(streams[t]));
    }
    CUDA_CHECK(cudaFree(d_localDist));
    CUDA_CHECK(cudaFree(d_localPath));
    CUDA_CHECK(cudaFree(d_rowK));

    // ---- gather results on rank 0 --------------------------------------
    if (mpiRank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
        // Copy rank 0's own block
        std::memcpy(fullDist.data(), localDist.data(),
                    localN * numNodes * sizeof(unsigned int));
        std::memcpy(fullPath.data(), localPath.data(),
                    localN * numNodes * sizeof(unsigned int));
    }

    // Non-blocking gather (rank 0 receives, others send)
    std::vector<MPI_Request> gatherReqs;
    if (mpiRank == 0) {
        gatherReqs.resize(2 * (nRanks - 1));
        int idx = 0;
        for (int r = 1; r < nRanks; ++r) {
            size_t rStart = static_cast<size_t>(r) * rowsPerRank;
            size_t rEnd   = std::min(rStart + rowsPerRank, numNodes);
            size_t rN     = rEnd - rStart;
            MPI_Irecv(fullDist.data() + rStart * numNodes,
                      static_cast<int>(rN * numNodes),
                      MPI_UNSIGNED, r, 0, MPI_COMM_WORLD, &gatherReqs[idx++]);
            MPI_Irecv(fullPath.data() + rStart * numNodes,
                      static_cast<int>(rN * numNodes),
                      MPI_UNSIGNED, r, 1, MPI_COMM_WORLD, &gatherReqs[idx++]);
        }
    } else {
        gatherReqs.resize(2);
        MPI_Isend(localDist.data(), static_cast<int>(localN * numNodes),
                  MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &gatherReqs[0]);
        MPI_Isend(localPath.data(), static_cast<int>(localN * numNodes),
                  MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, &gatherReqs[1]);
    }
    MPI_Waitall(static_cast<int>(gatherReqs.size()), gatherReqs.data(),
                MPI_STATUSES_IGNORE);

    // ---- output ---------------------------------------------------------
    if (mpiRank == 0) {
        printf("Computation time: %.0f ms\n", durationMs);

        double ops   = static_cast<double>(numNodes) *
                       static_cast<double>(numNodes) *
                       static_cast<double>(numNodes);
        double gflops = ops / (durationMs / 1000.0) / 1.0e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool ok = validateResult(fullDist, numNodes);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return 0;
}
