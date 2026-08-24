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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = call; \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_err)); \
            MPI_Abort(MPI_COMM_WORLD, _err); \
        } \
    } while (0)

// Index calculation for flattened 2D array (row-major: dist[row * n + col])
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// CUDA kernel: updates local rows for one iteration k of Floyd-Warshall
// Each thread handles one (i, j) element.
// ---------------------------------------------------------------------------
__global__ void floydKernel(unsigned int* __restrict__ localDist,
                             unsigned int* __restrict__ localPath,
                             const unsigned int* __restrict__ kRow,
                             unsigned int kCol,
                             unsigned int kVal,
                             unsigned int n,
                             unsigned int localRows) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localRows && j < n) {
        unsigned int distIK = localDist[i * n + kCol];
        unsigned int distKJ = kRow[j];
        unsigned int newDist = distIK + distKJ;
        unsigned int oldDist = localDist[i * n + j];
        if (newDist < oldDist) {
            localDist[i * n + j] = newDist;
            localPath[i * n + j] = kVal;
        }
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                               const size_t numNodes,
                               const unsigned int rangeMin,
                               const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
            range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    // Set diagonal to zero
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path,
                           const size_t numNodes) {
    #pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

bool validateResult(const std::vector<unsigned int>& dist,
                     const size_t numNodes) {
    // 1. Diagonal check
    bool diagOk = true;
    #pragma omp parallel for reduction(&& : diagOk)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) diagOk = false;
    }
    if (!diagOk) {
        printf("Validation failed: diagonal element is not zero\n");
        return false;
    }

    // 2. Triangle inequality on a sample
    bool triOk = true;
    size_t limit = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < limit && triOk; ++i) {
        for (size_t j = 0; j < limit && triOk; ++j) {
            #pragma omp parallel for reduction(&& : triOk)
            for (size_t kk = 0; kk < numNodes; ++kk) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(kk, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, kk, numNodes)];
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) triOk = false;
                }
            }
        }
    }

    if (!triOk) {
        printf("Validation failed: triangle inequality violated\n");
        return false;
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

// ---------------------------------------------------------------------------
// Helper: find which MPI rank owns a given global row and its local index
// ---------------------------------------------------------------------------
static inline void rowOwner(size_t k, size_t numNodes, int size,
                             int* owner, size_t* localK) {
    size_t base = numNodes / static_cast<size_t>(size);
    size_t rem  = numNodes % static_cast<size_t>(size);
    if (k < rem * (base + 1)) {
        *owner  = static_cast<int>(k / (base + 1));
        *localK = k - static_cast<size_t>(*owner) * (base + 1);
    } else {
        size_t offset = k - rem * (base + 1);
        *owner  = static_cast<int>(rem + offset / base);
        *localK = offset - static_cast<size_t>(*owner - static_cast<int>(rem)) * base;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Set CUDA device (round-robin across visible GPUs)
    {
        int devCount = 0;
        cudaGetDeviceCount(&devCount);
        if (devCount > 0) {
            cudaSetDevice(rank % devCount);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // ----  Parse arguments  ------------------------------------------------
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
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

    // ----  Row distribution  -----------------------------------------------
    size_t base = numNodes / static_cast<size_t>(mpiSize);
    size_t rem  = numNodes % static_cast<size_t>(mpiSize);

    size_t localRows;
    if (static_cast<size_t>(rank) < rem) {
        localRows = base + 1;
    } else {
        localRows = base;
    }

    // Build scatterv / gatherv descriptors
    std::vector<int> scounts(mpiSize), sdispls(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        size_t rRows, rStart;
        if (static_cast<size_t>(r) < rem) {
            rRows  = base + 1;
            rStart = static_cast<size_t>(r) * rRows;
        } else {
            rRows  = base;
            rStart = rem * (base + 1) +
                     (static_cast<size_t>(r) - rem) * base;
        }
        scounts[r] = static_cast<int>(rRows * numNodes);
        sdispls[r] = static_cast<int>(rStart * numNodes);
    }

    // ----  Allocate host memory  -------------------------------------------
    std::vector<unsigned int> h_fullDist;
    std::vector<unsigned int> h_fullPath;
    if (rank == 0) {
        h_fullDist.resize(numNodes * numNodes);
        h_fullPath.resize(numNodes * numNodes);
    }

    std::vector<unsigned int> h_localDist(localRows * numNodes);
    std::vector<unsigned int> h_localPath(localRows * numNodes);
    std::vector<unsigned int> h_kRow(numNodes);

    // ----  Initialise on rank 0, then scatter  -----------------------------
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", mpiSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");

        initializeDistanceMatrix(h_fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(h_fullPath, numNodes);
    }

    MPI_Scatterv(rank == 0 ? h_fullDist.data() : nullptr,
                 scounts.data(), sdispls.data(), MPI_UNSIGNED,
                 h_localDist.data(), static_cast<int>(localRows * numNodes),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_fullPath.data() : nullptr,
                 scounts.data(), sdispls.data(), MPI_UNSIGNED,
                 h_localPath.data(), static_cast<int>(localRows * numNodes),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // ----  GPU memory allocation  ------------------------------------------
    unsigned int *d_localDist = nullptr, *d_localPath = nullptr;
    unsigned int *d_kRow      = nullptr;

    CUDA_CHECK(cudaMalloc(&d_localDist, localRows * numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_localPath, localRows * numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_kRow,      numNodes * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_localDist, h_localDist.data(),
                          localRows * numNodes * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_localPath, h_localPath.data(),
                          localRows * numNodes * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // ----  CUDA kernel launch configuration  -------------------------------
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (static_cast<unsigned int>(numNodes)  + blockDim.x - 1) / blockDim.x,
        (static_cast<unsigned int>(localRows) + blockDim.y - 1) / blockDim.y
    );

    // ----  Timed computation  ----------------------------------------------
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < numNodes; ++k) {
        // Which rank owns row k ?
        int    owner;
        size_t localK;
        rowOwner(k, numNodes, mpiSize, &owner, &localK);

        // 1. Owner copies its k-th row from device -> host
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(h_kRow.data(),
                                  d_localDist + localK * numNodes,
                                  numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        // 2. Broadcast the pivot row to every rank
        MPI_Bcast(h_kRow.data(), static_cast<int>(numNodes),
                  MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // 3. Copy pivot row to device
        CUDA_CHECK(cudaMemcpy(d_kRow, h_kRow.data(),
                              numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // 4. Launch GPU kernel to update local rows
        floydKernel<<<gridDim, blockDim>>>(
            d_localDist, d_localPath, d_kRow,
            static_cast<unsigned int>(k),
            static_cast<unsigned int>(k), // kVal = intermediate node
            static_cast<unsigned int>(numNodes),
            static_cast<unsigned int>(localRows));

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long localDurationMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long durationMs = 0;
    MPI_Reduce(&localDurationMs, &durationMs, 1, MPI_LONG_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // ----  Copy results back  ----------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_localDist.data(), d_localDist,
                          localRows * numNodes * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_localPath.data(), d_localPath,
                          localRows * numNodes * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_localDist));
    CUDA_CHECK(cudaFree(d_localPath));
    CUDA_CHECK(cudaFree(d_kRow));

    // ----  Gather results on rank 0  ---------------------------------------
    MPI_Gatherv(h_localDist.data(), static_cast<int>(localRows * numNodes),
                MPI_UNSIGNED,
                rank == 0 ? h_fullDist.data() : nullptr,
                scounts.data(), sdispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // ----  Report  ---------------------------------------------------------
    if (rank == 0) {
        printf("Computation time: %lld ms\n", durationMs);

        double ops = static_cast<double>(numNodes) *
                     static_cast<double>(numNodes) *
                     static_cast<double>(numNodes);
        double perf = ops / (static_cast<double>(durationMs) / 1000.0) / 1.0e9;
        printf("Performance: %.3f GOPS\n", perf);

        if (printResults) {
            print_results_int(h_fullDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(h_fullDist, numNodes);
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
