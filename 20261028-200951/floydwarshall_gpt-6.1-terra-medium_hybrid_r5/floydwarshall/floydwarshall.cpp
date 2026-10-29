#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

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

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Each thread updates one destination in a local source row. The pivot row
// and pivot column do not change during an iteration (the diagonal is zero).
__global__ void updateRows(unsigned int* dist, unsigned int* path,
                           const unsigned int* pivot, size_t n, size_t rows,
                           size_t firstRow, size_t k) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y;
    if (i >= rows || j >= n || firstRow + i == k || j == k) return;
    size_t pos = i * n + j;
    unsigned int candidate = dist[i * n + k] + pivot[j];
    if (candidate < dist[pos]) {
        dist[pos] = candidate;
        path[pos] = static_cast<unsigned int>(k);
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            unsigned long long value = strtoull(argv[++i], &end, 10);
            if (*argv[i] == '-' || *argv[i] == '\0' || *end || value > INT_MAX) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            numNodes = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    // MPI_Scatterv/Gatherv use int counts and displacements.
    if (numNodes && numNodes > static_cast<size_t>(INT_MAX) / numNodes) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds MPI count limits\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks), starts(ranks + 1);
    for (int r = 0; r < ranks; ++r) {
        starts[r + 1] = starts[r] + static_cast<int>(numNodes / ranks + (static_cast<size_t>(r) < numNodes % ranks));
        counts[r] = (starts[r + 1] - starts[r]) * static_cast<int>(numNodes);
        offsets[r] = starts[r] * static_cast<int>(numNodes);
    }
    size_t rows = starts[rank + 1] - starts[rank];
    size_t localSize = rows * numNodes;
    std::vector<unsigned int> dist;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", numNodes, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    std::vector<unsigned int> local(localSize), path(localSize);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < numNodes; ++j) path[i * numNodes + j] = static_cast<unsigned int>(j);
    MPI_Scatterv(dist.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                 local.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    unsigned int *deviceDist = nullptr, *devicePath = nullptr, *devicePivot = nullptr;
    if (localSize) {
        CUDA_CHECK(cudaMalloc(&deviceDist, localSize * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&devicePath, localSize * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(deviceDist, local.data(), localSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, path.data(), localSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    std::vector<unsigned int> pivot(numNodes);
    if (numNodes && ranks > 1) CUDA_CHECK(cudaMalloc(&devicePivot, numNodes * sizeof(unsigned int)));
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int owner = 0;
    for (size_t k = 0; k < numNodes; ++k) {
        while (k >= static_cast<size_t>(starts[owner + 1])) ++owner;
        const unsigned int* pivotDevice;
        if (ranks == 1) {
            pivotDevice = deviceDist + k * numNodes;
        } else {
            if (rank == owner)
                CUDA_CHECK(cudaMemcpy(pivot.data(), deviceDist + (k - starts[rank]) * numNodes,
                                      numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(devicePivot, pivot.data(), numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
            pivotDevice = devicePivot;
        }
        if (rows) {
            dim3 block(256), grid((numNodes + 255) / 256, rows);
            updateRows<<<grid, block>>>(deviceDist, devicePath, pivotDevice, numNodes, rows, starts[rank], k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localSize) CUDA_CHECK(cudaMemcpy(local.data(), deviceDist, localSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (validate || printResults)
        MPI_Gatherv(local.data(), counts[rank], MPI_UNSIGNED, dist.data(), counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    int status = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / maxElapsed / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (deviceDist) CUDA_CHECK(cudaFree(deviceDist));
    if (devicePath) CUDA_CHECK(cudaFree(devicePath));
    if (devicePivot) CUDA_CHECK(cudaFree(devicePivot));
    MPI_Finalize();
    return status;
}
