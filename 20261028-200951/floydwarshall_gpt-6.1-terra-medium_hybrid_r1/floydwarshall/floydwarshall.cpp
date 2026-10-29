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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Each rank owns a consecutive set of source rows. The pivot row is
// replicated on all GPUs before updating the local rows for each k.
__global__ void updateRows(unsigned int* dist, unsigned int* path,
                           const unsigned int* pivot, size_t n,
                           size_t rows, size_t k) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (i < rows && j < n) {
        const size_t pos = i * n + j;
        const unsigned int candidate = dist[i * n + k] + pivot[j];
        if (candidate < dist[pos]) {
            dist[pos] = candidate;
            path[pos] = static_cast<unsigned int>(k);
        }
    }
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = strtoull(arg, &end, 10);
            if (*arg == '-' || *arg == '\0' || *end != '\0' || value > INT_MAX) {
                if (rank == 0) fprintf(stderr, "Invalid node count: %s\n", arg);
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
    // MPI's vector collectives use int counts and displacements.
    if (numNodes && numNodes > static_cast<size_t>(INT_MAX) / numNodes) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds MPI count limits\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks), rowStarts(ranks + 1);
    for (int r = 0; r < ranks; ++r) {
        rowStarts[r + 1] = rowStarts[r] + static_cast<int>(numNodes / ranks + (static_cast<size_t>(r) < numNodes % ranks));
        counts[r] = (rowStarts[r + 1] - rowStarts[r]) * static_cast<int>(numNodes);
        offsets[r] = rowStarts[r] * static_cast<int>(numNodes);
    }
    const size_t localRows = rowStarts[rank + 1] - rowStarts[rank];
    std::vector<unsigned int> dist, localDist(counts[rank]), localPath(counts[rank]);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\nValidation: %s\n", numNodes, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(dist.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows; ++i)
        for (size_t j = 0; j < numNodes; ++j)
            localPath[i * numNodes + j] = static_cast<unsigned int>(rowStarts[rank] + i);

    unsigned int *deviceDist = nullptr, *devicePath = nullptr, *devicePivot = nullptr;
    unsigned int* pivot = nullptr;
    if (numNodes) {
        cudaCheck(cudaMallocHost(&pivot, numNodes * sizeof(unsigned int)), "cudaMallocHost");
        cudaCheck(cudaMalloc(&devicePivot, numNodes * sizeof(unsigned int)), "cudaMalloc pivot");
        if (localRows) {
            cudaCheck(cudaMalloc(&deviceDist, localDist.size() * sizeof(unsigned int)), "cudaMalloc distance");
            cudaCheck(cudaMalloc(&devicePath, localPath.size() * sizeof(unsigned int)), "cudaMalloc path");
            cudaCheck(cudaMemcpy(deviceDist, localDist.data(), localDist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy distance");
            cudaCheck(cudaMemcpy(devicePath, localPath.data(), localPath.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy path");
        }
    }
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    int owner = 0;
    for (size_t k = 0; k < numNodes; ++k) {
        while (k >= static_cast<size_t>(rowStarts[owner + 1])) ++owner;
        if (rank == owner)
            cudaCheck(cudaMemcpy(pivot, deviceDist + (k - rowStarts[rank]) * numNodes,
                                 numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy pivot to host");
        MPI_Bcast(pivot, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        if (localRows) {
            cudaCheck(cudaMemcpy(devicePivot, pivot, numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy pivot to device");
            // Grid-stride in rows is unnecessary for the supported MPI matrix size
            // except when there are more than 65535 local rows; that cannot occur
            // with a square matrix whose element count fits in an int.
            dim3 grid((numNodes + 255) / 256, localRows);
            updateRows<<<grid, 256>>>(deviceDist, devicePath, devicePivot, numNodes, localRows, k);
            cudaCheck(cudaGetLastError(), "update kernel");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "finish computation");
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (localRows) cudaCheck(cudaMemcpy(localDist.data(), deviceDist, localDist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy result");
    MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, dist.data(), counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double seconds = std::chrono::duration<double>(end - start).count();
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (deviceDist) cudaFree(deviceDist);
    if (devicePath) cudaFree(devicePath);
    if (devicePivot) cudaFree(devicePivot);
    if (pivot) cudaFreeHost(pivot);
    MPI_Finalize();
    return result;
}
