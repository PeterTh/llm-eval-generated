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
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
        }
    }
}

// Each rank owns complete source rows. The pivot row is immutable during its
// iteration because the diagonal is zero and all edge weights are positive.
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

__global__ void updateRows(unsigned int* dist, unsigned int* path,
                           const unsigned int* pivot, size_t n, size_t rows,
                           size_t k) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y;
    if (i < rows && j < n) {
        size_t pos = i * n + j;
        unsigned int candidate = dist[i * n + k] + pivot[j];
        if (candidate < dist[pos]) {
            dist[pos] = candidate;
            path[pos] = static_cast<unsigned int>(k);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, int rank, int ranks, const std::vector<int>& counts,
                   const std::vector<int>& offsets) {
    size_t first = n ? static_cast<size_t>(offsets[rank]) / n : 0;
    size_t rows = n ? static_cast<size_t>(counts[rank]) / n : 0;
    std::vector<unsigned int> localDist(counts[rank]), localPath(counts[rank]);
    MPI_Scatterv(dist.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(path.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                 localPath.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    size_t bytes = localDist.size() * sizeof(unsigned int);
    if (bytes) {
        CUDA_CHECK(cudaMalloc(&dDist, bytes));
        CUDA_CHECK(cudaMalloc(&dPath, bytes));
        CUDA_CHECK(cudaMemcpy(dDist, localDist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, localPath.data(), bytes, cudaMemcpyHostToDevice));
    }
    std::vector<unsigned int> pivot(n);
    if (ranks > 1 && n) CUDA_CHECK(cudaMalloc(&dPivot, n * sizeof(unsigned int)));
    for (size_t k = 0; k < n; ++k) {
        const unsigned int* devicePivot;
        if (ranks == 1) {
            devicePivot = dDist + k * n;
        } else {
            int owner = 0;
            while (owner + 1 < ranks && static_cast<size_t>(offsets[owner + 1]) / n <= k) ++owner;
            if (rank == owner) {
                CUDA_CHECK(cudaMemcpy(pivot.data(), dDist + (k - first) * n,
                                      n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            }
            MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(dPivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
            devicePivot = dPivot;
        }
        if (rows) {
            dim3 grid(static_cast<unsigned int>((n + 255) / 256), static_cast<unsigned int>(rows));
            updateRows<<<grid, 256>>>(dDist, dPath, devicePivot, n, rows, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (bytes) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data(), dPath, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dDist));
        CUDA_CHECK(cudaFree(dPath));
    }
    if (dPivot) CUDA_CHECK(cudaFree(dPivot));
    MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, dist.data(), counts.data(),
                offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), counts[rank], MPI_UNSIGNED, path.data(), counts.data(),
                offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            unsigned long long value = strtoull(argv[++i], &end, 10);
            if (!*argv[i] || *end || argv[i][0] == '-' || value > 46340) {
                if (!rank) fprintf(stderr, "Invalid node count (must be between 0 and 46340)\n");
                MPI_Finalize(); return 1;
            }
            numNodes = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks);
    size_t offset = 0;
    for (int r = 0; r < ranks; ++r) {
        size_t rows = numNodes / ranks + (static_cast<size_t>(r) < numNodes % ranks);
        offsets[r] = static_cast<int>(offset * numNodes);
        counts[r] = static_cast<int>(rows * numNodes);
        offset += rows;
    }
    std::vector<unsigned int> dist, path;
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        dist.resize(numNodes * numNodes); path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes, rank, ranks, counts, offsets);
    auto end = std::chrono::high_resolution_clock::now();
    int status = 0;
    if (!rank) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / (duration.count() / 1000.0) / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
