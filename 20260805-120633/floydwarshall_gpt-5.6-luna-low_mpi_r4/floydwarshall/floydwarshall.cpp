#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t firstRow, const size_t rowCount,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // Preserve the reference's column-major RNG sequence while storing rows locally.
    for (size_t col = 0; col < numNodes; ++col)
        for (size_t globalRow = 0; globalRow < numNodes; ++globalRow) {
            unsigned int value = rangeMin +
                (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (globalRow >= firstRow && globalRow < firstRow + rowCount)
                dist[(globalRow - firstRow) * numNodes + col] = value;
        }
    for (size_t row = 0; row < rowCount; ++row)
        dist[row * numNodes + firstRow + row] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t firstRow, const size_t rowCount) {
    for (size_t row = 0; row < rowCount; ++row) {
        const size_t globalRow = firstRow + row;
        for (size_t col = 0; col < numNodes; ++col)
            path[row * numNodes + col] = col;
        path[row * numNodes + globalRow] = globalRow;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t firstRow, const size_t rowCount,
                   MPI_Comm comm) {
    std::vector<unsigned int> pivot(numNodes);
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = 0;
        // Contiguous block distribution, matching the row ownership below.
        // The owner is found cheaply because row blocks differ by at most one.
        for (int r = 0; r < 1; ++r) { int size; MPI_Comm_size(comm, &size); owner = 0;
            while (owner < size && !(k >= (size_t)((numNodes * owner) / size) &&
                                     k < (size_t)((numNodes * (owner + 1)) / size))) ++owner; }
        size_t ownerFirst; int size; MPI_Comm_size(comm, &size);
        ownerFirst = (numNodes * owner) / size;
        if (firstRow == ownerFirst && k >= firstRow && k < firstRow + rowCount)
            std::copy_n(&dist[(k - firstRow) * numNodes], numNodes, pivot.begin());
        MPI_Bcast(pivot.data(), (int)numNodes, MPI_UNSIGNED, owner, comm);
        for (size_t row = 0; row < rowCount; ++row) {
            const unsigned int distIK = dist[row * numNodes + k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[row * numNodes + j];
                const unsigned int distKJ = pivot[j];
                const unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    dist[row * numNodes + j] = newDist;
                    path[row * numNodes + j] = k;
                }
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
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (rank == 0) { printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    printf("Number of nodes: %zu\nValidation: %s\n", numNodes, validate ? "enabled" : "disabled"); }
    const size_t firstRow = (numNodes * (size_t)rank) / worldSize;
    const size_t rowCount = (numNodes * (size_t)(rank + 1)) / worldSize - firstRow;
    
    // Allocate matrices
    std::vector<unsigned int> dist(rowCount * numNodes);
    std::vector<unsigned int> path(rowCount * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, firstRow, rowCount, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, firstRow, rowCount);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, firstRow, rowCount, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long localMs = duration.count(), elapsedMs = 0;
    MPI_Reduce(&localMs, &elapsedMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %ld ms\n", elapsedMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);

    std::vector<unsigned int> gathered;
    if (rank == 0) gathered.resize(numNodes * numNodes);
    std::vector<int> counts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) { size_t f = numNodes * (size_t)r / worldSize;
        size_t c = numNodes * (size_t)(r + 1) / worldSize - f; counts[r] = (int)(c * numNodes); displs[r] = (int)(f * numNodes); }
    MPI_Gatherv(dist.data(), (int)dist.size(), MPI_UNSIGNED, gathered.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank != 0) { MPI_Finalize(); return 0; }
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        std::vector<unsigned int> output(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) for (size_t j = 0; j < numNodes; ++j) output[idx2(j, i, numNodes)] = gathered[i * numNodes + j];
        print_results_int(output, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        std::vector<unsigned int> output(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) for (size_t j = 0; j < numNodes; ++j) output[idx2(j, i, numNodes)] = gathered[i * numNodes + j];
        bool valid = validateResult(output, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize(); return 0;
}
