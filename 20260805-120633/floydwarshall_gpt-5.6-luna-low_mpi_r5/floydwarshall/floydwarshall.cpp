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

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t numNodes, size_t firstRow, size_t localRows, int rank, int worldSize) {
    std::vector<unsigned int> pivot(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        // The rank owning row k broadcasts that row. Blocks are nearly even.
        int pivotOwner = 0;
        size_t base = numNodes / static_cast<size_t>(worldSize);
        size_t rem = numNodes % static_cast<size_t>(worldSize);
        size_t cutoff = (base + 1) * rem;
        if (k < cutoff) pivotOwner = static_cast<int>(k / (base + 1));
        else pivotOwner = static_cast<int>(rem + (k - cutoff) / base);
        if (rank == pivotOwner) {
            size_t localK = k - firstRow;
            for (size_t j = 0; j < numNodes; ++j) pivot[j] = dist[j * localRows + localK];
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, pivotOwner, MPI_COMM_WORLD);
        for (size_t i = 0; i < localRows; ++i) {
            const unsigned int dik = dist[k * localRows + i];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int candidate = dik + pivot[j];
                unsigned int& dij = dist[j * localRows + i];
                if (candidate < dij) { dij = candidate; path[j * localRows + i] = static_cast<unsigned int>(k); }
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    
    if (rank == 0) { printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", numNodes, validate ? "enabled" : "disabled"); }
    
    // Allocate matrices
    const size_t base = numNodes / static_cast<size_t>(size), rem = numNodes % static_cast<size_t>(size);
    const size_t firstRow = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t localRows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    std::vector<unsigned int> globalDist(rank == 0 ? numNodes * numNodes : 0), globalPath(rank == 0 ? numNodes * numNodes : 0);
    std::vector<unsigned int> packedDist(rank == 0 ? numNodes * numNodes : 0), packedPath(rank == 0 ? numNodes * numNodes : 0);
    std::vector<unsigned int> dist(localRows * numNodes), path(localRows * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    if (rank == 0) { initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE); initializePathMatrix(globalPath, numNodes); }
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) { size_t fr = static_cast<size_t>(r)*base + std::min(static_cast<size_t>(r), rem); size_t nr = base + (static_cast<size_t>(r)<rem); counts[r] = static_cast<int>(nr*numNodes); displs[r] = static_cast<int>(fr*numNodes); }
    if (rank == 0) for (int r = 0; r < size; ++r) { size_t fr = static_cast<size_t>(r)*base + std::min(static_cast<size_t>(r), rem); size_t nr = base + (static_cast<size_t>(r)<rem); for (size_t i=0; i<nr; ++i) for (size_t j=0; j<numNodes; ++j) { packedDist[displs[r]+i*numNodes+j] = globalDist[idx2(j,fr+i,numNodes)]; packedPath[displs[r]+i*numNodes+j] = globalPath[idx2(j,fr+i,numNodes)]; } }
    MPI_Scatterv(rank == 0 ? packedDist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? packedPath.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, path.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD); auto start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, firstRow, localRows, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); double durationSeconds = MPI_Wtime() - start;
    
    double maxSeconds = 0; MPI_Reduce(&durationSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    if (rank == 0) printf("Computation time: %ld ms\nPerformance: %.3f GOPS\n", (long)(maxSeconds*1000), ops/maxSeconds/1e9);
    MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? packedDist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) for (int r = 0; r < size; ++r) { size_t fr = static_cast<size_t>(r)*base + std::min(static_cast<size_t>(r), rem); size_t nr = base + (static_cast<size_t>(r)<rem); for (size_t i=0; i<nr; ++i) for (size_t j=0; j<numNodes; ++j) globalDist[idx2(j,fr+i,numNodes)] = packedDist[displs[r]+i*numNodes+j]; }
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        if (rank == 0) print_results_int(globalDist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = rank == 0 ? validateResult(globalDist, numNodes) : true;
        
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
