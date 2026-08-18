#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t firstRow,
                   const std::vector<int>& rowOwner, MPI_Comm comm) {
    std::vector<unsigned int> pivot(numNodes);
    const size_t localRows = dist.size() / numNodes;
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[k];
        if (k >= firstRow && k < firstRow + localRows) {
            std::copy_n(dist.data() + (k - firstRow) * numNodes,
                        numNodes, pivot.data());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, comm);

        for (size_t i = 0; i < localRows; ++i) {
            unsigned int* const row = dist.data() + i * numNodes;
            unsigned int* const pathRow = path.data() + i * numNodes;
            const unsigned int distIK = row[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivot[j];
                if (newDist < row[j]) {
                    row[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
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
    int rank = 0, worldSize = 1;
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
    
    if (numNodes == 0 || numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        numNodes * numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) fprintf(stderr, "Matrix is empty or too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> rows(worldSize), counts(worldSize), displs(worldSize);
    std::vector<int> rowOwner(numNodes);
    size_t offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        rows[r] = static_cast<int>(numNodes / worldSize +
                                   (static_cast<size_t>(r) < numNodes % worldSize));
        counts[r] = rows[r] * static_cast<int>(numNodes);
        displs[r] = static_cast<int>(offset * numNodes);
        for (int q = 0; q < rows[r]; ++q) rowOwner[offset + q] = r;
        offset += rows[r];
    }
    const size_t firstRow = static_cast<size_t>(displs[rank]) / numNodes;
    
    // Allocate matrices
    std::vector<unsigned int> fullDist, fullPath;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
    }
    std::vector<unsigned int> dist(counts[rank]);
    std::vector<unsigned int> path(counts[rank]);
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }
    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, counts.data(), displs.data(),
                 MPI_UNSIGNED, dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullPath.data() : nullptr, counts.data(), displs.data(),
                 MPI_UNSIGNED, path.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    fullDist.clear(); fullDist.shrink_to_fit();
    fullPath.clear(); fullPath.shrink_to_fit();
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, firstRow, rowOwner, MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / seconds / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);

    if (printResults || validate) {
        if (rank == 0) fullDist.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr, counts.data(), displs.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(fullDist, "DistanceMatrix");
    }
    
    // Validation
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(fullDist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
