#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
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

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes, size_t firstRow, size_t localRows,
                   int rank, const std::vector<int>& rowCounts) {
    std::vector<unsigned int> pivot(numNodes);
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        const size_t processCount = rowCounts.size();
        const size_t baseRows = numNodes / processCount;
        const size_t extraRows = numNodes % processCount;
        const size_t firstRegion = (baseRows + 1) * extraRows;
        const int owner = static_cast<int>(k < firstRegion
            ? k / (baseRows + 1)
            : extraRows + (k - firstRegion) / baseRows);
        if (rank == owner) {
            const size_t localK = k - firstRow;
            std::copy_n(dist.data() + localK * numNodes, numNodes, pivot.data());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Each rank owns a contiguous block of the first matrix index. The
        // pivot vector is read-only, so local rows can be updated in parallel.
        for (size_t localJ = 0; localJ < localRows; ++localJ) {
            const unsigned int distJK = dist[localJ * numNodes + k];
            unsigned int* row = dist.data() + localJ * numNodes;
            unsigned int* pathRow = path.data() + localJ * numNodes;
            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDist = pivot[i] + distJK;
                const unsigned int distIJ = row[i];
                if (newDist < distIJ) {
                    row[i] = newDist;
                    pathRow[i] = static_cast<unsigned int>(k);
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

    if (numNodes == 0 || numNodes > static_cast<size_t>(INT_MAX) ||
        numNodes > static_cast<size_t>(INT_MAX) / static_cast<size_t>(worldSize)) {
        if (rank == 0) fprintf(stderr, "Invalid node count for MPI matrix communication\n");
        MPI_Finalize();
        return 1;
    }
    const size_t totalElements = numNodes * numNodes;
    if (totalElements > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds MPI count limits\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> rowCounts(worldSize), rowDisplacements(worldSize);
    size_t offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = numNodes / static_cast<size_t>(worldSize) +
                            (static_cast<size_t>(r) < numNodes % static_cast<size_t>(worldSize));
        rowCounts[r] = static_cast<int>(rows * numNodes);
        rowDisplacements[r] = static_cast<int>(offset);
        offset += rows * numNodes;
    }
    const size_t firstRow = (numNodes / static_cast<size_t>(worldSize)) * rank +
                            std::min(static_cast<size_t>(rank), numNodes % static_cast<size_t>(worldSize));
    const size_t localRows = rowCounts[rank] / numNodes;
    
    // Allocate matrices
    std::vector<unsigned int> dist(static_cast<size_t>(rowCounts[rank]));
    std::vector<unsigned int> path(static_cast<size_t>(rowCounts[rank]));
    std::vector<unsigned int> fullDist, fullPath;
    if (rank == 0) {
        fullDist.resize(totalElements);
        fullPath.resize(totalElements);
    }
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }
    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, rowCounts.data(), rowDisplacements.data(), MPI_UNSIGNED,
                 dist.data(), rowCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullPath.data() : nullptr, rowCounts.data(), rowDisplacements.data(), MPI_UNSIGNED,
                 path.data(), rowCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    floydWarshall(dist, path, numNodes, firstRow, localRows, rank, rowCounts);
    
    double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gatherv(dist.data(), rowCounts[rank], MPI_UNSIGNED,
                rank == 0 ? fullDist.data() : nullptr, rowCounts.data(), rowDisplacements.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    if (rank == 0) {
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = maxElapsed > 0 ? ops / maxElapsed / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(fullDist, "DistanceMatrix");
    }
    
    // Validation
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(fullDist, numNodes) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
