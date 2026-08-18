#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
                   const size_t numNodes, const size_t firstRow,
                   const size_t localRows, const std::vector<int>& rowOwner,
                   MPI_Comm comm) {
    std::vector<unsigned int> pivotRow(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[k];
        if (k >= firstRow && k < firstRow + localRows) {
            std::copy_n(dist.data() + (k - firstRow) * numNodes,
                        numNodes, pivotRow.data());
        }
        MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, comm);

        for (size_t i = 0; i < localRows; ++i) {
            unsigned int* const distRow = dist.data() + i * numNodes;
            unsigned int* const pathRow = path.data() + i * numNodes;
            const unsigned int distIK = distRow[k];
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivotRow[j];
                if (newDist < distRow[j]) {
                    distRow[j] = newDist;
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
    int rank = 0;
    int worldSize = 1;
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
    
    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (numNodes != 0 && numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) / numNodes)) {
        if (rank == 0) fprintf(stderr, "Matrix is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> rowCounts(worldSize), rowOffsets(worldSize);
    std::vector<int> elemCounts(worldSize), elemOffsets(worldSize);
    std::vector<int> rowOwner(numNodes);
    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t remainder = numNodes % static_cast<size_t>(worldSize);
    size_t offset = 0;
    for (int p = 0; p < worldSize; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < remainder ? 1 : 0);
        rowCounts[p] = static_cast<int>(rows);
        rowOffsets[p] = static_cast<int>(offset);
        elemCounts[p] = static_cast<int>(rows * numNodes);
        elemOffsets[p] = static_cast<int>(offset * numNodes);
        for (size_t row = offset; row < offset + rows; ++row) rowOwner[row] = p;
        offset += rows;
    }
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    const size_t firstRow = static_cast<size_t>(rowOffsets[rank]);
    
    // Allocate matrices
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
    }
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, elemCounts.data(),
                 elemOffsets.data(), MPI_UNSIGNED, dist.data(), elemCounts[rank],
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    for (size_t i = 0; i < localRows; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes,
                    static_cast<unsigned int>(firstRow + i));
    }
    globalDist.clear();
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, firstRow, localRows, rowOwner, MPI_COMM_WORLD);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computation time: %.3f ms\n", duration * 1000.0);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    if (rank == 0) {
        const double gflops = duration > 0.0 ? ops / duration / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults || validate) {
        if (rank == 0) globalDist.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), elemCounts[rank], MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr, elemCounts.data(),
                    elemOffsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(globalDist, "DistanceMatrix");
    }
    
    // Validation
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(globalDist, numNodes);
        
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
