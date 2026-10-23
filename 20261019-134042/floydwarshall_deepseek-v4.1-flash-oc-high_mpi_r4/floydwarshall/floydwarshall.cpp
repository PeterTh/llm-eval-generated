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

void partitionColumns(const size_t numNodes, const int rank, const int size,
                      size_t& colStart, size_t& colCount) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    colStart = r * base + std::min(r, rem);
    colCount = base + (r < rem ? 1 : 0);
}

int ownerOfColumn(const size_t k, const size_t numNodes, const int size) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t big = rem * (base + 1);
    if (k < big) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(rem + (k - big) / base);
}

void initializeLocalMatrices(std::vector<unsigned int>& dist,
                             std::vector<unsigned int>& path,
                             const size_t numNodes,
                             const size_t colStart, const size_t colCount,
                             const unsigned int rangeMin,
                             const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t offset = colStart * numNodes;
    for (size_t p = 0; p < offset; ++p) {
        rand_r(&seed);
    }

    const size_t localElements = colCount * numNodes;
    for (size_t p = 0; p < localElements; ++p) {
        dist[p] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t c = 0; c < colCount; ++c) {
        const size_t j = colStart + c;
        dist[c * numNodes + j] = 0;
    }

    for (size_t c = 0; c < colCount; ++c) {
        const size_t j = colStart + c;
        unsigned int* colP = path.data() + c * numNodes;
        for (size_t i = 0; i < numNodes; ++i) {
            colP[i] = static_cast<unsigned int>(j);
        }
    }
}

void floydWarshallMPI(std::vector<unsigned int>& dist,
                      std::vector<unsigned int>& path,
                      const size_t numNodes,
                      const size_t colStart, const size_t colCount,
                      const int rank, const std::vector<int>& owner) {
    std::vector<unsigned int> recvColumn(numNodes);

    unsigned int* d = dist.data();
    unsigned int* p = path.data();

    for (size_t k = 0; k < numNodes; ++k) {
        const int root = owner[k];

        const unsigned int* colK;
        if (rank == root) {
            unsigned int* sendColumn = d + (k - colStart) * numNodes;
            MPI_Bcast(sendColumn, static_cast<int>(numNodes), MPI_UNSIGNED,
                      root, MPI_COMM_WORLD);
            colK = sendColumn;
        } else {
            MPI_Bcast(recvColumn.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                      root, MPI_COMM_WORLD);
            colK = recvColumn.data();
        }

        for (size_t c = 0; c < colCount; ++c) {
            const size_t j = colStart + c;
            if (j == k) {
                continue;
            }

            const unsigned int distKJ = d[c * numNodes + k];
            unsigned int* __restrict colJ = d + c * numNodes;
            unsigned int* __restrict pathJ = p + c * numNodes;
            const unsigned int* __restrict colKPtr = colK;

            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDist = colKPtr[i] + distKJ;
                if (newDist < colJ[i]) {
                    colJ[i] = newDist;
                    pathJ[i] = static_cast<unsigned int>(k);
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
    int size = 1;
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    size_t colStart = 0;
    size_t colCount = 0;
    partitionColumns(numNodes, rank, size, colStart, colCount);

    std::vector<int> owner(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        owner[k] = ownerOfColumn(k, numNodes, size);
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    std::vector<unsigned int> dist(colCount * numNodes);
    std::vector<unsigned int> path(colCount * numNodes);
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeLocalMatrices(dist, path, numNodes, colStart, colCount, 1, MAX_DISTANCE);
    
    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshallMPI(dist, path, numNodes, colStart, colCount, rank, owner);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            size_t rStart = 0;
            size_t rCount = 0;
            partitionColumns(numNodes, r, size, rStart, rCount);
            counts[r] = static_cast<int>(rCount * numNodes);
            displs[r] = static_cast<int>(rStart * numNodes);
        }

        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
        }

        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr,
                    counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }
    }
    
    // Validation
    int result = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return result;
}
