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

// Number of rows owned by a rank for a contiguous row-block distribution.
inline size_t blockCount(const int rank, const size_t numNodes, const int numProcs) noexcept {
    const size_t p = static_cast<size_t>(numProcs);
    const size_t base = numNodes / p;
    const size_t rem = numNodes % p;
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

// First row owned by a rank for a contiguous row-block distribution.
inline size_t blockStart(const int rank, const size_t numNodes, const int numProcs) noexcept {
    const size_t p = static_cast<size_t>(numProcs);
    const size_t base = numNodes / p;
    const size_t rem = numNodes % p;
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
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

// The original path matrix initialization reduces to path[row][col] = row.
// Only the locally owned rows are materialized on each rank.
void initializeLocalPathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                               const size_t rowBegin, const size_t numLocalRows) {
    for (size_t r = 0; r < numLocalRows; ++r) {
        const unsigned int rowValue = static_cast<unsigned int>(rowBegin + r);
        unsigned int* row = path.data() + r * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = rowValue;
        }
    }
}

// Distributed Floyd-Warshall: every rank owns a contiguous block of source
// rows of both matrices. For each intermediate node k the owner of row k
// broadcasts that row and all ranks relax their local rows against it.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t rowBegin,
                   const size_t numLocalRows,
                   const int rank,
                   const int numProcs,
                   std::vector<unsigned int>& pivotRow) {
    if (numNodes == 0) {
        return;
    }

    const size_t p = static_cast<size_t>(numProcs);
    const size_t base = numNodes / p;
    const size_t rem = numNodes % p;
    const size_t firstLongBlockEnd = rem * (base + 1);

    unsigned int* distData = dist.data();
    unsigned int* pathData = path.data();
    unsigned int* pivot = pivotRow.data();
    const int bcastCount = static_cast<int>(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns row k.
        int owner;
        if (base == 0) {
            owner = static_cast<int>(k);
        } else if (k < firstLongBlockEnd) {
            owner = static_cast<int>(k / (base + 1));
        } else {
            owner = static_cast<int>(rem + (k - firstLongBlockEnd) / base);
        }

        if (rank == owner) {
            const unsigned int* pivotSrc = distData + (k - rowBegin) * numNodes;
            std::memcpy(pivot, pivotSrc, numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(pivot, bcastCount, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        const unsigned int kValue = static_cast<unsigned int>(k);
        for (size_t r = 0; r < numLocalRows; ++r) {
            if (rowBegin + r == k) {
                // Relaxing row k through k never changes it (dist[k][k] == 0).
                continue;
            }

            unsigned int* __restrict__ dRow = distData + r * numNodes;
            unsigned int* __restrict__ pRow = pathData + r * numNodes;
            const unsigned int distIK = dRow[k];
            const unsigned int* __restrict__ kRow = pivot;

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + kRow[j];
                if (newDist < dRow[j]) {
                    dRow[j] = newDist;
                    pRow[j] = kValue;
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
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }

    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
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
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    const size_t localRows = blockCount(rank, numNodes, numProcs);
    const size_t rowBegin = blockStart(rank, numNodes, numProcs);
    
    // Allocate local row blocks of the matrices
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    
    // A contiguous MPI datatype describing one full matrix row
    MPI_Datatype rowType = MPI_DATATYPE_NULL;
    if (numNodes > 0) {
        MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
        MPI_Type_commit(&rowType);
    }
    
    std::vector<int> rowCounts(numProcs);
    std::vector<int> rowDispls(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        rowCounts[r] = static_cast<int>(blockCount(r, numNodes, numProcs));
        rowDispls[r] = static_cast<int>(blockStart(r, numNodes, numProcs));
    }
    
    // Initialize the distance matrix on rank 0 and scatter the row blocks
    {
        std::vector<unsigned int> fullDist;
        unsigned int* sendBuf = nullptr;
        
        if (rank == 0) {
            if (rank == 0) {
                printf("Initializing graph...\n");
            }
            fullDist.resize(numNodes * numNodes);
            initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
            sendBuf = fullDist.data();
        }
        
        if (numNodes > 0) {
            MPI_Scatterv(sendBuf, rowCounts.data(), rowDispls.data(), rowType,
                         dist.data(), static_cast<int>(localRows), rowType,
                         0, MPI_COMM_WORLD);
        }
    }
    
    initializeLocalPathMatrix(path, numNodes, rowBegin, localRows);
    
    std::vector<unsigned int> pivotRow(numNodes);
    
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rowBegin, localRows, rank, numProcs, pivotRow);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n^3) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    bool valid = true;
    if (printResults || validate) {
        // Gather the distributed row blocks back to rank 0 for output/validation
        std::vector<unsigned int> fullDist;
        unsigned int* recvBuf = nullptr;
        
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            recvBuf = fullDist.data();
        }
        
        if (numNodes > 0) {
            MPI_Gatherv(dist.data(), static_cast<int>(localRows), rowType,
                        recvBuf, rowCounts.data(), rowDispls.data(), rowType,
                        0, MPI_COMM_WORLD);
        }
        
        if (rank == 0) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(fullDist, "DistanceMatrix");
            }
            
            // Validation
            if (validate) {
                printf("Validating result...\n");
                valid = validateResult(fullDist, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }
    
    // Keep the exit status consistent across all ranks
    int validFlag = valid ? 1 : 0;
    MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rowType != MPI_DATATYPE_NULL) {
        MPI_Type_free(&rowType);
    }
    
    MPI_Finalize();
    return validFlag ? 0 : 1;
}
