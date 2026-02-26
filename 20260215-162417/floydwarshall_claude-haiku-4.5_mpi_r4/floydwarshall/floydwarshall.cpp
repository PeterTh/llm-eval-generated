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

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes,
                   int rank,
                   int numProcs,
                   size_t localRows,
                   size_t startRow) {
    // MPI-parallelized Floyd-Warshall algorithm
    // dist[idx2(j, i, numNodes)] = distance from node i to node j (matrix[i][j])
    // Each process owns rows startRow to startRow+localRows-1
    // For each intermediate node k
    std::vector<unsigned int> kthRow(numNodes);
    
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which process owns row k
        int kOwner = (k * numProcs) / numNodes;
        
        // Gather the k-th row: dist[j][k] for all j (i.e., matrix[k][*])
        // Row k stored locally at position (k - startRow)
        if (rank == kOwner) {
            size_t localK = k - startRow;
            for (size_t j = 0; j < numNodes; ++j) {
                kthRow[j] = dist[idx2(j, localK, numNodes)];
            }
        }
        
        // Broadcast k-th row from owner to all processes
        MPI_Bcast(kthRow.data(), numNodes, MPI_UNSIGNED, kOwner, MPI_COMM_WORLD);
        
        // Each process updates its local rows using the k-th row
        for (size_t i = startRow; i < startRow + localRows; ++i) {
            size_t localI = i - startRow;
            // matrix[i][k] is in local storage since this process owns row i
            const unsigned int distIK = dist[idx2(k, localI, numNodes)];
            
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, localI, numNodes)];  // matrix[i][j]
                const unsigned int distKJ = kthRow[j];  // matrix[k][j]
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, localI, numNodes)] = newDist;
                    path[idx2(j, localI, numNodes)] = k;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
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
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate work distribution
    size_t localRows = numNodes / numProcs;
    size_t remainder = numNodes % numProcs;
    if (rank < (int)remainder) {
        localRows++;
    }
    size_t startRow = (rank < (int)remainder) ? 
                      rank * localRows : 
                      remainder * (remainder + 1) / 2 + (rank - remainder) * localRows;
    
    // Allocate local matrices
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    
    // Initialize (compute full matrix deterministically, then keep local rows)
    if (rank == 0) printf("Initializing graph...\n");
    
    // Compute the full distance matrix deterministically on every process
    std::vector<unsigned int> fullDist(numNodes * numNodes);
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE - 1) + 1.0;
    
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            fullDist[idx2(j, i, numNodes)] = 1 + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
        fullDist[idx2(i, i, numNodes)] = 0;
    }
    
    // Each process extracts its local rows from the full matrix
    for (size_t localI = 0; localI < localRows; ++localI) {
        size_t i = startRow + localI;
        for (size_t j = 0; j < numNodes; ++j) {
            dist[idx2(j, localI, numNodes)] = fullDist[idx2(j, i, numNodes)];
        }
    }
    
    // Initialize path matrix (same for all processes)
    for (size_t localI = 0; localI < localRows; ++localI) {
        size_t i = startRow + localI;
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx2(j, localI, numNodes)] = j;
        }
        path[idx2(i, localI, numNodes)] = i;
    }

    
    // Synchronize initialization
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, numProcs, localRows, startRow);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Gather results on rank 0 for validation and printing
    std::vector<unsigned int> globalDist;
    std::vector<int> sendcounts(numProcs);
    std::vector<int> displs(numProcs, 0);
    
    // Calculate sendcounts and displacements
    for (int i = 0; i < numProcs; ++i) {
        size_t iLocalRows = numNodes / numProcs;
        if (i < (int)(numNodes % numProcs)) {
            iLocalRows++;
        }
        sendcounts[i] = iLocalRows * numNodes;
        if (i > 0) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }
    
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
    }
    
    MPI_Gatherv(dist.data(), localRows * numNodes, MPI_UNSIGNED,
                globalDist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation (on rank 0)
    if (rank == 0 && printResults) {
        print_results_int(globalDist, "DistanceMatrix");
    }
    
    // Validation (on rank 0)
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(globalDist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    // Finalize MPI
    MPI_Finalize();
    
    if (rank == 0 && validate) {
        return validateResult(globalDist, numNodes) ? 0 : 1;
    }
    
    return 0;
}
