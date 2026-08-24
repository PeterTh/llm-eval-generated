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

void floydWarshallMPI(std::vector<unsigned int>& local_dist, 
                      std::vector<unsigned int>& local_path, 
                      const size_t numNodes,
                      int rank,
                      int numProcs,
                      int rowsPerProc,
                      int remainder) {
    
    std::vector<unsigned int> kRowDist(numNodes);
    
    // Start index of rows for this process
    // Distribution: first 'remainder' processes get rowsPerProc + 1, others get rowsPerProc
    size_t myStartRow;
    if (rank < remainder) {
        myStartRow = rank * (rowsPerProc + 1);
    } else {
        myStartRow = remainder * (rowsPerProc + 1) + (rank - remainder) * rowsPerProc;
    }
    size_t myNumRows = (rank < remainder) ? rowsPerProc + 1 : rowsPerProc;

    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        
        // Find owner of row k
        int ownerRank;
        if (k < (size_t)remainder * (rowsPerProc + 1)) {
            ownerRank = k / (rowsPerProc + 1);
        } else {
            ownerRank = remainder + (k - (remainder * (rowsPerProc + 1))) / rowsPerProc;
        }

        if (rank == ownerRank) {
            // Copy row k to buffer
            // k is global row index. Local index is k - myStartRow
            size_t localK = k - myStartRow;
            for (size_t j = 0; j < numNodes; ++j) {
                // Accessing dist[localK][j]
                // The local_dist is flattened, size myNumRows * numNodes.
                // Row 'localK' starts at localK * numNodes
                kRowDist[j] = local_dist[localK * numNodes + j];
            }
        }

        // Broadcast row k
        MPI_Bcast(kRowDist.data(), numNodes, MPI_UNSIGNED, ownerRank, MPI_COMM_WORLD);

        // Update local rows
        // For each source node i (local row)
        for (size_t i = 0; i < myNumRows; ++i) {
            // Global row index for i is myStartRow + i
            // But we don't need global index for access, just for logic if needed.
            
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                // distIJ is in local_dist at row i, col j
                unsigned int& distIJ = local_dist[i * numNodes + j];
                unsigned int& pathIJ = local_path[i * numNodes + j];
                
                // distIK is in local_dist at row i, col k.
                // Wait, dist[i][k] is local.
                const unsigned int distIK = local_dist[i * numNodes + k];
                
                // distKJ is in kRowDist at col j
                const unsigned int distKJ = kRowDist[j];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    distIJ = newDist;
                    pathIJ = k; // k is the global index of the intermediate node
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
        printf("Processes: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine row distribution
    int rowsPerProc = numNodes / numProcs;
    int remainder = numNodes % numProcs;

    // Calculate counts and displacements for scatter/gather
    std::vector<int> sendcounts(numProcs);
    std::vector<int> displs(numProcs);
    int offset = 0;
    for (int i = 0; i < numProcs; ++i) {
        int rows = (i < remainder) ? rowsPerProc + 1 : rowsPerProc;
        sendcounts[i] = rows * numNodes; // Number of elements (ints)
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int myRows = (rank < remainder) ? rowsPerProc + 1 : rowsPerProc;
    std::vector<unsigned int> local_dist(myRows * numNodes);
    std::vector<unsigned int> local_path(myRows * numNodes);
    
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;

    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        
        // Initialize
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Scatter data
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), myRows * numNodes, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
                 
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), myRows * numNodes, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshallMPI(local_dist, local_path, numNodes, rank, numProcs, rowsPerProc, remainder);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Gather results
    MPI_Gatherv(local_dist.data(), myRows * numNodes, MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(local_path.data(), myRows * numNodes, MPI_UNSIGNED,
                rank == 0 ? path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                // We should propagate failure but main returns int.
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
