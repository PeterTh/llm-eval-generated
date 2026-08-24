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
                   int size) {
    
    // 1D row block distribution
    size_t rowsPerProc = numNodes / size;
    size_t remainder = numNodes % size;
    
    // Calculate local start and end rows
    // size_t localStartRow = rank * rowsPerProc + std::min((size_t)rank, remainder);
    size_t localNumRows = rowsPerProc + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Buffer for the k-th row, broadcasted from owner
    std::vector<unsigned int> kRowDist(numNodes);
    std::vector<unsigned int> kRowPath(numNodes); // Not needed for calculation but maybe for consistency? No, only dist is needed.

    // Distribute the data
    // We need to scatter the initial full matrix if we are doing it inside here, 
    // or assume the caller handles distribution.
    // However, given the signature, 'dist' and 'path' are likely the full matrices on rank 0
    // and potentially empty on others.
    // But since the signature takes full matrices, let's assume we work with local parts.
    // Actually, to keep it simple and match "change existing files only", 
    // let's distribute inside this function if strictly necessary, 
    // or better, allocate local buffers.

    // For simplicity, let's allocate local buffers
    std::vector<unsigned int> localDist(localNumRows * numNodes);
    std::vector<unsigned int> localPath(localNumRows * numNodes);

    // Scatter data from rank 0
    // Since rowsPerProc might vary, we use MPI_Scatterv
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        size_t offset = 0;
        for (int i = 0; i < size; ++i) {
            size_t count = rowsPerProc + (static_cast<size_t>(i) < remainder ? 1 : 0);
            sendcounts[i] = count * numNodes;
            displs[i] = offset * numNodes;
            offset += count;
        }
    }

    MPI_Scatterv(dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), localNumRows * numNodes, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
                 
    MPI_Scatterv(path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), localNumRows * numNodes, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Classic Floyd-Warshall algorithm distributed
    for (size_t k = 0; k < numNodes; ++k) {
        
        // Find owner of row k
        int owner = -1;
        // Reconstruct owner logic (inverse of localStartRow logic)
        // Or simpler: iterate
        // A faster way:
        if (k < remainder * (rowsPerProc + 1)) {
            owner = k / (rowsPerProc + 1);
        } else {
            owner = remainder + (k - remainder * (rowsPerProc + 1)) / rowsPerProc;
        }

        // Prepare k-th row
        if (rank == owner) {
            size_t localK = k - (owner * rowsPerProc + std::min((size_t)owner, remainder));
            // Copy to buffer
            // localDist is flattened, access row localK
            // idx2(col, row, n) -> row * n + col.
            // Here localDist is (localNumRows x numNodes).
            // Row localK in localDist corresponds to row k in global matrix.
            // Access: localDist[localK * numNodes + j]
            for (size_t j = 0; j < numNodes; ++j) {
                kRowDist[j] = localDist[localK * numNodes + j];
            }
        }

        // Broadcast k-th row
        MPI_Bcast(kRowDist.data(), numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update local rows
        // For each local row i (which corresponds to global row globalI)
        for (size_t i = 0; i < localNumRows; ++i) {
            
            // Optimization: if dist[i][k] is INF, we can skip
            // globalI is source. k is intermediate.
            // dist[i][k] is localDist[i * numNodes + k]
            unsigned int distIK = localDist[i * numNodes + k];

            if (distIK == INF) continue;

            for (size_t j = 0; j < numNodes; ++j) {
                // dist[i][j]
                unsigned int distIJ = localDist[i * numNodes + j];
                // dist[k][j] comes from broadcast
                unsigned int distKJ = kRowDist[j];

                if (distKJ != INF && distIK + distKJ < distIJ) {
                    localDist[i * numNodes + j] = distIK + distKJ;
                    localPath[i * numNodes + j] = k;
                }
            }
        }
    }

    // Gather results back to rank 0
    MPI_Gatherv(localDist.data(), localNumRows * numNodes, MPI_UNSIGNED,
                dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
                
    MPI_Gatherv(localPath.data(), localNumRows * numNodes, MPI_UNSIGNED,
                path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
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

    int rank, size;
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Size: %d\n", size);
    }
    
    // Allocate matrices
    // Only rank 0 needs the full matrix initially and finally
    // However, to keep existing signature and structure, we can allocate on all or just handle in floydWarshall
    // Since initialize functions fill the whole matrix, we should probably just let rank 0 do initialization.
    
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;

    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
    
        // Initialize
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
        
        printf("Computing shortest paths...\n");
    }
    
    // Barrier to sync before start
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration = 0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration);
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration / 1000.0) / 1e9;
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
                // Don't return 1 here to avoid MPI abort issues, just print
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
