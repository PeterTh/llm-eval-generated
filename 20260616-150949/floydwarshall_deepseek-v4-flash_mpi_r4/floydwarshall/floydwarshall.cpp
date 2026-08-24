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

// Index calculation for flattened 2D array (row-major)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Initialize full distance matrix on a single rank
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality (sample limited rows for performance)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
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
    
    int numProcs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse for MPI_Init compat)
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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // ---- 1D row-block distribution setup ----
    size_t localRows = numNodes / static_cast<size_t>(numProcs);
    size_t remainder = numNodes % static_cast<size_t>(numProcs);
    
    // sendCounts[p] = number of matrix elements for process p
    // displs[p]     = offset (in elements) into the full matrix for process p
    // procStartRow[p] = global row index where process p's data starts
    std::vector<size_t> sendCounts(static_cast<size_t>(numProcs));
    std::vector<size_t> displs(static_cast<size_t>(numProcs));
    std::vector<size_t> procStartRow(static_cast<size_t>(numProcs));
    
    {
        size_t offset = 0;
        for (int p = 0; p < numProcs; ++p) {
            size_t rows = localRows + (static_cast<size_t>(p) < remainder ? 1 : 0);
            sendCounts[p] = rows * numNodes;
            displs[p] = offset;
            procStartRow[p] = offset / numNodes;
            offset += rows * numNodes;
        }
    }
    
    size_t myRows = sendCounts[static_cast<size_t>(rank)] / numNodes;
    size_t myStart = procStartRow[static_cast<size_t>(rank)];
    
    // Precompute which rank owns each global row
    std::vector<int> rowOwner(static_cast<size_t>(numNodes));
    for (size_t k = 0; k < numNodes; ++k) {
        for (int p = 0; p < numProcs; ++p) {
            if (k >= procStartRow[p] && 
                k < procStartRow[p] + sendCounts[p] / numNodes) {
                rowOwner[k] = p;
                break;
            }
        }
    }
    
    // Local matrix storage (at least 1 element to avoid null pointers with MPI)
    const size_t localAlloc = myRows * numNodes > 0 ? myRows * numNodes : 1;
    std::vector<unsigned int> localDist(localAlloc);
    std::vector<unsigned int> localPath(localAlloc);
    
    // ---- Initialization ----
    // Rank 0 initializes the full distance matrix, then scatters rows
    if (rank == 0) {
        printf("Initializing graph...\n");
        std::vector<unsigned int> fullDist(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        
        // Convert sizes to int for MPI (safe for benchmark-sized problems)
        std::vector<int> sc(numProcs), disp(numProcs);
        for (int p = 0; p < numProcs; ++p) {
            sc[p] = static_cast<int>(sendCounts[p]);
            disp[p] = static_cast<int>(displs[p]);
        }
        
        MPI_Scatterv(fullDist.data(), sc.data(), disp.data(), MPI_UNSIGNED,
                     localDist.data(), static_cast<int>(myRows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    } else {
        std::vector<int> sc(numProcs), disp(numProcs);
        for (int p = 0; p < numProcs; ++p) {
            sc[p] = static_cast<int>(sendCounts[p]);
            disp[p] = static_cast<int>(displs[p]);
        }
        
        MPI_Scatterv(nullptr, sc.data(), disp.data(), MPI_UNSIGNED,
                     localDist.data(), static_cast<int>(myRows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }
    
    // Initialize path matrix locally (each row i has direct predecessor i)
    for (size_t row = 0; row < myRows; ++row) {
        unsigned int* pathRow = localPath.data() + row * numNodes;
        unsigned int globalRow = static_cast<unsigned int>(myStart + row);
        for (size_t j = 0; j < numNodes; ++j) {
            pathRow[j] = globalRow;
        }
    }
    
    // Broadcast buffer for row k of the distance matrix
    std::vector<unsigned int> rowK(numNodes);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // ---- Parallel Floyd-Warshall (1D row-block distribution) ----
    // For each k, the owner of row k broadcasts it to all processes.
    // Each process then updates its local rows using the broadcast row k
    // and its locally available column k values.
    //
    // During iteration k, row k and column k remain unchanged (since
    // dist[k][k] = 0, no updates alter them), so broadcasting row k
    // once per iteration is correct.
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = rowOwner[k];
        
        // Owner extracts row k into the broadcast buffer
        if (rank == owner) {
            size_t localK = k - myStart;
            std::copy_n(localDist.data() + localK * numNodes, numNodes, rowK.data());
        }
        
        MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, 
                  owner, MPI_COMM_WORLD);
        
        // Update all local rows
        for (size_t row = 0; row < myRows; ++row) {
            unsigned int* distRow = localDist.data() + row * numNodes;
            unsigned int* pathRow = localPath.data() + row * numNodes;
            unsigned int distIK = distRow[k];  // dist[i][k] (column k, local row i)
            
            for (size_t j = 0; j < numNodes; ++j) {
                unsigned int newDist = distIK + rowK[j];  // dist[i][k] + dist[k][j]
                if (newDist < distRow[j]) {
                    distRow[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = static_cast<long>(duration.count());
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);
        
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (static_cast<double>(maxDuration) / 1000.0) / 1.0e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // ---- Gather results to rank 0 ----
    std::vector<unsigned int> fullDist, fullPath;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
    }
    
    std::vector<int> sc(numProcs), disp(numProcs);
    for (int p = 0; p < numProcs; ++p) {
        sc[p] = static_cast<int>(sendCounts[p]);
        disp[p] = static_cast<int>(displs[p]);
    }
    
    MPI_Gatherv(localDist.data(), static_cast<int>(myRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? fullDist.data() : nullptr, sc.data(), disp.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    
    MPI_Gatherv(localPath.data(), static_cast<int>(myRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? fullPath.data() : nullptr, sc.data(), disp.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    
    // ---- Post-processing (rank 0 only) ----
    if (rank == 0) {
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
