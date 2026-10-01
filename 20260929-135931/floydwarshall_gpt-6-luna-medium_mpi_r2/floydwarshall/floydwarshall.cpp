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
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // For each source node i
        for (size_t i = 0; i < numNodes; ++i) {
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
}

// Each rank owns complete source rows. The pivot row is broadcast at each
// iteration, so the cubic work is distributed while communication is O(n^2).
void distributedFloydWarshall(std::vector<unsigned int>& localDist,
                              std::vector<unsigned int>& localPath,
                              size_t n, size_t firstRow, int size) {
    std::vector<unsigned int> pivot(n);
    for (size_t k = 0; k < n; ++k) {
        if (k >= firstRow && k < firstRow + localDist.size() / (n ? n : 1)) {
            const size_t localK = k - firstRow;
            std::copy_n(localDist.data() + localK * n, n, pivot.data());
        }
        int owner = 0;
        // Balanced contiguous partition: first remainder ranks own one extra row.
        const size_t base = n / size, rem = n % size;
        owner = k < (base + 1) * rem ? static_cast<int>(k / (base + 1))
                                    : static_cast<int>(rem + (k - (base + 1) * rem) / base);
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t li = 0, rows = localDist.size() / (n ? n : 1); li < rows; ++li) {
            const unsigned int dik = localDist[li * n + k];
            unsigned int* row = localDist.data() + li * n;
            unsigned int* path = localPath.data() + li * n;
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = dik + pivot[j];
                if (candidate < row[j]) { row[j] = candidate; path[j] = static_cast<unsigned int>(k); }
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
    int rank, worldSize;
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
            printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes), path(numNodes * numNodes);
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    const size_t base = numNodes / worldSize, rem = numNodes % worldSize;
    const size_t first = rank * base + std::min<size_t>(rank, rem);
    const size_t rows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    std::vector<int> counts(worldSize), displs(worldSize);
    for (int p = 0; p < worldSize; ++p) {
        const size_t pRows = base + (static_cast<size_t>(p) < rem ? 1 : 0);
        const size_t pFirst = p * base + std::min<size_t>(p, rem);
        counts[p] = static_cast<int>(pRows * numNodes);
        displs[p] = static_cast<int>(pFirst * numNodes);
    }
    // Convert to source-major layout for contiguous row distribution.
    std::vector<unsigned int> packedDist, packedPath;
    if (rank == 0) {
        packedDist.resize(numNodes * numNodes); packedPath.resize(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) for (size_t j = 0; j < numNodes; ++j) {
            packedDist[i * numNodes + j] = dist[idx2(j, i, numNodes)];
            packedPath[i * numNodes + j] = path[idx2(j, i, numNodes)];
        }
    }
    std::vector<unsigned int> localDist(rows * numNodes), localPath(rows * numNodes);
    MPI_Scatterv(rank == 0 ? packedDist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? packedPath.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localPath.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    distributedFloydWarshall(localDist, localPath, numNodes, first, worldSize);

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                rank == 0 ? packedDist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) for (size_t i = 0; i < numNodes; ++i) for (size_t j = 0; j < numNodes; ++j)
        dist[idx2(j, i, numNodes)] = packedDist[i * numNodes + j];
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (rank == 0 && printResults) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
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
