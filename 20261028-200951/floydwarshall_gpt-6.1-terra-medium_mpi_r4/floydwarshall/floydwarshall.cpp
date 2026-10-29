#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
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

void floydWarshall(std::vector<unsigned int>& dist,
                   const size_t numNodes, const size_t firstRow,
                   const size_t localRows, int rank, int ranks) {
    std::vector<unsigned int> pivot(numNodes);
    const size_t base = numNodes / ranks;
    const size_t extra = numNodes % ranks;
    for (size_t k = 0; k < numNodes; ++k) {
        const size_t wideRows = (base + 1) * extra;
        const int owner = static_cast<int>(k < wideRows ? k / (base + 1)
                                                        : extra + (k - wideRows) / base);
        unsigned int* pivotRow = pivot.data();
        if (rank == owner) {
            pivotRow = dist.data() + (k - firstRow) * numNodes;
        }
        MPI_Bcast(pivotRow, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t i = 0; i < localRows; ++i) {
            unsigned int* row = dist.data() + i * numNodes;
            const unsigned int dik = row[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                if (candidate < row[j]) row[j] = candidate;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
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
    // MPI_Scatterv and MPI_Gatherv use int counts and displacements.
    if (numNodes > static_cast<size_t>(INT_MAX) ||
        (numNodes != 0 && numNodes > static_cast<size_t>(INT_MAX) / numNodes)) {
        if (rank == 0) fprintf(stderr, "Matrix is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }
    const size_t base = numNodes / ranks, extra = numNodes % ranks;
    const size_t localRows = base + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), extra);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = static_cast<int>((base + (static_cast<size_t>(r) < extra)) * numNodes);
        offsets[r] = static_cast<int>((static_cast<size_t>(r) * base +
                                       std::min(static_cast<size_t>(r), extra)) * numNodes);
    }
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> full;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        full.resize(numNodes * numNodes);
        initializeDistanceMatrix(full, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(full.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::vector<unsigned int>().swap(full);
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, numNodes, firstRow, localRows, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    if (validate || printResults) {
        if (rank == 0) full.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, full.data(),
                    counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    int status = 0;
    if (rank == 0) {
        if (printResults) print_results_int(full, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(full, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
