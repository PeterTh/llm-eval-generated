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

// Transfer arbitrarily large contiguous arrays without exceeding MPI's int count.
void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count != 0) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

size_t rowCount(size_t n, int rank, int ranks) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}
size_t rowStart(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, size_t first, size_t rows, int rank, int ranks) {
    std::vector<unsigned int> pivot(n);
    int owner = 0;
    size_t ownerEnd = rowCount(n, 0, ranks);
    for (size_t k = 0; k < n; ++k) {
        while (k >= ownerEnd) ownerEnd += rowCount(n, ++owner, ranks);
        unsigned int* pivotData = rank == owner ? dist.data() + (k - first) * n : pivot.data();
        // A row may exceed the count accepted by MPI_Bcast.
        for (size_t offset = 0; offset < n;) {
            int chunk = static_cast<int>(std::min(n - offset, static_cast<size_t>(INT_MAX)));
            MPI_Bcast(pivotData + offset, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            offset += chunk;
        }
        for (size_t i = 0; i < rows; ++i) {
            unsigned int* row = dist.data() + i * n;
            unsigned int* pathRow = path.data() + i * n;
            const unsigned int dik = row[k];
            for (size_t j = 0; j < n; ++j) {
                unsigned int candidate = dik + pivotData[j];
                if (candidate < row[j]) {
                    row[j] = candidate;
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
    const size_t first = rowStart(numNodes, rank, ranks);
    const size_t rows = rowCount(numNodes, rank, ranks);
    std::vector<unsigned int> dist(rows * numNodes);
    std::vector<unsigned int> path(rows * numNodes);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        unsigned int seed = 42;
        for (int r = 0; r < ranks; ++r) {
            size_t rfirst = rowStart(numNodes, r, ranks);
            size_t rrows = rowCount(numNodes, r, ranks);
            std::vector<unsigned int> buffer;
            if (r != 0) buffer.resize(rrows * numNodes);
            unsigned int* target = r == 0 ? dist.data() : buffer.data();
            for (size_t i = 0; i < rrows; ++i) {
                for (size_t j = 0; j < numNodes; ++j) {
                    unsigned int value = 1 + static_cast<unsigned int>(200.0 * rand_r(&seed) / (double)RAND_MAX);
                    target[i * numNodes + j] = (rfirst + i == j) ? 0 : value;
                }
            }
            if (r != 0) transfer(target, rrows * numNodes, r, 0, true);
        }
    } else {
        transfer(dist.data(), dist.size(), 0, 0, false);
    }
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < numNodes; ++j)
            path[i * numNodes + j] = static_cast<unsigned int>(j);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes, first, rows, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    std::vector<unsigned int> full;
    if (printResults || validate) {
        if (rank == 0) {
            full.resize(numNodes * numNodes);
            std::copy(dist.begin(), dist.end(), full.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t count = rowCount(numNodes, r, ranks) * numNodes;
                if (count) transfer(full.data() + rowStart(numNodes, r, ranks) * numNodes, count, r, 1, false);
            }
        } else {
            transfer(dist.data(), dist.size(), 0, 1, true);
        }
    }
    int status = 0;
    if (rank == 0) {
        if (printResults) print_results_int(full, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
