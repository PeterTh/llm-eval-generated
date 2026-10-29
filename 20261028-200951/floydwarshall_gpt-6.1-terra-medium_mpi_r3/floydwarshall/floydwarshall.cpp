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

// Transfer large buffers in chunks because MPI counts are int-sized.
void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, size_t n, size_t first,
                   size_t rows, int ranks) {
    std::vector<unsigned int> pivot(n);
    const size_t base = n / ranks, extra = n % ranks;
    for (size_t k = 0; k < n; ++k) {
        const size_t split = (base + 1) * extra;
        const int owner = static_cast<int>(k < split ? k / (base + 1)
                                                    : extra + (k - split) / base);
        if (k >= first && k - first < rows)
            std::copy_n(dist.data() + (k - first) * n, n, pivot.data());
        for (size_t offset = 0; offset < n;) {
            const int chunk = static_cast<int>(std::min(n - offset, static_cast<size_t>(INT_MAX)));
            MPI_Bcast(pivot.data() + offset, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            offset += chunk;
        }
        for (size_t i = 0; i < rows; ++i) {
            unsigned int* row = dist.data() + i * n;
            const unsigned int dik = row[k];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = dik + pivot[j];
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
    const size_t base = numNodes / ranks, extra = numNodes % ranks;
    const size_t rows = base + (static_cast<size_t>(rank) < extra);
    const size_t first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    std::vector<unsigned int> local(rows * numNodes);
    std::vector<unsigned int> full;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        full.resize(numNodes * numNodes);
        initializeDistanceMatrix(full, numNodes, 1, MAX_DISTANCE);
        std::copy_n(full.data(), local.size(), local.data());
        for (int r = 1; r < ranks; ++r) {
            const size_t start = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra);
            const size_t count = (base + (static_cast<size_t>(r) < extra)) * numNodes;
            if (count) transfer(full.data() + start * numNodes, count, r, 0, true);
        }
        if (!validate && !printResults) std::vector<unsigned int>().swap(full);
        printf("Computing shortest paths...\n");
    } else {
        transfer(local.data(), local.size(), 0, 0, false);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(local, numNodes, first, rows, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (validate || printResults) {
        if (rank == 0) {
            std::copy_n(local.data(), local.size(), full.data());
            for (int r = 1; r < ranks; ++r) {
                const size_t begin = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra);
                const size_t count = (base + (static_cast<size_t>(r) < extra)) * numNodes;
                if (count) transfer(full.data() + begin * numNodes, count, r, 1, false);
            }
        } else {
            transfer(local.data(), local.size(), 0, 1, true);
        }
    }
    int status = 0;
    if (rank == 0) {
        const auto milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / maxElapsed / 1e9);
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
