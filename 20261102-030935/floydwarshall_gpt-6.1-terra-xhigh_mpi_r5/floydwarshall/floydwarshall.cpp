#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous, balanced source-row distribution.
size_t rowCount(size_t n, int ranks, int rank) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}

size_t firstRow(size_t n, int ranks, int rank) {
    return static_cast<size_t>(rank) * (n / ranks) +
           std::min(static_cast<size_t>(rank), n % ranks);
}

// Generate the original rand_r stream in order, including the diagonal values
// that are subsequently overwritten. Stream it to ranks without a full copy.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              int rank, int ranks) {
    constexpr size_t chunkSize = 1u << 20;
    if (rank == 0) {
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        auto generate = [&](unsigned int* data, size_t count, size_t offset) {
            for (size_t i = 0; i < count; ++i) {
                unsigned int value = 1 + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
                size_t index = offset + i;
                data[i] = (index / n == index % n) ? 0 : value;
            }
        };
        if (!dist.empty()) generate(dist.data(), dist.size(), 0);
        std::vector<unsigned int> buffer(std::min(chunkSize, n * rowCount(n, ranks, 0)));
        for (int target = 1; target < ranks; ++target) {
            size_t count = n * rowCount(n, ranks, target);
            size_t offset = n * firstRow(n, ranks, target);
            for (size_t done = 0; done < count;) {
                size_t chunk = std::min(buffer.size(), count - done);
                generate(buffer.data(), chunk, offset + done);
                MPI_Send(buffer.data(), static_cast<int>(chunk), MPI_UNSIGNED, target, 0, MPI_COMM_WORLD);
                done += chunk;
            }
        }
    } else {
        for (size_t done = 0; done < dist.size();) {
            size_t chunk = std::min(chunkSize, dist.size() - done);
            MPI_Recv(dist.data() + done, static_cast<int>(chunk), MPI_UNSIGNED, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            done += chunk;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   int rank, int ranks) {
    const size_t first = firstRow(n, ranks, rank);
    const size_t rows = rowCount(n, ranks, rank);
    std::vector<unsigned int> pivot(n);
    int owner = 0;
    size_t ownerEnd = rowCount(n, ranks, 0);
    for (size_t k = 0; k < n; ++k) {
        while (k >= ownerEnd) {
            ++owner;
            ownerEnd += rowCount(n, ranks, owner);
        }
        unsigned int* pivotRow = rank == owner ? dist.data() + (k - first) * n : pivot.data();
        for (size_t offset = 0; offset < n;) {
            int count = static_cast<int>(std::min(n - offset, static_cast<size_t>(INT_MAX)));
            MPI_Bcast(pivotRow + offset, count, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            offset += count;
        }
        for (size_t i = 0; i < rows; ++i) {
            unsigned int* row = dist.data() + i * n;
            unsigned int* pathRow = path.data() + i * n;
            const unsigned int distIK = row[k];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int newDist = distIK + pivotRow[j];
                if (newDist < row[j]) {
                    row[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

// Collect only when reporting or validating the complete result.
std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& local,
                                          size_t n, int rank, int ranks) {
    constexpr size_t chunkSize = 1u << 20;
    std::vector<unsigned int> full;
    if (rank == 0) {
        full.resize(n * n);
        std::copy(local.begin(), local.end(), full.begin());
        for (int source = 1; source < ranks; ++source) {
            size_t count = n * rowCount(n, ranks, source);
            size_t offset = n * firstRow(n, ranks, source);
            for (size_t done = 0; done < count;) {
                size_t chunk = std::min(chunkSize, count - done);
                MPI_Recv(full.data() + offset + done, static_cast<int>(chunk), MPI_UNSIGNED,
                         source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                done += chunk;
            }
        }
    } else {
        for (size_t done = 0; done < local.size();) {
            size_t chunk = std::min(chunkSize, local.size() - done);
            MPI_Send(local.data() + done, static_cast<int>(chunk), MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
            done += chunk;
        }
    }
    return full;
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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    size_t first = firstRow(numNodes, ranks, rank);
    size_t rows = rowCount(numNodes, ranks, rank);
    std::vector<unsigned int> dist(rows * numNodes);
    std::vector<unsigned int> path(rows * numNodes);
    initializeDistanceMatrix(dist, numNodes, rank, ranks);
    for (size_t i = 0; i < rows; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes,
                    static_cast<unsigned int>(first + i));
    }

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, rank, ranks);
    double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        long duration = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int status = 0;
    if (printResults || validate) {
        std::vector<unsigned int> full = gatherDistances(dist, numNodes, rank, ranks);
        if (rank == 0) {
            if (printResults) print_results_int(full, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
