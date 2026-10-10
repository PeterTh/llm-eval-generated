#include <algorithm>
#include <cerrno>
#include <climits>
#include <exception>
#include <limits>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous, balanced row ownership; ranks beyond numNodes remain idle.
size_t firstRow(size_t numNodes, int rank, int ranks) {
    return (numNodes / ranks) * rank +
           std::min(static_cast<size_t>(rank), numNodes % ranks);
}

// Chunk transfers so the total matrix size need not fit an MPI int count.
void transfer(unsigned int* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) {
            MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, comm);
        } else {
            MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, comm, MPI_STATUS_IGNORE);
        }
        data += chunk;
        count -= chunk;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t numNodes,
                              int rank, int ranks, MPI_Comm comm) {
    if (rank != 0) {
        transfer(dist.data(), dist.size(), 0, false, comm);
        return;
    }

    // Generate the original rand_r stream once, in its original row-major order.
    // Only one remote block is buffered, avoiding a replicated global matrix.
    unsigned int seed = 42;
    std::vector<unsigned int> buffer;
    for (int peer = 0; peer < ranks; ++peer) {
        const size_t begin = firstRow(numNodes, peer, ranks);
        const size_t rows = firstRow(numNodes, peer + 1, ranks) - begin;
        if (peer != 0) buffer.resize(rows * numNodes);
        unsigned int* block = peer == 0 ? dist.data() : buffer.data();
        for (size_t i = 0; i < rows * numNodes; ++i) {
            block[i] = 1 + static_cast<unsigned int>(
                static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / (double)RAND_MAX);
        }
        for (size_t i = 0; i < rows; ++i) {
            block[i * numNodes + begin + i] = 0;
        }
        if (peer != 0) transfer(block, rows * numNodes, peer, true, comm);
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t numNodes,
                          size_t begin, size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes,
                    static_cast<unsigned int>(begin + i));
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t numNodes,
                   size_t begin, size_t rows, int rank, int ranks, MPI_Comm comm) {
    std::vector<unsigned int> pivot(numNodes);
    int owner = 0;
    size_t ownerEnd = firstRow(numNodes, 1, ranks);
    for (size_t k = 0; k < numNodes; ++k) {
        if (k == ownerEnd) {
            ++owner;
            ownerEnd = firstRow(numNodes, owner + 1, ranks);
        }
        unsigned int* pivotRow = rank == owner
            ? dist.data() + (k - begin) * numNodes : pivot.data();
        MPI_Bcast(pivotRow, static_cast<int>(numNodes), MPI_UNSIGNED, owner, comm);

        for (size_t i = 0; i < rows; ++i) {
            unsigned int* row = dist.data() + i * numNodes;
            unsigned int* pathRow = path.data() + i * numNodes;
            const unsigned int distIK = row[k];
            // With positive weights and a zero diagonal, the pivot row never
            // changes during this step. It can be broadcast directly in place.
            if (distIK == 0) continue;
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivotRow[j];
                const bool shorter = newDist < row[j];
                pathRow[j] = shorter ? static_cast<unsigned int>(k) : pathRow[j];
                row[j] = shorter ? newDist : row[j];
            }
        }
    }
}

std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& dist,
                                         size_t numNodes, int rank, int ranks,
                                         MPI_Comm comm) {
    std::vector<unsigned int> result;
    if (rank == 0) {
        result.resize(numNodes * numNodes);
        std::copy(dist.begin(), dist.end(), result.begin());
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t begin = firstRow(numNodes, peer, ranks);
            const size_t rows = firstRow(numNodes, peer + 1, ranks) - begin;
            transfer(result.data() + begin * numNodes, rows * numNodes,
                     peer, false, comm);
        }
    } else {
        // MPI_Send does not modify its buffer.
        transfer(const_cast<unsigned int*>(dist.data()), dist.size(), 0, true, comm);
    }
    return result;
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

int runBenchmark(int argc, char** argv, int rank, int worldSize) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno != 0 || value == end || *end != '\0' || value[0] == '-' ||
                parsed > INT_MAX ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() /
                                        sizeof(unsigned int) / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", value);
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const int ranks = static_cast<int>(std::min(numNodes, static_cast<size_t>(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    std::vector<unsigned int> result;
    double elapsed = 0.0;
    if (comm != MPI_COMM_NULL) {
        const size_t begin = firstRow(numNodes, rank, ranks);
        const size_t rows = firstRow(numNodes, rank + 1, ranks) - begin;
        std::vector<unsigned int> dist(rows * numNodes);
        std::vector<unsigned int> path(rows * numNodes);
        initializeDistanceMatrix(dist, numNodes, rank, ranks, comm);
        initializePathMatrix(path, numNodes, begin, rows);
        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, begin, rows, rank, ranks, comm);
        const double localElapsed = MPI_Wtime() - start;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (printResults || validate) {
            result = gatherDistances(dist, numNodes, rank, ranks, comm);
        }
        MPI_Comm_free(&comm);
    }

    int status = 0;
    if (rank == 0) {
        if (numNodes == 0) printf("Computing shortest paths...\n");
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0);
        if (printResults) print_results_int(result, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(result, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, worldSize);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
