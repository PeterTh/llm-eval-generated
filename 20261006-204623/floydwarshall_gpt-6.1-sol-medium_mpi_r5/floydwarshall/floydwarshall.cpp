#include <algorithm>
#include <cerrno>
#include <climits>
#include <limits>
#include <exception>
#include <mpi.h>
#include <cmath>
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

// Contiguous, balanced row ownership also keeps the output in its original order.
size_t firstRow(const size_t n, const int rank, const int ranks) {
    return (n / ranks) * rank + std::min(n % ranks, static_cast<size_t>(rank));
}

// Keep MPI counts bounded even when a rank owns more than INT_MAX elements.
constexpr size_t transferElements = 1 << 20;

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t n,
                              const int rank, const int ranks) {
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    std::vector<unsigned int> buffer;
    if (rank == 0 && ranks > 1) {
        buffer.resize(std::min(transferElements, (firstRow(n, 2, ranks) -
                                                   firstRow(n, 1, ranks)) * n));
    }
    // Only rank zero consumes the original rand_r stream. Stream each partition
    // to its owner so initialization never requires a full matrix on rank zero.
    for (int owner = 0; owner < ranks; ++owner) {
        if (rank != 0 && rank != owner) continue;
        const size_t begin = firstRow(n, owner, ranks) * n;
        const size_t count = (firstRow(n, owner + 1, ranks) * n) - begin;
        for (size_t offset = 0; offset < count; offset += transferElements) {
            const int length = static_cast<int>(std::min(transferElements, count - offset));
            if (rank == 0) {
                unsigned int* values = owner == 0 ? dist.data() + offset : buffer.data();
                for (int j = 0; j < length; ++j) {
                    const size_t index = begin + offset + j;
                    values[j] = 1 + static_cast<unsigned int>(range * rand_r(&seed) /
                                                             static_cast<double>(RAND_MAX));
                    if (index / n == index % n) values[j] = 0;
                }
                if (owner != 0) MPI_Send(values, length, MPI_UNSIGNED, owner, 0, MPI_COMM_WORLD);
            } else {
                MPI_Recv(dist.data() + offset, length, MPI_UNSIGNED, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t n) {
    for (size_t offset = 0; offset < path.size(); offset += n) {
        for (size_t j = 0; j < n; ++j) path[offset + j] = static_cast<unsigned int>(j);
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, const size_t n,
                   const int rank, const int ranks,
                   std::vector<unsigned int>& pivot) {
    const size_t begin = firstRow(n, rank, ranks);
    const size_t rows = firstRow(n, rank + 1, ranks) - begin;
    int owner = 0;
    for (size_t k = 0; k < n; ++k) {
        while (k >= firstRow(n, owner + 1, ranks)) ++owner;
        // The owner can broadcast directly from its matrix: nonnegative weights
        // and the zero diagonal ensure that the pivot row cannot change at k.
        const unsigned int* pivotRow = rank == owner ? dist.data() + (k - begin) * n
                                                     : pivot.data();
        MPI_Bcast(const_cast<unsigned int*>(pivotRow), static_cast<int>(n),
                  MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t i = 0; i < rows; ++i) {
            unsigned int* __restrict__ row = dist.data() + i * n;
            unsigned int* __restrict__ pathRow = path.data() + i * n;
            const unsigned int distanceToPivot = row[k];
            if (begin + i == k) continue;
            // Contiguous independent updates allow the compiler to use SIMD.
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = distanceToPivot + pivotRow[j];
                if (candidate < row[j]) {
                    row[j] = candidate;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

std::vector<unsigned int> gatherDistanceMatrix(const std::vector<unsigned int>& dist,
                                               const size_t n, const int rank,
                                               const int ranks) {
    std::vector<unsigned int> result;
    if (rank == 0) {
        result.resize(n * n);
        std::copy(dist.begin(), dist.end(), result.begin());
    }
    for (int owner = 1; owner < ranks; ++owner) {
        if (rank != 0 && rank != owner) continue;
        const size_t begin = firstRow(n, owner, ranks) * n;
        const size_t count = firstRow(n, owner + 1, ranks) * n - begin;
        for (size_t offset = 0; offset < count; offset += transferElements) {
            const int length = static_cast<int>(std::min(transferElements, count - offset));
            if (rank == 0) {
                MPI_Recv(result.data() + begin + offset, length, MPI_UNSIGNED,
                         owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Send(dist.data() + offset, length, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
            }
        }
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(argument, &end, 10);
            if (errno != 0 || argument[0] == '-' || end == argument || *end != '\0' ||
                value > INT_MAX || (value != 0 &&
                value > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / value)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(value);
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

    int result = 0;
    try {
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        const size_t localRows = firstRow(numNodes, rank + 1, ranks) -
                                 firstRow(numNodes, rank, ranks);
        std::vector<unsigned int> dist(localRows * numNodes);
        std::vector<unsigned int> path(localRows * numNodes);
        std::vector<unsigned int> pivot(numNodes);
        initializeDistanceMatrix(dist, numNodes, rank, ranks);
        initializePathMatrix(path, numNodes);

        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, rank, ranks, pivot);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", duration > 0.0 ? ops / duration / 1e9 : 0.0);
        }

        if (printResults || validate) {
            const auto fullDistance = gatherDistanceMatrix(dist, numNodes, rank, ranks);
            if (rank == 0) {
                if (printResults) print_results_int(fullDistance, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(fullDistance, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    result = valid ? 0 : 1;
                }
            }
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // A local allocation failure must not strand peers inside a collective.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
