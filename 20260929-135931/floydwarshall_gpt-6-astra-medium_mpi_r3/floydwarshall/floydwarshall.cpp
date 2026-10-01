#include <algorithm>
#include <cerrno>
#include <climits>
#include <limits>
#include <exception>
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

// Balanced contiguous row ownership, including non-divisible graph sizes.
size_t firstRow(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(n % ranks, static_cast<size_t>(rank));
}

int rowOwner(size_t row, size_t n, int ranks) {
    const size_t small = n / ranks;
    const size_t largeRows = (small + 1) * (n % ranks);
    return row < largeRows ? static_cast<int>(row / (small + 1))
                          : static_cast<int>(n % ranks + (row - largeRows) / small);
}

// Stream the original rand_r sequence from rank zero, using bounded storage.
// Chunked transfers also avoid MPI's int count limit for entire row blocks.
constexpr size_t transferChunk = 1 << 20;
void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n,
                        size_t first, int rank, int ranks, MPI_Comm comm) {
    if (rank == 0) {
        unsigned int seed = 42;
        std::vector<unsigned int> buffer(std::min(transferChunk, dist.size()));
        for (int r = 0; r < ranks; ++r) {
            const size_t count = (firstRow(n, r + 1, ranks) - firstRow(n, r, ranks)) * n;
            for (size_t offset = 0; offset < count; offset += transferChunk) {
                const int chunk = static_cast<int>(std::min(transferChunk, count - offset));
                unsigned int* target = r == 0 ? dist.data() + offset : buffer.data();
                for (int j = 0; j < chunk; ++j) {
                    target[j] = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / (double)RAND_MAX);
                }
                if (r != 0) MPI_Send(target, chunk, MPI_UNSIGNED, r, 0, comm);
            }
        }
    } else {
        for (size_t offset = 0; offset < dist.size(); offset += transferChunk) {
            const int chunk = static_cast<int>(std::min(transferChunk, dist.size() - offset));
            MPI_Recv(dist.data() + offset, chunk, MPI_UNSIGNED, 0, 0, comm, MPI_STATUS_IGNORE);
        }
    }
    for (size_t i = 0; i < dist.size() / n; ++i) {
        dist[i * n + first + i] = 0;
        std::fill_n(path.data() + i * n, n, static_cast<unsigned int>(first + i));
    }
}

inline void updateRow(unsigned int* row, unsigned int* path,
                      const unsigned int* pivot, size_t n, size_t k) {
    const unsigned int dik = row[k];
    // With nonnegative weights, the pivot row cannot change at this step.
    if (dik == 0) return;
    for (size_t j = 0; j < n; ++j) {
        const unsigned int candidate = dik + pivot[j];
        const bool improve = candidate < row[j];
        path[j] = improve ? static_cast<unsigned int>(k) : path[j];
        row[j] = improve ? candidate : row[j];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n, size_t first,
                   int rank, int ranks, MPI_Comm comm) {
    const size_t rows = dist.size() / n;
    std::vector<unsigned int> pivot(n), next(n);
    if (rank == 0) std::copy_n(dist.data(), n, pivot.data());
    MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, 0, comm);
    for (size_t k = 0; k < n; ++k) {
        MPI_Request request = MPI_REQUEST_NULL;
        size_t earlyRow = rows;
        if (k + 1 < n) {
            const int owner = rowOwner(k + 1, n, ranks);
            if (rank == owner) {
                earlyRow = k + 1 - first;
                updateRow(dist.data() + earlyRow * n, path.data() + earlyRow * n,
                          pivot.data(), n, k);
                std::copy_n(dist.data() + earlyRow * n, n, next.data());
            }
            // Rows are independent within k. Compute the next pivot first and
            // overlap its broadcast with the remaining local updates. Double
            // buffering keeps the in-flight buffer untouched until MPI_Wait.
            MPI_Ibcast(next.data(), static_cast<int>(n), MPI_UNSIGNED, owner, comm, &request);
        }
        for (size_t i = 0; i < rows; ++i) {
            if (i != earlyRow) {
                updateRow(dist.data() + i * n, path.data() + i * n, pivot.data(), n, k);
            }
        }
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        pivot.swap(next);
    }
}

void gatherDistances(const std::vector<unsigned int>& local,
                     std::vector<unsigned int>& full, size_t n,
                     int rank, int ranks, MPI_Comm comm) {
    if (rank == 0) {
        full.resize(n * n);
        std::copy(local.begin(), local.end(), full.begin());
        for (int r = 1; r < ranks; ++r) {
            const size_t start = firstRow(n, r, ranks) * n;
            const size_t count = (firstRow(n, r + 1, ranks) - firstRow(n, r, ranks)) * n;
            for (size_t offset = 0; offset < count; offset += transferChunk) {
                const int chunk = static_cast<int>(std::min(transferChunk, count - offset));
                MPI_Recv(full.data() + start + offset, chunk, MPI_UNSIGNED, r, 1, comm,
                         MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size(); offset += transferChunk) {
            const int chunk = static_cast<int>(std::min(transferChunk, local.size() - offset));
            MPI_Send(local.data() + offset, chunk, MPI_UNSIGNED, 0, 1, comm);
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

int run(int argc, char** argv, int rank, int ranks) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno != 0 || arg[0] == '-' || end == arg || *end != '\0' ||
                value > INT_MAX || value > std::numeric_limits<size_t>::max() ||
                (value != 0 && value > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / value)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", arg);
                return 1;
            }
            numNodes = static_cast<size_t>(value);
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

    // Avoid involving empty ranks in every pivot broadcast. Rank zero remains
    // active for the empty graph so output and validation retain their semantics.
    const int activeRanks = static_cast<int>(std::min(
        static_cast<size_t>(ranks), std::max(numNodes, size_t{1})));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = 0;
    if (rank < activeRanks) {
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        const size_t first = firstRow(numNodes, rank, activeRanks);
        const size_t rows = firstRow(numNodes, rank + 1, activeRanks) - first;
        std::vector<unsigned int> dist(rows * numNodes), path(rows * numNodes);
        if (numNodes != 0) {
            initializeMatrices(dist, path, numNodes, first, rank, activeRanks, comm);
        }
        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        if (numNodes != 0) {
            floydWarshall(dist, path, numNodes, first, rank, activeRanks, comm);
        }
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
        }
        if (printResults || validate) {
            std::vector<unsigned int> full;
            gatherDistances(dist, full, numNodes, rank, activeRanks, comm);
            if (rank == 0) {
                if (printResults) print_results_int(full, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(full, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    status = valid ? 0 : 1;
                }
            }
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
