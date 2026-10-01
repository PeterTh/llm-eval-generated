#include <algorithm>
#include <mpi.h>
#include <cerrno>
#include <climits>
#include <limits>
#include <exception>
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

// Contiguous, balanced row ownership; also supports ranks with no rows.
size_t firstRow(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(n % ranks, static_cast<size_t>(rank));
}

// Chunk point-to-point transfers so a local matrix can exceed MPI's int count.
void transfer(unsigned int* data, size_t count, int peer, bool send) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send)
            MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD);
        else
            MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path,
                        size_t n, int rank, int ranks) {
    const size_t first = firstRow(n, rank, ranks);
    const size_t rows = firstRow(n, rank + 1, ranks) - first;
    if (rank == 0) {
        // Generate the original rand_r stream exactly once, in its original
        // order. A reusable block avoids allocating the entire graph on root.
        unsigned int seed = 42;
        std::vector<unsigned int> block;
        if (ranks > 1) block.resize(dist.size());
        for (int peer = 0; peer < ranks; ++peer) {
            const size_t begin = firstRow(n, peer, ranks);
            const size_t count = (firstRow(n, peer + 1, ranks) - begin) * n;
            unsigned int* data = peer == 0 ? dist.data() : block.data();
            for (size_t i = 0; i < count; ++i)
                data[i] = 1 + static_cast<unsigned int>(
                    static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
            for (size_t i = 0; i < count / n; ++i)
                data[i * n + begin + i] = 0;
            if (peer != 0) transfer(data, count, peer, true);
        }
    } else {
        transfer(dist.data(), dist.size(), 0, false);
    }
    // Original path[j + i*n] is initialized to i, including the diagonal.
    for (size_t i = 0; i < rows; ++i)
        std::fill_n(path.data() + i * n, n, static_cast<unsigned int>(first + i));
}

void floydWarshall(std::vector<unsigned int>& dist,
                  std::vector<unsigned int>& path,
                  size_t n, int rank, int ranks,
                  std::vector<unsigned int>& pivot) {
    const size_t first = firstRow(n, rank, ranks);
    const size_t rows = firstRow(n, rank + 1, ranks) - first;
    int owner = 0;
    for (size_t k = 0; k < n; ++k) {
        while (k >= firstRow(n, owner + 1, ranks)) ++owner;
        unsigned int* pivotRow = rank == owner
            ? dist.data() + (k - first) * n : pivot.data();
        MPI_Bcast(pivotRow, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t i = 0; i < rows; ++i) {
            // Nonnegative weights and a zero diagonal mean the pivot row
            // cannot change. Keeping it read-only also permits vectorization.
            if (first + i == k) continue;
            unsigned int* __restrict__ row = dist.data() + i * n;
            unsigned int* __restrict__ paths = path.data() + i * n;
            const unsigned int via = row[k];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = via + pivotRow[j];
                const bool improve = candidate < row[j];
                paths[j] = improve ? static_cast<unsigned int>(k) : paths[j];
                row[j] = improve ? candidate : row[j];
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
    int status = 0;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value == end || *end || *value == '-' || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", value);
                status = 1;
                break;
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
            break;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            status = 1;
            break;
        }
    }
    if (help || status) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return status;
    }
    try {
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        const size_t first = firstRow(numNodes, rank, ranks);
        const size_t rows = firstRow(numNodes, rank + 1, ranks) - first;
        std::vector<unsigned int> dist(rows * numNodes);
        std::vector<unsigned int> path(rows * numNodes);
        std::vector<unsigned int> pivot(numNodes);
        if (numNodes) initializeMatrices(dist, path, numNodes, rank, ranks);
        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, rank, ranks, pivot);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0.0);
        }
        // Gathering is outside the timed kernel and only needed by the
        // original global validation and order-sensitive result hash.
        if (printResults || validate) {
            std::vector<unsigned int> result;
            if (rank == 0) {
                result.resize(numNodes * numNodes);
                std::copy(dist.begin(), dist.end(), result.begin());
                for (int peer = 1; peer < ranks; ++peer) {
                    const size_t begin = firstRow(numNodes, peer, ranks);
                    const size_t count = (firstRow(numNodes, peer + 1, ranks) - begin) * numNodes;
                    if (count) transfer(result.data() + begin * numNodes, count, peer, false);
                }
                if (printResults) print_results_int(result, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(result, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    status = valid ? 0 : 1;
                }
            } else {
                transfer(dist.data(), dist.size(), 0, true);
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
