#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

struct Partition {
    size_t n;
    int ranks;

    size_t first(int rank) const {
        return (n / ranks) * rank + std::min<size_t>(rank, n % ranks);
    }

    size_t rows(int rank) const {
        return first(rank + 1) - first(rank);
    }
};

// Generate the original rand_r stream once. Bounded transfers avoid both a
// full matrix on the generator and MPI's int limit on message counts.
constexpr size_t transferSize = 256 * 1024;

void initialize(std::vector<unsigned int>& dist, const Partition& part, int rank) {
    if (rank == 0) {
        unsigned int seed = 42;
        std::vector<unsigned int> buffer(std::min(transferSize, part.rows(0) * part.n));
        for (int dest = 0; dest < part.ranks; ++dest) {
            const size_t elements = part.rows(dest) * part.n;
            for (size_t offset = 0; offset < elements; offset += transferSize) {
                const int count = std::min(transferSize, elements - offset);
                unsigned int* chunk = dest == 0 ? dist.data() + offset : buffer.data();
                for (int j = 0; j < count; ++j) {
                    chunk[j] = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                }
                if (dest != 0)
                    MPI_Send(chunk, count, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (size_t offset = 0; offset < dist.size(); offset += transferSize) {
            const int count = std::min(transferSize, dist.size() - offset);
            MPI_Recv(dist.data() + offset, count, MPI_UNSIGNED, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    for (size_t i = 0; i < part.rows(rank); ++i)
        dist[i * part.n + part.first(rank) + i] = 0;
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const Partition& part, int rank) {
    const size_t n = part.n;
    if (n == 0) return;
    const size_t width = std::min({size_t{32}, part.rows(0), INT_MAX / n});
    std::vector<unsigned int> panel(width * n);
    for (int owner = 0; owner < part.ranks; ++owner) {
        for (size_t first = part.first(owner); first < part.first(owner + 1); first += width) {
            const size_t count = std::min(width, part.first(owner + 1) - first);
            if (rank == owner) {
                // Snapshot each pivot row as it would appear at its own k step.
                const auto begin = dist.data() + (first - part.first(owner)) * n;
                std::copy_n(begin, count * n, panel.data());
                for (size_t p = 0; p < count; ++p) {
                    auto row = panel.data() + p * n;
                    for (size_t q = 0; q < p; ++q) {
                        const auto pivot = panel.data() + q * n;
                        const auto via = row[first + q];
                        for (size_t j = 0; j < n; ++j)
                            row[j] = std::min(row[j], via + pivot[j]);
                    }
                }
            }
            MPI_Bcast(panel.data(), static_cast<int>(count * n), MPI_UNSIGNED,
                      owner, MPI_COMM_WORLD);
            for (size_t i = 0; i < part.rows(rank); ++i) {
                auto row = dist.data() + i * n;
                auto predecessors = path.data() + i * n;
                for (size_t p = 0; p < count; ++p) {
                    const unsigned int k = first + p;
                    const auto via = row[k];
                    const auto pivot = panel.data() + p * n;
                    for (size_t j = 0; j < n; ++j) {
                        const auto candidate = via + pivot[j];
                        if (candidate < row[j]) {
                            row[j] = candidate;
                            predecessors[j] = k;
                        }
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const Partition& part, int rank) {
    int valid = 1;
    const size_t start = part.first(rank);
    for (size_t i = 0; i < part.rows(rank); ++i)
        if (dist[i * part.n + start + i] != 0) valid = 0;

    const size_t sample = std::min<size_t>(10, part.n);
    std::vector<unsigned int> source(part.n);
    int owner = 0;
    for (size_t i = 0; i < sample; ++i) {
        while (part.first(owner + 1) <= i) ++owner;
        if (rank == owner)
            std::copy_n(dist.data() + (i - start) * part.n, part.n, source.data());
        MPI_Bcast(source.data(), static_cast<int>(part.n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t k = 0; k < part.rows(rank); ++k) {
            const auto via = source[start + k];
            for (size_t j = 0; j < sample; ++j) {
                const auto kj = dist[k * part.n + j];
                if (via < INF && kj < INF && via + kj < source[j]) valid = 0;
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return valid != 0;
}

void printResults(const std::vector<unsigned int>& dist, const Partition& part, int rank) {
    if (part.ranks == 1) {
        print_results_int(dist, "DistanceMatrix");
        return;
    }
    // The shared results formatter hashes the complete matrix in row order.
    if (rank == 0) {
        std::vector<unsigned int> full(part.n * part.n);
        std::copy(dist.begin(), dist.end(), full.begin());
        for (int src = 1; src < part.ranks; ++src) {
            const size_t end = part.first(src + 1) * part.n;
            for (size_t pos = part.first(src) * part.n; pos < end; pos += transferSize)
                MPI_Recv(full.data() + pos, std::min(transferSize, end - pos), MPI_UNSIGNED,
                         src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        print_results_int(full, "DistanceMatrix");
    } else {
        for (size_t pos = 0; pos < dist.size(); pos += transferSize)
            MPI_Send(dist.data() + pos, std::min(transferSize, dist.size() - pos),
                     MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
    }
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
    bool results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const int n = atoi(argv[++i]);
            if (n < 0) {
                if (rank == 0) printf("Number of nodes must be nonnegative\n");
                return 1;
            }
            numNodes = n;
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            results = true;
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
    const Partition part{numNodes, ranks};
    std::vector<unsigned int> dist(part.rows(rank) * numNodes);
    std::vector<unsigned int> path(dist.size());
    initialize(dist, part, rank);
    for (size_t i = 0; i < part.rows(rank); ++i)
        std::fill_n(path.data() + i * numNodes, numNodes, part.first(rank) + i);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, part, rank);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }

    if (results) printResults(dist, part, rank);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(dist, part, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        if (!valid) return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
