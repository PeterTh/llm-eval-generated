#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

constexpr unsigned int MAX_DISTANCE = 200;

// Give the first n % ranks ranks one additional row.
size_t firstRow(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(n % ranks, static_cast<size_t>(rank));
}

void initialize(std::vector<unsigned int>& dist, size_t n, int rank, int ranks) {
    if (n == 0) return;
    const size_t batchRows = std::max(size_t{1}, (size_t{1} << 18) / n);
    std::vector<unsigned int> buffer;
    unsigned int seed = 42;
    for (int owner = 0; owner < ranks; ++owner) {
        if (rank != 0 && rank != owner) continue;
        const size_t first = firstRow(n, owner, ranks);
        const size_t end = firstRow(n, owner + 1, ranks);
        for (size_t row = first; row < end; row += batchRows) {
            const size_t rows = std::min(batchRows, end - row);
            const size_t count = rows * n;
            if (rank == 0 && owner != 0) buffer.resize(count);
            unsigned int* data = rank == owner ? dist.data() + (row - first) * n : buffer.data();
            if (rank == 0) {
                for (size_t i = 0; i < count; ++i) {
                    data[i] = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                }
                for (size_t i = 0; i < rows; ++i) data[i * n + row + i] = 0;
                if (owner != 0) MPI_Send(data, count, MPI_UNSIGNED, owner, 0, MPI_COMM_WORLD);
            } else {
                MPI_Recv(data, count, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

inline void relax(unsigned int* row, const unsigned int* pivot, size_t k, size_t n) {
    const unsigned int via = row[k];
    for (size_t j = 0; j < n; ++j) row[j] = std::min(row[j], via + pivot[j]);
}

void floydWarshall(std::vector<unsigned int>& dist, size_t n, int rank, int ranks) {
    if (n == 0) return;
    const size_t rows = dist.size() / n;
    const size_t batch = std::min({size_t{16}, static_cast<size_t>(INT_MAX) / n,
                                   firstRow(n, 1, ranks)});
    std::vector<unsigned int> pivots(batch * n);
    for (int owner = 0; owner < ranks; ++owner) {
        const size_t first = firstRow(n, owner, ranks);
        const size_t end = firstRow(n, owner + 1, ranks);
        for (size_t k = first; k < end; k += batch) {
            const size_t count = std::min(batch, end - k);
            if (rank == owner) {
                std::copy_n(dist.data() + (k - first) * n, count * n, pivots.data());
                // Reconstruct each pivot as it stands at its own iteration.
                for (size_t r = 1; r < count; ++r) {
                    for (size_t t = 0; t < r; ++t) {
                        relax(pivots.data() + r * n, pivots.data() + t * n, k + t, n);
                    }
                }
            }
            MPI_Bcast(pivots.data(), count * n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            for (size_t i = 0; i < rows; ++i) {
                unsigned int* row = dist.data() + i * n;
                for (size_t t = 0; t < count; ++t) {
                    relax(row, pivots.data() + t * n, k + t, n);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n, int rank, int ranks) {
    const size_t first = firstRow(n, rank, ranks);
    const size_t rows = firstRow(n, rank + 1, ranks) - first;
    int valid = 1;
    for (size_t r = 0; r < rows; ++r) {
        if (dist[r * n + first + r] != 0) valid = 0;
    }
    const size_t sample = std::min(n, size_t{10});
    std::vector<unsigned int> source(n);
    int owner = 0;
    for (size_t i = 0; i < sample; ++i) {
        while (firstRow(n, owner + 1, ranks) <= i) ++owner;
        if (rank == owner) std::copy_n(dist.data() + (i - first) * n, n, source.data());
        MPI_Bcast(source.data(), n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t j = 0; j < sample; ++j) {
                if (source[first + r] + dist[r * n + j] < source[j]) valid = 0;
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return valid != 0;
}

void printResults(const std::vector<unsigned int>& dist, size_t n, int rank, int ranks) {
    if (ranks == 1) {
        print_results_int(dist, "DistanceMatrix");
        return;
    }
    std::vector<unsigned int> result;
    if (rank == 0) {
        result.resize(n * n);
        std::copy(dist.begin(), dist.end(), result.begin());
    }
    for (int owner = 1; owner < ranks; ++owner) {
        if (rank != 0 && rank != owner) continue;
        const size_t start = firstRow(n, owner, ranks) * n;
        const size_t length = firstRow(n, owner + 1, ranks) * n - start;
        for (size_t offset = 0; offset < length;) {
            const int count = std::min(length - offset, static_cast<size_t>(INT_MAX));
            if (rank == 0) {
                MPI_Recv(result.data() + start + offset, count, MPI_UNSIGNED, owner, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Send(dist.data() + offset, count, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
            }
            offset += count;
        }
    }
    if (rank == 0) print_results_int(result, "DistanceMatrix");
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

    size_t n = 512;
    bool validate = false;
    bool results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            int nodes = atoi(argv[++i]);
            if (nodes < 0) {
                if (rank == 0) fprintf(stderr, "Number of nodes must be nonnegative\n");
                MPI_Finalize();
                return 1;
            }
            n = nodes;
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            results = true;
        } else {
            bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    const size_t rows = firstRow(n, rank + 1, ranks) - firstRow(n, rank, ranks);
    std::vector<unsigned int> dist(rows * n);
    initialize(dist, n, rank, ranks);
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, n, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        const double ops = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0);
    }
    if (results) printResults(dist, n, rank, ranks);
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(dist, n, rank, ranks);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
