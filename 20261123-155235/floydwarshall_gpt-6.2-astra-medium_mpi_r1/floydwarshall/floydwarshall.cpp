#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Give the first n % ranks processes one extra row.
size_t firstRow(int rank, size_t n, int ranks) {
    return rank * (n / ranks) + std::min<size_t>(rank, n % ranks);
}

void initialize(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                size_t n, int rank, int ranks, MPI_Datatype rowType) {
    const size_t first = firstRow(rank, n, ranks);
    const size_t rows = firstRow(rank + 1, n, ranks) - first;
    for (size_t i = 0; i < rows; ++i)
        std::fill_n(path.data() + i * n, n, first + i);

    // Stream the original rand_r sequence in row order with bounded scratch space.
    constexpr size_t chunk = 32;
    if (rank == 0) {
        unsigned int seed = 42;
        std::vector<unsigned int> buffer(std::min(chunk, firstRow(1, n, ranks)) * n);
        for (int owner = 0; owner < ranks; ++owner) {
            const size_t end = firstRow(owner + 1, n, ranks);
            for (size_t i = firstRow(owner, n, ranks); i < end; i += chunk) {
                const size_t count = std::min(chunk, end - i);
                auto* out = owner == 0 ? dist.data() + i * n : buffer.data();
                for (size_t j = 0; j < count * n; ++j)
                    out[j] = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                for (size_t j = 0; j < count; ++j)
                    out[j * n + i + j] = 0;
                if (owner != 0)
                    MPI_Send(out, count, rowType, owner, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (size_t i = 0; i < rows; i += chunk)
            MPI_Recv(dist.data() + i * n, std::min(chunk, rows - i), rowType,
                     0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, int rank, int ranks, MPI_Datatype rowType) {
    constexpr size_t batch = 16;
    std::vector<unsigned int> pivots(std::min(batch, firstRow(1, n, ranks)) * n);
    const size_t rows = firstRow(rank + 1, n, ranks) - firstRow(rank, n, ranks);
    for (int owner = 0; owner < ranks; ++owner) {
        const size_t first = firstRow(owner, n, ranks);
        const size_t end = firstRow(owner + 1, n, ranks);
        for (size_t k = first; k < end; k += batch) {
            const size_t count = std::min(batch, end - k);
            if (rank == owner) {
                // Snapshot each pivot as it would appear at its own iteration.
                for (size_t t = 0; t < count; ++t) {
                    auto* dst = pivots.data() + t * n;
                    std::copy_n(dist.data() + (k - first + t) * n, n, dst);
                    for (size_t s = 0; s < t; ++s) {
                        const auto* src = pivots.data() + s * n;
                        const auto via = dst[k + s];
                        for (size_t j = 0; j < n; ++j)
                            dst[j] = std::min(dst[j], via + src[j]);
                    }
                }
            }
            MPI_Bcast(pivots.data(), count, rowType, owner, MPI_COMM_WORLD);
            // Reuse the local row in cache through all pivots in the batch.
            for (size_t i = 0; i < rows; ++i) {
                auto* __restrict__ row = dist.data() + i * n;
                auto* __restrict__ intermediate = path.data() + i * n;
                for (size_t t = 0; t < count; ++t) {
                    const auto* pivot = pivots.data() + t * n;
                    const auto via = row[k + t];
                    for (size_t j = 0; j < n; ++j) {
                        const auto candidate = via + pivot[j];
                        if (candidate < row[j]) {
                            row[j] = candidate;
                            intermediate[j] = k + t;
                        }
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n, int rank, int ranks,
                    MPI_Datatype rowType) {
    const size_t first = firstRow(rank, n, ranks);
    const size_t end = firstRow(rank + 1, n, ranks);
    const size_t sample = std::min<size_t>(10, n);
    std::vector<unsigned int> sampleRows(sample * n);
    for (int owner = 0; owner < ranks; ++owner) {
        const size_t begin = firstRow(owner, n, ranks);
        if (begin >= sample) break;
        const size_t count = std::min(sample, firstRow(owner + 1, n, ranks)) - begin;
        auto* dest = sampleRows.data() + begin * n;
        if (owner == rank) std::copy_n(dist.data(), count * n, dest);
        MPI_Bcast(dest, count, rowType, owner, MPI_COMM_WORLD);
    }

    // Each rank checks its diagonal and the inequalities that use its local k rows.
    int valid = 1;
    for (size_t k = first; k < end && valid; ++k) {
        const auto* row = dist.data() + (k - first) * n;
        if (row[k] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", k, k);
            valid = 0;
        }
        for (size_t i = 0; i < sample && valid; ++i) {
            const auto* source = sampleRows.data() + i * n;
            for (size_t j = 0; j < sample; ++j) {
                if (source[k] < INF && row[j] < INF &&
                    source[k] + row[j] < source[j]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                           i, j, k);
                    valid = 0;
                    break;
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return valid;
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
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
            if (n < 0) {
                if (rank == 0) fprintf(stderr, "Number of nodes must be nonnegative\n");
                MPI_Finalize();
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
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
        printf("Number of nodes: %d\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    MPI_Datatype rowType;
    MPI_Type_contiguous(n, MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);
    const size_t rows = firstRow(rank + 1, n, ranks) - firstRow(rank, n, ranks);
    std::vector<unsigned int> dist(rows * n), path(rows * n);
    initialize(dist, path, n, rank, ranks, rowType);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    floydWarshall(dist, path, n, rank, ranks, rowType);
    double elapsed = MPI_Wtime() - start, seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        printf("Performance: %.3f GOPS\n", static_cast<double>(n) * n * n / seconds / 1e9);
    }

    bool valid = true;
    if (printResults) {
        std::vector<unsigned int> full;
        std::vector<int> counts, offsets;
        if (rank == 0) {
            full.resize(static_cast<size_t>(n) * n);
            counts.resize(ranks);
            offsets.resize(ranks);
            for (int p = 0; p < ranks; ++p) {
                offsets[p] = firstRow(p, n, ranks);
                counts[p] = firstRow(p + 1, n, ranks) - offsets[p];
            }
        }
        MPI_Gatherv(dist.data(), rows, rowType, full.data(), counts.data(),
                    offsets.data(), rowType, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results_int(full, "DistanceMatrix");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(dist, n, rank, ranks, rowType);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Type_free(&rowType);
    MPI_Finalize();
    return valid ? 0 : 1;
}
