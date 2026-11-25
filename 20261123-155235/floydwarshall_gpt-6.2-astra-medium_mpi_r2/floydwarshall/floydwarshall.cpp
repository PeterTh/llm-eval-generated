#include <algorithm>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The first remainder ranks own one extra row.
struct Layout {
    int nodes;
    int ranks;

    int rows(int rank) const { return nodes / ranks + (rank < nodes % ranks); }
    int first(int rank) const { return rank * (nodes / ranks) + std::min(rank, nodes % ranks); }
};

void initialize(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                const Layout& layout, int rank, MPI_Datatype rowType) {
    constexpr int batchRows = 64;
    const size_t n = layout.nodes;
    if (rank == 0) {
        // Stream the original rand_r sequence in row order. Bound the send buffer
        // to one batch, even if a rank owns a large fraction of the graph.
        std::vector<unsigned int> buffer(n * std::min(batchRows, layout.rows(0)));
        unsigned int seed = 42;
        for (int target = 0; target < layout.ranks; ++target) {
            for (int start = 0; start < layout.rows(target);) {
                int count = std::min(batchRows, layout.rows(target) - start);
                auto* batch = target == 0 ? dist.data() + start * n : buffer.data();
                for (size_t i = 0; i < count * n; ++i) {
                    batch[i] = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                }
                if (target != 0)
                    MPI_Send(batch, count, rowType, target, 0, MPI_COMM_WORLD);
                start += count;
            }
        }
    } else {
        for (int start = 0; start < layout.rows(rank);) {
            const int count = std::min(batchRows, layout.rows(rank) - start);
            MPI_Recv(dist.data() + start * n, count, rowType, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            start += count;
        }
    }
    for (int i = 0; i < layout.rows(rank); ++i) {
        const int globalRow = layout.first(rank) + i;
        dist[i * n + globalRow] = 0;
        std::fill_n(path.data() + i * n, n, globalRow);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const Layout& layout, int rank, MPI_Datatype rowType) {
    constexpr int panelRows = 32;
    const size_t n = layout.nodes;
    std::vector<unsigned int> panel(n * std::min(panelRows, layout.rows(0)));
    for (int owner = 0; owner < layout.ranks; ++owner) {
        for (int start = 0; start < layout.rows(owner);) {
            const int count = std::min(panelRows, layout.rows(owner) - start);
            const int firstPivot = layout.first(owner) + start;
            if (rank == owner) {
                // Snapshot each pivot as it would appear at its own iteration.
                // Later pivots in the batch depend on the earlier snapshots.
                for (int p = 0; p < count; ++p) {
                    auto* pivot = panel.data() + p * n;
                    std::copy_n(dist.data() + (start + p) * n, n, pivot);
                    for (int q = 0; q < p; ++q) {
                        const auto* previous = panel.data() + q * n;
                        const unsigned int cost = pivot[firstPivot + q];
                        for (size_t j = 0; j < n; ++j)
                            pivot[j] = std::min(pivot[j], cost + previous[j]);
                    }
                }
            }
            MPI_Bcast(panel.data(), count, rowType, owner, MPI_COMM_WORLD);
            for (int i = 0; i < layout.rows(rank); ++i) {
                auto* __restrict__ row = dist.data() + i * n;
                auto* __restrict__ via = path.data() + i * n;
                // Distances only decrease, and all edge weights are nonnegative.
                const unsigned int upper = *std::max_element(row, row + n);
                for (int p = 0; p < count; ++p) {
                    const int k = firstPivot + p;
                    const unsigned int cost = row[k];
                    if (cost >= upper) continue;
                    const auto* pivot = panel.data() + p * n;
                    for (size_t j = 0; j < n; ++j) {
                        const unsigned int candidate = cost + pivot[j];
                        if (candidate < row[j]) {
                            row[j] = candidate;
                            via[j] = k;
                        }
                    }
                }
            }
            start += count;
        }
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
    
    if (numNodes < 0) {
        if (rank == 0) fprintf(stderr, "Number of nodes must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %d\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    const Layout layout{numNodes, ranks};
    const size_t n = numNodes;
    MPI_Datatype rowType;
    MPI_Type_contiguous(numNodes, MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);
    std::vector<unsigned int> dist(layout.rows(rank) * n);
    std::vector<unsigned int> path(dist.size());
    initialize(dist, path, layout, rank, rowType);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    floydWarshall(dist, path, layout, rank, rowType);
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        double ops = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0);
    }

    if (printResults) {
        std::vector<int> counts, offsets;
        std::vector<unsigned int> result;
        if (rank == 0) {
            result.resize(n * n);
            for (int r = 0; r < ranks; ++r) {
                counts.push_back(layout.rows(r));
                offsets.push_back(layout.first(r));
            }
        }
        MPI_Gatherv(dist.data(), layout.rows(rank), rowType, result.data(), counts.data(),
                    offsets.data(), rowType, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results_int(result, "DistanceMatrix");
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        for (int i = 0; i < layout.rows(rank); ++i)
            if (dist[i * n + layout.first(rank) + i] != 0) valid = 0;

        const int samples = std::min(numNodes, 10);
        std::vector<unsigned int> row(n);
        for (int owner = 0; layout.first(owner) < samples; ++owner) {
            const int count = std::min(layout.rows(owner), samples - layout.first(owner));
            for (int i = 0; i < count; ++i) {
                if (rank == owner) std::copy_n(dist.data() + i * n, n, row.data());
                MPI_Bcast(row.data(), 1, rowType, owner, MPI_COMM_WORLD);
                for (int k = 0; k < layout.rows(rank); ++k) {
                    const auto ik = row[layout.first(rank) + k];
                    for (int j = 0; j < samples; ++j) {
                        const auto kj = dist[k * n + j];
                        if (ik < INF && kj < INF && ik + kj < row[j]) valid = 0;
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Type_free(&rowType);
    MPI_Finalize();
    return valid ? 0 : 1;
}
