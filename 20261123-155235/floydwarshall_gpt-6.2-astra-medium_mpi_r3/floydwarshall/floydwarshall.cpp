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
    size_t nodes;
    int ranks;

    size_t first(int rank) const {
        return (nodes / ranks) * rank + std::min<size_t>(rank, nodes % ranks);
    }

    size_t rows(int rank) const {
        return first(rank + 1) - first(rank);
    }
};

void initialize(std::vector<unsigned>& dist, std::vector<unsigned>& path,
                const Partition& grid, int rank) {
    unsigned seed = 42;
#ifdef __GLIBC__
    // glibc rand_r advances this affine generator three times per value.
    // Exponentiate the recurrence to start at this rank's serial offset.
    unsigned multiplier = 1103515245u, increment = 12345u;
    for (size_t steps = 3 * grid.first(rank) * grid.nodes; steps; steps >>= 1) {
        if (steps & 1) seed = multiplier * seed + increment;
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
#else
    // Other implementations may have a different rand_r state transition.
    if (rank > 0)
        MPI_Recv(&seed, 1, MPI_UNSIGNED, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
    for (unsigned& value : dist)
        value = 1 + static_cast<unsigned>(double(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
#ifndef __GLIBC__
    if (rank + 1 < grid.ranks)
        MPI_Send(&seed, 1, MPI_UNSIGNED, rank + 1, 0, MPI_COMM_WORLD);
#endif
    for (size_t i = 0; i < grid.rows(rank); ++i) {
        const size_t source = grid.first(rank) + i;
        dist[i * grid.nodes + source] = 0;
        std::fill_n(path.data() + i * grid.nodes, grid.nodes, source);
    }
}

void floydWarshall(std::vector<unsigned>& dist, std::vector<unsigned>& path,
                   const Partition& grid, int rank, MPI_Datatype rowType) {
    constexpr size_t batch = 16;
    const size_t n = grid.nodes;
    const size_t localRows = grid.rows(rank);
    std::vector<unsigned> pivots(std::min(batch, grid.rows(0)) * n);

    for (int owner = 0; owner < grid.ranks; ++owner) {
        for (size_t offset = 0; offset < grid.rows(owner); offset += batch) {
            const size_t count = std::min(batch, grid.rows(owner) - offset);
            const size_t first = grid.first(owner) + offset;
            if (rank == owner) {
                std::copy_n(dist.data() + offset * n, count * n, pivots.data());
                // Snapshot each pivot as it would appear at its serial iteration.
                for (size_t i = 1; i < count; ++i) {
                    unsigned* row = pivots.data() + i * n;
                    for (size_t k = 0; k < i; ++k) {
                        const unsigned via = row[first + k];
                        const unsigned* pivot = pivots.data() + k * n;
                        for (size_t j = 0; j < n; ++j)
                            row[j] = std::min(row[j], via + pivot[j]);
                    }
                }
            }
            MPI_Bcast(pivots.data(), count, rowType, owner, MPI_COMM_WORLD);
            for (size_t i = 0; i < localRows; ++i) {
                unsigned* row = dist.data() + i * n;
                unsigned* route = path.data() + i * n;
                for (size_t k = 0; k < count; ++k) {
                    const unsigned via = row[first + k];
                    const unsigned* pivot = pivots.data() + k * n;
                    for (size_t j = 0; j < n; ++j) {
                        const unsigned candidate = via + pivot[j];
                        if (candidate < row[j]) {
                            row[j] = candidate;
                            route[j] = first + k;
                        }
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned>& dist, const Partition& grid, int rank) {
    const size_t n = grid.nodes;
    const size_t sample = std::min<size_t>(10, n);
    int valid = 1;
    for (size_t i = 0; i < grid.rows(rank); ++i)
        if (dist[i * n + grid.first(rank) + i] != 0) valid = 0;

    // Each rank checks the intermediate vertices whose rows it owns.
    std::vector<unsigned> source(n);
    for (int owner = 0; owner < grid.ranks && grid.first(owner) < sample; ++owner) {
        for (size_t i = grid.first(owner); i < std::min(sample, grid.first(owner + 1)); ++i) {
            if (rank == owner)
                std::copy_n(dist.data() + (i - grid.first(owner)) * n, n, source.data());
            MPI_Bcast(source.data(), n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            for (size_t k = 0; k < grid.rows(rank); ++k) {
                const unsigned via = source[grid.first(rank) + k];
                for (size_t j = 0; j < sample; ++j) {
                    const unsigned end = dist[k * n + j];
                    if (via < INF && end < INF && via + end < source[j]) valid = 0;
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
    size_t numNodes = 512;
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

    if (numNodes > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Number of nodes exceeds the supported range.\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const Partition grid{numNodes, ranks};
    std::vector<unsigned> dist(grid.rows(rank) * numNodes), path(dist.size());
    initialize(dist, path, grid, rank);
    MPI_Datatype rowType;
    MPI_Type_contiguous(numNodes, MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, grid, rank, rowType);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        const double ops = double(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / seconds / 1e9);
    }

    if (printResults) {
        std::vector<unsigned> result;
        std::vector<int> counts, offsets;
        if (rank == 0) {
            result.resize(numNodes * numNodes);
            for (int r = 0; r < ranks; ++r) {
                counts.push_back(grid.rows(r));
                offsets.push_back(grid.first(r));
            }
        }
        MPI_Gatherv(dist.data(), grid.rows(rank), rowType, result.data(), counts.data(),
                    offsets.data(), rowType, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results_int(result, "DistanceMatrix");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(dist, grid, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Type_free(&rowType);
    MPI_Finalize();
    return valid ? 0 : 1;
}
