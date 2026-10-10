#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous blocks in each dimension, with at most one extra element per rank.
struct Partition {
    size_t begin;
    size_t size;
};

Partition partition(size_t n, int parts, int coordinate) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    return {base * coordinate + std::min(extra, static_cast<size_t>(coordinate)),
            base + (static_cast<size_t>(coordinate) < extra)};
}

int owner(size_t index, size_t n, int parts) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    const size_t boundary = (base + 1) * extra;
    return index < boundary ? static_cast<int>(index / (base + 1))
                            : static_cast<int>(extra + (index - boundary) / base);
}

// Generate the original rand_r stream once, in its original order. Stream
// bounded row batches to the blocks, without allocating a global matrix.
void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n,
                        const int dims[2], const int coords[2], MPI_Comm grid) {
    int rank;
    MPI_Comm_rank(grid, &rank);
    const Partition rows = partition(n, dims[0], coords[0]);
    const Partition cols = partition(n, dims[1], coords[1]);
    const size_t batchSize = std::min<size_t>(64, std::max<size_t>(1, INT_MAX / std::max<size_t>(n, 1)));
    if (rank == 0) {
        unsigned int seed = 42;
        std::vector<unsigned int> batch(std::min(batchSize, n) * n);
        for (int r = 0; r < dims[0]; ++r) {
            const Partition targetRows = partition(n, dims[0], r);
            for (size_t first = 0; first < targetRows.size; first += batchSize) {
                const size_t count = std::min(batchSize, targetRows.size - first);
                for (size_t i = 0; i < count; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        batch[i * n + j] = 1 + static_cast<unsigned int>(
                            static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                    }
                    batch[i * n + targetRows.begin + first + i] = 0;
                }
                for (int c = 0; c < dims[1]; ++c) {
                    const Partition targetCols = partition(n, dims[1], c);
                    if (targetCols.size == 0) continue;
                    int targetCoords[2] = {r, c};
                    int target;
                    MPI_Cart_rank(grid, targetCoords, &target);
                    if (target == 0) {
                        for (size_t i = 0; i < count; ++i) {
                            std::copy_n(batch.data() + i * n + targetCols.begin,
                                        targetCols.size, dist.data() + (first + i) * cols.size);
                        }
                    } else {
                        MPI_Datatype block;
                        MPI_Type_vector(static_cast<int>(count), static_cast<int>(targetCols.size),
                                        static_cast<int>(n), MPI_UNSIGNED, &block);
                        MPI_Type_commit(&block);
                        MPI_Send(batch.data() + targetCols.begin, 1, block, target, 0, grid);
                        MPI_Type_free(&block);
                    }
                }
            }
        }
    } else if (cols.size != 0) {
        for (size_t first = 0; first < rows.size; first += batchSize) {
            const size_t count = std::min(batchSize, rows.size - first);
            MPI_Recv(dist.data() + first * cols.size, static_cast<int>(count * cols.size),
                     MPI_UNSIGNED, 0, 0, grid, MPI_STATUS_IGNORE);
        }
    }
    for (size_t i = 0; i < rows.size; ++i) {
        for (size_t j = 0; j < cols.size; ++j) {
            path[i * cols.size + j] = static_cast<unsigned int>(rows.begin + i);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   const int dims[2], const int coords[2],
                   MPI_Comm rowComm, MPI_Comm colComm) {
    const Partition rows = partition(n, dims[0], coords[0]);
    const Partition cols = partition(n, dims[1], coords[1]);
    std::vector<unsigned int> pivotRow(cols.size);
    std::vector<unsigned int> pivotCol(rows.size);
    for (size_t k = 0; k < n; ++k) {
        const int rowOwner = owner(k, n, dims[0]);
        const int colOwner = owner(k, n, dims[1]);
        if (coords[0] == rowOwner && cols.size != 0) {
            std::copy_n(dist.data() + (k - rows.begin) * cols.size,
                        cols.size, pivotRow.data());
        }
        if (coords[1] == colOwner) {
            for (size_t i = 0; i < rows.size; ++i) {
                pivotCol[i] = dist[i * cols.size + k - cols.begin];
            }
        }
        // Independent row/column collectives can progress concurrently.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(cols.size), MPI_UNSIGNED,
                   rowOwner, colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), static_cast<int>(rows.size), MPI_UNSIGNED,
                   colOwner, rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        if (cols.size == 0) continue;
        for (size_t i = 0; i < rows.size; ++i) {
            const unsigned int dik = pivotCol[i];
            if (dik == 0) continue; // The pivot row cannot improve itself.
            unsigned int* distanceRow = dist.data() + i * cols.size;
            unsigned int* pathRow = path.data() + i * cols.size;
            // Contiguous inner loops let the compiler vectorize the updates.
            for (size_t j = 0; j < cols.size; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                if (candidate < distanceRow[j]) {
                    distanceRow[j] = candidate;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

// Gathering is needed only for the original validation and hash output.
std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& dist,
                                         size_t n, const int dims[2],
                                         const int coords[2], MPI_Comm grid) {
    int rank, ranks;
    MPI_Comm_rank(grid, &rank);
    MPI_Comm_size(grid, &ranks);
    const Partition rows = partition(n, dims[0], coords[0]);
    const Partition cols = partition(n, dims[1], coords[1]);
    std::vector<unsigned int> result;
    if (rank == 0) {
        result.resize(n * n);
        for (int source = 0; source < ranks; ++source) {
            int sourceCoords[2];
            MPI_Cart_coords(grid, source, 2, sourceCoords);
            const Partition sourceRows = partition(n, dims[0], sourceCoords[0]);
            const Partition sourceCols = partition(n, dims[1], sourceCoords[1]);
            if (sourceCols.size == 0) continue;
            const size_t batchSize = std::max<size_t>(1, INT_MAX / sourceCols.size);
            for (size_t first = 0; first < sourceRows.size; first += batchSize) {
                const size_t count = std::min(batchSize, sourceRows.size - first);
                unsigned int* destination = result.data() + (sourceRows.begin + first) * n + sourceCols.begin;
                if (source == 0) {
                    for (size_t i = 0; i < count; ++i) {
                        std::copy_n(dist.data() + (first + i) * cols.size,
                                    cols.size, destination + i * n);
                    }
                } else {
                    MPI_Datatype block;
                    MPI_Type_vector(static_cast<int>(count), static_cast<int>(sourceCols.size),
                                    static_cast<int>(n), MPI_UNSIGNED, &block);
                    MPI_Type_commit(&block);
                    MPI_Recv(destination, 1, block, source, 1, grid, MPI_STATUS_IGNORE);
                    MPI_Type_free(&block);
                }
            }
        }
    } else if (cols.size != 0) {
        const size_t batchSize = std::max<size_t>(1, INT_MAX / cols.size);
        for (size_t first = 0; first < rows.size; first += batchSize) {
            const size_t count = std::min(batchSize, rows.size - first);
            MPI_Send(dist.data() + first * cols.size, static_cast<int>(count * cols.size),
                     MPI_UNSIGNED, 0, 1, grid);
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int status = 0;
    bool stop = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end != '\0' || errno != 0 ||
                value > INT_MAX || value > std::numeric_limits<size_t>::max() / sizeof(unsigned int) /
                                                  std::max<unsigned long long>(value, 1)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                status = 1;
                stop = true;
                break;
            }
            numNodes = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            stop = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            stop = true;
            break;
        }
    }
    if (stop) {
        MPI_Finalize();
        return status;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const int periods[2] = {0, 0};
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    // Split keys make communicator ranks equal to their grid coordinates.
    MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
    MPI_Comm_split(grid, coords[1], coords[0], &colComm);
    const Partition rows = partition(numNodes, dims[0], coords[0]);
    const Partition cols = partition(numNodes, dims[1], coords[1]);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    try {
        std::vector<unsigned int> dist(rows.size * cols.size);
        std::vector<unsigned int> path(rows.size * cols.size);
        initializeMatrices(dist, path, numNodes, dims, coords, grid);
        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(grid);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, dims, coords, rowComm, colComm);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0.0);
        }
        if (printResults || validate) {
            const auto result = gatherDistances(dist, numNodes, dims, coords, grid);
            if (rank == 0) {
                if (printResults) print_results_int(result, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(result, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    status = valid ? 0 : 1;
                }
            }
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, grid);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return status;
}
