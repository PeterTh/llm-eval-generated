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

// Contiguous, balanced partitions also support more ranks than nodes.
size_t blockStart(size_t n, int parts, int coordinate) {
    return (n / parts) * coordinate + std::min(n % parts, size_t(coordinate));
}

struct Grid {
    int rank, size, dims[2] = {0, 0}, row, col;
    size_t rowStart, colStart, rows, cols;
    MPI_Comm rowComm, colComm;

    explicit Grid(size_t n) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Dims_create(size, 2, dims);
        row = rank / dims[1];
        col = rank % dims[1];
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
        rowStart = blockStart(n, dims[0], row);
        colStart = blockStart(n, dims[1], col);
        rows = blockStart(n, dims[0], row + 1) - rowStart;
        cols = blockStart(n, dims[1], col + 1) - colStart;
    }

    ~Grid() {
        // Do not enter collective cleanup while peers are still computing if
        // a local allocation fails; terminate the entire MPI job instead.
        if (std::uncaught_exceptions()) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
};

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n, const Grid& g) {
    // Pass the rand_r state between process rows to reproduce the original
    // sequence exactly, without replicating the matrix or the random work.
    unsigned int seed = 42;
    std::vector<unsigned int> row(g.col == 0 ? n : 0);
    std::vector<int> counts(g.dims[1]), offsets(g.dims[1]);
    for (int c = 0; c < g.dims[1]; ++c) {
        offsets[c] = static_cast<int>(blockStart(n, g.dims[1], c));
        counts[c] = static_cast<int>(blockStart(n, g.dims[1], c + 1)) - offsets[c];
    }
    if (g.col == 0 && g.row > 0)
        MPI_Recv(&seed, 1, MPI_UNSIGNED, g.row - 1, 0, g.colComm, MPI_STATUS_IGNORE);
    for (size_t i = 0; i < g.rows; ++i) {
        if (g.col == 0) {
            for (size_t j = 0; j < n; ++j)
                row[j] = 1 + static_cast<unsigned int>(
                    double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
            row[g.rowStart + i] = 0;
        }
        MPI_Scatterv(row.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                     g.cols ? dist.data() + i * g.cols : nullptr,
                     static_cast<int>(g.cols), MPI_UNSIGNED, 0, g.rowComm);
        for (size_t j = 0; j < g.cols; ++j)
            path[i * g.cols + j] = static_cast<unsigned int>(g.rowStart + i);
    }
    if (g.col == 0 && g.row + 1 < g.dims[0])
        MPI_Send(&seed, 1, MPI_UNSIGNED, g.row + 1, 0, g.colComm);
}

void floydWarshall(std::vector<unsigned int>& dist,
                  std::vector<unsigned int>& path, size_t n, const Grid& g) {
    std::vector<unsigned int> pivotRow(g.cols), pivotCol(g.rows);
    int ownerRow = 0, ownerCol = 0;
    for (size_t k = 0; k < n; ++k) {
        while (k >= blockStart(n, g.dims[0], ownerRow + 1)) ++ownerRow;
        while (k >= blockStart(n, g.dims[1], ownerCol + 1)) ++ownerCol;
        if (g.row == ownerRow && g.cols)
            std::copy_n(dist.data() + (k - g.rowStart) * g.cols,
                        g.cols, pivotRow.data());
        if (g.col == ownerCol)
            for (size_t i = 0; i < g.rows; ++i)
                pivotCol[i] = dist[i * g.cols + k - g.colStart];

        // Independent broadcasts on the two grid axes can progress together.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(g.cols), MPI_UNSIGNED,
                   ownerRow, g.colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), static_cast<int>(g.rows), MPI_UNSIGNED,
                   ownerCol, g.rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        // Preserve increasing-k order and strict comparisons, including the
        // original path choices for ties. Nonnegative weights and zero diagonal
        // mean that the pivot row and column cannot change during this step.
        for (size_t i = 0; i < g.rows && g.cols; ++i) {
            const unsigned int dik = pivotCol[i];
            auto* d = dist.data() + i * g.cols;
            auto* p = path.data() + i * g.cols;
            for (size_t j = 0; j < g.cols; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                if (candidate < d[j]) {
                    d[j] = candidate;
                    p[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& dist,
                                          size_t n, const Grid& g) {
    std::vector<unsigned int> result;
    if (g.rank == 0) {
        result.resize(n * n);
        for (size_t i = 0; i < g.rows; ++i)
            std::copy_n(dist.data() + i * g.cols, g.cols, result.data() + i * n);
        for (int rank = 1; rank < g.size; ++rank) {
            const size_t r = blockStart(n, g.dims[0], rank / g.dims[1]);
            const size_t c = blockStart(n, g.dims[1], rank % g.dims[1]);
            const int rows = static_cast<int>(blockStart(n, g.dims[0], rank / g.dims[1] + 1) - r);
            const int cols = static_cast<int>(blockStart(n, g.dims[1], rank % g.dims[1] + 1) - c);
            if (!rows || !cols) continue;
            MPI_Datatype block;
            MPI_Type_vector(rows, cols, static_cast<int>(n), MPI_UNSIGNED, &block);
            MPI_Type_commit(&block);
            MPI_Recv(result.data() + r * n + c, 1, block, rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
    } else if (g.rows && g.cols) {
        // A row datatype avoids an MPI int count limit on the whole local tile.
        MPI_Datatype row;
        MPI_Type_contiguous(static_cast<int>(g.cols), MPI_UNSIGNED, &row);
        MPI_Type_commit(&row);
        MPI_Send(dist.data(), static_cast<int>(g.rows), row, 0, 1, MPI_COMM_WORLD);
        MPI_Type_free(&row);
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

int run(int argc, char** argv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || *value == '-' || end == value || *end || parsed > INT_MAX ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / parsed)) {
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    Grid grid(numNodes);
    std::vector<unsigned int> dist(grid.rows * grid.cols);
    std::vector<unsigned int> path(grid.rows * grid.cols);
    initializeMatrices(dist, path, numNodes, grid);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, grid);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        const double ops = double(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0);
    }

    int status = 0;
    if (printResults || validate) {
        auto result = gatherDistances(dist, numNodes, grid);
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
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
