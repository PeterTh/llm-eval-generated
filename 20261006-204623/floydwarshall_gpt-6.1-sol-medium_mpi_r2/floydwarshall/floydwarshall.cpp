#include <algorithm>
#include <climits>
#include <cerrno>
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

// Balanced contiguous blocks, including empty tiles when there are more ranks
// than nodes. Cartesian ranks are ordered by process row, then process column.
size_t blockStart(size_t n, int coordinate, int parts) {
    return n * static_cast<size_t>(coordinate) / static_cast<size_t>(parts);
}

struct Grid {
    MPI_Comm cart, rowComm, columnComm;
    int rank, dims[2] = {0, 0}, coords[2];
    size_t rowStart, columnStart, rows, columns;

    explicit Grid(size_t n) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Dims_create(size, 2, dims);
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
        MPI_Comm_rank(cart, &rank);
        MPI_Cart_coords(cart, rank, 2, coords);
        MPI_Comm_split(cart, coords[0], coords[1], &rowComm);
        MPI_Comm_split(cart, coords[1], coords[0], &columnComm);
        rowStart = blockStart(n, coords[0], dims[0]);
        columnStart = blockStart(n, coords[1], dims[1]);
        rows = blockStart(n, coords[0] + 1, dims[0]) - rowStart;
        columns = blockStart(n, coords[1] + 1, dims[1]) - columnStart;
    }

    ~Grid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&columnComm);
        MPI_Comm_free(&cart);
    }
};

// A tile datatype avoids MPI's int limit on the total number of elements.
MPI_Datatype tileType(size_t rows, size_t columns, size_t stride) {
    MPI_Datatype type;
    MPI_Type_vector(static_cast<int>(rows), static_cast<int>(columns),
                    static_cast<int>(stride), MPI_UNSIGNED, &type);
    MPI_Type_commit(&type);
    return type;
}

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n, const Grid& grid) {
    for (size_t i = 0; i < grid.rows; ++i)
        for (size_t j = 0; j < grid.columns; ++j)
            path[i * grid.columns + j] = static_cast<unsigned int>(grid.columnStart + j);

    MPI_Datatype localType = tileType(grid.rows, grid.columns, grid.columns);
    if (grid.rank == 0) {
        // Stream the original rand_r sequence once, using only one process-row
        // slab of temporary storage rather than a replicated full matrix.
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        std::vector<unsigned int> slab;
        for (int r = 0; r < grid.dims[0]; ++r) {
            const size_t begin = blockStart(n, r, grid.dims[0]);
            const size_t height = blockStart(n, r + 1, grid.dims[0]) - begin;
            slab.resize(height * n);
            for (auto& value : slab)
                value = 1 + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
            for (size_t i = 0; i < height; ++i)
                slab[i * n + begin + i] = 0;
            for (int c = 0; c < grid.dims[1]; ++c) {
                const size_t column = blockStart(n, c, grid.dims[1]);
                const size_t width = blockStart(n, c + 1, grid.dims[1]) - column;
                const int destination = r * grid.dims[1] + c;
                if (destination == 0) {
                    for (size_t i = 0; i < height && width != 0; ++i)
                        std::copy_n(slab.data() + i * n + column, width,
                                    dist.data() + i * width);
                } else {
                    MPI_Datatype type = tileType(height, width, n);
                    MPI_Send(height && width ? slab.data() + column : nullptr,
                             1, type, destination, 0, grid.cart);
                    MPI_Type_free(&type);
                }
            }
        }
    } else {
        MPI_Recv(dist.data(), 1, localType, 0, 0, grid.cart, MPI_STATUS_IGNORE);
    }
    MPI_Type_free(&localType);
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n, const Grid& grid) {
    std::vector<unsigned int> pivotRow(grid.columns), pivotColumn(grid.rows);
    for (size_t k = 0; k < n; ++k) {
        const int ownerRow = static_cast<int>(((k + 1) * grid.dims[0] - 1) / n);
        const int ownerColumn = static_cast<int>(((k + 1) * grid.dims[1] - 1) / n);
        if (grid.coords[0] == ownerRow && grid.columns != 0)
            std::copy_n(dist.data() + (k - grid.rowStart) * grid.columns,
                        grid.columns, pivotRow.data());
        if (grid.coords[1] == ownerColumn)
            for (size_t i = 0; i < grid.rows; ++i)
                pivotColumn[i] = dist[i * grid.columns + k - grid.columnStart];

        // Independent row/column collectives can progress concurrently. Their
        // completion supplies all dependencies; no per-step global barrier.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(grid.columns), MPI_UNSIGNED,
                   ownerRow, grid.columnComm, &requests[0]);
        MPI_Ibcast(pivotColumn.data(), static_cast<int>(grid.rows), MPI_UNSIGNED,
                   ownerColumn, grid.rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        // Keep k strictly ordered to preserve both distances and path tie
        // handling. Positive edge weights and a zero diagonal make snapshots
        // of the pivot row/column equivalent to the original in-place loop.
        for (size_t i = 0; i < grid.rows && grid.columns != 0; ++i) {
            const unsigned int dik = pivotColumn[i];
            if (dik == 0) continue; // The pivot row cannot improve itself.
            unsigned int* distanceRow = dist.data() + i * grid.columns;
            unsigned int* pathRow = path.data() + i * grid.columns;
            for (size_t j = 0; j < grid.columns; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                if (candidate < distanceRow[j]) {
                    distanceRow[j] = candidate;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& dist,
                                          size_t n, const Grid& grid) {
    std::vector<unsigned int> full;
    if (grid.rank == 0) {
        full.resize(n * n);
        for (int r = 0; r < grid.dims[0]; ++r) {
            const size_t row = blockStart(n, r, grid.dims[0]);
            const size_t height = blockStart(n, r + 1, grid.dims[0]) - row;
            for (int c = 0; c < grid.dims[1]; ++c) {
                const size_t column = blockStart(n, c, grid.dims[1]);
                const size_t width = blockStart(n, c + 1, grid.dims[1]) - column;
                const int source = r * grid.dims[1] + c;
                if (source == 0) {
                    for (size_t i = 0; i < height && width != 0; ++i)
                        std::copy_n(dist.data() + i * width, width,
                                    full.data() + (row + i) * n + column);
                } else {
                    MPI_Datatype type = tileType(height, width, n);
                    MPI_Recv(height && width ? full.data() + row * n + column : nullptr,
                             1, type, source, 1, grid.cart, MPI_STATUS_IGNORE);
                    MPI_Type_free(&type);
                }
            }
        }
    } else {
        MPI_Datatype type = tileType(grid.rows, grid.columns, grid.columns);
        MPI_Send(dist.data(), 1, type, 0, 1, grid.cart);
        MPI_Type_free(&type);
    }
    return full;
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

int runBenchmark(int argc, char** argv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(text, &end, 10);
            if (errno || text == end || *end || *text == '-' || value > INT_MAX ||
                (value && value > std::numeric_limits<size_t>::max() / value / sizeof(unsigned int))) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", text);
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

    Grid grid(numNodes);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(grid.rows * grid.columns);
    std::vector<unsigned int> path(grid.rows * grid.columns);
    initializeMatrices(dist, path, numNodes, grid);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(grid.cart);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, grid);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid.cart);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }

    int status = 0;
    if (printResults || validate) {
        auto full = gatherDistances(dist, numNodes, grid);
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
    MPI_Bcast(&status, 1, MPI_INT, 0, grid.cart);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = 1;
    try {
        status = runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
