#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Balanced contiguous partitions, including empty blocks when there are more
// process rows/columns than vertices.
size_t partitionSize(size_t n, int parts, int coordinate) {
    return n / parts + (static_cast<size_t>(coordinate) < n % parts);
}

size_t partitionBegin(size_t n, int parts, int coordinate) {
    return (n / parts) * coordinate + std::min(n % parts, static_cast<size_t>(coordinate));
}

int partitionOwner(size_t vertex, size_t n, int parts) {
    const size_t base = n / parts;
    const size_t remainder = n % parts;
    const size_t boundary = (base + 1) * remainder;
    // If base is zero, every valid vertex is in the first branch.
    return static_cast<int>(vertex < boundary ? vertex / (base + 1)
                                              : remainder + (vertex - boundary) / base);
}

struct ProcessGrid {
    MPI_Comm communicator, rowCommunicator, columnCommunicator;
    int rank, dimensions[2] = {0, 0}, coordinates[2];
    size_t rowBegin, columnBegin, rows, columns;

    explicit ProcessGrid(size_t n) {
        int processes;
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dimensions);
        const int periods[2] = {0, 0};
        // Preserve world rank zero as the sole output/initialization rank.
        MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &communicator);
        MPI_Comm_rank(communicator, &rank);
        MPI_Cart_coords(communicator, rank, 2, coordinates);
        MPI_Comm_split(communicator, coordinates[0], coordinates[1], &rowCommunicator);
        MPI_Comm_split(communicator, coordinates[1], coordinates[0], &columnCommunicator);
        rowBegin = partitionBegin(n, dimensions[0], coordinates[0]);
        columnBegin = partitionBegin(n, dimensions[1], coordinates[1]);
        rows = partitionSize(n, dimensions[0], coordinates[0]);
        columns = partitionSize(n, dimensions[1], coordinates[1]);
    }

    void release() {
        MPI_Comm_free(&rowCommunicator);
        MPI_Comm_free(&columnCommunicator);
        MPI_Comm_free(&communicator);
    }
};

// Bound temporary storage and keep each MPI message below the int count limit.
size_t transferRows(size_t n) {
    return n == 0 ? 1 : std::min(static_cast<size_t>(64), static_cast<size_t>(INT_MAX) / n);
}

MPI_Datatype matrixStrip(size_t rows, size_t columns, size_t stride) {
    MPI_Datatype datatype;
    MPI_Type_vector(static_cast<int>(rows), static_cast<int>(columns),
                    static_cast<int>(stride), MPI_UNSIGNED, &datatype);
    MPI_Type_commit(&datatype);
    return datatype;
}

void initializeMatrices(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                        size_t n, const ProcessGrid& grid) {
    for (size_t i = 0; i < grid.rows; ++i) {
        if (grid.columns != 0) {
            std::fill_n(path.data() + i * grid.columns, grid.columns,
                        static_cast<unsigned int>(grid.rowBegin + i));
        }
    }

    const size_t stripRows = transferRows(n);
    if (grid.rank != 0) {
        if (grid.columns != 0) {
            for (size_t i = 0; i < grid.rows; i += stripRows) {
                const size_t rows = std::min(stripRows, grid.rows - i);
                MPI_Recv(dist.data() + i * grid.columns, static_cast<int>(rows * grid.columns),
                         MPI_UNSIGNED, 0, 0, grid.communicator, MPI_STATUS_IGNORE);
            }
        }
        return;
    }

    // Stream the original rand_r sequence once, without replicating the matrix
    // or assuming a particular libc implementation of the generator.
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    std::vector<unsigned int> strip(std::min(stripRows, n) * n);
    for (int processRow = 0; processRow < grid.dimensions[0]; ++processRow) {
        const size_t begin = partitionBegin(n, grid.dimensions[0], processRow);
        const size_t count = partitionSize(n, grid.dimensions[0], processRow);
        for (size_t i = 0; i < count; i += stripRows) {
            const size_t rows = std::min(stripRows, count - i);
            for (size_t element = 0; element < rows * n; ++element) {
                strip[element] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
            }
            for (size_t row = 0; row < rows; ++row) {
                strip[row * n + begin + i + row] = 0;
            }
            for (int processColumn = 0; processColumn < grid.dimensions[1]; ++processColumn) {
                const size_t column = partitionBegin(n, grid.dimensions[1], processColumn);
                const size_t columns = partitionSize(n, grid.dimensions[1], processColumn);
                if (columns == 0) {
                    continue;
                }
                const int coordinates[2] = {processRow, processColumn};
                int destination;
                MPI_Cart_rank(grid.communicator, coordinates, &destination);
                if (destination == 0) {
                    for (size_t row = 0; row < rows; ++row) {
                        std::copy_n(strip.data() + row * n + column, columns,
                                    dist.data() + (i + row) * columns);
                    }
                } else {
                    MPI_Datatype datatype = matrixStrip(rows, columns, n);
                    MPI_Send(strip.data() + column, 1, datatype, destination, 0, grid.communicator);
                    MPI_Type_free(&datatype);
                }
            }
        }
    }
}

// These arrays do not alias. Keeping the destination loop contiguous lets the
// compiler vectorize the strict comparison and both conditional updates.
inline void updateRow(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                      const unsigned int* __restrict__ pivot, size_t columns,
                      unsigned int distanceToPivot, unsigned int k) {
    for (size_t j = 0; j < columns; ++j) {
        const unsigned int candidate = distanceToPivot + pivot[j];
        if (candidate < dist[j]) {
            dist[j] = candidate;
            path[j] = k;
        }
    }
}

struct Pivot {
    std::vector<unsigned int> row, column;
    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    explicit Pivot(const ProcessGrid& grid) : row(grid.columns), column(grid.rows) {}
};

void broadcastPivot(const std::vector<unsigned int>& dist, size_t n, size_t k,
                    const ProcessGrid& grid, Pivot& pivot) {
    const int processRow = partitionOwner(k, n, grid.dimensions[0]);
    const int processColumn = partitionOwner(k, n, grid.dimensions[1]);
    if (grid.coordinates[0] == processRow && grid.columns != 0) {
        std::copy_n(dist.data() + (k - grid.rowBegin) * grid.columns,
                    grid.columns, pivot.row.data());
    }
    if (grid.coordinates[1] == processColumn) {
        for (size_t i = 0; i < grid.rows; ++i) {
            pivot.column[i] = dist[i * grid.columns + k - grid.columnBegin];
        }
    }
    MPI_Ibcast(pivot.row.data(), static_cast<int>(grid.columns), MPI_UNSIGNED,
               processRow, grid.columnCommunicator, &pivot.requests[0]);
    MPI_Ibcast(pivot.column.data(), static_cast<int>(grid.rows), MPI_UNSIGNED,
               processColumn, grid.rowCommunicator, &pivot.requests[1]);
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, const ProcessGrid& grid) {
    if (n == 0) {
        return;
    }
    Pivot pivots[2] = {Pivot(grid), Pivot(grid)};
    broadcastPivot(dist, n, 0, grid, pivots[0]);
    for (size_t k = 0; k < n; ++k) {
        Pivot& pivot = pivots[k % 2];
        MPI_Waitall(2, pivot.requests, MPI_STATUSES_IGNORE);

        size_t nextRow = grid.rows;
        size_t nextColumn = grid.columns;
        if (k + 1 < n) {
            // Update the next pivot's row and column first. With nonnegative
            // weights and a zero diagonal, the current pivot is immutable.
            // Its snapshots therefore give exactly the original k/i/j order,
            // including the intermediate-vertex values on strict improvements.
            if (k + 1 >= grid.rowBegin && k + 1 - grid.rowBegin < grid.rows) {
                nextRow = k + 1 - grid.rowBegin;
                if (grid.columns != 0) {
                    updateRow(dist.data() + nextRow * grid.columns,
                              path.data() + nextRow * grid.columns, pivot.row.data(),
                              grid.columns, pivot.column[nextRow], static_cast<unsigned int>(k));
                }
            }
            if (k + 1 >= grid.columnBegin && k + 1 - grid.columnBegin < grid.columns) {
                nextColumn = k + 1 - grid.columnBegin;
                for (size_t i = 0; i < grid.rows; ++i) {
                    if (i == nextRow) {
                        continue;
                    }
                    const size_t element = i * grid.columns + nextColumn;
                    const unsigned int candidate = pivot.column[i] + pivot.row[nextColumn];
                    if (candidate < dist[element]) {
                        dist[element] = candidate;
                        path[element] = static_cast<unsigned int>(k);
                    }
                }
            }
            // Double buffering allows communication of k+1 to overlap the
            // remaining local work for k, without touching an MPI send buffer.
            broadcastPivot(dist, n, k + 1, grid, pivots[(k + 1) % 2]);
        }

        if (grid.columns == 0) {
            continue;
        }
        for (size_t i = 0; i < grid.rows; ++i) {
            if (i == nextRow || pivot.column[i] == 0) {
                continue;
            }
            unsigned int* row = dist.data() + i * grid.columns;
            unsigned int* pathRow = path.data() + i * grid.columns;
            updateRow(row, pathRow, pivot.row.data(), nextColumn,
                      pivot.column[i], static_cast<unsigned int>(k));
            if (nextColumn < grid.columns) {
                const size_t begin = nextColumn + 1;
                updateRow(row + begin, pathRow + begin, pivot.row.data() + begin,
                          grid.columns - begin, pivot.column[i], static_cast<unsigned int>(k));
            }
        }
    }
}

// Assemble in the original row-major order only for requested output/checks.
// Strips also support matrices whose total element count exceeds INT_MAX.
std::vector<unsigned int> gatherMatrix(const std::vector<unsigned int>& local,
                                      size_t n, const ProcessGrid& grid) {
    const size_t stripRows = transferRows(n);
    if (grid.rank != 0) {
        if (grid.columns != 0) {
            for (size_t i = 0; i < grid.rows; i += stripRows) {
                const size_t rows = std::min(stripRows, grid.rows - i);
                MPI_Send(local.data() + i * grid.columns, static_cast<int>(rows * grid.columns),
                         MPI_UNSIGNED, 0, 1, grid.communicator);
            }
        }
        return {};
    }

    std::vector<unsigned int> result(n * n);
    for (int source = 0; source < grid.dimensions[0] * grid.dimensions[1]; ++source) {
        int coordinates[2];
        MPI_Cart_coords(grid.communicator, source, 2, coordinates);
        const size_t rowBegin = partitionBegin(n, grid.dimensions[0], coordinates[0]);
        const size_t columnBegin = partitionBegin(n, grid.dimensions[1], coordinates[1]);
        const size_t rows = partitionSize(n, grid.dimensions[0], coordinates[0]);
        const size_t columns = partitionSize(n, grid.dimensions[1], coordinates[1]);
        if (columns == 0) {
            continue;
        }
        for (size_t i = 0; i < rows; i += stripRows) {
            const size_t count = std::min(stripRows, rows - i);
            unsigned int* destination = result.data() + (rowBegin + i) * n + columnBegin;
            if (source == 0) {
                for (size_t row = 0; row < count; ++row) {
                    std::copy_n(local.data() + (i + row) * columns, columns, destination + row * n);
                }
            } else {
                MPI_Datatype datatype = matrixStrip(count, columns, n);
                MPI_Recv(destination, 1, datatype, source, 1, grid.communicator, MPI_STATUS_IGNORE);
                MPI_Type_free(&datatype);
            }
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

int runBenchmark(size_t numNodes, bool validate, bool printResults) {
    ProcessGrid grid(numNodes);
    if (grid.rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> dist(grid.rows * grid.columns);
    std::vector<unsigned int> path(grid.rows * grid.columns);
    initializeMatrices(dist, path, numNodes, grid);

    if (grid.rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(grid.communicator);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, grid);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid.communicator);
    if (grid.rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }

    int status = 0;
    if (printResults || validate) {
        const std::vector<unsigned int> result = gatherMatrix(dist, numNodes, grid);
        if (grid.rank == 0) {
            if (printResults) {
                print_results_int(result, "DistanceMatrix");
            }
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(result, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, grid.communicator);
    grid.release();
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse once and share both options and early-exit status with every rank.
    // Status 2 denotes a successful help request; status 1 denotes an error.
    unsigned long long options[4] = {512, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* argument = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long value = std::strtoull(argument, &end, 10);
                if (argument[0] == '-' || end == argument || *end != '\0' || errno == ERANGE ||
                    value > static_cast<unsigned long long>(INT_MAX) ||
                    (value != 0 && value > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / value)) {
                    printf("Invalid number of nodes: %s\n", argument);
                    options[3] = 1;
                    break;
                }
                options[0] = value;
            } else if (strcmp(argv[i], "-v") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[2] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                options[3] = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                options[3] = 1;
                break;
            }
        }
    }
    MPI_Bcast(options, 4, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (options[3] != 0) {
        MPI_Finalize();
        return options[3] == 2 ? 0 : 1;
    }

    int status;
    try {
        status = runBenchmark(static_cast<size_t>(options[0]), options[1] != 0, options[2] != 0);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
