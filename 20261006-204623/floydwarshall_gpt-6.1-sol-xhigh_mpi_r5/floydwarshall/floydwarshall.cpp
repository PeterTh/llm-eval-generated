#include <algorithm>
#include <cerrno>
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

struct Block {
    size_t begin;
    size_t count;
};

Block blockFor(const size_t n, const int coordinate, const int parts) {
    const size_t width = n / parts;
    const size_t remainder = n % parts;
    const size_t c = static_cast<size_t>(coordinate);
    return {c * width + std::min(c, remainder), width + (c < remainder)};
}

int ownerOf(const size_t k, const size_t n, const int parts) {
    const size_t width = n / parts;
    const size_t remainder = n % parts;
    const size_t boundary = (width + 1) * remainder;
    // When width is zero, every valid k is in the first branch.
    return static_cast<int>(k < boundary ? k / (width + 1)
                                        : remainder + (k - boundary) / width);
}

struct ProcessGrid {
    int dimensions[2] = {0, 0};
    int row;
    int column;
    Block rows;
    Block columns;
    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm columnComm = MPI_COMM_NULL;

    ProcessGrid(const size_t n, const int rank, const int processes) {
        MPI_Dims_create(processes, 2, dimensions);
        row = rank / dimensions[1];
        column = rank % dimensions[1];
        rows = blockFor(n, row, dimensions[0]);
        columns = blockFor(n, column, dimensions[1]);
        // Communicator ranks coincide with the corresponding grid coordinate.
        MPI_Comm_split(MPI_COMM_WORLD, row, column, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, column, row, &columnComm);
    }

    ~ProcessGrid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&columnComm);
    }
};

MPI_Datatype matrixBlockType(const size_t rows, const size_t columns,
                            const size_t stride) {
    MPI_Datatype type;
    // Using one derived datatype also supports blocks exceeding INT_MAX elements.
    MPI_Type_vector(static_cast<int>(rows), static_cast<int>(columns),
                    static_cast<int>(stride), MPI_UNSIGNED, &type);
    MPI_Type_commit(&type);
    return type;
}

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, const size_t n,
                        const ProcessGrid& grid, const int rank,
                        const MPI_Datatype localType) {
    for (size_t i = 0; i < grid.rows.count && grid.columns.count != 0; ++i) {
        // The original path matrix starts with the source row index.
        std::fill_n(path.data() + i * grid.columns.count, grid.columns.count,
                    static_cast<unsigned int>(grid.rows.begin + i));
    }

    if (rank != 0) {
        if (!dist.empty()) {
            MPI_Recv(dist.data(), 1, localType, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        return;
    }

    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    std::vector<unsigned int> strip;
    // Generate the original rand_r stream in order, staging only one row strip.
    // No process retains a replicated global matrix during the computation.
    for (int r = 0; r < grid.dimensions[0]; ++r) {
        const Block rows = blockFor(n, r, grid.dimensions[0]);
        if (rows.count == 0) {
            continue;
        }
        strip.resize(rows.count * n);
        for (auto& value : strip) {
            value = 1 + static_cast<unsigned int>(range * rand_r(&seed) /
                                                  static_cast<double>(RAND_MAX));
        }
        for (size_t i = 0; i < rows.count; ++i) {
            strip[i * n + rows.begin + i] = 0;
        }
        for (int c = 0; c < grid.dimensions[1]; ++c) {
            const Block columns = blockFor(n, c, grid.dimensions[1]);
            if (columns.count == 0) {
                continue;
            }
            const int destination = r * grid.dimensions[1] + c;
            if (destination == 0) {
                for (size_t i = 0; i < rows.count; ++i) {
                    std::copy_n(strip.data() + i * n + columns.begin,
                                columns.count, dist.data() + i * columns.count);
                }
            } else {
                MPI_Datatype type = matrixBlockType(rows.count, columns.count, n);
                MPI_Send(strip.data() + columns.begin, 1, type, destination, 0,
                         MPI_COMM_WORLD);
                MPI_Type_free(&type);
            }
        }
    }
}

// Separate pivot buffers and non-aliasing rows allow SIMD masked updates.
inline void updateRow(unsigned int* __restrict__ dist,
                      unsigned int* __restrict__ path,
                      const unsigned int* __restrict__ pivotRow,
                      const size_t columns, const unsigned int distIK,
                      const unsigned int k) {
    if (distIK == 0) {
        return; // The zero-diagonal pivot row cannot improve itself.
    }
    for (size_t j = 0; j < columns; ++j) {
        const unsigned int candidate = distIK + pivotRow[j];
        if (candidate < dist[j]) {
            dist[j] = candidate;
            path[j] = k;
        }
    }
}

void broadcastPivot(const std::vector<unsigned int>& dist, const size_t n,
                    const size_t k, const ProcessGrid& grid,
                    std::vector<unsigned int>& pivotRow,
                    std::vector<unsigned int>& pivotColumn,
                    MPI_Request* requests) {
    const int rowOwner = ownerOf(k, n, grid.dimensions[0]);
    const int columnOwner = ownerOf(k, n, grid.dimensions[1]);
    if (grid.row == rowOwner && grid.columns.count != 0) {
        std::copy_n(dist.data() + (k - grid.rows.begin) * grid.columns.count,
                    grid.columns.count, pivotRow.data());
    }
    if (grid.column == columnOwner) {
        for (size_t i = 0; i < grid.rows.count; ++i) {
            pivotColumn[i] = dist[i * grid.columns.count + k - grid.columns.begin];
        }
    }
    MPI_Ibcast(pivotRow.data(), static_cast<int>(grid.columns.count), MPI_UNSIGNED,
               rowOwner, grid.columnComm, &requests[0]);
    MPI_Ibcast(pivotColumn.data(), static_cast<int>(grid.rows.count), MPI_UNSIGNED,
               columnOwner, grid.rowComm, &requests[1]);
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, const size_t n,
                   const ProcessGrid& grid) {
    if (n == 0) {
        return;
    }
    std::vector<unsigned int> pivotRows[2] = {
        std::vector<unsigned int>(grid.columns.count),
        std::vector<unsigned int>(grid.columns.count)};
    std::vector<unsigned int> pivotColumns[2] = {
        std::vector<unsigned int>(grid.rows.count),
        std::vector<unsigned int>(grid.rows.count)};
    MPI_Request requests[2][2];
    broadcastPivot(dist, n, 0, grid, pivotRows[0], pivotColumns[0], requests[0]);

    for (size_t k = 0; k < n; ++k) {
        const size_t slot = k % 2;
        MPI_Waitall(2, requests[slot], MPI_STATUSES_IGNORE);
        const unsigned int* pivotRow = pivotRows[slot].data();
        const unsigned int* pivotColumn = pivotColumns[slot].data();

        if (k + 1 < n) {
            // Update the next pivot's row and column first, then communicate them
            // while the rest of this iteration is computed. All k remain ordered.
            const size_t next = k + 1;
            if (next >= grid.rows.begin && next - grid.rows.begin < grid.rows.count
                && grid.columns.count != 0) {
                const size_t i = next - grid.rows.begin;
                updateRow(dist.data() + i * grid.columns.count,
                          path.data() + i * grid.columns.count, pivotRow,
                          grid.columns.count, pivotColumn[i],
                          static_cast<unsigned int>(k));
            }
            if (next >= grid.columns.begin &&
                next - grid.columns.begin < grid.columns.count) {
                const size_t j = next - grid.columns.begin;
                for (size_t i = 0; i < grid.rows.count; ++i) {
                    const size_t offset = i * grid.columns.count + j;
                    const unsigned int candidate = pivotColumn[i] + pivotRow[j];
                    if (candidate < dist[offset]) {
                        dist[offset] = candidate;
                        path[offset] = static_cast<unsigned int>(k);
                    }
                }
            }
            const size_t nextSlot = 1 - slot;
            broadcastPivot(dist, n, next, grid, pivotRows[nextSlot],
                           pivotColumns[nextSlot], requests[nextSlot]);
        }

        if (grid.columns.count != 0) {
            for (size_t i = 0; i < grid.rows.count; ++i) {
                // Repeating the early row/column updates is idempotent: pivot
                // distances are unchanged with these nonnegative, bounded weights.
                updateRow(dist.data() + i * grid.columns.count,
                          path.data() + i * grid.columns.count, pivotRow,
                          grid.columns.count, pivotColumn[i],
                          static_cast<unsigned int>(k));
            }
        }
    }
}

std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& dist,
                                         const size_t n, const ProcessGrid& grid,
                                         const int rank, const MPI_Datatype localType) {
    std::vector<unsigned int> result;
    if (rank != 0) {
        if (!dist.empty()) {
            MPI_Send(dist.data(), 1, localType, 0, 1, MPI_COMM_WORLD);
        }
        return result;
    }

    result.resize(n * n);
    for (int r = 0; r < grid.dimensions[0]; ++r) {
        const Block rows = blockFor(n, r, grid.dimensions[0]);
        for (int c = 0; c < grid.dimensions[1]; ++c) {
            const Block columns = blockFor(n, c, grid.dimensions[1]);
            if (rows.count == 0 || columns.count == 0) {
                continue;
            }
            const int source = r * grid.dimensions[1] + c;
            unsigned int* target = result.data() + rows.begin * n + columns.begin;
            if (source == 0) {
                for (size_t i = 0; i < rows.count; ++i) {
                    std::copy_n(dist.data() + i * columns.count, columns.count,
                                target + i * n);
                }
            } else {
                MPI_Datatype type = matrixBlockType(rows.count, columns.count, n);
                MPI_Recv(target, 1, type, source, 1, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Type_free(&type);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    // Only rank zero parses and prints; all ranks follow the same control flow.
    // Options: node count, validation, result output, and early-exit status.
    unsigned long long options[4] = {512, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* argument = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long n = strtoull(argument, &end, 10);
                if (argument[0] == '-' || end == argument || *end != '\0' ||
                    errno == ERANGE || n > static_cast<unsigned long long>(
                        std::numeric_limits<int>::max()) ||
                    (n != 0 && n > std::numeric_limits<size_t>::max() /
                                      sizeof(unsigned int) / n)) {
                    printf("Invalid number of nodes: %s\n", argument);
                    options[3] = 2;
                    break;
                }
                options[0] = n;
            } else if (strcmp(argv[i], "-v") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[2] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                options[3] = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                options[3] = 2;
                break;
            }
        }
    }
    MPI_Bcast(options, 4, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (options[3] != 0) {
        MPI_Finalize();
        return options[3] == 1 ? 0 : 1;
    }

    const size_t numNodes = static_cast<size_t>(options[0]);
    const bool validate = options[1] != 0;
    const bool printResults = options[2] != 0;
    int resultStatus = 0;
    try {
        ProcessGrid grid(numNodes, rank, processes);
        const size_t elements = grid.rows.count * grid.columns.count;
        std::vector<unsigned int> dist(elements);
        std::vector<unsigned int> path(elements);
        MPI_Datatype localType = MPI_DATATYPE_NULL;
        if (elements != 0) {
            localType = matrixBlockType(grid.rows.count, grid.columns.count,
                                        grid.columns.count);
        }

        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        initializeMatrices(dist, path, numNodes, grid, rank, localType);

        if (rank == 0) {
            printf("Computing shortest paths...\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, grid);
        const double localSeconds = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %lld ms\n",
                    static_cast<long long>(seconds * 1000.0));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
        }

        if (printResults || validate) {
            const std::vector<unsigned int> result =
                gatherDistances(dist, numNodes, grid, rank, localType);
            if (rank == 0) {
                if (printResults) {
                    print_results_int(result, "DistanceMatrix");
                }
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(result, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    resultStatus = valid ? 0 : 1;
                }
            }
        }
        MPI_Bcast(&resultStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (localType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&localType);
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return resultStatus;
}
