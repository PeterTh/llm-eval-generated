#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t PANEL_SIZE = 64;
constexpr size_t TILE_ROWS = 16;
constexpr size_t TILE_COLS = 512;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

size_t blockSize(size_t n, int parts, int coordinate) {
    return n / parts + (static_cast<size_t>(coordinate) < n % parts);
}

size_t blockBegin(size_t n, int parts, int coordinate) {
    return (n / parts) * coordinate + std::min(n % parts, static_cast<size_t>(coordinate));
}

int blockOwner(size_t index, size_t n, int parts) {
    const size_t small = n / parts;
    const size_t largeEnd = (small + 1) * (n % parts);
    if (index < largeEnd) {
        return static_cast<int>(index / (small + 1));
    }
    return static_cast<int>(n % parts + (index - largeEnd) / small);
}

struct ProcessGrid {
    int rank, processes;
    int dimensions[2] = {0, 0};
    int row, col;
    size_t rows, cols, rowBegin, colBegin;
    MPI_Comm rowComm, colComm;

    explicit ProcessGrid(size_t n) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dimensions);
        row = rank / dimensions[1];
        col = rank % dimensions[1];
        rows = blockSize(n, dimensions[0], row);
        cols = blockSize(n, dimensions[1], col);
        rowBegin = blockBegin(n, dimensions[0], row);
        colBegin = blockBegin(n, dimensions[1], col);
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }

    ~ProcessGrid() {
        // On a rank-local exception, reach the main MPI_Abort handler before
        // attempting collective cleanup that other ranks might not enter.
        if (std::uncaught_exceptions() != 0) {
            return;
        }
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
};

#if defined(__GLIBC__)
// glibc rand_r advances its 32-bit LCG three times per draw. Compose the
// affine transform by squaring to jump to an arbitrary position exactly.
// The generated values still come from rand_r, including its rounding.
struct RandomJump {
    unsigned int multiplier = 1;
    unsigned int increment = 0;

    explicit RandomJump(size_t draws) {
        unsigned int a = 1, c = 0;
        for (int step = 0; step < 3; ++step) {
            a *= 1103515245U;
            c = c * 1103515245U + 12345U;
        }
        while (draws != 0) {
            if (draws & 1) {
                increment = a * increment + c;
                multiplier *= a;
            }
            c = a * c + c;
            a *= a;
            draws >>= 1;
        }
    }

    unsigned int operator()(unsigned int seed) const {
        return multiplier * seed + increment;
    }
};
#else
// rand_r's sequence is implementation-defined. Preserve the host libc's
// sequence on other platforms without assuming glibc's state transition.
struct RandomJump {
    size_t draws;
    explicit RandomJump(size_t count) : draws(count) {}
    unsigned int operator()(unsigned int seed) const {
        for (size_t i = 0; i < draws; ++i) {
            rand_r(&seed);
        }
        return seed;
    }
};
#endif

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path,
                        size_t n, const ProcessGrid& grid) {
    if (grid.rows == 0 || grid.cols == 0) {
        return;
    }
    unsigned int seed = RandomJump(grid.rowBegin * n + grid.colBegin)(42);
    const RandomJump nextRow(n - grid.cols);
    const double range = static_cast<double>(MAX_DISTANCE);
    for (size_t i = 0; i < grid.rows; ++i) {
        for (size_t j = 0; j < grid.cols; ++j) {
            const size_t index = i * grid.cols + j;
            dist[index] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
            if (grid.rowBegin + i == grid.colBegin + j) {
                dist[index] = 0;
            }
            // Both assignments in the original path initialization set the
            // entry to its source row, rather than its destination column.
            path[index] = static_cast<unsigned int>(grid.rowBegin + i);
        }
        seed = nextRow(seed);
    }
}

// Separate, nonaliasing arrays let the compiler vectorize the conditional
// distance and path writes together. Keep strict '<' for identical ties.
inline void relax(unsigned int* __restrict__ dist,
                  unsigned int* __restrict__ path,
                  const unsigned int* __restrict__ pivot,
                  size_t count, unsigned int via, unsigned int k) {
    if (via == 0) {
        return;
    }
    for (size_t j = 0; j < count; ++j) {
        const unsigned int candidate = via + pivot[j];
        if (candidate < dist[j]) {
            dist[j] = candidate;
            path[j] = k;
        }
    }
}

// Chunking also supports panels whose element count exceeds MPI's int count.
void broadcastPanel(unsigned int* data, size_t count, int root, MPI_Comm comm,
                    std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset,
                                                   static_cast<size_t>(INT_MAX)));
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Ibcast(data + offset, chunk, MPI_UNSIGNED, root, comm, &requests.back());
        offset += chunk;
    }
}

void updateTrailing(std::vector<unsigned int>& dist,
                    std::vector<unsigned int>& path,
                    const std::vector<unsigned int>& rowPanel,
                    const std::vector<unsigned int>& colPanel,
                    size_t rows, size_t cols, size_t begin, size_t width,
                    size_t rowFirst, size_t rowLast, size_t colFirst, size_t colLast) {
    // A tile remains in cache through all pivots. Each cell sees precisely
    // the same increasing k sequence as in the original scalar algorithm.
    for (size_t ib = rowFirst; ib < rowLast; ib += TILE_ROWS) {
        const size_t ie = std::min(ib + TILE_ROWS, rowLast);
        for (size_t jb = colFirst; jb < colLast; jb += TILE_COLS) {
            const size_t je = std::min(jb + TILE_COLS, colLast);
            for (size_t t = 0; t < width; ++t) {
                const unsigned int* pivot = rowPanel.data() + t * cols + jb;
                const unsigned int* column = colPanel.data() + t * rows;
                for (size_t i = ib; i < ie; ++i) {
                    const size_t offset = i * cols + jb;
                    relax(dist.data() + offset, path.data() + offset, pivot,
                          je - jb, column[i], static_cast<unsigned int>(begin + t));
                }
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   size_t n, const ProcessGrid& grid) {
    std::vector<unsigned int> rowPanel(PANEL_SIZE * grid.cols);
    std::vector<unsigned int> colPanel(PANEL_SIZE * grid.rows);
    std::vector<unsigned int> diagonalRows(PANEL_SIZE * PANEL_SIZE);
    std::vector<unsigned int> diagonalCols(PANEL_SIZE * PANEL_SIZE);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * PANEL_SIZE);

    for (size_t begin = 0; begin < n;) {
        const int ownerRow = blockOwner(begin, n, grid.dimensions[0]);
        const int ownerCol = blockOwner(begin, n, grid.dimensions[1]);
        const size_t ownerRowBegin = blockBegin(n, grid.dimensions[0], ownerRow);
        const size_t ownerColBegin = blockBegin(n, grid.dimensions[1], ownerCol);
        // Keep the entire diagonal panel on a single rank, including when
        // the row/column partitions differ or do not divide n evenly.
        const size_t width = std::min({PANEL_SIZE, n - begin,
            ownerRowBegin + blockSize(n, grid.dimensions[0], ownerRow) - begin,
            ownerColBegin + blockSize(n, grid.dimensions[1], ownerCol) - begin});
        const bool ownsRows = grid.row == ownerRow;
        const bool ownsCols = grid.col == ownerCol;
        const size_t localRow = ownsRows ? begin - grid.rowBegin : 0;
        const size_t localCol = ownsCols ? begin - grid.colBegin : 0;

        // First solve the diagonal panel. Record the row and column BEFORE
        // each pivot. Using only the final panel values would change strict
        // improvement decisions and therefore the original path matrix.
        if (ownsRows && ownsCols) {
            for (size_t t = 0; t < width; ++t) {
                const unsigned int* pivot = dist.data() + (localRow + t) * grid.cols + localCol;
                std::copy_n(pivot, width, diagonalRows.data() + t * width);
                for (size_t i = 0; i < width; ++i) {
                    diagonalCols[t * width + i] = dist[(localRow + i) * grid.cols + localCol + t];
                }
                for (size_t i = 0; i < width; ++i) {
                    const size_t offset = (localRow + i) * grid.cols + localCol;
                    relax(dist.data() + offset, path.data() + offset,
                          diagonalRows.data() + t * width, width,
                          diagonalCols[t * width + i], static_cast<unsigned int>(begin + t));
                }
            }
        }

        requests.clear();
        if (ownsCols) {
            broadcastPanel(diagonalRows.data(), width * width, ownerRow, grid.colComm, requests);
        }
        if (ownsRows) {
            broadcastPanel(diagonalCols.data(), width * width, ownerCol, grid.rowComm, requests);
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        // Solve the pivot row panel and capture every outgoing pivot row.
        // The diagonal snapshots supply dependencies on earlier pivots.
        if (ownsRows && grid.cols != 0) {
            for (size_t t = 0; t < width; ++t) {
                unsigned int* pivot = rowPanel.data() + t * grid.cols;
                std::copy_n(dist.data() + (localRow + t) * grid.cols, grid.cols, pivot);
                if (ownsCols) {
                    std::copy_n(diagonalRows.data() + t * width, width, pivot + localCol);
                }
                for (size_t i = 0; i < width; ++i) {
                    const size_t offset = (localRow + i) * grid.cols;
                    const unsigned int via = diagonalCols[t * width + i];
                    const unsigned int k = static_cast<unsigned int>(begin + t);
                    if (ownsCols) {
                        relax(dist.data() + offset, path.data() + offset, pivot, localCol, via, k);
                        const size_t after = localCol + width;
                        relax(dist.data() + offset + after, path.data() + offset + after,
                              pivot + after, grid.cols - after, via, k);
                    } else {
                        relax(dist.data() + offset, path.data() + offset, pivot, grid.cols, via, k);
                    }
                }
            }
        }

        // Solve the pivot column panel independently of the row panel.
        if (ownsCols && grid.rows != 0) {
            for (size_t t = 0; t < width; ++t) {
                unsigned int* column = colPanel.data() + t * grid.rows;
                for (size_t i = 0; i < grid.rows; ++i) {
                    column[i] = dist[i * grid.cols + localCol + t];
                }
                if (ownsRows) {
                    std::copy_n(diagonalCols.data() + t * width, width, column + localRow);
                }
                for (size_t i = 0; i < grid.rows; ++i) {
                    if (ownsRows && i >= localRow && i < localRow + width) {
                        continue;
                    }
                    const size_t offset = i * grid.cols + localCol;
                    relax(dist.data() + offset, path.data() + offset,
                          diagonalRows.data() + t * width, width, column[i],
                          static_cast<unsigned int>(begin + t));
                }
            }
        }

        // Two full panel broadcasts per batch, on orthogonal communicators.
        requests.clear();
        broadcastPanel(rowPanel.data(), width * grid.cols, ownerRow, grid.colComm, requests);
        broadcastPanel(colPanel.data(), width * grid.rows, ownerCol, grid.rowComm, requests);
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        // The pivot panels are finished already. Update only their complement.
        const size_t rowStarts[2] = {0, ownsRows ? localRow + width : grid.rows};
        const size_t rowEnds[2] = {ownsRows ? localRow : grid.rows, grid.rows};
        const size_t colStarts[2] = {0, ownsCols ? localCol + width : grid.cols};
        const size_t colEnds[2] = {ownsCols ? localCol : grid.cols, grid.cols};
        for (int ri = 0; ri < 2; ++ri) {
            for (int ci = 0; ci < 2; ++ci) {
                updateTrailing(dist, path, rowPanel, colPanel, grid.rows, grid.cols,
                               begin, width, rowStarts[ri], rowEnds[ri], colStarts[ci], colEnds[ci]);
            }
        }
        begin += width;
    }
}

// Gather only for -r, directly into the original row-major layout. MPI
// subarray types avoid packing buffers and the int limit on n*n elements.
std::vector<unsigned int> gatherDistanceMatrix(const std::vector<unsigned int>& local,
                                              size_t n, const ProcessGrid& grid) {
    std::vector<unsigned int> global;
    if (grid.rank == 0) {
        global.resize(n * n);
        for (size_t i = 0; i < grid.rows; ++i) {
            if (grid.cols != 0) {
                std::copy_n(local.data() + i * grid.cols, grid.cols,
                            global.data() + (grid.rowBegin + i) * n + grid.colBegin);
            }
        }
        for (int rank = 1; rank < grid.processes; ++rank) {
            const int row = rank / grid.dimensions[1];
            const int col = rank % grid.dimensions[1];
            const int sizes[2] = {static_cast<int>(n), static_cast<int>(n)};
            const int subsizes[2] = {static_cast<int>(blockSize(n, grid.dimensions[0], row)),
                                     static_cast<int>(blockSize(n, grid.dimensions[1], col))};
            if (subsizes[0] == 0 || subsizes[1] == 0) {
                continue;
            }
            const int starts[2] = {static_cast<int>(blockBegin(n, grid.dimensions[0], row)),
                                  static_cast<int>(blockBegin(n, grid.dimensions[1], col))};
            MPI_Datatype block;
            MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_UNSIGNED, &block);
            MPI_Type_commit(&block);
            MPI_Recv(global.data(), 1, block, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
    } else if (!local.empty()) {
        MPI_Datatype row, block;
        MPI_Type_contiguous(static_cast<int>(grid.cols), MPI_UNSIGNED, &row);
        MPI_Type_contiguous(static_cast<int>(grid.rows), row, &block);
        MPI_Type_commit(&block);
        MPI_Send(local.data(), 1, block, 0, 0, MPI_COMM_WORLD);
        MPI_Type_free(&block);
        MPI_Type_free(&row);
    }
    return global;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n, const ProcessGrid& grid) {
    int localValid = 1, valid = 1;
    for (size_t i = 0; i < grid.rows; ++i) {
        const size_t globalRow = grid.rowBegin + i;
        if (globalRow >= grid.colBegin && globalRow - grid.colBegin < grid.cols &&
            dist[i * grid.cols + globalRow - grid.colBegin] != 0) {
            localValid = 0;
        }
    }
    MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!valid) {
        if (grid.rank == 0) {
            printf("Validation failed: a diagonal element is not zero\n");
        }
        return false;
    }

    // Preserve the original triangle checks with only O(10*n) workspace.
    // Entries have a unique owner; missing entries are zero for MPI_MAX.
    const size_t sample = std::min(n, static_cast<size_t>(10));
    std::vector<unsigned int> sampled(2 * sample * n, 0);
    for (size_t i = 0; i < grid.rows; ++i) {
        const size_t globalRow = grid.rowBegin + i;
        if (globalRow < sample && grid.cols != 0) {
            std::copy_n(dist.data() + i * grid.cols, grid.cols,
                        sampled.data() + globalRow * n + grid.colBegin);
        }
        for (size_t j = 0; j < grid.cols && grid.colBegin + j < sample; ++j) {
            sampled[sample * n + (grid.colBegin + j) * n + globalRow] = dist[i * grid.cols + j];
        }
    }
    for (size_t offset = 0; offset < sampled.size();) {
        const int count = static_cast<int>(std::min(sampled.size() - offset,
                                                   static_cast<size_t>(INT_MAX)));
        if (grid.rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, sampled.data() + offset, count, MPI_UNSIGNED,
                       MPI_MAX, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(sampled.data() + offset, sampled.data() + offset, count, MPI_UNSIGNED,
                       MPI_MAX, 0, MPI_COMM_WORLD);
        }
        offset += count;
    }
    if (grid.rank == 0) {
        for (size_t i = 0; i < sample && valid; ++i) {
            for (size_t j = 0; j < sample && valid; ++j) {
                const unsigned int distIJ = sampled[i * n + j];
                for (size_t k = 0; k < n; ++k) {
                    const unsigned int distIK = sampled[i * n + k];
                    const unsigned int distKJ = sampled[sample * n + j * n + k];
                    if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                        valid = 0;
                        break;
                    }
                }
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank) {
    // Root parses once and shares the configuration, including early exits.
    uint64_t numNodes = 512;
    int options[3] = {0, 0, -1}; // validation, results, early-exit status
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* value = argv[++i];
                char* end = nullptr;
                errno = 0;
                numNodes = strtoull(value, &end, 10);
                if (errno != 0 || *value == '-' || end == value || *end != '\0' ||
                    numNodes > static_cast<uint64_t>(INT_MAX) ||
                    (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes / sizeof(unsigned int))) {
                    fprintf(stderr, "Invalid number of nodes: %s\n", value);
                    options[2] = 1;
                    break;
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                options[0] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                options[2] = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                options[2] = 1;
                break;
            }
        }
    }
    MPI_Bcast(options, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (options[2] != -1) {
        return options[2];
    }
    MPI_Bcast(&numNodes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    const size_t n = static_cast<size_t>(numNodes);
    ProcessGrid grid(n);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", n);
        printf("Validation: %s\n", options[0] ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d grid)\n", grid.processes, grid.dimensions[0], grid.dimensions[1]);
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(grid.rows * grid.cols);
    std::vector<unsigned int> path(grid.rows * grid.cols);
    initializeMatrices(dist, path, n, grid);
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, n, grid);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration * 1000.0);
        const double ops = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0.0);
    }
    if (options[1]) {
        const auto global = gatherDistanceMatrix(dist, n, grid);
        if (rank == 0) {
            print_results_int(global, "DistanceMatrix");
        }
    }
    if (options[0]) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(dist, n, grid);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
