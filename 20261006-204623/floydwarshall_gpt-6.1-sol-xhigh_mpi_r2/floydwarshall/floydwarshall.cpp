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

struct Partition {
    size_t begin;
    size_t size;
};

Partition partition(size_t n, int parts, int coordinate) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    return {coordinate * base + std::min(extra, static_cast<size_t>(coordinate)),
            base + (static_cast<size_t>(coordinate) < extra)};
}

int owner(size_t k, size_t n, int parts) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    const size_t boundary = (base + 1) * extra;
    return static_cast<int>(k < boundary ? k / (base + 1)
                                         : extra + (k - boundary) / base);
}

struct ProcessGrid {
    MPI_Comm comm;
    MPI_Comm rowComm;
    MPI_Comm colComm;
    int rank;
    int dims[2] = {0, 0};
    int coords[2];
    Partition rows;
    Partition cols;

    explicit ProcessGrid(size_t n) {
        int processes;
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dims);
        const int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 2, coords);
        MPI_Comm_split(comm, coords[0], coords[1], &rowComm);
        MPI_Comm_split(comm, coords[1], coords[0], &colComm);
        rows = partition(n, dims[0], coords[0]);
        cols = partition(n, dims[1], coords[1]);
    }

    void free() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
        MPI_Comm_free(&comm);
    }
};

#if defined(__GLIBC__)
// glibc rand_r advances this 32-bit LCG three times per result. Exponentiate
// its affine transformation to jump directly to any position in the ORIGINAL
// random stream; changing the seed per rank would change the benchmark graph.
struct RandomJump {
    unsigned int multiplier = 1;
    unsigned int increment = 0;

    explicit RandomJump(size_t samples) {
        unsigned int a = 1;
        unsigned int c = 0;
        for (int i = 0; i < 3; ++i) {
            a *= 1103515245u;
            c = c * 1103515245u + 12345u;
        }
        while (samples != 0) {
            if (samples & 1) {
                increment = increment * a + c;
                multiplier *= a;
            }
            c *= a + 1;
            a *= a;
            samples >>= 1;
        }
    }

    unsigned int operator()(unsigned int seed) const {
        return multiplier * seed + increment;
    }
};
#endif

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n,
                        const ProcessGrid& grid) {
    const size_t rows = grid.rows.size;
    const size_t cols = grid.cols.size;
    const double range = static_cast<double>(MAX_DISTANCE);
#if defined(__GLIBC__)
    if (rows != 0 && cols != 0) {
        unsigned int seed = RandomJump(grid.rows.begin * n + grid.cols.begin)(42);
        const RandomJump nextRow(n - cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                dist[i * cols + j] = 1 + static_cast<unsigned int>(
                    range * rand_r(&seed) / static_cast<double>(RAND_MAX));
            }
            seed = nextRow(seed);
        }
    }
#else
    // Other libc implementations need not use the same RNG recurrence.
    // Stream one original row at a time without allocating a global matrix.
    if (grid.rank == 0) {
        unsigned int seed = 42;
        std::vector<unsigned int> row(n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                row[j] = 1 + static_cast<unsigned int>(
                    range * rand_r(&seed) / static_cast<double>(RAND_MAX));
            }
            for (int c = 0; c < grid.dims[1]; ++c) {
                const Partition columns = partition(n, grid.dims[1], c);
                if (columns.size == 0) continue;
                int coordinates[2] = {owner(i, n, grid.dims[0]), c};
                int target;
                MPI_Cart_rank(grid.comm, coordinates, &target);
                if (target == 0) {
                    std::copy_n(row.data() + columns.begin, columns.size,
                                dist.data() + (i - grid.rows.begin) * cols);
                } else {
                    MPI_Send(row.data() + columns.begin, static_cast<int>(columns.size),
                             MPI_UNSIGNED, target, 0, grid.comm);
                }
            }
        }
    } else if (cols != 0) {
        for (size_t i = 0; i < rows; ++i) {
            MPI_Recv(dist.data() + i * cols, static_cast<int>(cols), MPI_UNSIGNED,
                     0, 0, grid.comm, MPI_STATUS_IGNORE);
        }
    }
#endif
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // The original path initializer stores the source vertex.
            path[i * cols + j] = static_cast<unsigned int>(grid.rows.begin + i);
            if (grid.rows.begin + i == grid.cols.begin + j) dist[i * cols + j] = 0;
        }
    }
}

// Independent, contiguous updates permit SIMD without changing strict tie
// handling or the original meaning of path[i,j] (last improving pivot).
inline void updateRow(unsigned int* __restrict__ dist,
                      unsigned int* __restrict__ path,
                      const unsigned int* __restrict__ pivot,
                      unsigned int via, size_t count, unsigned int k) {
    for (size_t j = 0; j < count; ++j) {
        const unsigned int candidate = via + pivot[j];
        const unsigned int previous = dist[j];
        if (candidate < previous) {
            dist[j] = candidate;
            path[j] = k;
        }
    }
}

void updateRectangle(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                     const std::vector<unsigned int>& rowPanel,
                     const std::vector<unsigned int>& colPanel,
                     size_t rows, size_t cols, size_t width, size_t firstPivot,
                     size_t rowBegin, size_t rowEnd, size_t colBegin, size_t colEnd) {
    // Keep distance/path tiles in cache while replaying a whole pivot batch.
    constexpr size_t tileRows = 32;
    constexpr size_t tileCols = 256;
    for (size_t ib = rowBegin; ib < rowEnd; ib += tileRows) {
        const size_t ie = std::min(ib + tileRows, rowEnd);
        for (size_t jb = colBegin; jb < colEnd; jb += tileCols) {
            const size_t count = std::min(tileCols, colEnd - jb);
            for (size_t t = 0; t < width; ++t) {
                const unsigned int* pivot = rowPanel.data() + t * cols + jb;
                for (size_t i = ib; i < ie; ++i) {
                    updateRow(dist.data() + i * cols + jb,
                              path.data() + i * cols + jb, pivot,
                              colPanel[t * rows + i], count,
                              static_cast<unsigned int>(firstPivot + t));
                }
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, const ProcessGrid& grid) {
    const size_t rows = grid.rows.size;
    const size_t cols = grid.cols.size;
    // Bound MPI counts even when a panel would otherwise exceed INT_MAX.
    const int smallerDimension = std::min(grid.dims[0], grid.dims[1]);
    const size_t maxEdge = std::max<size_t>(1, (n + smallerDimension - 1) / smallerDimension);
    const size_t batch = std::min<size_t>(64, INT_MAX / maxEdge);
    std::vector<unsigned int> diagonalRows(batch * batch);
    std::vector<unsigned int> diagonalCols(batch * batch);
    std::vector<unsigned int> rowPanel(batch * cols);
    std::vector<unsigned int> colPanel(batch * rows);

    for (size_t first = 0; first < n;) {
        const int rowOwner = owner(first, n, grid.dims[0]);
        const int colOwner = owner(first, n, grid.dims[1]);
        const Partition pivotRows = partition(n, grid.dims[0], rowOwner);
        const Partition pivotCols = partition(n, grid.dims[1], colOwner);
        const size_t width = std::min({batch, pivotRows.begin + pivotRows.size - first,
                                      pivotCols.begin + pivotCols.size - first});
        const bool ownsRows = grid.coords[0] == rowOwner;
        const bool ownsCols = grid.coords[1] == colOwner;
        const size_t ri = ownsRows ? first - grid.rows.begin : 0;
        const size_t cj = ownsCols ? first - grid.cols.begin : 0;

        // 1. Solve the diagonal tile, retaining its row/column at EACH k.
        // Using only its final values would change strict-improvement paths.
        if (ownsRows && ownsCols) {
            for (size_t t = 0; t < width; ++t) {
                std::copy_n(dist.data() + (ri + t) * cols + cj, width,
                            diagonalRows.data() + t * width);
                for (size_t i = 0; i < width; ++i) {
                    diagonalCols[t * width + i] = dist[(ri + i) * cols + cj + t];
                }
                for (size_t i = 0; i < width; ++i) {
                    updateRow(dist.data() + (ri + i) * cols + cj,
                              path.data() + (ri + i) * cols + cj,
                              diagonalRows.data() + t * width,
                              diagonalCols[t * width + i], width,
                              static_cast<unsigned int>(first + t));
                }
            }
        }
        MPI_Request requests[2];
        int requestCount = 0;
        if (ownsRows) {
            MPI_Ibcast(diagonalCols.data(), static_cast<int>(width * width), MPI_UNSIGNED,
                       colOwner, grid.rowComm, &requests[requestCount++]);
        }
        if (ownsCols) {
            MPI_Ibcast(diagonalRows.data(), static_cast<int>(width * width), MPI_UNSIGNED,
                       rowOwner, grid.colComm, &requests[requestCount++]);
        }
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);

        // 2. Solve the pivot row/column panels, saving the chronological
        // pivot values needed by all the independent remaining rectangles.
        if (ownsRows && cols != 0) {
            for (size_t t = 0; t < width; ++t) {
                std::copy_n(dist.data() + (ri + t) * cols, cols,
                            rowPanel.data() + t * cols);
                for (size_t i = 0; i < width; ++i) {
                    unsigned int* d = dist.data() + (ri + i) * cols;
                    unsigned int* p = path.data() + (ri + i) * cols;
                    const unsigned int via = diagonalCols[t * width + i];
                    if (ownsCols) {
                        updateRow(d, p, rowPanel.data() + t * cols, via, cj,
                                  static_cast<unsigned int>(first + t));
                        const size_t after = cj + width;
                        updateRow(d + after, p + after, rowPanel.data() + t * cols + after,
                                  via, cols - after, static_cast<unsigned int>(first + t));
                    } else {
                        updateRow(d, p, rowPanel.data() + t * cols, via, cols,
                                  static_cast<unsigned int>(first + t));
                    }
                }
            }
        }
        if (ownsCols) {
            for (size_t t = 0; t < width; ++t) {
                for (size_t i = 0; i < rows; ++i) {
                    colPanel[t * rows + i] = dist[i * cols + cj + t];
                }
                for (size_t i = 0; i < rows; ++i) {
                    if (ownsRows && i >= ri && i < ri + width) continue;
                    updateRow(dist.data() + i * cols + cj, path.data() + i * cols + cj,
                              diagonalRows.data() + t * width, colPanel[t * rows + i],
                              width, static_cast<unsigned int>(first + t));
                }
            }
        }
        MPI_Ibcast(rowPanel.data(), static_cast<int>(width * cols), MPI_UNSIGNED,
                   rowOwner, grid.colComm, &requests[0]);
        MPI_Ibcast(colPanel.data(), static_cast<int>(width * rows), MPI_UNSIGNED,
                   colOwner, grid.rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        // 3. Replay exactly the original increasing-k updates locally.
        const size_t rowCuts[4] = {0, ownsRows ? ri : rows,
                                  ownsRows ? ri + width : rows, rows};
        const size_t colCuts[4] = {0, ownsCols ? cj : cols,
                                  ownsCols ? cj + width : cols, cols};
        for (int r = 0; r < 4; r += 2) {
            for (int c = 0; c < 4; c += 2) {
                updateRectangle(dist, path, rowPanel, colPanel, rows, cols, width, first,
                                rowCuts[r], rowCuts[r + 1], colCuts[c], colCuts[c + 1]);
            }
        }
        first += width;
    }
}

// Only materialize the global matrix when the requested output needs it.
std::vector<unsigned int> gatherMatrix(const std::vector<unsigned int>& local,
                                      size_t n, const ProcessGrid& grid) {
    std::vector<unsigned int> result;
    if (grid.rank == 0) {
        result.resize(n * n);
        int processes;
        MPI_Comm_size(grid.comm, &processes);
        for (int rank = 0; rank < processes; ++rank) {
            int coordinates[2];
            MPI_Cart_coords(grid.comm, rank, 2, coordinates);
            const Partition rows = partition(n, grid.dims[0], coordinates[0]);
            const Partition cols = partition(n, grid.dims[1], coordinates[1]);
            if (rows.size == 0 || cols.size == 0) continue;
            unsigned int* destination = result.data() + rows.begin * n + cols.begin;
            if (rank == 0) {
                for (size_t i = 0; i < rows.size; ++i) {
                    std::copy_n(local.data() + i * cols.size, cols.size, destination + i * n);
                }
            } else {
                MPI_Datatype block;
                MPI_Type_vector(static_cast<int>(rows.size), static_cast<int>(cols.size),
                                static_cast<int>(n), MPI_UNSIGNED, &block);
                MPI_Type_commit(&block);
                MPI_Recv(destination, 1, block, rank, 1, grid.comm, MPI_STATUS_IGNORE);
                MPI_Type_free(&block);
            }
        }
    } else if (!local.empty()) {
        MPI_Datatype row;
        MPI_Type_contiguous(static_cast<int>(grid.cols.size), MPI_UNSIGNED, &row);
        MPI_Type_commit(&row);
        MPI_Send(local.data(), static_cast<int>(grid.rows.size), row, 0, 1, grid.comm);
        MPI_Type_free(&row);
    }
    return result;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n, const ProcessGrid& grid) {
    unsigned long long firstBadDiagonal = ULLONG_MAX;
    const size_t begin = std::max(grid.rows.begin, grid.cols.begin);
    const size_t end = std::min(grid.rows.begin + grid.rows.size,
                                grid.cols.begin + grid.cols.size);
    for (size_t i = begin; i < end; ++i) {
        if (dist[(i - grid.rows.begin) * grid.cols.size + i - grid.cols.begin] != 0) {
            firstBadDiagonal = i;
            break;
        }
    }
    unsigned long long badDiagonal = ULLONG_MAX;
    MPI_Reduce(&firstBadDiagonal, &badDiagonal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
               0, grid.comm);

    // The original check needs only the first ten rows and columns. Assemble
    // these O(n) samples rather than gathering an O(n^2) matrix for validation.
    const size_t samples = std::min<size_t>(n, 10);
    std::vector<unsigned int> localRows(samples * n, 0);
    std::vector<unsigned int> localCols(n * samples, 0);
    const size_t sampledCols = grid.cols.begin < samples
        ? std::min(grid.cols.size, samples - grid.cols.begin) : 0;
    for (size_t i = 0; i < grid.rows.size; ++i) {
        const size_t globalRow = grid.rows.begin + i;
        if (globalRow < samples && grid.cols.size != 0) {
            std::copy_n(dist.data() + i * grid.cols.size, grid.cols.size,
                        localRows.data() + globalRow * n + grid.cols.begin);
        }
        if (sampledCols != 0) {
            std::copy_n(dist.data() + i * grid.cols.size, sampledCols,
                        localCols.data() + globalRow * samples + grid.cols.begin);
        }
    }
    std::vector<unsigned int> sampleRows(grid.rank == 0 ? samples * n : 0);
    std::vector<unsigned int> sampleCols(grid.rank == 0 ? n * samples : 0);
    for (size_t offset = 0; offset < samples * n;) {
        const int count = static_cast<int>(std::min<size_t>(INT_MAX, samples * n - offset));
        MPI_Reduce(localRows.data() + offset,
                   grid.rank == 0 ? sampleRows.data() + offset : nullptr,
                   count, MPI_UNSIGNED, MPI_SUM, 0, grid.comm);
        MPI_Reduce(localCols.data() + offset,
                   grid.rank == 0 ? sampleCols.data() + offset : nullptr,
                   count, MPI_UNSIGNED, MPI_SUM, 0, grid.comm);
        offset += count;
    }
    if (grid.rank != 0) return true;
    if (badDiagonal != ULLONG_MAX) {
        printf("Validation failed: diagonal element [%llu,%llu] is not zero\n",
               badDiagonal, badDiagonal);
        return false;
    }
    for (size_t i = 0; i < samples; ++i) {
        for (size_t j = 0; j < samples; ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int distIJ = sampleRows[i * n + j];
                const unsigned int distIK = sampleRows[i * n + k];
                const unsigned int distKJ = sampleCols[k * samples + j];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                           i, j, k);
                    return false;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end != '\0' || errno == ERANGE ||
                parsed > INT_MAX ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() / parsed /
                                         sizeof(unsigned int))) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
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

    int status = 0;
    try {
        ProcessGrid grid(numNodes);
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        std::vector<unsigned int> dist(grid.rows.size * grid.cols.size);
        std::vector<unsigned int> path(dist.size());
        initializeMatrices(dist, path, numNodes, grid);

        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(grid.comm);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, grid);
        const double localTime = MPI_Wtime() - start;
        double elapsed = 0;
        MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", elapsed > 0 ? ops / elapsed / 1e9 : 0.0);
        }

        if (printResults) {
            const std::vector<unsigned int> result = gatherMatrix(dist, numNodes, grid);
            if (rank == 0) print_results_int(result, "DistanceMatrix");
        }
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes, grid);
            if (rank == 0) {
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, grid.comm);
        // Communicators must be freed before MPI_Finalize; avoid collective
        // destructors during unwinding if an allocation fails on just one rank.
        grid.free();
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
