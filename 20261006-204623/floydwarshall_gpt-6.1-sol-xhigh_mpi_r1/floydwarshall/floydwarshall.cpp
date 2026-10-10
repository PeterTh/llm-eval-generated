#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t PANEL_SIZE = 64;
constexpr size_t TRANSFER_ROWS = 64;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous, balanced partitions, including empty partitions for tiny graphs.
size_t blockBegin(size_t n, int parts, int coordinate) {
    return (n / parts) * coordinate + std::min(n % parts, static_cast<size_t>(coordinate));
}

size_t blockSize(size_t n, int parts, int coordinate) {
    return n / parts + (static_cast<size_t>(coordinate) < n % parts);
}

int blockOwner(size_t index, size_t n, int parts) {
    const size_t base = n / parts;
    const size_t remainder = n % parts;
    const size_t largerEnd = remainder * (base + 1);
    if (index < largerEnd) {
        return static_cast<int>(index / (base + 1));
    }
    return static_cast<int>(remainder + (index - largerEnd) / base);
}

struct ProcessGrid {
    int dims[2] = {0, 0};
    int row, col;
    size_t rowBegin, colBegin, rows, cols;
    MPI_Comm rowComm, colComm;

    ProcessGrid(size_t n, int rank, int ranks) {
        MPI_Dims_create(ranks, 2, dims);
        row = rank / dims[1];
        col = rank % dims[1];
        rowBegin = blockBegin(n, dims[0], row);
        colBegin = blockBegin(n, dims[1], col);
        rows = blockSize(n, dims[0], row);
        cols = blockSize(n, dims[1], col);
        // Communicator ranks equal the corresponding grid coordinates.
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }
};

// A transfer slab never crosses a process-row boundary. Bound its total element
// count as well, so all MPI counts and displacements fit the MPI int interface.
size_t transferRows(size_t first, size_t n, const ProcessGrid& grid, int owner) {
    return std::min({TRANSFER_ROWS, static_cast<size_t>(INT_MAX) / n,
                     blockBegin(n, grid.dims[0], owner) +
                         blockSize(n, grid.dims[0], owner) - first});
}

void transferLayout(size_t n, size_t rows, int owner, const ProcessGrid& grid,
                    std::vector<int>& counts, std::vector<int>& displacements) {
    std::fill(counts.begin(), counts.end(), 0);
    std::fill(displacements.begin(), displacements.end(), 0);
    size_t offset = 0;
    for (int col = 0; col < grid.dims[1]; ++col) {
        const int rank = owner * grid.dims[1] + col;
        counts[rank] = static_cast<int>(rows * blockSize(n, grid.dims[1], col));
        displacements[rank] = static_cast<int>(offset);
        offset += static_cast<size_t>(counts[rank]);
    }
}

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n,
                        const ProcessGrid& grid, int rank, int ranks) {
    // Preserve the original source-node initialization of every path entry.
    for (size_t i = 0; i < grid.rows; ++i) {
        for (size_t j = 0; j < grid.cols; ++j) {
            path[i * grid.cols + j] = static_cast<unsigned int>(grid.rowBegin + i);
        }
    }

    // Stream the original libc rand_r sequence on rank zero. Sending bounded
    // slabs preserves every input bit without replicating the global matrix or
    // assuming a particular libc random-number implementation.
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    std::vector<unsigned int> packed;
    std::vector<int> counts(ranks), displacements(ranks);
    for (size_t first = 0; first < n;) {
        const int owner = blockOwner(first, n, grid.dims[0]);
        const size_t rows = transferRows(first, n, grid, owner);
        transferLayout(n, rows, owner, grid, counts, displacements);
        if (rank == 0) {
            packed.resize(rows * n);
            for (size_t i = 0; i < rows; ++i) {
                for (int col = 0; col < grid.dims[1]; ++col) {
                    const size_t colBegin = blockBegin(n, grid.dims[1], col);
                    const size_t cols = blockSize(n, grid.dims[1], col);
                    unsigned int* dst = packed.data() +
                        displacements[owner * grid.dims[1] + col] + i * cols;
                    for (size_t j = 0; j < cols; ++j) {
                        const unsigned int value = 1 +
                            static_cast<unsigned int>(range * rand_r(&seed) /
                                                      static_cast<double>(RAND_MAX));
                        dst[j] = first + i == colBegin + j ? 0 : value;
                    }
                }
            }
        }
        unsigned int* dst = counts[rank] != 0
            ? dist.data() + (first - grid.rowBegin) * grid.cols : nullptr;
        MPI_Scatterv(packed.data(), counts.data(), displacements.data(), MPI_UNSIGNED,
                     dst, counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        first += rows;
    }
}

void relaxDistances(unsigned int* __restrict dst,
                    const unsigned int* __restrict src,
                    unsigned int left, size_t count) {
    for (size_t j = 0; j < count; ++j) {
        dst[j] = std::min(dst[j], left + src[j]);
    }
}

void updateLocalMatrix(unsigned int* __restrict dist,
                       unsigned int* __restrict path,
                       const unsigned int* __restrict rowPanel,
                       const unsigned int* __restrict colPanel,
                       size_t rows, size_t cols, size_t band, size_t first) {
    // Cache tiles retain both distance and path entries across the entire band.
    // Every entry still observes pivots in exactly the original increasing order.
    constexpr size_t TILE_ROWS = 64;
    constexpr size_t TILE_COLS = 256;
    for (size_t rowStart = 0; rowStart < rows; rowStart += TILE_ROWS) {
        const size_t rowEnd = std::min(rows, rowStart + TILE_ROWS);
        for (size_t colStart = 0; colStart < cols; colStart += TILE_COLS) {
            const size_t colEnd = std::min(cols, colStart + TILE_COLS);
            for (size_t k = 0; k < band; ++k) {
                const unsigned int* pivotRow = rowPanel + k * cols;
                const unsigned int* pivotCol = colPanel + k * rows;
                const unsigned int pivot = static_cast<unsigned int>(first + k);
                for (size_t i = rowStart; i < rowEnd; ++i) {
                    unsigned int* distRow = dist + i * cols;
                    unsigned int* pathRow = path + i * cols;
                    const unsigned int left = pivotCol[i];
                    // Vectorizable conditional stores avoid touching the path
                    // matrix or writing distances when no entry improves.
                    for (size_t j = colStart; j < colEnd; ++j) {
                        const unsigned int previous = distRow[j];
                        const unsigned int candidate = left + pivotRow[j];
                        if (candidate < previous) {
                            pathRow[j] = pivot;
                            distRow[j] = candidate;
                        }
                    }
                }
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   const ProcessGrid& grid, int rank) {
    const size_t capacity = std::min(PANEL_SIZE, static_cast<size_t>(INT_MAX) /
        std::max({size_t(1), blockSize(n, grid.dims[0], 0),
                  blockSize(n, grid.dims[1], 0)}));
    // Nonempty backing storage also makes zero-count collectives safe for ranks
    // whose graph partition is empty.
    std::vector<unsigned int> rowPanel(std::max(size_t(1), capacity * grid.cols));
    std::vector<unsigned int> colPanel(std::max(size_t(1), capacity * grid.rows));
    std::vector<unsigned int> corner(capacity * capacity);

    for (size_t first = 0; first < n;) {
        const int rowOwner = blockOwner(first, n, grid.dims[0]);
        const int colOwner = blockOwner(first, n, grid.dims[1]);
        // A band belongs to one process row and one process column.
        const size_t band = std::min({
            capacity, blockBegin(n, grid.dims[0], rowOwner) +
                          blockSize(n, grid.dims[0], rowOwner) - first,
            blockBegin(n, grid.dims[1], colOwner) +
                blockSize(n, grid.dims[1], colOwner) - first});
        if (grid.row == rowOwner && grid.cols != 0) {
            for (size_t k = 0; k < band; ++k) {
                std::copy_n(dist.data() + (first - grid.rowBegin + k) * grid.cols,
                            grid.cols, rowPanel.data() + k * grid.cols);
            }
        }
        if (grid.col == colOwner && grid.rows != 0) {
            for (size_t k = 0; k < band; ++k) {
                for (size_t i = 0; i < grid.rows; ++i) {
                    colPanel[k * grid.rows + i] =
                        dist[i * grid.cols + first - grid.colBegin + k];
                }
            }
        }
        const int cornerOwner = rowOwner * grid.dims[1] + colOwner;
        if (rank == cornerOwner) {
            for (size_t k = 0; k < band; ++k) {
                std::copy_n(dist.data() + (first - grid.rowBegin + k) * grid.cols +
                                first - grid.colBegin,
                            band, corner.data() + k * band);
            }
        }

        // Exchange only pivot panels, not matrices. Independent broadcasts on
        // the two grid axes and the small corner can progress concurrently.
        MPI_Request requests[3];
        MPI_Ibcast(rowPanel.data(), static_cast<int>(band * grid.cols), MPI_UNSIGNED,
                   rowOwner, grid.colComm, &requests[0]);
        MPI_Ibcast(colPanel.data(), static_cast<int>(band * grid.rows), MPI_UNSIGNED,
                   colOwner, grid.rowComm, &requests[1]);
        MPI_Ibcast(corner.data(), static_cast<int>(band * band), MPI_UNSIGNED,
                   cornerOwner, MPI_COMM_WORLD, &requests[2]);
        MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);

        // Evolve only future panel pivots. Freeze each pivot row and column at
        // its original k-step state before moving to the next pivot. Because
        // weights are nonnegative and diagonals are zero, a pivot's own row and
        // column do not change during its step. These snapshots let every local
        // cache tile run all band steps without further communication, retaining
        // the original strict comparisons and path tie-breaking.
        for (size_t k = 0; k < band; ++k) {
            for (size_t next = k + 1; next < band; ++next) {
                relaxDistances(rowPanel.data() + next * grid.cols,
                               rowPanel.data() + k * grid.cols,
                               corner[next * band + k], grid.cols);
                relaxDistances(colPanel.data() + next * grid.rows,
                               colPanel.data() + k * grid.rows,
                               corner[k * band + next], grid.rows);
            }
            for (size_t i = k + 1; i < band; ++i) {
                relaxDistances(corner.data() + i * band + k + 1,
                               corner.data() + k * band + k + 1,
                               corner[i * band + k], band - k - 1);
            }
        }
        updateLocalMatrix(dist.data(), path.data(), rowPanel.data(), colPanel.data(),
                          grid.rows, grid.cols, band, first);
        first += band;
    }
}

// Reconstruct the row-major matrix only when reporting or validation needs it.
// Normal benchmark runs retain O(n^2 / ranks) matrix storage per rank.
std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& dist,
                                         size_t n, const ProcessGrid& grid,
                                         int rank, int ranks) {
    std::vector<unsigned int> result, packed;
    if (rank == 0) {
        result.resize(n * n);
    }
    std::vector<int> counts(ranks), displacements(ranks);
    for (size_t first = 0; first < n;) {
        const int owner = blockOwner(first, n, grid.dims[0]);
        const size_t rows = transferRows(first, n, grid, owner);
        transferLayout(n, rows, owner, grid, counts, displacements);
        if (rank == 0) {
            packed.resize(rows * n);
        }
        const unsigned int* src = counts[rank] != 0
            ? dist.data() + (first - grid.rowBegin) * grid.cols : nullptr;
        MPI_Gatherv(src, counts[rank], MPI_UNSIGNED, packed.data(), counts.data(),
                    displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int col = 0; col < grid.dims[1]; ++col) {
                const size_t colBegin = blockBegin(n, grid.dims[1], col);
                const size_t cols = blockSize(n, grid.dims[1], col);
                for (size_t i = 0; i < rows; ++i) {
                    std::copy_n(packed.data() +
                                    displacements[owner * grid.dims[1] + col] + i * cols,
                                cols, result.data() + (first + i) * n + colBegin);
                }
            }
        }
        first += rows;
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


int runBenchmark(int argc, char** argv, int rank, int ranks) {
    // Parse once and share the configuration so every rank follows the same
    // collective sequence, including help and error exits.
    unsigned long long options[3] = {512, 0, 0};
    int parseStatus = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* argument = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long value = strtoull(argument, &end, 10);
                if (argument[0] == '-' || end == argument || *end != '\0' ||
                    errno == ERANGE || value > static_cast<unsigned long long>(INT_MAX) ||
                    (value != 0 && value > std::vector<unsigned int>().max_size() / value)) {
                    fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                    parseStatus = 1;
                    break;
                }
                options[0] = value;
            } else if (strcmp(argv[i], "-v") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[2] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
    }
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        return parseStatus == 2 ? 0 : 1;
    }
    MPI_Bcast(options, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    const size_t numNodes = static_cast<size_t>(options[0]);
    const bool validate = options[1] != 0;
    const bool printResults = options[2] != 0;

    ProcessGrid grid(numNodes, rank, ranks);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d x %d process grid)\n", ranks, grid.dims[0], grid.dims[1]);
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(grid.rows * grid.cols);
    std::vector<unsigned int> path(grid.rows * grid.cols);
    initializeMatrices(dist, path, numNodes, grid, rank, ranks);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, grid, rank);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }

    int valid = 1;
    if (printResults || validate) {
        const std::vector<unsigned int> result =
            gatherDistances(dist, numNodes, grid, rank, ranks);
        if (rank == 0) {
            if (printResults) {
                print_results_int(result, "DistanceMatrix");
            }
            if (validate) {
                printf("Validating result...\n");
                valid = validateResult(result, numNodes) ? 1 : 0;
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Comm_free(&grid.rowComm);
    MPI_Comm_free(&grid.colComm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
