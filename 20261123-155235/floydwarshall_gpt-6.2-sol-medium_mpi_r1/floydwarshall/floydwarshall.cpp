#include <mpi.h>

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/hash.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t MAX_TILE_SIZE = 48;

struct ProcessGrid {
    int rows;
    int cols;
    int row;
    int col;
    MPI_Comm rowComm;
    MPI_Comm colComm;
};

struct Tile {
    size_t rows;
    size_t cols;
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;

    Tile(size_t height, size_t width)
        : rows(height), cols(width), dist(height * width), path(height * width) {}
};

size_t localTileCount(size_t count, int coord, int stride) {
    return (count + stride - 1 - coord) / stride;
}

struct LocalMatrix {
    size_t nodes;
    size_t tileSize;
    size_t tileCount;
    size_t tileRows;
    size_t tileCols;
    std::vector<Tile> tiles;

    LocalMatrix(size_t numNodes, const ProcessGrid& grid)
        : nodes(numNodes),
          tileSize(numNodes == 0 ? 1 : std::min({MAX_TILE_SIZE,
                      numNodes / std::max(grid.rows, grid.cols), INT_MAX / numNodes})),
          tileCount((nodes + tileSize - 1) / tileSize),
          tileRows(localTileCount(tileCount, grid.row, grid.rows)),
          tileCols(localTileCount(tileCount, grid.col, grid.cols)) {
        tiles.reserve(tileRows * tileCols);
        for (size_t i = grid.row; i < tileCount; i += grid.rows) {
            for (size_t j = grid.col; j < tileCount; j += grid.cols) {
                tiles.emplace_back(extent(i), extent(j));
            }
        }
    }

    size_t extent(size_t index) const {
        return std::min(tileSize, nodes - index * tileSize);
    }

    Tile& tile(size_t row, size_t col) {
        return tiles[row * tileCols + col];
    }

    const Tile& tile(size_t row, size_t col) const {
        return tiles[row * tileCols + col];
    }
};

void skipRandomNumbers(unsigned int& seed, size_t count) {
#if defined(__GLIBC__)
    // glibc's rand_r advances its unsigned LCG three times per sample.
    // Square the affine transformation to skip directly to the next local entry.
    constexpr unsigned int multiplier = 1103515245u;
    constexpr unsigned int increment = 12345u;
    unsigned int a = multiplier * multiplier * multiplier;
    unsigned int b = increment * (multiplier * multiplier + multiplier + 1u);
    while (count != 0) {
        if (count & 1) {
            seed = a * seed + b;
        }
        b = a * b + b;
        a *= a;
        count >>= 1;
    }
#else
    while (count != 0) {
        rand_r(&seed);
        --count;
    }
#endif
}

void initializeMatrices(LocalMatrix& matrix, const ProcessGrid& grid) {
    unsigned int seed = 42;
    size_t position = 0;
    for (size_t i = 0; i < matrix.tileRows; ++i) {
        const size_t firstRow = (i * grid.rows + grid.row) * matrix.tileSize;
        for (size_t row = 0; row < matrix.tile(i, 0).rows; ++row) {
            const size_t globalRow = firstRow + row;
            for (size_t j = 0; j < matrix.tileCols; ++j) {
                const size_t firstCol = (j * grid.cols + grid.col) * matrix.tileSize;
                const size_t next = globalRow * matrix.nodes + firstCol;
                skipRandomNumbers(seed, next - position);
                Tile& tile = matrix.tile(i, j);
                for (size_t col = 0; col < tile.cols; ++col) {
                    const size_t globalCol = firstCol + col;
                    const unsigned int weight = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                    tile.dist[row * tile.cols + col] = globalRow == globalCol ? 0 : weight;
                    tile.path[row * tile.cols + col] = globalRow;
                }
                position = next + tile.cols;
            }
        }
    }
}

inline void relaxRow(unsigned int* __restrict dist, unsigned int* __restrict path,
                     const unsigned int* __restrict right, size_t width,
                     unsigned int left, unsigned int pivot) {
    for (size_t j = 0; j < width; ++j) {
        const unsigned int old = dist[j];
        const unsigned int candidate = left + right[j];
        dist[j] = std::min(old, candidate);
        path[j] = candidate < old ? pivot : path[j];
    }
}

template <bool saveColumn = false, bool saveRow = false>
void relaxTile(Tile& tile, unsigned int* leftPanel, unsigned int* rightPanel,
               size_t firstPivot, size_t steps) {
    for (size_t k = 0; k < steps; ++k) {
        unsigned int* left = leftPanel + k * tile.rows;
        unsigned int* right = rightPanel + k * tile.cols;
        if constexpr (saveColumn) {
            for (size_t i = 0; i < tile.rows; ++i) {
                left[i] = tile.dist[i * tile.cols + k];
            }
        }
        if constexpr (saveRow) {
            std::copy_n(tile.dist.data() + k * tile.cols, tile.cols, right);
        }
        for (size_t i = 0; i < tile.rows; ++i) {
            relaxRow(tile.dist.data() + i * tile.cols, tile.path.data() + i * tile.cols,
                     right, tile.cols, left[i], firstPivot + k);
        }
    }
}

void floydWarshall(LocalMatrix& matrix, const ProcessGrid& grid) {
    // Panel snapshots use exact tile extents. Only the last local tile can be short.
    const size_t rowSpan = (matrix.tileRows - 1) * matrix.tileSize
        + matrix.tile(matrix.tileRows - 1, 0).rows;
    const size_t colSpan = (matrix.tileCols - 1) * matrix.tileSize
        + matrix.tile(0, matrix.tileCols - 1).cols;
    std::vector<unsigned int> left(matrix.tileSize * rowSpan);
    std::vector<unsigned int> right(matrix.tileSize * colSpan);

    for (size_t pivot = 0; pivot < matrix.tileCount; ++pivot) {
        const size_t steps = matrix.extent(pivot);
        const size_t firstPivot = pivot * matrix.tileSize;
        const size_t stride = steps * matrix.tileSize;
        const int ownerRow = pivot % grid.rows;
        const int ownerCol = pivot % grid.cols;
        const size_t localRow = pivot / grid.rows;
        const size_t localCol = pivot / grid.cols;
        const bool inPivotRow = grid.row == ownerRow;
        const bool inPivotCol = grid.col == ownerCol;

        // Record the operands at each k before relaxing that step. Keeping
        // those snapshots lets every tile reproduce the original pivot order.
        if (inPivotRow && inPivotCol) {
            relaxTile<true, true>(matrix.tile(localRow, localCol), left.data() + localRow * stride,
                                  right.data() + localCol * stride, firstPivot, steps);
        }

        MPI_Request transfers[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (inPivotRow && grid.cols > 1) {
            MPI_Ibcast(left.data() + localRow * stride, steps * steps, MPI_UNSIGNED,
                       ownerCol, grid.rowComm, &transfers[0]);
        }
        if (inPivotCol && grid.rows > 1) {
            MPI_Ibcast(right.data() + localCol * stride, steps * steps, MPI_UNSIGNED,
                       ownerRow, grid.colComm, &transfers[1]);
        }

        if (inPivotRow) {
            MPI_Wait(&transfers[0], MPI_STATUS_IGNORE);
            for (size_t j = 0; j < matrix.tileCols; ++j) {
                if (!inPivotCol || j != localCol) {
                    relaxTile<false, true>(matrix.tile(localRow, j), left.data() + localRow * stride,
                                           right.data() + j * stride, firstPivot, steps);
                }
            }
        }
        if (inPivotCol) {
            MPI_Wait(&transfers[1], MPI_STATUS_IGNORE);
            for (size_t i = 0; i < matrix.tileRows; ++i) {
                if (!inPivotRow || i != localRow) {
                    relaxTile<true, false>(matrix.tile(i, localCol), left.data() + i * stride,
                                           right.data() + localCol * stride, firstPivot, steps);
                }
            }
        }

        if (grid.cols > 1) {
            MPI_Ibcast(left.data(), steps * rowSpan, MPI_UNSIGNED, ownerCol,
                       grid.rowComm, &transfers[0]);
        }
        if (grid.rows > 1) {
            MPI_Ibcast(right.data(), steps * colSpan, MPI_UNSIGNED, ownerRow,
                       grid.colComm, &transfers[1]);
        }
        MPI_Waitall(2, transfers, MPI_STATUSES_IGNORE);

        for (size_t i = 0; i < matrix.tileRows; ++i) {
            if (inPivotRow && i == localRow) {
                continue;
            }
            for (size_t j = 0; j < matrix.tileCols; ++j) {
                if (!inPivotCol || j != localCol) {
                    relaxTile(matrix.tile(i, j), left.data() + i * stride,
                              right.data() + j * stride, firstPivot, steps);
                }
            }
        }
    }
}

void printResults(const LocalMatrix& matrix, const ProcessGrid& grid, MPI_Comm comm, int rank) {
    if (rank != 0) {
        for (const Tile& tile : matrix.tiles) {
            MPI_Send(tile.dist.data(), tile.dist.size(), MPI_UNSIGNED, 0, 0, comm);
        }
        return;
    }

    const size_t count = matrix.nodes * matrix.nodes;
    const size_t indices[] = {0, count / 4, count / 2, count - (count + 3) / 4,
                              count == 0 ? 0 : count - 1};
    unsigned int samples[5] = {};
    unsigned int minDistance = UINT_MAX;
    unsigned int maxDistance = 0;
    hash digest;

    // Stream one tile row at a time to preserve the row-major hash without
    // allocating the complete matrix on the reporting rank.
    std::vector<unsigned int> stripe(matrix.tileSize * matrix.nodes);
    std::vector<unsigned int> incoming(matrix.tileSize * matrix.tileSize);
    for (size_t i = 0; i < matrix.tileCount; ++i) {
        const size_t rows = matrix.extent(i);
        const int ownerRow = i % grid.rows;
        for (size_t j = 0; j < matrix.tileCount; ++j) {
            const size_t cols = matrix.extent(j);
            const int owner = ownerRow * grid.cols + j % grid.cols;
            const unsigned int* tileData;
            if (owner == 0) {
                tileData = matrix.tile(i / grid.rows, j / grid.cols).dist.data();
            } else {
                MPI_Recv(incoming.data(), rows * cols, MPI_UNSIGNED,
                         owner, 0, comm, MPI_STATUS_IGNORE);
                tileData = incoming.data();
            }
            for (size_t row = 0; row < rows; ++row) {
                std::copy_n(tileData + row * cols, cols,
                            stripe.data() + row * matrix.nodes + j * matrix.tileSize);
            }
        }
        const size_t offset = i * matrix.tileSize * matrix.nodes;
        const size_t length = rows * matrix.nodes;
        for (size_t k = 0; k < length; ++k) {
            const unsigned int value = stripe[k];
            digest.add(value);
            minDistance = std::min(minDistance, value);
            maxDistance = std::max(maxDistance, value);
        }
        for (size_t s = 0; s < 5; ++s) {
            if (indices[s] >= offset && indices[s] - offset < length) {
                samples[s] = stripe[indices[s] - offset];
            }
        }
    }

    printf("=== RESULTS ===\n");
    printf("Name: DistanceMatrix\n");
    printf("Elements: %zu\n", count);
    if (count != 0) {
        printf("Min: %u\n", minDistance);
        printf("Max: %u\n", maxDistance);
        for (size_t s = 0; s < 5; ++s) {
            printf("Sample[%zu]: %u\n", indices[s], samples[s]);
        }
        printf("Hash: %016" PRIx64 "\n", digest.get());
    }
    printf("=== END RESULTS ===\n");
}

bool validateResult(const LocalMatrix& matrix, const ProcessGrid& grid, MPI_Comm comm, int rank) {
    const size_t n = matrix.nodes;
    const size_t sample = std::min(n, size_t{10});
    // Exchange only the rows and columns needed by the triangle checks.
    std::vector<unsigned int> edges(2 * sample * n, 0);
    unsigned long long badDiagonal = n;
    for (size_t i = 0; i < matrix.tileRows; ++i) {
        const size_t tileRow = i * grid.rows + grid.row;
        const size_t rowStart = tileRow * matrix.tileSize;
        for (size_t j = 0; j < matrix.tileCols; ++j) {
            const size_t tileCol = j * grid.cols + grid.col;
            const size_t colStart = tileCol * matrix.tileSize;
            const Tile& tile = matrix.tile(i, j);
            if (tileRow == tileCol) {
                for (size_t d = 0; d < tile.rows; ++d) {
                    if (tile.dist[d * tile.cols + d] != 0) {
                        badDiagonal = std::min(badDiagonal,
                                               static_cast<unsigned long long>(rowStart + d));
                    }
                }
            }
            for (size_t row = 0; row < tile.rows && rowStart + row < sample; ++row) {
                std::copy_n(tile.dist.data() + row * tile.cols, tile.cols,
                            edges.data() + (rowStart + row) * n + colStart);
            }
            for (size_t col = 0; col < tile.cols && colStart + col < sample; ++col) {
                for (size_t row = 0; row < tile.rows; ++row) {
                    edges[(sample + colStart + col) * n + rowStart + row]
                        = tile.dist[row * tile.cols + col];
                }
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &badDiagonal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    if (badDiagonal != n) {
        if (rank == 0) {
            printf("Validation failed: diagonal element [%llu,%llu] is not zero\n",
                   badDiagonal, badDiagonal);
        }
        return false;
    }

    for (size_t offset = 0; offset < edges.size();) {
        const int length = std::min(edges.size() - offset, size_t{INT_MAX});
        unsigned int* chunk = edges.data() + offset;
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : chunk, chunk, length, MPI_UNSIGNED,
                   MPI_MAX, 0, comm);
        offset += length;
    }
    if (rank == 0) {
        for (size_t i = 0; i < sample; ++i) {
            for (size_t j = 0; j < sample; ++j) {
                const unsigned int distance = edges[i * n + j];
                for (size_t k = 0; k < n; ++k) {
                    const unsigned int left = edges[i * n + k];
                    const unsigned int right = edges[(sample + j) * n + k];
                    if (left < INF && right < INF && left + right < distance) {
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numNodes = 512;
    bool validate = false;
    bool report = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            report = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (numNodes < 0) {
        if (rank == 0) {
            printf("Number of nodes must be nonnegative\n");
        }
        return 1;
    }

    int dims[2];
    const int maxDimension = std::max(1, numNodes);
    int activeRanks = std::min<long long>(ranks, 1LL * maxDimension * maxDimension);
    do {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(activeRanks, 2, dims);
        if (std::max(dims[0], dims[1]) <= maxDimension) {
            break;
        }
        --activeRanks;
    } while (true);

    MPI_Comm active = MPI_COMM_WORLD;
    if (activeRanks != ranks) {
        MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
    }
    int status = 0;
    if (active != MPI_COMM_NULL) {
        ProcessGrid grid{dims[0], dims[1], rank / dims[1], rank % dims[1],
                         MPI_COMM_NULL, MPI_COMM_NULL};
        if (grid.cols > 1) {
            MPI_Comm_split(active, grid.row, grid.col, &grid.rowComm);
        }
        if (grid.rows > 1) {
            MPI_Comm_split(active, grid.col, grid.row, &grid.colComm);
        }

        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %d\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        LocalMatrix matrix(numNodes, grid);
        initializeMatrices(matrix, grid);

        if (rank == 0) {
            printf("Computing shortest paths...\n");
        }
        MPI_Barrier(active);
        const double start = MPI_Wtime();
        if (numNodes != 0) {
            floydWarshall(matrix, grid);
        }
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active);
        if (rank == 0) {
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
            printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
        }

        if (report) {
            printResults(matrix, grid, active, rank);
        }
        if (validate) {
            if (rank == 0) {
                printf("Validating result...\n");
            }
            const bool valid = validateResult(matrix, grid, active, rank);
            if (rank == 0) {
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
        if (grid.rowComm != MPI_COMM_NULL) {
            MPI_Comm_free(&grid.rowComm);
        }
        if (grid.colComm != MPI_COMM_NULL) {
            MPI_Comm_free(&grid.colComm);
        }
        if (active != MPI_COMM_WORLD) {
            MPI_Comm_free(&active);
        }
    }
    if (validate) {
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int status = runBenchmark(argc, argv);
    MPI_Finalize();
    return status;
}
