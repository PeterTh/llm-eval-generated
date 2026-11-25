#include <algorithm>
#include <array>
#include <climits>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/hash.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t MAX_BLOCK = 32;
constexpr size_t COLUMN_TILE = 128;
constexpr unsigned int PIVOT_SCALE = 64;
static_assert(MAX_BLOCK < PIVOT_SCALE);
static_assert(2 * (MAX_DISTANCE + 1) * PIVOT_SCALE + MAX_BLOCK <= UINT16_MAX);

struct Grid {
    MPI_Comm comm;
    MPI_Comm row;
    MPI_Comm col;
    int rowCount;
    int colCount;
    int rowRank;
    int colRank;
    size_t nodes;
    size_t blockSize;

    size_t blocks() const {
        return (nodes + blockSize - 1) / blockSize;
    }

    size_t width(size_t block) const {
        return std::min(blockSize, nodes - block * blockSize);
    }

    size_t localSize(int parts, int coord) const {
        const size_t whole = nodes / blockSize;
        return (whole / parts + (static_cast<size_t>(coord) < whole % parts)) * blockSize
               + (static_cast<size_t>(coord) == whole % parts ? nodes % blockSize : 0);
    }

    size_t localOffset(size_t index, int parts) const {
        return index / blockSize / parts * blockSize + index % blockSize;
    }

    size_t globalIndex(size_t index, int parts, int coord) const {
        return (index / blockSize * parts + coord) * blockSize + index % blockSize;
    }

    bool root() const {
        return rowRank == 0 && colRank == 0;
    }
};

// Pack a block row in process-column order for MPI. Within each process-column
// segment the rows have the same layout as in that rank's local matrix.
struct Stripe {
    std::vector<int> counts;
    std::vector<int> offsets;
    std::vector<unsigned int> data;

    Stripe(const Grid& grid, size_t height, bool holdData)
        : counts(grid.colCount), offsets(grid.colCount),
          data(holdData ? height * grid.nodes : 0) {
        size_t offset = 0;
        for (int col = 0; col < grid.colCount; ++col) {
            counts[col] = height * grid.localSize(grid.colCount, col);
            offsets[col] = offset;
            offset += counts[col];
        }
    }

    template <typename Visitor>
    void visit(const Grid& grid, size_t height, Visitor&& visitor) {
        for (size_t i = 0; i < height; ++i) {
            for (size_t block = 0; block < grid.blocks(); ++block) {
                const int col = block % grid.colCount;
                const size_t pitch = grid.localSize(grid.colCount, col);
                const size_t start = offsets[col] + i * pitch + block / grid.colCount * grid.blockSize;
                for (size_t j = 0; j < grid.width(block); ++j) {
                    visitor(data[start + j]);
                }
            }
        }
    }
};

unsigned int nextDistance(unsigned int& seed) {
    return 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
}

#if defined(__GLIBC__)
// glibc's rand_r advances its 32-bit LCG three times per result. Compose
// those transitions so each rank can skip the entries owned by other ranks.
struct SeedTransform {
    uint32_t multiplier;
    uint32_t increment;

    uint32_t operator()(uint32_t seed) const {
        return multiplier * seed + increment;
    }

    SeedTransform then(SeedTransform next) const {
        return {next.multiplier * multiplier, next(increment)};
    }
};

SeedTransform skipDistances(size_t count) {
    const SeedTransform step{1103515245u, 12345u};
    SeedTransform power = step.then(step).then(step);
    SeedTransform result{1, 0};
    while (count != 0) {
        if (count & 1) {
            result = result.then(power);
        }
        power = power.then(power);
        count >>= 1;
    }
    return result;
}
#endif

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const Grid& grid,
                              size_t rows, size_t cols) {
#if defined(__GLIBC__)
    const auto skipRow = skipDistances(grid.nodes);
    const auto skipRowBlock = skipDistances((grid.rowCount - 1) * grid.blockSize * grid.nodes);
    const auto skipColBlocks = skipDistances((grid.colCount - 1) * grid.blockSize);
    uint32_t rowSeed = skipDistances(grid.rowRank * grid.blockSize * grid.nodes
                                    + grid.colRank * grid.blockSize)(42);
    for (size_t i = 0; i < rows; ++i) {
        unsigned int seed = rowSeed;
        for (size_t j = 0; j < cols;) {
            const size_t end = std::min(j + grid.blockSize, cols);
            for (; j < end; ++j) {
                dist[i * cols + j] = nextDistance(seed);
            }
            seed = skipColBlocks(seed);
        }
        rowSeed = skipRow(rowSeed);
        if ((i + 1) % grid.blockSize == 0) {
            rowSeed = skipRowBlock(rowSeed);
        }
    }
#else
    unsigned int seed = 42;
    for (size_t block = 0; block < grid.blocks(); ++block) {
        const int owner = block % grid.rowCount;
        if (!grid.root() && grid.rowRank != owner) {
            continue;
        }
        const size_t height = grid.width(block);
        Stripe stripe(grid, height, grid.colRank == 0);
        if (grid.root()) {
            stripe.visit(grid, height, [&](unsigned int& value) { value = nextDistance(seed); });
            if (owner != 0) {
                MPI_Send(stripe.data.data(), stripe.data.size(), MPI_UNSIGNED,
                         owner * grid.colCount, 0, grid.comm);
            }
        } else if (grid.colRank == 0) {
            MPI_Recv(stripe.data.data(), stripe.data.size(), MPI_UNSIGNED, 0, 0,
                     grid.comm, MPI_STATUS_IGNORE);
        }
        if (grid.rowRank == owner) {
            auto* local = dist.data() + block / grid.rowCount * grid.blockSize * cols;
            MPI_Scatterv(stripe.data.data(), stripe.counts.data(), stripe.offsets.data(),
                         MPI_UNSIGNED, local, height * cols, MPI_UNSIGNED, 0, grid.row);
        }
    }
#endif

    for (size_t i = 0; i < rows; ++i) {
        const size_t global = grid.globalIndex(i, grid.rowCount, grid.rowRank);
        if (global / grid.blockSize % grid.colCount == static_cast<size_t>(grid.colRank)) {
            dist[i * cols + grid.localOffset(global, grid.colCount)] = 0;
        }
    }
}

template <typename Distance>
inline void relax(unsigned int* __restrict dist, unsigned int* __restrict path,
                  const Distance* __restrict pivot, size_t count,
                  unsigned int toPivot, unsigned int k) {
    for (size_t j = 0; j < count; ++j) {
        const unsigned int candidate = toPivot + pivot[j];
        const unsigned int oldDistance = dist[j];
        const unsigned int update = 0u - (candidate < oldDistance);
        dist[j] = std::min(candidate, oldDistance);
        path[j] = (update & k) | (~update & path[j]);
    }
}

// For an output tile all candidate distances are known independently. Order
// them by distance and then by pivot, with the old value first on a tie.
// Packing that pair lets the compiler vectorize the reduction in 16-bit lanes.
void relaxTileRow(unsigned int* __restrict dist, unsigned int* __restrict path,
                  const uint16_t* __restrict pivotRows, const uint16_t* __restrict pivotCol,
                  size_t count, size_t width, size_t first) {
    uint16_t best[COLUMN_TILE];
    for (size_t j = 0; j < count; ++j) {
        best[j] = dist[j] * PIVOT_SCALE;
    }
    for (size_t k = 0; k < width; ++k) {
        const auto* pivot = pivotRows + k * count;
        for (size_t j = 0; j < count; ++j) {
            best[j] = std::min(best[j], static_cast<uint16_t>(pivotCol[k] + pivot[j]));
        }
    }
    for (size_t j = 0; j < count; ++j) {
        dist[j] = best[j] / PIVOT_SCALE;
        const unsigned int offset = best[j] % PIVOT_SCALE;
        if (offset != 0) {
            path[j] = first + offset - 1;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const Grid& grid, size_t rows, size_t cols) {
    std::vector<uint16_t> pivotRows(grid.blockSize * cols);
    std::vector<uint16_t> pivotCols(grid.blockSize * rows);
    std::vector<std::pair<size_t, size_t>> columnTiles;
    columnTiles.reserve(cols / COLUMN_TILE + 2);
    std::array<unsigned int, MAX_BLOCK * MAX_BLOCK> diagonalRows;
    std::array<unsigned int, MAX_BLOCK * MAX_BLOCK> diagonalCols;
    constexpr size_t rowTile = 16;

    for (size_t block = 0; block < grid.blocks(); ++block) {
        const size_t first = block * grid.blockSize;
        const size_t width = grid.width(block);
        const size_t rowStart = block / grid.rowCount * grid.blockSize;
        const size_t colStart = block / grid.colCount * grid.blockSize;
        const int rowOwner = block % grid.rowCount;
        const int colOwner = block % grid.colCount;
        const bool ownsRows = grid.rowRank == rowOwner;
        const bool ownsCols = grid.colRank == colOwner;

        columnTiles.clear();
        for (size_t j = 0; j < cols;) {
            if (ownsCols && j == colStart) {
                j += width;
                continue;
            }
            const size_t limit = ownsCols && j < colStart ? colStart : cols;
            const size_t count = std::min(COLUMN_TILE, limit - j);
            columnTiles.emplace_back(j, count);
            j += count;
        }

        // Record each pivot row and column at the start of its serial step.
        // Keeping these snapshots preserves strict improvements and tie ordering.
        if (ownsRows && ownsCols) {
            for (size_t k = 0; k < width; ++k) {
                const auto* pivot = dist.data() + (rowStart + k) * cols + colStart;
                auto* rowSnapshot = diagonalRows.data() + k * width;
                auto* colSnapshot = diagonalCols.data() + k * width;
                std::copy_n(pivot, width, rowSnapshot);
                for (size_t i = 0; i < width; ++i) {
                    const size_t base = (rowStart + i) * cols + colStart;
                    colSnapshot[i] = dist[base + k];
                    relax(dist.data() + base, path.data() + base, rowSnapshot,
                          width, colSnapshot[i], first + k);
                }
            }
        }

        MPI_Request diagonalRequests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (ownsRows && grid.colCount > 1) {
            MPI_Ibcast(diagonalCols.data(), width * width, MPI_UNSIGNED, colOwner,
                       grid.row, &diagonalRequests[0]);
        }
        if (ownsCols && grid.rowCount > 1) {
            MPI_Ibcast(diagonalRows.data(), width * width, MPI_UNSIGNED, rowOwner,
                       grid.col, &diagonalRequests[1]);
        }
        MPI_Waitall(2, diagonalRequests, MPI_STATUSES_IGNORE);

        if (ownsRows) {
            // The diagonal tile has already completed this entire block.
            for (const auto& [j, count] : columnTiles) {
                for (size_t k = 0; k < width; ++k) {
                    auto* snapshot = pivotRows.data() + j * width + k * count;
                    const auto* pivot = dist.data() + (rowStart + k) * cols + j;
                    std::copy_n(pivot, count, snapshot);
                    for (size_t i = 0; i < width; ++i) {
                        const unsigned int toPivot = diagonalCols[k * width + i];
                        const size_t base = (rowStart + i) * cols + j;
                        relax(dist.data() + base, path.data() + base, snapshot,
                              count, toPivot, first + k);
                    }
                    for (size_t c = 0; c < count; ++c) {
                        snapshot[c] *= PIVOT_SCALE;
                    }
                }
            }
        }

        MPI_Request panelRequests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (grid.rowCount > 1) {
            MPI_Ibcast(pivotRows.data(), width * cols, MPI_UINT16_T, rowOwner,
                       grid.col, &panelRequests[0]);
        }

        if (ownsCols) {
            for (size_t i = 0; i < rows; ++i) {
                if (ownsRows && i >= rowStart && i < rowStart + width) {
                    continue;
                }
                const size_t base = i * cols + colStart;
                for (size_t k = 0; k < width; ++k) {
                    const unsigned int toPivot = dist[base + k];
                    pivotCols[i * width + k] = toPivot * PIVOT_SCALE + k + 1;
                    relax(dist.data() + base, path.data() + base,
                          diagonalRows.data() + k * width, width, toPivot, first + k);
                }
            }
        }

        if (grid.colCount > 1) {
            MPI_Ibcast(pivotCols.data(), width * rows, MPI_UINT16_T, colOwner,
                       grid.row, &panelRequests[1]);
        }
        MPI_Waitall(2, panelRequests, MPI_STATUSES_IGNORE);

        // Keep each output tile in cache across the entire group of pivots.
        for (size_t i = 0; i < rows;) {
            if (ownsRows && i == rowStart) {
                i += width;
                continue;
            }
            const size_t rowLimit = ownsRows && i < rowStart ? rowStart : rows;
            const size_t rowEnd = std::min(i + rowTile, rowLimit);
            for (const auto& [j, count] : columnTiles) {
                const auto* pivots = pivotRows.data() + j * width;
                for (size_t r = i; r < rowEnd; ++r) {
                    const size_t base = r * cols + j;
                    relaxTileRow(dist.data() + base, path.data() + base, pivots,
                                 pivotCols.data() + r * width, count, width, first);
                }
            }
            i = rowEnd;
        }
    }
}

void printResults(const std::vector<unsigned int>& dist, const Grid& grid, size_t cols) {
    const size_t elements = grid.nodes * grid.nodes;
    const size_t indices[] = {0, elements / 4, elements / 2, 3 * (elements / 4) + 3 * (elements % 4) / 4,
                              elements == 0 ? 0 : elements - 1};
    unsigned int samples[5] = {};
    unsigned int minValue = std::numeric_limits<unsigned int>::max();
    unsigned int maxValue = 0;
    size_t index = 0;
    size_t sample = 0;
    hash digest;

    for (size_t block = 0; block < grid.blocks(); ++block) {
        const int owner = block % grid.rowCount;
        if (!grid.root() && grid.rowRank != owner) {
            continue;
        }
        const size_t height = grid.width(block);
        Stripe stripe(grid, height, grid.colRank == 0);
        if (grid.rowRank == owner) {
            const auto* local = dist.data() + block / grid.rowCount * grid.blockSize * cols;
            MPI_Gatherv(local, height * cols, MPI_UNSIGNED, stripe.data.data(),
                        stripe.counts.data(), stripe.offsets.data(), MPI_UNSIGNED, 0, grid.row);
            if (!grid.root() && grid.colRank == 0) {
                MPI_Send(stripe.data.data(), stripe.data.size(), MPI_UNSIGNED, 0, 0, grid.comm);
            }
        } else {
            MPI_Recv(stripe.data.data(), stripe.data.size(), MPI_UNSIGNED,
                     owner * grid.colCount, 0, grid.comm, MPI_STATUS_IGNORE);
        }
        if (grid.root()) {
            stripe.visit(grid, height, [&](unsigned int value) {
                digest.add(value);
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
                while (sample < 5 && index == indices[sample]) {
                    samples[sample++] = value;
                }
                ++index;
            });
        }
    }

    if (grid.root()) {
        printf("=== RESULTS ===\n");
        printf("Name: DistanceMatrix\n");
        printf("Elements: %zu\n", elements);
        if (elements != 0) {
            printf("Min: %u\n", minValue);
            printf("Max: %u\n", maxValue);
            for (size_t i = 0; i < 5; ++i) {
                printf("Sample[%zu]: %u\n", indices[i], samples[i]);
            }
            printf("Hash: %016" PRIx64 "\n", digest.get());
        }
        printf("=== END RESULTS ===\n");
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const Grid& grid,
                    size_t rows, size_t cols) {
    const size_t sample = std::min(grid.nodes, size_t{10});
    const size_t values = sample * grid.nodes;
    std::vector<unsigned int> strips(2 * values);
    size_t badDiagonal = grid.nodes;
    for (size_t i = 0; i < rows; ++i) {
        const size_t globalRow = grid.globalIndex(i, grid.rowCount, grid.rowRank);
        if (globalRow / grid.blockSize % grid.colCount == static_cast<size_t>(grid.colRank)
            && dist[i * cols + grid.localOffset(globalRow, grid.colCount)] != 0) {
            badDiagonal = std::min(badDiagonal, globalRow);
        }
        for (size_t j = 0; j < cols; ++j) {
            const size_t globalCol = grid.globalIndex(j, grid.colCount, grid.colRank);
            if (globalRow >= sample && globalCol >= sample) {
                break;
            }
            if (globalRow < sample) {
                strips[globalRow * grid.nodes + globalCol] = dist[i * cols + j];
            }
            if (globalCol < sample) {
                strips[values + globalCol * grid.nodes + globalRow] = dist[i * cols + j];
            }
        }
    }

    unsigned long long diagonal = badDiagonal;
    unsigned long long firstBadDiagonal = grid.nodes;
    MPI_Reduce(&diagonal, &firstBadDiagonal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, grid.comm);
    if (grid.root()) {
        MPI_Reduce(MPI_IN_PLACE, strips.data(), strips.size(), MPI_UNSIGNED, MPI_SUM, 0, grid.comm);
    } else {
        MPI_Reduce(strips.data(), nullptr, strips.size(), MPI_UNSIGNED, MPI_SUM, 0, grid.comm);
    }

    bool valid = true;
    if (grid.root()) {
        if (firstBadDiagonal != grid.nodes) {
            printf("Validation failed: diagonal element [%llu,%llu] is not zero\n",
                   firstBadDiagonal, firstBadDiagonal);
            valid = false;
        }
        for (size_t i = 0; valid && i < sample; ++i) {
            for (size_t j = 0; valid && j < sample; ++j) {
                const auto* row = strips.data() + i * grid.nodes;
                const auto* col = strips.data() + values + j * grid.nodes;
                for (size_t k = 0; k < grid.nodes; ++k) {
                    if (row[k] < INF && col[k] < INF && row[k] + col[k] < row[j]) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        valid = false;
                        break;
                    }
                }
            }
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
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

int run(int argc, char** argv, int rank, int size) {
    size_t numNodes = 512;
    bool validate = false;
    bool print = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            print = true;
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

    if (numNodes > INT_MAX / MAX_BLOCK) {
        if (rank == 0) {
            fprintf(stderr, "Number of nodes exceeds the supported range.\n");
        }
        return 1;
    }

    int dims[2] = {};
    MPI_Dims_create(size, 2, dims);
    const size_t blockSize = std::clamp(numNodes / dims[0], size_t{1}, MAX_BLOCK);
    const int blocks = std::max(size_t{1}, (numNodes + blockSize - 1) / blockSize);
    const int rowCount = std::min(dims[0], blocks);
    const int colCount = std::min(dims[1], blocks);
    const bool active = rank < rowCount * colCount;
    Grid grid{MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL, rowCount, colCount,
              rank / colCount, rank % colCount, numNodes, blockSize};
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &grid.comm);
    if (!active) {
        return 0;
    }
    MPI_Comm_split(grid.comm, grid.rowRank, grid.colRank, &grid.row);
    MPI_Comm_split(grid.comm, grid.colRank, grid.rowRank, &grid.col);

    if (grid.root()) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const size_t rows = grid.localSize(grid.rowCount, grid.rowRank);
    const size_t cols = grid.localSize(grid.colCount, grid.colRank);
    std::vector<unsigned int> dist(rows * cols);
    std::vector<unsigned int> path(rows * cols);
    initializeDistanceMatrix(dist, grid, rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        std::fill_n(path.data() + i * cols, cols, grid.globalIndex(i, grid.rowCount, grid.rowRank));
    }

    if (grid.root()) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(grid.comm);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, grid, rows, cols);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);
    if (grid.root()) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / duration / 1e9);
    }

    if (print) {
        printResults(dist, grid, cols);
    }
    int status = 0;
    if (validate) {
        if (grid.root()) {
            printf("Validating result...\n");
        }
        status = validateResult(dist, grid, rows, cols) ? 0 : 1;
    }

    MPI_Comm_free(&grid.col);
    MPI_Comm_free(&grid.row);
    MPI_Comm_free(&grid.comm);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = run(argc, argv, rank, size);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
