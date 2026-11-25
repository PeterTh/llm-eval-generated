#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include <mpi.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int MAX_TILE_SIZE = 64;
using Distance = unsigned char;
static_assert(MAX_DISTANCE + 1 < UCHAR_MAX);

struct Grid {
    int rows;
    int cols;
    int row;
    int col;
    MPI_Comm across;
    MPI_Comm down;
};

struct Matrix {
    size_t nodes;
    size_t tileSize;
    size_t tileCount;
    size_t tileArea;
    size_t localRows;
    size_t localCols;
    std::vector<Distance> dist;
    std::vector<unsigned int> path;

    Matrix(size_t nodes, const Grid& grid)
        : nodes(nodes),
          tileSize(std::clamp(nodes / std::max(grid.rows, grid.cols),
                              size_t{1}, size_t{MAX_TILE_SIZE})),
          tileCount((nodes + tileSize - 1) / tileSize),
          tileArea(tileSize * tileSize),
          localRows((tileCount + grid.rows - 1 - grid.row) / grid.rows),
          localCols((tileCount + grid.cols - 1 - grid.col) / grid.cols),
          dist(localRows * localCols * tileArea),
          path(dist.size()) {}

    size_t extent(size_t tile) const {
        return std::min(tileSize, nodes - tile * tileSize);
    }

    size_t offset(size_t localRow, size_t localCol) const {
        return (localRow * localCols + localCol) * tileArea;
    }
};

unsigned int advanceSeed(unsigned int seed, size_t count) {
#ifdef __GLIBC__
    // glibc rand_r advances its 32-bit LCG three times per value. Compose
    // the affine transforms by squaring to seek into the original stream.
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;
    for (size_t steps = 3 * count; steps != 0; steps >>= 1) {
        if (steps & 1) {
            seed = multiplier * seed + increment;
        }
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
#else
    while (count-- != 0) {
        rand_r(&seed);
    }
#endif
    return seed;
}

void initialize(Matrix& matrix, const Grid& grid) {
    const size_t b = matrix.tileSize;
    unsigned int seed = 42;
    size_t streamPosition = 0;
    for (size_t tr = grid.row; tr < matrix.tileCount; tr += grid.rows) {
        for (size_t i = 0; i < matrix.extent(tr); ++i) {
            const size_t globalRow = tr * b + i;
            for (size_t tc = grid.col; tc < matrix.tileCount; tc += grid.cols) {
                const size_t globalCol = tc * b;
                const size_t position = globalRow * matrix.nodes + globalCol;
                seed = advanceSeed(seed, position - streamPosition);
                streamPosition = position + matrix.extent(tc);
                const size_t start = matrix.offset(tr / grid.rows, tc / grid.cols) + i * b;
                for (size_t j = 0; j < matrix.extent(tc); ++j) {
                    const auto distance = 1u + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                    matrix.dist[start + j] = globalRow == globalCol + j ? 0 : distance;
                    matrix.path[start + j] = static_cast<unsigned int>(globalRow);
                }
            }
        }
    }
}

inline void relax(Distance* __restrict dist, unsigned int* __restrict path,
                  const Distance* __restrict row, const Distance* __restrict col,
                  size_t height, size_t width, size_t stride, unsigned int pivot) {
    for (size_t i = 0; i < height; ++i) {
        const unsigned int toPivot = col[i];
        size_t j = 0;
#ifdef __AVX2__
        const __m256i base = _mm256_set1_epi8(static_cast<char>(toPivot));
        for (; j + 32 <= width; j += 32) {
            auto* distance = reinterpret_cast<__m256i_u*>(dist + j);
            const auto previous = _mm256_loadu_si256(distance);
            const auto fromPivot = _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(row + j));
            // A sum that saturates at 255 cannot improve a stored distance.
            const auto best = _mm256_min_epu8(previous, _mm256_adds_epu8(base, fromPivot));
            auto changed = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(previous, best)));
            if (changed != 0) {
                _mm256_storeu_si256(distance, best);
                do {
                    path[j + std::countr_zero(changed)] = pivot;
                    changed &= changed - 1;
                } while (changed != 0);
            }
        }
#endif
        for (; j < width; ++j) {
            const unsigned int candidate = toPivot + row[j];
            if (candidate < dist[j]) {
                dist[j] = candidate;
                path[j] = pivot;
            }
        }
        dist += stride;
        path += stride;
    }
}

// MPI-3 counts are int. Normal panels fit in a single collective.
void broadcastPanel(std::vector<Distance>& panel, int root, MPI_Comm comm,
                    MPI_Request& request) {
    size_t offset = 0;
    while (panel.size() - offset > INT_MAX) {
        MPI_Bcast(panel.data() + offset, INT_MAX, MPI_UNSIGNED_CHAR, root, comm);
        offset += INT_MAX;
    }
    MPI_Ibcast(panel.data() + offset, static_cast<int>(panel.size() - offset),
               MPI_UNSIGNED_CHAR, root, comm, &request);
}

void floydWarshall(Matrix& matrix, const Grid& grid) {
    const size_t b = matrix.tileSize;
    const size_t area = matrix.tileArea;
    std::vector<Distance> diagonalRows(area), diagonalCols(area);
    std::vector<Distance> pivotRows(matrix.localCols * area);
    std::vector<Distance> pivotCols(matrix.localRows * area);
    MPI_Request requests[2];

    for (size_t block = 0; block < matrix.tileCount; ++block) {
        const size_t pivots = matrix.extent(block);
        const int ownerRow = block % grid.rows;
        const int ownerCol = block % grid.cols;

        // Save each pivot as it appeared at its original Floyd-Warshall step.
        // Relaxations cannot change the current pivot row or column because
        // the diagonal is zero and updates require a strict improvement.
        if (grid.row == ownerRow && grid.col == ownerCol) {
            const size_t offset = matrix.offset(block / grid.rows, block / grid.cols);
            auto* dist = matrix.dist.data() + offset;
            auto* path = matrix.path.data() + offset;
            for (size_t k = 0; k < pivots; ++k) {
                auto* row = diagonalRows.data() + k * b;
                auto* col = diagonalCols.data() + k * b;
                std::copy_n(dist + k * b, pivots, row);
                for (size_t i = 0; i < pivots; ++i) {
                    col[i] = dist[i * b + k];
                }
                relax(dist, path, row, col, pivots, pivots, b, block * b + k);
            }
            std::copy_n(diagonalRows.data(), area, pivotRows.data() + block / grid.cols * area);
            std::copy_n(diagonalCols.data(), area, pivotCols.data() + block / grid.rows * area);
        }

        requests[0] = MPI_REQUEST_NULL;
        requests[1] = MPI_REQUEST_NULL;
        if (grid.row == ownerRow && grid.cols > 1) {
            broadcastPanel(diagonalCols, ownerCol, grid.across, requests[0]);
        }
        if (grid.col == ownerCol && grid.rows > 1) {
            broadcastPanel(diagonalRows, ownerRow, grid.down, requests[1]);
        }
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        if (grid.row == ownerRow) {
            for (size_t tc = grid.col; tc < matrix.tileCount; tc += grid.cols) {
                if (tc == block) {
                    continue;
                }
                const size_t offset = matrix.offset(block / grid.rows, tc / grid.cols);
                auto* dist = matrix.dist.data() + offset;
                auto* path = matrix.path.data() + offset;
                const size_t width = matrix.extent(tc);
                auto* row = pivotRows.data() + tc / grid.cols * area;
                for (size_t k = 0; k < pivots; ++k, row += b) {
                    std::copy_n(dist + k * b, width, row);
                    relax(dist, path, row, diagonalCols.data() + k * b,
                          pivots, width, b, block * b + k);
                }
            }
        }
        if (grid.col == ownerCol) {
            for (size_t tr = grid.row; tr < matrix.tileCount; tr += grid.rows) {
                if (tr == block) {
                    continue;
                }
                const size_t offset = matrix.offset(tr / grid.rows, block / grid.cols);
                auto* dist = matrix.dist.data() + offset;
                auto* path = matrix.path.data() + offset;
                const size_t height = matrix.extent(tr);
                auto* col = pivotCols.data() + tr / grid.rows * area;
                for (size_t k = 0; k < pivots; ++k, col += b) {
                    for (size_t i = 0; i < height; ++i) {
                        col[i] = dist[i * b + k];
                    }
                    relax(dist, path, diagonalRows.data() + k * b, col,
                          height, pivots, b, block * b + k);
                }
            }
        }

        requests[0] = MPI_REQUEST_NULL;
        requests[1] = MPI_REQUEST_NULL;
        if (grid.cols > 1) {
            broadcastPanel(pivotCols, ownerCol, grid.across, requests[0]);
        }
        if (grid.rows > 1) {
            broadcastPanel(pivotRows, ownerRow, grid.down, requests[1]);
        }
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        for (size_t tr = grid.row; tr < matrix.tileCount; tr += grid.rows) {
            if (tr == block) {
                continue;
            }
            const auto* cols = pivotCols.data() + tr / grid.rows * area;
            const size_t height = matrix.extent(tr);
            for (size_t tc = grid.col; tc < matrix.tileCount; tc += grid.cols) {
                if (tc == block) {
                    continue;
                }
                const size_t offset = matrix.offset(tr / grid.rows, tc / grid.cols);
                const auto* rows = pivotRows.data() + tc / grid.cols * area;
                const size_t width = matrix.extent(tc);
                for (size_t k = 0; k < pivots; ++k) {
                    relax(matrix.dist.data() + offset, matrix.path.data() + offset,
                          rows + k * b, cols + k * b, height, width, b, block * b + k);
                }
            }
        }
    }
}

void printResults(const Matrix& matrix, const Grid& grid, int rank, MPI_Comm comm) {
    const size_t n = matrix.nodes;
    const size_t elements = n * n;
    if (elements == 0) {
        if (rank == 0) {
            print_results_int(matrix.dist, "DistanceMatrix");
        }
        return;
    }

    const size_t b = matrix.tileSize;
    const size_t area = matrix.tileArea;
    const std::array<size_t, 5> indices{0, elements / 4, elements / 2,
                                        elements / 4 * 3 + elements % 4 * 3 / 4, elements - 1};
    std::array<unsigned int, 5> samples{};
    unsigned int minValue = UINT_MAX;
    unsigned int maxValue = 0;
    hash digest;
    // Receive one block row at a time so the ordered hash needs only O(b*n)
    // storage on rank zero. Tile groups occupy consecutive slots by rank.
    std::vector<Distance> buffer(rank == 0 ? matrix.tileCount * area : 0);
    for (size_t tr = 0; tr < matrix.tileCount; ++tr) {
        const int ownerRow = tr % grid.rows;
        if (rank != 0 && grid.row != ownerRow) {
            continue;
        }
        size_t bufferOffset = 0;
        const int firstCol = rank == 0 ? 0 : grid.col;
        const int lastCol = rank == 0 ? grid.cols : grid.col + 1;
        for (int col = firstCol; col < lastCol; ++col) {
            const int source = ownerRow * grid.cols + col;
            const size_t count = (matrix.tileCount + grid.cols - 1 - col) / grid.cols * area;
            for (size_t i = 0; i < count;) {
                const int chunk = static_cast<int>(std::min(count - i, size_t{INT_MAX}));
                if (source == 0) {
                    std::copy_n(matrix.dist.data() + matrix.offset(tr / grid.rows, 0) + i,
                                chunk, buffer.data() + bufferOffset + i);
                } else if (rank == 0) {
                    MPI_Recv(buffer.data() + bufferOffset + i, chunk, MPI_UNSIGNED_CHAR, source, 0,
                             comm, MPI_STATUS_IGNORE);
                } else {
                    MPI_Send(matrix.dist.data() + matrix.offset(tr / grid.rows, 0) + i,
                             chunk, MPI_UNSIGNED_CHAR, 0, 0, comm);
                }
                i += chunk;
            }
            bufferOffset += count;
        }

        if (rank != 0) {
            continue;
        }
        const size_t baseTiles = matrix.tileCount / grid.cols;
        const size_t extraTiles = matrix.tileCount % grid.cols;
        for (size_t i = 0; i < matrix.extent(tr); ++i) {
            for (size_t tc = 0; tc < matrix.tileCount; ++tc) {
                const size_t col = tc % grid.cols;
                const size_t slot = col * baseTiles + std::min(col, extraTiles) + tc / grid.cols;
                const auto* row = buffer.data() + slot * area + i * b;
                for (size_t j = 0; j < matrix.extent(tc); ++j) {
                    // The serial benchmark hashes the bytes of an unsigned int.
                    const unsigned int value = row[j];
                    digest.add(value);
                    minValue = std::min(minValue, value);
                    maxValue = std::max(maxValue, value);
                }
            }
        }
        for (size_t s = 0; s < indices.size(); ++s) {
            const size_t i = indices[s] / n;
            if (i / b == tr) {
                const size_t j = indices[s] % n;
                const size_t tc = j / b;
                const size_t col = tc % grid.cols;
                const size_t slot = col * baseTiles + std::min(col, extraTiles) + tc / grid.cols;
                samples[s] = buffer[slot * area + (i % b) * b + j % b];
            }
        }
    }

    if (rank == 0) {
        printf("=== RESULTS ===\nName: DistanceMatrix\nElements: %zu\n", elements);
        printf("Min: %u\nMax: %u\n", minValue, maxValue);
        for (size_t s = 0; s < indices.size(); ++s) {
            printf("Sample[%zu]: %u\n", indices[s], samples[s]);
        }
        printf("Hash: %016" PRIx64 "\n=== END RESULTS ===\n", digest.get());
    }
}

bool validateResult(const Matrix& matrix, const Grid& grid, int rank, MPI_Comm comm) {
    const size_t n = matrix.nodes;
    const size_t b = matrix.tileSize;
    const size_t sample = std::min(n, size_t{10});
    std::vector<Distance> rows(sample * n), cols(sample * n);
    uint64_t badDiagonal = n;
    for (size_t tr = grid.row; tr < matrix.tileCount; tr += grid.rows) {
        for (size_t tc = grid.col; tc < matrix.tileCount; tc += grid.cols) {
            const auto* tile = matrix.dist.data() + matrix.offset(tr / grid.rows, tc / grid.cols);
            if (tr == tc) {
                for (size_t i = 0; i < matrix.extent(tr); ++i) {
                    if (tile[i * b + i] != 0) {
                        badDiagonal = std::min(badDiagonal, static_cast<uint64_t>(tr * b + i));
                    }
                }
            }
            for (size_t i = 0; i < matrix.extent(tr) && tr * b + i < sample; ++i) {
                std::copy_n(tile + i * b, matrix.extent(tc), rows.data() + (tr * b + i) * n + tc * b);
            }
            for (size_t i = 0; i < matrix.extent(tr); ++i) {
                for (size_t j = 0; j < matrix.extent(tc) && tc * b + j < sample; ++j) {
                    cols[(tc * b + j) * n + tr * b + i] = tile[i * b + j];
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &badDiagonal, 1, MPI_UINT64_T, MPI_MIN, comm);
    if (badDiagonal != n) {
        if (rank == 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                   static_cast<size_t>(badDiagonal), static_cast<size_t>(badDiagonal));
        }
        return false;
    }

    for (size_t i = 0; i < rows.size();) {
        const int count = static_cast<int>(std::min(rows.size() - i, size_t{INT_MAX}));
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : rows.data() + i,
                   rank == 0 ? rows.data() + i : nullptr, count, MPI_UNSIGNED_CHAR, MPI_MAX, 0, comm);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : cols.data() + i,
                   rank == 0 ? cols.data() + i : nullptr, count, MPI_UNSIGNED_CHAR, MPI_MAX, 0, comm);
        i += count;
    }
    if (rank == 0) {
        for (size_t i = 0; i < sample; ++i) {
            for (size_t j = 0; j < sample; ++j) {
                const unsigned int direct = rows[i * n + j];
                for (size_t k = 0; k < n; ++k) {
                    const unsigned int left = rows[i * n + k];
                    const unsigned int right = cols[j * n + k];
                    if (left < INF && right < INF && left + right < direct) {
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

int run(size_t numNodes, bool validate, bool report, int rank, int rows, int cols, MPI_Comm comm) {
    Grid grid{rows, cols, rank / cols, rank % cols, MPI_COMM_NULL, MPI_COMM_NULL};
    MPI_Comm_split(comm, grid.row, grid.col, &grid.across);
    MPI_Comm_split(comm, grid.col, grid.row, &grid.down);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    Matrix matrix(numNodes, grid);
    initialize(matrix, grid);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    floydWarshall(matrix, grid);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }
    if (report) {
        printResults(matrix, grid, rank, comm);
    }
    int result = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(matrix, grid, rank, comm);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        result = valid ? 0 : 1;
    }
    MPI_Comm_free(&grid.across);
    MPI_Comm_free(&grid.down);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int options[] = {512, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const long nodes = std::strtol(argv[++i], nullptr, 10);
                if (nodes < 0 || nodes > INT_MAX) {
                    printf("Invalid number of nodes: %s\n", argv[i]);
                    options[3] = 2;
                    break;
                }
                options[0] = static_cast<int>(nodes);
            } else if (strcmp(argv[i], "-v") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[2] = 1;
            } else {
                const bool help = strcmp(argv[i], "-h") == 0;
                if (!help) {
                    printf("Unknown option: %s\n", argv[i]);
                }
                printUsage(argv[0]);
                options[3] = help ? 1 : 2;
                break;
            }
        }
    }
    MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (options[3] != 0) {
        MPI_Finalize();
        return options[3] == 2 ? 1 : 0;
    }

    const int nodes = options[0];
    int dims[2]{};
    MPI_Dims_create(size, 2, dims);
    dims[0] = std::clamp(dims[0], 1, std::max(1, nodes));
    dims[1] = std::clamp(dims[1], 1, std::max(1, nodes));
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < dims[0] * dims[1] ? 0 : MPI_UNDEFINED, rank, &active);

    int result = 0;
    if (active != MPI_COMM_NULL) {
        try {
            result = run(nodes, options[1], options[2], rank, dims[0], dims[1], active);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
