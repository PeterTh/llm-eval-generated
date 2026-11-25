#include <algorithm>
#include <climits>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "../common/hash.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t PIVOT_BATCH = 64;
constexpr size_t COLUMN_TILE = 512;
#ifdef __AVX2__
constexpr size_t SIMD_COLUMNS = 32;
#endif

struct Partition {
    size_t nodes;
    int parts;

    size_t begin(int part) const { return nodes * part / parts; }
    size_t size(int part) const { return begin(part + 1) - begin(part); }
    int owner(size_t index) const { return ((index + 1) * parts - 1) / nodes; }
};

struct Grid {
    Partition rows;
    Partition cols;
    int row;
    int col;
    MPI_Comm comm;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    int rank(int r, int c) const { return r * cols.parts + c; }
};

// Split large buffers into counts supported by both MPI-3 and MPI-4.
void broadcast(unsigned int* data, size_t count, int root, MPI_Comm comm) {
    while (count > 0) {
        const int chunk = std::min(count, static_cast<size_t>(INT_MAX));
        MPI_Bcast(data, chunk, MPI_UNSIGNED, root, comm);
        data += chunk;
        count -= chunk;
    }
}

MPI_Request startBroadcast(unsigned int* data, size_t count, int root, MPI_Comm comm) {
    if (count > INT_MAX) {
        broadcast(data, count, root, comm);
        return MPI_REQUEST_NULL;
    }
    MPI_Request request;
    MPI_Ibcast(data, count, MPI_UNSIGNED, root, comm, &request);
    return request;
}

#ifdef __GLIBC__
struct RandAdvance {
    unsigned int multiplier = 1;
    unsigned int increment = 0;

    unsigned int operator()(unsigned int seed) const {
        return seed * multiplier + increment;
    }
};

RandAdvance randAdvance(size_t draws) {
    // glibc rand_r advances this unsigned LCG three times per draw.
    RandAdvance advance;
    RandAdvance power{1103515245u, 12345u};
    for (size_t steps = 3 * draws; steps > 0; steps >>= 1) {
        if (steps & 1) {
            advance = {power.multiplier * advance.multiplier, power(advance.increment)};
        }
        power = {power.multiplier * power.multiplier, power(power.increment)};
    }
    return advance;
}
#endif

void initializeMatrices(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                        const Grid& grid) {
    const size_t nodes = grid.rows.nodes;
    const size_t firstRow = grid.rows.begin(grid.row);
    const size_t firstCol = grid.cols.begin(grid.col);
    const size_t rows = grid.rows.size(grid.row);
    const size_t cols = grid.cols.size(grid.col);
#ifdef __GLIBC__
    unsigned int seed = randAdvance(firstRow * nodes + firstCol)(42);
    const auto skip = randAdvance(nodes - cols);
#else
    // On other C libraries obtain starting states by replaying their rand_r stream.
    std::vector<unsigned int> rowSeeds(nodes);
    if (grid.row == 0 && grid.col == 0) {
        unsigned int seed = 42;
        for (auto& start : rowSeeds) {
            start = seed;
            for (size_t j = 0; j < nodes; ++j) {
                rand_r(&seed);
            }
        }
    }
    broadcast(rowSeeds.data(), nodes, 0, grid.comm);
#endif
    for (size_t i = 0; i < rows; ++i) {
        const size_t globalRow = firstRow + i;
#ifndef __GLIBC__
        unsigned int seed = rowSeeds[globalRow];
        for (size_t j = 0; j < firstCol; ++j) {
            rand_r(&seed);
        }
#endif
        for (size_t j = 0; j < cols; ++j) {
            dist[i * cols + j] = 1 + static_cast<unsigned int>(
                static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
            path[i * cols + j] = globalRow;
        }
        if (firstCol <= globalRow && globalRow < firstCol + cols) {
            dist[i * cols + globalRow - firstCol] = 0;
        }
#ifdef __GLIBC__
        seed = skip(seed);
#endif
    }
}

inline void relax(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                  const unsigned int* __restrict__ pivotRow, size_t count,
                  unsigned int toPivot, unsigned int pivot) {
    for (size_t j = 0; j < count; ++j) {
        const unsigned int previous = dist[j];
        const unsigned int candidate = toPivot + pivotRow[j];
        const bool improved = candidate < previous;
        dist[j] = improved ? candidate : previous;
        path[j] = improved ? pivot : path[j];
    }
}

void relaxTile(unsigned int* dist, unsigned int* path, const unsigned int* fromPivot,
               size_t stride, const unsigned int* toPivot, size_t width,
               size_t count, unsigned int first, [[maybe_unused]] const unsigned int* packed) {
#ifdef __AVX2__
    // Track the distances and their last improving pivots in registers across the batch.
    constexpr size_t lanes = 8;
    constexpr size_t vectors = SIMD_COLUMNS / lanes;
    size_t j = 0;
    for (; j + SIMD_COLUMNS <= width; j += SIMD_COLUMNS) {
        __m256i distance[vectors];
        __m256i previousPivot[vectors];
        for (size_t v = 0; v < vectors; ++v) {
            distance[v] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dist + j + v * lanes));
            previousPivot[v] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(path + j + v * lanes));
        }
        for (size_t k = 0; k < count; ++k) {
            const __m256i left = _mm256_set1_epi32(toPivot[k]);
            const __m256i pivot = _mm256_set1_epi32(first + k);
            const auto* row = packed + j * count + k * SIMD_COLUMNS;
            for (size_t v = 0; v < vectors; ++v) {
                const __m256i right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + v * lanes));
                const __m256i candidate = _mm256_add_epi32(left, right);
                // The generated distances and their sums fit in signed 32-bit lanes.
                const __m256i improved = _mm256_cmpgt_epi32(distance[v], candidate);
                previousPivot[v] = _mm256_blendv_epi8(previousPivot[v], pivot, improved);
                distance[v] = _mm256_min_epu32(distance[v], candidate);
            }
        }
        for (size_t v = 0; v < vectors; ++v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dist + j + v * lanes), distance[v]);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(path + j + v * lanes), previousPivot[v]);
        }
    }
    dist += j;
    path += j;
    fromPivot += j;
    width -= j;
#endif
    for (size_t k = 0; k < count; ++k) {
        relax(dist, path, fromPivot + k * stride, width, toPivot[k], first + k);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const Grid& grid) {
    const size_t rows = grid.rows.size(grid.row);
    const size_t cols = grid.cols.size(grid.col);
    const size_t nodes = grid.rows.nodes;
    // Save each pivot's distances as they stood at its original iteration.
    // A zero diagonal keeps the pivot row and column fixed during that iteration.
    // Within a batch the diagonal feeds the two panels, and both panels feed
    // the rest of the matrix. Each cell still sees pivots in ascending order.
    std::vector<unsigned int> toPivot(rows * PIVOT_BATCH);
    std::vector<unsigned int> fromPivot(cols * PIVOT_BATCH);
    std::vector<unsigned int> diagonalTo(PIVOT_BATCH * PIVOT_BATCH);
    std::vector<unsigned int> diagonalFrom(PIVOT_BATCH * PIVOT_BATCH);
    std::vector<unsigned int> packed;
#ifdef __AVX2__
    packed.resize(COLUMN_TILE * PIVOT_BATCH);
#endif

    for (size_t first = 0; first < nodes;) {
        const int ownerRow = grid.rows.owner(first);
        const int ownerCol = grid.cols.owner(first);
        const size_t end = std::min({first + PIVOT_BATCH, grid.rows.begin(ownerRow + 1),
                                     grid.cols.begin(ownerCol + 1)});
        const size_t count = end - first;
        const bool ownsRows = grid.row == ownerRow;
        const bool ownsCols = grid.col == ownerCol;
        const size_t rowStart = ownsRows ? first - grid.rows.begin(grid.row) : 0;
        const size_t colStart = ownsCols ? first - grid.cols.begin(grid.col) : 0;
        const size_t rowEnd = rowStart + (ownsRows ? count : 0);
        const size_t colEnd = colStart + (ownsCols ? count : 0);

        if (ownsRows && ownsCols) {
            for (size_t k = 0; k < count; ++k) {
                auto* right = diagonalFrom.data() + k * count;
                const auto* pivot = dist.data() + (rowStart + k) * cols + colStart;
                std::copy_n(pivot, count, right);
                for (size_t i = 0; i < count; ++i) {
                    const size_t offset = (rowStart + i) * cols + colStart;
                    const unsigned int left = dist[offset + k];
                    diagonalTo[i * count + k] = left;
                    relax(dist.data() + offset, path.data() + offset,
                          right, count, left, first + k);
                }
            }
        }

        MPI_Request diagonalRight = MPI_REQUEST_NULL;
        if (ownsCols && grid.rows.parts > 1) {
            diagonalRight = startBroadcast(diagonalFrom.data(), count * count, ownerRow, grid.colComm);
        }

        if (ownsRows) {
            if (grid.cols.parts > 1) {
                broadcast(diagonalTo.data(), count * count, ownerCol, grid.rowComm);
            }
            for (size_t k = 0; k < count; ++k) {
                auto* right = fromPivot.data() + k * cols;
                const auto* pivot = dist.data() + (rowStart + k) * cols;
                std::copy_n(pivot, cols, right);
                if (ownsCols) {
                    std::copy_n(diagonalFrom.data() + k * count, count, right + colStart);
                }
                for (size_t i = 0; i < count; ++i) {
                    auto* d = dist.data() + (rowStart + i) * cols;
                    auto* p = path.data() + (rowStart + i) * cols;
                    const unsigned int left = diagonalTo[i * count + k];
                    relax(d, p, right, colStart, left, first + k);
                    relax(d + colEnd, p + colEnd, right + colEnd, cols - colEnd, left, first + k);
                }
            }
        }

        if (ownsCols) {
            MPI_Wait(&diagonalRight, MPI_STATUS_IGNORE);
            for (size_t i = 0; i < rows; ++i) {
                auto* left = toPivot.data() + i * count;
                if (rowStart <= i && i < rowEnd) {
                    std::copy_n(diagonalTo.data() + (i - rowStart) * count, count, left);
                    continue;
                }
                const size_t offset = i * cols + colStart;
                auto* d = dist.data() + offset;
                auto* p = path.data() + offset;
                for (size_t k = 0; k < count; ++k) {
                    left[k] = d[k];
                    relax(d, p, diagonalFrom.data() + k * count, count, left[k], first + k);
                }
            }
        }

        MPI_Request panels[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (grid.rows.parts > 1) {
            panels[0] = startBroadcast(fromPivot.data(), count * cols, ownerRow, grid.colComm);
        }
        if (grid.cols.parts > 1) {
            panels[1] = startBroadcast(toPivot.data(), rows * count, ownerCol, grid.rowComm);
        }
        MPI_Waitall(2, panels, MPI_STATUSES_IGNORE);

        // Keep a tile of each row in cache while applying the pivots in order.
        for (size_t j = 0; j < cols;) {
            if (ownsCols && j == colStart) {
                j = colEnd;
                continue;
            }
            const size_t width = std::min(COLUMN_TILE, (j < colStart ? colStart : cols) - j);
#ifdef __AVX2__
            // Pack the pivot rows by SIMD block so successive pivots are adjacent.
            for (size_t x = 0; x + SIMD_COLUMNS <= width; x += SIMD_COLUMNS) {
                for (size_t k = 0; k < count; ++k) {
                    std::copy_n(fromPivot.data() + k * cols + j + x, SIMD_COLUMNS,
                                packed.data() + x * count + k * SIMD_COLUMNS);
                }
            }
#endif
            for (size_t i = 0; i < rows; ++i) {
                if (rowStart <= i && i < rowEnd) {
                    i = rowEnd - 1;
                    continue;
                }
                auto* d = dist.data() + i * cols + j;
                auto* p = path.data() + i * cols + j;
                const auto* left = toPivot.data() + i * count;
                relaxTile(d, p, fromPivot.data() + j, cols, left, width, count, first, packed.data());
            }
            j += width;
        }
        first = end;
    }
}

// Stream blocks to rank zero in global row order, using a strided receive to
// place each column partition directly in the row buffer.
void printDistanceResults(const std::vector<unsigned int>& dist, const Grid& grid) {
    const bool root = grid.row == 0 && grid.col == 0;
    const size_t nodes = grid.rows.nodes;
    const size_t total = nodes * nodes;
    const size_t sampleIndices[] = {0, total / 4, total / 2,
                                    total / 4 * 3 + total % 4 * 3 / 4, total ? total - 1 : 0};
    unsigned int samples[5]{};
    unsigned int minimum = std::numeric_limits<unsigned int>::max();
    unsigned int maximum = 0;
    hash digest;

    const size_t blockRows = std::max<size_t>(1, (256 * 1024) / std::max<size_t>(1, nodes));
    const size_t localCols = grid.cols.size(grid.col);
    std::vector<unsigned int> buffer(root ? std::min(nodes, blockRows) * nodes : 0);
    for (int r = 0; r < grid.rows.parts; ++r) {
        if (!root && r != grid.row) {
            continue;
        }
        const size_t rowCount = grid.rows.size(r);
        for (size_t i = 0; i < rowCount; i += blockRows) {
            const size_t count = std::min(blockRows, rowCount - i);
            if (!root) {
                MPI_Send(dist.data() + i * localCols, count * localCols, MPI_UNSIGNED,
                         0, 0, grid.comm);
                continue;
            }
            for (int c = 0; c < grid.cols.parts; ++c) {
                const size_t firstCol = grid.cols.begin(c);
                const size_t cols = grid.cols.size(c);
                if (r == 0 && c == 0) {
                    for (size_t j = 0; j < count; ++j) {
                        std::copy_n(dist.data() + (i + j) * cols, cols, buffer.data() + j * nodes);
                    }
                } else {
                    MPI_Datatype slice;
                    MPI_Type_vector(count, cols, nodes, MPI_UNSIGNED, &slice);
                    MPI_Type_commit(&slice);
                    MPI_Recv(buffer.data() + firstCol, 1, slice, grid.rank(r, c), 0,
                             grid.comm, MPI_STATUS_IGNORE);
                    MPI_Type_free(&slice);
                }
            }

            const size_t offset = (grid.rows.begin(r) + i) * nodes;
            const size_t size = count * nodes;
            for (size_t j = 0; j < size; ++j) {
                const unsigned int value = buffer[j];
                digest.add(value);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            for (size_t s = 0; s < 5; ++s) {
                if (offset <= sampleIndices[s] && sampleIndices[s] < offset + size) {
                    samples[s] = buffer[sampleIndices[s] - offset];
                }
            }
        }
    }

    if (root) {
        printf("=== RESULTS ===\n");
        printf("Name: DistanceMatrix\n");
        printf("Elements: %zu\n", total);
        if (total > 0) {
            printf("Min: %u\n", minimum);
            printf("Max: %u\n", maximum);
            for (size_t s = 0; s < 5; ++s) {
                printf("Sample[%zu]: %u\n", sampleIndices[s], samples[s]);
            }
            printf("Hash: %016" PRIx64 "\n", digest.get());
        }
        printf("=== END RESULTS ===\n");
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const Grid& grid) {
    const size_t nodes = grid.rows.nodes;
    const size_t firstRow = grid.rows.begin(grid.row);
    const size_t lastRow = grid.rows.begin(grid.row + 1);
    const size_t firstCol = grid.cols.begin(grid.col);
    const size_t lastCol = grid.cols.begin(grid.col + 1);
    const size_t cols = lastCol - firstCol;
    const bool root = grid.row == 0 && grid.col == 0;

    unsigned int badDiagonal = nodes;
    for (size_t i = std::max(firstRow, firstCol); i < std::min(lastRow, lastCol); ++i) {
        if (dist[(i - firstRow) * cols + i - firstCol] != 0) {
            badDiagonal = i;
            break;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &badDiagonal, 1, MPI_UNSIGNED, MPI_MIN, grid.comm);
    if (badDiagonal != nodes) {
        if (root) {
            printf("Validation failed: diagonal element [%u,%u] is not zero\n", badDiagonal, badDiagonal);
        }
        return false;
    }

    // Only the first ten rows and columns are needed for the triangle checks.
    const size_t sample = std::min<size_t>(nodes, 10);
    std::vector<unsigned int> edges(2 * sample * nodes, 0);
    for (size_t i = firstRow; i < std::min(lastRow, sample); ++i) {
        std::copy_n(dist.data() + (i - firstRow) * cols, cols,
                    edges.data() + i * nodes + firstCol);
    }
    for (size_t i = firstRow; i < lastRow; ++i) {
        for (size_t j = firstCol; j < std::min(lastCol, sample); ++j) {
            edges[sample * nodes + j * nodes + i] = dist[(i - firstRow) * cols + j - firstCol];
        }
    }
    for (size_t offset = 0; offset < edges.size();) {
        const int chunk = std::min(edges.size() - offset, static_cast<size_t>(INT_MAX));
        auto* data = edges.data() + offset;
        MPI_Reduce(root ? MPI_IN_PLACE : data, root ? data : nullptr, chunk,
                   MPI_UNSIGNED, MPI_MAX, 0, grid.comm);
        offset += chunk;
    }

    int valid = 1;
    if (root) {
        for (size_t i = 0; valid && i < sample; ++i) {
            for (size_t j = 0; valid && j < sample; ++j) {
                const unsigned int direct = edges[i * nodes + j];
                const auto* fromSource = edges.data() + i * nodes;
                const auto* toDestination = edges.data() + (sample + j) * nodes;
                for (size_t k = 0; k < nodes; ++k) {
                    if (fromSource[k] < INF && toDestination[k] < INF &&
                        fromSource[k] + toDestination[k] < direct) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        valid = 0;
                        break;
                    }
                }
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, grid.comm);
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

int runBenchmark(int argc, char** argv) {
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = worldRank == 0;
    int numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numNodes < 0) {
        if (root) {
            fprintf(stderr, "Number of nodes must be nonnegative\n");
        }
        return 1;
    }
    const size_t nodes = numNodes;
    int dims[2]{};
    MPI_Dims_create(worldSize, 2, dims);
    dims[0] = std::min(dims[0], std::max(numNodes, 1));
    dims[1] = std::min(dims[1], std::max(numNodes, 1));
    const bool active = worldRank < dims[0] * dims[1];
    MPI_Comm comm = MPI_COMM_WORLD;
    if (dims[0] * dims[1] != worldSize) {
        MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    }

    int result = 0;
    if (active) {
        Grid grid{{nodes, dims[0]}, {nodes, dims[1]}, worldRank / dims[1], worldRank % dims[1],
                  comm, MPI_COMM_NULL, MPI_COMM_NULL};
        MPI_Comm_split(comm, grid.row, grid.col, &grid.rowComm);
        MPI_Comm_split(comm, grid.col, grid.row, &grid.colComm);

        if (root) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", nodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        const size_t elements = grid.rows.size(grid.row) * grid.cols.size(grid.col);
        std::vector<unsigned int> dist(elements);
        std::vector<unsigned int> path(elements);
        initializeMatrices(dist, path, grid);

        if (root) {
            printf("Computing shortest paths...\n");
        }
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, grid);
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (root) {
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
            const double operations = static_cast<double>(nodes) * nodes * nodes;
            printf("Performance: %.3f GOPS\n", seconds > 0 ? operations / seconds / 1e9 : 0.0);
        }

        if (printResults) {
            printDistanceResults(dist, grid);
        }
        if (validate) {
            if (root) {
                printf("Validating result...\n");
            }
            const bool valid = validateResult(dist, grid);
            if (root) {
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            result = valid ? 0 : 1;
        }

        MPI_Comm_free(&grid.rowComm);
        MPI_Comm_free(&grid.colComm);
        if (comm != MPI_COMM_WORLD) {
            MPI_Comm_free(&comm);
        }
    }

    if (dims[0] * dims[1] != worldSize) {
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int status = runBenchmark(argc, argv);
    MPI_Finalize();
    return status;
}
