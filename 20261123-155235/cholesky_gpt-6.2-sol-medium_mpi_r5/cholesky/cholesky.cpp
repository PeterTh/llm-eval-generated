#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/hash.hpp"

namespace {

constexpr int tileSize = 64;
constexpr int tileElements = tileSize * tileSize;

int localCount(int end, int coordinate, int stride) {
    return end <= coordinate ? 0 : 1 + (end - 1 - coordinate) / stride;
}

struct ProcessGrid {
    int rank;
    int rows;
    int cols;
    int row;
    int col;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    ProcessGrid() {
        int size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int dims[2] = {};
        MPI_Dims_create(size, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }

    ~ProcessGrid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
};

// Only the lower triangle of tiles is stored. Each tile is column-major.
struct Matrix {
    int order;
    int tiles;
    int localRows;
    int localCols;
    const ProcessGrid& grid;
    std::vector<size_t> offsets;
    std::vector<double> values;

    Matrix(int n, const ProcessGrid& g)
        : order(n), tiles(n / tileSize + (n % tileSize != 0)),
          localRows(localCount(tiles, g.row, g.rows)),
          localCols(localCount(tiles, g.col, g.cols)), grid(g),
          offsets(localRows + 1) {
        for (int i = 0; i < localRows; ++i) {
            offsets[i + 1] = offsets[i] + localCount(globalRow(i) + 1, g.col, g.cols);
        }
        values.resize(offsets.back() * tileElements);
    }

    int globalRow(int i) const { return grid.row + i * grid.rows; }
    int globalCol(int j) const { return grid.col + j * grid.cols; }
    int extent(int tile) const { return std::min(tileSize, order - tile * tileSize); }
    int columnsInRow(int i) const { return static_cast<int>(offsets[i + 1] - offsets[i]); }

    double* tile(int i, int j) { return values.data() + (offsets[i] + j) * tileElements; }
    const double* tile(int i, int j) const {
        return values.data() + (offsets[i] + j) * tileElements;
    }
};

// Update C by a panel product X*Y^T. A microtile keeps four output columns
// in registers while the innermost loop operates on contiguous rows.
template <bool subtract>
void multiplyTile(double* c, const double* x, const double* y, int height, int width,
                  int depth) {
    constexpr int microRows = 8;
    constexpr int microCols = 4;
    for (int j = 0; j < width; j += microCols) {
        for (int i = 0; i < height; i += microRows) {
            double accum[microCols][microRows];
            for (int col = 0; col < microCols; ++col) {
                std::copy_n(c + (j + col) * tileSize + i, microRows, accum[col]);
            }
            for (int k = 0; k < depth; ++k) {
                const double* a = x + k * tileSize + i;
                const double* b = y + k * tileSize + j;
                for (int col = 0; col < microCols; ++col) {
                    for (int row = 0; row < microRows; ++row) {
                        if constexpr (subtract) {
                            accum[col][row] -= a[row] * b[col];
                        } else {
                            accum[col][row] += a[row] * b[col];
                        }
                    }
                }
            }
            for (int col = 0; col < microCols; ++col) {
                std::copy_n(accum[col], microRows, c + (j + col) * tileSize + i);
            }
        }
    }
}

std::vector<unsigned int> randomRowSeeds(int n, int rank) {
    std::vector<unsigned int> seeds(n);
#if defined(__GLIBC__)
    // glibc rand_r advances a 32-bit LCG three times per sample. Compose
    // the affine transition by squaring to jump directly to the next row.
    (void)rank;
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;
    uint32_t rowMultiplier = 1;
    uint32_t rowIncrement = 0;
    for (uint64_t steps = 3 * static_cast<uint64_t>(n); steps > 0; steps >>= 1) {
        if (steps & 1) {
            rowMultiplier *= multiplier;
            rowIncrement = rowIncrement * multiplier + increment;
        }
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
    uint32_t seed = 42;
    for (auto& rowSeed : seeds) {
        rowSeed = seed;
        seed = seed * rowMultiplier + rowIncrement;
    }
#else
    // Obtain the same rand_r stream on libcs with a different generator.
    if (rank == 0) {
        unsigned int seed = 42;
        for (auto& rowSeed : seeds) {
            rowSeed = seed;
            for (int k = 0; k < n; ++k) {
                rand_r(&seed);
            }
        }
    }
    MPI_Bcast(seeds.data(), n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    return seeds;
}

void randomPanel(const Matrix& matrix, std::vector<double>& panel,
                 std::vector<unsigned int>& seeds, int coordinate, int stride, int depth) {
    const int blocks = static_cast<int>(panel.size() / tileElements);
    for (int block = 0; block < blocks; ++block) {
        const int global = coordinate + block * stride;
        double* tile = panel.data() + static_cast<size_t>(block) * tileElements;
        for (int i = 0; i < matrix.extent(global); ++i) {
            auto& seed = seeds[static_cast<size_t>(block) * tileSize + i];
            for (int k = 0; k < depth; ++k) {
                tile[k * tileSize + i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            }
        }
    }
}

void generatePositiveDefiniteMatrix(Matrix& matrix) {
    const auto& grid = matrix.grid;
    const auto seeds = randomRowSeeds(matrix.order, grid.rank);
    if (matrix.values.empty()) {
        return;
    }

    const bool samePanels = grid.rows == grid.cols && grid.row == grid.col;
    std::vector<double> left(static_cast<size_t>(matrix.localRows) * tileElements);
    std::vector<double> right(samePanels ? 0 : static_cast<size_t>(matrix.localCols) * tileElements);
    std::vector<unsigned int> leftSeeds(static_cast<size_t>(matrix.localRows) * tileSize);
    std::vector<unsigned int> rightSeeds(samePanels ? 0 : static_cast<size_t>(matrix.localCols) * tileSize);
    for (int i = 0; i < matrix.localRows; ++i) {
        const int row = matrix.globalRow(i);
        std::copy_n(seeds.data() + row * tileSize, matrix.extent(row),
                    leftSeeds.data() + static_cast<size_t>(i) * tileSize);
    }
    if (!samePanels) {
        for (int j = 0; j < matrix.localCols; ++j) {
            const int col = matrix.globalCol(j);
            std::copy_n(seeds.data() + col * tileSize, matrix.extent(col),
                        rightSeeds.data() + static_cast<size_t>(j) * tileSize);
        }
    }

    const double* columnPanel = samePanels ? left.data() : right.data();
    for (int k = 0; k < matrix.tiles; ++k) {
        const int depth = matrix.extent(k);
        randomPanel(matrix, left, leftSeeds, grid.row, grid.rows, depth);
        if (!samePanels) {
            randomPanel(matrix, right, rightSeeds, grid.col, grid.cols, depth);
        }
        for (int i = 0; i < matrix.localRows; ++i) {
            const double* x = left.data() + static_cast<size_t>(i) * tileElements;
            const int height = matrix.extent(matrix.globalRow(i));
            for (int j = 0; j < matrix.columnsInRow(i); ++j) {
                const double* y = columnPanel + static_cast<size_t>(j) * tileElements;
                multiplyTile<false>(matrix.tile(i, j), x, y, height,
                                    matrix.extent(matrix.globalCol(j)), depth);
            }
        }
    }
    for (int i = 0; i < matrix.localRows; ++i) {
        const int row = matrix.globalRow(i);
        if (row % grid.cols == grid.col) {
            double* diagonal = matrix.tile(i, row / grid.cols);
            for (int j = 0; j < matrix.extent(row); ++j) {
                diagonal[j * tileSize + j] += matrix.order;
            }
        }
    }
}

// Share the column of tiles L(:,k) across process rows and then down process
// columns. Grouping by source process row gives a single column broadcast on
// square grids and independent packed broadcasts on rectangular grids.
class Panel {
public:
    explicit Panel(const Matrix& matrix)
        : matrix(matrix), square(matrix.grid.rows == matrix.grid.cols),
          direct(square && matrix.grid.row == matrix.grid.col),
          left(1 + static_cast<size_t>(matrix.localRows) * tileElements),
          right(direct ? 0 : static_cast<size_t>(matrix.localCols) * tileElements),
          columnSlots(matrix.localCols), groups(matrix.grid.rows),
          requests(matrix.grid.rows) {
        if (square) {
            return;
        }
        for (int j = 0; j < matrix.localCols; ++j) {
            groups[matrix.globalCol(j) % matrix.grid.rows].push_back(j);
        }
        int slot = 0;
        for (const auto& group : groups) {
            for (int j : group) {
                columnSlots[j] = slot++;
            }
        }
    }

    double* rowTile(int i) { return left.data() + 1 + static_cast<size_t>(i) * tileElements; }
    const double* colTile(int j) const {
        if (direct) {
            return left.data() + 1 + static_cast<size_t>(j) * tileElements;
        }
        return right.data() + static_cast<size_t>(square ? j : columnSlots[j]) * tileElements;
    }

    int shareRows(int firstTile, int root, int failure) {
        const auto& grid = matrix.grid;
        const int firstRow = localCount(firstTile, grid.row, grid.rows);
        // Prefix the active tile suffix with the failing pivot, if any.
        double* buffer = rowTile(firstRow) - 1;
        if (grid.col == root) {
            buffer[0] = failure;
        }
        MPI_Bcast(buffer, 1 + (matrix.localRows - firstRow) * tileElements, MPI_DOUBLE,
                  root, grid.rowComm);
        return static_cast<int>(buffer[0]);
    }

    void shareColumns(int firstCol) {
        const auto& grid = matrix.grid;
        if (firstCol == matrix.localCols || grid.rows == 1) {
            return;
        }
        if (square) {
            double* buffer = direct ? rowTile(firstCol)
                                    : right.data() + static_cast<size_t>(firstCol) * tileElements;
            MPI_Bcast(buffer, (matrix.localCols - firstCol) * tileElements, MPI_DOUBLE,
                      grid.col, grid.colComm);
            return;
        }
        int count = 0;
        for (int root = 0; root < grid.rows; ++root) {
            const auto& group = groups[root];
            auto begin = std::lower_bound(group.begin(), group.end(), firstCol);
            if (begin == group.end()) {
                continue;
            }
            double* buffer = right.data() + static_cast<size_t>(columnSlots[*begin]) * tileElements;
            if (grid.row == root) {
                double* dest = buffer;
                for (auto j = begin; j != group.end(); ++j, dest += tileElements) {
                    const int global = matrix.globalCol(*j);
                    std::copy_n(rowTile(global / grid.rows), tileElements, dest);
                }
            }
            MPI_Ibcast(buffer, static_cast<int>(group.end() - begin) * tileElements,
                       MPI_DOUBLE, root, grid.colComm, &requests[count++]);
        }
        MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
    }

private:
    const Matrix& matrix;
    bool square;
    bool direct;
    std::vector<double> left;
    std::vector<double> right;
    std::vector<int> columnSlots;
    std::vector<std::vector<int>> groups;
    std::vector<MPI_Request> requests;
};

template <bool subtract>
void updateTrailingTiles(Matrix& matrix, Panel& panel, int firstRow, int firstCol, int depth) {
    for (int i = firstRow; i < matrix.localRows; ++i) {
        const int height = matrix.extent(matrix.globalRow(i));
        const double* x = panel.rowTile(i);
        for (int j = firstCol; j < matrix.columnsInRow(i); ++j) {
            multiplyTile<subtract>(matrix.tile(i, j), x, panel.colTile(j), height,
                                   matrix.extent(matrix.globalCol(j)), depth);
        }
    }
}

int factorDiagonal(double* tile, int size) {
    for (int j = 0; j < size; ++j) {
        double* column = tile + j * tileSize;
        if (column[j] <= 0.0) {
            return j;
        }
        column[j] = std::sqrt(column[j]);
        for (int i = j + 1; i < size; ++i) {
            column[i] /= column[j];
        }
        for (int k = j + 1; k < size; ++k) {
            double* next = tile + k * tileSize;
            const double multiplier = column[k];
            for (int i = k; i < size; ++i) {
                next[i] -= column[i] * multiplier;
            }
        }
        std::fill_n(column, j, 0.0);
    }
    return -1;
}

void solvePanelTile(double* tile, const double* diagonal, int height, int width) {
    for (int j = 0; j < width; ++j) {
        const double* factor = diagonal + j * tileSize;
        double* column = tile + j * tileSize;
        for (int i = 0; i < height; ++i) {
            column[i] /= factor[j];
        }
        for (int k = j + 1; k < width; ++k) {
            double* next = tile + k * tileSize;
            const double multiplier = factor[k];
            for (int i = 0; i < height; ++i) {
                next[i] -= column[i] * multiplier;
            }
        }
    }
}

int choleskyDecomposition(Matrix& matrix, Panel& panel) {
    const auto& grid = matrix.grid;
    std::array<double, tileElements + 1> diagonal;
    for (int k = 0; k < matrix.tiles; ++k) {
        const int depth = matrix.extent(k);
        const int firstRow = localCount(k + 1, grid.row, grid.rows);
        const int firstCol = localCount(k + 1, grid.col, grid.cols);
        const int rootRow = k % grid.rows;
        const int rootCol = k % grid.cols;
        int failure = -1;

        if (grid.col == rootCol) {
            const int localCol = k / grid.cols;
            if (grid.row == rootRow) {
                double* tile = matrix.tile(k / grid.rows, localCol);
                const int pivot = factorDiagonal(tile, depth);
                diagonal[0] = pivot < 0 ? -1 : k * tileSize + pivot;
                std::copy_n(tile, tileElements, diagonal.data() + 1);
            }
            MPI_Bcast(diagonal.data(), tileElements + 1, MPI_DOUBLE, rootRow, grid.colComm);
            failure = static_cast<int>(diagonal[0]);
            if (failure < 0) {
                for (int i = firstRow; i < matrix.localRows; ++i) {
                    double* tile = matrix.tile(i, localCol);
                    solvePanelTile(tile, diagonal.data() + 1, matrix.extent(matrix.globalRow(i)), depth);
                    std::copy_n(tile, tileElements, panel.rowTile(i));
                }
            }
        }

        failure = panel.shareRows(k + 1, rootCol, failure);
        if (failure >= 0) {
            return failure;
        }
        panel.shareColumns(firstCol);
        updateTrailingTiles<true>(matrix, panel, firstRow, firstCol, depth);
    }
    return -1;
}

bool validateCholesky(const Matrix& matrix, const Matrix& original, Panel& panel) {
    const auto& grid = matrix.grid;
    Matrix reconstructed(matrix.order, grid);
    for (int k = 0; k < matrix.tiles; ++k) {
        const int firstRow = localCount(k, grid.row, grid.rows);
        const int firstCol = localCount(k, grid.col, grid.cols);
        const int root = k % grid.cols;
        if (grid.col == root) {
            for (int i = firstRow; i < matrix.localRows; ++i) {
                std::copy_n(matrix.tile(i, k / grid.cols), tileElements, panel.rowTile(i));
            }
        }
        panel.shareRows(k, root, -1);
        panel.shareColumns(firstCol);
        updateTrailingTiles<false>(reconstructed, panel, firstRow, firstCol, matrix.extent(k));
    }

    double localErrors[2] = {};
    for (int i = 0; i < matrix.localRows; ++i) {
        const int row = matrix.globalRow(i);
        for (int j = 0; j < matrix.columnsInRow(i); ++j) {
            const int col = matrix.globalCol(j);
            const double* input = original.tile(i, j);
            const double* actual = reconstructed.tile(i, j);
            for (int c = 0; c < matrix.extent(col); ++c) {
                for (int r = row == col ? c : 0; r < matrix.extent(row); ++r) {
                    const int index = c * tileSize + r;
                    const double error = std::abs(actual[index] - input[index]);
                    localErrors[0] = std::max(localErrors[0], error);
                    localErrors[1] = std::max(localErrors[1], error / (std::abs(input[index]) + 1e-10));
                }
            }
        }
    }
    double errors[2];
    MPI_Allreduce(localErrors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (grid.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) {
            printf("Validation failed: relative error too large\n");
        }
    }
    return errors[1] <= 1e-6;
}

void printFactor(const Matrix& matrix) {
    const auto& grid = matrix.grid;
    if (grid.rank != 0) {
        for (int i = 0; i < matrix.localRows; ++i) {
            const int count = matrix.columnsInRow(i) * tileElements;
            if (count > 0) {
                MPI_Send(matrix.tile(i, 0), count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
        return;
    }

    const size_t n = matrix.order;
    const size_t elements = n * n;
    printf("=== RESULTS ===\n");
    printf("Name: CholeskyL\n");
    printf("Elements: %zu\n", elements);
    if (elements == 0) {
        printf("=== END RESULTS ===\n");
        return;
    }

    // The digest and compensated sum depend on row-major order. Stream one
    // tile row at a time so reporting does not gather a dense matrix on rank 0.
    const std::array<size_t, 5> indices = {0, elements / 4, elements / 2, 3 * elements / 4, elements - 1};
    std::array<double, 5> samples = {};
    double sum = 0.0;
    double correction = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -minimum;
    hash digest;
    size_t position = 0;
    size_t sample = 0;
    std::vector<double> result(n * matrix.extent(0));
    std::vector<double> received(static_cast<size_t>(matrix.localCols) * tileElements);
    for (int i = 0; i < matrix.tiles; ++i) {
        const int height = matrix.extent(i);
        const size_t stripeSize = n * height;
        std::fill_n(result.data(), stripeSize, 0.0);
        for (int col = 0; col < grid.cols && col <= i; ++col) {
            const int source = (i % grid.rows) * grid.cols + col;
            const int count = localCount(i + 1, col, grid.cols);
            const double* tiles = received.data();
            if (source == 0) {
                tiles = matrix.tile(i / grid.rows, 0);
            } else {
                MPI_Recv(received.data(), count * tileElements, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int j = 0; j < count; ++j) {
                const int globalCol = col + j * grid.cols;
                for (int r = 0; r < height; ++r) {
                    double* output = result.data() + r * n + globalCol * tileSize;
                    const int width = i == globalCol ? r + 1 : matrix.extent(globalCol);
                    for (int c = 0; c < width; ++c) {
                        output[c] = tiles[static_cast<size_t>(j) * tileElements + c * tileSize + r];
                    }
                }
            }
        }
        for (size_t j = 0; j < stripeSize; ++j, ++position) {
            const double value = result[j];
            const double adjusted = value - correction;
            const double next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            digest.add(value);
            while (sample < indices.size() && position == indices[sample]) {
                samples[sample++] = value;
            }
        }
    }
    printf("Sum: %.17e\n", sum);
    printf("Min: %.17e\n", minimum);
    printf("Max: %.17e\n", maximum);
    for (size_t i = 0; i < indices.size(); ++i) {
        printf("Sample[%zu]: %.17e\n", indices[i], samples[i]);
    }
    printf("Hash: %016" PRIx64 "\n", digest.get());
    printf("=== END RESULTS ===\n");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
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
    if (n < 0) {
        if (rank == 0) {
            printf("Error: Matrix size must be nonnegative\n");
        }
        return 1;
    }

    const ProcessGrid grid;
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix matrix(n, grid);
    generatePositiveDefiniteMatrix(matrix);
    Matrix original(validate ? n : 0, grid);
    if (validate) {
        original.values = matrix.values;
    }
    Panel panel(matrix);

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int failedPivot = choleskyDecomposition(matrix, panel);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (failedPivot >= 0) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failedPivot);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) {
        printFactor(matrix);
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateCholesky(matrix, original, panel);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int result = runBenchmark(argc, argv);
    MPI_Finalize();
    return result;
}
