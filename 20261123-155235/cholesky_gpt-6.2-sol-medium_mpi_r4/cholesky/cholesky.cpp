#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

constexpr int maxBlockSize = 64;

int firstOwned(int first, int coordinate, int stride) {
    return first + (coordinate + stride - first % stride) % stride;
}

struct Tile {
    int row;
    int col;
    size_t offset;
};

struct Matrix {
    int n;
    int rank;
    int rows;
    int cols;
    int row;
    int col;
    int block;
    int blocks;
    int tileSize;
    MPI_Comm rowComm;
    MPI_Comm colComm;
    std::vector<Tile> tiles;
    std::vector<size_t> columnStarts;
    std::vector<double> values;

    Matrix(int order, int worldRank, int worldSize) : n(order), rank(worldRank) {
        int dims[2] = {};
        MPI_Dims_create(worldSize, 2, dims);
        rows = dims[0];
        cols = dims[1];

        // Keep several tiles per process coordinate where possible.
        block = maxBlockSize;
        while (block > 8 && n / block < 4LL * rows) {
            block /= 2;
        }
        blocks = n / block + (n % block != 0);
        tileSize = block * block;
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);

        for (int j = col; j < blocks; j += cols) {
            columnStarts.push_back(tiles.size());
            for (int i = firstOwned(j, row, rows); i < blocks; i += rows) {
                tiles.push_back({i, j, tiles.size() * tileSize});
            }
        }
        columnStarts.push_back(tiles.size());
        values.resize(tiles.size() * tileSize);
    }

    ~Matrix() {
        MPI_Comm_free(&colComm);
        MPI_Comm_free(&rowComm);
    }

    int extent(int index) const {
        return std::min(block, n - index * block);
    }

    int owner(int i, int j) const {
        return (i % rows) * cols + j % cols;
    }

    std::span<const Tile> column(int j) const {
        const int localCol = j / cols;
        return std::span(tiles).subspan(columnStarts[localCol],
                                        columnStarts[localCol + 1] - columnStarts[localCol]);
    }

    std::span<const Tile> trailing(int first) const {
        const int localCol = first / cols + (first % cols > col);
        return std::span(tiles).subspan(columnStarts[localCol]);
    }
};

#if defined(__AVX__)
using RowVector = double __attribute__((vector_size(32)));
#else
using RowVector = double __attribute__((vector_size(16)));
#endif

// Column-major tiles let each vector lane accumulate a different element.
// The inner dimension advances in order, including across successive updates.
void addProduct(double* result, const double* left, const double* right,
                int block, int inner, bool diagonal) {
    constexpr int rowStrip = 8;
    constexpr int colStrip = 4;
    constexpr int vectors = rowStrip * sizeof(double) / sizeof(RowVector);

    for (int j = 0; j < block; j += colStrip) {
        const int firstRow = diagonal ? j / rowStrip * rowStrip : 0;
        for (int i = firstRow; i < block; i += rowStrip) {
            RowVector sums[colStrip][vectors];
            for (int c = 0; c < colStrip; ++c) {
                std::memcpy(sums[c], result + (j + c) * block + i, sizeof(sums[c]));
            }
            for (int k = 0; k < inner; ++k) {
                RowVector a[vectors];
                std::memcpy(a, left + k * block + i, sizeof(a));
                const double* b = right + k * block + j;
                for (int c = 0; c < colStrip; ++c) {
                    for (int r = 0; r < vectors; ++r) {
                        sums[c][r] += a[r] * b[c];
                    }
                }
            }
            for (int c = 0; c < colStrip; ++c) {
                std::memcpy(result + (j + c) * block + i, sums[c], sizeof(sums[c]));
            }
        }
    }
}

void skipRandom(unsigned int& seed, size_t count) {
#if defined(__GLIBC__)
    // glibc's rand_r advances this 32-bit LCG three times per output.
    constexpr std::uint32_t multiplier = 1103515245u;
    constexpr std::uint32_t increment = 12345u;
    std::uint32_t scale = multiplier * multiplier * multiplier;
    std::uint32_t offset = (multiplier * multiplier + multiplier + 1u) * increment;
    std::uint32_t state = seed;
    while (count != 0) {
        if (count & 1) {
            state = scale * state + offset;
        }
        offset = scale * offset + offset;
        scale *= scale;
        count >>= 1;
    }
    seed = state;
#else
    for (size_t i = 0; i < count; ++i) {
        rand_r(&seed);
    }
#endif
}

void generatePositiveDefiniteMatrix(Matrix& matrix) {
    const int block = matrix.block;
    std::vector<int> needed;
    std::vector<size_t> offsets(matrix.blocks);
    for (const Tile& tile : matrix.tiles) {
        offsets[tile.row] = 1;
        offsets[tile.col] = 1;
    }
    for (int i = 0; i < matrix.blocks; ++i) {
        if (offsets[i] != 0) {
            offsets[i] = needed.size() * matrix.tileSize;
            needed.push_back(i);
        }
    }

    // Maintain each needed row's place in the original row-major random stream.
    std::vector<unsigned int> seeds(needed.size() * block);
    unsigned int seed = 42;
    size_t previous = 0;
    for (size_t p = 0; p < needed.size(); ++p) {
        for (int i = 0; i < matrix.extent(needed[p]); ++i) {
            const size_t position = (static_cast<size_t>(needed[p]) * block + i) * matrix.n;
            skipRandom(seed, position - previous);
            seeds[p * block + i] = seed;
            previous = position;
        }
    }

    // Build A = B B^T + n I with just one slice of the needed rows of B in memory.
    std::vector<double> randomTiles(needed.size() * matrix.tileSize);
    std::array<double, maxBlockSize> randomRow;
    for (int band = 0; band < matrix.blocks; ++band) {
        const int inner = matrix.extent(band);
        for (size_t p = 0; p < needed.size(); ++p) {
            double* data = randomTiles.data() + p * matrix.tileSize;
            for (int i = 0; i < matrix.extent(needed[p]); ++i) {
                unsigned int& state = seeds[p * block + i];
                for (int k = 0; k < inner; ++k) {
                    randomRow[k] = (rand_r(&state) / static_cast<double>(RAND_MAX)) - 0.5;
                }
                for (int k = 0; k < inner; ++k) {
                    data[k * block + i] = randomRow[k];
                }
            }
        }
        for (const Tile& tile : matrix.tiles) {
            addProduct(matrix.values.data() + tile.offset,
                       randomTiles.data() + offsets[tile.row],
                       randomTiles.data() + offsets[tile.col], block, inner, tile.row == tile.col);
        }
    }
    for (const Tile& tile : matrix.tiles) {
        if (tile.row == tile.col) {
            for (int i = 0; i < matrix.extent(tile.row); ++i) {
                matrix.values[tile.offset + i * (block + 1)] += matrix.n;
            }
        }
    }
}

// Exchange the active tiles in block column k. Each process row broadcasts
// its tiles, then each process column obtains the tiles needed as right operands.
class Panel {
public:
    explicit Panel(const Matrix& matrix)
        : matrix(matrix), rowData(1 + capacity(matrix.rows)),
          colData(capacity(matrix.cols)), colOffsets(matrix.blocks),
          counts(matrix.rows), displacements(matrix.rows) {}

    void setRange(int first) {
        firstRow = firstOwned(first, matrix.row, matrix.rows);
        firstCol = firstOwned(first, matrix.col, matrix.cols);
        rowCount = countFrom(firstRow, matrix.rows);
        colCount = countFrom(firstCol, matrix.cols);
    }

    double* rowTile(int i) {
        return rowData.data() + 1 + static_cast<size_t>((i - firstRow) / matrix.rows) * matrix.tileSize;
    }

    const double* colTile(int j) const {
        if (matrix.rows == 1 || (matrix.rows == matrix.cols && matrix.row == matrix.col)) {
            return rowData.data() + 1 + static_cast<size_t>((j - firstCol) / matrix.cols) * matrix.tileSize;
        }
        const size_t offset = matrix.rows == matrix.cols
            ? static_cast<size_t>((j - firstCol) / matrix.cols) * matrix.tileSize
            : colOffsets[j];
        return colData.data() + offset;
    }

    MPI_Request broadcastRows(int root, int failure) {
        // Carry pivot failure with the panel so every rank can exit together.
        rowData[0] = failure;
        MPI_Request request = MPI_REQUEST_NULL;
        if (matrix.cols > 1) {
            MPI_Ibcast(rowData.data(), 1 + rowCount * matrix.tileSize, MPI_DOUBLE, root, matrix.rowComm, &request);
        }
        return request;
    }

    int finishRows(MPI_Request request) const {
        if (request != MPI_REQUEST_NULL) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        return static_cast<int>(rowData[0]);
    }

    void exchangeColumns() {
        if (colCount == 0 || matrix.rows == 1) {
            return;
        }
        if (matrix.rows == matrix.cols) {
            double* buffer = matrix.row == matrix.col ? rowData.data() + 1 : colData.data();
            MPI_Bcast(buffer, colCount * matrix.tileSize, MPI_DOUBLE, matrix.col, matrix.colComm);
            return;
        }

        // A rectangular grid can obtain a column's tiles from several process rows.
        std::fill(counts.begin(), counts.end(), 0);
        for (int j = firstCol; j < matrix.blocks; j += matrix.cols) {
            counts[j % matrix.rows] += matrix.tileSize;
        }
        int offset = 0;
        for (int r = 0; r < matrix.rows; ++r) {
            displacements[r] = offset;
            offset += counts[r];
        }
        std::fill(counts.begin(), counts.end(), 0);
        for (int j = firstCol; j < matrix.blocks; j += matrix.cols) {
            const int r = j % matrix.rows;
            colOffsets[j] = displacements[r] + counts[r];
            if (r == matrix.row) {
                std::copy_n(rowTile(j), matrix.tileSize, colData.data() + colOffsets[j]);
            }
            counts[r] += matrix.tileSize;
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, colData.data(),
                       counts.data(), displacements.data(), MPI_DOUBLE, matrix.colComm);
    }

private:
    const Matrix& matrix;
    std::vector<double> rowData;
    std::vector<double> colData;
    std::vector<size_t> colOffsets;
    std::vector<int> counts;
    std::vector<int> displacements;
    int firstRow = 0;
    int firstCol = 0;
    int rowCount = 0;
    int colCount = 0;

    size_t capacity(int stride) const {
        return static_cast<size_t>(countFrom(0, stride)) * matrix.tileSize;
    }

    int countFrom(int first, int stride) const {
        return first < matrix.blocks ? 1 + (matrix.blocks - 1 - first) / stride : 0;
    }
};

void addPanelProducts(const Matrix& matrix, std::vector<double>& sums, std::span<const Tile> tiles,
                      Panel& panel, int width) {
    for (const Tile& tile : tiles) {
        addProduct(sums.data() + tile.offset, panel.rowTile(tile.row), panel.colTile(tile.col),
                   matrix.block, width, tile.row == tile.col);
    }
}

int factorDiagonal(double* tile, double* sums, int block, int width) {
    for (int j = 0; j < width; ++j) {
        double* column = tile + j * block;
        const double* sum = sums + j * block;
        const double value = column[j] - sum[j];
        if (value <= 0.0) {
            return j;
        }
        column[j] = std::sqrt(value);
        std::fill_n(column, j, 0.0);
        for (int i = j + 1; i < width; ++i) {
            column[i] = (column[i] - sum[i]) / column[j];
        }
        for (int c = j + 1; c < width; ++c) {
            double* next = sums + c * block;
            for (int i = c; i < width; ++i) {
                next[i] += column[i] * column[c];
            }
        }
    }
    return -1;
}

void solvePanelTile(double* tile, double* sums, const double* diagonal,
                    int block, int height, int width) {
    for (int j = 0; j < width; ++j) {
        double* column = tile + j * block;
        const double* sum = sums + j * block;
        const double* pivot = diagonal + j * block;
        for (int i = 0; i < height; ++i) {
            column[i] = (column[i] - sum[i]) / pivot[j];
        }
        for (int c = j + 1; c < width; ++c) {
            double* next = sums + c * block;
            for (int i = 0; i < height; ++i) {
                next[i] += column[i] * pivot[c];
            }
        }
    }
}

int choleskyDecomposition(Matrix& matrix, std::vector<double>& sums, Panel& first, Panel& second) {
    // Keep the dot products separate from A so each entry is subtracted once.
    std::array<double, 1 + maxBlockSize * maxBlockSize> diagonal;
    Panel* current = &first;
    Panel* previous = &second;
    for (int k = 0; k < matrix.blocks; ++k) {
        const int width = matrix.extent(k);
        const int rootCol = k % matrix.cols;
        const int rootRow = k % matrix.rows;
        if (matrix.rank == matrix.owner(k, k)) {
            const size_t offset = matrix.column(k).front().offset;
            double* data = matrix.values.data() + offset;
            const int pivot = factorDiagonal(data, sums.data() + offset, matrix.block, width);
            diagonal[0] = pivot < 0 ? 0 : k * matrix.block + pivot + 1;
            std::copy_n(data, matrix.tileSize, diagonal.data() + 1);
        }
        if (k + 1 == matrix.blocks) {
            MPI_Bcast(diagonal.data(), 1, MPI_DOUBLE, matrix.owner(k, k), MPI_COMM_WORLD);
            return static_cast<int>(diagonal[0]) - 1;
        }

        int failure = 0;
        current->setRange(k + 1);
        if (matrix.col == rootCol) {
            if (matrix.rows > 1) {
                MPI_Bcast(diagonal.data(), 1 + matrix.tileSize, MPI_DOUBLE, rootRow, matrix.colComm);
            }
            failure = static_cast<int>(diagonal[0]);
            if (failure == 0) {
                for (const Tile& tile : matrix.column(k)) {
                    if (tile.row > k) {
                        double* data = matrix.values.data() + tile.offset;
                        solvePanelTile(data, sums.data() + tile.offset, diagonal.data() + 1,
                                       matrix.block, matrix.extent(tile.row), width);
                        std::copy_n(data, matrix.tileSize, current->rowTile(tile.row));
                    }
                }
            }
        }
        const MPI_Request request = current->broadcastRows(rootCol, failure);
        // The next block column was updated first. Finish the previous panel's
        // trailing updates while the current panel is in flight.
        if (k > 0) {
            addPanelProducts(matrix, sums, matrix.trailing(k + 1), *previous, matrix.extent(k - 1));
        }
        failure = current->finishRows(request);
        if (failure != 0) {
            return failure - 1;
        }
        current->exchangeColumns();

        if (matrix.col == (k + 1) % matrix.cols) {
            addPanelProducts(matrix, sums, matrix.column(k + 1), *current, width);
        }
        std::swap(current, previous);
    }
    return -1;
}

bool validateCholesky(const Matrix& matrix, const std::vector<double>& original,
                      std::vector<double>& reconstructed, Panel& panel) {
    std::fill(reconstructed.begin(), reconstructed.end(), 0.0);
    for (int k = 0; k < matrix.blocks; ++k) {
        const int root = k % matrix.cols;
        panel.setRange(k);
        if (matrix.col == root) {
            for (const Tile& tile : matrix.column(k)) {
                std::copy_n(matrix.values.data() + tile.offset, matrix.tileSize, panel.rowTile(tile.row));
            }
        }
        panel.finishRows(panel.broadcastRows(root, 0));
        panel.exchangeColumns();
        addPanelProducts(matrix, reconstructed, matrix.trailing(k), panel, matrix.extent(k));
    }

    double localErrors[2] = {};
    for (const Tile& tile : matrix.tiles) {
        const int height = matrix.extent(tile.row);
        const int width = matrix.extent(tile.col);
        for (int j = 0; j < width; ++j) {
            const int start = tile.row == tile.col ? j : 0;
            for (int i = start; i < height; ++i) {
                const size_t offset = tile.offset + j * matrix.block + i;
                const double error = std::abs(reconstructed[offset] - original[offset]);
                localErrors[0] = std::max(localErrors[0], error);
                localErrors[1] = std::max(localErrors[1], error / (std::abs(original[offset]) + 1e-10));
            }
        }
    }
    double errors[2];
    MPI_Allreduce(localErrors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const bool valid = errors[1] <= 1e-6;
    if (matrix.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (!valid) {
            printf("Validation failed: relative error too large\n");
        }
    }
    return valid;
}

void printDistributedResults(const Matrix& matrix) {
    constexpr int tag = 0;
    if (matrix.rank != 0) {
        for (const Tile& tile : matrix.tiles) {
            MPI_Send(matrix.values.data() + tile.offset, matrix.tileSize, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
        return;
    }

    std::vector<double> result(static_cast<size_t>(matrix.n) * matrix.n);
    std::vector<double> buffer(matrix.tileSize);
    // Match the senders' tile order and restore the benchmark's row-major layout.
    size_t local = 0;
    for (int j = 0; j < matrix.blocks; ++j) {
        for (int i = j; i < matrix.blocks; ++i) {
            const int source = matrix.owner(i, j);
            const double* data = buffer.data();
            if (source == 0) {
                data = matrix.values.data() + matrix.tiles[local++].offset;
            } else {
                MPI_Recv(buffer.data(), matrix.tileSize, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int row = 0; row < matrix.extent(i); ++row) {
                double* output = result.data() + (static_cast<size_t>(i) * matrix.block + row) * matrix.n
                    + static_cast<size_t>(j) * matrix.block;
                const int width = i == j ? row + 1 : matrix.extent(j);
                for (int col = 0; col < width; ++col) {
                    output[col] = data[col * matrix.block + row];
                }
            }
        }
    }
    print_results(result, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int size) {
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
            printf("Matrix size must be nonnegative\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix matrix(n, rank, size);
    generatePositiveDefiniteMatrix(matrix);
    std::vector<double> original;
    if (validate) {
        original = matrix.values;
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    std::vector<double> sums(matrix.values.size());
    Panel panel(matrix);
    Panel lookahead(matrix);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int failure = choleskyDecomposition(matrix, sums, panel, lookahead);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (failure >= 0) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failure);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / elapsed / 1e9);
    }
    if (printResults) {
        printDistributedResults(matrix);
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateCholesky(matrix, original, sums, panel);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int result = run(argc, argv, rank, size);
    MPI_Finalize();
    return result;
}
