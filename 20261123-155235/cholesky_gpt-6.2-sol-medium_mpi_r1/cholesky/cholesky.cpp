#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Tile boundaries are multiples of the serial dot product's SIMD group size.
constexpr int dotGroup = 8;

struct ProcessGrid {
    int rank;
    int rows;
    int cols;
    int row;
    int col;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    ProcessGrid() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int size;
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

    int owner(int i, int j) const {
        return (i % rows) * cols + j % cols;
    }
};

// The first local index whose global index is at least limit.
int firstLocal(int limit, int coordinate, int period) {
    return limit / period + (coordinate < limit % period);
}

int mpiChunk(size_t count) {
    return static_cast<int>(std::min(count, size_t(std::numeric_limits<int>::max())));
}

void transferDoubles(double* data, size_t count, int source, int destination, int rank) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = mpiChunk(count - offset);
        if (rank == source) {
            MPI_Send(data + offset, chunk, MPI_DOUBLE, destination, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(data + offset, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        offset += chunk;
    }
}

void broadcastDoubles(double* data, size_t count, int root, MPI_Comm comm) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = mpiChunk(count - offset);
        MPI_Bcast(data + offset, chunk, MPI_DOUBLE, root, comm);
        offset += chunk;
    }
}

double dot(const double* left, const double* right, int length, double sum = 0.0) {
    for (int k = 0; k < length; ++k) {
        sum += left[k] * right[k];
    }
    return sum;
}

template <int Width>
using TileData = std::array<double, Width * Width>;

template <int Width>
struct Tile {
    TileData<Width> input{};
    // Holds the dot products from preceding panels until this tile is factored.
    TileData<Width> factor{};
};

template <int Width>
struct LocalMatrix {
    const ProcessGrid& grid;
    const int n;
    const int blocks;
    // Block-cyclic ownership on both axes, with only lower-triangular tiles.
    std::vector<std::vector<Tile<Width>>> tiles;

    LocalMatrix(int n, const ProcessGrid& grid)
        : grid(grid), n(n), blocks(n / Width + (n % Width != 0)),
          tiles(firstLocal(blocks, grid.row, grid.rows)) {
        for (int i = 0; i < static_cast<int>(tiles.size()); ++i) {
            tiles[i].resize(firstLocal(globalRow(i) + 1, grid.col, grid.cols));
        }
    }

    int globalRow(int i) const { return i * grid.rows + grid.row; }
    int globalCol(int j) const { return j * grid.cols + grid.col; }
    int extent(int block) const { return std::min(Width, n - block * Width); }
};

// Accumulate each dot product in scalar k order, vectorizing across output
// columns. With GCC, use the same separate multiply and add as the serial
// dot product's vector groups. dot() handles any scalar remainder.
template <int Width>
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void addProducts(double* __restrict output, const double* left, const double* right, int depth) {
    constexpr int rowGroup = 4;
    constexpr int colGroup = 8;
    static_assert(Width % rowGroup == 0 && Width % colGroup == 0);
    for (int i = 0; i < Width; i += rowGroup) {
        for (int j = 0; j < Width; j += colGroup) {
            double sums[rowGroup][colGroup];
            for (int r = 0; r < rowGroup; ++r) {
                std::copy_n(output + (i + r) * Width + j, colGroup, sums[r]);
            }
            for (int k = 0; k < depth; ++k) {
                const double* lhs = left + k * Width + i;
                const double* rhs = right + k * Width + j;
                for (int r = 0; r < rowGroup; ++r) {
                    for (int c = 0; c < colGroup; ++c) {
                        sums[r][c] += lhs[r] * rhs[c];
                    }
                }
            }
            for (int r = 0; r < rowGroup; ++r) {
                std::copy_n(sums[r], colGroup, output + (i + r) * Width + j);
            }
        }
    }
}

// Store complete SIMD groups in depth-major order for the tile kernel.
// Keep the remainder row-major so dot() handles it as in the serial loop.
template <int Width>
void packInputBlock(const double* source, double* packed, int rows, int n, int bulk) {
    for (int r = 0; r < rows; ++r) {
        const double* row = source + size_t(r) * n;
        for (int k = 0; k < bulk; ++k) {
            packed[size_t(k) * Width + r] = row[k];
        }
        std::copy_n(row + bulk, n - bulk, packed + size_t(bulk) * Width + r * (n - bulk));
    }
}

// Generate the original rand_r stream at rank zero in block rows. Separate
// row and column broadcasts give each rank only the B rows needed for its
// tiles of B * B^T.
template <int Width>
void generatePositiveDefiniteMatrix(LocalMatrix<Width>& matrix) {
    const auto& grid = matrix.grid;
    const int n = matrix.n;
    std::vector<std::vector<double>> rowBlocks(matrix.tiles.size());
    std::vector<std::vector<double>> colBlocks(firstLocal(matrix.blocks, grid.col, grid.cols));
    std::vector<double> buffer(static_cast<size_t>(Width) * n);
    const int bulk = n - n % dotGroup;
    unsigned int seed = 42;

    for (int block = 0; block < matrix.blocks; ++block) {
        const int row = block % grid.rows;
        const int col = block % grid.cols;
        const int owner = grid.owner(block, block);
        const int extent = matrix.extent(block);
        const size_t count = size_t(extent) * n;
        if (grid.rank == 0) {
            for (size_t index = 0; index < count; ++index) {
                buffer[index] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            }
        }
        if (owner != 0 && (grid.rank == 0 || grid.rank == owner)) {
            transferDoubles(buffer.data(), count, 0, owner, grid.rank);
        }
        if (grid.row == row) {
            broadcastDoubles(buffer.data(), count, col, grid.rowComm);
        }
        if (grid.col == col) {
            broadcastDoubles(buffer.data(), count, row, grid.colComm);
        }
        if (grid.row == row || grid.col == col) {
            auto& packed = grid.row == row ? rowBlocks[block / grid.rows]
                                          : colBlocks[block / grid.cols];
            packed.resize(size_t(Width) * n);
            packInputBlock<Width>(buffer.data(), packed.data(), extent, n, bulk);
        }
    }

    for (int i = 0; i < static_cast<int>(matrix.tiles.size()); ++i) {
        const int blockRow = matrix.globalRow(i);
        const auto& left = rowBlocks[i];
        for (int j = 0; j < static_cast<int>(matrix.tiles[i].size()); ++j) {
            const int blockCol = matrix.globalCol(j);
            const bool diagonal = blockRow == blockCol;
            const auto& right = blockCol % grid.rows == grid.row
                                ? rowBlocks[blockCol / grid.rows] : colBlocks[j];
            auto& input = matrix.tiles[i][j].input;
            constexpr int depthBlock = 128;
            for (int k = 0; k < bulk;) {
                const int depth = std::min(depthBlock, bulk - k);
                const size_t offset = size_t(k) * Width;
                addProducts<Width>(input.data(), left.data() + offset, right.data() + offset, depth);
                k += depth;
            }
            const int tail = n - bulk;
            if (tail) {
                const double* leftTail = left.data() + size_t(bulk) * Width;
                const double* rightTail = right.data() + size_t(bulk) * Width;
                for (int r = 0; r < matrix.extent(blockRow); ++r) {
                    const int columns = diagonal ? r + 1 : matrix.extent(blockCol);
                    for (int c = 0; c < columns; ++c) {
                        double& value = input[r * Width + c];
                        value = dot(leftTail + r * tail, rightTail + c * tail, tail, value);
                    }
                }
            }
            if (diagonal) {
                for (int r = 0; r < matrix.extent(blockRow); ++r) {
                    input[r * Width + r] += n;
                }
            }
        }
    }
}

template <int Width>
void transposeTile(const double* tile, double* packed) {
    for (int column = 0; column < Width; ++column) {
        for (int row = 0; row < Width; ++row) {
            packed[column * Width + row] = tile[row * Width + column];
        }
    }
}

template <int Width>
class Panel {
    const LocalMatrix<Width>& matrix;
    std::vector<double> rowData;
    std::vector<double> colData;
    std::vector<int> colCounts;
    std::vector<int> colDisplacements;
    std::vector<int> colOffsets;
    MPI_Datatype tileType = MPI_DATATYPE_NULL;

    double* rowTileData(int i) { return rowData.data() + 1 + size_t(i) * Width * Width; }

public:
    explicit Panel(const LocalMatrix<Width>& matrix)
        : matrix(matrix), rowData(1 + matrix.tiles.size() * Width * Width),
          colData(size_t(firstLocal(matrix.blocks, matrix.grid.col, matrix.grid.cols))
                  * Width * Width),
          colCounts(matrix.grid.rows), colDisplacements(matrix.grid.rows),
          colOffsets(firstLocal(matrix.blocks, matrix.grid.col, matrix.grid.cols)) {
        if (matrix.grid.rows != matrix.grid.cols) {
            MPI_Type_contiguous(Width * Width, MPI_DOUBLE, &tileType);
            MPI_Type_commit(&tileType);
        }
    }

    ~Panel() {
        if (tileType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&tileType);
        }
    }

    const double* rowTile(int i) const { return rowData.data() + 1 + size_t(i) * Width * Width; }
    const double* colTile(int j) const {
        const auto& grid = matrix.grid;
        if (grid.rows == grid.cols && grid.row == grid.col) {
            return rowTile(j);
        }
        return colData.data() + size_t(colOffsets[j]) * Width * Width;
    }

    // First fan out L(i,k) along process rows, then redistribute the same
    // tiles along process columns to supply L(j,k) to owners of A(i,j).
    int exchange(int k, int firstBlock, int failedPivot = 0) {
        const auto& grid = matrix.grid;
        const int rowBegin = firstLocal(firstBlock, grid.row, grid.rows);
        const int colBegin = firstLocal(firstBlock, grid.col, grid.cols);
        const int panelCol = k % grid.cols;
        if (grid.col == panelCol && !failedPivot) {
            for (int i = rowBegin; i < static_cast<int>(matrix.tiles.size()); ++i) {
                transposeTile<Width>(matrix.tiles[i][k / grid.cols].factor.data(), rowTileData(i));
            }
        }
        // Send the pivot status in the slot immediately before the active tiles.
        double* message = rowTileData(rowBegin) - 1;
        *message = failedPivot;
        broadcastDoubles(message, rowData.size() - size_t(rowBegin) * Width * Width,
                         panelCol, grid.rowComm);
        if (*message != 0.0) {
            return static_cast<int>(*message);
        }

        const int columns = static_cast<int>(colOffsets.size());
        if (colBegin == columns) {
            return 0;
        }
        if (grid.rows == grid.cols) {
            // All tiles needed by this process column come from one row.
            for (int j = colBegin; j < columns; ++j) {
                colOffsets[j] = j - colBegin;
            }
            double* data = grid.row == grid.col ? rowTileData(colBegin) : colData.data();
            broadcastDoubles(data, size_t(columns - colBegin) * Width * Width,
                             grid.col, grid.colComm);
            return 0;
        }

        std::fill(colCounts.begin(), colCounts.end(), 0);
        for (int j = colBegin; j < columns; ++j) {
            const int sourceRow = matrix.globalCol(j) % grid.rows;
            ++colCounts[sourceRow];
        }
        int offset = 0;
        for (int r = 0; r < grid.rows; ++r) {
            colDisplacements[r] = offset;
            offset += colCounts[r];
            colCounts[r] = 0;
        }
        for (int j = colBegin; j < columns; ++j) {
            const int block = matrix.globalCol(j);
            const int sourceRow = block % grid.rows;
            colOffsets[j] = colDisplacements[sourceRow] + colCounts[sourceRow]++;
            if (grid.row == sourceRow) {
                std::copy_n(rowTile(block / grid.rows), Width * Width,
                            colData.data() + size_t(colOffsets[j]) * Width * Width);
            }
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, tileType, colData.data(),
                       colCounts.data(), colDisplacements.data(), tileType, grid.colComm);
        return 0;
    }
};

template <int Width>
int solveTile(Tile<Width>& tile, const double* diagonal, int rows, int columns,
              bool isDiagonal) {
    for (int i = 0; i < rows; ++i) {
        double* row = tile.factor.data() + i * Width;
        const int end = isDiagonal ? i : columns;
        for (int j = 0; j < end; ++j) {
            const double* pivot = diagonal + j * Width;
            const double sum = dot(row, pivot, j, row[j]);
            row[j] = (tile.input[i * Width + j] - sum) / pivot[j];
        }
        if (isDiagonal) {
            const double sum = dot(row, row, i, row[i]);
            const double value = tile.input[i * Width + i] - sum;
            if (value <= 0.0) {
                return i + 1;
            }
            row[i] = std::sqrt(value);
            std::fill(row + i + 1, row + Width, 0.0);
        }
    }
    return 0;
}

template <int Width>
int choleskyDecomposition(LocalMatrix<Width>& matrix, Panel<Width>& panel) {
    const auto& grid = matrix.grid;
    std::array<double, Width * Width + 1> diagonal{};
    for (int k = 0; k < matrix.blocks; ++k) {
        const int pivotRow = k % grid.rows;
        const int pivotCol = k % grid.cols;
        const int extent = matrix.extent(k);
        int failedPivot = 0;
        if (grid.col == pivotCol) {
            if (grid.row == pivotRow) {
                auto& tile = matrix.tiles[k / grid.rows][k / grid.cols];
                const int failure = solveTile(tile, tile.factor.data(), extent, extent, true);
                diagonal[0] = failure ? k * Width + failure : 0;
                std::copy(tile.factor.begin(), tile.factor.end(), diagonal.begin() + 1);
            }
            MPI_Bcast(diagonal.data(), diagonal.size(), MPI_DOUBLE, pivotRow, grid.colComm);
            failedPivot = static_cast<int>(diagonal[0]);
            if (!failedPivot) {
                const int firstRow = firstLocal(k + 1, grid.row, grid.rows);
                for (int i = firstRow; i < static_cast<int>(matrix.tiles.size()); ++i) {
                    solveTile(matrix.tiles[i][k / grid.cols], diagonal.data() + 1,
                              matrix.extent(matrix.globalRow(i)), extent, false);
                }
            }
        }
        failedPivot = panel.exchange(k, k + 1, failedPivot);
        if (failedPivot) {
            return failedPivot;
        }

        const int firstRow = firstLocal(k + 1, grid.row, grid.rows);
        const int firstCol = firstLocal(k + 1, grid.col, grid.cols);
        for (int i = firstRow; i < static_cast<int>(matrix.tiles.size()); ++i) {
            const double* left = panel.rowTile(i);
            for (int j = firstCol; j < static_cast<int>(matrix.tiles[i].size()); ++j) {
                addProducts<Width>(matrix.tiles[i][j].factor.data(), left, panel.colTile(j), extent);
            }
        }
    }
    return 0;
}

template <int Width>
bool validateCholesky(const LocalMatrix<Width>& matrix, Panel<Width>& panel) {
    const auto& grid = matrix.grid;
    std::vector<std::vector<TileData<Width>>> reconstructed(matrix.tiles.size());
    for (size_t i = 0; i < matrix.tiles.size(); ++i) {
        reconstructed[i].resize(matrix.tiles[i].size());
    }

    double localErrors[2] = {};
    for (int k = 0; k < matrix.blocks; ++k) {
        panel.exchange(k, k);
        const int extent = matrix.extent(k);
        const int tail = extent % dotGroup;
        const int bulk = extent - tail;
        const int firstRow = firstLocal(k, grid.row, grid.rows);
        const int firstCol = firstLocal(k, grid.col, grid.cols);
        for (int i = firstRow; i < static_cast<int>(matrix.tiles.size()); ++i) {
            const double* left = panel.rowTile(i);
            const int blockRow = matrix.globalRow(i);
            for (int j = firstCol; j < static_cast<int>(matrix.tiles[i].size()); ++j) {
                auto& values = reconstructed[i][j];
                const double* right = panel.colTile(j);
                if (bulk) {
                    addProducts<Width>(values.data(), left, right, bulk);
                }
                if (matrix.globalCol(j) == k) {
                    const auto& input = matrix.tiles[i][j].input;
                    TileData<Width> leftRows, rightRows;
                    if (tail) {
                        transposeTile<Width>(left, leftRows.data());
                        transposeTile<Width>(right, rightRows.data());
                    }
                    for (int r = 0; r < matrix.extent(blockRow); ++r) {
                        const int columns = blockRow == k ? r + 1 : extent;
                        for (int c = 0; c < columns; ++c) {
                            const int index = r * Width + c;
                            double value = values[index];
                            if (tail) {
                                value = dot(leftRows.data() + r * Width + bulk,
                                            rightRows.data() + c * Width + bulk, tail, value);
                            }
                            const double error = std::abs(value - input[index]);
                            localErrors[0] = std::max(localErrors[0], error);
                            localErrors[1] = std::max(localErrors[1],
                                                     error / (std::abs(input[index]) + 1e-10));
                        }
                    }
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
        printf("Validation: %s\n", errors[1] <= 1e-6 ? "PASSED" : "FAILED");
    }
    return errors[1] <= 1e-6;
}

template <int Width>
void printFactor(const LocalMatrix<Width>& matrix) {
    const auto& grid = matrix.grid;
    const int n = matrix.n;
    std::vector<double> result;
    if (grid.rank == 0) {
        result.resize(size_t(n) * n);
    }
    std::vector<double> packed;
    for (int i = 0; i < matrix.blocks; ++i) {
        const int ownerRow = i % grid.rows;
        if (grid.rank != 0 && grid.row != ownerRow) {
            continue;
        }
        for (int col = 0; col < grid.cols; ++col) {
            const int owner = ownerRow * grid.cols + col;
            const int count = firstLocal(i + 1, col, grid.cols);
            if (count == 0 || (grid.rank != 0 && grid.rank != owner)) {
                continue;
            }
            if (owner != 0) {
                packed.resize(size_t(count) * Width * Width);
                if (grid.rank == owner) {
                    for (int j = 0; j < count; ++j) {
                        const auto& tile = matrix.tiles[i / grid.rows][j].factor;
                        std::copy(tile.begin(), tile.end(), packed.data() + size_t(j) * Width * Width);
                    }
                }
                transferDoubles(packed.data(), packed.size(), owner, 0, grid.rank);
            }
            if (grid.rank == 0) {
                for (int j = 0; j < count; ++j) {
                    const int blockCol = j * grid.cols + col;
                    const double* tile = owner == 0 ? matrix.tiles[i / grid.rows][j].factor.data()
                                                   : packed.data() + size_t(j) * Width * Width;
                    for (int r = 0; r < matrix.extent(i); ++r) {
                        const size_t offset = size_t(i * Width + r) * n + blockCol * Width;
                        std::copy_n(tile + r * Width, matrix.extent(blockCol), result.data() + offset);
                    }
                }
            }
        }
    }
    if (grid.rank == 0) {
        print_results(result, "CholeskyL");
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

template <int Width>
int runBenchmark(int n, bool validate, bool printResults, const ProcessGrid& grid) {
    if (grid.rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    LocalMatrix<Width> matrix(n, grid);
    generatePositiveDefiniteMatrix(matrix);
    Panel<Width> panel(matrix);

    if (grid.rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int failedPivot = choleskyDecomposition(matrix, panel);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (failedPivot) {
        if (grid.rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failedPivot - 1);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }
    if (grid.rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }
    if (printResults) {
        printFactor(matrix);
    }
    if (validate) {
        if (grid.rank == 0) {
            printf("Validating result...\n");
        }
        return validateCholesky(matrix, panel) ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    enum Option { Size, Validate, PrintResults, Exit, OptionCount };
    int options[OptionCount] = {512, 0, 0, -1};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                options[Size] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                options[Validate] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[PrintResults] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                options[Exit] = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                options[Exit] = 1;
                break;
            }
        }
        if (options[Exit] == -1 && options[Size] < 0) {
            printf("Matrix size must be nonnegative\n");
            options[Exit] = 1;
        }
    }
    MPI_Bcast(options, OptionCount, MPI_INT, 0, MPI_COMM_WORLD);

    int status = options[Exit];
    if (status == -1) {
        const ProcessGrid grid;
        const int n = options[Size];
        if (n / grid.rows >= 128) {
            status = runBenchmark<32>(n, options[Validate], options[PrintResults], grid);
        } else if (n / grid.rows >= 48) {
            status = runBenchmark<16>(n, options[Validate], options[PrintResults], grid);
        } else {
            status = runBenchmark<8>(n, options[Validate], options[PrintResults], grid);
        }
    }
    MPI_Finalize();
    return status;
}
