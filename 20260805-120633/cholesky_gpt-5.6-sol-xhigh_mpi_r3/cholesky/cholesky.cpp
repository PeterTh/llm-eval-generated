#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr size_t kNoOffset = std::numeric_limits<size_t>::max();

struct ProcessGrid {
    MPI_Comm world = MPI_COMM_WORLD;
    MPI_Comm row = MPI_COMM_NULL;
    MPI_Comm column = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int rows = 1;
    int columns = 1;
    int rowCoordinate = 0;
    int columnCoordinate = 0;

    ProcessGrid() {
        MPI_Comm_rank(world, &rank);
        MPI_Comm_size(world, &size);

        int dimensions[2] = {0, 0};
        MPI_Dims_create(size, 2, dimensions);
        rows = dimensions[0];
        columns = dimensions[1];
        rowCoordinate = rank / columns;
        columnCoordinate = rank % columns;

        // The keys make communicator ranks equal to the corresponding grid
        // coordinate, which simplifies collective roots and panel placement.
        MPI_Comm_split(world, rowCoordinate, columnCoordinate, &row);
        MPI_Comm_split(world, columnCoordinate, rowCoordinate, &column);
    }

    ~ProcessGrid() {
        if (row != MPI_COMM_NULL) {
            MPI_Comm_free(&row);
        }
        if (column != MPI_COMM_NULL) {
            MPI_Comm_free(&column);
        }
    }

    ProcessGrid(const ProcessGrid&) = delete;
    ProcessGrid& operator=(const ProcessGrid&) = delete;
};

int chooseTileSize(const size_t n, const ProcessGrid& grid) {
    if (n == 0) {
        return 1;
    }

    size_t tileSize = std::min<size_t>(64, n);
    if (grid.size == 1) {
        return static_cast<int>(tileSize);
    }

    // Keep several cyclic tiles per process-grid dimension for load balance.
    const size_t desiredTiles =
        4 * static_cast<size_t>(std::max(grid.rows, grid.columns));
    while (tileSize > 1 && (n + tileSize - 1) / tileSize < desiredTiles) {
        tileSize = (tileSize + 1) / 2;
    }
    return static_cast<int>(tileSize);
}

class DistributedMatrix {
  public:
    DistributedMatrix(const size_t order, const int blockSize,
                      const ProcessGrid& processGrid)
        : n(order), tileSize(blockSize),
          blockCount(static_cast<int>((order + blockSize - 1) / blockSize)),
          processRows(processGrid.rows), processColumns(processGrid.columns),
          myProcessRow(processGrid.rowCoordinate),
          myProcessColumn(processGrid.columnCoordinate),
          localBlockRows(localBlockCount(blockCount, myProcessRow, processRows)),
          localBlockColumns(
              localBlockCount(blockCount, myProcessColumn, processColumns)),
          tiles(localBlockRows * localBlockColumns) {
        for (int blockRow = myProcessRow; blockRow < blockCount;
             blockRow += processRows) {
            for (int blockColumn = myProcessColumn;
                 blockColumn <= blockRow && blockColumn < blockCount;
                 blockColumn += processColumns) {
                tile(blockRow, blockColumn)
                    .assign(rowsInBlock(blockRow) * rowsInBlock(blockColumn), 0.0);
            }
        }
    }

    bool owns(const int blockRow, const int blockColumn) const {
        return blockRow % processRows == myProcessRow &&
               blockColumn % processColumns == myProcessColumn;
    }

    size_t rowsInBlock(const int block) const {
        const size_t first = static_cast<size_t>(block) * tileSize;
        return std::min(static_cast<size_t>(tileSize), n - first);
    }

    std::vector<double>& tile(const int blockRow, const int blockColumn) {
        return tiles[localIndex(blockRow, blockColumn)];
    }

    const std::vector<double>& tile(const int blockRow,
                                    const int blockColumn) const {
        return tiles[localIndex(blockRow, blockColumn)];
    }

    size_t n;
    int tileSize;
    int blockCount;
    int processRows;
    int processColumns;
    int myProcessRow;
    int myProcessColumn;
    size_t localBlockRows;
    size_t localBlockColumns;

  private:
    static size_t localBlockCount(const int total, const int coordinate,
                                  const int stride) {
        if (coordinate >= total) {
            return 0;
        }
        return static_cast<size_t>((total - 1 - coordinate) / stride + 1);
    }

    size_t localIndex(const int blockRow, const int blockColumn) const {
        return static_cast<size_t>(blockRow / processRows) * localBlockColumns +
               static_cast<size_t>(blockColumn / processColumns);
    }

    std::vector<std::vector<double>> tiles;
};

[[noreturn]] void abortForLargeMessage(const ProcessGrid& grid) {
    if (grid.rank == 0) {
        std::fprintf(stderr,
                     "Error: an MPI message exceeds this MPI implementation's "
                     "count limit\n");
    }
    MPI_Abort(grid.world, 2);
    std::abort();
}

int checkedMpiCount(const size_t count, const ProcessGrid& grid) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        abortForLargeMessage(grid);
    }
    return static_cast<int>(count);
}

// glibc rand_r advances the POSIX LCG three times per returned value.  Jumping
// directly to the beginning of a row lets every rank generate exactly the rows
// it needs without replicating B or serializing generation through rank zero.
unsigned int advanceRandomState(unsigned int state, uint64_t advances) {
    uint32_t accumulatedMultiplier = 1;
    uint32_t accumulatedIncrement = 0;
    uint32_t multiplier = 1103515245U;
    uint32_t increment = 12345U;

    while (advances != 0) {
        if ((advances & 1U) != 0) {
            accumulatedMultiplier = accumulatedMultiplier * multiplier;
            accumulatedIncrement = accumulatedIncrement * multiplier + increment;
        }
        increment = (multiplier + 1U) * increment;
        multiplier = multiplier * multiplier;
        advances >>= 1U;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

bool randomJumpIsCompatible() {
    if (std::numeric_limits<unsigned int>::digits != 32) {
        return false;
    }
    unsigned int actualState = 42;
    (void)rand_r(&actualState);
    return actualState == advanceRandomState(42, 3);
}

void generateRandomRow(double* row, const size_t globalRow, const size_t n,
                       const bool canJump) {
    unsigned int seed = 42;
    const uint64_t firstElement = static_cast<uint64_t>(globalRow) * n;
    if (canJump) {
        seed = advanceRandomState(seed, 3 * firstElement);
    } else {
        // A portability fallback for a libc with a different rand_r state
        // transition.  It is slower, but preserves the original generator.
        for (uint64_t element = 0; element < firstElement; ++element) {
            (void)rand_r(&seed);
        }
    }
    for (size_t column = 0; column < n; ++column) {
        row[column] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    }
}

std::vector<double> generateRandomBlock(const int block, const size_t n,
                                        const int tileSize,
                                        const bool canJump) {
    const size_t firstRow = static_cast<size_t>(block) * tileSize;
    const size_t rowCount = std::min(static_cast<size_t>(tileSize), n - firstRow);
    std::vector<double> result(rowCount * n);
    for (size_t row = 0; row < rowCount; ++row) {
        generateRandomRow(result.data() + row * n, firstRow + row, n, canJump);
    }
    return result;
}

double originalDotProduct(const double* left, const double* right,
                          const size_t count) {
    double sum = 0.0;
    for (size_t index = 0; index < count; ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

std::vector<double> generateOriginalTile(const int blockRow,
                                         const int blockColumn,
                                         const DistributedMatrix& matrix,
                                         const bool canJump) {
    const size_t rows = matrix.rowsInBlock(blockRow);
    const size_t columns = matrix.rowsInBlock(blockColumn);
    const std::vector<double> left =
        generateRandomBlock(blockRow, matrix.n, matrix.tileSize, canJump);
    const std::vector<double> right = blockRow == blockColumn
                                          ? left
                                          : generateRandomBlock(blockColumn,
                                                                matrix.n,
                                                                matrix.tileSize,
                                                                canJump);
    std::vector<double> result(rows * columns, 0.0);

    for (size_t row = 0; row < rows; ++row) {
        const size_t columnLimit =
            blockRow == blockColumn ? row + 1 : columns;
        for (size_t column = 0; column < columnLimit; ++column) {
            result[row * columns + column] =
                originalDotProduct(left.data() + row * matrix.n,
                                   right.data() + column * matrix.n, matrix.n);
        }
    }

    if (blockRow == blockColumn) {
        for (size_t diagonal = 0; diagonal < rows; ++diagonal) {
            result[diagonal * columns + diagonal] +=
                static_cast<double>(matrix.n);
        }
    }
    return result;
}

void generatePositiveDefiniteMatrix(DistributedMatrix& matrix,
                                    const bool canJump) {
    for (int blockRow = matrix.myProcessRow; blockRow < matrix.blockCount;
         blockRow += matrix.processRows) {
        for (int blockColumn = matrix.myProcessColumn;
             blockColumn <= blockRow && blockColumn < matrix.blockCount;
             blockColumn += matrix.processColumns) {
            matrix.tile(blockRow, blockColumn) =
                generateOriginalTile(blockRow, blockColumn, matrix, canJump);
        }
    }
}

bool factorDiagonalTile(std::vector<double>& diagonal, const size_t size,
                        size_t& failedAt) {
    for (size_t column = 0; column < size; ++column) {
        double diagonalSum = 0.0;
        for (size_t inner = 0; inner < column; ++inner) {
            const double value = diagonal[column * size + inner];
            diagonalSum += value * value;
        }

        const double value = diagonal[column * size + column] - diagonalSum;
        if (!(value > 0.0) || !std::isfinite(value)) {
            failedAt = column;
            return false;
        }
        diagonal[column * size + column] = std::sqrt(value);

        for (size_t row = column + 1; row < size; ++row) {
            double sum = 0.0;
            for (size_t inner = 0; inner < column; ++inner) {
                sum += diagonal[row * size + inner] *
                       diagonal[column * size + inner];
            }
            diagonal[row * size + column] =
                (diagonal[row * size + column] - sum) /
                diagonal[column * size + column];
        }
    }
    // Prior symmetric rank-k updates touch the complete diagonal tile.  The
    // upper half is workspace only and must not be exposed later as part of L.
    for (size_t row = 0; row < size; ++row) {
        std::fill(diagonal.begin() + row * size + row + 1,
                  diagonal.begin() + (row + 1) * size, 0.0);
    }
    return true;
}

void triangularSolve(std::vector<double>& panel,
                     const std::vector<double>& diagonal, const size_t rows,
                     const size_t width) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < width; ++column) {
            double sum = 0.0;
            for (size_t inner = 0; inner < column; ++inner) {
                sum += panel[row * width + inner] *
                       diagonal[column * width + inner];
            }
            panel[row * width + column] =
                (panel[row * width + column] - sum) /
                diagonal[column * width + column];
        }
    }
}

void subtractProduct(std::vector<double>& result, const double* left,
                     const double* right, const size_t rows,
                     const size_t columns, const size_t innerSize) {
    for (size_t row = 0; row < rows; ++row) {
        const double* leftRow = left + row * innerSize;
        for (size_t column = 0; column < columns; ++column) {
            const double* rightRow = right + column * innerSize;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t inner = 0; inner < innerSize; ++inner) {
                sum += leftRow[inner] * rightRow[inner];
            }
            result[row * columns + column] -= sum;
        }
    }
}

void subtractSymmetricProduct(std::vector<double>& result,
                              const double* panel, const size_t size,
                              const size_t innerSize) {
    for (size_t row = 0; row < size; ++row) {
        const double* leftRow = panel + row * innerSize;
        for (size_t column = 0; column <= row; ++column) {
            const double* rightRow = panel + column * innerSize;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t inner = 0; inner < innerSize; ++inner) {
                sum += leftRow[inner] * rightRow[inner];
            }
            result[row * size + column] -= sum;
        }
    }
}

struct DistributedPanel {
    std::vector<double> processRowData;
    std::vector<double> processColumnData;
    std::vector<size_t> processRowOffsets;
    std::vector<size_t> processColumnOffsets;
};

DistributedPanel distributePanel(const DistributedMatrix& matrix,
                                 const int panelBlock,
                                 const int firstIncludedBlock,
                                 const ProcessGrid& grid) {
    const size_t panelWidth = matrix.rowsInBlock(panelBlock);
    DistributedPanel panel;
    panel.processRowOffsets.assign(matrix.blockCount, kNoOffset);
    panel.processColumnOffsets.assign(matrix.blockCount, kNoOffset);

    size_t rowElementCount = 0;
    for (int block = firstIncludedBlock; block < matrix.blockCount; ++block) {
        if (block % grid.rows == grid.rowCoordinate) {
            panel.processRowOffsets[block] = rowElementCount;
            rowElementCount += matrix.rowsInBlock(block) * panelWidth;
        }
    }
    panel.processRowData.resize(rowElementCount);

    const int panelOwnerColumn = panelBlock % grid.columns;
    if (grid.columnCoordinate == panelOwnerColumn) {
        for (int block = firstIncludedBlock; block < matrix.blockCount; ++block) {
            if (block % grid.rows != grid.rowCoordinate) {
                continue;
            }
            const std::vector<double>& source = matrix.tile(block, panelBlock);
            std::copy(source.begin(), source.end(),
                      panel.processRowData.begin() +
                          panel.processRowOffsets[block]);
        }
    }
    MPI_Bcast(panel.processRowData.data(),
              checkedMpiCount(panel.processRowData.size(), grid), MPI_DOUBLE,
              panelOwnerColumn, grid.row);

    std::vector<double> columnContribution;
    size_t contributionCount = 0;
    for (int block = firstIncludedBlock; block < matrix.blockCount; ++block) {
        if (block % grid.rows == grid.rowCoordinate &&
            block % grid.columns == grid.columnCoordinate) {
            contributionCount += matrix.rowsInBlock(block) * panelWidth;
        }
    }
    columnContribution.reserve(contributionCount);
    for (int block = firstIncludedBlock; block < matrix.blockCount; ++block) {
        if (block % grid.rows == grid.rowCoordinate &&
            block % grid.columns == grid.columnCoordinate) {
            const size_t offset = panel.processRowOffsets[block];
            const size_t count = matrix.rowsInBlock(block) * panelWidth;
            columnContribution.insert(
                columnContribution.end(), panel.processRowData.begin() + offset,
                panel.processRowData.begin() + offset + count);
        }
    }

    std::vector<int> receiveCounts(grid.rows, 0);
    std::vector<int> displacements(grid.rows, 0);
    size_t totalColumnElements = 0;
    for (int processRow = 0; processRow < grid.rows; ++processRow) {
        size_t count = 0;
        for (int block = firstIncludedBlock; block < matrix.blockCount; ++block) {
            if (block % grid.rows == processRow &&
                block % grid.columns == grid.columnCoordinate) {
                count += matrix.rowsInBlock(block) * panelWidth;
            }
        }
        receiveCounts[processRow] = checkedMpiCount(count, grid);
        displacements[processRow] = checkedMpiCount(totalColumnElements, grid);
        totalColumnElements += count;
    }
    checkedMpiCount(totalColumnElements, grid);
    panel.processColumnData.resize(totalColumnElements);

    MPI_Allgatherv(columnContribution.data(),
                   checkedMpiCount(columnContribution.size(), grid), MPI_DOUBLE,
                   panel.processColumnData.data(), receiveCounts.data(),
                   displacements.data(), MPI_DOUBLE, grid.column);

    for (int processRow = 0; processRow < grid.rows; ++processRow) {
        size_t offset = static_cast<size_t>(displacements[processRow]);
        for (int block = firstIncludedBlock; block < matrix.blockCount; ++block) {
            if (block % grid.rows == processRow &&
                block % grid.columns == grid.columnCoordinate) {
                panel.processColumnOffsets[block] = offset;
                offset += matrix.rowsInBlock(block) * panelWidth;
            }
        }
    }
    return panel;
}

bool choleskyDecomposition(DistributedMatrix& matrix,
                           const ProcessGrid& grid) {
    for (int panelBlock = 0; panelBlock < matrix.blockCount; ++panelBlock) {
        const int diagonalProcessRow = panelBlock % grid.rows;
        const int diagonalProcessColumn = panelBlock % grid.columns;
        const size_t panelWidth = matrix.rowsInBlock(panelBlock);
        std::vector<double> diagonal(panelWidth * panelWidth);

        if (grid.rowCoordinate == diagonalProcessRow &&
            grid.columnCoordinate == diagonalProcessColumn) {
            std::vector<double>& localDiagonal =
                matrix.tile(panelBlock, panelBlock);
            size_t failedAt = 0;
            if (!factorDiagonalTile(localDiagonal, panelWidth, failedAt)) {
                std::fprintf(stderr,
                             "Error: Matrix is not positive definite at "
                             "diagonal element %zu\n",
                             static_cast<size_t>(panelBlock) * matrix.tileSize +
                                 failedAt);
                std::fflush(stderr);
                MPI_Abort(grid.world, 1);
                return false;
            }
            diagonal = localDiagonal;
        }

        // Only the process column owning the active panel needs L(k,k).
        if (grid.columnCoordinate == diagonalProcessColumn) {
            MPI_Bcast(diagonal.data(), checkedMpiCount(diagonal.size(), grid),
                      MPI_DOUBLE, diagonalProcessRow, grid.column);

            for (int blockRow = panelBlock + 1;
                 blockRow < matrix.blockCount; ++blockRow) {
                if (blockRow % grid.rows != grid.rowCoordinate) {
                    continue;
                }
                std::vector<double>& localPanel =
                    matrix.tile(blockRow, panelBlock);
                triangularSolve(localPanel, diagonal,
                                matrix.rowsInBlock(blockRow), panelWidth);
            }
        }

        if (panelBlock + 1 == matrix.blockCount) {
            continue;
        }

        const DistributedPanel panel =
            distributePanel(matrix, panelBlock, panelBlock + 1, grid);

        for (int blockRow = grid.rowCoordinate; blockRow < matrix.blockCount;
             blockRow += grid.rows) {
            if (blockRow <= panelBlock) {
                continue;
            }
            const double* left = panel.processRowData.data() +
                                 panel.processRowOffsets[blockRow];
            for (int blockColumn = grid.columnCoordinate;
                 blockColumn <= blockRow && blockColumn < matrix.blockCount;
                 blockColumn += grid.columns) {
                if (blockColumn <= panelBlock) {
                    continue;
                }
                const double* right = panel.processColumnData.data() +
                                      panel.processColumnOffsets[blockColumn];
                if (blockRow == blockColumn) {
                    subtractSymmetricProduct(
                        matrix.tile(blockRow, blockColumn), left,
                        matrix.rowsInBlock(blockRow), panelWidth);
                } else {
                    subtractProduct(matrix.tile(blockRow, blockColumn), left,
                                    right, matrix.rowsInBlock(blockRow),
                                    matrix.rowsInBlock(blockColumn), panelWidth);
                }
            }
        }
    }
    return true;
}

bool validateCholesky(const DistributedMatrix& factor,
                      DistributedMatrix& residual,
                      const ProcessGrid& grid, const bool canJump) {
    // residual starts as A.  Subtract one distributed panel product at a time,
    // leaving A - L*L^T without gathering either matrix to one process.
    for (int panelBlock = 0; panelBlock < factor.blockCount; ++panelBlock) {
        const size_t panelWidth = factor.rowsInBlock(panelBlock);
        const DistributedPanel panel =
            distributePanel(factor, panelBlock, panelBlock, grid);

        for (int blockRow = grid.rowCoordinate; blockRow < factor.blockCount;
             blockRow += grid.rows) {
            if (blockRow < panelBlock) {
                continue;
            }
            const double* left = panel.processRowData.data() +
                                 panel.processRowOffsets[blockRow];
            for (int blockColumn = grid.columnCoordinate;
                 blockColumn <= blockRow && blockColumn < factor.blockCount;
                 blockColumn += grid.columns) {
                if (blockColumn < panelBlock) {
                    continue;
                }
                const double* right = panel.processColumnData.data() +
                                      panel.processColumnOffsets[blockColumn];
                if (blockRow == blockColumn) {
                    subtractSymmetricProduct(
                        residual.tile(blockRow, blockColumn), left,
                        factor.rowsInBlock(blockRow), panelWidth);
                } else {
                    subtractProduct(residual.tile(blockRow, blockColumn), left,
                                    right, factor.rowsInBlock(blockRow),
                                    factor.rowsInBlock(blockColumn), panelWidth);
                }
            }
        }
    }

    double localMaxAbsoluteError = 0.0;
    double localMaxRelativeError = 0.0;
    for (int blockRow = factor.myProcessRow; blockRow < factor.blockCount;
         blockRow += factor.processRows) {
        for (int blockColumn = factor.myProcessColumn;
             blockColumn <= blockRow && blockColumn < factor.blockCount;
             blockColumn += factor.processColumns) {
            const std::vector<double> original =
                generateOriginalTile(blockRow, blockColumn, factor, canJump);
            const std::vector<double>& difference =
                residual.tile(blockRow, blockColumn);
            const size_t rows = factor.rowsInBlock(blockRow);
            const size_t columns = factor.rowsInBlock(blockColumn);
            for (size_t row = 0; row < rows; ++row) {
                const size_t columnLimit =
                    blockRow == blockColumn ? row + 1 : columns;
                for (size_t column = 0; column < columnLimit; ++column) {
                    const size_t index = row * columns + column;
                    const double absoluteError = std::fabs(difference[index]);
                    localMaxAbsoluteError =
                        std::max(localMaxAbsoluteError, absoluteError);
                    localMaxRelativeError = std::max(
                        localMaxRelativeError,
                        absoluteError / (std::fabs(original[index]) + 1e-10));
                }
            }
        }
    }

    double localErrors[2] = {localMaxAbsoluteError, localMaxRelativeError};
    double globalErrors[2] = {0.0, 0.0};
    MPI_Reduce(localErrors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, 0,
               grid.world);

    if (grid.rank == 0) {
        std::printf("Max absolute error: %.10e\n", globalErrors[0]);
        std::printf("Max relative error: %.10e\n", globalErrors[1]);
        if (globalErrors[1] > 1e-6 || !std::isfinite(globalErrors[1])) {
            std::printf("Validation failed: relative error too large\n");
            return false;
        }
    }
    return true;
}

size_t packedElementCountForRank(const DistributedMatrix& matrix,
                                 const int rank,
                                 const ProcessGrid& grid) {
    const int processRow = rank / grid.columns;
    const int processColumn = rank % grid.columns;
    size_t count = 0;
    for (int blockRow = processRow; blockRow < matrix.blockCount;
         blockRow += grid.rows) {
        for (int blockColumn = processColumn;
             blockColumn <= blockRow && blockColumn < matrix.blockCount;
             blockColumn += grid.columns) {
            count += matrix.rowsInBlock(blockRow) *
                     matrix.rowsInBlock(blockColumn);
        }
    }
    return count;
}

std::vector<double> packLocalTiles(const DistributedMatrix& matrix,
                                   const ProcessGrid& grid) {
    std::vector<double> packed;
    packed.reserve(packedElementCountForRank(matrix, grid.rank, grid));
    for (int blockRow = grid.rowCoordinate; blockRow < matrix.blockCount;
         blockRow += grid.rows) {
        for (int blockColumn = grid.columnCoordinate;
             blockColumn <= blockRow && blockColumn < matrix.blockCount;
             blockColumn += grid.columns) {
            const std::vector<double>& source =
                matrix.tile(blockRow, blockColumn);
            packed.insert(packed.end(), source.begin(), source.end());
        }
    }
    return packed;
}

std::vector<double> gatherMatrix(const DistributedMatrix& matrix,
                                 const ProcessGrid& grid) {
    const std::vector<double> local = packLocalTiles(matrix, grid);
    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    std::vector<double> packed;

    if (grid.rank == 0) {
        receiveCounts.resize(grid.size);
        displacements.resize(grid.size);
        size_t total = 0;
        for (int rank = 0; rank < grid.size; ++rank) {
            receiveCounts[rank] = checkedMpiCount(
                packedElementCountForRank(matrix, rank, grid), grid);
            displacements[rank] = checkedMpiCount(total, grid);
            total += static_cast<size_t>(receiveCounts[rank]);
        }
        checkedMpiCount(total, grid);
        packed.resize(total);
    }

    MPI_Gatherv(local.data(), checkedMpiCount(local.size(), grid), MPI_DOUBLE,
                packed.data(), receiveCounts.data(), displacements.data(),
                MPI_DOUBLE, 0, grid.world);

    std::vector<double> full;
    if (grid.rank != 0) {
        return full;
    }
    if (matrix.n != 0 &&
        matrix.n > std::numeric_limits<size_t>::max() / matrix.n) {
        std::fprintf(stderr, "Error: matrix size is too large\n");
        MPI_Abort(grid.world, 2);
    }
    full.assign(matrix.n * matrix.n, 0.0);
    for (int rank = 0; rank < grid.size; ++rank) {
        const int processRow = rank / grid.columns;
        const int processColumn = rank % grid.columns;
        size_t offset = static_cast<size_t>(displacements[rank]);
        for (int blockRow = processRow; blockRow < matrix.blockCount;
             blockRow += grid.rows) {
            const size_t rows = matrix.rowsInBlock(blockRow);
            const size_t globalRow =
                static_cast<size_t>(blockRow) * matrix.tileSize;
            for (int blockColumn = processColumn;
                 blockColumn <= blockRow && blockColumn < matrix.blockCount;
                 blockColumn += grid.columns) {
                const size_t columns = matrix.rowsInBlock(blockColumn);
                const size_t globalColumn =
                    static_cast<size_t>(blockColumn) * matrix.tileSize;
                for (size_t row = 0; row < rows; ++row) {
                    std::copy_n(packed.data() + offset + row * columns, columns,
                                full.data() + (globalRow + row) * matrix.n +
                                    globalColumn);
                }
                offset += rows * columns;
            }
        }
    }
    return full;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, size_t& value) {
    if (text[0] == '\0' || text[0] == '-') {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (*end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int returnCode = 0;

    {
        ProcessGrid grid;
        size_t n = 512;
        bool validate = false;
        bool printResults = false;
        bool showHelp = false;
        bool argumentsValid = true;

        for (int argument = 1; argument < argc; ++argument) {
            if (std::strcmp(argv[argument], "-n") == 0 &&
                argument + 1 < argc) {
                if (!parseSize(argv[++argument], n)) {
                    if (grid.rank == 0) {
                        std::printf("Invalid matrix size: %s\n", argv[argument]);
                    }
                    argumentsValid = false;
                }
            } else if (std::strcmp(argv[argument], "-v") == 0) {
                validate = true;
            } else if (std::strcmp(argv[argument], "-r") == 0) {
                printResults = true;
            } else if (std::strcmp(argv[argument], "-h") == 0) {
                showHelp = true;
            } else {
                if (grid.rank == 0) {
                    std::printf("Unknown option: %s\n", argv[argument]);
                }
                argumentsValid = false;
            }
        }

        if (showHelp || !argumentsValid) {
            if (grid.rank == 0) {
                printUsage(argv[0]);
            }
            returnCode = argumentsValid ? 0 : 1;
        } else {
            const int tileSize = chooseTileSize(n, grid);
            if (grid.rank == 0) {
                std::printf("Cholesky Decomposition Benchmark\n");
                std::printf("Matrix size: %zu x %zu\n", n, n);
                std::printf("Validation: %s\n",
                            validate ? "enabled" : "disabled");
                std::printf("MPI processes: %d (%d x %d grid)\n", grid.size,
                            grid.rows, grid.columns);
                std::printf("Generating positive definite matrix...\n");
            }

            const bool canJump = randomJumpIsCompatible();
            DistributedMatrix matrix(n, tileSize, grid);
            generatePositiveDefiniteMatrix(matrix, canJump);
            std::unique_ptr<DistributedMatrix> original;
            if (validate) {
                original = std::make_unique<DistributedMatrix>(matrix);
            }

            if (grid.rank == 0) {
                std::printf("Computing Cholesky decomposition...\n");
            }
            MPI_Barrier(grid.world);
            const double start = MPI_Wtime();
            const bool success = choleskyDecomposition(matrix, grid);
            const double localElapsed = MPI_Wtime() - start;
            double elapsed = 0.0;
            MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                       grid.world);

            int localSuccess = success ? 1 : 0;
            int globalSuccess = 0;
            MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_LAND,
                          grid.world);
            if (!globalSuccess) {
                if (grid.rank == 0) {
                    std::printf("Cholesky decomposition failed\n");
                }
                returnCode = 1;
            } else {
                if (grid.rank == 0) {
                    const long long milliseconds =
                        static_cast<long long>(elapsed * 1000.0);
                    const double operations =
                        static_cast<double>(n) * static_cast<double>(n) *
                        static_cast<double>(n) / 3.0;
                    const double gflops =
                        elapsed > 0.0 ? operations / elapsed / 1e9 : 0.0;
                    std::printf("Computation time: %lld ms\n", milliseconds);
                    std::printf("Performance: %.3f GFLOPS\n", gflops);
                }

                if (printResults) {
                    std::vector<double> result = gatherMatrix(matrix, grid);
                    if (grid.rank == 0) {
                        print_results(result, "CholeskyL");
                    }
                }

                if (validate) {
                    if (grid.rank == 0) {
                        std::printf("Validating result...\n");
                    }
                    const bool valid =
                        validateCholesky(matrix, *original, grid, canJump);
                    int validInteger = valid ? 1 : 0;
                    MPI_Bcast(&validInteger, 1, MPI_INT, 0, grid.world);
                    if (grid.rank == 0) {
                        std::printf("Validation: %s\n",
                                    validInteger ? "PASSED" : "FAILED");
                    }
                    if (!validInteger) {
                        returnCode = 1;
                    }
                }
            }
        }
    }

    MPI_Finalize();
    return returnCode;
}
