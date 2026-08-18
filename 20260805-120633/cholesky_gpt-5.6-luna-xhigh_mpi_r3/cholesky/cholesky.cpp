#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// The matrix is distributed as fixed-size tiles in a 2D block-cyclic layout.
// Fixed-size storage makes the local kernels contiguous and avoids a large
// number of MPI derived datatypes.  Entries outside the matrix are padding and
// are always ignored by the kernels.
class DistributedMatrix {
public:
    DistributedMatrix(const std::size_t n,
                      const std::size_t block_size,
                      const int process_rows,
                      const int process_columns,
                      const int my_process_row,
                      const int my_process_column)
        : n_(n),
          block_size_(block_size),
          tile_area_(block_size * block_size),
          block_count_((n + block_size - 1) / block_size),
          process_rows_(process_rows),
          process_columns_(process_columns),
          my_process_row_(my_process_row),
          my_process_column_(my_process_column),
          local_block_rows_(localBlockCount(block_count_, process_rows, my_process_row)),
          local_block_columns_(localBlockCount(block_count_, process_columns, my_process_column)),
          values_(local_block_rows_ * local_block_columns_ * tile_area_, 0.0) {}

    std::size_t n() const { return n_; }
    std::size_t blockSize() const { return block_size_; }
    std::size_t tileArea() const { return tile_area_; }
    std::size_t blockCount() const { return block_count_; }
    int processRows() const { return process_rows_; }
    int processColumns() const { return process_columns_; }
    int myProcessRow() const { return my_process_row_; }
    int myProcessColumn() const { return my_process_column_; }
    std::size_t localBlockRows() const { return local_block_rows_; }
    std::size_t localBlockColumns() const { return local_block_columns_; }
    std::size_t storageSize() const { return values_.size(); }

    std::size_t blockRows(const std::size_t block_row) const {
        return std::min(block_size_, n_ - block_row * block_size_);
    }

    std::size_t blockColumns(const std::size_t block_column) const {
        return std::min(block_size_, n_ - block_column * block_size_);
    }

    double* block(const std::size_t block_row, const std::size_t block_column) {
        if (!owns(block_row, block_column)) {
            return nullptr;
        }
        const std::size_t local_row = (block_row - static_cast<std::size_t>(my_process_row_)) /
                                      static_cast<std::size_t>(process_rows_);
        const std::size_t local_column = (block_column - static_cast<std::size_t>(my_process_column_)) /
                                         static_cast<std::size_t>(process_columns_);
        return values_.data() +
               (local_row * local_block_columns_ + local_column) * tile_area_;
    }

    const double* block(const std::size_t block_row, const std::size_t block_column) const {
        if (!owns(block_row, block_column)) {
            return nullptr;
        }
        const std::size_t local_row = (block_row - static_cast<std::size_t>(my_process_row_)) /
                                      static_cast<std::size_t>(process_rows_);
        const std::size_t local_column = (block_column - static_cast<std::size_t>(my_process_column_)) /
                                         static_cast<std::size_t>(process_columns_);
        return values_.data() +
               (local_row * local_block_columns_ + local_column) * tile_area_;
    }

    std::vector<double>& values() { return values_; }
    const std::vector<double>& values() const { return values_; }

private:
    static std::size_t localBlockCount(const std::size_t block_count,
                                       const int process_count,
                                       const int coordinate) {
        if (block_count <= static_cast<std::size_t>(coordinate)) {
            return 0;
        }
        return (block_count - 1 - static_cast<std::size_t>(coordinate)) /
                   static_cast<std::size_t>(process_count) +
               1;
    }

    bool owns(const std::size_t block_row, const std::size_t block_column) const {
        return block_row < block_count_ && block_column < block_count_ &&
               block_row % static_cast<std::size_t>(process_rows_) ==
                   static_cast<std::size_t>(my_process_row_) &&
               block_column % static_cast<std::size_t>(process_columns_) ==
                   static_cast<std::size_t>(my_process_column_);
    }

    std::size_t n_;
    std::size_t block_size_;
    std::size_t tile_area_;
    std::size_t block_count_;
    int process_rows_;
    int process_columns_;
    int my_process_row_;
    int my_process_column_;
    std::size_t local_block_rows_;
    std::size_t local_block_columns_;
    std::vector<double> values_;
};

std::size_t checkedMatrixElements(const std::size_t n, const int rank) {
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large\n");
        }
        return 0;
    }
    return n * n;
}

int mpiCount(const std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return -1;
    }
    return static_cast<int>(count);
}

bool sendDoubles(const std::vector<double>& values,
                 const int destination,
                 const int tag,
                 MPI_Comm communicator) {
    const std::size_t maximum_count = static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (std::size_t offset = 0; offset < values.size();) {
        const std::size_t count = std::min(maximum_count, values.size() - offset);
        if (MPI_Send(values.data() + offset,
                     static_cast<int>(count),
                     MPI_DOUBLE,
                     destination,
                     tag,
                     communicator) != MPI_SUCCESS) {
            return false;
        }
        offset += count;
    }
    return true;
}

bool receiveDoubles(std::vector<double>& values,
                    const int source,
                    const int tag,
                    MPI_Comm communicator) {
    const std::size_t maximum_count = static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (std::size_t offset = 0; offset < values.size();) {
        const std::size_t count = std::min(maximum_count, values.size() - offset);
        if (MPI_Recv(values.data() + offset,
                     static_cast<int>(count),
                     MPI_DOUBLE,
                     source,
                     tag,
                     communicator,
                     MPI_STATUS_IGNORE) != MPI_SUCCESS) {
            return false;
        }
        offset += count;
    }
    return true;
}

// Choose enough tiles to keep a 2D grid busy while retaining a cache-friendly
// tile size.  The command line interface remains identical to the original.
std::size_t chooseBlockSize(const std::size_t n, const int process_rows, const int process_columns) {
    if (n == 0) {
        return 1;
    }

    const std::size_t grid_extent = static_cast<std::size_t>(std::max(process_rows, process_columns));
    const std::size_t target_tiles = std::max<std::size_t>(2, 2 * grid_extent);
    const std::size_t raw_block_size = (n + target_tiles - 1) / target_tiles;

    // Multiples of 16 give the simple kernels good SIMD-friendly dimensions;
    // the bounds prevent excessive panel communication or tiny tiles.
    std::size_t block_size = ((raw_block_size + 8) / 16) * 16;
    block_size = std::max<std::size_t>(16, block_size);
    block_size = std::min<std::size_t>(128, block_size);
    return std::min(block_size, n);
}

// Generate the same deterministic positive-definite input as the original
// benchmark.  It is generated only on rank 0 and then distributed, so the
// benchmark's timed region contains the distributed factorization only.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const std::size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (std::size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        A[i * n + i] += static_cast<double>(n);
    }
}

void packMatrixForRank(const std::vector<double>& full_matrix,
                       const std::size_t n,
                       const std::size_t block_size,
                       const std::size_t block_count,
                       const int process_rows,
                       const int process_columns,
                       const int process_row,
                       const int process_column,
                       std::vector<double>& packed) {
    const std::size_t tile_area = block_size * block_size;
    const std::size_t local_block_rows =
        block_count <= static_cast<std::size_t>(process_row)
            ? 0
            : (block_count - 1 - static_cast<std::size_t>(process_row)) /
                      static_cast<std::size_t>(process_rows) +
                  1;
    const std::size_t local_block_columns =
        block_count <= static_cast<std::size_t>(process_column)
            ? 0
            : (block_count - 1 - static_cast<std::size_t>(process_column)) /
                      static_cast<std::size_t>(process_columns) +
                  1;

    packed.assign(local_block_rows * local_block_columns * tile_area, 0.0);
    for (std::size_t local_row = 0; local_row < local_block_rows; ++local_row) {
        const std::size_t block_row = static_cast<std::size_t>(process_row) +
                                      local_row * static_cast<std::size_t>(process_rows);
        const std::size_t row_start = block_row * block_size;
        const std::size_t rows = std::min(block_size, n - row_start);
        for (std::size_t local_column = 0; local_column < local_block_columns; ++local_column) {
            const std::size_t block_column = static_cast<std::size_t>(process_column) +
                                             local_column * static_cast<std::size_t>(process_columns);
            const std::size_t column_start = block_column * block_size;
            const std::size_t columns = std::min(block_size, n - column_start);
            double* tile = packed.data() +
                           (local_row * local_block_columns + local_column) * tile_area;
            for (std::size_t i = 0; i < rows; ++i) {
                std::memcpy(tile + i * block_size,
                            full_matrix.data() + (row_start + i) * n + column_start,
                            columns * sizeof(double));
            }
        }
    }
}

void unpackMatrixForRank(const std::vector<double>& packed,
                         const std::size_t n,
                         const std::size_t block_size,
                         const std::size_t block_count,
                         const int process_rows,
                         const int process_columns,
                         const int process_row,
                         const int process_column,
                         std::vector<double>& full_matrix) {
    const std::size_t tile_area = block_size * block_size;
    const std::size_t local_block_rows =
        block_count <= static_cast<std::size_t>(process_row)
            ? 0
            : (block_count - 1 - static_cast<std::size_t>(process_row)) /
                      static_cast<std::size_t>(process_rows) +
                  1;
    const std::size_t local_block_columns =
        block_count <= static_cast<std::size_t>(process_column)
            ? 0
            : (block_count - 1 - static_cast<std::size_t>(process_column)) /
                      static_cast<std::size_t>(process_columns) +
                  1;

    for (std::size_t local_row = 0; local_row < local_block_rows; ++local_row) {
        const std::size_t block_row = static_cast<std::size_t>(process_row) +
                                      local_row * static_cast<std::size_t>(process_rows);
        const std::size_t row_start = block_row * block_size;
        const std::size_t rows = std::min(block_size, n - row_start);
        for (std::size_t local_column = 0; local_column < local_block_columns; ++local_column) {
            const std::size_t block_column = static_cast<std::size_t>(process_column) +
                                             local_column * static_cast<std::size_t>(process_columns);
            const std::size_t column_start = block_column * block_size;
            const std::size_t columns = std::min(block_size, n - column_start);
            const double* tile = packed.data() +
                                 (local_row * local_block_columns + local_column) * tile_area;
            for (std::size_t i = 0; i < rows; ++i) {
                std::memcpy(full_matrix.data() + (row_start + i) * n + column_start,
                            tile + i * block_size,
                            columns * sizeof(double));
            }
        }
    }
}

bool distributeMatrix(DistributedMatrix& matrix,
                      std::vector<double>& root_matrix,
                      MPI_Comm grid,
                      const int rank,
                      const int world_size,
                      const int process_rows,
                      const int process_columns) {
    const std::size_t block_count = matrix.blockCount();
    std::vector<double> packed;

    if (rank == 0) {
        for (int destination = 0; destination < world_size; ++destination) {
            int coordinates[2] = {0, 0};
            MPI_Cart_coords(grid, destination, 2, coordinates);
            packMatrixForRank(root_matrix,
                              matrix.n(),
                              matrix.blockSize(),
                              block_count,
                              process_rows,
                              process_columns,
                              coordinates[0],
                              coordinates[1],
                              packed);
            if (destination == 0) {
                matrix.values() = packed;
            } else if (!sendDoubles(packed, destination, 17, grid)) {
                return false;
            }
        }
    } else {
        if (!receiveDoubles(matrix.values(), 0, 17, grid)) {
            return false;
        }
    }

    return true;
}

bool gatherMatrix(const DistributedMatrix& matrix,
                  std::vector<double>& root_matrix,
                  MPI_Comm grid,
                  const int rank,
                  const int world_size,
                  const int process_rows,
                  const int process_columns) {
    const std::size_t block_count = matrix.blockCount();
    std::vector<double> packed;

    if (rank == 0) {
        root_matrix.assign(matrix.n() * matrix.n(), 0.0);
        for (int source = 0; source < world_size; ++source) {
            int coordinates[2] = {0, 0};
            MPI_Cart_coords(grid, source, 2, coordinates);
            if (source == 0) {
                packed = matrix.values();
            } else {
                const std::size_t local_block_rows =
                    block_count <= static_cast<std::size_t>(coordinates[0])
                        ? 0
                        : (block_count - 1 - static_cast<std::size_t>(coordinates[0])) /
                                  static_cast<std::size_t>(process_rows) +
                              1;
                const std::size_t local_block_columns =
                    block_count <= static_cast<std::size_t>(coordinates[1])
                        ? 0
                        : (block_count - 1 - static_cast<std::size_t>(coordinates[1])) /
                                  static_cast<std::size_t>(process_columns) +
                              1;
                packed.resize(local_block_rows * local_block_columns * matrix.tileArea());
                if (!receiveDoubles(packed, source, 23, grid)) {
                    return false;
                }
            }
            unpackMatrixForRank(packed,
                                matrix.n(),
                                matrix.blockSize(),
                                block_count,
                                process_rows,
                                process_columns,
                                coordinates[0],
                                coordinates[1],
                                root_matrix);
        }
    } else {
        if (!sendDoubles(matrix.values(), 0, 23, grid)) {
            return false;
        }
    }

    return true;
}

bool factorDiagonal(double* diagonal, const std::size_t size, const std::size_t stride, std::size_t& failure_index) {
    for (std::size_t j = 0; j < size; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < j; ++k) {
            const double value = diagonal[j * stride + k];
            sum += value * value;
        }

        const double value = diagonal[j * stride + j] - sum;
        if (value <= 0.0) {
            failure_index = j;
            return false;
        }
        diagonal[j * stride + j] = std::sqrt(value);

        for (std::size_t i = j + 1; i < size; ++i) {
            sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += diagonal[i * stride + k] * diagonal[j * stride + k];
            }
            diagonal[i * stride + j] =
                (diagonal[i * stride + j] - sum) / diagonal[j * stride + j];
        }
    }

    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = i + 1; j < size; ++j) {
            diagonal[i * stride + j] = 0.0;
        }
    }
    return true;
}

// Solve X * L^T = B, where L is the factored diagonal tile.  Each row is
// independent, so this kernel is easy to vectorize and has no synchronization.
void triangularSolve(double* panel,
                     const double* diagonal,
                     const std::size_t rows,
                     const std::size_t diagonal_size,
                     const std::size_t stride) {
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < diagonal_size; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += panel[i * stride + k] * diagonal[j * stride + k];
            }
            panel[i * stride + j] =
                (panel[i * stride + j] - sum) / diagonal[j * stride + j];
        }
    }
}

// C -= A * B^T.  For diagonal tiles only the lower triangle is stored as a
// meaningful result; skipping the upper half saves roughly half the work.
void symmetricOrGeneralUpdate(double* result,
                              const double* left,
                              const double* right,
                              const std::size_t result_rows,
                              const std::size_t result_columns,
                              const std::size_t inner,
                              const std::size_t stride,
                              const bool diagonal_tile) {
    for (std::size_t i = 0; i < result_rows; ++i) {
        const std::size_t last_column = diagonal_tile ? std::min(i, result_columns - 1) :
                                                         result_columns - 1;
        for (std::size_t k = 0; k < inner; ++k) {
            const double left_value = left[i * stride + k];
            for (std::size_t j = 0; j <= last_column; ++j) {
                result[i * stride + j] -= left_value * right[j * stride + k];
            }
        }
    }
}

// Right-looking blocked Cholesky on a 2D block-cyclic matrix.  A panel tile
// is broadcast across its process row and then down the destination process
// column, giving each owner of a trailing tile both operands for its local
// update without replicating the matrix.
bool choleskyDecomposition(DistributedMatrix& matrix,
                           MPI_Comm grid,
                           MPI_Comm row_comm,
                           MPI_Comm column_comm,
                           const int rank) {
    const std::size_t block_count = matrix.blockCount();
    const std::size_t block_size = matrix.blockSize();
    const std::size_t tile_area = matrix.tileArea();
    const int process_rows = matrix.processRows();
    const int process_columns = matrix.processColumns();
    const int my_process_row = matrix.myProcessRow();
    const int my_process_column = matrix.myProcessColumn();

    if (block_count == 0) {
        return true;
    }

    // Each rank retains only the panel tiles that can be used by its local
    // trailing tiles.  This is O(number of tiles * block_size^2), much smaller
    // than a replicated dense matrix for a large problem.
    std::vector<double> row_panels(block_count * tile_area, 0.0);
    std::vector<double> column_panels(block_count * tile_area, 0.0);
    std::vector<double> diagonal(tile_area, 0.0);

    for (std::size_t k = 0; k < block_count; ++k) {
        const std::size_t diagonal_size = matrix.blockRows(k);
        const int diagonal_process_row = static_cast<int>(k % static_cast<std::size_t>(process_rows));
        const int diagonal_process_column = static_cast<int>(k % static_cast<std::size_t>(process_columns));

        int local_success = 1;
        if (my_process_row == diagonal_process_row && my_process_column == diagonal_process_column) {
            double* diagonal_tile = matrix.block(k, k);
            std::copy(diagonal_tile, diagonal_tile + tile_area, diagonal.begin());
            std::size_t failure_index = 0;
            local_success = factorDiagonal(diagonal.data(), diagonal_size, block_size, failure_index) ? 1 : 0;
            if (!local_success && rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                            k * block_size + failure_index);
            }
            if (local_success) {
                std::copy(diagonal.begin(), diagonal.end(), diagonal_tile);
            }
        }

        // The diagonal factor is needed by every owner of a block in panel k.
        if (my_process_column == diagonal_process_column) {
            MPI_Bcast(diagonal.data(), mpiCount(diagonal_size * block_size), MPI_DOUBLE,
                      diagonal_process_row, column_comm);
        }

        int global_success = 0;
        MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, grid);
        if (!global_success) {
            return false;
        }

        // Broadcast the diagonal tile across its process row, and solve every
        // block below it in the panel on the owning process-row rank.
        for (std::size_t i = k; i < block_count; ++i) {
            if (my_process_row != static_cast<int>(i % static_cast<std::size_t>(process_rows))) {
                continue;
            }

            double* panel = row_panels.data() + i * tile_area;
            std::fill(panel, panel + tile_area, 0.0);
            if (my_process_column == diagonal_process_column) {
                if (i == k) {
                    std::copy(diagonal.begin(), diagonal.end(), panel);
                } else {
                    double* panel_tile = matrix.block(i, k);
                    triangularSolve(panel_tile, diagonal.data(), matrix.blockRows(i), diagonal_size, block_size);
                    std::copy(panel_tile, panel_tile + tile_area, panel);
                }
            }

            MPI_Bcast(panel,
                      mpiCount(matrix.blockRows(i) * block_size),
                      MPI_DOUBLE,
                      diagonal_process_column,
                      row_comm);
        }

        // Move each panel block from process row i to process column j.  The
        // second broadcast makes L(j,k) available to every owner of A(i,j).
        for (std::size_t j = k + 1; j < block_count; ++j) {
            if (my_process_column != static_cast<int>(j % static_cast<std::size_t>(process_columns))) {
                continue;
            }

            double* panel = column_panels.data() + j * tile_area;
            std::fill(panel, panel + tile_area, 0.0);
            if (my_process_row == static_cast<int>(j % static_cast<std::size_t>(process_rows))) {
                std::copy(row_panels.data() + j * tile_area,
                          row_panels.data() + (j + 1) * tile_area,
                          panel);
            }
            MPI_Bcast(panel,
                      mpiCount(matrix.blockRows(j) * block_size),
                      MPI_DOUBLE,
                      static_cast<int>(j % static_cast<std::size_t>(process_rows)),
                      column_comm);
        }

        // Every rank updates exactly the trailing tiles it owns.
        for (std::size_t local_i = 0; local_i < matrix.localBlockRows(); ++local_i) {
            const std::size_t i = static_cast<std::size_t>(my_process_row) +
                                  local_i * static_cast<std::size_t>(process_rows);
            if (i <= k) {
                continue;
            }
            const double* left = row_panels.data() + i * tile_area;
            for (std::size_t local_j = 0; local_j < matrix.localBlockColumns(); ++local_j) {
                const std::size_t j = static_cast<std::size_t>(my_process_column) +
                                      local_j * static_cast<std::size_t>(process_columns);
                if (j <= k || j > i) {
                    continue;
                }
                double* result = matrix.block(i, j);
                const double* right = column_panels.data() + j * tile_area;
                symmetricOrGeneralUpdate(result,
                                         left,
                                         right,
                                         matrix.blockRows(i),
                                         matrix.blockColumns(j),
                                         diagonal_size,
                                         block_size,
                                         i == j);
            }
        }
    }

    // The block kernels update only the lower triangle.  Clear the upper
    // triangle so gathered output has exactly the documented L representation.
    for (std::size_t local_i = 0; local_i < matrix.localBlockRows(); ++local_i) {
        const std::size_t block_row = static_cast<std::size_t>(my_process_row) +
                                      local_i * static_cast<std::size_t>(process_rows);
        for (std::size_t local_j = 0; local_j < matrix.localBlockColumns(); ++local_j) {
            const std::size_t block_column = static_cast<std::size_t>(my_process_column) +
                                             local_j * static_cast<std::size_t>(process_columns);
            double* tile = matrix.block(block_row, block_column);
            const std::size_t rows = matrix.blockRows(block_row);
            const std::size_t columns = matrix.blockColumns(block_column);
            for (std::size_t i = 0; i < rows; ++i) {
                const std::size_t global_row = block_row * block_size + i;
                for (std::size_t j = 0; j < columns; ++j) {
                    const std::size_t global_column = block_column * block_size + j;
                    if (global_row < global_column) {
                        tile[i * block_size + j] = 0.0;
                    }
                }
            }
        }
    }

    return true;
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig,
                      const std::size_t n) {
    std::vector<double> reconstructed(n * n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double max_error = 0.0;
    double relative_error = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        max_error = std::max(max_error, error);
        const double relative = error / (std::fabs(A_orig[i]) + 1e-10);
        relative_error = std::max(relative_error, relative);
    }

    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", relative_error);
    if (relative_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, std::size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (text[0] == '\0' || end == text || *end != '\0') {
        return false;
    }
    if (parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::size_t n = 512;
    bool validate = false;
    bool print_results_enabled = false;
    bool parse_ok = true;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            parse_ok = parseSize(argv[++i], n);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_enabled = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            parse_ok = false;
        }
        if (!parse_ok) {
            break;
        }
    }

    if (show_help && parse_ok) {
        if (world_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    const std::size_t matrix_elements = checkedMatrixElements(n, world_rank);
    if (!parse_ok || (n != 0 && matrix_elements == 0)) {
        if (world_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dimensions);
    int periods[2] = {0, 0};
    MPI_Comm grid = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &grid);

    int grid_rank = 0;
    MPI_Comm_rank(grid, &grid_rank);
    int coordinates[2] = {0, 0};
    MPI_Cart_coords(grid, grid_rank, 2, coordinates);

    MPI_Comm row_comm = MPI_COMM_NULL;
    MPI_Comm column_comm = MPI_COMM_NULL;
    MPI_Comm_split(grid, coordinates[0], coordinates[1], &row_comm);
    MPI_Comm_split(grid, coordinates[1], coordinates[0], &column_comm);

    const std::size_t block_size = chooseBlockSize(n, dimensions[0], dimensions[1]);
    DistributedMatrix matrix(n,
                             block_size,
                             dimensions[0],
                             dimensions[1],
                             coordinates[0],
                             coordinates[1]);

    std::vector<double> root_matrix;
    std::vector<double> original_matrix;
    if (grid_rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
        root_matrix.resize(matrix_elements);
        generatePositiveDefiniteMatrix(root_matrix, n);
        if (validate) {
            original_matrix = root_matrix;
        }
    }

    int distribution_ok = distributeMatrix(matrix,
                                           root_matrix,
                                           grid,
                                           grid_rank,
                                           world_size,
                                           dimensions[0],
                                           dimensions[1])
                              ? 1
                              : 0;
    int all_distribution_ok = 0;
    MPI_Allreduce(&distribution_ok, &all_distribution_ok, 1, MPI_INT, MPI_MIN, grid);
    if (!all_distribution_ok) {
        if (grid_rank == 0) {
            std::fprintf(stderr, "MPI matrix distribution failed\n");
        }
        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&column_comm);
        MPI_Comm_free(&grid);
        MPI_Finalize();
        return 1;
    }

    if (grid_rank == 0 && !validate && !print_results_enabled) {
        std::vector<double>().swap(root_matrix);
    }

    if (grid_rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(matrix, grid, row_comm, column_comm, grid_rank);
    const double elapsed = MPI_Wtime() - start;

    double maximum_elapsed = 0.0;
    MPI_Reduce(&elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, grid);

    int local_success = success ? 1 : 0;
    int global_success = 0;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, grid);
    if (!global_success) {
        if (grid_rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&column_comm);
        MPI_Comm_free(&grid);
        MPI_Finalize();
        return 1;
    }

    if (grid_rank == 0) {
        long duration_ms = static_cast<long>(maximum_elapsed * 1000.0);
        duration_ms = std::max<long>(1, duration_ms);
        std::printf("Computation time: %ld ms\n", duration_ms);
        const double operations = static_cast<double>(n) * static_cast<double>(n) *
                                  static_cast<double>(n) / 3.0;
        const double gflops = operations / maximum_elapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (validate || print_results_enabled) {
        const bool gather_ok = gatherMatrix(matrix,
                                            root_matrix,
                                            grid,
                                            grid_rank,
                                            world_size,
                                            dimensions[0],
                                            dimensions[1]);
        int local_gather_ok = gather_ok ? 1 : 0;
        int global_gather_ok = 0;
        MPI_Allreduce(&local_gather_ok, &global_gather_ok, 1, MPI_INT, MPI_MIN, grid);
        if (!global_gather_ok) {
            if (grid_rank == 0) {
                std::fprintf(stderr, "MPI result gathering failed\n");
            }
            MPI_Comm_free(&row_comm);
            MPI_Comm_free(&column_comm);
            MPI_Comm_free(&grid);
            MPI_Finalize();
            return 1;
        }

        if (grid_rank == 0) {
            if (print_results_enabled) {
                print_results(root_matrix, "CholeskyL");
            }
            if (validate) {
                std::printf("Validating result...\n");
                const bool valid = validateCholesky(root_matrix, original_matrix, n);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                MPI_Comm_free(&row_comm);
                MPI_Comm_free(&column_comm);
                MPI_Comm_free(&grid);
                MPI_Finalize();
                return valid ? 0 : 1;
            }
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&column_comm);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}
