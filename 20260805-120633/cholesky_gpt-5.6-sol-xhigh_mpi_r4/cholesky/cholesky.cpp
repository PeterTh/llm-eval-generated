#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// The matrix is distributed by square blocks over a two-dimensional process
// grid.  Only lower-triangular blocks are stored; block (i,j) belongs to
// process (i % process_rows, j % process_cols).
struct ProcessGrid {
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Comm row = MPI_COMM_NULL;
    MPI_Comm column = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int rows = 1;
    int columns = 1;
    int row_coordinate = 0;
    int column_coordinate = 0;
};

[[noreturn]] void abortMpi(MPI_Comm communicator, const char* operation,
                           const int error_code) {
    char error[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error_code, error, &length);
    int rank = -1;
    MPI_Comm_rank(communicator, &rank);
    std::fprintf(stderr, "Rank %d: %s failed: %.*s\n", rank, operation,
                 length, error);
    MPI_Abort(communicator, error_code);
    std::abort();
}

void mpiCheck(const int error_code, MPI_Comm communicator,
              const char* operation) {
    if (error_code != MPI_SUCCESS) {
        abortMpi(communicator, operation, error_code);
    }
}

ProcessGrid createProcessGrid() {
    ProcessGrid grid;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &grid.size), MPI_COMM_WORLD,
             "MPI_Comm_size");

    int dimensions[2] = {0, 0};
    mpiCheck(MPI_Dims_create(grid.size, 2, dimensions), MPI_COMM_WORLD,
             "MPI_Dims_create");
    grid.rows = dimensions[0];
    grid.columns = dimensions[1];

    const int periods[2] = {0, 0};
    mpiCheck(MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0,
                             &grid.cart),
             MPI_COMM_WORLD, "MPI_Cart_create");
    mpiCheck(MPI_Comm_rank(grid.cart, &grid.rank), grid.cart,
             "MPI_Comm_rank");

    int coordinates[2] = {0, 0};
    mpiCheck(MPI_Cart_coords(grid.cart, grid.rank, 2, coordinates), grid.cart,
             "MPI_Cart_coords");
    grid.row_coordinate = coordinates[0];
    grid.column_coordinate = coordinates[1];

    // The split keys make row-communicator ranks equal process-column
    // coordinates and column-communicator ranks equal process-row coordinates.
    mpiCheck(MPI_Comm_split(grid.cart, grid.row_coordinate,
                            grid.column_coordinate, &grid.row),
             grid.cart, "MPI_Comm_split(row)");
    mpiCheck(MPI_Comm_split(grid.cart, grid.column_coordinate,
                            grid.row_coordinate, &grid.column),
             grid.cart, "MPI_Comm_split(column)");
    return grid;
}

void destroyProcessGrid(ProcessGrid& grid) {
    if (grid.row != MPI_COMM_NULL) {
        MPI_Comm_free(&grid.row);
    }
    if (grid.column != MPI_COMM_NULL) {
        MPI_Comm_free(&grid.column);
    }
    if (grid.cart != MPI_COMM_NULL) {
        MPI_Comm_free(&grid.cart);
    }
}

size_t chooseBlockSize(const size_t n, const ProcessGrid& grid) {
    // Small cache-resident blocks give the custom kernels good locality.  At
    // least four block cycles per process-grid dimension also smooth out the
    // shrinking triangular workload late in the factorization.
    const size_t largest_dimension =
        static_cast<size_t>(std::max(grid.rows, grid.columns));
    size_t block_size = n / std::max<size_t>(1, 4 * largest_dimension);
    block_size = ((block_size + 8) / 16) * 16;
    block_size = std::clamp<size_t>(block_size, 16, 64);
    block_size = std::min(block_size, n);

    // This is deliberately an environment tuning knob rather than a new CLI
    // option, so the benchmark's command-line interface remains unchanged.
    if (const char* value = std::getenv("CHOLESKY_BLOCK_SIZE")) {
        char* end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (errno == 0 && end != value && *end == '\0' && parsed > 0 &&
            parsed <= n) {
            block_size = static_cast<size_t>(parsed);
        }
    }
    return block_size;
}

class DistributedLowerMatrix {
  public:
    DistributedLowerMatrix(const size_t matrix_size, const size_t block_size,
                           const ProcessGrid& grid)
        : n_(matrix_size), block_size_(block_size),
          block_count_((matrix_size + block_size - 1) / block_size),
          block_area_(block_size * block_size), grid_(grid),
          local_index_(block_count_ * block_count_, -1) {
        size_t local_blocks = 0;
        for (size_t i = 0; i < block_count_; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                if (owns(i, j)) {
                    local_index_[i * block_count_ + j] =
                        static_cast<long long>(local_blocks++);
                }
            }
        }
        if (local_blocks >
            std::numeric_limits<size_t>::max() / block_area_) {
            throw std::overflow_error("local matrix allocation is too large");
        }
        data_.assign(local_blocks * block_area_, 0.0);
    }

    size_t matrixSize() const { return n_; }
    size_t blockSize() const { return block_size_; }
    size_t blockCount() const { return block_count_; }
    size_t blockArea() const { return block_area_; }

    size_t activeSize(const size_t block) const {
        return std::min(block_size_, n_ - block * block_size_);
    }

    bool owns(const size_t block_row, const size_t block_column) const {
        return static_cast<int>(block_row % grid_.rows) ==
                   grid_.row_coordinate &&
               static_cast<int>(block_column % grid_.columns) ==
                   grid_.column_coordinate;
    }

    int ownerRank(const size_t block_row, const size_t block_column) const {
        int coordinates[2] = {
            static_cast<int>(block_row % grid_.rows),
            static_cast<int>(block_column % grid_.columns)};
        int owner = -1;
        mpiCheck(MPI_Cart_rank(grid_.cart, coordinates, &owner), grid_.cart,
                 "MPI_Cart_rank");
        return owner;
    }

    double* block(const size_t block_row, const size_t block_column) {
        const long long local =
            local_index_[block_row * block_count_ + block_column];
        assert(local >= 0);
        return data_.data() + static_cast<size_t>(local) * block_area_;
    }

    const double* block(const size_t block_row,
                        const size_t block_column) const {
        const long long local =
            local_index_[block_row * block_count_ + block_column];
        assert(local >= 0);
        return data_.data() + static_cast<size_t>(local) * block_area_;
    }

    std::vector<double>& localData() { return data_; }
    const std::vector<double>& localData() const { return data_; }

  private:
    size_t n_;
    size_t block_size_;
    size_t block_count_;
    size_t block_area_;
    const ProcessGrid& grid_;
    std::vector<long long> local_index_;
    std::vector<double> data_;
};

// Generate exactly the same rand_r stream and B*B^T matrix as the original
// benchmark.  A rank only retains B rows needed by the block rows and block
// columns of A that it owns.
void generatePositiveDefiniteMatrix(DistributedLowerMatrix& matrix,
                                    const ProcessGrid& grid) {
    const size_t n = matrix.matrixSize();
    const size_t block_size = matrix.blockSize();
    std::vector<long long> row_offset(n, -1);
    size_t needed_row_count = 0;

    for (size_t row = 0; row < n; ++row) {
        const size_t block = row / block_size;
        const bool needed_as_left =
            static_cast<int>(block % grid.rows) == grid.row_coordinate;
        const bool needed_as_right =
            static_cast<int>(block % grid.columns) == grid.column_coordinate;
        if (needed_as_left || needed_as_right) {
            row_offset[row] = static_cast<long long>(needed_row_count++);
        }
    }

    if (needed_row_count > std::numeric_limits<size_t>::max() / n) {
        throw std::overflow_error("matrix generator allocation is too large");
    }
    std::vector<double> needed_rows(needed_row_count * n);
    unsigned int seed = 42;
    for (size_t row = 0; row < n; ++row) {
        double* destination = row_offset[row] >= 0
                                  ? needed_rows.data() +
                                        static_cast<size_t>(row_offset[row]) * n
                                  : nullptr;
        for (size_t column = 0; column < n; ++column) {
            const double value =
                (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (destination != nullptr) {
                destination[column] = value;
            }
        }
    }

    const size_t blocks = matrix.blockCount();
    for (size_t block_row = 0; block_row < blocks; ++block_row) {
        for (size_t block_column = 0; block_column <= block_row;
             ++block_column) {
            if (!matrix.owns(block_row, block_column)) {
                continue;
            }
            const size_t row_count = matrix.activeSize(block_row);
            const size_t column_count = matrix.activeSize(block_column);
            double* a = matrix.block(block_row, block_column);
            for (size_t local_row = 0; local_row < row_count; ++local_row) {
                const size_t global_row = block_row * block_size + local_row;
                const double* left =
                    needed_rows.data() +
                    static_cast<size_t>(row_offset[global_row]) * n;
                for (size_t local_column = 0; local_column < column_count;
                     ++local_column) {
                    const size_t global_column =
                        block_column * block_size + local_column;
                    const double* right =
                        needed_rows.data() +
                        static_cast<size_t>(row_offset[global_column]) * n;
                    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                    for (size_t k = 0; k < n; ++k) {
                        sum += left[k] * right[k];
                    }
                    if (global_row == global_column) {
                        sum += static_cast<double>(n);
                    }
                    a[local_row * block_size + local_column] = sum;
                }
            }
        }
    }
}

bool factorDiagonalBlock(double* diagonal, const size_t active_size,
                         const size_t stride, size_t& failed_diagonal) {
    for (size_t j = 0; j < active_size; ++j) {
        double value = diagonal[j * stride + j];
#pragma omp simd reduction(- : value)
        for (size_t k = 0; k < j; ++k) {
            value -= diagonal[j * stride + k] * diagonal[j * stride + k];
        }
        if (!(value > 0.0) || !std::isfinite(value)) {
            failed_diagonal = j;
            return false;
        }
        diagonal[j * stride + j] = std::sqrt(value);

        const double reciprocal = 1.0 / diagonal[j * stride + j];
        for (size_t i = j + 1; i < active_size; ++i) {
            double entry = diagonal[i * stride + j];
#pragma omp simd reduction(- : entry)
            for (size_t k = 0; k < j; ++k) {
                entry -= diagonal[i * stride + k] *
                         diagonal[j * stride + k];
            }
            diagonal[i * stride + j] = entry * reciprocal;
        }
        std::fill(diagonal + j * stride + j + 1,
                  diagonal + j * stride + active_size, 0.0);
    }
    return true;
}

void solvePanelBlock(double* panel, const double* diagonal,
                     const size_t panel_rows, const size_t diagonal_size,
                     const size_t stride) {
    for (size_t row = 0; row < panel_rows; ++row) {
        double* values = panel + row * stride;
        for (size_t j = 0; j < diagonal_size; ++j) {
            double entry = values[j];
#pragma omp simd reduction(- : entry)
            for (size_t k = 0; k < j; ++k) {
                entry -= values[k] * diagonal[j * stride + k];
            }
            values[j] = entry / diagonal[j * stride + j];
        }
    }
}

void updateTrailingBlock(double* target, const double* left,
                         const double* right, const size_t target_rows,
                         const size_t target_columns, const size_t panel_width,
                         const size_t stride, const bool diagonal_block) {
    for (size_t i = 0; i < target_rows; ++i) {
        const size_t last_column =
            diagonal_block ? std::min(i + 1, target_columns) : target_columns;
        const double* left_row = left + i * stride;
        size_t j = 0;
        // Four simultaneous dot products reuse L(i,k) and substantially reduce
        // load traffic without requiring an external BLAS implementation.
        for (; j + 3 < last_column; j += 4) {
            double value0 = target[i * stride + j];
            double value1 = target[i * stride + j + 1];
            double value2 = target[i * stride + j + 2];
            double value3 = target[i * stride + j + 3];
            const double* right0 = right + j * stride;
            const double* right1 = right0 + stride;
            const double* right2 = right1 + stride;
            const double* right3 = right2 + stride;
#pragma omp simd reduction(- : value0, value1, value2, value3)
            for (size_t k = 0; k < panel_width; ++k) {
                const double left_value = left_row[k];
                value0 -= left_value * right0[k];
                value1 -= left_value * right1[k];
                value2 -= left_value * right2[k];
                value3 -= left_value * right3[k];
            }
            target[i * stride + j] = value0;
            target[i * stride + j + 1] = value1;
            target[i * stride + j + 2] = value2;
            target[i * stride + j + 3] = value3;
        }
        for (; j < last_column; ++j) {
            double value = target[i * stride + j];
            const double* right_row = right + j * stride;
#pragma omp simd reduction(- : value)
            for (size_t k = 0; k < panel_width; ++k) {
                value -= left_row[k] * right_row[k];
            }
            target[i * stride + j] = value;
        }
    }
}

bool distributedCholesky(DistributedLowerMatrix& matrix,
                         const ProcessGrid& grid,
                         size_t& failed_global_diagonal) {
    const size_t block_size = matrix.blockSize();
    const size_t block_area = matrix.blockArea();
    const size_t block_count = matrix.blockCount();

    // Every process row caches the factor blocks whose block-row it owns.
    const size_t row_panel_slots =
        (block_count + static_cast<size_t>(grid.rows) - 1) / grid.rows;
    std::vector<double> row_panel_workspace(row_panel_slots * block_area);
    std::vector<double*> row_panels(block_count, nullptr);
    std::vector<MPI_Request> row_requests;
    row_requests.reserve(row_panel_slots);

    // Double buffering overlaps the next process-column broadcast with the
    // local matrix multiplication for the current trailing block column.
    std::vector<double> column_workspace(2 * block_area);

    for (size_t k_block = 0; k_block < block_count; ++k_block) {
        const int diagonal_process_row =
            static_cast<int>(k_block % grid.rows);
        const int diagonal_process_column =
            static_cast<int>(k_block % grid.columns);
        const size_t diagonal_size = matrix.activeSize(k_block);
        int factorization_ok = 1;
        size_t failed_in_block = 0;

        if (grid.row_coordinate == diagonal_process_row &&
            grid.column_coordinate == diagonal_process_column) {
            factorization_ok =
                factorDiagonalBlock(matrix.block(k_block, k_block),
                                    diagonal_size, block_size, failed_in_block)
                    ? 1
                    : 0;
        }

        const int diagonal_rank = matrix.ownerRank(k_block, k_block);
        mpiCheck(MPI_Bcast(&factorization_ok, 1, MPI_INT, diagonal_rank,
                           grid.cart),
                 grid.cart, "MPI_Bcast(factorization status)");
        if (!factorization_ok) {
            unsigned long long failed =
                static_cast<unsigned long long>(failed_in_block);
            mpiCheck(MPI_Bcast(&failed, 1, MPI_UNSIGNED_LONG_LONG,
                               diagonal_rank, grid.cart),
                     grid.cart, "MPI_Bcast(failed diagonal)");
            failed_global_diagonal =
                k_block * block_size + static_cast<size_t>(failed);
            return false;
        }

        // Broadcast L(k,k) down its process column, then solve all local
        // L(i,k) blocks in that column.
        if (grid.column_coordinate == diagonal_process_column) {
            double* diagonal =
                grid.row_coordinate == diagonal_process_row
                    ? matrix.block(k_block, k_block)
                    : column_workspace.data();
            mpiCheck(MPI_Bcast(diagonal,
                               static_cast<int>(diagonal_size * block_size),
                               MPI_DOUBLE, diagonal_process_row, grid.column),
                     grid.column, "MPI_Bcast(diagonal block)");

            for (size_t i_block = k_block + 1; i_block < block_count;
                 ++i_block) {
                if (static_cast<int>(i_block % grid.rows) ==
                    grid.row_coordinate) {
                    solvePanelBlock(matrix.block(i_block, k_block), diagonal,
                                    matrix.activeSize(i_block), diagonal_size,
                                    block_size);
                }
            }
        }

        if (k_block + 1 == block_count) {
            continue;
        }

        std::fill(row_panels.begin(), row_panels.end(), nullptr);
        row_requests.clear();

        // Broadcast all newly solved L(i,k) blocks across their process rows.
        // Nonblocking collectives allow MPI to schedule the independent panel
        // transfers together.
        for (size_t i_block = k_block + 1; i_block < block_count; ++i_block) {
            if (static_cast<int>(i_block % grid.rows) != grid.row_coordinate) {
                continue;
            }
            double* panel = nullptr;
            if (grid.column_coordinate == diagonal_process_column) {
                panel = matrix.block(i_block, k_block);
            } else {
                const size_t slot = i_block / static_cast<size_t>(grid.rows);
                panel = row_panel_workspace.data() + slot * block_area;
            }
            row_panels[i_block] = panel;
            MPI_Request request = MPI_REQUEST_NULL;
            mpiCheck(MPI_Ibcast(
                         panel,
                         static_cast<int>(matrix.activeSize(i_block) *
                                          block_size),
                         MPI_DOUBLE, diagonal_process_column, grid.row,
                         &request),
                     grid.row, "MPI_Ibcast(panel row)");
            row_requests.push_back(request);
        }
        if (!row_requests.empty()) {
            mpiCheck(MPI_Waitall(static_cast<int>(row_requests.size()),
                                 row_requests.data(), MPI_STATUSES_IGNORE),
                     grid.row, "MPI_Waitall(panel rows)");
        }

        // Each process column receives the L(j,k) blocks for its local block
        // columns.  Pipeline those broadcasts with local GEMM/SYRK updates.
        std::vector<size_t> local_block_columns;
        for (size_t j_block = k_block + 1; j_block < block_count; ++j_block) {
            if (static_cast<int>(j_block % grid.columns) ==
                grid.column_coordinate) {
                local_block_columns.push_back(j_block);
            }
        }

        MPI_Request column_request = MPI_REQUEST_NULL;
        double* broadcast_buffers[2] = {nullptr, nullptr};
        auto startColumnBroadcast = [&](const size_t j_block,
                                        const int slot) {
            const int root = static_cast<int>(j_block % grid.rows);
            broadcast_buffers[slot] =
                grid.row_coordinate == root
                    ? row_panels[j_block]
                    : column_workspace.data() +
                          static_cast<size_t>(slot) * block_area;
            mpiCheck(MPI_Ibcast(
                         broadcast_buffers[slot],
                         static_cast<int>(matrix.activeSize(j_block) *
                                          block_size),
                         MPI_DOUBLE, root, grid.column, &column_request),
                     grid.column, "MPI_Ibcast(panel column)");
        };

        if (!local_block_columns.empty()) {
            startColumnBroadcast(local_block_columns.front(), 0);
        }
        for (size_t column_index = 0;
             column_index < local_block_columns.size(); ++column_index) {
            const int slot = static_cast<int>(column_index % 2);
            mpiCheck(MPI_Wait(&column_request, MPI_STATUS_IGNORE), grid.column,
                     "MPI_Wait(panel column)");
            if (column_index + 1 < local_block_columns.size()) {
                startColumnBroadcast(local_block_columns[column_index + 1],
                                     1 - slot);
            }

            const size_t j_block = local_block_columns[column_index];
            const double* right = broadcast_buffers[slot];
            for (size_t i_block = j_block; i_block < block_count; ++i_block) {
                if (static_cast<int>(i_block % grid.rows) !=
                    grid.row_coordinate) {
                    continue;
                }
                updateTrailingBlock(
                    matrix.block(i_block, j_block), row_panels[i_block], right,
                    matrix.activeSize(i_block), matrix.activeSize(j_block),
                    diagonal_size, block_size, i_block == j_block);
            }
        }
    }
    return true;
}

std::vector<double> gatherMatrix(const DistributedLowerMatrix& matrix,
                                 const ProcessGrid& grid,
                                 const bool symmetric) {
    const size_t local_values = matrix.localData().size();
    int local_count_ok =
        local_values <= static_cast<size_t>(std::numeric_limits<int>::max());
    int all_counts_ok = 0;
    mpiCheck(MPI_Allreduce(&local_count_ok, &all_counts_ok, 1, MPI_INT,
                           MPI_LAND, grid.cart),
             grid.cart, "MPI_Allreduce(gather sizes)");
    if (!all_counts_ok) {
        if (grid.rank == 0) {
            std::fprintf(stderr,
                         "Cannot gather matrix: an MPI count exceeds INT_MAX\n");
        }
        MPI_Abort(grid.cart, EXIT_FAILURE);
    }

    const int local_count = static_cast<int>(local_values);
    std::vector<int> counts(grid.rank == 0 ? grid.size : 0);
    mpiCheck(MPI_Gather(&local_count, 1, MPI_INT,
                        grid.rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
                        grid.cart),
             grid.cart, "MPI_Gather(block counts)");

    std::vector<int> displacements;
    std::vector<double> packed;
    if (grid.rank == 0) {
        displacements.resize(grid.size);
        size_t total = 0;
        for (int process = 0; process < grid.size; ++process) {
            if (total >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
                std::fprintf(stderr,
                             "Cannot gather matrix: total MPI count exceeds "
                             "INT_MAX\n");
                MPI_Abort(grid.cart, EXIT_FAILURE);
            }
            displacements[process] = static_cast<int>(total);
            total += static_cast<size_t>(counts[process]);
        }
        if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr,
                         "Cannot gather matrix: total MPI count exceeds "
                         "INT_MAX\n");
            MPI_Abort(grid.cart, EXIT_FAILURE);
        }
        packed.resize(total);
    }

    mpiCheck(MPI_Gatherv(
                 matrix.localData().data(), local_count, MPI_DOUBLE,
                 grid.rank == 0 ? packed.data() : nullptr,
                 grid.rank == 0 ? counts.data() : nullptr,
                 grid.rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                 grid.cart),
             grid.cart, "MPI_Gatherv(matrix blocks)");

    std::vector<double> result;
    if (grid.rank != 0) {
        return result;
    }

    const size_t n = matrix.matrixSize();
    const size_t block_size = matrix.blockSize();
    const size_t block_area = matrix.blockArea();
    result.assign(n * n, 0.0);
    std::vector<size_t> cursors(grid.size);
    for (int process = 0; process < grid.size; ++process) {
        cursors[process] = static_cast<size_t>(displacements[process]);
    }

    for (size_t block_row = 0; block_row < matrix.blockCount(); ++block_row) {
        for (size_t block_column = 0; block_column <= block_row;
             ++block_column) {
            const int owner = matrix.ownerRank(block_row, block_column);
            const double* block = packed.data() + cursors[owner];
            cursors[owner] += block_area;
            const size_t row_count = matrix.activeSize(block_row);
            const size_t column_count = matrix.activeSize(block_column);
            for (size_t i = 0; i < row_count; ++i) {
                const size_t global_i = block_row * block_size + i;
                for (size_t j = 0; j < column_count; ++j) {
                    const size_t global_j = block_column * block_size + j;
                    const double value = block[i * block_size + j];
                    result[global_i * n + global_j] = value;
                    if (symmetric && block_row != block_column) {
                        result[global_j * n + global_i] = value;
                    }
                }
            }
        }
    }
    return result;
}

bool validateCholesky(const std::vector<double>& lower,
                      const std::vector<double>& original, const size_t n) {
    double max_error = 0.0;
    double relative_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            const size_t terms = std::min(i, j) + 1;
            for (size_t k = 0; k < terms; ++k) {
                reconstructed += lower[i * n + k] * lower[j * n + k];
            }
            const double reference = original[i * n + j];
            const double error = std::fabs(reconstructed - reference);
            max_error = std::max(max_error, error);
            relative_error =
                std::max(relative_error,
                         error / (std::fabs(reference) + 1.0e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", relative_error);
    if (relative_error > 1.0e-6) {
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

struct Options {
    size_t n = 512;
    bool validate = false;
    bool print_results = false;
    bool help = false;
    bool valid = true;
    std::string error;
};

Options parseOptions(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) {
                options.valid = false;
                options.error = "Invalid matrix size: " + std::string(argv[i]);
                return options;
            }
            options.n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.print_results = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            options.valid = false;
            options.error = "Unknown option: " + std::string(argv[i]);
            return options;
        }
    }
    if (options.n > std::numeric_limits<size_t>::max() / options.n) {
        options.valid = false;
        options.error = "Matrix size is too large";
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    const int init_error = MPI_Init(&argc, &argv);
    if (init_error != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init failed\n");
        return EXIT_FAILURE;
    }

    ProcessGrid grid = createProcessGrid();
    const Options options = parseOptions(argc, argv);
    if (!options.valid || options.help) {
        if (grid.rank == 0) {
            if (!options.valid) {
                std::printf("%s\n", options.error.c_str());
            }
            printUsage(argv[0]);
        }
        destroyProcessGrid(grid);
        MPI_Finalize();
        return options.valid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const size_t block_size = chooseBlockSize(options.n, grid);
    if (block_size > static_cast<size_t>(std::sqrt(
                         static_cast<double>(std::numeric_limits<int>::max())))) {
        if (grid.rank == 0) {
            std::fprintf(stderr, "Block size exceeds MPI count limits\n");
        }
        destroyProcessGrid(grid);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (grid.rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", options.n, options.n);
        std::printf("Validation: %s\n",
                    options.validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (%d x %d process grid)\n", grid.size,
                    grid.rows, grid.columns);
        std::printf("Block size: %zu\n", block_size);
        std::printf("Generating positive definite matrix...\n");
    }

    int return_code = EXIT_SUCCESS;
    try {
        DistributedLowerMatrix matrix(options.n, block_size, grid);
        generatePositiveDefiniteMatrix(matrix, grid);

        std::vector<double> original;
        if (options.validate) {
            original = gatherMatrix(matrix, grid, true);
        }

        if (grid.rank == 0) {
            std::printf("Computing Cholesky decomposition...\n");
        }
        mpiCheck(MPI_Barrier(grid.cart), grid.cart, "MPI_Barrier(start)");
        const double start = MPI_Wtime();

        size_t failed_diagonal = 0;
        const bool success =
            distributedCholesky(matrix, grid, failed_diagonal);

        const double local_elapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        mpiCheck(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                            grid.cart),
                 grid.cart, "MPI_Reduce(elapsed time)");

        if (!success) {
            if (grid.rank == 0) {
                std::printf(
                    "Error: Matrix is not positive definite at diagonal "
                    "element %zu\n",
                    failed_diagonal);
                std::printf("Cholesky decomposition failed\n");
            }
            return_code = EXIT_FAILURE;
        } else {
            if (grid.rank == 0) {
                std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
                const double size = static_cast<double>(options.n);
                const double operations = size * size * size / 3.0;
                const double gflops =
                    elapsed > 0.0 ? operations / elapsed / 1.0e9 : 0.0;
                std::printf("Performance: %.3f GFLOPS\n", gflops);
            }

            std::vector<double> lower;
            if (options.print_results || options.validate) {
                lower = gatherMatrix(matrix, grid, false);
            }
            if (grid.rank == 0 && options.print_results) {
                print_results(lower, "CholeskyL");
            }
            if (grid.rank == 0 && options.validate) {
                std::printf("Validating result...\n");
                if (validateCholesky(lower, original, options.n)) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    return_code = EXIT_FAILURE;
                }
            }
        }
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "Rank %d: %s\n", grid.rank, exception.what());
        MPI_Abort(grid.cart, EXIT_FAILURE);
        return_code = EXIT_FAILURE;
    }

    destroyProcessGrid(grid);
    MPI_Finalize();
    return return_code;
}
