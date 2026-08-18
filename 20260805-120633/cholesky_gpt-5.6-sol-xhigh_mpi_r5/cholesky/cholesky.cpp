#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// Small tiles expose ample distributed work at the benchmark's default size
// and keep the hand-vectorized update kernel in the first-level cache.
#ifndef CHOLESKY_BLOCK_SIZE
#define CHOLESKY_BLOCK_SIZE 32
#endif
constexpr int kBlockSize = CHOLESKY_BLOCK_SIZE;

struct Tile {
    int block_row;
    int block_col;
    std::vector<double> values;
};

struct DistributedMatrix {
    size_t n;
    int block_size;
    int num_blocks;
    int process_rows;
    int process_cols;
    int process_row;
    int process_col;
    std::vector<Tile> tiles;
    std::vector<int> tile_index;

    DistributedMatrix(size_t matrix_size, int bs, int pr, int pc, int my_row,
                      int my_col)
        : n(matrix_size),
          block_size(bs),
          num_blocks(static_cast<int>((matrix_size + bs - 1) / bs)),
          process_rows(pr),
          process_cols(pc),
          process_row(my_row),
          process_col(my_col),
          tile_index(static_cast<size_t>(num_blocks) * num_blocks, -1) {
        const size_t tile_elements = static_cast<size_t>(block_size) * block_size;
        for (int bi = 0; bi < num_blocks; ++bi) {
            if (bi % process_rows != process_row) {
                continue;
            }
            for (int bj = 0; bj <= bi; ++bj) {
                if (bj % process_cols != process_col) {
                    continue;
                }
                const int index = static_cast<int>(tiles.size());
                tile_index[static_cast<size_t>(bi) * num_blocks + bj] = index;
                tiles.push_back({bi, bj, std::vector<double>(tile_elements, 0.0)});
            }
        }
    }

    int extent(int block) const {
        const size_t begin = static_cast<size_t>(block) * block_size;
        return static_cast<int>(std::min(static_cast<size_t>(block_size), n - begin));
    }

    Tile* localTile(int bi, int bj) {
        const int index = tile_index[static_cast<size_t>(bi) * num_blocks + bj];
        return index < 0 ? nullptr : &tiles[static_cast<size_t>(index)];
    }
};

struct RandomRows {
    std::vector<size_t> offsets;
    std::vector<double> values;

    const double* row(size_t global_row) const {
        return values.data() + offsets[global_row];
    }

    void release() {
        std::vector<size_t>().swap(offsets);
        std::vector<double>().swap(values);
    }
};

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseArguments(int argc, char** argv, size_t& n, bool& validate,
                    bool& print_results_requested, bool& help_requested,
                    int rank) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) {
                    std::printf("Invalid matrix size: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                return false;
            }
            n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help_requested = true;
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return false;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return false;
        }
    }
    return true;
}

// rand_r is deliberately evaluated in the same order as the original
// benchmark. Each rank retains only rows needed by its local A tiles; rank 0
// retains every row only when serial validation has explicitly been requested.
RandomRows generateRandomRows(const DistributedMatrix& matrix,
                              bool retain_every_row) {
    RandomRows result{std::vector<size_t>(matrix.n,
                                         std::numeric_limits<size_t>::max()),
                      {}};
    size_t selected_rows = 0;
    for (size_t row = 0; row < matrix.n; ++row) {
        const int block = static_cast<int>(row / matrix.block_size);
        if (retain_every_row || block % matrix.process_rows == matrix.process_row ||
            block % matrix.process_cols == matrix.process_col) {
            result.offsets[row] = selected_rows++ * matrix.n;
        }
    }
    result.values.resize(selected_rows * matrix.n);

    unsigned int seed = 42;
    for (size_t row = 0; row < matrix.n; ++row) {
        double* destination = nullptr;
        if (result.offsets[row] != std::numeric_limits<size_t>::max()) {
            destination = result.values.data() + result.offsets[row];
        }
        for (size_t column = 0; column < matrix.n; ++column) {
            const double value =
                (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (destination != nullptr) {
                destination[column] = value;
            }
        }
    }
    return result;
}

void generatePositiveDefiniteMatrix(DistributedMatrix& matrix,
                                    const RandomRows& b) {
    const int ld = matrix.block_size;
    for (Tile& tile : matrix.tiles) {
        const int rows = matrix.extent(tile.block_row);
        const int cols = matrix.extent(tile.block_col);
        const size_t row_base = static_cast<size_t>(tile.block_row) * ld;
        const size_t col_base = static_cast<size_t>(tile.block_col) * ld;
        for (int ii = 0; ii < rows; ++ii) {
            const size_t global_i = row_base + static_cast<size_t>(ii);
            const double* const b_i = b.row(global_i);
            const int last_col = tile.block_row == tile.block_col
                                     ? std::min(cols, ii + 1)
                                     : cols;
            for (int jj = 0; jj < last_col; ++jj) {
                const size_t global_j = col_base + static_cast<size_t>(jj);
                const double* const b_j = b.row(global_j);
                double sum = 0.0;
                for (size_t k = 0; k < matrix.n; ++k) {
                    sum += b_i[k] * b_j[k];
                }
                if (global_i == global_j) {
                    sum += static_cast<double>(matrix.n);
                }
                tile.values[static_cast<size_t>(ii) * ld + jj] = sum;
            }
        }
    }
}

bool factorDiagonal(double* diagonal, int n, int ld) {
    for (int j = 0; j < n; ++j) {
        double diagonal_update = 0.0;
        for (int k = 0; k < j; ++k) {
            const double value = diagonal[static_cast<size_t>(j) * ld + k];
            diagonal_update += value * value;
        }
        const size_t jj = static_cast<size_t>(j) * ld + j;
        const double pivot = diagonal[jj] - diagonal_update;
        if (!(pivot > 0.0) || !std::isfinite(pivot)) {
            return false;
        }
        diagonal[jj] = std::sqrt(pivot);

        const double inverse_pivot = 1.0 / diagonal[jj];
        for (int i = j + 1; i < n; ++i) {
            double update = 0.0;
            for (int k = 0; k < j; ++k) {
                update += diagonal[static_cast<size_t>(i) * ld + k] *
                          diagonal[static_cast<size_t>(j) * ld + k];
            }
            const size_t ij = static_cast<size_t>(i) * ld + j;
            diagonal[ij] = (diagonal[ij] - update) * inverse_pivot;
        }
    }
    return true;
}

void triangularSolveRight(double* tile, const double* diagonal, int rows,
                          int columns, int ld) {
    for (int i = 0; i < rows; ++i) {
        double* const row = tile + static_cast<size_t>(i) * ld;
        for (int j = 0; j < columns; ++j) {
            double update = 0.0;
            const double* const diagonal_row =
                diagonal + static_cast<size_t>(j) * ld;
            for (int k = 0; k < j; ++k) {
                update += row[k] * diagonal_row[k];
            }
            row[j] = (row[j] - update) / diagonal_row[j];
        }
    }
}

// C -= X * Y^T. Y is communicated already transposed so that the innermost
// loop is contiguous and vectorizable. Diagonal tiles update only their lower
// triangle; off-diagonal lower tiles are dense.
void trailingUpdate(double* c, const double* x, const double* y_transposed,
                    int rows, int columns, int inner, int ld,
                    bool diagonal_tile) {
    for (int i = 0; i < rows; ++i) {
        double* const c_row = c + static_cast<size_t>(i) * ld;
        const double* const x_row = x + static_cast<size_t>(i) * ld;
        const int active_columns = diagonal_tile ? std::min(columns, i + 1) : columns;
        int j = 0;
        // Eight independent accumulators give the compiler a compact SIMD
        // microkernel while keeping C in registers for the complete dot product.
        for (; j + 7 < active_columns; j += 8) {
            double c0 = c_row[j];
            double c1 = c_row[j + 1];
            double c2 = c_row[j + 2];
            double c3 = c_row[j + 3];
            double c4 = c_row[j + 4];
            double c5 = c_row[j + 5];
            double c6 = c_row[j + 6];
            double c7 = c_row[j + 7];
            for (int k = 0; k < inner; ++k) {
                const double multiplier = x_row[k];
                const double* const y_row =
                    y_transposed + static_cast<size_t>(k) * ld + j;
                c0 -= multiplier * y_row[0];
                c1 -= multiplier * y_row[1];
                c2 -= multiplier * y_row[2];
                c3 -= multiplier * y_row[3];
                c4 -= multiplier * y_row[4];
                c5 -= multiplier * y_row[5];
                c6 -= multiplier * y_row[6];
                c7 -= multiplier * y_row[7];
            }
            c_row[j] = c0;
            c_row[j + 1] = c1;
            c_row[j + 2] = c2;
            c_row[j + 3] = c3;
            c_row[j + 4] = c4;
            c_row[j + 5] = c5;
            c_row[j + 6] = c6;
            c_row[j + 7] = c7;
        }
        for (; j < active_columns; ++j) {
            double value = c_row[j];
            for (int k = 0; k < inner; ++k) {
                value -= x_row[k] *
                         y_transposed[static_cast<size_t>(k) * ld + j];
            }
            c_row[j] = value;
        }
    }
}

bool distributedCholesky(DistributedMatrix& matrix, MPI_Comm row_communicator,
                         MPI_Comm column_communicator, int rank) {
    const int bs = matrix.block_size;
    const int tile_elements = bs * bs;
    const size_t tile_size = static_cast<size_t>(tile_elements);
    std::vector<double> diagonal(tile_size);
    std::vector<int> row_offsets(static_cast<size_t>(matrix.num_blocks));
    std::vector<int> column_offsets(static_cast<size_t>(matrix.num_blocks));
    std::vector<int> receive_counts(static_cast<size_t>(matrix.process_rows));
    std::vector<int> receive_displacements(static_cast<size_t>(matrix.process_rows));
    std::vector<double> row_panel;
    std::vector<double> send_panel;
    std::vector<double> column_panel;

    for (int block = 0; block < matrix.num_blocks; ++block) {
        const int diagonal_process_row = block % matrix.process_rows;
        const int diagonal_process_col = block % matrix.process_cols;
        const int diagonal_rank =
            diagonal_process_row * matrix.process_cols + diagonal_process_col;
        int factor_failed = 0;

        if (rank == diagonal_rank) {
            Tile* const tile = matrix.localTile(block, block);
            if (tile == nullptr ||
                !factorDiagonal(tile->values.data(), matrix.extent(block), bs)) {
                factor_failed = 1;
            } else {
                std::copy(tile->values.begin(), tile->values.end(), diagonal.begin());
            }
        }
        MPI_Bcast(&factor_failed, 1, MPI_INT, diagonal_rank, MPI_COMM_WORLD);
        if (factor_failed) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal block %d\n",
                            block);
            }
            return false;
        }
        if (matrix.process_col == diagonal_process_col) {
            MPI_Bcast(diagonal.data(), tile_elements, MPI_DOUBLE,
                      diagonal_process_row, column_communicator);
        }

        // The process column owning the current panel performs all TRSMs.
        if (matrix.process_col == diagonal_process_col) {
            const int inner = matrix.extent(block);
            for (int bi = block + 1; bi < matrix.num_blocks; ++bi) {
                if (bi % matrix.process_rows != matrix.process_row) {
                    continue;
                }
                Tile* const panel_tile = matrix.localTile(bi, block);
                triangularSolveRight(panel_tile->values.data(), diagonal.data(),
                                     matrix.extent(bi), inner, bs);
            }
        }

        // Broadcast each process row's portion of the factored panel. Packing
        // all tiles into one message avoids one collective per matrix tile.
        std::fill(row_offsets.begin(), row_offsets.end(), -1);
        int row_tile_count = 0;
        for (int bi = block + 1; bi < matrix.num_blocks; ++bi) {
            if (bi % matrix.process_rows == matrix.process_row) {
                row_offsets[static_cast<size_t>(bi)] = row_tile_count * tile_elements;
                ++row_tile_count;
            }
        }
        row_panel.resize(static_cast<size_t>(row_tile_count) * tile_size);
        if (matrix.process_col == diagonal_process_col) {
            for (int bi = block + 1; bi < matrix.num_blocks; ++bi) {
                const int offset = row_offsets[static_cast<size_t>(bi)];
                if (offset >= 0) {
                    const Tile* const tile = matrix.localTile(bi, block);
                    std::copy(tile->values.begin(), tile->values.end(),
                              row_panel.begin() + offset);
                }
            }
        }
        MPI_Bcast(row_panel.data(), row_tile_count * tile_elements, MPI_DOUBLE,
                  diagonal_process_col, row_communicator);

        // Within each process column, gather the panel tiles needed as right
        // operands. They are transposed during packing for the update kernel.
        int total_received = 0;
        for (int process_row = 0; process_row < matrix.process_rows; ++process_row) {
            int count = 0;
            for (int bj = block + 1; bj < matrix.num_blocks; ++bj) {
                if (bj % matrix.process_rows == process_row &&
                    bj % matrix.process_cols == matrix.process_col) {
                    ++count;
                }
            }
            receive_counts[static_cast<size_t>(process_row)] = count * tile_elements;
            receive_displacements[static_cast<size_t>(process_row)] = total_received;
            total_received += count * tile_elements;
        }

        const int send_count = receive_counts[static_cast<size_t>(matrix.process_row)];
        send_panel.resize(static_cast<size_t>(send_count));
        int send_offset = 0;
        for (int bj = block + 1; bj < matrix.num_blocks; ++bj) {
            if (bj % matrix.process_rows != matrix.process_row ||
                bj % matrix.process_cols != matrix.process_col) {
                continue;
            }
            const double* const source =
                row_panel.data() + row_offsets[static_cast<size_t>(bj)];
            double* const destination = send_panel.data() + send_offset;
            const int rows = matrix.extent(bj);
            const int inner = matrix.extent(block);
            for (int j = 0; j < rows; ++j) {
                for (int k = 0; k < inner; ++k) {
                    destination[static_cast<size_t>(k) * bs + j] =
                        source[static_cast<size_t>(j) * bs + k];
                }
            }
            send_offset += tile_elements;
        }

        column_panel.resize(static_cast<size_t>(total_received));
        MPI_Allgatherv(send_panel.data(), send_count, MPI_DOUBLE,
                       column_panel.data(), receive_counts.data(),
                       receive_displacements.data(), MPI_DOUBLE,
                       column_communicator);

        std::fill(column_offsets.begin(), column_offsets.end(), -1);
        for (int process_row = 0; process_row < matrix.process_rows; ++process_row) {
            int offset = receive_displacements[static_cast<size_t>(process_row)];
            for (int bj = block + 1; bj < matrix.num_blocks; ++bj) {
                if (bj % matrix.process_rows == process_row &&
                    bj % matrix.process_cols == matrix.process_col) {
                    column_offsets[static_cast<size_t>(bj)] = offset;
                    offset += tile_elements;
                }
            }
        }

        const int inner = matrix.extent(block);
        for (Tile& tile : matrix.tiles) {
            if (tile.block_col <= block) {
                continue;
            }
            const double* const left_panel =
                row_panel.data() + row_offsets[static_cast<size_t>(tile.block_row)];
            const double* const right_panel =
                column_panel.data() + column_offsets[static_cast<size_t>(tile.block_col)];
            trailingUpdate(tile.values.data(), left_panel, right_panel,
                           matrix.extent(tile.block_row),
                           matrix.extent(tile.block_col), inner, bs,
                           tile.block_row == tile.block_col);
        }
    }
    return true;
}

std::vector<double> gatherFactor(const DistributedMatrix& matrix, int rank,
                                 int world_size) {
    const int tile_elements = matrix.block_size * matrix.block_size;
    const size_t local_elements = matrix.tiles.size() *
                                  static_cast<size_t>(tile_elements);
    if (local_elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::fprintf(stderr, "Local result is too large for MPI_Gatherv\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    std::vector<double> packed(local_elements);
    for (size_t i = 0; i < matrix.tiles.size(); ++i) {
        std::copy(matrix.tiles[i].values.begin(), matrix.tiles[i].values.end(),
                  packed.begin() + i * static_cast<size_t>(tile_elements));
    }

    const int local_count = static_cast<int>(local_elements);
    std::vector<int> counts(rank == 0 ? static_cast<size_t>(world_size) : 0);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    std::vector<int> displacements;
    std::vector<double> gathered;
    if (rank == 0) {
        displacements.resize(static_cast<size_t>(world_size));
        long long total = 0;
        for (int process = 0; process < world_size; ++process) {
            if (total > std::numeric_limits<int>::max()) {
                std::fprintf(stderr, "Gathered result is too large for MPI_Gatherv\n");
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            displacements[static_cast<size_t>(process)] = static_cast<int>(total);
            total += counts[static_cast<size_t>(process)];
        }
        if (total > std::numeric_limits<int>::max()) {
            std::fprintf(stderr, "Gathered result is too large for MPI_Gatherv\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        gathered.resize(static_cast<size_t>(total));
    }

    MPI_Gatherv(packed.data(), local_count, MPI_DOUBLE, gathered.data(),
                counts.data(), displacements.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank != 0) {
        return {};
    }

    std::vector<double> result(matrix.n * matrix.n, 0.0);
    for (int process = 0; process < world_size; ++process) {
        const int process_row = process / matrix.process_cols;
        const int process_col = process % matrix.process_cols;
        size_t source_offset = static_cast<size_t>(displacements[static_cast<size_t>(process)]);
        for (int bi = 0; bi < matrix.num_blocks; ++bi) {
            if (bi % matrix.process_rows != process_row) {
                continue;
            }
            const int rows = matrix.extent(bi);
            for (int bj = 0; bj <= bi; ++bj) {
                if (bj % matrix.process_cols != process_col) {
                    continue;
                }
                const int cols = matrix.extent(bj);
                for (int ii = 0; ii < rows; ++ii) {
                    const size_t global_i = static_cast<size_t>(bi) * matrix.block_size + ii;
                    const int last_col = bi == bj ? std::min(cols, ii + 1) : cols;
                    for (int jj = 0; jj < last_col; ++jj) {
                        const size_t global_j = static_cast<size_t>(bj) * matrix.block_size + jj;
                        result[global_i * matrix.n + global_j] =
                            gathered[source_offset + static_cast<size_t>(ii) *
                                     matrix.block_size + jj];
                    }
                }
                source_offset += static_cast<size_t>(tile_elements);
            }
        }
    }
    return result;
}

bool validateCholesky(const std::vector<double>& l,
                      const RandomRows& b, size_t n) {
    double max_error = 0.0;
    double max_relative_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            const size_t common = std::min(i, j);
            for (size_t k = 0; k <= common; ++k) {
                reconstructed += l[i * n + k] * l[j * n + k];
            }

            double original = 0.0;
            const double* const b_i = b.row(i);
            const double* const b_j = b.row(j);
            for (size_t k = 0; k < n; ++k) {
                original += b_i[k] * b_j[k];
            }
            if (i == j) {
                original += static_cast<double>(n);
            }

            const double error = std::fabs(reconstructed - original);
            max_error = std::max(max_error, error);
            max_relative_error = std::max(
                max_relative_error, error / (std::fabs(original) + 1.0e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", max_relative_error);
    if (max_relative_error > 1.0e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t n = 512;
    bool validate = false;
    bool print_results_requested = false;
    bool help_requested = false;
    const bool arguments_ok =
        parseArguments(argc, argv, n, validate, print_results_requested,
                       help_requested, rank);
    if (!arguments_ok) {
        MPI_Finalize();
        return help_requested ? 0 : 1;
    }

    if (n > std::numeric_limits<size_t>::max() / n) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dimensions);
    const int process_rows = dimensions[0];
    const int process_cols = dimensions[1];
    const int process_row = rank / process_cols;
    const int process_col = rank % process_cols;

    MPI_Comm row_communicator = MPI_COMM_NULL;
    MPI_Comm column_communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, process_row, process_col, &row_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, process_col, process_row, &column_communicator);

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (%d x %d grid)\n", world_size,
                    process_rows, process_cols);
        std::printf("Generating positive definite matrix...\n");
    }

    DistributedMatrix matrix(n, kBlockSize, process_rows, process_cols,
                             process_row, process_col);
    RandomRows b = generateRandomRows(matrix, validate && rank == 0);
    generatePositiveDefiniteMatrix(matrix, b);
    if (!validate || rank != 0) {
        b.release();
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(matrix, row_communicator,
                                             column_communicator, rank);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Comm_free(&row_communicator);
        MPI_Comm_free(&column_communicator);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const double operations = static_cast<double>(n) *
                                  static_cast<double>(n) *
                                  static_cast<double>(n) / 3.0;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", operations / seconds / 1.0e9);
    }

    std::vector<double> result;
    if (print_results_requested || validate) {
        result = gatherFactor(matrix, rank, world_size);
    }

    int exit_code = 0;
    if (rank == 0 && print_results_requested) {
        print_results(result, "CholeskyL");
    }
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        if (validateCholesky(result, b, n)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exit_code = 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&row_communicator);
    MPI_Comm_free(&column_communicator);
    MPI_Finalize();
    return exit_code;
}
