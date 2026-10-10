#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

namespace {

constexpr int MaxTileWidth = 64;

// Tiles (i,j) belong to process (i % rows, j % cols). Only the lower
// triangle is stored. Row/column communicators keep panel traffic off the
// world communicator, and block cycling balances the shrinking trailing matrix.
struct ProcessGrid {
    int rank, size, rows, cols, row, col;
    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm colComm = MPI_COMM_NULL;

    ProcessGrid() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int dims[2] = {0, 0};
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

    ProcessGrid(const ProcessGrid&) = delete;
    ProcessGrid& operator=(const ProcessGrid&) = delete;

    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
};

int localCount(int blocks, int coordinate, int dimension) {
    return blocks <= coordinate ? 0 : (blocks - 1 - coordinate) / dimension + 1;
}

// Local index of the first block whose global index is at least 'first'.
int localBegin(int first, int coordinate, int dimension) {
    return first <= coordinate ? 0 : (first - coordinate - 1) / dimension + 1;
}

void sendDoubles(const double* data, size_t count, int destination, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        MPI_Send(data, chunk, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

void receiveDoubles(double* data, size_t count, int source, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        MPI_Recv(data, chunk, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

struct DistributedMatrix {
    const ProcessGrid& grid;
    size_t n;
    int block, blocks, localRows, localCols;
    size_t tileSize;
    std::vector<size_t> rowOffsets;
    std::vector<double> values;

    DistributedMatrix(const ProcessGrid& g, size_t order, int tileWidth)
        : grid(g), n(order), block(tileWidth),
          blocks(static_cast<int>((order + tileWidth - 1) / tileWidth)),
          localRows(localCount(blocks, g.row, g.rows)),
          localCols(localCount(blocks, g.col, g.cols)),
          tileSize(size_t(block) * block), rowOffsets(size_t(localRows) + 1, 0) {
        for (int r = 0; r < localRows; ++r) {
            const int i = g.row + r * g.rows;
            rowOffsets[r + 1] = rowOffsets[r] + localCount(i + 1, g.col, g.cols);
        }
        values.resize(rowOffsets.back() * tileSize, 0.0);
    }

    int width(int i) const {
        return static_cast<int>(std::min(size_t(block), n - size_t(i) * block));
    }

    double* tile(int i, int j) {
        return values.data() + (rowOffsets[i / grid.rows] + j / grid.cols) * tileSize;
    }

    const double* tile(int i, int j) const {
        return values.data() + (rowOffsets[i / grid.rows] + j / grid.cols) * tileSize;
    }
};

// A panel is broadcast once along each process row. The process columns then
// exchange the subset they need as right operands. Square grids need just one
// column broadcast; rectangular grids use one batched allgather. Scratch space
// is O(n * block / sqrt(p)), rather than a replicated n-by-n matrix.
class Panels {
    const ProcessGrid& grid;
    int block, localRows, localCols;
    size_t tileSize;
    std::vector<double> sendBuffer, receiveBuffer;
    std::vector<int> counts, displacements, positions;

public:
    std::vector<double> left, right;

    explicit Panels(const DistributedMatrix& matrix)
        : grid(matrix.grid), block(matrix.block),
          localRows(matrix.localRows), localCols(matrix.localCols),
          tileSize(matrix.tileSize),
          sendBuffer(std::max(size_t(1), size_t(localCols) * tileSize)),
          receiveBuffer(std::max(size_t(1), size_t(localCols) * tileSize)),
          counts(grid.rows), displacements(grid.rows), positions(grid.rows),
          left(std::max(size_t(1), size_t(localRows) * tileSize)),
          right(std::max(size_t(1), size_t(localCols) * tileSize)) {}

    void exchange(int k, int first) {
        const int rowStart = std::min(localRows, localBegin(first, grid.row, grid.rows));
        const int colStart = std::min(localCols, localBegin(first, grid.col, grid.cols));
        MPI_Bcast(left.data() + size_t(rowStart) * tileSize,
                  static_cast<int>(size_t(localRows - rowStart) * tileSize), MPI_DOUBLE,
                  k % grid.cols, grid.rowComm);

        if (grid.rows == grid.cols) {
            const size_t start = size_t(colStart) * tileSize;
            const size_t count = size_t(localCols - colStart) * tileSize;
            if (grid.row == grid.col) {
                std::copy_n(left.data() + start, count, right.data() + start);
            }
            MPI_Bcast(right.data() + start, static_cast<int>(count), MPI_DOUBLE,
                      grid.col, grid.colComm);
            // Transpose right operands so the update kernel vectorizes over
            // adjacent output columns, without floating-point reassociation.
            for (int c = colStart; c < localCols; ++c) {
                double* tile = right.data() + size_t(c) * tileSize;
                for (int i = 0; i < block; ++i) {
                    for (int j = 0; j < i; ++j) {
                        std::swap(tile[size_t(i) * block + j], tile[size_t(j) * block + i]);
                    }
                }
            }
        } else {
            std::fill(counts.begin(), counts.end(), 0);
            for (int c = colStart; c < localCols; ++c) {
                counts[(grid.col + c * grid.cols) % grid.rows] += static_cast<int>(tileSize);
            }
            int total = 0;
            for (int r = 0; r < grid.rows; ++r) {
                displacements[r] = total;
                total += counts[r];
            }
            int sent = 0;
            for (int c = colStart; c < localCols; ++c) {
                const int j = grid.col + c * grid.cols;
                if (j % grid.rows == grid.row) {
                    std::copy_n(left.data() + size_t(j / grid.rows) * tileSize,
                                tileSize, sendBuffer.data() + sent);
                    sent += static_cast<int>(tileSize);
                }
            }
            MPI_Allgatherv(sendBuffer.data(), sent, MPI_DOUBLE, receiveBuffer.data(),
                           counts.data(), displacements.data(), MPI_DOUBLE, grid.colComm);
            positions = displacements;
            for (int c = colStart; c < localCols; ++c) {
                const int owner = (grid.col + c * grid.cols) % grid.rows;
                const double* source = receiveBuffer.data() + positions[owner];
                double* target = right.data() + size_t(c) * tileSize;
                for (int i = 0; i < block; ++i) {
                    for (int j = 0; j < block; ++j) {
                        target[size_t(j) * block + i] = source[size_t(i) * block + j];
                    }
                }
                positions[owner] += static_cast<int>(tileSize);
            }
        }
    }

    void fromFactor(const DistributedMatrix& matrix, int k, int first) {
        if (grid.col == k % grid.cols) {
            for (int r = localBegin(first, grid.row, grid.rows); r < localRows; ++r) {
                const int i = grid.row + r * grid.rows;
                std::copy_n(matrix.tile(i, k), tileSize, left.data() + size_t(r) * tileSize);
            }
        }
        exchange(k, first);
    }
};

// Register-blocked matrix multiplication, with a transposed right panel.
// Independent output columns allow SIMD at -O3 without -ffast-math or BLAS.
// Each accumulator visits the inner dimension in its original order.
template <bool Subtract>
void multiplyTile(double* output, const double* left, const double* right,
                  int rows, int cols, int inner, int stride) {
    int i = 0;
    for (; i + 4 <= rows; i += 4) {
        int j = 0;
        for (; j + 8 <= cols; j += 8) {
#if defined(__AVX2__) && defined(__FMA__)
            // Eight vector accumulators fit in the AVX2 register file. Loads
            // are unaligned so neither tile padding nor MPI buffers require a
            // special allocator. The portable kernel below handles other CPUs.
            double* c0 = output + size_t(i) * stride + j;
            double* c1 = c0 + stride;
            double* c2 = c1 + stride;
            double* c3 = c2 + stride;
            __m256d s00 = _mm256_loadu_pd(c0), s01 = _mm256_loadu_pd(c0 + 4);
            __m256d s10 = _mm256_loadu_pd(c1), s11 = _mm256_loadu_pd(c1 + 4);
            __m256d s20 = _mm256_loadu_pd(c2), s21 = _mm256_loadu_pd(c2 + 4);
            __m256d s30 = _mm256_loadu_pd(c3), s31 = _mm256_loadu_pd(c3 + 4);
            const double* a = left + size_t(i) * stride;
            const double* b = right + j;
            for (int k = 0; k < inner; ++k, b += stride) {
                const __m256d b0 = _mm256_loadu_pd(b), b1 = _mm256_loadu_pd(b + 4);
                const __m256d a0 = _mm256_broadcast_sd(a + k);
                const __m256d a1 = _mm256_broadcast_sd(a + stride + k);
                const __m256d a2 = _mm256_broadcast_sd(a + 2 * stride + k);
                const __m256d a3 = _mm256_broadcast_sd(a + 3 * stride + k);
                if constexpr (Subtract) {
                    s00 = _mm256_fnmadd_pd(a0, b0, s00); s01 = _mm256_fnmadd_pd(a0, b1, s01);
                    s10 = _mm256_fnmadd_pd(a1, b0, s10); s11 = _mm256_fnmadd_pd(a1, b1, s11);
                    s20 = _mm256_fnmadd_pd(a2, b0, s20); s21 = _mm256_fnmadd_pd(a2, b1, s21);
                    s30 = _mm256_fnmadd_pd(a3, b0, s30); s31 = _mm256_fnmadd_pd(a3, b1, s31);
                } else {
                    s00 = _mm256_fmadd_pd(a0, b0, s00); s01 = _mm256_fmadd_pd(a0, b1, s01);
                    s10 = _mm256_fmadd_pd(a1, b0, s10); s11 = _mm256_fmadd_pd(a1, b1, s11);
                    s20 = _mm256_fmadd_pd(a2, b0, s20); s21 = _mm256_fmadd_pd(a2, b1, s21);
                    s30 = _mm256_fmadd_pd(a3, b0, s30); s31 = _mm256_fmadd_pd(a3, b1, s31);
                }
            }
            _mm256_storeu_pd(c0, s00); _mm256_storeu_pd(c0 + 4, s01);
            _mm256_storeu_pd(c1, s10); _mm256_storeu_pd(c1 + 4, s11);
            _mm256_storeu_pd(c2, s20); _mm256_storeu_pd(c2 + 4, s21);
            _mm256_storeu_pd(c3, s30); _mm256_storeu_pd(c3 + 4, s31);
#else
            double s0[8], s1[8], s2[8], s3[8];
            for (int c = 0; c < 8; ++c) {
                s0[c] = output[size_t(i) * stride + j + c];
                s1[c] = output[size_t(i + 1) * stride + j + c];
                s2[c] = output[size_t(i + 2) * stride + j + c];
                s3[c] = output[size_t(i + 3) * stride + j + c];
            }
            for (int k = 0; k < inner; ++k) {
                const double a0 = left[size_t(i) * stride + k];
                const double a1 = left[size_t(i + 1) * stride + k];
                const double a2 = left[size_t(i + 2) * stride + k];
                const double a3 = left[size_t(i + 3) * stride + k];
                const double* b = right + size_t(k) * stride + j;
                for (int c = 0; c < 8; ++c) {
                    if constexpr (Subtract) {
                        s0[c] -= a0 * b[c];
                        s1[c] -= a1 * b[c];
                        s2[c] -= a2 * b[c];
                        s3[c] -= a3 * b[c];
                    } else {
                        s0[c] += a0 * b[c];
                        s1[c] += a1 * b[c];
                        s2[c] += a2 * b[c];
                        s3[c] += a3 * b[c];
                    }
                }
            }
            for (int c = 0; c < 8; ++c) {
                output[size_t(i) * stride + j + c] = s0[c];
                output[size_t(i + 1) * stride + j + c] = s1[c];
                output[size_t(i + 2) * stride + j + c] = s2[c];
                output[size_t(i + 3) * stride + j + c] = s3[c];
            }
#endif
        }
        for (; j < cols; ++j) {
            for (int r = 0; r < 4; ++r) {
                double value = output[size_t(i + r) * stride + j];
                for (int k = 0; k < inner; ++k) {
                    const double product = left[size_t(i + r) * stride + k] * right[size_t(k) * stride + j];
                    if constexpr (Subtract) value -= product;
                    else value += product;
                }
                output[size_t(i + r) * stride + j] = value;
            }
        }
    }
    for (; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double value = output[size_t(i) * stride + j];
            for (int k = 0; k < inner; ++k) {
                const double product = left[size_t(i) * stride + k] * right[size_t(k) * stride + j];
                if constexpr (Subtract) value -= product;
                else value += product;
            }
            output[size_t(i) * stride + j] = value;
        }
    }
}

template <bool Subtract>
void updateMatrix(DistributedMatrix& matrix, const Panels& panels, int first, int inner) {
    const ProcessGrid& g = matrix.grid;
    const int colStart = localBegin(first, g.col, g.cols);
    for (int r = localBegin(first, g.row, g.rows); r < matrix.localRows; ++r) {
        const int i = g.row + r * g.rows;
        const double* left = panels.left.data() + size_t(r) * matrix.tileSize;
        for (int c = colStart; c < matrix.localCols; ++c) {
            const int j = g.col + c * g.cols;
            if (j > i) break;
            multiplyTile<Subtract>(matrix.tile(i, j), left,
                                   panels.right.data() + size_t(c) * matrix.tileSize,
                                   matrix.width(i), matrix.width(j), inner, matrix.block);
        }
    }
}

void updateNextColumn(DistributedMatrix& matrix, const Panels& panels, int column, int inner) {
    const ProcessGrid& g = matrix.grid;
    if (g.col != column % g.cols) return;
    const double* right = panels.right.data() + size_t(column / g.cols) * matrix.tileSize;
    for (int r = localBegin(column, g.row, g.rows); r < matrix.localRows; ++r) {
        const int i = g.row + r * g.rows;
        multiplyTile<true>(matrix.tile(i, column), panels.left.data() + size_t(r) * matrix.tileSize,
                           right, matrix.width(i), matrix.width(column), inner, matrix.block);
    }
}

void generatePositiveDefiniteMatrix(DistributedMatrix& matrix, Panels& panels) {
    const ProcessGrid& g = matrix.grid;
    const size_t tileSize = matrix.tileSize;
    // Distribute B in row stripes, preserving exactly the original rand_r
    // sequence. Rank zero holds only block*n random numbers at any time;
    // the cubic B*B^T calculation is distributed over the entire process grid.
    std::vector<double> randomTiles(std::max(size_t(1), size_t(matrix.localRows) * matrix.localCols * tileSize), 0.0);
    if (g.rank == 0) {
        std::vector<double> stripe(size_t(matrix.block) * matrix.n);
        std::vector<double> packed(size_t(localCount(matrix.blocks, 0, g.cols)) * tileSize);
        unsigned int seed = 42;
        for (int i = 0; i < matrix.blocks; ++i) {
            const int height = matrix.width(i);
            for (size_t element = 0; element < size_t(height) * matrix.n; ++element) {
                stripe[element] = (rand_r(&seed) / double(RAND_MAX)) - 0.5;
            }
            for (int c = 0; c < g.cols; ++c) {
                const int columns = localCount(matrix.blocks, c, g.cols);
                const size_t count = size_t(columns) * tileSize;
                if (count == 0) continue;
                const int destination = (i % g.rows) * g.cols + c;
                double* target = destination == 0
                    ? randomTiles.data() + size_t(i / g.rows) * columns * tileSize
                    : packed.data();
                std::fill_n(target, count, 0.0);
                for (int j = c, local = 0; j < matrix.blocks; j += g.cols, ++local) {
                    const int width = matrix.width(j);
                    for (int r = 0; r < height; ++r) {
                        std::copy_n(stripe.data() + size_t(r) * matrix.n + size_t(j) * matrix.block,
                                    width, target + size_t(local) * tileSize + size_t(r) * matrix.block);
                    }
                }
                if (destination != 0) sendDoubles(target, count, destination, 100);
            }
        }
    } else {
        const size_t count = size_t(matrix.localCols) * tileSize;
        for (int r = 0; r < matrix.localRows; ++r) {
            receiveDoubles(randomTiles.data() + size_t(r) * count, count, 0, 100);
        }
    }

    for (int k = 0; k < matrix.blocks; ++k) {
        if (g.col == k % g.cols) {
            for (int r = 0; r < matrix.localRows; ++r) {
                std::copy_n(randomTiles.data() + (size_t(r) * matrix.localCols + k / g.cols) * tileSize,
                            tileSize, panels.left.data() + size_t(r) * tileSize);
            }
        }
        panels.exchange(k, 0);
        updateMatrix<false>(matrix, panels, 0, matrix.width(k));
    }
    for (int k = 0; k < matrix.blocks; ++k) {
        if (g.rank == g.owner(k, k)) {
            double* diagonal = matrix.tile(k, k);
            for (int i = 0; i < matrix.width(k); ++i) {
                diagonal[size_t(i) * matrix.block + i] += matrix.n;
            }
        }
    }
}

// Factor a small diagonal tile. Return a one-based failing diagonal, or zero.
int factorDiagonal(double* tile, int width, int stride) {
    double column[MaxTileWidth];
    for (int j = 0; j < width; ++j) {
        const double value = tile[size_t(j) * stride + j];
        if (value <= 0.0 || !std::isfinite(value)) return j + 1;
        const double pivot = std::sqrt(value);
        tile[size_t(j) * stride + j] = pivot;
        for (int i = j + 1; i < width; ++i) {
            column[i] = tile[size_t(i) * stride + j] / pivot;
            tile[size_t(i) * stride + j] = column[i];
        }
        for (int i = j + 1; i < width; ++i) {
            double* row = tile + size_t(i) * stride;
            const double multiplier = column[i];
            for (int k = j + 1; k <= i; ++k) row[k] -= multiplier * column[k];
        }
    }
    for (int i = 0; i < width; ++i) {
        std::fill(tile + size_t(i) * stride + i + 1, tile + size_t(i + 1) * stride, 0.0);
    }
    return 0;
}

void solvePanel(double* tile, const double* transposedDiagonal, const double* inverse,
                int height, int width, int stride) {
    for (int i = 0; i < height; ++i) {
        double* row = tile + size_t(i) * stride;
        for (int j = 0; j < width; ++j) {
            const double value = row[j] * inverse[j];
            row[j] = value;
            const double* diagonalRow = transposedDiagonal + size_t(j) * stride;
            for (int k = j + 1; k < width; ++k) row[k] -= value * diagonalRow[k];
        }
    }
}

bool choleskyDecomposition(DistributedMatrix& matrix, Panels& panels) {
    const ProcessGrid& g = matrix.grid;
    // The final entry carries the failure status in the same broadcast as the
    // diagonal factor. Every rank follows the same exit path on a failed pivot.
    std::vector<double> diagonal(matrix.tileSize + 1, 0.0);
    std::vector<double> transposedDiagonal(matrix.tileSize);
    std::vector<double> inverse(matrix.block);
    MPI_Request diagonalReady = MPI_REQUEST_NULL;
    auto startDiagonal = [&](int k) {
        const int owner = g.owner(k, k);
        if (g.rank == owner) {
            double* tile = matrix.tile(k, k);
            diagonal[matrix.tileSize] = factorDiagonal(tile, matrix.width(k), matrix.block);
            std::copy_n(tile, matrix.tileSize, diagonal.data());
        }
        MPI_Ibcast(diagonal.data(), static_cast<int>(diagonal.size()), MPI_DOUBLE,
                   owner, MPI_COMM_WORLD, &diagonalReady);
    };
    if (matrix.blocks != 0) startDiagonal(0);
    for (int k = 0; k < matrix.blocks; ++k) {
        MPI_Wait(&diagonalReady, MPI_STATUS_IGNORE);
        const int width = matrix.width(k);
        if (diagonal[matrix.tileSize] != 0.0) {
            if (g.rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       size_t(k) * matrix.block + size_t(diagonal[matrix.tileSize]) - 1);
            }
            return false;
        }
        if (g.col == k % g.cols) {
            for (int i = 0; i < width; ++i) {
                inverse[i] = 1.0 / diagonal[size_t(i) * matrix.block + i];
                for (int j = i + 1; j < width; ++j) {
                    transposedDiagonal[size_t(i) * matrix.block + j] = diagonal[size_t(j) * matrix.block + i];
                }
            }
            for (int r = localBegin(k + 1, g.row, g.rows); r < matrix.localRows; ++r) {
                const int i = g.row + r * g.rows;
                solvePanel(matrix.tile(i, k), transposedDiagonal.data(), inverse.data(),
                           matrix.width(i), width, matrix.block);
            }
        }
        if (k + 1 < matrix.blocks) {
            panels.fromFactor(matrix, k, k + 1);
            // Update the next panel first. Its diagonal factor and broadcast
            // can proceed while ranks compute the remaining trailing tiles.
            updateNextColumn(matrix, panels, k + 1, width);
            startDiagonal(k + 1);
            updateMatrix<true>(matrix, panels, k + 2, width);
        }
    }
    return true;
}

bool validateCholesky(const DistributedMatrix& factor, const std::vector<double>& original,
                      Panels& panels) {
    DistributedMatrix reconstructed(factor.grid, factor.n, factor.block);
    for (int k = 0; k < factor.blocks; ++k) {
        panels.fromFactor(factor, k, k);
        updateMatrix<false>(reconstructed, panels, k, factor.width(k));
    }
    double localErrors[2] = {0.0, 0.0};
    for (int r = 0; r < factor.localRows; ++r) {
        const int i = factor.grid.row + r * factor.grid.rows;
        for (int c = 0; c < factor.localCols; ++c) {
            const int j = factor.grid.col + c * factor.grid.cols;
            if (j > i) break;
            const size_t start = (factor.rowOffsets[r] + c) * factor.tileSize;
            for (int row = 0; row < factor.width(i); ++row) {
                const int columns = i == j ? row + 1 : factor.width(j);
                for (int col = 0; col < columns; ++col) {
                    const size_t index = start + size_t(row) * factor.block + col;
                    const double value = reconstructed.values[index];
                    const double error = std::isfinite(value)
                        ? std::fabs(value - original[index]) : std::numeric_limits<double>::infinity();
                    localErrors[0] = std::max(localErrors[0], error);
                    localErrors[1] = std::max(localErrors[1], error / (std::fabs(original[index]) + 1e-10));
                }
            }
        }
    }
    double errors[2];
    MPI_Allreduce(localErrors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (factor.grid.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    // The omitted upper half has the same errors by symmetry.
    return errors[1] <= 1e-6;
}

// Gathering is needed only for the optional, order-dependent results hash.
// Chunked point-to-point transfers avoid MPI's int displacement/count limits.
void printDistributedResults(const DistributedMatrix& matrix) {
    const ProcessGrid& g = matrix.grid;
    if (g.rank != 0) {
        sendDoubles(matrix.values.data(), matrix.values.size(), 0, 200);
        return;
    }
    std::vector<double> result(matrix.n * matrix.n, 0.0);
    std::vector<double> incoming;
    for (int source = 0; source < g.size; ++source) {
        const int processRow = source / g.cols;
        const int processCol = source % g.cols;
        size_t tiles = 0;
        for (int i = processRow; i < matrix.blocks; i += g.rows) {
            tiles += localCount(i + 1, processCol, g.cols);
        }
        const double* data = matrix.values.data();
        if (source != 0) {
            incoming.resize(tiles * matrix.tileSize);
            receiveDoubles(incoming.data(), incoming.size(), source, 200);
            data = incoming.data();
        }
        size_t tile = 0;
        for (int i = processRow; i < matrix.blocks; i += g.rows) {
            for (int j = processCol; j <= i; j += g.cols, ++tile) {
                for (int row = 0; row < matrix.width(i); ++row) {
                    std::copy_n(data + tile * matrix.tileSize + size_t(row) * matrix.block,
                                matrix.width(j), result.data() + (size_t(i) * matrix.block + row) * matrix.n
                                                     + size_t(j) * matrix.block);
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

int runBenchmark(int argc, char** argv, int rank) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(argument, &end, 10);
            if (errno != 0 || end == argument || *end != '\0' || argument[0] == '-'
                || parsed > static_cast<unsigned long long>(INT_MAX)
                || (parsed != 0 && parsed > std::vector<double>().max_size() / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argument);
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    ProcessGrid grid;
    // Keep several tiles per grid dimension on small problems; larger ones
    // use cache-sized tiles to amortize communication and triangular solves.
    const size_t target = n / (size_t(4) * std::max(grid.rows, grid.cols));
    const int block = static_cast<int>(std::clamp((target / 8) * 8, size_t(8), size_t(MaxTileWidth)));
    const size_t blocks = (n + block - 1) / block;
    if (blocks * size_t(block) * block > size_t(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Matrix size exceeds MPI panel count limits\n");
        return 1;
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d process grid)\n", grid.size, grid.rows, grid.cols);
        printf("Generating positive definite matrix...\n");
    }
    DistributedMatrix matrix(grid, n, block);
    Panels panels(matrix);
    generatePositiveDefiniteMatrix(matrix, panels);
    std::vector<double> original;
    if (validate) original = matrix.values;

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(matrix, panels);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        const double ops = double(n) * double(n) * double(n) / 3.0;
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? ops / duration / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(matrix);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(matrix, original, panels);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // An allocation failure must not leave peers waiting in a collective.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
