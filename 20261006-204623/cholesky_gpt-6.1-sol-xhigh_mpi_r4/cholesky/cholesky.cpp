#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
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

// Row-major rank order makes communicator-local roots grid coordinates.
// A nearly square process grid minimizes panel communication per rank.
struct ProcessGrid {
    int rank, size, rows, cols, row, col, block;
    MPI_Comm rowComm, colComm;

    explicit ProcessGrid(size_t n) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        block = 64;
        // Keep small problems distributed as well, including partial tiles.
        while (block > 1 && (n + block - 1) / block < static_cast<size_t>(rows))
            block /= 2;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }

    ~ProcessGrid() {
        // Leave collective cleanup to MPI_Abort when unwinding an exception.
        if (std::uncaught_exceptions() == 0) {
            MPI_Comm_free(&rowComm);
            MPI_Comm_free(&colComm);
        }
    }

    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
};

// Only owned tiles exist; A and L store just the lower triangle. Edge tiles
// have zero padding and a fixed leading dimension for the local kernels.
class DistributedMatrix {
public:
    const ProcessGrid& grid;
    size_t n;
    int blocks, localRows, localCols;
    std::vector<std::vector<double>> tiles;

    DistributedMatrix(const ProcessGrid& g, size_t order, bool lower = true)
        : grid(g), n(order), blocks(static_cast<int>((n + g.block - 1) / g.block)),
          localRows(static_cast<int>((static_cast<size_t>(blocks) + g.rows - 1 - g.row) / g.rows)),
          localCols(static_cast<int>((static_cast<size_t>(blocks) + g.cols - 1 - g.col) / g.cols)),
          tiles(static_cast<size_t>(localRows) * localCols) {
        for (int i = 0; i < localRows; ++i)
            for (int j = 0; j < localCols; ++j)
                if (!lower || g.row + i * g.rows >= g.col + j * g.cols)
                    tiles[static_cast<size_t>(i) * localCols + j].resize(g.block * g.block, 0.0);
    }

    double* tile(int i, int j) {
        return tiles[static_cast<size_t>(i / grid.rows) * localCols + j / grid.cols].data();
    }
    const double* tile(int i, int j) const {
        return tiles[static_cast<size_t>(i / grid.rows) * localCols + j / grid.cols].data();
    }
    int extent(int i) const {
        return static_cast<int>(std::min(static_cast<size_t>(grid.block), n - static_cast<size_t>(i) * grid.block));
    }
};

void broadcast(double* data, size_t count, int root, MPI_Comm comm) {
    // Chunk counts for MPI implementations whose count argument is an int.
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, comm);
        data += chunk;
        count -= chunk;
    }
}

void transpose(double* tile, int b) {
    for (int i = 0; i < b; ++i)
        for (int j = 0; j < i; ++j)
            std::swap(tile[i * b + j], tile[j * b + i]);
}

// C += X*Y (or C -= X*Y). A 4x8 native microkernel keeps accumulators in
// registers throughout the depth loop. Other targets use contiguous loops
// suitable for compiler vectorization. Products accumulate in increasing k.
template<bool Subtract>
void matrixProduct(double* __restrict c, const double* __restrict x,
                   const double* __restrict y, int m, int n, int depth, int b) {
    int column = 0;
#if defined(__AVX2__) && defined(__FMA__)
    // Reuse the same narrow slice of Y across all output rows.
    for (; column + 7 < n; column += 8) {
        for (int i = 0; i + 3 < m; i += 4) {
            double* c0 = c + i * b;
            double* c1 = c0 + b;
            double* c2 = c1 + b;
            double* c3 = c2 + b;
            __m256d v00 = _mm256_loadu_pd(c0 + column), v01 = _mm256_loadu_pd(c0 + column + 4);
            __m256d v10 = _mm256_loadu_pd(c1 + column), v11 = _mm256_loadu_pd(c1 + column + 4);
            __m256d v20 = _mm256_loadu_pd(c2 + column), v21 = _mm256_loadu_pd(c2 + column + 4);
            __m256d v30 = _mm256_loadu_pd(c3 + column), v31 = _mm256_loadu_pd(c3 + column + 4);
            for (int k = 0; k < depth; ++k) {
                const double* yk = y + k * b + column;
                const __m256d y0 = _mm256_loadu_pd(yk), y1 = _mm256_loadu_pd(yk + 4);
                const __m256d a0 = _mm256_broadcast_sd(x + i * b + k);
                const __m256d a1 = _mm256_broadcast_sd(x + (i + 1) * b + k);
                const __m256d a2 = _mm256_broadcast_sd(x + (i + 2) * b + k);
                const __m256d a3 = _mm256_broadcast_sd(x + (i + 3) * b + k);
                if constexpr (Subtract) {
                    v00 = _mm256_fnmadd_pd(a0, y0, v00); v01 = _mm256_fnmadd_pd(a0, y1, v01);
                    v10 = _mm256_fnmadd_pd(a1, y0, v10); v11 = _mm256_fnmadd_pd(a1, y1, v11);
                    v20 = _mm256_fnmadd_pd(a2, y0, v20); v21 = _mm256_fnmadd_pd(a2, y1, v21);
                    v30 = _mm256_fnmadd_pd(a3, y0, v30); v31 = _mm256_fnmadd_pd(a3, y1, v31);
                } else {
                    v00 = _mm256_fmadd_pd(a0, y0, v00); v01 = _mm256_fmadd_pd(a0, y1, v01);
                    v10 = _mm256_fmadd_pd(a1, y0, v10); v11 = _mm256_fmadd_pd(a1, y1, v11);
                    v20 = _mm256_fmadd_pd(a2, y0, v20); v21 = _mm256_fmadd_pd(a2, y1, v21);
                    v30 = _mm256_fmadd_pd(a3, y0, v30); v31 = _mm256_fmadd_pd(a3, y1, v31);
                }
            }
            _mm256_storeu_pd(c0 + column, v00); _mm256_storeu_pd(c0 + column + 4, v01);
            _mm256_storeu_pd(c1 + column, v10); _mm256_storeu_pd(c1 + column + 4, v11);
            _mm256_storeu_pd(c2 + column, v20); _mm256_storeu_pd(c2 + column + 4, v21);
            _mm256_storeu_pd(c3 + column, v30); _mm256_storeu_pd(c3 + column + 4, v31);
        }
    }
#endif
    int i = 0;
    for (; i + 3 < m; i += 4) {
        if (column == n)
            continue;
        double* c0 = c + i * b;
        double* c1 = c0 + b;
        double* c2 = c1 + b;
        double* c3 = c2 + b;
        for (int k = 0; k < depth; ++k) {
            const double a0 = x[i * b + k], a1 = x[(i + 1) * b + k];
            const double a2 = x[(i + 2) * b + k], a3 = x[(i + 3) * b + k];
            const double* yk = y + k * b;
            for (int j = column; j < n; ++j) {
                if constexpr (Subtract) {
                    c0[j] -= a0 * yk[j]; c1[j] -= a1 * yk[j];
                    c2[j] -= a2 * yk[j]; c3[j] -= a3 * yk[j];
                } else {
                    c0[j] += a0 * yk[j]; c1[j] += a1 * yk[j];
                    c2[j] += a2 * yk[j]; c3[j] += a3 * yk[j];
                }
            }
        }
    }
    for (; i < m; ++i)
        for (int k = 0; k < depth; ++k) {
            const double a = x[i * b + k];
            for (int j = 0; j < n; ++j) {
                if constexpr (Subtract)
                    c[i * b + j] -= a * y[k * b + j];
                else
                    c[i * b + j] += a * y[k * b + j];
            }
        }
}

// Preserve the original rand_r stream (seed 42, global row order). Root
// buffers one block row; B and B^T are distributed rather than replicated.
void generateRandomMatrix(DistributedMatrix& B, DistributedMatrix& BT) {
    const ProcessGrid& g = B.grid;
    const int b = g.block, elements = b * b;
    std::vector<double> rows(g.rank == 0 ? static_cast<size_t>(b) * B.n : 0);
    std::vector<double> tile(elements), transposed(elements);
    unsigned int seed = 42;
    for (int i = 0; i < B.blocks; ++i) {
        const int height = B.extent(i);
        if (g.rank == 0)
            for (size_t index = 0; index < static_cast<size_t>(height) * B.n; ++index)
                rows[index] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        for (int j = 0; j < B.blocks; ++j) {
            const int owner = g.owner(i, j), transposeOwner = g.owner(j, i);
            if (g.rank == 0) {
                std::fill(tile.begin(), tile.end(), 0.0);
                std::fill(transposed.begin(), transposed.end(), 0.0);
                for (int r = 0; r < height; ++r)
                    for (int c = 0; c < B.extent(j); ++c) {
                        double value = rows[static_cast<size_t>(r) * B.n + static_cast<size_t>(j) * b + c];
                        tile[r * b + c] = value;
                        transposed[c * b + r] = value;
                    }
                if (owner == 0)
                    std::copy(tile.begin(), tile.end(), B.tile(i, j));
                else
                    MPI_Send(tile.data(), elements, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD);
                if (transposeOwner == 0)
                    std::copy(transposed.begin(), transposed.end(), BT.tile(j, i));
                else
                    MPI_Send(transposed.data(), elements, MPI_DOUBLE, transposeOwner, 1, MPI_COMM_WORLD);
            } else {
                if (g.rank == owner)
                    MPI_Recv(B.tile(i, j), elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (g.rank == transposeOwner)
                    MPI_Recv(BT.tile(j, i), elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

void generatePositiveDefiniteMatrix(DistributedMatrix& A) {
    const ProcessGrid& g = A.grid;
    const int b = g.block;
    const size_t elements = static_cast<size_t>(b) * b;
    DistributedMatrix B(g, A.n, false), BT(g, A.n, false);
    generateRandomMatrix(B, BT);
    std::vector<double> left(static_cast<size_t>(A.localRows) * elements);
    std::vector<double> right(static_cast<size_t>(A.localCols) * elements);
    // SUMMA broadcasts only the block rows/columns needed on each rank.
    for (int k = 0; k < A.blocks; ++k) {
        if (g.col == k % g.cols)
            for (int i = 0; i < A.localRows; ++i)
                std::copy_n(B.tile(g.row + i * g.rows, k), elements, left.data() + i * elements);
        if (g.row == k % g.rows)
            for (int j = 0; j < A.localCols; ++j)
                std::copy_n(BT.tile(k, g.col + j * g.cols), elements, right.data() + j * elements);
        broadcast(left.data(), left.size(), k % g.cols, g.rowComm);
        broadcast(right.data(), right.size(), k % g.rows, g.colComm);
        for (int i = 0; i < A.localRows; ++i) {
            int globalRow = g.row + i * g.rows;
            for (int j = 0; j < A.localCols; ++j) {
                int globalCol = g.col + j * g.cols;
                if (globalRow >= globalCol)
                    matrixProduct<false>(A.tile(globalRow, globalCol), left.data() + i * elements,
                                         right.data() + j * elements, A.extent(globalRow),
                                         A.extent(globalCol), A.extent(k), b);
            }
        }
    }
    for (int k = 0; k < A.blocks; ++k)
        if (g.rank == g.owner(k, k))
            for (int i = 0; i < A.extent(k); ++i)
                A.tile(k, k)[i * b + i] += A.n;
}

// First local tile whose global index is >= first, including empty suffixes.
int firstLocal(int first, int coordinate, int stride) {
    return first <= coordinate ? 0 : (first - coordinate + stride - 1) / stride;
}

// Broadcast a column of L across process rows and its transpose down process
// columns. Nonblocking transfers reroute transpose panels to one process row,
// also for rectangular grids, and overlap the row broadcast.
void broadcastFactorPanel(const DistributedMatrix& L, int k, int first,
                          std::vector<double>& left, std::vector<double>& right) {
    const ProcessGrid& g = L.grid;
    const int b = g.block, elements = b * b;
    const int rowStart = firstLocal(first, g.row, g.rows);
    const int colStart = firstLocal(first, g.col, g.cols);
    const int rootRow = k % g.rows, rootCol = k % g.cols;
    std::vector<MPI_Request> requests;
    requests.reserve(L.localRows + L.localCols);
    if (g.row == rootRow) {
        for (int j = colStart; j < L.localCols; ++j) {
            int globalCol = g.col + j * g.cols;
            int source = g.owner(globalCol, k);
            double* target = right.data() + static_cast<size_t>(j - colStart) * elements;
            if (source == g.rank)
                std::copy_n(L.tile(globalCol, k), elements, target);
            else {
                requests.push_back(MPI_REQUEST_NULL);
                MPI_Irecv(target, elements, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &requests.back());
            }
        }
    }
    if (g.col == rootCol) {
        for (int i = rowStart; i < L.localRows; ++i) {
            int globalRow = g.row + i * g.rows;
            const double* source = L.tile(globalRow, k);
            std::copy_n(source, elements, left.data() + static_cast<size_t>(i - rowStart) * elements);
            int target = rootRow * g.cols + globalRow % g.cols;
            if (target != g.rank) {
                requests.push_back(MPI_REQUEST_NULL);
                MPI_Isend(source, elements, MPI_DOUBLE, target, 2, MPI_COMM_WORLD, &requests.back());
            }
        }
    }
    broadcast(left.data(), static_cast<size_t>(L.localRows - rowStart) * elements, rootCol, g.rowComm);
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    if (g.row == rootRow)
        for (int j = colStart; j < L.localCols; ++j)
            transpose(right.data() + static_cast<size_t>(j - colStart) * elements, b);
    broadcast(right.data(), static_cast<size_t>(L.localCols - colStart) * elements, rootRow, g.colComm);
}

template<bool Subtract>
void updateTrailingMatrix(DistributedMatrix& A, int k, int first,
                          const std::vector<double>& left, const std::vector<double>& right) {
    const ProcessGrid& g = A.grid;
    const int b = g.block;
    const size_t elements = static_cast<size_t>(b) * b;
    const int rowStart = firstLocal(first, g.row, g.rows);
    const int colStart = firstLocal(first, g.col, g.cols);
    for (int i = rowStart; i < A.localRows; ++i) {
        int globalRow = g.row + i * g.rows;
        for (int j = colStart; j < A.localCols; ++j) {
            int globalCol = g.col + j * g.cols;
            if (globalRow >= globalCol)
                matrixProduct<Subtract>(A.tile(globalRow, globalCol), left.data() + (i - rowStart) * elements,
                                        right.data() + (j - colStart) * elements, A.extent(globalRow),
                                        A.extent(globalCol), A.extent(k), b);
        }
    }
}

// Factor a diagonal tile, returning its failing local diagonal index or -1.
int factorDiagonal(double* a, int extent, int b) {
    for (int j = 0; j < extent; ++j) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k)
            sum += a[j * b + k] * a[j * b + k];
        double value = a[j * b + j] - sum;
        if (!(value > 0.0))
            return j;
        a[j * b + j] = std::sqrt(value);
        for (int i = j + 1; i < extent; ++i) {
            sum = 0.0;
            for (int k = 0; k < j; ++k)
                sum += a[i * b + k] * a[j * b + k];
            a[i * b + j] = (a[i * b + j] - sum) / a[j * b + j];
        }
        for (int i = j + 1; i < extent; ++i)
            a[j * b + i] = 0.0;
    }
    return -1;
}

void solvePanel(double* a, const double* diagonal, int rows, int width, int b) {
    // Transpose locally so independent solves occupy adjacent SIMD lanes.
    transpose(a, b);
    for (int j = 0; j < width; ++j) {
        double sum[64] = {}; // The maximum tile extent is 64.
        for (int k = 0; k < j; ++k) {
            double coefficient = diagonal[j * b + k];
            for (int i = 0; i < rows; ++i)
                sum[i] += a[k * b + i] * coefficient;
        }
        for (int i = 0; i < rows; ++i)
            a[j * b + i] = (a[j * b + i] - sum[i]) / diagonal[j * b + j];
    }
    transpose(a, b);
}

bool choleskyDecomposition(DistributedMatrix& A) {
    const ProcessGrid& g = A.grid;
    const int b = g.block;
    const size_t elements = static_cast<size_t>(b) * b;
    // Last entry carries the failure index: one collective propagates both
    // the diagonal factor and its status so failures cannot strand other ranks.
    std::vector<double> diagonal(elements + 1);
    std::vector<double> left(static_cast<size_t>(A.localRows) * elements);
    std::vector<double> right(static_cast<size_t>(A.localCols) * elements);
    for (int k = 0; k < A.blocks; ++k) {
        int owner = g.owner(k, k);
        if (g.rank == owner) {
            double* tile = A.tile(k, k);
            int failed = factorDiagonal(tile, A.extent(k), b);
            std::copy_n(tile, elements, diagonal.data());
            diagonal[elements] = failed < 0 ? -1.0 : static_cast<double>(k) * b + failed;
        }
        broadcast(diagonal.data(), diagonal.size(), owner, MPI_COMM_WORLD);
        if (diagonal[elements] >= 0.0) {
            if (g.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       static_cast<size_t>(diagonal[elements]));
            return false;
        }
        if (k + 1 == A.blocks)
            break;
        if (g.col == k % g.cols)
            for (int i = firstLocal(k + 1, g.row, g.rows); i < A.localRows; ++i) {
                int globalRow = g.row + i * g.rows;
                solvePanel(A.tile(globalRow, k), diagonal.data(), A.extent(globalRow), A.extent(k), b);
            }
        broadcastFactorPanel(A, k, k + 1, left, right);
        updateTrailingMatrix<true>(A, k, k + 1, left, right);
    }
    return true;
}

bool validateCholesky(const DistributedMatrix& L, const DistributedMatrix& original) {
    const ProcessGrid& g = L.grid;
    const size_t elements = static_cast<size_t>(g.block) * g.block;
    DistributedMatrix reconstructed(g, L.n);
    std::vector<double> left(static_cast<size_t>(L.localRows) * elements);
    std::vector<double> right(static_cast<size_t>(L.localCols) * elements);
    for (int k = 0; k < L.blocks; ++k) {
        broadcastFactorPanel(L, k, k, left, right);
        updateTrailingMatrix<false>(reconstructed, k, k, left, right);
    }
    double errors[2] = {0.0, 0.0}, globalErrors[2];
    // Symmetry makes comparison of the stored lower tiles sufficient.
    for (int i = 0; i < L.localRows; ++i) {
        int globalRow = g.row + i * g.rows;
        for (int j = 0; j < L.localCols; ++j) {
            int globalCol = g.col + j * g.cols;
            if (globalRow < globalCol)
                continue;
            const double* a = original.tile(globalRow, globalCol);
            const double* r = reconstructed.tile(globalRow, globalCol);
            for (int row = 0; row < L.extent(globalRow); ++row)
                for (int col = 0; col < L.extent(globalCol); ++col) {
                    int index = row * g.block + col;
                    double error = std::fabs(r[index] - a[index]);
                    if (!std::isfinite(error))
                        error = std::numeric_limits<double>::infinity();
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(a[index]) + 1e-10));
                }
        }
    }
    MPI_Allreduce(errors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (g.rank == 0) {
        printf("Max absolute error: %.10e\n", globalErrors[0]);
        printf("Max relative error: %.10e\n", globalErrors[1]);
        if (globalErrors[1] > 1e-6)
            printf("Validation failed: relative error too large\n");
    }
    return globalErrors[1] <= 1e-6;
}

std::vector<double> gatherResult(const DistributedMatrix& L) {
    const ProcessGrid& g = L.grid;
    std::vector<double> result(g.rank == 0 ? L.n * L.n : 0, 0.0);
    std::vector<double> buffer(g.block * g.block);
    for (int i = 0; i < L.blocks; ++i)
        for (int j = 0; j <= i; ++j) {
            int owner = g.owner(i, j);
            if (g.rank == 0) {
                const double* tile = owner == 0 ? L.tile(i, j) : buffer.data();
                if (owner != 0)
                    MPI_Recv(buffer.data(), g.block * g.block, MPI_DOUBLE, owner, 3,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int row = 0; row < L.extent(i); ++row)
                    std::copy_n(tile + row * g.block, L.extent(j),
                                result.data() + (static_cast<size_t>(i) * g.block + row) * L.n
                                    + static_cast<size_t>(j) * g.block);
            } else if (g.rank == owner) {
                MPI_Send(L.tile(i, j), g.block * g.block, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
            }
        }
    return result;
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
    uint64_t order = 512;
    int options[2] = {0, 0};
    int parseStatus = 0; // 0: run, 1: help, 2: invalid arguments
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* argument = argv[++i];
                char* end = nullptr;
                errno = 0;
                unsigned long long value = strtoull(argument, &end, 10);
                if (argument[0] == '-' || end == argument || *end || errno == ERANGE
                    || value > INT_MAX || (value && value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                    printf("Invalid matrix size: %s\n", argument);
                    parseStatus = 2;
                    break;
                }
                order = value;
            } else if (strcmp(argv[i], "-v") == 0) {
                options[0] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                parseStatus = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseStatus = 2;
                break;
            }
        }
        if (parseStatus)
            printUsage(argv[0]);
    }
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus)
        return parseStatus == 1 ? 0 : 1;
    MPI_Bcast(&order, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(options, 2, MPI_INT, 0, MPI_COMM_WORLD);
    const size_t n = static_cast<size_t>(order);
    const bool validate = options[0], printResults = options[1];
    ProcessGrid grid(n);
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    DistributedMatrix A(grid, n);
    generatePositiveDefiniteMatrix(A);
    DistributedMatrix original(grid, validate ? n : 0);
    if (validate)
        original.tiles = A.tiles;
    if (rank == 0)
        printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = choleskyDecomposition(A);
    double localTime = MPI_Wtime() - start, elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0)
            printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) {
        std::vector<double> result = gatherResult(A);
        if (rank == 0)
            print_results(result, "CholeskyL");
    }
    if (validate) {
        if (rank == 0)
            printf("Validating result...\n");
        bool valid = validateCholesky(A, original);
        if (rank == 0)
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int result = 1;
    try {
        result = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
