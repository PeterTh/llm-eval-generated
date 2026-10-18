#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

namespace {
constexpr int blockSize = 64;
constexpr int tileElements = blockSize * blockSize;

// Only lower-triangular tiles are stored. Tiles are distributed cyclically
// over both dimensions of a Cartesian process grid.
struct Matrix {
    size_t n;
    int nt, rank, rows, cols, row, col;
    MPI_Comm grid, rowComm, colComm;
    std::vector<size_t> offsets;
    std::vector<double> data;

    explicit Matrix(size_t size) : n(size) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int processes, dims[2] = {0, 0}, periods[2] = {0, 0};
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dims);
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
        rows = dims[0];
        cols = dims[1];
        int coords[2];
        MPI_Cart_coords(grid, rank, 2, coords);
        row = coords[0];
        col = coords[1];
        MPI_Comm_split(grid, row, col, &rowComm);
        MPI_Comm_split(grid, col, row, &colComm);
        nt = static_cast<int>((n + blockSize - 1) / blockSize);
        offsets.resize(static_cast<size_t>(localRows()) * localCols());
        size_t count = 0;
        for (int i = row; i < nt; i += rows)
            for (int j = col; j <= i; j += cols) {
                offsets[index(i, j)] = count;
                count += tileElements;
            }
        data.resize(count);
    }

    ~Matrix() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
        MPI_Comm_free(&grid);
    }
    int localRows() const { return nt / rows + (row < nt % rows); }
    int localCols() const { return nt / cols + (col < nt % cols); }
    int width(int i) const {
        return static_cast<int>(std::min<size_t>(blockSize, n - size_t(i) * blockSize));
    }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
    size_t index(int i, int j) const {
        return static_cast<size_t>(i / rows) * localCols() + j / cols;
    }
    double* tile(int i, int j) { return data.data() + offsets[index(i, j)]; }
};

// Preserve the original rand_r stream exactly, but retain only this process
// row's B rows. Column-communicator broadcasts supply the other operand of B B^T.
void generatePositiveDefiniteMatrix(Matrix& a) {
    std::vector<double> b(static_cast<size_t>(a.localRows()) * blockSize * a.n);
    unsigned int seed = 42;
    for (size_t i = 0; i < a.n; ++i) {
        const int tileRow = static_cast<int>(i / blockSize);
        double* dest = tileRow % a.rows == a.row
            ? b.data() + (size_t(tileRow / a.rows) * blockSize + i % blockSize) * a.n
            : nullptr;
        for (size_t k = 0; k < a.n; ++k) {
            const double value = rand_r(&seed) / double(RAND_MAX) - 0.5;
            if (dest) dest[k] = value;
        }
    }
    std::vector<double> other(static_cast<size_t>(blockSize) * a.n);
    for (int j = a.col; j < a.nt; j += a.cols) {
        const size_t count = size_t(a.width(j)) * a.n;
        if (a.row == j % a.rows)
            std::copy_n(b.data() + size_t(j / a.rows) * blockSize * a.n,
                        count, other.data());
        // Chunk large messages to avoid MPI's int count limit.
        for (size_t start = 0; start < count;) {
            const int chunk = static_cast<int>(std::min<size_t>(count - start, INT_MAX));
            MPI_Bcast(other.data() + start, chunk, MPI_DOUBLE, j % a.rows, a.colComm);
            start += chunk;
        }
        for (int i = a.row; i < a.nt; i += a.rows) {
            if (i < j) continue;
            double* tile = a.tile(i, j);
            for (int r = 0; r < a.width(i); ++r) {
                const double* left = b.data() + (size_t(i / a.rows) * blockSize + r) * a.n;
                for (int c = 0; c < a.width(j); ++c) {
                    const double* right = other.data() + size_t(c) * a.n;
                    double sum = 0.0;
                    for (size_t k = 0; k < a.n; ++k) sum += left[k] * right[k];
                    tile[r * blockSize + c] = sum + (i == j && r == c ? a.n : 0);
                }
            }
        }
    }
}

// Communicate only the panels required by the local trailing tiles: broadcast
// along process rows, then redistribute the transposed operand down columns.
// Per-rank panel storage and traffic scale as O(n * blockSize / sqrt(P)).
struct Panels {
    Matrix& a;
    std::vector<double> rowData, colData, send;
    std::vector<int> counts, displacements, colOffsets;

    explicit Panels(Matrix& matrix) : a(matrix),
        rowData(size_t(a.localRows()) * tileElements),
        colData(size_t(a.localCols()) * tileElements),
        send(size_t(a.localCols()) * tileElements),
        counts(a.rows), displacements(a.rows), colOffsets(a.localCols()) {}

    void exchange(int k, int first) {
        int rowCount = 0;
        for (int i = a.row; i < a.nt; i += a.rows) {
            if (i < first) continue;
            if (a.col == k % a.cols)
                std::copy_n(a.tile(i, k), tileElements, rowData.data() + rowCount);
            rowCount += tileElements;
        }
        MPI_Bcast(rowData.data(), rowCount, MPI_DOUBLE, k % a.cols, a.rowComm);
        int total = 0;
        for (int r = 0; r < a.rows; ++r) {
            displacements[r] = total;
            for (int j = a.col; j < a.nt; j += a.cols) {
                if (j < first || j % a.rows != r) continue;
                colOffsets[j / a.cols] = total;
                total += tileElements;
            }
            counts[r] = total - displacements[r];
        }
        int sent = 0, offset = 0;
        for (int i = a.row; i < a.nt; i += a.rows) {
            if (i < first) continue;
            if (i % a.cols == a.col) {
                std::copy_n(rowData.data() + offset, tileElements, send.data() + sent);
                sent += tileElements;
            }
            offset += tileElements;
        }
        MPI_Allgatherv(send.data(), sent, MPI_DOUBLE, colData.data(),
                       counts.data(), displacements.data(), MPI_DOUBLE, a.colComm);
        // Pack the right-hand operand in k-major order once per panel,
        // allowing contiguous SIMD loads in every trailing-tile update.
        for (int tile = 0; tile < total; tile += tileElements)
            for (int r = 0; r < blockSize; ++r)
                for (int c = 0; c < r; ++c)
                    std::swap(colData[tile + r * blockSize + c],
                              colData[tile + c * blockSize + r]);
    }
};

// Small register blocks vectorize across independent output elements, retaining
// ascending summation order in each dot product (no fast-math required).
void product(double* target, const double* left, const double* right,
             int nr, int nc, int nk, double sign) {
    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; c += 8) {
            const int end = std::min(8, nc - c);
            double sums[8] = {};
            for (int k = 0; k < nk; ++k) {
                const double value = left[r * blockSize + k];
                for (int t = 0; t < end; ++t)
                    sums[t] += value * right[k * blockSize + c + t];
            }
            for (int t = 0; t < end; ++t)
                target[r * blockSize + c + t] += sign * sums[t];
        }
    }
}

bool choleskyDecomposition(Matrix& a) {
    Panels panels(a);
    std::vector<double> diagonal(tileElements);
    for (int k = 0; k < a.nt; ++k) {
        const int width = a.width(k);
        if (a.rank == a.owner(k, k)) {
            std::copy_n(a.tile(k, k), tileElements, diagonal.data());
            for (int j = 0; j < width; ++j) {
                double sum = 0.0;
                for (int t = 0; t < j; ++t)
                    sum += diagonal[j * blockSize + t] * diagonal[j * blockSize + t];
                const double value = diagonal[j * blockSize + j] - sum;
                if (!(value > 0.0)) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                           size_t(k) * blockSize + j);
                    diagonal[0] = -1.0;
                    break;
                }
                diagonal[j * blockSize + j] = std::sqrt(value);
                for (int i = j + 1; i < width; ++i) {
                    sum = 0.0;
                    for (int t = 0; t < j; ++t)
                        sum += diagonal[i * blockSize + t] * diagonal[j * blockSize + t];
                    diagonal[i * blockSize + j] =
                        (diagonal[i * blockSize + j] - sum) / diagonal[j * blockSize + j];
                }
                for (int i = 0; i < j; ++i) diagonal[i * blockSize + j] = 0.0;
            }
            std::copy_n(diagonal.data(), tileElements, a.tile(k, k));
        }
        MPI_Bcast(diagonal.data(), tileElements, MPI_DOUBLE, a.owner(k, k), a.grid);
        if (diagonal[0] <= 0.0) return false;

        if (a.col == k % a.cols) {
            for (int i = a.row; i < a.nt; i += a.rows) {
                if (i <= k) continue;
                double* tile = a.tile(i, k);
                for (int r = 0; r < a.width(i); ++r)
                    for (int j = 0; j < width; ++j) {
                        double sum = 0.0;
                        for (int t = 0; t < j; ++t)
                            sum += tile[r * blockSize + t] * diagonal[j * blockSize + t];
                        tile[r * blockSize + j] =
                            (tile[r * blockSize + j] - sum) / diagonal[j * blockSize + j];
                    }
            }
        }
        panels.exchange(k, k + 1);
        int offset = 0;
        for (int i = a.row; i < a.nt; i += a.rows) {
            if (i <= k) continue;
            for (int j = a.col; j <= i; j += a.cols) {
                if (j <= k) continue;
                product(a.tile(i, j), panels.rowData.data() + offset,
                        panels.colData.data() + panels.colOffsets[j / a.cols],
                        a.width(i), a.width(j), width, -1.0);
            }
            offset += tileElements;
        }
    }
    return true;
}

bool validateCholesky(Matrix& a, const std::vector<double>& original) {
    std::vector<double> reconstructed(a.data.size(), 0.0);
    Panels panels(a);
    for (int k = 0; k < a.nt; ++k) {
        panels.exchange(k, k);
        int offset = 0;
        for (int i = a.row; i < a.nt; i += a.rows) {
            if (i < k) continue;
            for (int j = a.col; j <= i; j += a.cols) {
                if (j < k) continue;
                product(reconstructed.data() + a.offsets[a.index(i, j)],
                        panels.rowData.data() + offset,
                        panels.colData.data() + panels.colOffsets[j / a.cols],
                        a.width(i), a.width(j), a.width(k), 1.0);
            }
            offset += tileElements;
        }
    }
    double errors[2] = {};
    for (int i = a.row; i < a.nt; i += a.rows)
        for (int j = a.col; j <= i; j += a.cols) {
            const size_t offset = a.offsets[a.index(i, j)];
            for (int r = 0; r < a.width(i); ++r)
                for (int c = 0; c < a.width(j); ++c) {
                    const size_t pos = offset + r * blockSize + c;
                    const double error = std::fabs(reconstructed[pos] - original[pos]);
                    const double relative = error / (std::fabs(original[pos]) + 1e-10);
                    errors[0] = std::max(errors[0], std::isfinite(error) ? error : INFINITY);
                    errors[1] = std::max(errors[1], std::isfinite(relative) ? relative : INFINITY);
                }
        }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, a.grid);
    if (a.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

void printDistributedResults(Matrix& a) {
    std::vector<double> result, buffer(tileElements);
    if (a.rank == 0) result.resize(a.n * a.n, 0.0);
    for (int i = 0; i < a.nt; ++i)
        for (int j = 0; j <= i; ++j) {
            const int owner = a.owner(i, j);
            if (a.rank == owner && owner != 0)
                MPI_Send(a.tile(i, j), tileElements, MPI_DOUBLE, 0, 0, a.grid);
            if (a.rank != 0) continue;
            const double* tile = buffer.data();
            if (owner == 0) tile = a.tile(i, j);
            else MPI_Recv(buffer.data(), tileElements, MPI_DOUBLE, owner, 0, a.grid, MPI_STATUS_IGNORE);
            for (int r = 0; r < a.width(i); ++r)
                std::copy_n(tile + r * blockSize, a.width(j),
                            result.data() + (size_t(i) * blockSize + r) * a.n + size_t(j) * blockSize);
        }
    if (a.rank == 0) print_results(result, "CholeskyL");
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* argument = argv[++i];
            errno = 0;
            const unsigned long long value = std::strtoull(argument, &end, 10);
            // Bound tile-panel MPI counts and all size_t allocation arithmetic.
            const size_t limit = std::min<size_t>(INT_MAX / (blockSize * 2),
                static_cast<size_t>(std::sqrt(std::numeric_limits<size_t>::max() / sizeof(double))));
            if (errno || argument[0] == '-' || end == argument || *end || value > limit) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argument);
                return 1;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
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
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix a(n);
    generatePositiveDefiniteMatrix(a);
    const std::vector<double> original = validate ? a.data : std::vector<double>();
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(a.grid);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, a.grid);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        const double ops = double(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(a);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(a, original);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status = 1;
    try {
        status = run(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
