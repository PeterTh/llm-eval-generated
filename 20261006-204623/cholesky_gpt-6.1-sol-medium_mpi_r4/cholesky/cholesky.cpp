#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

namespace {
constexpr int blockSize = 64;
constexpr size_t tileElements = blockSize * blockSize;

// Only lower-triangular tiles are stored. Both tile coordinates are cyclically
// distributed, so the shrinking trailing matrix remains balanced across ranks.
struct Matrix {
    size_t n;
    int tiles, rows, cols, row, col, localRows, localCols;
    MPI_Comm rowComm = MPI_COMM_NULL, colComm = MPI_COMM_NULL;
    std::vector<size_t> offsets;
    std::vector<double> data;

    Matrix(size_t order, int processes, int rank) : n(order) {
        int dims[2] = {0, 0};
        MPI_Dims_create(processes, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
        tiles = static_cast<int>((n + blockSize - 1) / blockSize);
        localRows = tiles <= row ? 0 : (tiles - 1 - row) / rows + 1;
        localCols = tiles <= col ? 0 : (tiles - 1 - col) / cols + 1;
        offsets.assign(static_cast<size_t>(localRows) * localCols,
                       std::numeric_limits<size_t>::max());
        size_t count = 0;
        for (int i = 0; i < localRows; ++i)
            for (int j = 0; j < localCols; ++j)
                if (i * rows + row >= j * cols + col) {
                    offsets[static_cast<size_t>(i) * localCols + j] = count;
                    count += tileElements;
                }
        data.resize(count, 0.0);
    }

    ~Matrix() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }

    int extent(int tile) const {
        return static_cast<int>(std::min<size_t>(blockSize, n - size_t(tile) * blockSize));
    }

    double* tile(int i, int j) {
        return data.data() + offsets[static_cast<size_t>(i / rows) * localCols + j / cols];
    }
    const double* tile(int i, int j) const {
        return data.data() + offsets[static_cast<size_t>(i / rows) * localCols + j / cols];
    }
};

// Chunked transfers avoid MPI's int-count limit for large panels/matrices.
void broadcast(double* data, size_t count, int root, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min<size_t>(count, std::numeric_limits<int>::max()));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, comm);
        data += chunk;
        count -= chunk;
    }
}

// Reproduce exactly the original rand_r stream without replicating all of B.
// Each process keeps just its process-row's B rows and streams the other rows.
void generatePositiveDefiniteMatrix(Matrix& a) {
    const size_t stride = static_cast<size_t>(blockSize) * a.n;
    std::vector<double> ownRows(static_cast<size_t>(a.localRows) * stride);
    std::vector<double> otherRow(a.n);
    unsigned seed = 42;
    for (size_t i = 0; i < a.n; ++i) {
        for (size_t k = 0; k < a.n; ++k)
            otherRow[k] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
        if (static_cast<int>(i / blockSize) % a.rows == a.row) {
            size_t offset = (i / blockSize / a.rows * blockSize + i % blockSize) * a.n;
            std::copy(otherRow.begin(), otherRow.end(), ownRows.data() + offset);
        }
    }
    seed = 42;
    for (size_t j = 0; j < a.n; ++j) {
        for (size_t k = 0; k < a.n; ++k)
            otherRow[k] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
        int tj = static_cast<int>(j / blockSize);
        if (tj % a.cols != a.col) continue;
        for (int li = 0; li < a.localRows; ++li) {
            int ti = li * a.rows + a.row;
            if (ti < tj) continue;
            double* target = a.tile(ti, tj);
            for (int ii = 0; ii < a.extent(ti); ++ii) {
                size_t i = size_t(ti) * blockSize + ii;
                if (i < j) continue;
                const double* b = ownRows.data() + size_t(li) * stride + size_t(ii) * a.n;
                double sum = 0.0;
                for (size_t k = 0; k < a.n; ++k) sum += b[k] * otherRow[k];
                target[ii * blockSize + j % blockSize] = sum + (i == j ? a.n : 0.0);
            }
        }
    }
}

bool factorDiagonal(double* a, int size, size_t& failed) {
    for (int j = 0; j < size; ++j) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) sum += a[j * blockSize + k] * a[j * blockSize + k];
        double value = a[j * blockSize + j] - sum;
        if (!(value > 0.0) || !std::isfinite(value)) {
            failed = j;
            return false;
        }
        a[j * blockSize + j] = std::sqrt(value);
        for (int i = j + 1; i < size; ++i) {
            sum = 0.0;
            for (int k = 0; k < j; ++k) sum += a[i * blockSize + k] * a[j * blockSize + k];
            a[i * blockSize + j] = (a[i * blockSize + j] - sum) / a[j * blockSize + j];
        }
        for (int k = j + 1; k < size; ++k) a[j * blockSize + k] = 0.0;
    }
    return true;
}

void solvePanel(double* a, const double* diagonal, int height, int width) {
    for (int i = 0; i < height; ++i) {
        double* r = a + i * blockSize;
        for (int j = 0; j < width; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += r[k] * diagonal[j * blockSize + k];
            r[j] = (r[j] - sum) / diagonal[j * blockSize + j];
        }
    }
}

void transpose(const double* source, double* target) {
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            target[j * blockSize + i] = source[i * blockSize + j];
}

// Register-blocked C += sign * L * R^T. R is already transposed, allowing
// contiguous SIMD loads and reuse across four output rows; no BLAS dependency.
void updateTile(double* c, const double* l, const double* rt,
                int height, int width, int inner, double sign) {
    int i = 0;
    for (; i + 4 <= height; i += 4) {
        int j = 0;
        for (; j + 8 <= width; j += 8) {
            double s0[8] = {}, s1[8] = {}, s2[8] = {}, s3[8] = {};
            for (int k = 0; k < inner; ++k) {
                const double* r = rt + k * blockSize + j;
                double a0 = l[i * blockSize + k], a1 = l[(i + 1) * blockSize + k];
                double a2 = l[(i + 2) * blockSize + k], a3 = l[(i + 3) * blockSize + k];
                for (int q = 0; q < 8; ++q) {
                    s0[q] += a0 * r[q];
                    s1[q] += a1 * r[q];
                    s2[q] += a2 * r[q];
                    s3[q] += a3 * r[q];
                }
            }
            for (int q = 0; q < 8; ++q) {
                c[i * blockSize + j + q] += sign * s0[q];
                c[(i + 1) * blockSize + j + q] += sign * s1[q];
                c[(i + 2) * blockSize + j + q] += sign * s2[q];
                c[(i + 3) * blockSize + j + q] += sign * s3[q];
            }
        }
        for (; j < width; ++j)
            for (int ii = i; ii < i + 4; ++ii) {
                double sum = 0.0;
                for (int k = 0; k < inner; ++k) sum += l[ii * blockSize + k] * rt[k * blockSize + j];
                c[ii * blockSize + j] += sign * sum;
            }
    }
    for (; i < height; ++i)
        for (int j = 0; j < width; ++j) {
            double sum = 0.0;
            for (int k = 0; k < inner; ++k) sum += l[i * blockSize + k] * rt[k * blockSize + j];
            c[i * blockSize + j] += sign * sum;
        }
}

struct Panels {
    std::vector<double> left, right;
    std::vector<MPI_Request> requests;
    explicit Panels(const Matrix& a)
        : left(size_t(a.localRows) * tileElements),
          right(size_t(a.localCols) * tileElements), requests(a.localCols) {}

    // Broadcast the panel along process rows, then send its transposed tiles
    // down process columns. Nonblocking column broadcasts can progress together.
    void exchange(const Matrix& a, int k, int first) {
        int begin = std::max(0, (first - a.row + a.rows - 1) / a.rows);
        if (a.col == k % a.cols)
            for (int li = begin; li < a.localRows; ++li)
                std::copy_n(a.tile(li * a.rows + a.row, k), tileElements,
                            left.data() + size_t(li) * tileElements);
        size_t count = static_cast<size_t>(a.localRows - begin) * tileElements;
        if (count) broadcast(left.data() + size_t(begin) * tileElements, count, k % a.cols, a.rowComm);
        int pending = 0;
        for (int lj = 0; lj < a.localCols; ++lj) {
            int j = lj * a.cols + a.col;
            if (j < first) continue;
            double* target = right.data() + size_t(lj) * tileElements;
            if (a.row == j % a.rows)
                transpose(left.data() + size_t(j / a.rows) * tileElements, target);
            MPI_Ibcast(target, static_cast<int>(tileElements), MPI_DOUBLE,
                       j % a.rows, a.colComm, &requests[pending++]);
        }
        if (pending) MPI_Waitall(pending, requests.data(), MPI_STATUSES_IGNORE);
    }
};

bool choleskyDecomposition(Matrix& a) {
    Panels panels(a);
    std::vector<double> diagonal(tileElements);
    for (int k = 0; k < a.tiles; ++k) {
        int owner = (k % a.rows) * a.cols + k % a.cols;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == owner) {
            double* tile = a.tile(k, k);
            size_t failed = 0;
            if (!factorDiagonal(tile, a.extent(k), failed)) {
                // Carry the failure through the same collective as the diagonal.
                tile[0] = -1.0;
                tile[1] = static_cast<double>(size_t(k) * blockSize + failed);
            }
            std::copy_n(tile, tileElements, diagonal.data());
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(tileElements), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (diagonal[0] < 0.0) {
            if (rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", static_cast<size_t>(diagonal[1]));
            return false;
        }
        if (a.col == k % a.cols)
            for (int li = 0; li < a.localRows; ++li) {
                int i = li * a.rows + a.row;
                if (i > k) solvePanel(a.tile(i, k), diagonal.data(), a.extent(i), a.extent(k));
            }
        panels.exchange(a, k, k + 1);
        for (int li = 0; li < a.localRows; ++li) {
            int i = li * a.rows + a.row;
            if (i <= k) continue;
            for (int lj = 0; lj < a.localCols; ++lj) {
                int j = lj * a.cols + a.col;
                if (j <= k || j > i) continue;
                updateTile(a.tile(i, j), panels.left.data() + size_t(li) * tileElements,
                           panels.right.data() + size_t(lj) * tileElements,
                           a.extent(i), a.extent(j), a.extent(k), -1.0);
            }
        }
    }
    return true;
}

bool validateCholesky(const Matrix& a, const std::vector<double>& original, int rank) {
    std::vector<double> reconstructed(a.data.size(), 0.0);
    Panels panels(a);
    for (int k = 0; k < a.tiles; ++k) {
        panels.exchange(a, k, k);
        for (int li = 0; li < a.localRows; ++li) {
            int i = li * a.rows + a.row;
            if (i < k) continue;
            for (int lj = 0; lj < a.localCols; ++lj) {
                int j = lj * a.cols + a.col;
                if (j < k || j > i) continue;
                size_t offset = a.offsets[size_t(li) * a.localCols + lj];
                updateTile(reconstructed.data() + offset,
                           panels.left.data() + size_t(li) * tileElements,
                           panels.right.data() + size_t(lj) * tileElements,
                           a.extent(i), a.extent(j), a.extent(k), 1.0);
            }
        }
    }
    double errors[2] = {0.0, 0.0};
    for (int li = 0; li < a.localRows; ++li) {
        int i = li * a.rows + a.row;
        for (int lj = 0; lj < a.localCols; ++lj) {
            int j = lj * a.cols + a.col;
            if (j > i) continue;
            size_t offset = a.offsets[size_t(li) * a.localCols + lj];
            for (int ii = 0; ii < a.extent(i); ++ii)
                for (int jj = 0; jj < a.extent(j); ++jj) {
                    if (i == j && jj > ii) continue;
                    size_t pos = offset + ii * blockSize + jj;
                    double error = std::fabs(reconstructed[pos] - original[pos]);
                    if (!std::isfinite(error)) error = std::numeric_limits<double>::infinity();
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(original[pos]) + 1e-10));
                }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

// Only -r materializes a full row-major factor, preserving the original result
// statistics, samples, zero upper triangle, and hash ordering on rank zero.
void printDistributedResults(const Matrix& a, int rank) {
    std::vector<double> result;
    if (rank == 0) result.resize(a.n * a.n, 0.0);
    std::vector<double> buffer(tileElements);
    for (int i = 0; i < a.tiles; ++i)
        for (int j = 0; j <= i; ++j) {
            int owner = (i % a.rows) * a.cols + j % a.cols;
            const double* tile = nullptr;
            if (rank == owner) {
                tile = a.tile(i, j);
                if (owner != 0) MPI_Send(tile, static_cast<int>(tileElements), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            if (rank == 0) {
                if (owner != 0) {
                    MPI_Recv(buffer.data(), static_cast<int>(tileElements), MPI_DOUBLE, owner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    tile = buffer.data();
                }
                for (int ii = 0; ii < a.extent(i); ++ii)
                    for (int jj = 0; jj < a.extent(j); ++jj)
                        if (i != j || jj <= ii)
                            result[(size_t(i) * blockSize + ii) * a.n + size_t(j) * blockSize + jj] = tile[ii * blockSize + jj];
            }
        }
    if (rank == 0) print_results(result, "CholeskyL");
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int processes) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            unsigned long long value = std::strtoull(arg, &end, 10);
            size_t maxN = static_cast<size_t>(std::sqrt(static_cast<long double>(std::numeric_limits<size_t>::max() / sizeof(double))));
            if (errno || arg[0] == '-' || !arg[0] || *end || value > maxN ||
                value > static_cast<unsigned long long>(std::numeric_limits<int>::max() - blockSize)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
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
    Matrix a(n, processes, rank);
    generatePositiveDefiniteMatrix(a);
    std::vector<double> original;
    if (validate) original = a.data;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = choleskyDecomposition(a);
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(elapsed * 1000.0));
        double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(a, rank);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(a, original, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int status = 1;
    try {
        status = run(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
