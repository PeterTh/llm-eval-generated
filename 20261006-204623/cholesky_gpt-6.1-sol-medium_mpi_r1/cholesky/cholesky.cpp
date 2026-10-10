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

#include <mpi.h>
#include "../common/results_output.hpp"

namespace {
constexpr int blockSize = 64;
constexpr size_t tileSize = blockSize * blockSize;

// Chunk large messages instead of narrowing size_t to MPI's int count.
void broadcast(double* data, size_t count, int root, MPI_Comm comm) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, comm);
        data += chunk;
        count -= chunk;
    }
}

struct Matrix {
    size_t n;
    int blocks, rows, cols, row, col, localRows, localCols;
    MPI_Comm rowComm, colComm;
    std::vector<std::vector<double>> tiles;

    Matrix(size_t size, int rank, int ranks) : n(size) {
        blocks = static_cast<int>((n + blockSize - 1) / blockSize);
        int dims[2] = {0, 0};
        MPI_Dims_create(ranks, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
        localRows = blocks / rows + (row < blocks % rows);
        localCols = blocks / cols + (col < blocks % cols);
        tiles.resize(size_t(localRows) * localCols);
        for (int i = row; i < blocks; i += rows)
            for (int j = col; j <= i; j += cols)
                tile(i, j).resize(tileSize, 0.0);
    }
    ~Matrix() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    int extent(int i) const {
        return static_cast<int>(std::min(size_t(blockSize), n - size_t(i) * blockSize));
    }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
    std::vector<double>& tile(int i, int j) {
        return tiles[size_t(i / rows) * localCols + j / cols];
    }
    const std::vector<double>& tile(int i, int j) const {
        return tiles[size_t(i / rows) * localCols + j / cols];
    }
};

void generateProduct(double* c, const double* l, const double* rt,
                     int m, int n, int depth, size_t leftStride);

// Reproduce the original rand_r stream exactly. Store only the B rows needed
// by this process row; the cubic Gram matrix work is distributed in 2D.
void generatePositiveDefiniteMatrix(Matrix& a) {
    std::vector<double> b(size_t(a.localRows) * blockSize * a.n, 0.0);
    unsigned int seed = 42;
    for (size_t i = 0; i < a.n; ++i) {
        const int block = static_cast<int>(i / blockSize);
        double* dest = block % a.rows == a.row
            ? b.data() + (size_t(block / a.rows) * blockSize + i % blockSize) * a.n
            : nullptr;
        for (size_t k = 0; k < a.n; ++k) {
            const double value = rand_r(&seed) / double(RAND_MAX) - 0.5;
            if (dest) dest[k] = value;
        }
    }
    std::vector<double> other(size_t(blockSize) * a.n);
    std::vector<double> transposed(size_t(blockSize) * a.n);
    for (int j = a.col; j < a.blocks; j += a.cols) {
        const int nj = a.extent(j);
        if (a.row == j % a.rows)
            std::copy_n(b.data() + size_t(j / a.rows) * blockSize * a.n,
                        size_t(nj) * a.n, other.data());
        broadcast(other.data(), size_t(nj) * a.n, j % a.rows, a.colComm);
        for (size_t k = 0; k < a.n; ++k)
            for (int jj = 0; jj < nj; ++jj)
                transposed[k * blockSize + jj] = other[size_t(jj) * a.n + k];
        for (int i = a.row; i < a.blocks; i += a.rows) {
            if (i < j) continue;
            auto& tile = a.tile(i, j);
            generateProduct(tile.data(), b.data() + size_t(i / a.rows) * blockSize * a.n,
                            transposed.data(), a.extent(i), nj, static_cast<int>(a.n), a.n);
            if (i == j)
                for (int ii = 0; ii < nj; ++ii)
                    tile[size_t(ii) * blockSize + ii] += a.n;
        }
    }
}

// A panel is broadcast along process rows, then its transposed counterpart
// along process columns. Each rank receives only panels used by its tiles.
// Packing amortizes collective latency over all tiles in a process row.
struct Panels {
    std::vector<double> left, right, packed;
    explicit Panels(const Matrix& a)
        : left(size_t(a.localRows) * tileSize),
          right(size_t(a.localCols) * tileSize),
          packed(size_t(a.localCols) * tileSize) {}

    void exchange(const Matrix& a, int k, int first) {
        const int firstRow = first / a.rows + (a.row < first % a.rows);
        if (a.col == k % a.cols)
            for (int li = firstRow; li < a.localRows; ++li) {
                const int i = li * a.rows + a.row;
                std::copy_n(a.tile(i, k).data(), tileSize, left.data() + size_t(li) * tileSize);
            }
        if (firstRow < a.localRows)
            broadcast(left.data() + size_t(firstRow) * tileSize,
                      size_t(a.localRows - firstRow) * tileSize, k % a.cols, a.rowComm);
        for (int source = 0; source < a.rows; ++source) {
            size_t count = 0;
            for (int j = a.col; j < a.blocks; j += a.cols) {
                if (j < first || j % a.rows != source) continue;
                if (a.row == source) {
                    const double* src = left.data() + size_t(j / a.rows) * tileSize;
                    double* dst = packed.data() + count;
                    for (int i = 0; i < blockSize; ++i)
                        for (int t = 0; t < blockSize; ++t)
                            dst[size_t(t) * blockSize + i] = src[size_t(i) * blockSize + t];
                }
                count += tileSize;
            }
            broadcast(packed.data(), count, source, a.colComm);
            count = 0;
            for (int j = a.col; j < a.blocks; j += a.cols) {
                if (j < first || j % a.rows != source) continue;
                std::copy_n(packed.data() + count, tileSize, right.data() + size_t(j / a.cols) * tileSize);
                count += tileSize;
            }
        }
    }
};

// C += L * R^T. Four-by-eight register tiles reuse panel data
// and keep partial sums off the memory bus. Independent output columns
// allow SIMD without reassociating each dot product or requiring BLAS.
void generateProduct(double* c, const double* l, const double* rt,
            int m, int n, int depth, size_t leftStride) {
    int i = 0;
    for (; i + 3 < m; i += 4) {
        int j = 0;
        for (; j + 7 < n; j += 8) {
            double s0[8] = {}, s1[8] = {}, s2[8] = {}, s3[8] = {};
            for (int k = 0; k < depth; ++k) {
                const double v0 = l[size_t(i) * leftStride + k];
                const double v1 = l[size_t(i + 1) * leftStride + k];
                const double v2 = l[size_t(i + 2) * leftStride + k];
                const double v3 = l[size_t(i + 3) * leftStride + k];
                const double* r = rt + size_t(k) * blockSize + j;
                for (int t = 0; t < 8; ++t) {
                    s0[t] += v0 * r[t];
                    s1[t] += v1 * r[t];
                    s2[t] += v2 * r[t];
                    s3[t] += v3 * r[t];
                }
            }
            for (int t = 0; t < 8; ++t) {
                c[size_t(i) * blockSize + j + t] += s0[t];
                c[size_t(i + 1) * blockSize + j + t] += s1[t];
                c[size_t(i + 2) * blockSize + j + t] += s2[t];
                c[size_t(i + 3) * blockSize + j + t] += s3[t];
            }
        }
        for (; j < n; ++j)
            for (int ii = i; ii < i + 4; ++ii) {
                double sum = 0.0;
                for (int k = 0; k < depth; ++k)
                    sum += l[size_t(ii) * leftStride + k] * rt[size_t(k) * blockSize + j];
                c[size_t(ii) * blockSize + j] += sum;
            }
    }
    for (; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < depth; ++k)
                sum += l[size_t(i) * leftStride + k] * rt[size_t(k) * blockSize + j];
            c[size_t(i) * blockSize + j] += sum;
        }
}

// The factorization panels fit in cache. Stream whole RHS rows to maximize
// SIMD throughput and avoid repeatedly walking strided columns of a tile.
void update(double* c, const double* l, const double* rt,
            int m, int n, int depth, double sign) {
    int i = 0;
    for (; i + 3 < m; i += 4) {
        double* c0 = c + size_t(i) * blockSize;
        double* c1 = c0 + blockSize;
        double* c2 = c1 + blockSize;
        double* c3 = c2 + blockSize;
        for (int k = 0; k < depth; ++k) {
            const double v0 = sign * l[size_t(i) * blockSize + k];
            const double v1 = sign * l[size_t(i + 1) * blockSize + k];
            const double v2 = sign * l[size_t(i + 2) * blockSize + k];
            const double v3 = sign * l[size_t(i + 3) * blockSize + k];
            const double* r = rt + size_t(k) * blockSize;
            for (int j = 0; j < n; ++j) {
                c0[j] += v0 * r[j];
                c1[j] += v1 * r[j];
                c2[j] += v2 * r[j];
                c3[j] += v3 * r[j];
            }
        }
    }
    for (; i < m; ++i)
        for (int k = 0; k < depth; ++k) {
            const double value = sign * l[size_t(i) * blockSize + k];
            for (int j = 0; j < n; ++j)
                c[size_t(i) * blockSize + j] += value * rt[size_t(k) * blockSize + j];
        }
}

bool choleskyDecomposition(Matrix& a, int rank) {
    Panels panels(a);
    std::vector<double> diagonal(tileSize);
    for (int k = 0; k < a.blocks; ++k) {
        const int width = a.extent(k);
        int failed = -1;
        if (rank == a.owner(k, k)) {
            auto& d = a.tile(k, k);
            for (int j = 0; j < width; ++j) {
                double sum = 0.0;
                for (int t = 0; t < j; ++t) sum += d[size_t(j) * blockSize + t] * d[size_t(j) * blockSize + t];
                const double value = d[size_t(j) * blockSize + j] - sum;
                if (!(value > 0.0)) { failed = k * blockSize + j; break; }
                d[size_t(j) * blockSize + j] = std::sqrt(value);
                for (int i = j + 1; i < width; ++i) {
                    sum = 0.0;
                    for (int t = 0; t < j; ++t) sum += d[size_t(i) * blockSize + t] * d[size_t(j) * blockSize + t];
                    d[size_t(i) * blockSize + j] = (d[size_t(i) * blockSize + j] - sum) / d[size_t(j) * blockSize + j];
                }
                std::fill(d.begin() + size_t(j) * blockSize + j + 1,
                          d.begin() + size_t(j + 1) * blockSize, 0.0);
            }
            diagonal = d;
        }
        MPI_Bcast(&failed, 1, MPI_INT, a.owner(k, k), MPI_COMM_WORLD);
        if (failed >= 0) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", failed);
            return false;
        }
        // Only the process column owning this panel needs the diagonal tile.
        if (a.col == k % a.cols) {
            broadcast(diagonal.data(), tileSize, k % a.rows, a.colComm);
            for (int i = a.row; i < a.blocks; i += a.rows) {
                if (i <= k) continue;
                auto& panel = a.tile(i, k);
                for (int ii = 0; ii < a.extent(i); ++ii)
                    for (int j = 0; j < width; ++j) {
                        double sum = 0.0;
                        for (int t = 0; t < j; ++t)
                            sum += panel[size_t(ii) * blockSize + t] * diagonal[size_t(j) * blockSize + t];
                        panel[size_t(ii) * blockSize + j] =
                            (panel[size_t(ii) * blockSize + j] - sum) / diagonal[size_t(j) * blockSize + j];
                    }
            }
        }
        panels.exchange(a, k, k + 1);
        for (int i = a.row; i < a.blocks; i += a.rows) {
            if (i <= k) continue;
            for (int j = a.col; j <= i; j += a.cols) {
                if (j <= k) continue;
                update(a.tile(i, j).data(), panels.left.data() + size_t(i / a.rows) * tileSize,
                       panels.right.data() + size_t(j / a.cols) * tileSize,
                       a.extent(i), a.extent(j), width, -1.0);
            }
        }
    }
    return true;
}

bool validateCholesky(const Matrix& a, const std::vector<std::vector<double>>& original, int rank) {
    auto reconstructed = original;
    for (auto& tile : reconstructed) std::fill(tile.begin(), tile.end(), 0.0);
    Panels panels(a);
    for (int k = 0; k < a.blocks; ++k) {
        panels.exchange(a, k, k);
        for (int i = a.row; i < a.blocks; i += a.rows) {
            if (i < k) continue;
            for (int j = a.col; j <= i; j += a.cols) {
                if (j < k) continue;
                update(reconstructed[size_t(i / a.rows) * a.localCols + j / a.cols].data(),
                       panels.left.data() + size_t(i / a.rows) * tileSize,
                       panels.right.data() + size_t(j / a.cols) * tileSize,
                       a.extent(i), a.extent(j), a.extent(k), 1.0);
            }
        }
    }
    double errors[2] = {0.0, 0.0};
    for (int i = a.row; i < a.blocks; i += a.rows)
        for (int j = a.col; j <= i; j += a.cols) {
            const size_t index = size_t(i / a.rows) * a.localCols + j / a.cols;
            for (int ii = 0; ii < a.extent(i); ++ii)
                for (int jj = 0; jj < (i == j ? ii + 1 : a.extent(j)); ++jj) {
                    const size_t pos = size_t(ii) * blockSize + jj;
                    const double error = std::fabs(reconstructed[index][pos] - original[index][pos]);
                    if (!std::isfinite(error)) errors[0] = errors[1] = std::numeric_limits<double>::infinity();
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(original[index][pos]) + 1e-10));
                }
        }
    double global[2];
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

// Gather only on explicit request, preserving the original row-major output
// and zero upper triangle. Normal benchmark runs never allocate a full A.
void printDistributedResults(const Matrix& a, int rank) {
    std::vector<double> result;
    if (rank == 0) result.resize(a.n * a.n, 0.0);
    std::vector<double> buffer(tileSize);
    for (int i = 0; i < a.blocks; ++i)
        for (int j = 0; j <= i; ++j) {
            const int owner = a.owner(i, j);
            if (rank == owner && owner != 0)
                MPI_Send(a.tile(i, j).data(), int(tileSize), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                const double* data;
                if (owner == 0) data = a.tile(i, j).data();
                else {
                    MPI_Recv(buffer.data(), int(tileSize), MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    data = buffer.data();
                }
                for (int ii = 0; ii < a.extent(i); ++ii) {
                    const int count = i == j ? ii + 1 : a.extent(j);
                    std::copy_n(data + size_t(ii) * blockSize, count,
                                result.data() + (size_t(i) * blockSize + ii) * a.n + size_t(j) * blockSize);
                }
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

int run(int argc, char** argv, int rank, int ranks) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(text, &end, 10);
            if (text[0] == '-' || end == text || *end || errno == ERANGE ||
                value > size_t(INT_MAX) - blockSize ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", text);
                return 1;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix a(n, rank, ranks);
    generatePositiveDefiniteMatrix(a);
    std::vector<std::vector<double>> original;
    if (validate) original = a.tiles;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a, rank);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double ops = double(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? ops / duration / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(a, rank);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(a, original, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
