#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <vector>

#include "../common/results_output.hpp"

namespace {
constexpr int block = 64;
constexpr int tileSize = block * block;

struct Grid {
    int rank, rows, cols, row, col;
    MPI_Comm rowComm, colComm;

    Grid() {
        int size, dims[2] = {0, 0};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Dims_create(size, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }
    ~Grid() {
        // An allocation failure must reach MPI_Abort without a collective unwind.
        if (std::uncaught_exceptions()) return;
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
};

// Only local tiles are stored. Cholesky matrices omit the upper triangle;
// the temporary random matrix also stores upper tiles.
struct Matrix {
    size_t n;
    int nt, localRows, localCols;
    const Grid& grid;
    std::vector<std::vector<double>> tiles;

    Matrix(size_t order, const Grid& g, bool full = false)
        : n(order), nt(static_cast<int>((n + block - 1) / block)),
          localRows(nt / g.rows + (g.row < nt % g.rows)),
          localCols(nt / g.cols + (g.col < nt % g.cols)), grid(g),
          tiles(static_cast<size_t>(localRows) * localCols) {
        for (int i = g.row; i < nt; i += g.rows)
            for (int j = g.col; j < nt; j += g.cols)
                if (full || i >= j) tile(i, j).resize(tileSize, 0.0);
    }
    int extent(int i) const {
        return static_cast<int>(std::min<size_t>(block, n - size_t(i) * block));
    }
    std::vector<double>& tile(int i, int j) {
        return tiles[static_cast<size_t>(i / grid.rows) * localCols + j / grid.cols];
    }
    const std::vector<double>& tile(int i, int j) const {
        return tiles[static_cast<size_t>(i / grid.rows) * localCols + j / grid.cols];
    }
};

void broadcast(double* data, size_t count, int root, MPI_Comm comm) {
    // MPI's count is an int, even when the matrix uses size_t indexing.
    while (count) {
        int chunk = static_cast<int>(std::min<size_t>(count, std::numeric_limits<int>::max()));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, comm);
        data += chunk;
        count -= chunk;
    }
}

// Exchange a block column across process rows, and its transpose across
// process columns. Per-rank panel storage and traffic scale as n*b/sqrt(P).
struct Panels {
    const Matrix& matrix;
    std::vector<double> left, right, transposed;
    std::vector<MPI_Request> requests;

    explicit Panels(const Matrix& a) : matrix(a),
        left(static_cast<size_t>(a.localRows) * tileSize),
        right(static_cast<size_t>(a.localCols) * tileSize), transposed(right.size()) {
        requests.reserve(static_cast<size_t>(a.localRows) + a.localCols);
    }
    const double* rowTile(int i) const { return left.data() + size_t(i / matrix.grid.rows) * tileSize; }
    const double* colTile(int j) const { return transposed.data() + size_t(j / matrix.grid.cols) * tileSize; }

    void exchange(int k, int first) {
        const Grid& g = matrix.grid;
        requests.clear();
        // Route each panel tile to its transposed process-grid position.
        // Receives and sends are nonblocking, including for rectangular grids.
        for (int i = first; i < matrix.nt; ++i) {
            int source = g.owner(i, k);
            int target = g.owner(k, i);
            if (g.rank == source) {
                double* p = left.data() + size_t(i / g.rows) * tileSize;
                std::copy_n(matrix.tile(i, k).data(), tileSize, p);
                if (source == target) {
                    std::copy_n(p, tileSize, right.data() + size_t(i / g.cols) * tileSize);
                } else {
                    requests.emplace_back();
                    MPI_Isend(p, tileSize, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &requests.back());
                }
            }
            if (g.rank == target && source != target) {
                requests.emplace_back();
                MPI_Irecv(right.data() + size_t(i / g.cols) * tileSize, tileSize,
                          MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &requests.back());
            }
        }
        if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        // Exclude the already completed part of the panel from broadcasts.
        int rowStart = first / g.rows + (g.row < first % g.rows);
        int colStart = first / g.cols + (g.col < first % g.cols);
        if (rowStart < matrix.localRows)
            broadcast(left.data() + size_t(rowStart) * tileSize,
                      size_t(matrix.localRows - rowStart) * tileSize, k % g.cols, g.rowComm);
        if (colStart < matrix.localCols)
            broadcast(right.data() + size_t(colStart) * tileSize,
                      size_t(matrix.localCols - colStart) * tileSize, k % g.rows, g.colComm);
        for (int t = colStart; t < matrix.localCols; ++t) {
            const double* src = right.data() + size_t(t) * tileSize;
            double* dst = transposed.data() + size_t(t) * tileSize;
            for (int i = 0; i < block; ++i)
                for (int j = 0; j < block; ++j) dst[j * block + i] = src[i * block + j];
        }
    }
};

// Small register blocks expose independent contiguous columns to SIMD without
// requiring BLAS, OpenMP, or relaxed floating-point compiler flags. Tiles
// are zero-padded to 64x64, so edge microblocks also use fixed SIMD bounds.
template<bool subtract>
void product(double* c, const double* a, const double* bt, int m, int n, int depth) {
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 8) {
            double accum[4][8] = {};
            for (int r = 0; r < 4; ++r)
                for (int s = 0; s < 8; ++s) accum[r][s] = c[(i + r) * block + j + s];
            for (int k = 0; k < depth; ++k) {
                for (int r = 0; r < 4; ++r) {
                    double x = a[(i + r) * block + k];
                    for (int s = 0; s < 8; ++s) {
                        if constexpr (subtract) accum[r][s] -= x * bt[k * block + j + s];
                        else accum[r][s] += x * bt[k * block + j + s];
                    }
                }
            }
            for (int r = 0; r < 4; ++r)
                for (int s = 0; s < 8; ++s) c[(i + r) * block + j + s] = accum[r][s];
        }
    }
}

template<bool subtract>
void update(Matrix& a, const Panels& panels, int first, int depth) {
    const Grid& g = a.grid;
    for (int i = g.row; i < a.nt; i += g.rows) {
        if (i < first) continue;
        for (int j = g.col; j <= i; j += g.cols) {
            if (j < first) continue;
            product<subtract>(a.tile(i, j).data(), panels.rowTile(i), panels.colTile(j),
                              a.extent(i), a.extent(j), depth);
        }
    }
}

void generatePositiveDefiniteMatrix(Matrix& a) {
    Matrix random(a.n, a.grid, true);
    const Grid& g = a.grid;
    // Reproduce the original rand_r sequence, retaining only locally owned
    // entries. No rank holds a complete random or positive definite matrix.
    unsigned int seed = 42;
    for (size_t i = 0; i < a.n; ++i) {
        int ti = static_cast<int>(i / block);
        for (int tj = 0; tj < a.nt; ++tj) {
            bool owned = g.owner(ti, tj) == g.rank;
            double* dst = owned ? random.tile(ti, tj).data() + (i % block) * block : nullptr;
            for (int j = 0; j < a.extent(tj); ++j) {
                double value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
                if (owned) dst[j] = value;
            }
        }
    }
    Panels panels(random);
    for (int k = 0; k < a.nt; ++k) {
        panels.exchange(k, 0);
        update<false>(a, panels, 0, a.extent(k));
    }
    for (int i = 0; i < a.nt; ++i)
        if (g.owner(i, i) == g.rank)
            for (int j = 0; j < a.extent(i); ++j) a.tile(i, i)[j * block + j] += a.n;
}

bool choleskyDecomposition(Matrix& a) {
    const Grid& g = a.grid;
    Panels panels(a);
    std::vector<double> diagonal(tileSize + 1);
    for (int k = 0; k < a.nt; ++k) {
        int width = a.extent(k);
        int owner = g.owner(k, k);
        if (g.rank == owner) {
            auto& d = a.tile(k, k);
            diagonal[tileSize] = -1;
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (int t = 0; t < j; ++t) sum += d[i * block + t] * d[j * block + t];
                    double value = d[i * block + j] - sum;
                    if (i == j) {
                        if (!(value > 0.0) || !std::isfinite(value)) {
                            diagonal[tileSize] = static_cast<double>(size_t(k) * block + i);
                            break;
                        }
                        d[i * block + j] = std::sqrt(value);
                    } else d[i * block + j] = value / d[j * block + j];
                }
                if (diagonal[tileSize] >= 0) break;
                std::fill(d.begin() + i * block + i + 1, d.begin() + (i + 1) * block, 0.0);
            }
            std::copy(d.begin(), d.end(), diagonal.begin());
        }
        // The failure index travels with the diagonal, so all ranks exit
        // together without leaving a peer blocked in a panel collective.
        MPI_Bcast(diagonal.data(), tileSize + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (diagonal[tileSize] >= 0) {
            if (g.rank == 0) printf("Error: Matrix is not positive definite at diagonal element %.0f\n", diagonal[tileSize]);
            return false;
        }
        if (g.col == k % g.cols) {
            for (int i = g.row; i < a.nt; i += g.rows) {
                if (i <= k) continue;
                auto& p = a.tile(i, k);
                for (int r = 0; r < a.extent(i); ++r)
                    for (int j = 0; j < width; ++j) {
                        double sum = 0.0;
                        for (int t = 0; t < j; ++t) sum += p[r * block + t] * diagonal[j * block + t];
                        p[r * block + j] = (p[r * block + j] - sum) / diagonal[j * block + j];
                    }
            }
        }
        panels.exchange(k, k + 1);
        update<true>(a, panels, k + 1, width);
    }
    return true;
}

bool validateCholesky(const Matrix& l, const Matrix& original) {
    Matrix reconstructed(l.n, l.grid);
    Panels panels(l);
    for (int k = 0; k < l.nt; ++k) {
        panels.exchange(k, k);
        update<false>(reconstructed, panels, k, l.extent(k));
    }
    double errors[2] = {0.0, 0.0}, global[2];
    const Grid& g = l.grid;
    for (int i = g.row; i < l.nt; i += g.rows)
        for (int j = g.col; j <= i; j += g.cols)
            for (int r = 0; r < l.extent(i); ++r)
                for (int c = 0; c < l.extent(j); ++c) {
                    if (i == j && c > r) continue;
                    double ref = original.tile(i, j)[r * block + c];
                    double value = reconstructed.tile(i, j)[r * block + c];
                    double error = std::isfinite(value) ? std::fabs(value - ref) : std::numeric_limits<double>::infinity();
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(ref) + 1e-10));
                }
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (g.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

void printDistributedResults(const Matrix& a) {
    // Gather only for the explicitly requested serial, row-major result hash.
    std::vector<double> result, buffer(tileSize);
    if (a.grid.rank == 0) result.resize(a.n * a.n, 0.0);
    for (int i = 0; i < a.nt; ++i)
        for (int j = 0; j <= i; ++j) {
            int owner = a.grid.owner(i, j);
            if (a.grid.rank == owner && owner != 0)
                MPI_Send(a.tile(i, j).data(), tileSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            if (a.grid.rank == 0) {
                const double* src;
                if (owner == 0) src = a.tile(i, j).data();
                else {
                    MPI_Recv(buffer.data(), tileSize, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    src = buffer.data();
                }
                for (int r = 0; r < a.extent(i); ++r)
                    std::copy_n(src + r * block, a.extent(j),
                                result.data() + (size_t(i) * block + r) * a.n + size_t(j) * block);
            }
        }
    if (a.grid.rank == 0) print_results(result, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    Grid grid;
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            unsigned long long value = std::strtoull(arg, &end, 10);
            if (errno || *arg == '-' || end == arg || *end ||
                value > static_cast<unsigned long long>(std::numeric_limits<int>::max() - block) ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                if (grid.rank == 0) printf("Invalid matrix size: %s\n", arg);
                return 1;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (grid.rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (grid.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (grid.rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix a(n, grid);
    generatePositiveDefiniteMatrix(a);
    // Avoid allocating the validation copy in ordinary benchmark runs.
    std::optional<Matrix> original;
    if (validate) original.emplace(a);
    if (grid.rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = choleskyDecomposition(a);
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (grid.rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (grid.rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", duration > 0 ? ops / duration / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(a);
    if (validate) {
        if (grid.rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(a, *original);
        if (grid.rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& e) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
