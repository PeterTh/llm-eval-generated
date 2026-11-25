#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

namespace {
constexpr int width = 32;
using Tile = std::array<double, width * width>;

struct Grid {
    int rank, rows, cols, row, col, n, blocks;
    MPI_Comm horizontal, vertical;

    explicit Grid(int size) : n(size), blocks(n / width + (n % width != 0)) {
        int ranks, dims[2]{};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        MPI_Dims_create(ranks, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &horizontal);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &vertical);
    }
    ~Grid() {
        MPI_Comm_free(&horizontal);
        MPI_Comm_free(&vertical);
    }
    int extent(int i) const { return std::min(width, n - i * width); }
};

int localCount(int end, int coordinate, int stride) { return end / stride + (coordinate < end % stride); }

// Each local block row contains consecutive locally owned tiles.
struct Matrix {
    const Grid &g;
    std::vector<std::vector<Tile>> data;

    explicit Matrix(const Grid &grid, bool full = false) : g(grid) {
        for (int i = g.row; i < g.blocks; i += g.rows)
            data.emplace_back(localCount(full ? g.blocks : i + 1, g.col, g.cols));
    }
    Tile &at(int i, int j) { return data[i / g.rows][j / g.cols]; }
    const Tile &at(int i, int j) const { return data[i / g.rows][j / g.cols]; }
};

void generateRandom(Matrix &b) {
    const Grid &g = b.g;
    unsigned seed = 42;
    // Pack one block row at a time in process-column order, retaining
    // the original row-major rand_r stream, including the partial tiles.
    std::vector<int> counts(g.cols), offsets(g.cols);
    int total = 0;
    for (int c = 0; c < g.cols; ++c) {
        offsets[c] = total;
        counts[c] = localCount(g.blocks, c, g.cols) * width * width;
        total += counts[c];
    }
    std::vector<double> buffer(g.rank == 0 || g.col == 0 ? total : 0);
    for (int i = 0; i < g.blocks; ++i) {
        int owner = i % g.rows;
        if (g.rank == 0) {
            for (int r = 0; r < g.extent(i); ++r)
                for (int j = 0; j < g.blocks; ++j) {
                    double *out = buffer.data() + offsets[j % g.cols] + (j / g.cols * width + r) * width;
                    for (int c = 0; c < g.extent(j); ++c)
                        out[c] = rand_r(&seed) / double(RAND_MAX) - 0.5;
                }
            if (owner)
                MPI_Send(buffer.data(), total, MPI_DOUBLE, owner * g.cols, 0, MPI_COMM_WORLD);
        }
        if (g.row == owner) {
            if (owner && g.col == 0)
                MPI_Recv(buffer.data(), total, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Scatterv(buffer.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                         b.data[i / g.rows].data(), counts[g.col], MPI_DOUBLE, 0, g.horizontal);
        }
    }
}

// Replicate active panel tiles along process rows, then exchange the
// subset needed as transposed operands within each process column.
struct Panel {
    const Grid &g;
    std::vector<Tile> left, right, send;
    std::vector<int> counts, offsets, index;

    explicit Panel(const Grid &grid)
        : g(grid), left(localCount(g.blocks, g.row, g.rows)), right(localCount(g.blocks, g.col, g.cols)),
          send(right.size()), counts(g.rows), offsets(g.rows), index(right.size()) {}

    void exchange(const Matrix &a, int k, int first) {
        int begin = localCount(first, g.row, g.rows);
        int count = static_cast<int>(left.size()) - begin;
        if (g.col == k % g.cols)
            for (int t = begin; t < static_cast<int>(left.size()); ++t)
                left[t] = a.at(t * g.rows + g.row, k);
        if (count)
            MPI_Bcast(left[begin].data(), count * width * width, MPI_DOUBLE, k % g.cols, g.horizontal);

        std::fill(counts.begin(), counts.end(), 0);
        int start = localCount(first, g.col, g.cols);
        for (int t = start; t < static_cast<int>(right.size()); ++t) {
            int i = t * g.cols + g.col;
            int owner = i % g.rows;
            index[t] = counts[owner]++;
            if (owner == g.row)
                send[index[t]] = left[i / g.rows];
        }
        int total = 0;
        for (int r = 0; r < g.rows; ++r) {
            offsets[r] = total;
            counts[r] *= width * width;
            total += counts[r];
        }
        if (total)
            MPI_Allgatherv(send.data(), counts[g.row], MPI_DOUBLE, right.data(), counts.data(),
                           offsets.data(), MPI_DOUBLE, g.vertical);
        for (int t = start; t < static_cast<int>(right.size()); ++t)
            index[t] += offsets[(t * g.cols + g.col) % g.rows] / (width * width);
    }
};

double dot(double sum, const double *x, const double *y, int length) {
    for (int k = 0; k < length; ++k)
        sum += x[k] * y[k];
    return sum;
}

// Four independent rows share the transposed operand. SIMD lanes span
// output columns so each dot product accumulates in its original order.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void updateFull(Tile& sum, const Tile& x, const Tile& transposed, int rows) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (int i = 0; i < rows; i += 4)
        for (int j = 0; j < width; j += 8) {
            using Lanes = double __attribute__((vector_size(4 * sizeof(double))));
            Lanes acc[4][2];
            for (int r = 0; r < 4; ++r)
                std::memcpy(acc[r], &sum[(i + r) * width + j], sizeof(acc[r]));
            for (int k = 0; k < width; ++k) {
                Lanes y[2];
                std::memcpy(y, &transposed[k * width + j], sizeof(y));
                for (int r = 0; r < 4; ++r) {
                    double value = x[(i + r) * width + k];
                    acc[r][0] += value * y[0];
                    acc[r][1] += value * y[1];
                }
            }
            for (int r = 0; r < 4; ++r)
                std::memcpy(&sum[(i + r) * width + j], acc[r], sizeof(acc[r]));
        }
}

void accumulate(Matrix &sums, Panel &panel, const Matrix &source, int k, int first) {
    const Grid &g = sums.g;
    panel.exchange(source, k, first);
    for (int t = localCount(first, g.col, g.cols); t < static_cast<int>(panel.right.size()); ++t) {
        int j = t * g.cols + g.col;
        const Tile &y = panel.right[panel.index[t]];
        Tile transposed;
        if (g.extent(k) == width)
            for (int r = 0; r < width; ++r)
                for (int c = 0; c < width; ++c)
                    transposed[c * width + r] = y[r * width + c];
        for (int s = localCount(j, g.row, g.rows); s < static_cast<int>(panel.left.size()); ++s) {
            int i = s * g.rows + g.row;
            Tile &out = sums.at(i, j);
            const Tile &x = panel.left[s];
            if (g.extent(k) == width) {
                updateFull(out, x, transposed, g.extent(i));
            } else {
                for (int r = 0; r < g.extent(i); ++r)
                    for (int c = 0; c < g.extent(j); ++c)
                        out[r * width + c] =
                            dot(out[r * width + c], &x[r * width], &y[c * width], g.extent(k));
            }
        }
    }
}

Matrix generate(const Grid &g, Panel &panel) {
    Matrix b(g, true), a(g);
    generateRandom(b);
    for (int k = 0; k < g.blocks; ++k)
        accumulate(a, panel, b, k, 0);
    for (int i = g.row; i < g.blocks; i += g.rows)
        if (i % g.cols == g.col)
            for (int r = 0; r < g.extent(i); ++r)
                a.at(i, i)[r * width + r] += g.n;
    return a;
}

// Factor the diagonal tile and solve its block column. Earlier panels
// have already contributed to sums, leaving only in-tile dot products.
int factor(const Matrix &a, Matrix &l, Panel &panel) {
    const Grid &g = a.g;
    for (int k = 0; k < g.blocks; ++k) {
        if (g.col == k % g.cols) {
            Tile diagonal{};
            if (g.row == k % g.rows) {
                Tile &sums = l.at(k, k);
                for (int r = 0; r < g.extent(k); ++r)
                    for (int c = 0; c <= r; ++c) {
                        double value =
                            a.at(k, k)[r * width + c] -
                            dot(sums[r * width + c], &diagonal[r * width], &diagonal[c * width], c);
                        diagonal[r * width + c] =
                            r == c ? std::sqrt(value) : value / diagonal[c * width + c];
                    }
                sums = diagonal;
            }
            MPI_Bcast(diagonal.data(), width * width, MPI_DOUBLE, k % g.rows, g.vertical);
            for (int s = localCount(k + 1, g.row, g.rows); s < static_cast<int>(panel.left.size()); ++s) {
                int i = s * g.rows + g.row;
                Tile &out = l.at(i, k);
                const Tile &input = a.at(i, k);
                for (int r = 0; r < g.extent(i); ++r)
                    for (int c = 0; c < g.extent(k); ++c) {
                        int p = r * width + c;
                        out[p] = (input[p] - dot(out[p], &out[r * width], &diagonal[c * width], c)) /
                                 diagonal[c * width + c];
                    }
            }
        }
        if (k + 1 < g.blocks)
            accumulate(l, panel, l, k, k + 1);
    }
    int bad = g.n;
    for (int i = g.row; i < g.blocks; i += g.rows)
        if (i % g.cols == g.col)
            for (int r = 0; r < g.extent(i); ++r)
                if (!(l.at(i, i)[r * width + r] > 0))
                    bad = std::min(bad, i * width + r);
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return bad;
}

bool validate(const Matrix &a, const Matrix &l, Panel &panel) {
    const Grid &g = a.g;
    Matrix product(g);
    for (int k = 0; k < g.blocks; ++k)
        accumulate(product, panel, l, k, k);
    double errors[2]{};
    for (int i = g.row; i < g.blocks; i += g.rows)
        for (int j = g.col; j <= i; j += g.cols)
            for (int r = 0; r < g.extent(i); ++r)
                for (int c = 0; c < g.extent(j); ++c) {
                    int p = r * width + c;
                    double v = a.at(i, j)[p];
                    double error = std::abs(product.at(i, j)[p] - v);
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::abs(v) + 1e-10));
                }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    bool valid = errors[1] <= 1e-6;
    if (g.rank == 0) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
        if (!valid)
            printf("Validation failed: relative error too large\n");
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

void results(const Matrix &l) {
    const Grid &g = l.g;
    std::vector<double> dense(g.rank == 0 ? size_t(g.n) * g.n : 0);
    Tile buffer;
    for (int i = 0; i < g.blocks; ++i)
        for (int j = 0; j <= i; ++j) {
            int owner = i % g.rows * g.cols + j % g.cols;
            if (g.rank == 0) {
                if (owner)
                    MPI_Recv(buffer.data(), width * width, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                const Tile &tile = owner ? buffer : l.at(i, j);
                for (int r = 0; r < g.extent(i); ++r)
                    std::copy_n(&tile[r * width], g.extent(j),
                                &dense[size_t(i * width + r) * g.n + j * width]);
            } else if (g.rank == owner)
                MPI_Send(l.at(i, j).data(), width * width, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    if (g.rank == 0)
        print_results(dense, "CholeskyL");
}

void usage(const char *name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char **argv) {
    int rank, n = 512;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool check = false, report = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n = atoi(argv[++i]);
            if (n < 0) {
                if (!rank)
                    printf("Matrix size must be nonnegative\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "-v"))
            check = true;
        else if (!strcmp(argv[i], "-r"))
            report = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank)
                usage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                usage(argv[0]);
            }
            return 1;
        }
    }

    Grid g(n);
    Panel panel(g);
    if (!rank) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nGenerating positive definite matrix...\n", check ? "enabled" : "disabled");
    }
    Matrix a = generate(g, panel);
    Matrix l(g);
    if (!rank)
        printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int bad = factor(a, l, panel);
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (bad < n) {
        if (!rank)
            printf("Error: Matrix is not positive definite at diagonal element %d\nCholesky decomposition "
                   "failed\n",
                   bad);
        return 1;
    }
    if (!rank) {
        printf("Computation time: %ld ms\n", long(duration * 1000));
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * duration));
    }
    if (report)
        results(l);
    if (check && !rank)
        printf("Validating result...\n");
    return check && !validate(a, l, panel) ? 1 : 0;
}
} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int status = run(argc, argv);
    MPI_Finalize();
    return status;
}
