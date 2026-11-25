#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {
constexpr int side = 32;
using Tile = std::array<double, side * side>;
using Matrix = std::vector<Tile>;

int localCount(int blocks, int coordinate, int stride) {
    return blocks / stride + (coordinate < blocks % stride);
}

// Tile (i,j) belongs to process (i % rows, j % cols).
struct Grid {
    int n, blocks, rows, cols, row, col, nr, nc;
    MPI_Comm across, down;

    Grid(int size, int rank, int ranks) : n(size), blocks(size / side + (size % side != 0)) {
        int dims[2] = {};
        MPI_Dims_create(ranks, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        nr = localCount(blocks, row, rows);
        nc = localCount(blocks, col, cols);
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &across);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &down);
    }
    ~Grid() {
        MPI_Comm_free(&across);
        MPI_Comm_free(&down);
    }
    size_t index(int i, int j) const { return size_t(i / rows) * nc + j / cols; }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
    int extent(int i) const { return std::min(side, n - i * side); }
};

// Stream the original rand_r sequence once, distributing a block row at a time.
Matrix randomMatrix(const Grid &g, int rank) {
    Matrix b(size_t(g.nr) * g.nc);
    Matrix strip(rank == 0 ? g.blocks : 0);
    unsigned seed = 42;
    for (int i = 0; i < g.blocks; ++i) {
        if (rank == 0) {
            for (int r = 0; r < g.extent(i); ++r)
                for (int j = 0; j < g.blocks; ++j)
                    for (int c = 0; c < g.extent(j); ++c)
                        strip[j][r * side + c] = rand_r(&seed) / double(RAND_MAX) - 0.5;
            for (int j = 0; j < g.blocks; ++j) {
                int dest = g.owner(i, j);
                if (dest == 0)
                    b[g.index(i, j)] = strip[j];
                else
                    MPI_Send(strip[j].data(), side * side, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        } else if (i % g.rows == g.row) {
            for (int j = g.col; j < g.blocks; j += g.cols)
                MPI_Recv(b[g.index(i, j)].data(), side * side, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }
    }
    return b;
}

double dot(double sum, const double *a, const double *b, int length) {
    for (int k = 0; k < length; ++k)
        sum += a[k] * b[k];
    return sum;
}

// Accumulate in the original k order. Vectorize independent output columns.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void fullProduct(Tile& sums, const Tile& a, const Tile& bt) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; j += 8) {
            double values[8];
            std::copy_n(&sums[i * side + j], 8, values);
            for (int k = 0; k < side; ++k)
                for (int t = 0; t < 8; ++t)
                    values[t] += a[i * side + k] * bt[k * side + j + t];
            std::copy_n(values, 8, &sums[i * side + j]);
        }
    }
}

// Share the active part of a tile column across process rows, then redistribute
// the right operands down process columns. Pack one message per collective.
void panelProduct(const Grid &g, const Matrix &input, Matrix &sums, int k, int first) {
    int r0 = localCount(first, g.row, g.rows);
    int c0 = localCount(first, g.col, g.cols);
    Matrix left(g.nr - r0);
    if (g.col == k % g.cols)
        for (int r = r0; r < g.nr; ++r)
            left[r - r0] = input[size_t(r) * g.nc + k / g.cols];
    MPI_Bcast(left.data(), int(left.size()) * side * side, MPI_DOUBLE, k % g.cols, g.across);

    std::vector<int> counts(g.rows), offsets(g.rows), slots(g.nc - c0);
    for (int c = c0; c < g.nc; ++c)
        counts[(g.col + c * g.cols) % g.rows] += side * side;
    for (int p = 1; p < g.rows; ++p)
        offsets[p] = offsets[p - 1] + counts[p - 1];
    auto cursor = offsets;
    Matrix right(g.nc - c0);
    for (int c = c0; c < g.nc; ++c) {
        int j = g.col + c * g.cols, src = j % g.rows;
        int slot = cursor[src] / (side * side);
        cursor[src] += side * side;
        slots[c - c0] = slot;
        if (src == g.row) {
            const Tile &tile = left[j / g.rows - r0];
            if (g.extent(k) == side) {
                for (int t = 0; t < side; ++t)
                    for (int u = 0; u < side; ++u)
                        right[slot][t * side + u] = tile[u * side + t];
            } else
                right[slot] = tile;
        }
    }
    if (g.rows == g.cols) {
        MPI_Bcast(right.data(), int(right.size()) * side * side, MPI_DOUBLE, g.col, g.down);
    } else
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, right.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                       g.down);
    for (int r = r0; r < g.nr; ++r) {
        int i = g.row + r * g.rows;
        for (int c = c0; c < g.nc && g.col + c * g.cols <= i; ++c) {
            Tile &out = sums[size_t(r) * g.nc + c];
            const Tile &a = left[r - r0];
            const Tile &b = right[slots[c - c0]];
            if (g.extent(k) == side)
                fullProduct(out, a, b);
            else {
                for (int x = 0; x < g.extent(i); ++x)
                    for (int y = 0; y < g.extent(g.col + c * g.cols); ++y)
                        out[x * side + y] = dot(out[x * side + y], &a[x * side], &b[y * side], g.extent(k));
            }
        }
    }
}

Matrix generate(const Grid &g, int rank) {
    Matrix b = randomMatrix(g, rank), a(b.size());
    for (int k = 0; k < g.blocks; ++k)
        panelProduct(g, b, a, k, 0);
    for (int k = g.row; k < g.blocks; k += g.rows)
        if (k % g.cols == g.col)
            for (int d = 0; d < g.extent(k); ++d)
                a[g.index(k, k)][d * side + d] += g.n;
    return a;
}

// Sums holds products from completed panels so each element is computed
// with a single subtraction from the original matrix, as in the scalar solve.
bool factor(const Grid &g, Matrix &a, Matrix &sums, int rank) {
    for (int k = 0; k < g.blocks; ++k) {
        int width = g.extent(k);
        Tile diag{};
        if (rank == g.owner(k, k)) {
            Tile &block = a[g.index(k, k)];
            const Tile &prior = sums[g.index(k, k)];
            for (int j = 0; j < width; ++j) {
                for (int i = j; i < width; ++i) {
                    int ij = i * side + j;
                    double value = block[ij] - dot(prior[ij], &block[i * side], &block[j * side], j);
                    if (i == j && value <= 0) {
                        printf("Error: Matrix is not positive definite at diagonal element %d\n",
                               k * side + j);
                        block[ij] = -1;
                        break;
                    }
                    block[ij] = i == j ? std::sqrt(value) : value / block[j * side + j];
                }
                if (block[j * side + j] < 0) {
                    block[0] = -1;
                    break;
                }
                std::fill(block.begin() + j * side + j + 1, block.begin() + (j + 1) * side, 0.0);
            }
            diag = block;
        }
        MPI_Bcast(diag.data(), side * side, MPI_DOUBLE, g.owner(k, k), MPI_COMM_WORLD);
        if (diag[0] < 0)
            return false;
        if (g.col == k % g.cols) {
            for (int r = localCount(k + 1, g.row, g.rows); r < g.nr; ++r) {
                int i = g.row + r * g.rows;
                Tile &block = a[g.index(i, k)];
                const Tile &prior = sums[g.index(i, k)];
                for (int x = 0; x < g.extent(i); ++x)
                    for (int j = 0; j < width; ++j) {
                        int ij = x * side + j;
                        block[ij] = (block[ij] - dot(prior[ij], &block[x * side], &diag[j * side], j)) /
                                    diag[j * side + j];
                    }
            }
        }
        if (k + 1 < g.blocks)
            panelProduct(g, a, sums, k, k + 1);
    }
    return true;
}

bool check(const Grid &g, const Matrix &l, const Matrix &orig, Matrix &sums, int rank) {
    std::fill(sums.begin(), sums.end(), Tile{});
    for (int k = 0; k < g.blocks; ++k)
        panelProduct(g, l, sums, k, k);
    double local[2] = {}, errors[2];
    for (int i = g.row; i < g.blocks; i += g.rows)
        for (int j = g.col; j <= i; j += g.cols) {
            size_t pos = g.index(i, j);
            for (int x = 0; x < g.extent(i); ++x)
                for (int y = 0; y < g.extent(j); ++y) {
                    int xy = x * side + y;
                    double err = std::abs(sums[pos][xy] - orig[pos][xy]);
                    local[0] = std::max(local[0], err);
                    local[1] = std::max(local[1], err / (std::abs(orig[pos][xy]) + 1e-10));
                }
        }
    MPI_Allreduce(local, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
        if (errors[1] > 1e-6)
            printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

void report(const Grid &g, const Matrix &a, int rank) {
    std::vector<double> result(rank == 0 ? size_t(g.n) * g.n : 0);
    Tile received;
    for (int i = 0; i < g.blocks; ++i)
        for (int j = 0; j <= i; ++j) {
            int owner = g.owner(i, j);
            if (rank == 0) {
                if (owner != 0)
                    MPI_Recv(received.data(), side * side, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                const Tile &tile = owner == 0 ? a[g.index(i, j)] : received;
                for (int x = 0; x < g.extent(i); ++x)
                    std::copy_n(&tile[x * side], g.extent(j),
                                &result[size_t(i * side + x) * g.n + j * side]);
            } else if (rank == owner)
                MPI_Send(a[g.index(i, j)].data(), side * side, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    if (rank == 0)
        print_results(result, "CholeskyL");
}

void printUsage(const char *progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char **argv, int rank, int ranks) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0)
                printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0)
                printf("Unknown option: %s\n", argv[i]);
            if (rank == 0)
                printUsage(argv[0]);
            return 1;
        }
    }

    if (n > size_t(std::numeric_limits<int>::max())) {
        if (rank == 0)
            printf("Invalid matrix size\n");
        return 1;
    }
    Grid grid(static_cast<int>(n), rank, ranks);
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix a = generate(grid, rank);
    Matrix original;
    if (validate)
        original = a;
    Matrix sums(a.size());
    if (rank == 0)
        printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = factor(grid, a, sums, rank);
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0)
            printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * seconds));
    }
    if (printResults)
        report(grid, a, rank);
    if (validate) {
        if (rank == 0)
            printf("Validating result...\n");
        bool valid = check(grid, a, original, sums, rank);
        if (rank == 0)
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = benchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
