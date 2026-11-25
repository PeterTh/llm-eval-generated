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
constexpr int block = 32;
using Tile = std::array<double, block * block>;

struct Grid {
    int n, tiles, rank, rows, cols, row, col;
    MPI_Comm horizontal, vertical;

    explicit Grid(int size) : n(size), tiles(size / block + (size % block != 0)) {
        int ranks, dims[2] = {};
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

    int extent(int i) const { return std::min(block, n - i * block); }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
};

// Tile (i,j) lives on process (i % rows, j % cols). Omit upper tiles
// for symmetric matrices and for the lower triangular factor.
struct Matrix {
    const Grid& g;
    std::vector<std::vector<Tile>> data;

    Matrix(const Grid& grid, bool triangular) : g(grid) {
        for (int i = g.row; i < g.tiles; i += g.rows) {
            int end = triangular ? i + 1 : g.tiles;
            data.emplace_back((end + g.cols - 1 - g.col) / g.cols);
        }
    }
    Tile& at(int i, int j) { return data[i / g.rows][j / g.cols]; }
};

// Generate in the original rand_r order, distributing one tile row at a time.
void generateRandom(Matrix& b) {
    const Grid& g = b.g;
    std::vector<Tile> stripe(g.rank == 0 ? g.tiles : 0);
    unsigned seed = 42;
    for (int i = 0; i < g.tiles; ++i) {
        if (g.rank == 0) {
            for (int r = 0; r < g.extent(i); ++r)
                for (int j = 0; j < g.tiles; ++j)
                    for (int c = 0; c < g.extent(j); ++c)
                        stripe[j][r * block + c] = rand_r(&seed) / double(RAND_MAX) - 0.5;
            for (int j = 0; j < g.tiles; ++j) {
                int dest = g.owner(i, j);
                if (dest == 0) b.at(i, j) = stripe[j];
                else MPI_Send(stripe[j].data(), block * block, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        } else if (i % g.rows == g.row) {
            for (int j = g.col; j < g.tiles; j += g.cols)
                MPI_Recv(b.at(i, j).data(), block * block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

double dot(double sum, const double* a, const double* b, int width) {
    for (int k = 0; k < width; ++k) sum += a[k] * b[k];
    return sum;
}

// Accumulate independent output columns in SIMD lanes. Preserve the serial
// dot product's ordered, separately rounded products over each full block.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void updateFullTile(Tile& sums, const Tile& lhs, const Tile& transposed, int height) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (int r = 0; r < height; ++r) {
        std::array<double, block> acc;
        std::copy_n(&sums[r * block], block, acc.begin());
        for (int k = 0; k < block; ++k) {
            const double x = lhs[r * block + k];
            for (int c = 0; c < block; ++c) acc[c] += x * transposed[k * block + c];
        }
        std::copy(acc.begin(), acc.end(), &sums[r * block]);
    }
}

// Replicate the active part of tile column k along process rows, then
// redistribute its transpose along process columns for local outer products.
void accumulatePanel(Matrix& target, Matrix& source, int k, int first) {
    const Grid& g = target.g;
    int start = first + (g.row - first % g.rows + g.rows) % g.rows;
    int leftCount = (g.tiles - start + g.rows - 1) / g.rows;
    std::vector<Tile> left(leftCount);
    if (g.col == k % g.cols)
        for (int i = start; i < g.tiles; i += g.rows)
            left[(i - start) / g.rows] = source.at(i, k);

    constexpr int area = block * block;
    MPI_Bcast(left.data(), leftCount * area, MPI_DOUBLE, k % g.cols, g.horizontal);

    int begin = first + (g.col - first % g.cols + g.cols) % g.cols;
    int rightCount = (g.tiles - begin + g.cols - 1) / g.cols;
    std::vector<Tile> right;
    std::vector<int> index;
    Tile* columns;
    if (g.rows == g.cols) {
        // The diagonal rank already has the entire panel for its column.
        if (g.row == g.col) columns = left.data();
        else {
            right.resize(rightCount);
            columns = right.data();
        }
        MPI_Bcast(columns, rightCount * area, MPI_DOUBLE, g.col, g.vertical);
    } else {
        right.resize(rightCount);
        columns = right.data();
        std::vector<int> counts(g.rows), offsets(g.rows);
        for (int j = begin; j < g.tiles; j += g.cols) counts[j % g.rows] += area;
        for (int r = 1; r < g.rows; ++r) offsets[r] = offsets[r - 1] + counts[r - 1];
        auto next = offsets;
        for (int j = begin; j < g.tiles; j += g.cols) {
            int r = j % g.rows;
            int slot = next[r] / area;
            next[r] += area;
            index.push_back(slot);
            if (r == g.row) right[slot] = left[(j - start) / g.rows];
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, columns, counts.data(),
                       offsets.data(), MPI_DOUBLE, g.vertical);
    }

    for (int j = begin; j < g.tiles; j += g.cols) {
        int slot = (j - begin) / g.cols;
        const Tile& rhs = columns[index.empty() ? slot : index[slot]];
        Tile transposed;
        if (g.extent(k) == block)
            for (int r = 0; r < block; ++r)
                for (int c = 0; c < block; ++c) transposed[c * block + r] = rhs[r * block + c];
        for (int i = start; i < g.tiles; i += g.rows) {
            if (i < j) continue;
            const Tile& lhs = left[(i - start) / g.rows];
            Tile& sums = target.at(i, j);
            if (g.extent(k) == block) {
                updateFullTile(sums, lhs, transposed, g.extent(i));
                continue;
            }
            for (int r = 0; r < g.extent(i); ++r)
                for (int c = 0; c < g.extent(j); ++c) {
                    int pos = r * block + c;
                    sums[pos] = dot(sums[pos], &lhs[r * block], &rhs[c * block], g.extent(k));
                }
        }
    }
}

void gram(Matrix& a, Matrix& b, bool triangular) {
    for (int k = 0; k < a.g.tiles; ++k)
        accumulatePanel(a, b, k, triangular ? k : 0);
}

bool factor(Matrix& a, Matrix& sums) {
    const Grid& g = a.g;
    for (int k = 0; k < g.tiles; ++k) {
        Tile diagonal{};
        int width = g.extent(k);
        if (g.rank == g.owner(k, k)) {
            Tile& d = a.at(k, k);
            const Tile& partial = sums.at(k, k);
            for (int r = 0; r < width; ++r) {
                for (int c = 0; c <= r; ++c) {
                    int pos = r * block + c;
                    double v = d[pos] - dot(partial[pos], &d[r * block], &d[c * block], c);
                    d[pos] = r == c ? std::sqrt(v) : v / d[c * block + c];
                }
                std::fill(d.begin() + r * block + r + 1, d.begin() + (r + 1) * block, 0.0);
            }
            diagonal = d;
        }
        MPI_Bcast(diagonal.data(), block * block, MPI_DOUBLE, g.owner(k, k), MPI_COMM_WORLD);
        for (int c = 0; c < width; ++c) {
            if (!(diagonal[c * block + c] > 0)) {
                if (!g.rank) std::printf("Error: Matrix is not positive definite at diagonal element %d\n", k * block + c);
                return false;
            }
        }
        if (g.col == k % g.cols) {
            for (int i = g.row; i < g.tiles; i += g.rows) {
                if (i <= k) continue;
                Tile& panel = a.at(i, k);
                const Tile& partial = sums.at(i, k);
                for (int r = 0; r < g.extent(i); ++r)
                    for (int c = 0; c < width; ++c) {
                        int pos = r * block + c;
                        panel[pos] = (panel[pos] - dot(partial[pos], &panel[r * block],
                                                       &diagonal[c * block], c)) / diagonal[c * block + c];
                    }
            }
        }
        if (k + 1 < g.tiles) accumulatePanel(sums, a, k, k + 1);
    }
    return true;
}

void printFactor(Matrix& a) {
    const Grid& g = a.g;
    std::vector<double> dense(g.rank == 0 ? size_t(g.n) * g.n : 0);
    Tile received;
    for (int i = 0; i < g.tiles; ++i)
        for (int j = 0; j <= i; ++j) {
            int owner = g.owner(i, j);
            if (g.rank == 0) {
                if (owner) MPI_Recv(received.data(), block * block, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                const Tile& tile = owner ? received : a.at(i, j);
                for (int r = 0; r < g.extent(i); ++r)
                    std::copy_n(&tile[r * block], g.extent(j), &dense[size_t(i * block + r) * g.n + j * block]);
            } else if (g.rank == owner) {
                MPI_Send(a.at(i, j).data(), block * block, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
        }
    if (!g.rank) print_results(dense, "CholeskyL");
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n = atoi(argv[++i]);
            if (n < 0) {
                if (!rank) printf("Matrix size must be nonnegative\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    Grid g(n);
    if (!rank) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nGenerating positive definite matrix...\n", validate ? "enabled" : "disabled");
    }
    Matrix a(g, true);
    {
        Matrix b(g, false);
        generateRandom(b);
        gram(a, b, false);
        for (int i = g.row; i < g.tiles; i += g.rows)
            if (i % g.cols == g.col)
                for (int r = 0; r < g.extent(i); ++r) a.at(i, i)[r * block + r] += n;
    }
    Matrix sums(g, true);
    auto original = validate ? a.data : decltype(a.data){};
    if (!rank) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = factor(a, sums);
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (!rank) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (!rank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? double(n) * n * n / (3e9 * seconds) : 0);
    }
    if (results) printFactor(a);
    if (validate) {
        if (!rank) printf("Validating result...\n");
        for (auto& row : sums.data)
            for (auto& t : row) t.fill(0);
        gram(sums, a, true);
        double errors[2] = {}, global[2];
        for (int i = g.row; i < g.tiles; i += g.rows)
            for (int j = g.col; j <= i; j += g.cols) {
                const Tile& ref = original[i / g.rows][j / g.cols];
                for (int r = 0; r < g.extent(i); ++r)
                    for (int c = 0; c < g.extent(j); ++c) {
                        int pos = r * block + c;
                        double e = std::abs(sums.at(i, j)[pos] - ref[pos]);
                        errors[0] = std::max(errors[0], e);
                        errors[1] = std::max(errors[1], e / (std::abs(ref[pos]) + 1e-10));
                    }
            }
        MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        bool valid = global[1] <= 1e-6;
        if (!rank) {
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = run(argc, argv);
    MPI_Finalize();
    return status;
}
