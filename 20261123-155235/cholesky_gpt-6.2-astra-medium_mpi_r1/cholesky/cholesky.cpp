#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

namespace {

constexpr int block = 32;
using Tile = std::array<double, block * block>;

struct Grid {
    int rank, rows, cols, row, col, n, blocks;
    MPI_Comm across, down;

    explicit Grid(int size) : n(size), blocks(n / block + (n % block != 0)) {
        int count, dims[2] = {};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &count);
        MPI_Dims_create(count, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &across);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &down);
    }

    ~Grid() {
        MPI_Comm_free(&across);
        MPI_Comm_free(&down);
    }

    int extent(int i) const { return std::min(block, n - i * block); }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
};

// Local block rows each contain only their owned block columns.
struct Matrix {
    const Grid& g;
    std::vector<std::vector<Tile>> data;

    Matrix(const Grid& grid, bool triangular = true) : g(grid) {
        for (int i = g.row; i < g.blocks; i += g.rows) {
            int end = triangular ? i + 1 : g.blocks;
            data.emplace_back((end + g.cols - 1 - g.col) / g.cols);
        }
    }

    Tile& at(int i, int j) { return data[i / g.rows][j / g.cols]; }
    const Tile& at(int i, int j) const { return data[i / g.rows][j / g.cols]; }
};

double dot(double sum, const double* a, const double* b, int length) {
    for (int k = 0; k < length; ++k)
        sum += a[k] * b[k];
    return sum;
}

// Keep a strip of independent dot products in registers. Each accumulates
// products in the same k order as the serial dot product.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void accumulate(Tile& sums, const Tile& left, const Tile& rightT,
                int height, int width) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; j += 8) {
            double acc[8];
            std::copy_n(&sums[i * block + j], 8, acc);
            for (int k = 0; k < block; ++k)
                for (int t = 0; t < 8; ++t)
                    acc[t] += left[i * block + k] * rightT[k * block + j + t];
            std::copy_n(acc, 8, &sums[i * block + j]);
        }
    }
}

// Broadcast the local part of the panel across process rows. Its transposed
// counterpart is assembled down each process column for the local rank-k update.
void update(const Matrix& source, Matrix& sums, int k, int first) {
    const Grid& g = source.g;
    int start = first + (g.row - first % g.rows + g.rows) % g.rows;
    int localRows = (g.blocks - start + g.rows - 1) / g.rows;
    std::vector<Tile> left(localRows);
    if (g.col == k % g.cols)
        for (int t = 0; t < localRows; ++t)
            left[t] = source.at(start + t * g.rows, k);
    MPI_Bcast(left.data(), localRows * block * block, MPI_DOUBLE, k % g.cols, g.across);

    std::vector<int> counts(g.rows), offsets(g.rows);
    int firstCol = first + (g.col - first % g.cols + g.cols) % g.cols;
    for (int j = firstCol; j < g.blocks; j += g.cols)
        counts[j % g.rows] += block * block;
    int total = 0;
    for (int r = 0; r < g.rows; ++r) {
        offsets[r] = total;
        total += counts[r];
    }
    std::vector<Tile> right(total / (block * block));
    std::vector<int> next = offsets;
    std::vector<int> index;
    for (int j = firstCol; j < g.blocks; j += g.cols) {
        int r = j % g.rows;
        int slot = next[r] / (block * block);
        next[r] += block * block;
        index.push_back(slot);
        if (r == g.row)
            right[slot] = left[(j - start) / g.rows];
    }
    if (g.rows == g.cols) {
        // On a square grid a single process row holds this entire column panel.
        MPI_Bcast(right.data(), total, MPI_DOUBLE, g.col, g.down);
    } else {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, right.data(), counts.data(), offsets.data(),
                       MPI_DOUBLE, g.down);
    }

    for (size_t c = 0; c < index.size(); ++c) {
        int j = firstCol + static_cast<int>(c) * g.cols;
        const Tile& rhs = right[index[c]];
        Tile transposed;
        int depth = g.extent(k);
        if (depth == block)
            for (int b = 0; b < block; ++b)
                for (int a = 0; a < block; ++a)
                    transposed[b * block + a] = rhs[a * block + b];
        int firstRow = j + (g.row - j % g.rows + g.rows) % g.rows;
        for (int i = firstRow; i < g.blocks; i += g.rows) {
            Tile& dest = sums.at(i, j);
            const Tile& lhs = left[(i - start) / g.rows];
            if (depth == block) {
                accumulate(dest, lhs, transposed, g.extent(i), g.extent(j));
            } else {
                for (int a = 0; a < g.extent(i); ++a)
                    for (int b = 0; b < g.extent(j); ++b) {
                        double& s = dest[a * block + b];
                        s = dot(s, &lhs[a * block], &rhs[b * block], depth);
                    }
            }
        }
    }
}

void generate(Matrix& matrix) {
    const Grid& g = matrix.g;
    Matrix random(g, false);
    unsigned int seed = 42;
    // Stream block rows so rand_r follows the original row-major sequence.
    std::vector<Tile> strip(g.rank == 0 ? g.blocks : 0);
    for (int i = 0; i < g.blocks; ++i) {
        if (g.rank == 0)
            for (int a = 0; a < g.extent(i); ++a)
                for (int j = 0; j < g.blocks; ++j)
                    for (int b = 0; b < g.extent(j); ++b)
                        strip[j][a * block + b] = rand_r(&seed) / (double)RAND_MAX - 0.5;
        for (int j = 0; j < g.blocks; ++j) {
            int owner = g.owner(i, j);
            if (g.rank == 0 && owner == 0)
                random.at(i, j) = strip[j];
            else if (g.rank == 0)
                MPI_Send(strip[j].data(), block * block, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD);
            else if (g.rank == owner)
                MPI_Recv(random.at(i, j).data(), block * block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }
    }
    for (int k = 0; k < g.blocks; ++k)
        update(random, matrix, k, 0);
    for (int i = g.row; i < g.blocks; i += g.rows)
        if (i % g.cols == g.col)
            for (int a = 0; a < g.extent(i); ++a)
                matrix.at(i, i)[a * block + a] += g.n;
}

int factor(Matrix& matrix, Matrix& sums) {
    const Grid& g = matrix.g;
    for (int k = 0; k < g.blocks; ++k) {
        Tile diagonal{};
        int width = g.extent(k);
        if (g.rank == g.owner(k, k)) {
            Tile& d = matrix.at(k, k);
            const Tile& s = sums.at(k, k);
            for (int a = 0; a < width; ++a) {
                for (int b = 0; b <= a; ++b) {
                    int p = a * block + b;
                    double v = d[p] - dot(s[p], &d[a * block], &d[b * block], b);
                    if (a == b) {
                        d[p] = v > 0 ? std::sqrt(v) : 0;
                        if (!d[p])
                            break;
                    } else {
                        d[p] = v / d[b * block + b];
                    }
                }
                std::fill(d.begin() + a * block + a + 1, d.begin() + (a + 1) * block, 0.);
                if (!d[a * block + a])
                    break;
            }
            diagonal = d;
        }
        MPI_Bcast(diagonal.data(), block * block, MPI_DOUBLE, g.owner(k, k), MPI_COMM_WORLD);
        for (int a = 0; a < width; ++a)
            if (!diagonal[a * block + a])
                return k * block + a;
        if (k + 1 == g.blocks)
            break;
        if (g.col == k % g.cols) {
            for (int i = g.row; i < g.blocks; i += g.rows) {
                if (i <= k)
                    continue;
                Tile& x = matrix.at(i, k);
                const Tile& s = sums.at(i, k);
                for (int a = 0; a < g.extent(i); ++a)
                    for (int b = 0; b < width; ++b) {
                        int p = a * block + b;
                        x[p] = (x[p] - dot(s[p], &x[a * block], &diagonal[b * block], b)) /
                               diagonal[b * block + b];
                    }
            }
        }
        update(matrix, sums, k, k + 1);
    }
    return -1;
}

void printFactor(const Matrix& matrix) {
    const Grid& g = matrix.g;
    std::vector<double> dense(g.rank == 0 ? size_t(g.n) * g.n : 0);
    for (int i = 0; i < g.blocks; ++i)
        for (int j = 0; j <= i; ++j) {
            Tile received;
            int owner = g.owner(i, j);
            if (g.rank == owner && owner != 0)
                MPI_Send(matrix.at(i, j).data(), block * block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (g.rank != 0)
                continue;
            if (owner != 0)
                MPI_Recv(received.data(), block * block, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            const Tile& tile = owner == 0 ? matrix.at(i, j) : received;
            for (int a = 0; a < g.extent(i); ++a)
                std::copy_n(&tile[a * block], g.extent(j),
                            &dense[size_t(i * block + a) * g.n + j * block]);
        }
    if (g.rank == 0)
        print_results(dense, "CholeskyL");
}

bool validateFactor(const Matrix& matrix, const Matrix& original) {
    const Grid& g = matrix.g;
    double errors[2] = {};
    Matrix product(g);
    for (int k = 0; k < g.blocks; ++k)
        update(matrix, product, k, k);
    for (int i = g.row; i < g.blocks; i += g.rows)
        for (int j = g.col; j <= i; j += g.cols)
            for (int a = 0; a < g.extent(i); ++a)
                for (int b = 0; b < g.extent(j); ++b) {
                    int p = a * block + b;
                    double expected = original.at(i, j)[p];
                    double error = std::abs(product.at(i, j)[p] - expected);
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::abs(expected) + 1e-10));
                }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (g.rank == 0) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
        if (errors[1] > 1e-6)
            printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
            if (n < 0) {
                if (!rank)
                    printf("Matrix size must be nonnegative\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            results = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (!rank)
                printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    Grid grid(n);
    if (!rank) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nGenerating positive definite matrix...\n",
               validate ? "enabled" : "disabled");
    }
    Matrix matrix(grid);
    generate(matrix);
    std::optional<Matrix> original;
    if (validate)
        original.emplace(matrix);
    if (!rank)
        printf("Computing Cholesky decomposition...\n");
    {
        Matrix sums(grid);
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        int failed = factor(matrix, sums);
        double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (failed >= 0) {
            if (!rank)
                printf("Error: Matrix is not positive definite at diagonal element %d\n"
                       "Cholesky decomposition failed\n",
                       failed);
            return 1;
        }
        if (!rank) {
            printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
            printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * seconds));
        }
    }
    if (results)
        printFactor(matrix);
    if (validate) {
        if (!rank)
            printf("Validating result...\n");
        bool valid = validateFactor(matrix, *original);
        if (!rank)
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = benchmark(argc, argv);
    MPI_Finalize();
    return status;
}
