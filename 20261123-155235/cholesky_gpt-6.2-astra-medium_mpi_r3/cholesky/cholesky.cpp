#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

namespace {
constexpr int width = 32;
struct Tile {
    double a[width][width]{};
};
static_assert(sizeof(Tile) == width * width * sizeof(double));

struct Grid {
    int rows, cols, row, col, blocks, n;
    MPI_Comm horizontal, vertical;
    MPI_Datatype tileType;
    std::vector<int> rowBlocks, colBlocks;
    // The lower tiles of a local block row are contiguous.
    std::vector<size_t> offsets{0};

    Grid(int size, int rank, int order) : blocks(order / width + (order % width != 0)), n(order) {
        MPI_Type_contiguous(width * width, MPI_DOUBLE, &tileType);
        MPI_Type_commit(&tileType);
        int dims[2]{};
        MPI_Dims_create(size, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &horizontal);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &vertical);
        for (int j = col; j < blocks; j += cols)
            colBlocks.push_back(j);
        for (int i = row; i < blocks; i += rows) {
            rowBlocks.push_back(i);
            offsets.push_back(offsets.back() + (std::upper_bound(colBlocks.begin(), colBlocks.end(), i) - colBlocks.begin()));
        }
    }
    ~Grid() {
        MPI_Type_free(&tileType);
        MPI_Comm_free(&horizontal);
        MPI_Comm_free(&vertical);
    }
    int extent(int block) const { return std::min(width, n - block * width); }
    size_t index(int i, int j) const { return offsets[i / rows] + j / cols; }
};

double dot(double sum, const double* x, const double* y, int count) {
    for (int k = 0; k < count; ++k)
        sum += x[k] * y[k];
    return sum;
}

// Vectorize over independent output columns while retaining the order of
// additions within each dot product. Transposed panel tiles are reused.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void addProduct(Tile& sums, const Tile& left, const Tile& transposed, int height, int columns) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < columns; j += 8) {
            double acc[8];
            std::copy_n(&sums.a[i][j], 8, acc);
            for (int k = 0; k < width; ++k)
                for (int v = 0; v < 8; ++v)
                    acc[v] += left.a[i][k] * transposed.a[k][j + v];
            std::copy_n(acc, 8, &sums.a[i][j]);
        }
    }
}

struct Panels {
    std::vector<Tile> left, right, transposed;
    std::vector<int> counts, offsets, cursor;
    explicit Panels(const Grid& g) : left(g.rowBlocks.size()), right(g.colBlocks.size()), transposed(right.size()),
        counts(g.rows), offsets(g.rows), cursor(g.rows) {}

    void accumulate(const Grid& g, std::vector<Tile>& sums, int first, int depth) {
        if (depth == width) {
            for (size_t c = 0; c < right.size(); ++c) {
                if (g.colBlocks[c] < first) continue;
                for (int j = 0; j < g.extent(g.colBlocks[c]); ++j)
                    for (int k = 0; k < width; ++k)
                        transposed[c].a[k][j] = right[c].a[j][k];
            }
        }
        size_t begin = std::lower_bound(g.colBlocks.begin(), g.colBlocks.end(), first) - g.colBlocks.begin();
        for (size_t r = 0; r < left.size(); ++r) {
            int i = g.rowBlocks[r];
            for (size_t c = begin; c < right.size() && g.colBlocks[c] <= i; ++c) {
                int j = g.colBlocks[c];
                Tile& tile = sums[g.offsets[r] + c];
                if (depth == width) {
                    addProduct(tile, left[r], transposed[c], g.extent(i), g.extent(j));
                } else {
                    for (int u = 0; u < g.extent(i); ++u)
                        for (int v = 0; v < g.extent(j); ++v)
                            tile.a[u][v] = dot(tile.a[u][v], left[r].a[u], right[c].a[v], depth);
                }
            }
        }
    }

    // Broadcast the solved column panel along process rows, then exchange
    // the corresponding row tiles within each process column.
    void distribute(const Grid& g, const std::vector<Tile>& factor, int k, int first) {
        size_t begin = std::lower_bound(g.rowBlocks.begin(), g.rowBlocks.end(), first) - g.rowBlocks.begin();
        if (g.col == k % g.cols)
            for (size_t r = begin; r < left.size(); ++r)
                left[r] = factor[g.index(g.rowBlocks[r], k)];
        if (begin < left.size())
            MPI_Bcast(&left[begin], static_cast<int>(left.size() - begin), g.tileType, k % g.cols, g.horizontal);

        size_t start = std::lower_bound(g.colBlocks.begin(), g.colBlocks.end(), first) - g.colBlocks.begin();
        if (start == right.size()) return;
        if (g.rows == g.cols) {
            if (g.row == g.col)
                std::copy(left.begin() + start, left.end(), right.begin() + start);
            MPI_Bcast(&right[start], static_cast<int>(right.size() - start), g.tileType, g.col, g.vertical);
            return;
        }

        // On a rectangular grid multiple process rows can supply this column.
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t c = start; c < right.size(); ++c)
            ++counts[g.colBlocks[c] % g.rows];
        for (int r = 1; r < g.rows; ++r)
            offsets[r] = offsets[r - 1] + counts[r - 1];
        int slot = offsets[g.row];
        for (size_t c = start; c < right.size(); ++c) {
            int j = g.colBlocks[c];
            if (j % g.rows == g.row) transposed[slot++] = left[j / g.rows];
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, g.tileType, transposed.data(), counts.data(), offsets.data(),
                       g.tileType, g.vertical);
        cursor = offsets;
        for (size_t c = start; c < right.size(); ++c)
            right[c] = transposed[cursor[g.colBlocks[c] % g.rows]++];
    }
};

void generate(const Grid& g, std::vector<Tile>& matrix, Panels& panels, int rank) {
    // Each row stream starts at exactly the same rand_r state as in the
    // sequential generator, regardless of tile ownership.
    std::vector<unsigned> rowSeed(g.n);
    if (rank == 0) {
        unsigned seed = 42;
        for (int i = 0; i < g.n; ++i) {
            rowSeed[i] = seed;
            for (int j = 0; j < g.n; ++j) rand_r(&seed);
        }
    }
    MPI_Bcast(rowSeed.data(), g.n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    auto colSeed = rowSeed;
    auto fill = [&](std::vector<Tile>& panel, const std::vector<int>& blocks,
                    std::vector<unsigned>& seeds, int depth) {
        for (size_t b = 0; b < blocks.size(); ++b)
            for (int i = 0; i < g.extent(blocks[b]); ++i) {
                auto& seed = seeds[blocks[b] * width + i];
                for (int k = 0; k < depth; ++k)
                    panel[b].a[i][k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
            }
    };
    for (int k = 0; k < g.blocks; ++k) {
        int depth = g.extent(k);
        fill(panels.left, g.rowBlocks, rowSeed, depth);
        if (g.rowBlocks == g.colBlocks) panels.right = panels.left;
        else fill(panels.right, g.colBlocks, colSeed, depth);
        panels.accumulate(g, matrix, 0, depth);
    }
    for (int i : g.rowBlocks)
        if (i % g.cols == g.col)
            for (int u = 0; u < g.extent(i); ++u)
                matrix[g.index(i, i)].a[u][u] += g.n;
}

// Before a tile is solved it holds the accumulated dot products from all
// preceding panels. Subtract once at the solve to preserve rounding.
bool factorize(const Grid& g, const std::vector<Tile>& matrix, std::vector<Tile>& factor, Panels& panels) {
    for (int k = 0; k < g.blocks; ++k) {
        Tile diag;
        int depth = g.extent(k);
        if (g.row == k % g.rows && g.col == k % g.cols) {
            size_t pos = g.index(k, k);
            for (int i = 0; i < depth; ++i)
                for (int j = 0; j <= i; ++j) {
                    double sum = dot(factor[pos].a[i][j], diag.a[i], diag.a[j], j);
                    double value = matrix[pos].a[i][j] - sum;
                    diag.a[i][j] = i == j ? std::sqrt(value) : value / diag.a[j][j];
                }
            factor[pos] = diag;
        }
        MPI_Bcast(&diag, 1, g.tileType, (k % g.rows) * g.cols + k % g.cols, MPI_COMM_WORLD);
        for (int j = 0; j < depth; ++j)
            if (!(diag.a[j][j] > 0.0)) {
                if (g.row == 0 && g.col == 0)
                    printf("Error: Matrix is not positive definite at diagonal element %d\n", k * width + j);
                return false;
            }
        if (g.col == k % g.cols) {
            for (int i : g.rowBlocks) {
                if (i <= k) continue;
                size_t pos = g.index(i, k);
                Tile& tile = factor[pos];
                for (int u = 0; u < g.extent(i); ++u)
                    for (int v = 0; v < depth; ++v) {
                        double sum = dot(tile.a[u][v], tile.a[u], diag.a[v], v);
                        tile.a[u][v] = (matrix[pos].a[u][v] - sum) / diag.a[v][v];
                    }
            }
        }
        panels.distribute(g, factor, k, k + 1);
        panels.accumulate(g, factor, k + 1, depth);
    }
    return true;
}

bool validate(const Grid& g, const std::vector<Tile>& factor, const std::vector<Tile>& matrix, Panels& panels) {
    std::vector<Tile> product(factor.size());
    for (int k = 0; k < g.blocks; ++k) {
        panels.distribute(g, factor, k, k);
        panels.accumulate(g, product, k, g.extent(k));
    }
    double errors[2]{};
    for (int i : g.rowBlocks)
        for (int j : g.colBlocks) {
            if (j > i) break;
            size_t pos = g.index(i, j);
            for (int u = 0; u < g.extent(i); ++u)
                for (int v = 0; v < g.extent(j); ++v) {
                    double orig = matrix[pos].a[u][v];
                    double error = std::abs(product[pos].a[u][v] - orig);
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::abs(orig) + 1e-10));
                }
        }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (g.row == 0 && g.col == 0) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

void report(const Grid& g, const std::vector<Tile>& factor, int rank) {
    std::vector<double> dense;
    if (rank == 0) dense.resize(size_t(g.n) * g.n);
    for (int i = 0; i < g.blocks; ++i)
        for (int j = 0; j <= i; ++j) {
            int owner = (i % g.rows) * g.cols + j % g.cols;
            Tile incoming;
            if (rank == 0) {
                if (owner) MPI_Recv(&incoming, 1, g.tileType, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                const Tile& tile = owner ? incoming : factor[g.index(i, j)];
                for (int u = 0; u < g.extent(i); ++u)
                    std::copy_n(tile.a[u], g.extent(j), dense.data() + size_t(i * width + u) * g.n + j * width);
            } else if (rank == owner) {
                MPI_Send(&factor[g.index(i, j)], 1, g.tileType, 0, 0, MPI_COMM_WORLD);
            }
        }
    if (rank == 0) print_results(dense, "CholeskyL");
}
void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int size, int rank) {
    int n = 512;
    bool check = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            long value = std::strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > std::numeric_limits<int>::max()) {
                if (rank == 0) printf("Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            check = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            results = true;
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
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", check ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Grid grid(size, rank, n);
    std::vector<Tile> matrix(grid.offsets.back()), factor(matrix.size());
    Panels panels(grid);
    generate(grid, matrix, panels, rank);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = factorize(grid, matrix, factor, panels);
    double elapsed = MPI_Wtime() - start, longest = 0.0;
    MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(longest * 1000));
        printf("Performance: %.3f GFLOPS\n", longest > 0 ? double(n) * n * n / (3e9 * longest) : 0.0);
    }
    if (results) report(grid, factor, rank);
    if (check) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validate(grid, factor, matrix, panels);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = run(argc, argv, size, rank);
    MPI_Finalize();
    return status;
}
