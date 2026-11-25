#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

struct Grid {
    int rank;
    int rows;
    int cols;
    int row;
    int col;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    Grid() {
        int size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }

    ~Grid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }

    int owner(int i, int j) const {
        return (i % rows) * cols + j % cols;
    }
};

int tilesOnRank(int tiles, int coordinate, int stride) {
    return tiles / stride + (coordinate < tiles % stride);
}

int tileWidth(int n, const Grid& grid) {
    int width = 64;
    // Keep several tile rows and columns per process when the matrix is small.
    while (width > 8 && n / width < 3 * std::max(grid.rows, grid.cols)) {
        width /= 2;
    }
    return width;
}

struct Matrix {
    const Grid& grid;
    const int n;
    const int width;
    const int area;
    const int tiles;
    const int localRows;
    const int localCols;
    std::vector<std::vector<double>> data;

    Matrix(const Grid& grid, int n, int width, bool lower)
        : grid(grid), n(n), width(width), area(width * width),
          tiles(n / width + (n % width != 0)),
          localRows(tilesOnRank(tiles, grid.row, grid.rows)),
          localCols(tilesOnRank(tiles, grid.col, grid.cols)),
          data(static_cast<size_t>(localRows) * localCols) {
        for (int i = grid.row; i < tiles; i += grid.rows) {
            for (int j = grid.col; j < tiles && (!lower || j <= i); j += grid.cols) {
                tile(i, j).resize(area);
            }
        }
    }

    int extent(int i) const {
        return std::min(width, n - i * width);
    }

    std::vector<double>& tile(int i, int j) {
        return data[static_cast<size_t>(i / grid.rows) * localCols + j / grid.cols];
    }

    const std::vector<double>& tile(int i, int j) const {
        return data[static_cast<size_t>(i / grid.rows) * localCols + j / grid.cols];
    }
};

// Exchange the tiles X(i,k) needed for local products X(i,k)*X(j,k)^T.
// After the row broadcast, one copy of each tile is sent down the column
// that owns output tile column i.
class Panels {
    const Grid& grid;
    const int width;
    const int area;
    std::vector<double> rowPanel;
    std::vector<double> colPanel;
    std::vector<int> slot;
    std::vector<std::vector<int>> columnTiles;
    std::vector<int> offsets;
    std::vector<int> counts;

    int firstLocalRow(int first) const {
        return tilesOnRank(first, grid.row, grid.rows);
    }

public:
    explicit Panels(const Matrix& matrix)
        : grid(matrix.grid), width(matrix.width), area(matrix.area),
          rowPanel(static_cast<size_t>(matrix.localRows) * area + 1),
          colPanel(static_cast<size_t>(matrix.localCols) * area),
          slot(matrix.tiles), columnTiles(grid.rows), offsets(grid.rows), counts(grid.rows) {
        for (int j = grid.col; j < matrix.tiles; j += grid.cols) {
            columnTiles[j % grid.rows].push_back(j);
        }
        int pos = 0;
        for (const auto& indices : columnTiles) {
            for (int j : indices) {
                slot[j] = pos++;
            }
        }
    }

    const double* rowTile(int i) const {
        return rowPanel.data() + 1 + static_cast<size_t>(i / grid.rows) * area;
    }

    double* colTile(int j) {
        return colPanel.data() + static_cast<size_t>(slot[j]) * area;
    }

    int broadcastRows(const Matrix& matrix, int k, int first, int failure = 0) {
        // Put the status just before the first tile in the active suffix.
        const int skip = firstLocalRow(first);
        double* buffer = rowPanel.data() + static_cast<size_t>(skip) * area;
        const int root = k % grid.cols;
        if (grid.col == root) {
            buffer[0] = failure;
            if (failure == 0) {
                for (int r = skip; r < matrix.localRows; ++r) {
                    const int i = grid.row + r * grid.rows;
                    const auto& tile = matrix.tile(i, k);
                    std::copy_n(tile.data(), area,
                                buffer + 1 + static_cast<size_t>(r - skip) * area);
                }
            }
        }
        MPI_Bcast(buffer, 1 + (matrix.localRows - skip) * area, MPI_DOUBLE,
                  root, grid.rowComm);
        return static_cast<int>(buffer[0]);
    }

    void exchangeColumns(int first, bool transpose) {
        int base = 0;
        for (int r = 0; r < grid.rows; ++r) {
            const auto& indices = columnTiles[r];
            const auto begin = std::lower_bound(indices.begin(), indices.end(), first);
            offsets[r] = base + static_cast<int>(begin - indices.begin()) * area;
            counts[r] = static_cast<int>(indices.end() - begin) * area;
            base += static_cast<int>(indices.size()) * area;
            if (r == grid.row) {
                for (auto it = begin; it != indices.end(); ++it) {
                    const double* source = rowTile(*it);
                    double* target = colTile(*it);
                    if (transpose) {
                        for (int i = 0; i < width; ++i) {
                            for (int j = 0; j < width; ++j) {
                                target[j * width + i] = source[i * width + j];
                            }
                        }
                    } else {
                        std::copy_n(source, area, target);
                    }
                }
            }
        }
        if (!colPanel.empty()) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, colPanel.data(), counts.data(),
                           offsets.data(), MPI_DOUBLE, grid.colComm);
        }
    }
};

double dot(const double* a, const double* b, int count, double sum) {
    for (int k = 0; k < count; ++k) {
        sum += a[k] * b[k];
    }
    return sum;
}

// Vectorize across output columns, preserving the order of products and
// additions along k. Full panels end on a dot-product vector boundary.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
void accumulateTile(double* sums, const double* left, const double* rightT,
                    int rows, int cols, int width) {
    constexpr int rowBlock = 2;
    constexpr int colBlock = 8;
    for (int i = 0; i < rows; i += rowBlock) {
        for (int j = 0; j < cols; j += colBlock) {
            double* out0 = sums + i * width + j;
            double* out1 = out0 + width;
            double acc0[colBlock];
            double acc1[colBlock];
            std::copy_n(out0, colBlock, acc0);
            std::copy_n(out1, colBlock, acc1);
            for (int k = 0; k < width; ++k) {
                const double a = left[i * width + k];
                const double b = left[(i + 1) * width + k];
                const double* rhs = rightT + k * width + j;
                for (int c = 0; c < colBlock; ++c) {
                    acc0[c] += a * rhs[c];
                    acc1[c] += b * rhs[c];
                }
            }
            std::copy_n(acc0, colBlock, out0);
            std::copy_n(acc1, colBlock, out1);
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

void updateProducts(Matrix& sums, Panels& panels, int k, int first) {
    const auto& grid = sums.grid;
    const int width = sums.width;
    const int depth = sums.extent(k);
    const int firstCol = tilesOnRank(first, grid.col, grid.cols) * grid.cols + grid.col;
    panels.exchangeColumns(first, depth == width);

    for (int j = firstCol; j < sums.tiles; j += grid.cols) {
        const double* right = panels.colTile(j);
        const int minRow = tilesOnRank(j, grid.row, grid.rows) * grid.rows + grid.row;
        for (int i = minRow; i < sums.tiles; i += grid.rows) {
            double* tile = sums.tile(i, j).data();
            const double* left = panels.rowTile(i);
            const int rows = sums.extent(i);
            const int cols = sums.extent(j);
            if (depth == width) {
                accumulateTile(tile, left, right, rows, cols, width);
            } else {
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        double& sum = tile[r * width + c];
                        sum = dot(left + r * width, right + c * width, depth, sum);
                    }
                }
            }
        }
    }
}

#if defined(__GLIBC__)
struct SeedAdvance {
    unsigned int multiplier;
    unsigned int increment;

    unsigned int operator()(unsigned int seed) const {
        return multiplier * seed + increment;
    }
};

SeedAdvance advanceSeed(uint64_t count) {
    // glibc's rand_r applies this 32-bit recurrence three times per sample.
    SeedAdvance step{1103515245u, 12345u};
    SeedAdvance advance{1u, 0u};
    for (uint64_t steps = count * 3; steps != 0; steps >>= 1) {
        if (steps & 1) {
            advance = {step.multiplier * advance.multiplier, step(advance.increment)};
        }
        step = {step.multiplier * step.multiplier, step(step.increment)};
    }
    return advance;
}
#endif

void generateRandomTiles(Matrix& random) {
    const auto& grid = random.grid;
#if defined(__GLIBC__)
    const auto nextRow = advanceSeed(random.n);
    for (int i = grid.row; i < random.tiles; i += grid.rows) {
        for (int j = grid.col; j < random.tiles; j += grid.cols) {
            const uint64_t first = static_cast<uint64_t>(i) * random.width * random.n
                                   + j * random.width;
            unsigned int rowSeed = advanceSeed(first)(42);
            auto& tile = random.tile(i, j);
            for (int r = 0; r < random.extent(i); ++r) {
                unsigned int seed = rowSeed;
                rowSeed = nextRow(rowSeed);
                for (int c = 0; c < random.extent(j); ++c) {
                    tile[r * random.width + c] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
                }
            }
        }
    }
#else
    // Other libraries may use a different rand_r sequence.
    if (grid.rank == 0) {
        unsigned int seed = 42;
        std::vector<double> stripe(static_cast<size_t>(random.tiles) * random.area);
        for (int i = 0; i < random.tiles; ++i) {
            // Generate in the original row-major order, buffering one tile row.
            for (int r = 0; r < random.extent(i); ++r) {
                for (int j = 0; j < random.tiles; ++j) {
                    double* row = stripe.data() + static_cast<size_t>(j) * random.area
                                  + r * random.width;
                    for (int c = 0; c < random.extent(j); ++c) {
                        row[c] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
                    }
                }
            }
            for (int j = 0; j < random.tiles; ++j) {
                const int owner = grid.owner(i, j);
                const double* tile = stripe.data() + static_cast<size_t>(j) * random.area;
                if (owner == 0) {
                    std::copy_n(tile, random.area, random.tile(i, j).data());
                } else {
                    MPI_Send(tile, random.area, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        for (int i = grid.row; i < random.tiles; i += grid.rows) {
            for (int j = grid.col; j < random.tiles; j += grid.cols) {
                MPI_Recv(random.tile(i, j).data(), random.area, MPI_DOUBLE,
                         0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
#endif
}

void generatePositiveDefiniteMatrix(Matrix& original) {
    Matrix random(original.grid, original.n, original.width, false);
    generateRandomTiles(random);
    Panels panels(random);
    for (int k = 0; k < random.tiles; ++k) {
        panels.broadcastRows(random, k, 0);
        updateProducts(original, panels, k, 0);
    }
    const auto& grid = original.grid;
    for (int i = grid.row; i < original.tiles; i += grid.rows) {
        if (i % grid.cols == grid.col) {
            auto& tile = original.tile(i, i);
            for (int r = 0; r < original.extent(i); ++r) {
                tile[r * original.width + r] += original.n;
            }
        }
    }
}

int factorDiagonal(const Matrix& original, Matrix& factor, int k) {
    const int width = factor.width;
    const int depth = factor.extent(k);
    const auto& input = original.tile(k, k);
    auto& tile = factor.tile(k, k);
    for (int i = 0; i < depth; ++i) {
        double* row = tile.data() + i * width;
        for (int j = 0; j < i; ++j) {
            const double* prev = tile.data() + j * width;
            const double sum = dot(row, prev, j, row[j]);
            row[j] = (input[i * width + j] - sum) / prev[j];
        }
        const double sum = dot(row, row, i, row[i]);
        const double diagonal = input[i * width + i] - sum;
        if (diagonal <= 0.0) {
            return k * width + i + 1;
        }
        row[i] = std::sqrt(diagonal);
        std::fill(row + i + 1, row + width, 0.0);
    }
    return 0;
}

void solvePanel(const Matrix& original, Matrix& factor, int k, const double* diagonal) {
    const auto& grid = factor.grid;
    const int width = factor.width;
    const int depth = factor.extent(k);
    const int firstRow = tilesOnRank(k + 1, grid.row, grid.rows) * grid.rows + grid.row;
    for (int i = firstRow; i < factor.tiles; i += grid.rows) {
        const auto& input = original.tile(i, k);
        auto& tile = factor.tile(i, k);
        for (int r = 0; r < factor.extent(i); ++r) {
            double* row = tile.data() + r * width;
            for (int j = 0; j < depth; ++j) {
                const double* prev = diagonal + j * width;
                const double sum = dot(row, prev, j, row[j]);
                row[j] = (input[r * width + j] - sum) / prev[j];
            }
        }
    }
}

int choleskyDecomposition(const Matrix& original, Matrix& factor) {
    const auto& grid = factor.grid;
    Panels panels(factor);
    std::vector<double> diagonal(factor.area + 1);
    // Pending tiles hold sums of products from prior panels. Subtract each
    // completed sum from the input once, as in the scalar recurrence.
    for (int k = 0; k < factor.tiles; ++k) {
        int failure = 0;
        if (grid.col == k % grid.cols) {
            if (grid.row == k % grid.rows) {
                diagonal[0] = factorDiagonal(original, factor, k);
                const auto& tile = factor.tile(k, k);
                std::copy_n(tile.data(), factor.area, diagonal.data() + 1);
            }
            MPI_Bcast(diagonal.data(), factor.area + 1, MPI_DOUBLE, k % grid.rows,
                      grid.colComm);
            failure = static_cast<int>(diagonal[0]);
            if (failure == 0) {
                solvePanel(original, factor, k, diagonal.data() + 1);
            }
        }
        failure = panels.broadcastRows(factor, k, k + 1, failure);
        if (failure != 0) {
            return failure;
        }
        if (k + 1 < factor.tiles) {
            updateProducts(factor, panels, k, k + 1);
        }
    }
    return 0;
}

bool validateCholesky(const Matrix& factor, const Matrix& original) {
    const auto& grid = factor.grid;
    Matrix reconstructed(grid, factor.n, factor.width, true);
    Panels panels(factor);
    for (int k = 0; k < factor.tiles; ++k) {
        panels.broadcastRows(factor, k, k);
        updateProducts(reconstructed, panels, k, k);
    }
    double localErrors[2] = {};
    for (int i = grid.row; i < factor.tiles; i += grid.rows) {
        for (int j = grid.col; j <= i; j += grid.cols) {
            const auto& input = original.tile(i, j);
            const auto& product = reconstructed.tile(i, j);
            for (int r = 0; r < factor.extent(i); ++r) {
                for (int c = 0; c < factor.extent(j); ++c) {
                    const int idx = r * factor.width + c;
                    const double error = std::abs(product[idx] - input[idx]);
                    localErrors[0] = std::max(localErrors[0], error);
                    const double relative = error / (std::abs(input[idx]) + 1e-10);
                    localErrors[1] = std::max(localErrors[1], relative);
                }
            }
        }
    }
    double errors[2];
    MPI_Allreduce(localErrors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (grid.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) {
            printf("Validation failed: relative error too large\n");
        }
    }
    return errors[1] <= 1e-6;
}

void printFactor(const Matrix& factor) {
    const auto& grid = factor.grid;
    if (grid.rank == 0) {
        std::vector<double> result(static_cast<size_t>(factor.n) * factor.n);
        std::vector<double> received(factor.area);
        for (int i = 0; i < factor.tiles; ++i) {
            for (int j = 0; j <= i; ++j) {
                const double* tile = received.data();
                const int owner = grid.owner(i, j);
                if (owner == 0) {
                    tile = factor.tile(i, j).data();
                } else {
                    MPI_Recv(received.data(), factor.area, MPI_DOUBLE, owner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                for (int r = 0; r < factor.extent(i); ++r) {
                    double* row = result.data()
                        + (static_cast<size_t>(i) * factor.width + r) * factor.n;
                    std::copy_n(tile + r * factor.width, factor.extent(j),
                                row + j * factor.width);
                }
            }
        }
        print_results(result, "CholeskyL");
    } else {
        for (int i = grid.row; i < factor.tiles; i += grid.rows) {
            for (int j = grid.col; j <= i; j += grid.cols) {
                MPI_Send(factor.tile(i, j).data(), factor.area, MPI_DOUBLE, 0, 0,
                         MPI_COMM_WORLD);
            }
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, const Grid& grid) {
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (grid.rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (grid.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (n < 0) {
        if (grid.rank == 0) {
            printf("Matrix size must be nonnegative\n");
        }
        return 1;
    }

    if (grid.rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    const int width = tileWidth(n, grid);
    Matrix original(grid, n, width, true);
    generatePositiveDefiniteMatrix(original);
    Matrix factor(grid, n, width, true);

    if (grid.rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int failure = choleskyDecomposition(original, factor);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (failure != 0) {
        if (grid.rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failure - 1);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }
    if (grid.rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / (duration * 1e9));
    }
    if (printResults) {
        printFactor(factor);
    }
    if (validate) {
        if (grid.rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateCholesky(factor, original);
        if (grid.rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status;
    {
        const Grid grid;
        status = run(argc, argv, grid);
    }
    MPI_Finalize();
    return status;
}
