#include <mpi.h>

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

#include "../common/results_output.hpp"

namespace {
constexpr int blockSize = 64;
constexpr int tileElements = blockSize * blockSize;

struct Tile {
    int row, col;
    std::vector<double> data;
    Tile(int i, int j) : row(i), col(j), data(tileElements, 0.0) {}
};

// Only lower-triangular tiles are stored. Tile (i,j) belongs to process
// (i % processRows, j % processCols); communicator ranks follow coordinates.
struct Matrix {
    size_t n;
    int blocks, rank, dims[2] = {0, 0}, row, col;
    int localRows, localCols;
    MPI_Comm rows, cols;
    std::vector<Tile> tiles;
    std::vector<size_t> index;

    explicit Matrix(size_t size) : n(size) {
        int processes;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dims);
        row = rank / dims[1];
        col = rank % dims[1];
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols);
        blocks = static_cast<int>((n + blockSize - 1) / blockSize);
        localRows = blocks > row ? (blocks - 1 - row) / dims[0] + 1 : 0;
        localCols = blocks > col ? (blocks - 1 - col) / dims[1] + 1 : 0;
        index.resize(static_cast<size_t>(localRows) * localCols);
        for (int i = row; i < blocks; i += dims[0]) {
            for (int j = col; j <= i; j += dims[1]) {
                index[static_cast<size_t>(i / dims[0]) * localCols + j / dims[1]] = tiles.size();
                tiles.emplace_back(i, j);
            }
        }
    }
    ~Matrix() {
        MPI_Comm_free(&rows);
        MPI_Comm_free(&cols);
    }
    int extent(int k) const {
        return static_cast<int>(std::min<size_t>(blockSize, n - static_cast<size_t>(k) * blockSize));
    }
    int owner(int i, int j) const { return (i % dims[0]) * dims[1] + j % dims[1]; }
    double* tile(int i, int j) {
        return tiles[index[static_cast<size_t>(i / dims[0]) * localCols + j / dims[1]]].data.data();
    }
};

// C -= X * Y^T. Y is packed transposed so independent output columns
// vectorize without changing the order of a dot product's additions.
void updateTile(double* c, const double* x, const double* yt,
                int m, int n, int depth) {
    for (int i = 0; i < m; ++i) {
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            double sums[8] = {};
            for (int k = 0; k < depth; ++k) {
                const double a = x[i * blockSize + k];
                for (int q = 0; q < 8; ++q)
                    sums[q] += a * yt[k * blockSize + j + q];
            }
            for (int q = 0; q < 8; ++q)
                c[i * blockSize + j + q] -= sums[q];
        }
        for (; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < depth; ++k)
                sum += x[i * blockSize + k] * yt[k * blockSize + j];
            c[i * blockSize + j] -= sum;
        }
    }
}

void transpose(const double* src, double* dst) {
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            dst[j * blockSize + i] = src[i * blockSize + j];
}

// Retain rand_r's exact original stream without replicating the n*n matrix B.
// Row seeds cost O(n) memory; each process regenerates only needed row blocks.
void generatePositiveDefiniteMatrix(Matrix& a) {
    std::vector<unsigned int> seeds(a.n);
    if (a.rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < a.n; ++i) {
            seeds[i] = seed;
            for (size_t j = 0; j < a.n; ++j) (void)rand_r(&seed);
        }
    }
    MPI_Bcast(seeds.data(), static_cast<int>(a.n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<double> bi(static_cast<size_t>(blockSize) * a.n);
    std::vector<double> bj(bi.size());
    auto generateRows = [&](int block, std::vector<double>& b) {
        for (int i = 0; i < a.extent(block); ++i) {
            unsigned int seed = seeds[static_cast<size_t>(block) * blockSize + i];
            for (size_t k = 0; k < a.n; ++k)
                b[static_cast<size_t>(i) * a.n + k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
        }
    };
    int previousRow = -1;
    for (Tile& t : a.tiles) {
        if (previousRow != t.row) {
            generateRows(t.row, bi);
            previousRow = t.row;
        }
        if (t.col != t.row) generateRows(t.col, bj);
        const double* right = t.col == t.row ? bi.data() : bj.data();
        for (int i = 0; i < a.extent(t.row); ++i) {
            for (int j = 0; j < a.extent(t.col); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < a.n; ++k)
                    sum += bi[static_cast<size_t>(i) * a.n + k] * right[static_cast<size_t>(j) * a.n + k];
                if (t.row == t.col && i == j) sum += a.n;
                t.data[i * blockSize + j] = sum;
            }
        }
    }
}

struct Panels {
    std::vector<double> horizontal, vertical;
    std::vector<MPI_Request> requests;
    explicit Panels(const Matrix& a)
        : horizontal(static_cast<size_t>(a.localRows) * tileElements),
          vertical(static_cast<size_t>(a.localCols) * tileElements) {
        requests.reserve(a.localCols);
    }
    const double* left(const Matrix& a, int i) const {
        return horizontal.data() + static_cast<size_t>(i / a.dims[0]) * tileElements;
    }
    const double* right(const Matrix& a, int j) const {
        return vertical.data() + static_cast<size_t>(j / a.dims[1]) * tileElements;
    }
    void exchange(Matrix& a, int k, int first) {
        // One packed broadcast per process row, followed by column broadcasts.
        // Only the active suffix of the panel is communicated.
        const int begin = std::min(a.localRows,
            std::max(0, (first - a.row + a.dims[0] - 1) / a.dims[0]));
        if (a.col == k % a.dims[1]) {
            for (int r = begin; r < a.localRows; ++r)
                std::copy_n(a.tile(a.row + r * a.dims[0], k), tileElements,
                            horizontal.data() + static_cast<size_t>(r) * tileElements);
        }
        if (begin < a.localRows) {
            // Chunking keeps MPI counts valid even for very large matrices.
            size_t offset = static_cast<size_t>(begin) * tileElements;
            while (offset < horizontal.size()) {
                const int count = static_cast<int>(std::min<size_t>(INT_MAX, horizontal.size() - offset));
                MPI_Bcast(horizontal.data() + offset, count, MPI_DOUBLE, k % a.dims[1], a.rows);
                offset += count;
            }
        }
        requests.clear();
        for (int j = a.col; j < a.blocks; j += a.dims[1]) {
            if (j < first) continue;
            double* target = vertical.data() + static_cast<size_t>(j / a.dims[1]) * tileElements;
            if (a.row == j % a.dims[0]) transpose(left(a, j), target);
            requests.emplace_back();
            MPI_Ibcast(target, tileElements, MPI_DOUBLE, j % a.dims[0], a.cols, &requests.back());
        }
        if (!requests.empty())
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
};

bool choleskyDecomposition(Matrix& a, Panels& panels) {
    std::vector<double> diagonal(tileElements);
    for (int k = 0; k < a.blocks; ++k) {
        const int width = a.extent(k);
        const int root = a.owner(k, k);
        if (a.rank == root) {
            double* d = a.tile(k, k);
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (int p = 0; p < j; ++p) sum += d[i * blockSize + p] * d[j * blockSize + p];
                    const double value = d[i * blockSize + j] - sum;
                    if (i == j) {
                        if (!(value > 0.0) || !std::isfinite(value)) {
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                                   static_cast<size_t>(k) * blockSize + i);
                            // A sentinel in the usual broadcast makes all ranks exit together.
                            d[0] = -1.0;
                            goto diagonal_done;
                        }
                        d[i * blockSize + j] = std::sqrt(value);
                    } else {
                        d[i * blockSize + j] = value / d[j * blockSize + j];
                    }
                }
                std::fill(d + i * blockSize + i + 1, d + (i + 1) * blockSize, 0.0);
            }
        diagonal_done:
            std::copy_n(d, tileElements, diagonal.data());
        }
        MPI_Bcast(diagonal.data(), tileElements, MPI_DOUBLE, root, MPI_COMM_WORLD);
        if (diagonal[0] < 0.0) return false;

        // Right triangular solve: L_ik * L_kk^T = A_ik.
        if (a.col == k % a.dims[1]) {
            for (int i = a.row; i < a.blocks; i += a.dims[0]) {
                if (i <= k) continue;
                double* t = a.tile(i, k);
                for (int r = 0; r < a.extent(i); ++r) {
                    for (int j = 0; j < width; ++j) {
                        double sum = 0.0;
                        for (int p = 0; p < j; ++p)
                            sum += t[r * blockSize + p] * diagonal[j * blockSize + p];
                        t[r * blockSize + j] = (t[r * blockSize + j] - sum) / diagonal[j * blockSize + j];
                    }
                }
            }
        }
        panels.exchange(a, k, k + 1);
        for (Tile& t : a.tiles) {
            if (t.col > k)
                updateTile(t.data.data(), panels.left(a, t.row), panels.right(a, t.col),
                           a.extent(t.row), a.extent(t.col), width);
        }
    }
    return true;
}

bool validateCholesky(Matrix& a, Panels& panels, const std::vector<Tile>& original) {
    std::vector<Tile> residual = original;
    // Subtract L*L^T from the saved distributed input. Symmetry means checking
    // the lower triangle is equivalent to checking the complete matrix.
    for (int k = 0; k < a.blocks; ++k) {
        panels.exchange(a, k, k);
        for (Tile& t : residual) {
            if (t.col >= k)
                updateTile(t.data.data(), panels.left(a, t.row), panels.right(a, t.col),
                           a.extent(t.row), a.extent(t.col), a.extent(k));
        }
    }
    double errors[2] = {0.0, 0.0};
    for (size_t t = 0; t < residual.size(); ++t) {
        const Tile& tile = residual[t];
        for (int i = 0; i < a.extent(tile.row); ++i) {
            const int columns = tile.row == tile.col ? i + 1 : a.extent(tile.col);
            for (int j = 0; j < columns; ++j) {
                const int pos = i * blockSize + j;
                const double error = std::fabs(tile.data[pos]);
                const double relative = error / (std::fabs(original[t].data[pos]) + 1e-10);
                errors[0] = std::max(errors[0], std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
                errors[1] = std::max(errors[1], std::isfinite(relative) ? relative : std::numeric_limits<double>::infinity());
            }
        }
    }
    double global[2];
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (a.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

void printResults(Matrix& a) {
    // A full row-major result is needed only for the optional external hash.
    std::vector<double> result;
    if (a.rank == 0) result.resize(a.n * a.n, 0.0);
    std::vector<double> buffer(tileElements);
    for (int i = 0; i < a.blocks; ++i) {
        for (int j = 0; j <= i; ++j) {
            const int owner = a.owner(i, j);
            if (owner != 0 && a.rank == owner)
                MPI_Send(a.tile(i, j), tileElements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (a.rank == 0) {
                const double* data;
                if (owner == 0) data = a.tile(i, j);
                else {
                    MPI_Recv(buffer.data(), tileElements, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    data = buffer.data();
                }
                for (int r = 0; r < a.extent(i); ++r)
                    std::copy_n(data + r * blockSize, a.extent(j),
                                result.data() + (static_cast<size_t>(i) * blockSize + r) * a.n + static_cast<size_t>(j) * blockSize);
            }
        }
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
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end;
            errno = 0;
            const unsigned long long value = std::strtoull(arg, &end, 10);
            // n also bounds MPI row-seed counts and tile-index arithmetic.
            if (*arg == '-' || end == arg || *end || errno || value > INT_MAX ||
                value > std::sqrt(static_cast<long double>(std::numeric_limits<size_t>::max() / sizeof(double)))) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                return 1;
            }
            n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) results = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
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
    std::vector<Tile> original;
    if (validate) original = a.tiles;
    Panels panels(a);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a, panels);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }
    if (results) printResults(a);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(a, panels, original);
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
