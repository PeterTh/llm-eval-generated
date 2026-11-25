#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

constexpr int maxTile = 64;
// Leave room for a padded panel plus its status element in an MPI int count.
constexpr int maxMatrixSize = (std::numeric_limits<int>::max() - 1) /
                              (maxTile * maxTile) * maxTile;

int ceilDiv(int n, int d) {
    return n / d + (n % d > 0);
}

int firstOwned(int start, int coord, int stride) {
    return start + (coord + stride - start % stride) % stride;
}

struct Grid {
    int n;
    int block;
    int tiles;
    int rows;
    int cols;
    int rank;
    int row;
    int col;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm across = MPI_COMM_NULL;
    MPI_Comm down = MPI_COMM_NULL;

    Grid(int size, int worldRank, int worldSize) : n(size), rank(worldRank) {
        int dims[2] = {};
        MPI_Dims_create(worldSize, 2, dims);
        // Use a near-square grid even when the available rank count is prime.
        const int shortSide = static_cast<int>(std::sqrt(worldSize));
        if (dims[0] > 2LL * dims[1] && shortSide > 1) {
            dims[0] = worldSize / shortSide;
            dims[1] = shortSide;
        }
        const int target = ceilDiv(ceilDiv(n, dims[0]), 3);
        block = std::clamp(ceilDiv(target, 8) * 8, 16, maxTile);
        tiles = ceilDiv(n, block);
        rows = std::min(dims[0], std::max(1, tiles));
        cols = std::min(dims[1], std::max(1, tiles));
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row < rows ? 0 : MPI_UNDEFINED, rank, &comm);
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_split(comm, row, col, &across);
            MPI_Comm_split(comm, col, row, &down);
        }
    }

    ~Grid() {
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&across);
            MPI_Comm_free(&down);
            MPI_Comm_free(&comm);
        }
    }

    int extent(int tile) const {
        return std::min(block, n - tile * block);
    }

    int tileSize() const { return block * block; }
};

// Store just the local lower-triangular tiles, in column-major order within a tile.
struct Matrix {
    const Grid& grid;
    std::vector<size_t> offsets;
    std::vector<double> values;

    explicit Matrix(const Grid& g) : grid(g) {
        size_t count = 0;
        for (int j = g.col; j < g.tiles; j += g.cols) {
            offsets.push_back(count);
            count += static_cast<size_t>(ceilDiv(g.tiles - firstRow(j), g.rows)) *
                     g.tileSize();
        }
        values.resize(count);
    }

    int firstRow(int j) const { return firstOwned(j, grid.row, grid.rows); }

    size_t offset(int i, int j) const {
        return offsets[j / grid.cols] +
               static_cast<size_t>((i - firstRow(j)) / grid.rows) * grid.tileSize();
    }

    double* tile(int i, int j) { return values.data() + offset(i, j); }
    const double* tile(int i, int j) const { return values.data() + offset(i, j); }
};

double dot(const double* a, const double* b, int length, double sum = 0.0) {
    for (int k = 0; k < length; ++k) {
        sum += a[k] * b[k];
    }
    return sum;
}

// Jump to a position in the original row-major rand_r stream.
class RandomJump {
#ifdef __GLIBC__
    std::uint32_t multiplier = 1;
    std::uint32_t increment = 0;

public:
    explicit RandomJump(std::uint64_t count) {
        // glibc advances the same 32-bit LCG three times per rand_r call.
        std::uint32_t mul = 1103515245u;
        std::uint32_t add = 12345u;
        for (auto steps = 3 * count; steps != 0; steps >>= 1) {
            if (steps & 1) {
                multiplier *= mul;
                increment = mul * increment + add;
            }
            add *= mul + 1;
            mul *= mul;
        }
    }

    unsigned int operator()(unsigned int seed) const {
        return multiplier * seed + increment;
    }
#else
    std::uint64_t count;

public:
    explicit RandomJump(std::uint64_t calls) : count(calls) {}

    unsigned int operator()(unsigned int seed) const {
        for (std::uint64_t i = 0; i < count; ++i) {
            rand_r(&seed);
        }
        return seed;
    }
#endif
};

std::vector<unsigned int> initialSeeds(const Grid& g) {
    std::vector<unsigned int> local(static_cast<size_t>(ceilDiv(g.tiles, g.rows)) * g.block);
#ifdef __GLIBC__
    const RandomJump nextRow(g.n);
#else
    // Build one checkpoint per input row for C libraries with another RNG.
    std::vector<unsigned int> rowSeeds(g.n);
    if (g.rank == 0) {
        unsigned int seed = 42;
        const RandomJump nextRow(g.n);
        for (auto& rowSeed : rowSeeds) {
            rowSeed = seed;
            seed = nextRow(seed);
        }
    }
    MPI_Bcast(rowSeeds.data(), g.n, MPI_UNSIGNED, 0, g.comm);
    const RandomJump skipColumns(g.col * g.block);
#endif
    for (int i = g.row; i < g.tiles; i += g.rows) {
        const int row = i * g.block;
        auto* seeds = local.data() + (i / g.rows) * g.block;
#ifdef __GLIBC__
        const auto offset = static_cast<std::uint64_t>(row) * g.n + g.col * g.block;
        unsigned int seed = RandomJump(offset)(42);
        for (int r = 0; r < g.extent(i); ++r) {
            seeds[r] = seed;
            seed = nextRow(seed);
        }
#else
        for (int r = 0; r < g.extent(i); ++r) {
            seeds[r] = skipColumns(rowSeeds[row + r]);
        }
#endif
    }
    return local;
}

// The prefix contains dot products from completed tiles. Finish each entry
// using the preceding elements within the current diagonal tile.
int factorDiagonal(double* tile, const double* prefix, double* diag, int b, int width) {
    for (int r = 0; r < width; ++r) {
        for (int c = 0; c < r; ++c) {
            const double sum = dot(diag + r * b, diag + c * b, c, prefix[c * b + r]);
            diag[r * b + c] = tile[c * b + r] = (tile[c * b + r] - sum) / diag[c * b + c];
        }
        const double sum = dot(diag + r * b, diag + r * b, r, prefix[r * b + r]);
        const double val = tile[r * b + r] - sum;
        if (val <= 0.0) {
            return r;
        }
        diag[r * b + r] = tile[r * b + r] = std::sqrt(val);
        std::fill_n(tile + r * b, r, 0.0);
    }
    return -1;
}

void solvePanel(double* tile, const double* prefix, const double* diag,
                int b, int height, int width) {
    for (int r = 0; r < height; ++r) {
        double row[maxTile];
        for (int c = 0; c < width; ++c) {
            const double sum = dot(row, diag + c * b, c, prefix[c * b + r]);
            row[c] = tile[c * b + r] = (tile[c * b + r] - sum) / diag[c * b + c];
        }
    }
}

// Keep a strip of outputs in registers and vectorize across independent rows.
// GCC vectorizes the serial dot's full groups with separate multiplies and
// additions. Its scalar tail is computed by dot() in the diagonal/panel solves.
template <int Rows, int Cols>
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void updateStrip(double* out, const double* left, const double* right, int b, int inner) {
#if defined(__AVX__)
    using Vector = double __attribute__((vector_size(32)));
#elif defined(__GNUC__)
    using Vector = double __attribute__((vector_size(16)));
#else
    using Vector = double;
#endif
    constexpr int vectors = Rows * sizeof(double) / sizeof(Vector);
    Vector sums[Cols][vectors];
    for (int c = 0; c < Cols; ++c) {
        std::memcpy(sums[c], out + c * b, sizeof(sums[c]));
    }
    for (int k = 0; k < inner; ++k) {
        Vector a[vectors];
        std::memcpy(a, left, sizeof(a));
        for (int c = 0; c < Cols; ++c) {
            for (int r = 0; r < vectors; ++r) {
                sums[c][r] += a[r] * right[c];
            }
        }
        left += b;
        right += b;
    }
    for (int c = 0; c < Cols; ++c) {
        std::memcpy(out + c * b, sums[c], sizeof(sums[c]));
    }
}

void updateTile(double* out, const double* left, const double* right, int b, int inner) {
    constexpr int rows = 8;
    constexpr int cols = 4;
    for (int c = 0; c < b; c += cols) {
        for (int r = 0; r < b; r += rows) {
            updateStrip<rows, cols>(out + c * b + r, left + r, right + c, b, inner);
        }
    }
}

void accumulateTile(double* out, const double* left, const double* right,
                    int b, int inner) {
    constexpr int group = 8;
    const int prefix = inner / group * group;
    if (prefix != 0) {
        updateTile(out, left, right, b, prefix);
    }
    if (prefix == inner) {
        return;
    }

    // Match the serial dot's handling of its partial SIMD group.
    const int tail = inner - prefix;
    double leftTail[maxTile][group], rightTail[maxTile][group];
    for (int k = 0; k < tail; ++k) {
        for (int r = 0; r < b; ++r) {
            leftTail[r][k] = left[(prefix + k) * b + r];
            rightTail[r][k] = right[(prefix + k) * b + r];
        }
    }
    for (int c = 0; c < b; ++c) {
        for (int r = 0; r < b; ++r) {
            double& value = out[c * b + r];
            value = dot(leftTail[r], rightTail[c], tail, value);
        }
    }
}

// Redistribute one column panel for a lower-triangular rank-b update.
// Each process gets the tiles indexed by its output rows and columns.
class Panels {
    const Grid& g;
    const int tileSize;
    std::vector<double> rowPanel;
    std::vector<double> colPanel;
    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<int> locations;
    int startRow = 0;
    int startCol = 0;
    const double* columnData = nullptr;

    int rowOffset(int i) const {
        return 1 + (i - startRow) / g.rows * tileSize;
    }

public:
    explicit Panels(const Grid& grid)
        : g(grid), tileSize(g.tileSize()),
          rowPanel(static_cast<size_t>(ceilDiv(g.tiles, g.rows)) * tileSize + 1),
          colPanel(static_cast<size_t>(ceilDiv(g.tiles, g.cols)) * tileSize),
          counts(g.rows), displs(g.rows), locations(ceilDiv(g.tiles, g.cols)) {}

    void begin(int first) {
        startRow = firstOwned(first, g.row, g.rows);
        startCol = firstOwned(first, g.col, g.cols);
    }

    int firstRow() const { return startRow; }

    double* rowTile(int i) {
        return rowPanel.data() + rowOffset(i);
    }

    void setFailure(double failure) { rowPanel[0] = failure; }

    int share(int ownerCol) {
        const int rowCount = ceilDiv(g.tiles - startRow, g.rows);
        MPI_Bcast(rowPanel.data(), 1 + rowCount * tileSize, MPI_DOUBLE, ownerCol, g.across);
        if (rowPanel[0] != 0) {
            return static_cast<int>(rowPanel[0]) - 1;
        }
        if (startCol >= g.tiles) {
            return -1;
        }

        if (g.rows == g.cols) {
            // The diagonal process already has the entire column's panel
            // in order from its process row broadcast.
            double* buffer = g.row == g.col ? rowPanel.data() + 1 : colPanel.data();
            MPI_Bcast(buffer, ceilDiv(g.tiles - startCol, g.cols) * tileSize,
                      MPI_DOUBLE, g.col, g.down);
            columnData = buffer;
        } else {
            // Pack by owning process row for one collective per process column.
            std::fill(counts.begin(), counts.end(), 0);
            for (int j = startCol; j < g.tiles; j += g.cols) {
                counts[j % g.rows] += tileSize;
            }
            int total = 0;
            for (int r = 0; r < g.rows; ++r) {
                displs[r] = total;
                total += counts[r];
                counts[r] = 0;
            }
            for (int j = startCol; j < g.tiles; j += g.cols) {
                const int r = j % g.rows;
                const int pos = displs[r] + counts[r];
                counts[r] += tileSize;
                locations[j / g.cols] = pos;
                if (r == g.row) {
                    std::copy_n(rowTile(j), tileSize, colPanel.data() + pos);
                }
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, colPanel.data(), counts.data(),
                           displs.data(), MPI_DOUBLE, g.down);
            columnData = colPanel.data();
        }
        return -1;
    }

    void accumulate(const Matrix& layout, std::vector<double>& sums, int width) const {
        for (int j = startCol; j < g.tiles; j += g.cols) {
            const int offset = g.rows == g.cols ? (j - startCol) / g.cols * tileSize
                                               : locations[j / g.cols];
            const double* right = columnData + offset;
            for (int i = layout.firstRow(j); i < g.tiles; i += g.rows) {
                const double* left = rowPanel.data() + rowOffset(i);
                accumulateTile(sums.data() + layout.offset(i, j), left, right, g.block, width);
            }
        }
    }
};

void generateMatrix(Matrix& a) {
    const Grid& g = a.grid;
    const int b = g.block;
    auto seeds = initialSeeds(g);
    const RandomJump nextTile((g.cols - 1) * b);
    Panels panels(g);
    panels.begin(0);

    // Stream B one block of columns at a time while forming A = B * B^T.
    for (int s = 0; s < g.tiles; ++s) {
        const int ownerCol = s % g.cols;
        const int width = g.extent(s);
        if (g.col == ownerCol) {
            for (int i = panels.firstRow(); i < g.tiles; i += g.rows) {
                double* tile = panels.rowTile(i);
                auto* rowSeeds = seeds.data() + i / g.rows * b;
                const int height = g.extent(i);
                for (int r = 0; r < height; ++r) {
                    auto seed = rowSeeds[r];
                    for (int k = 0; k < width; ++k) {
                        tile[k * b + r] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
                    }
                    if (s + g.cols < g.tiles) {
                        rowSeeds[r] = nextTile(seed);
                    }
                }
                for (int k = 0; k < width; ++k) {
                    std::fill_n(tile + k * b + height, b - height, 0.0);
                }
            }
        }
        panels.share(ownerCol);
        panels.accumulate(a, a.values, width);
    }

    for (int i = g.col; i < g.tiles; i += g.cols) {
        if (i % g.rows == g.row) {
            double* tile = a.tile(i, i);
            for (int c = 0; c < g.extent(i); ++c) {
                tile[c * b + c] += g.n;
            }
        }
    }
}

int factor(Matrix& a) {
    const Grid& g = a.grid;
    const int b = g.block;
    const int tileSize = g.tileSize();
    std::vector<double> prefix(a.values.size());
    // The first element of each broadcast carries the failure index, or zero.
    std::vector<double> diag(tileSize + 1);
    Panels panels(g);

    for (int s = 0; s < g.tiles; ++s) {
        const int ownerRow = s % g.rows;
        const int ownerCol = s % g.cols;
        const int width = g.extent(s);
        panels.begin(s + 1);

        if (g.col == ownerCol) {
            if (g.row == ownerRow) {
                const int bad = factorDiagonal(a.tile(s, s), prefix.data() + a.offset(s, s),
                                               diag.data() + 1, b, width);
                diag[0] = bad < 0 ? 0 : s * b + bad + 1;
            }
            MPI_Bcast(diag.data(), tileSize + 1, MPI_DOUBLE, ownerRow, g.down);
            panels.setFailure(diag[0]);
            if (diag[0] == 0) {
                for (int i = panels.firstRow(); i < g.tiles; i += g.rows) {
                    double* tile = a.tile(i, s);
                    solvePanel(tile, prefix.data() + a.offset(i, s), diag.data() + 1,
                               b, g.extent(i), width);
                    std::copy_n(tile, tileSize, panels.rowTile(i));
                }
            }
        }
        const int bad = panels.share(ownerCol);
        if (bad >= 0) {
            return bad;
        }
        panels.accumulate(a, prefix, width);
    }
    return -1;
}

bool validateFactor(const Matrix& a, const std::vector<double>& original) {
    const Grid& g = a.grid;
    const int b = g.block;
    std::vector<double> reconstructed(a.values.size());
    Panels panels(g);
    for (int s = 0; s < g.tiles; ++s) {
        const int ownerCol = s % g.cols;
        panels.begin(s);
        if (g.col == ownerCol) {
            for (int i = panels.firstRow(); i < g.tiles; i += g.rows) {
                std::copy_n(a.tile(i, s), g.tileSize(), panels.rowTile(i));
            }
        }
        panels.share(ownerCol);
        panels.accumulate(a, reconstructed, g.extent(s));
    }

    std::array<double, 2> local{};
    for (int j = g.col; j < g.tiles; j += g.cols) {
        for (int i = a.firstRow(j); i < g.tiles; i += g.rows) {
            const size_t offset = a.offset(i, j);
            for (int c = 0; c < g.extent(j); ++c) {
                for (int r = (i == j ? c : 0); r < g.extent(i); ++r) {
                    const size_t index = offset + c * b + r;
                    const double expected = original[index];
                    const double error = std::abs(reconstructed[index] - expected);
                    local[0] = std::max(local[0], error);
                    local[1] = std::max(local[1], error / (std::abs(expected) + 1e-10));
                }
            }
        }
    }

    std::array<double, 2> errors{};
    MPI_Allreduce(local.data(), errors.data(), errors.size(), MPI_DOUBLE, MPI_MAX, g.comm);
    const bool valid = errors[1] <= 1e-6;
    if (g.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (!valid) {
            printf("Validation failed: relative error too large\n");
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

void reportFactor(const Matrix& a) {
    const Grid& g = a.grid;
    std::vector<double> dense;
    if (g.rank == 0) {
        dense.resize(static_cast<size_t>(g.n) * g.n);
    }

    // Gather a tile row at a time to keep MPI counts and staging storage small.
    std::vector<double> buffer(static_cast<size_t>(ceilDiv(g.tiles, g.cols)) * g.tileSize());
    for (int i = 0; i < g.tiles; ++i) {
        if (g.row == i % g.rows) {
            int offset = 0;
            for (int j = g.col; j <= i; j += g.cols) {
                std::copy_n(a.tile(i, j), g.tileSize(), buffer.data() + offset);
                offset += g.tileSize();
            }
            if (g.rank != 0 && offset != 0) {
                MPI_Send(buffer.data(), offset, MPI_DOUBLE, 0, 0, g.comm);
            }
        }
        if (g.rank == 0) {
            for (int c = 0; c < std::min(g.cols, i + 1); ++c) {
                const int source = (i % g.rows) * g.cols + c;
                if (source != 0) {
                    MPI_Recv(buffer.data(), ceilDiv(i + 1 - c, g.cols) * g.tileSize(),
                             MPI_DOUBLE, source, 0, g.comm, MPI_STATUS_IGNORE);
                }
                int offset = 0;
                for (int j = c; j <= i; j += g.cols) {
                    for (int r = 0; r < g.extent(i); ++r) {
                        double* dest = dense.data() +
                                       static_cast<size_t>(i * g.block + r) * g.n + j * g.block;
                        for (int k = 0; k < (i == j ? r + 1 : g.extent(j)); ++k) {
                            dest[k] = buffer[offset + k * g.block + r];
                        }
                    }
                    offset += g.tileSize();
                }
            }
        }
    }

    if (g.rank == 0) {
        print_results(dense, "CholeskyL");
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

int benchmark(const Grid& grid, bool validate, bool printResults) {
    const int n = grid.n;
    const bool root = grid.rank == 0;
    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix a(grid);
    generateMatrix(a);
    const std::vector<double> original = validate ? a.values : std::vector<double>{};

    if (root) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(grid.comm);
    const double start = MPI_Wtime();
    const int bad = factor(a);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);

    if (bad >= 0) {
        if (root) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", bad);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }
    if (root) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? ops / duration / 1e9 : 0.0);
    }
    if (printResults) {
        reportFactor(a);
    }
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        return validateFactor(a, original) ? 0 : 1;
    }
    return 0;
}

int run(int argc, char** argv, int rank, int size) {
    // Parse once so every process takes the same exit or collective path.
    constexpr int runBenchmark = -1;
    int options[] = {512, 0, 0, runBenchmark};
    auto& [n, validate, printResults, status] = options;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const long parsed = std::strtol(argv[++i], nullptr, 10);
                if (parsed < 0 || parsed > maxMatrixSize) {
                    printf("Invalid matrix size: %s\n", argv[i]);
                    status = 1;
                    break;
                }
                n = static_cast<int>(parsed);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                status = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                status = 1;
                break;
            }
        }
    }
    MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != runBenchmark) {
        return status;
    }

    Grid grid(n, rank, size);
    int result = 0;
    if (grid.comm != MPI_COMM_NULL) {
        try {
            result = benchmark(grid, validate, printResults);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int result = run(argc, argv, rank, size);
    MPI_Finalize();
    return result;
}
