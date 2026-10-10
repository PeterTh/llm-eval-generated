#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

namespace {

// All matrices use square, row-major tiles on a 2D block-cyclic process grid.
// A rank stores O(n^2 / ranks) matrix entries and O(n * block / sqrt(ranks))
// panel entries on a balanced grid; complete matrices are never replicated.
struct ProcessGrid {
    int rank = 0, size = 0;
    int rows = 0, cols = 0, row = 0, col = 0;
    size_t block = 64, tiles = 0, localRows = 0, localCols = 0;
    MPI_Comm rowComm = MPI_COMM_NULL, colComm = MPI_COMM_NULL;

    explicit ProcessGrid(size_t n) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        // Smaller tiles expose enough work for small matrices and large grids.
        while (block > 8 && n / block < 2 * static_cast<size_t>(std::max(rows, cols)))
            block /= 2;
        tiles = n / block + (n % block != 0);
        localRows = count(tiles, row, rows);
        localCols = count(tiles, col, cols);
        // Allgatherv uses int counts/displacements for panel buffers.
        const size_t maxPanelTiles = std::max(count(tiles, 0, rows),
                                             count(tiles, 0, cols));
        if (maxPanelTiles > static_cast<size_t>(std::numeric_limits<int>::max()) / (block * block))
            throw std::length_error("Matrix panels exceed MPI count limits");
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }

    static size_t count(size_t total, int coordinate, int stride) {
        const size_t start = static_cast<size_t>(coordinate);
        return total > start ? 1 + (total - 1 - start) / static_cast<size_t>(stride) : 0;
    }

    size_t firstLocalRow(size_t first) const {
        return first > static_cast<size_t>(row)
            ? std::min(localRows, (first - row + rows - 1) / rows) : 0;
    }

    ~ProcessGrid() {
        // The outer exception handler aborts all ranks; avoid collective
        // cleanup while peers may still be waiting in matrix operations.
        if (std::uncaught_exceptions() != 0) return;
        if (rowComm != MPI_COMM_NULL) MPI_Comm_free(&rowComm);
        if (colComm != MPI_COMM_NULL) MPI_Comm_free(&colComm);
    }
};

struct Matrix {
    const ProcessGrid& grid;
    std::vector<double> data;

    explicit Matrix(const ProcessGrid& g) : grid(g) {
        const size_t area = g.block * g.block;
        if (g.localRows && g.localCols > data.max_size() / area / g.localRows)
            throw std::length_error("Matrix is too large");
        data.resize(g.localRows * g.localCols * area, 0.0);
    }

    double* local(size_t i, size_t j) {
        return data.data() + (i * grid.localCols + j) * grid.block * grid.block;
    }

    const double* local(size_t i, size_t j) const {
        return data.data() + (i * grid.localCols + j) * grid.block * grid.block;
    }

    double* tile(size_t i, size_t j) {
        return local(i / grid.rows, j / grid.cols);
    }
};

#if defined(__AVX2__)
inline __m256d accumulateProduct(__m256d a, __m256d b, __m256d sum) {
#if defined(__FMA__)
    return _mm256_fmadd_pd(a, b, sum);
#else
    return _mm256_add_pd(_mm256_mul_pd(a, b), sum);
#endif
}

template<bool Subtract>
inline void storeSum(double* target, __m256d sum) {
    const __m256d previous = _mm256_loadu_pd(target);
    _mm256_storeu_pd(target, Subtract ? _mm256_sub_pd(previous, sum)
                                    : _mm256_add_pd(previous, sum));
}
#endif

// Transposed RHS panels are reused for every local tile. The 4 x 8
// register block avoids horizontal reductions and keeps independent SIMD
// accumulators; other architectures use the same portable kernel below.
template<bool Subtract>
void multiplyTile(double* target, const double* left,
                  const double* rightTransposed, size_t b) {
    for (size_t i = 0; i < b; i += 4) {
        for (size_t j = 0; j < b; j += 8) {
#if defined(__AVX2__)
            __m256d s00 = _mm256_setzero_pd(), s01 = _mm256_setzero_pd();
            __m256d s10 = _mm256_setzero_pd(), s11 = _mm256_setzero_pd();
            __m256d s20 = _mm256_setzero_pd(), s21 = _mm256_setzero_pd();
            __m256d s30 = _mm256_setzero_pd(), s31 = _mm256_setzero_pd();
            for (size_t k = 0; k < b; ++k) {
                const double* rhs = rightTransposed + k * b + j;
                const __m256d r0 = _mm256_loadu_pd(rhs);
                const __m256d r1 = _mm256_loadu_pd(rhs + 4);
                const __m256d a0 = _mm256_broadcast_sd(left + i * b + k);
                const __m256d a1 = _mm256_broadcast_sd(left + (i + 1) * b + k);
                const __m256d a2 = _mm256_broadcast_sd(left + (i + 2) * b + k);
                const __m256d a3 = _mm256_broadcast_sd(left + (i + 3) * b + k);
                s00 = accumulateProduct(a0, r0, s00);
                s01 = accumulateProduct(a0, r1, s01);
                s10 = accumulateProduct(a1, r0, s10);
                s11 = accumulateProduct(a1, r1, s11);
                s20 = accumulateProduct(a2, r0, s20);
                s21 = accumulateProduct(a2, r1, s21);
                s30 = accumulateProduct(a3, r0, s30);
                s31 = accumulateProduct(a3, r1, s31);
            }
            storeSum<Subtract>(target + i * b + j, s00);
            storeSum<Subtract>(target + i * b + j + 4, s01);
            storeSum<Subtract>(target + (i + 1) * b + j, s10);
            storeSum<Subtract>(target + (i + 1) * b + j + 4, s11);
            storeSum<Subtract>(target + (i + 2) * b + j, s20);
            storeSum<Subtract>(target + (i + 2) * b + j + 4, s21);
            storeSum<Subtract>(target + (i + 3) * b + j, s30);
            storeSum<Subtract>(target + (i + 3) * b + j + 4, s31);
#else
            double s0[8] = {}, s1[8] = {}, s2[8] = {}, s3[8] = {};
            for (size_t k = 0; k < b; ++k) {
                const double a0 = left[i * b + k];
                const double a1 = left[(i + 1) * b + k];
                const double a2 = left[(i + 2) * b + k];
                const double a3 = left[(i + 3) * b + k];
                const double* rhs = rightTransposed + k * b + j;
                for (size_t v = 0; v < 8; ++v) {
                    s0[v] += a0 * rhs[v];
                    s1[v] += a1 * rhs[v];
                    s2[v] += a2 * rhs[v];
                    s3[v] += a3 * rhs[v];
                }
            }
            for (size_t v = 0; v < 8; ++v) {
                if constexpr (Subtract) {
                    target[i * b + j + v] -= s0[v];
                    target[(i + 1) * b + j + v] -= s1[v];
                    target[(i + 2) * b + j + v] -= s2[v];
                    target[(i + 3) * b + j + v] -= s3[v];
                } else {
                    target[i * b + j + v] += s0[v];
                    target[(i + 1) * b + j + v] += s1[v];
                    target[(i + 2) * b + j + v] += s2[v];
                    target[(i + 3) * b + j + v] += s3[v];
                }
            }
#endif
        }
    }
}

struct Panels {
    const ProcessGrid& grid;
    std::vector<double> left, right, transposed;
    std::vector<size_t> slots;
    std::vector<int> counts, displacements;

    explicit Panels(const ProcessGrid& g)
        : grid(g), left(g.localRows * g.block * g.block),
          right(g.localCols * g.block * g.block), transposed(right.size()),
          slots(g.localCols), counts(g.rows), displacements(g.rows) {
        // Group RHS tiles by their source process row for a column allgather.
        std::vector<size_t> next(g.rows, 0);
        for (size_t j = 0; j < g.localCols; ++j)
            ++next[(g.col + j * g.cols) % g.rows];
        size_t offset = 0;
        for (int r = 0; r < g.rows; ++r) {
            const size_t length = next[r];
            next[r] = offset;
            offset += length;
        }
        for (size_t j = 0; j < g.localCols; ++j)
            slots[j] = next[(g.col + j * g.cols) % g.rows]++;
    }

    void collect(const Matrix& matrix, size_t k, size_t first) {
        const auto& g = grid;
        const size_t area = g.block * g.block;
        const size_t begin = g.firstLocalRow(first);
        const size_t length = (g.localRows - begin) * area;
        if (g.col == static_cast<int>(k % g.cols)) {
            for (size_t i = begin; i < g.localRows; ++i)
                std::copy_n(matrix.local(i, k / g.cols), area, left.data() + i * area);
        }
        // Roots rotate with the block-cyclic panel ownership.
        if (length)
            MPI_Bcast(left.data() + begin * area, static_cast<int>(length),
                      MPI_DOUBLE, static_cast<int>(k % g.cols), g.rowComm);

        std::fill(counts.begin(), counts.end(), 0);
        std::fill(displacements.begin(), displacements.end(), 0);
        for (size_t j = 0; j < g.localCols; ++j) {
            const size_t global = g.col + j * g.cols;
            if (global < first) continue;
            const int owner = static_cast<int>(global % g.rows);
            if (!counts[owner])
                displacements[owner] = static_cast<int>(slots[j] * area);
            counts[owner] += static_cast<int>(area);
            if (owner == g.row)
                std::copy_n(left.data() + (global / g.rows) * area, area,
                            right.data() + slots[j] * area);
        }
        double dummy = 0.0;
        double* receive = right.empty() ? &dummy : right.data();
        if (g.rows == g.cols) {
            // On a square grid all RHS tiles in a process column share a root.
            if (counts[g.col])
                MPI_Bcast(receive + displacements[g.col], counts[g.col],
                          MPI_DOUBLE, g.col, g.colComm);
        } else {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, receive,
                           counts.data(), displacements.data(), MPI_DOUBLE, g.colComm);
        }
        for (size_t j = 0; j < g.localCols; ++j) {
            if (g.col + j * g.cols < first) continue;
            const double* source = receive + slots[j] * area;
            double* dest = transposed.data() + j * area;
            for (size_t i = 0; i < g.block; ++i)
                for (size_t t = 0; t < g.block; ++t)
                    dest[t * g.block + i] = source[i * g.block + t];
        }
    }

    template<bool Subtract>
    void update(Matrix& matrix, size_t first) const {
        const auto& g = grid;
        const size_t area = g.block * g.block;
        for (size_t i = g.firstLocalRow(first); i < g.localRows; ++i) {
            const size_t globalI = g.row + i * g.rows;
            for (size_t j = 0; j < g.localCols; ++j) {
                const size_t globalJ = g.col + j * g.cols;
                if (globalJ < first || globalJ > globalI) continue;
                multiplyTile<Subtract>(matrix.local(i, j), left.data() + i * area,
                                       transposed.data() + j * area, g.block);
            }
        }
    }
};

// Distributed symmetric product. Only the lower triangle is needed; it is
// also sufficient to check every entry of the symmetric reconstruction.
void symmetricProduct(const Matrix& input, Matrix& output, bool triangular) {
    Panels panels(input.grid);
    for (size_t k = 0; k < input.grid.tiles; ++k) {
        const size_t first = triangular ? k : 0;
        panels.collect(input, k, first);
        panels.update<false>(output, first);
    }
}

void generatePositiveDefiniteMatrix(Matrix& matrix, size_t n) {
    const auto& g = matrix.grid;
    Matrix random(g);
    std::vector<unsigned int> rowSeeds(n);
    // Record the original rand_r state at each row. This retains the exact
    // seed-42 input while other ranks generate just their process-row data.
    if (g.rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            rowSeeds[i] = seed;
            for (size_t j = 0; j < n; ++j) {
                const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
                if ((i / g.block) % g.rows == 0 && (j / g.block) % g.cols == 0)
                    random.tile(i / g.block, j / g.block)[(i % g.block) * g.block + j % g.block] = value;
            }
        }
    }
    for (size_t offset = 0; offset < n;) {
        const int length = static_cast<int>(std::min(n - offset,
                                      static_cast<size_t>(std::numeric_limits<int>::max())));
        MPI_Bcast(rowSeeds.data() + offset, length, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        offset += static_cast<size_t>(length);
    }
    if (g.rank != 0) {
        for (size_t local = 0; local < g.localRows; ++local) {
            const size_t tileRow = g.row + local * g.rows;
            const size_t end = std::min(n, (tileRow + 1) * g.block);
            for (size_t i = tileRow * g.block; i < end; ++i) {
                unsigned int seed = rowSeeds[i];
                for (size_t j = 0; j < n; ++j) {
                    const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
                    if ((j / g.block) % g.cols == static_cast<size_t>(g.col))
                        random.local(local, (j / g.block) / g.cols)[(i % g.block) * g.block + j % g.block] = value;
                }
            }
        }
    }
    symmetricProduct(random, matrix, false);
    for (size_t k = 0; k < g.tiles; ++k) {
        if (k % g.rows != static_cast<size_t>(g.row) ||
            k % g.cols != static_cast<size_t>(g.col)) continue;
        double* diagonal = matrix.tile(k, k);
        const size_t width = std::min(g.block, n - k * g.block);
        for (size_t i = 0; i < width; ++i)
            diagonal[i * g.block + i] += static_cast<double>(n);
    }
}

bool factorDiagonal(double* tile, size_t b, size_t width, size_t& failed) {
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += tile[i * b + k] * tile[j * b + k];
            const double value = tile[i * b + j] - sum;
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    failed = i;
                    return false;
                }
                tile[i * b + j] = std::sqrt(value);
            } else {
                tile[i * b + j] = value / tile[j * b + j];
            }
        }
        std::fill(tile + i * b + i + 1, tile + (i + 1) * b, 0.0);
    }
    return true;
}

void solvePanel(double* tile, const double* diagonal, const double* inverse,
                size_t b, size_t height, size_t width) {
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += tile[i * b + k] * diagonal[j * b + k];
            tile[i * b + j] = (tile[i * b + j] - sum) * inverse[j];
        }
    }
}

bool choleskyDecomposition(Matrix& matrix, size_t n) {
    const auto& g = matrix.grid;
    const size_t area = g.block * g.block;
    Panels panels(g);
    // The extra entry communicates a failing diagonal in the same broadcast.
    std::vector<double> diagonal(area + 1), inverse(g.block);
    for (size_t k = 0; k < g.tiles; ++k) {
        const int owner = static_cast<int>(k % g.rows) * g.cols + static_cast<int>(k % g.cols);
        const size_t width = std::min(g.block, n - k * g.block);
        if (g.rank == owner) {
            double* tile = matrix.tile(k, k);
            size_t failed = 0;
            diagonal[area] = factorDiagonal(tile, g.block, width, failed)
                ? 0.0 : static_cast<double>(k * g.block + failed + 1);
            std::copy_n(tile, area, diagonal.data());
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(area + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (diagonal[area] != 0.0) {
            if (g.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       static_cast<size_t>(diagonal[area] - 1));
            return false;
        }
        if (k + 1 == g.tiles) break;
        for (size_t j = 0; j < width; ++j)
            inverse[j] = 1.0 / diagonal[j * g.block + j];
        if (g.col == static_cast<int>(k % g.cols)) {
            for (size_t i = g.firstLocalRow(k + 1); i < g.localRows; ++i) {
                const size_t global = g.row + i * g.rows;
                solvePanel(matrix.local(i, k / g.cols), diagonal.data(), inverse.data(),
                           g.block, std::min(g.block, n - global * g.block), width);
            }
        }
        panels.collect(matrix, k, k + 1);
        panels.update<true>(matrix, k + 1);
    }
    return true;
}

bool validateCholesky(const Matrix& factor, const Matrix& original, size_t n) {
    const auto& g = factor.grid;
    Matrix reconstructed(g);
    symmetricProduct(factor, reconstructed, true);
    double errors[2] = {0.0, 0.0};
    for (size_t i = 0; i < g.localRows; ++i) {
        const size_t globalI = g.row + i * g.rows;
        const size_t height = std::min(g.block, n - globalI * g.block);
        for (size_t j = 0; j < g.localCols; ++j) {
            const size_t globalJ = g.col + j * g.cols;
            if (globalJ > globalI) continue;
            const size_t width = std::min(g.block, n - globalJ * g.block);
            const double* result = reconstructed.local(i, j);
            const double* reference = original.local(i, j);
            for (size_t r = 0; r < height; ++r) {
                const size_t end = globalI == globalJ ? r + 1 : width;
                for (size_t c = 0; c < end; ++c) {
                    const size_t index = r * g.block + c;
                    const double error = std::fabs(result[index] - reference[index]);
                    if (!std::isfinite(error)) {
                        errors[0] = errors[1] = std::numeric_limits<double>::infinity();
                    } else {
                        errors[0] = std::max(errors[0], error);
                        errors[1] = std::max(errors[1], error / (std::fabs(reference[index]) + 1e-10));
                    }
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (g.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6)
            printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

// Gathering is solely for the original -r interface, outside timed work.
// Streaming one rank at a time limits root scratch storage to a local matrix,
// and chunked messages avoid MPI's int count limit.
std::vector<double> gatherFactor(const Matrix& matrix, size_t n) {
    const auto& g = matrix.grid;
    std::vector<double> result;
    if (g.rank != 0) {
        for (size_t offset = 0; offset < matrix.data.size();) {
            const int length = static_cast<int>(std::min(matrix.data.size() - offset,
                                          static_cast<size_t>(std::numeric_limits<int>::max())));
            MPI_Send(matrix.data.data() + offset, length, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
            offset += static_cast<size_t>(length);
        }
        return result;
    }
    result.resize(n * n, 0.0);
    std::vector<double> scratch;
    const size_t area = g.block * g.block;
    for (int source = 0; source < g.size; ++source) {
        const int row = source / g.cols, col = source % g.cols;
        const size_t nr = ProcessGrid::count(g.tiles, row, g.rows);
        const size_t nc = ProcessGrid::count(g.tiles, col, g.cols);
        const double* packed = matrix.data.data();
        if (source != 0) {
            scratch.resize(nr * nc * area);
            for (size_t offset = 0; offset < scratch.size();) {
                const int length = static_cast<int>(std::min(scratch.size() - offset,
                                              static_cast<size_t>(std::numeric_limits<int>::max())));
                MPI_Recv(scratch.data() + offset, length, MPI_DOUBLE, source, 11,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += static_cast<size_t>(length);
            }
            packed = scratch.data();
        }
        for (size_t i = 0; i < nr; ++i) {
            const size_t globalI = row + i * g.rows;
            const size_t height = std::min(g.block, n - globalI * g.block);
            for (size_t j = 0; j < nc; ++j) {
                const size_t globalJ = col + j * g.cols;
                if (globalJ > globalI) continue;
                const size_t width = std::min(g.block, n - globalJ * g.block);
                const double* tile = packed + (i * nc + j) * area;
                for (size_t r = 0; r < height; ++r)
                    std::copy_n(tile + r * g.block, width,
                                result.data() + (globalI * g.block + r) * n + globalJ * g.block);
            }
        }
    }
    return result;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '-' || value[0] == '\0' || *end != '\0' || errno == ERANGE ||
                parsed > std::numeric_limits<size_t>::max()) {
                if (rank == 0) printf("Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<size_t>(parsed);
            if (n && n > std::vector<double>().max_size() / n) {
                if (rank == 0) printf("Matrix size is too large\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
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

    ProcessGrid grid(n);
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix matrix(grid);
    generatePositiveDefiniteMatrix(matrix, n);
    std::optional<Matrix> original;
    if (validate) {
        original.emplace(grid);
        original->data = matrix.data;
    }
    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(matrix, n);
    double elapsed = MPI_Wtime() - start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0);
    }
    if (printResults) {
        const std::vector<double> factor = gatherFactor(matrix, n);
        if (rank == 0) print_results(factor, "CholeskyL");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(matrix, *original, n);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // Do not leave other ranks blocked in a collective on allocation failure.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
