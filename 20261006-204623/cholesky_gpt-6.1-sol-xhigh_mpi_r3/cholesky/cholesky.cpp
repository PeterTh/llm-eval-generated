#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

struct ProcessGrid {
    int rank, size, rows, cols, row, col;
    MPI_Comm rowComm, colComm;

    ProcessGrid() {
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

    ~ProcessGrid() {
        // An exception is followed by MPI_Abort; avoid collective cleanup
        // while other ranks may still be inside a different collective.
        if (std::uncaught_exceptions() == 0) {
            MPI_Comm_free(&rowComm);
            MPI_Comm_free(&colComm);
        }
    }

    int owner(size_t i, size_t j) const {
        return static_cast<int>(i % rows) * cols + static_cast<int>(j % cols);
    }
};

struct Tile {
    size_t i, j, offset;
};

// Only locally owned tiles are allocated. The factor stores the lower half;
// the temporary random matrix uses the same layout with all tiles present.
struct DistributedMatrix {
    const ProcessGrid& grid;
    size_t n, blocks, localRows, localCols, area;
    int block;
    std::vector<size_t> offsets;
    std::vector<Tile> tiles;
    std::vector<double> data;

    DistributedMatrix(const ProcessGrid& g, size_t size, int b, bool lower = true)
        : grid(g), n(size), blocks(size / b + (size % b != 0)),
          localRows(blocks / g.rows + (static_cast<size_t>(g.row) < blocks % g.rows)),
          localCols(blocks / g.cols + (static_cast<size_t>(g.col) < blocks % g.cols)),
          area(static_cast<size_t>(b) * b), block(b),
          offsets(localRows * localCols, std::numeric_limits<size_t>::max()) {
        // Panel collectives use MPI's int counts; the matrix itself may exceed
        // this limit and is transferred in chunks when results are requested.
        if (std::max(localRows, localCols) > static_cast<size_t>(INT_MAX) / area)
            throw std::length_error("Matrix panels exceed the MPI count limit");
        for (size_t i = g.row; i < blocks; i += g.rows) {
            for (size_t j = g.col; j < blocks && (!lower || j <= i); j += g.cols) {
                const size_t offset = tiles.size() * area;
                offsets[(i / g.rows) * localCols + j / g.cols] = offset;
                tiles.push_back({i, j, offset});
            }
        }
        data.resize(tiles.size() * area, 0.0);
    }

    const double* tile(size_t i, size_t j) const {
        const size_t offset = offsets[(i / grid.rows) * localCols + j / grid.cols];
        return offset == std::numeric_limits<size_t>::max() ? nullptr : data.data() + offset;
    }

    double* tile(size_t i, size_t j) {
        return const_cast<double*>(static_cast<const DistributedMatrix&>(*this).tile(i, j));
    }

    int extent(size_t i) const {
        return static_cast<int>(std::min(static_cast<size_t>(block), n - i * block));
    }
};

struct PanelWorkspace {
    std::vector<double> left, right;
    std::vector<int> counts, displacements;
    std::vector<size_t> offsets;

    explicit PanelWorkspace(const DistributedMatrix& a)
        : left(a.localRows * a.area), right(a.localCols * a.area),
          counts(a.grid.rows), displacements(a.grid.rows), offsets(a.blocks) {}

    // Broadcast the column panel along grid rows, then redistribute its
    // transpose within grid columns. Aggregating tiles gives two collectives
    // per panel, including on nonsquare process grids. The communicated volume
    // per rank is O(n * block / sqrt(p)) on a balanced grid.
    void exchange(const DistributedMatrix& a, size_t k, size_t first) {
        const ProcessGrid& g = a.grid;
        const size_t firstRow = first / g.rows + (static_cast<size_t>(g.row) < first % g.rows);
        if (g.col == static_cast<int>(k % g.cols)) {
            for (size_t r = firstRow; r < a.localRows; ++r) {
                const double* source = a.tile(r * g.rows + g.row, k);
                double* target = left.data() + r * a.area;
                if (source)
                    std::copy_n(source, a.area, target);
                else
                    std::fill_n(target, a.area, 0.0);
            }
        }
        const int count = static_cast<int>((a.localRows - firstRow) * a.area);
        MPI_Bcast(count ? left.data() + firstRow * a.area : left.data(), count,
                  MPI_DOUBLE, static_cast<int>(k % g.cols), g.rowComm);

        std::fill(counts.begin(), counts.end(), 0);
        const size_t firstCol = first / g.cols + (static_cast<size_t>(g.col) < first % g.cols);
        for (size_t c = firstCol; c < a.localCols; ++c) {
            const size_t j = c * g.cols + g.col;
            offsets[j] = counts[j % g.rows];
            counts[j % g.rows] += static_cast<int>(a.area);
        }
        int total = 0;
        for (int r = 0; r < g.rows; ++r) {
            displacements[r] = total;
            total += counts[r];
        }
        for (size_t c = firstCol; c < a.localCols; ++c) {
            const size_t j = c * g.cols + g.col;
            offsets[j] += displacements[j % g.rows];
            if (static_cast<int>(j % g.rows) == g.row) {
                const double* source = left.data() + (j / g.rows) * a.area;
                double* target = right.data() + offsets[j];
                for (int i = 0; i < a.block; ++i)
                    for (int t = 0; t < a.block; ++t)
                        target[t * a.block + i] = source[i * a.block + t];
            }
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, right.data(), counts.data(),
                       displacements.data(), MPI_DOUBLE, g.colComm);
    }
};

#if defined(__AVX2__)
template <bool add>
inline __m256d accumulate(__m256d sum, __m256d a, __m256d b) {
#if defined(__FMA__)
    if constexpr (!add)
        return _mm256_fmadd_pd(a, b, sum);
#endif
    return _mm256_add_pd(sum, _mm256_mul_pd(a, b));
}
#endif

// A 4-by-8 register kernel, with a portable fallback. SIMD runs across
// independent output entries, avoiding horizontal reductions or unsafe
// floating-point flags. add=true retains k-order for generation/validation.
template <bool add>
void tileProduct(double* __restrict target, const double* __restrict left,
                 const double* __restrict right, int block) {
    for (int i = 0; i < block; i += 4) {
        for (int j = 0; j < block; j += 8) {
#if defined(__AVX2__)
            double* d0 = target + (i + 0) * block + j;
            double* d1 = target + (i + 1) * block + j;
            double* d2 = target + (i + 2) * block + j;
            double* d3 = target + (i + 3) * block + j;
            __m256d c00 = add ? _mm256_loadu_pd(d0) : _mm256_setzero_pd();
            __m256d c01 = add ? _mm256_loadu_pd(d0 + 4) : _mm256_setzero_pd();
            __m256d c10 = add ? _mm256_loadu_pd(d1) : _mm256_setzero_pd();
            __m256d c11 = add ? _mm256_loadu_pd(d1 + 4) : _mm256_setzero_pd();
            __m256d c20 = add ? _mm256_loadu_pd(d2) : _mm256_setzero_pd();
            __m256d c21 = add ? _mm256_loadu_pd(d2 + 4) : _mm256_setzero_pd();
            __m256d c30 = add ? _mm256_loadu_pd(d3) : _mm256_setzero_pd();
            __m256d c31 = add ? _mm256_loadu_pd(d3 + 4) : _mm256_setzero_pd();
            for (int k = 0; k < block; ++k) {
                const double* b = right + k * block + j;
                const __m256d b0 = _mm256_loadu_pd(b);
                const __m256d b1 = _mm256_loadu_pd(b + 4);
                __m256d v = _mm256_broadcast_sd(left + (i + 0) * block + k);
                c00 = accumulate<add>(c00, v, b0);
                c01 = accumulate<add>(c01, v, b1);
                v = _mm256_broadcast_sd(left + (i + 1) * block + k);
                c10 = accumulate<add>(c10, v, b0);
                c11 = accumulate<add>(c11, v, b1);
                v = _mm256_broadcast_sd(left + (i + 2) * block + k);
                c20 = accumulate<add>(c20, v, b0);
                c21 = accumulate<add>(c21, v, b1);
                v = _mm256_broadcast_sd(left + (i + 3) * block + k);
                c30 = accumulate<add>(c30, v, b0);
                c31 = accumulate<add>(c31, v, b1);
            }
            if constexpr (!add) {
                c00 = _mm256_sub_pd(_mm256_loadu_pd(d0), c00);
                c01 = _mm256_sub_pd(_mm256_loadu_pd(d0 + 4), c01);
                c10 = _mm256_sub_pd(_mm256_loadu_pd(d1), c10);
                c11 = _mm256_sub_pd(_mm256_loadu_pd(d1 + 4), c11);
                c20 = _mm256_sub_pd(_mm256_loadu_pd(d2), c20);
                c21 = _mm256_sub_pd(_mm256_loadu_pd(d2 + 4), c21);
                c30 = _mm256_sub_pd(_mm256_loadu_pd(d3), c30);
                c31 = _mm256_sub_pd(_mm256_loadu_pd(d3 + 4), c31);
            }
            _mm256_storeu_pd(d0, c00);
            _mm256_storeu_pd(d0 + 4, c01);
            _mm256_storeu_pd(d1, c10);
            _mm256_storeu_pd(d1 + 4, c11);
            _mm256_storeu_pd(d2, c20);
            _mm256_storeu_pd(d2 + 4, c21);
            _mm256_storeu_pd(d3, c30);
            _mm256_storeu_pd(d3 + 4, c31);
#else
            double c0[8], c1[8], c2[8], c3[8];
            for (int t = 0; t < 8; ++t) {
                c0[t] = add ? target[(i + 0) * block + j + t] : 0.0;
                c1[t] = add ? target[(i + 1) * block + j + t] : 0.0;
                c2[t] = add ? target[(i + 2) * block + j + t] : 0.0;
                c3[t] = add ? target[(i + 3) * block + j + t] : 0.0;
            }
            for (int k = 0; k < block; ++k) {
                const double a0 = left[(i + 0) * block + k];
                const double a1 = left[(i + 1) * block + k];
                const double a2 = left[(i + 2) * block + k];
                const double a3 = left[(i + 3) * block + k];
                const double* b = right + k * block + j;
                for (int t = 0; t < 8; ++t) {
                    c0[t] += a0 * b[t];
                    c1[t] += a1 * b[t];
                    c2[t] += a2 * b[t];
                    c3[t] += a3 * b[t];
                }
            }
            for (int t = 0; t < 8; ++t) {
                if constexpr (add) {
                    target[(i + 0) * block + j + t] = c0[t];
                    target[(i + 1) * block + j + t] = c1[t];
                    target[(i + 2) * block + j + t] = c2[t];
                    target[(i + 3) * block + j + t] = c3[t];
                } else {
                    target[(i + 0) * block + j + t] -= c0[t];
                    target[(i + 1) * block + j + t] -= c1[t];
                    target[(i + 2) * block + j + t] -= c2[t];
                    target[(i + 3) * block + j + t] -= c3[t];
                }
            }
#endif
        }
    }
}

void generatePositiveDefiniteMatrix(DistributedMatrix& a, PanelWorkspace& panels) {
    DistributedMatrix random(a.grid, a.n, a.block, false);
    std::vector<double> slab, packed;
    if (a.grid.rank == 0) {
        // Stream the exact original rand_r sequence without a replicated n^2
        // input allocation or relying on platform-specific RNG jump-ahead.
        slab.resize(static_cast<size_t>(a.block) * a.n);
        packed.resize(a.area);
    }
    unsigned int seed = 42;
    for (size_t i = 0; i < a.blocks; ++i) {
        const int rows = a.extent(i);
        if (a.grid.rank == 0)
            for (size_t t = 0; t < static_cast<size_t>(rows) * a.n; ++t)
                slab[t] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
        for (size_t j = 0; j < a.blocks; ++j) {
            const int owner = a.grid.owner(i, j);
            if (a.grid.rank == 0) {
                std::fill(packed.begin(), packed.end(), 0.0);
                const int cols = a.extent(j);
                for (int r = 0; r < rows; ++r)
                    std::copy_n(slab.data() + r * a.n + j * a.block, cols,
                                packed.data() + r * a.block);
                if (owner == 0)
                    std::copy(packed.begin(), packed.end(), random.tile(i, j));
                else
                    MPI_Send(packed.data(), static_cast<int>(a.area), MPI_DOUBLE,
                             owner, 0, MPI_COMM_WORLD);
            } else if (a.grid.rank == owner) {
                MPI_Recv(random.tile(i, j), static_cast<int>(a.area), MPI_DOUBLE,
                         0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    for (size_t k = 0; k < a.blocks; ++k) {
        panels.exchange(random, k, 0);
        for (const Tile& tile : a.tiles)
            tileProduct<true>(a.data.data() + tile.offset,
                              panels.left.data() + (tile.i / a.grid.rows) * a.area,
                              panels.right.data() + panels.offsets[tile.j], a.block);
    }
    for (const Tile& tile : a.tiles) {
        if (tile.i == tile.j) {
            double* diagonal = a.data.data() + tile.offset;
            for (int i = 0; i < a.extent(tile.i); ++i)
                diagonal[i * a.block + i] += a.n;
        }
    }
}

bool choleskyDecomposition(DistributedMatrix& a, PanelWorkspace& panels) {
    std::vector<double> diagonal(a.area + 1);
    std::vector<double> transpose(a.area);
    for (size_t k = 0; k < a.blocks; ++k) {
        const int width = a.extent(k);
        const int owner = a.grid.owner(k, k);
        if (a.grid.rank == owner) {
            double* d = a.tile(k, k);
            diagonal[a.area] = 0.0;
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (int t = 0; t < j; ++t)
                        sum += d[i * a.block + t] * d[j * a.block + t];
                    const double value = d[i * a.block + j] - sum;
                    if (i == j) {
                        if (!(value > 0.0) || !std::isfinite(value)) {
                            diagonal[a.area] = static_cast<double>(k * a.block + i + 1);
                            break;
                        }
                        d[i * a.block + j] = std::sqrt(value);
                    } else {
                        d[i * a.block + j] = value / d[j * a.block + j];
                    }
                }
                if (diagonal[a.area] != 0.0)
                    break;
                std::fill(d + i * a.block + i + 1, d + (i + 1) * a.block, 0.0);
            }
            std::copy_n(d, a.area, diagonal.data());
        }
        // Include the failure flag in the diagonal broadcast so every rank
        // takes the same exit path, without an extra reduction per panel.
        MPI_Bcast(diagonal.data(), static_cast<int>(a.area + 1), MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);
        if (diagonal[a.area] != 0.0) {
            if (a.grid.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       static_cast<size_t>(diagonal[a.area] - 1));
            return false;
        }
        if (k + 1 == a.blocks)
            continue;
        if (a.grid.col == static_cast<int>(k % a.grid.cols)) {
            for (int i = 0; i < width; ++i)
                for (int j = 0; j < width; ++j)
                    transpose[i * a.block + j] = diagonal[j * a.block + i];
            for (const Tile& tile : a.tiles) {
                if (tile.j != k || tile.i <= k)
                    continue;
                double* panel = a.data.data() + tile.offset;
                for (int i = 0; i < a.extent(tile.i); ++i) {
                    double* row = panel + i * a.block;
                    for (int j = 0; j < width; ++j) {
                        row[j] /= diagonal[j * a.block + j];
                        const double value = row[j];
                        const double* coefficients = transpose.data() + j * a.block;
                        // Contiguous rank-one updates vectorize without a
                        // reduction, keeping the triangular solve in cache.
                        for (int t = j + 1; t < width; ++t)
                            row[t] -= value * coefficients[t];
                    }
                }
            }
        }
        panels.exchange(a, k, k + 1);
        for (const Tile& tile : a.tiles) {
            if (tile.j > k)
                tileProduct<false>(a.data.data() + tile.offset,
                                   panels.left.data() + (tile.i / a.grid.rows) * a.area,
                                   panels.right.data() + panels.offsets[tile.j], a.block);
        }
    }
    return true;
}

bool validateCholesky(const DistributedMatrix& a, const std::vector<double>& original,
                      PanelWorkspace& panels) {
    DistributedMatrix reconstructed(a.grid, a.n, a.block);
    for (size_t k = 0; k < a.blocks; ++k) {
        panels.exchange(a, k, k);
        for (const Tile& tile : a.tiles) {
            if (tile.j >= k)
                tileProduct<true>(reconstructed.data.data() + tile.offset,
                                  panels.left.data() + (tile.i / a.grid.rows) * a.area,
                                  panels.right.data() + panels.offsets[tile.j], a.block);
        }
    }
    double errors[2] = {0.0, 0.0};
    for (const Tile& tile : a.tiles) {
        for (int i = 0; i < a.extent(tile.i); ++i) {
            for (int j = 0; j < a.extent(tile.j); ++j) {
                const size_t offset = tile.offset + i * a.block + j;
                const double value = reconstructed.data[offset];
                const double error = std::abs(value - original[offset]);
                const double relative = error / (std::abs(original[offset]) + 1e-10);
                if (!std::isfinite(error) || !std::isfinite(relative)) {
                    errors[0] = errors[1] = std::numeric_limits<double>::infinity();
                } else {
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], relative);
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (a.grid.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6)
            printf("Validation failed: relative error too large\n");
    }
    // Both A and L*L^T are symmetric, so checking the distributed lower
    // triangle covers the full matrix without gathering either matrix.
    return errors[1] <= 1e-6;
}

void printDistributedResults(const DistributedMatrix& a) {
    if (a.grid.rank != 0) {
        for (size_t offset = 0; offset < a.data.size();) {
            const int count = static_cast<int>(std::min(a.data.size() - offset,
                                                       static_cast<size_t>(INT_MAX)));
            MPI_Send(a.data.data() + offset, count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            offset += count;
        }
        return;
    }
    std::vector<double> result(a.n * a.n, 0.0), received;
    for (int rank = 0; rank < a.grid.size; ++rank) {
        const int row = rank / a.grid.cols;
        const int col = rank % a.grid.cols;
        size_t count = 0;
        for (size_t i = row; i < a.blocks; i += a.grid.rows)
            for (size_t j = col; j < a.blocks && j <= i; j += a.grid.cols)
                count += a.area;
        const double* source = a.data.data();
        if (rank != 0) {
            received.resize(count);
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(count - offset,
                                                           static_cast<size_t>(INT_MAX)));
                MPI_Recv(received.data() + offset, chunk, MPI_DOUBLE, rank, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
            source = received.data();
        }
        size_t offset = 0;
        for (size_t i = row; i < a.blocks; i += a.grid.rows) {
            for (size_t j = col; j < a.blocks && j <= i; j += a.grid.cols) {
                for (int r = 0; r < a.extent(i); ++r)
                    std::copy_n(source + offset + r * a.block, a.extent(j),
                                result.data() + (i * a.block + r) * a.n + j * a.block);
                offset += a.area;
            }
        }
    }
    print_results(result, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
    ProcessGrid grid;
    unsigned long long options[3] = {512, 0, 0};
    int status = 0;
    if (grid.rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* text = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long value = strtoull(text, &end, 10);
                if (*text == '-' || end == text || *end != '\0' || errno == ERANGE ||
                    value > std::numeric_limits<size_t>::max() ||
                    (value && value > std::vector<double>().max_size() / value)) {
                    printf("Invalid matrix size: %s\n", text);
                    status = 1;
                    break;
                }
                options[0] = value;
            } else if (strcmp(argv[i], "-v") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[2] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                status = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                status = 1;
                break;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status)
        return status == 2 ? 0 : 1;
    MPI_Bcast(options, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    const size_t n = static_cast<size_t>(options[0]);
    const bool validate = options[1] != 0;
    const bool printResults = options[2] != 0;
    // Retain cache-sized tiles while exposing more tiles on larger grids.
    int block = 64;
    while (block > 16 && n / block < 2 * static_cast<size_t>(grid.rows))
        block /= 2;
    DistributedMatrix a(grid, n, block);
    PanelWorkspace panels(a);
    if (grid.rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(a, panels);
    std::vector<double> original;
    if (validate)
        original = a.data;
    if (grid.rank == 0)
        printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a, panels);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (grid.rank == 0)
            printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (grid.rank == 0) {
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? ops / duration / 1e9 : 0.0);
    }
    if (printResults)
        printDistributedResults(a);
    if (validate) {
        if (grid.rank == 0)
            printf("Validating result...\n");
        const bool valid = validateCholesky(a, original, panels);
        if (grid.rank == 0)
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 1;
    try {
        result = runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // Allocation failures must not strand other ranks inside collectives.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
