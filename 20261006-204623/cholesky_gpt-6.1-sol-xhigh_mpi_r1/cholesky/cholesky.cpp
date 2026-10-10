#include <mpi.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// A two-dimensional block-cyclic layout balances the shrinking trailing matrix
// and limits panel communication to process rows and columns.
struct ProcessGrid {
    int rank, rows, columns, row, column;
    MPI_Comm rowComm, columnComm;

    explicit ProcessGrid(int processes) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int dimensions[2] = {0, 0};
        MPI_Dims_create(processes, 2, dimensions);
        rows = dimensions[0];
        columns = dimensions[1];
        row = rank / columns;
        column = rank % columns;
        MPI_Comm_split(MPI_COMM_WORLD, row, column, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, column, row, &columnComm);
    }

    ~ProcessGrid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&columnComm);
    }

    int owner(size_t i, size_t j) const {
        return static_cast<int>(i % rows) * columns + static_cast<int>(j % columns);
    }
};

size_t localCount(size_t count, int coordinate, int processes) {
    return count > static_cast<size_t>(coordinate)
        ? 1 + (count - 1 - coordinate) / processes : 0;
}

// Chunk large panels rather than overflowing MPI's int count argument.
void broadcast(double* data, size_t count, int root, MPI_Comm comm) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, comm);
        data += chunk;
        count -= chunk;
    }
}

struct DistributedMatrix {
    const ProcessGrid& grid;
    size_t n, blocks, localRows, localColumns;
    int blockSize;
    size_t tileElements;
    std::vector<std::vector<double>> tiles;

    DistributedMatrix(const ProcessGrid& g, size_t size, int width, bool triangular)
        : grid(g), n(size), blocks(size / width + (size % width != 0)),
          localRows(localCount(blocks, g.row, g.rows)),
          localColumns(localCount(blocks, g.column, g.columns)),
          blockSize(width), tileElements(static_cast<size_t>(width) * width),
          tiles(localRows * localColumns) {
        for (size_t i = g.row; i < blocks; i += g.rows) {
            for (size_t j = g.column; j < blocks; j += g.columns) {
                if (!triangular || i >= j) {
                    tile(i, j).resize(tileElements, 0.0);
                }
            }
        }
    }

    std::vector<double>& tile(size_t i, size_t j) {
        return tiles[(i / grid.rows) * localColumns + j / grid.columns];
    }
    const std::vector<double>& tile(size_t i, size_t j) const {
        return tiles[(i / grid.rows) * localColumns + j / grid.columns];
    }
    int extent(size_t i) const {
        return static_cast<int>(std::min(n - i * blockSize, static_cast<size_t>(blockSize)));
    }
};

// C +=/-= X Y^T. Y is packed transposed so independent output elements are
// contiguous. A 4-by-8 register block reuses both operands without an external
// BLAS library. Use native SIMD when available, retaining a portable fallback
// and the same accumulation order. Block sizes are always multiples of eight.
template <bool subtract>
void multiplyTile(const double* x, const double* yt, double* c, int b, int width) {
    for (int i = 0; i < b; i += 4) {
        for (int j = 0; j < b; j += 8) {
#if defined(__AVX2__) && defined(__FMA__)
            __m256d accumulators[4][2];
            for (int r = 0; r < 4; ++r) {
                accumulators[r][0] = _mm256_loadu_pd(c + (i + r) * b + j);
                accumulators[r][1] = _mm256_loadu_pd(c + (i + r) * b + j + 4);
            }
            for (int k = 0; k < width; ++k) {
                const __m256d y0 = _mm256_loadu_pd(yt + k * b + j);
                const __m256d y1 = _mm256_loadu_pd(yt + k * b + j + 4);
                for (int r = 0; r < 4; ++r) {
                    const __m256d value = _mm256_broadcast_sd(x + (i + r) * b + k);
                    if constexpr (subtract) {
                        accumulators[r][0] = _mm256_fnmadd_pd(value, y0, accumulators[r][0]);
                        accumulators[r][1] = _mm256_fnmadd_pd(value, y1, accumulators[r][1]);
                    } else {
                        accumulators[r][0] = _mm256_fmadd_pd(value, y0, accumulators[r][0]);
                        accumulators[r][1] = _mm256_fmadd_pd(value, y1, accumulators[r][1]);
                    }
                }
            }
            for (int r = 0; r < 4; ++r) {
                _mm256_storeu_pd(c + (i + r) * b + j, accumulators[r][0]);
                _mm256_storeu_pd(c + (i + r) * b + j + 4, accumulators[r][1]);
            }
#else
            double c0[8], c1[8], c2[8], c3[8];
            for (int t = 0; t < 8; ++t) {
                c0[t] = c[i * b + j + t];
                c1[t] = c[(i + 1) * b + j + t];
                c2[t] = c[(i + 2) * b + j + t];
                c3[t] = c[(i + 3) * b + j + t];
            }
            for (int k = 0; k < width; ++k) {
                const double x0 = x[i * b + k];
                const double x1 = x[(i + 1) * b + k];
                const double x2 = x[(i + 2) * b + k];
                const double x3 = x[(i + 3) * b + k];
                for (int t = 0; t < 8; ++t) {
                    const double y = yt[k * b + j + t];
                    if constexpr (subtract) {
                        c0[t] -= x0 * y;
                        c1[t] -= x1 * y;
                        c2[t] -= x2 * y;
                        c3[t] -= x3 * y;
                    } else {
                        c0[t] += x0 * y;
                        c1[t] += x1 * y;
                        c2[t] += x2 * y;
                        c3[t] += x3 * y;
                    }
                }
            }
            for (int t = 0; t < 8; ++t) {
                c[i * b + j + t] = c0[t];
                c[(i + 1) * b + j + t] = c1[t];
                c[(i + 2) * b + j + t] = c2[t];
                c[(i + 3) * b + j + t] = c3[t];
            }
#endif
        }
    }
}

// Cache one distributed block column in both orientations. Aggregate tiles into
// broadcasts, avoiding a separate collective for every tile. On square process
// grids each column needs only one column-communicator broadcast.
struct Panels {
    const DistributedMatrix& matrix;
    std::vector<double> rowData, columnData, exchange;

    explicit Panels(const DistributedMatrix& a)
        : matrix(a), rowData(a.localRows * a.tileElements),
          columnData(a.localColumns * a.tileElements),
          exchange(a.localColumns * a.tileElements) {}

    const double* rowTile(size_t i) const {
        return rowData.data() + (i / matrix.grid.rows) * matrix.tileElements;
    }
    const double* columnTile(size_t j) const {
        return columnData.data() + (j / matrix.grid.columns) * matrix.tileElements;
    }

    void distribute(size_t k, size_t first) {
        const auto& g = matrix.grid;
        const size_t elements = matrix.tileElements;
        const int b = matrix.blockSize;
        const size_t rowStart = localCount(first, g.row, g.rows);
        if (g.column == static_cast<int>(k % g.columns)) {
            for (size_t r = rowStart; r < matrix.localRows; ++r) {
                const size_t i = g.row + r * g.rows;
                std::copy_n(matrix.tile(i, k).data(), elements, rowData.data() + r * elements);
            }
        }
        if (rowStart < matrix.localRows) {
            broadcast(rowData.data() + rowStart * elements,
                      (matrix.localRows - rowStart) * elements,
                      static_cast<int>(k % g.columns), g.rowComm);
        }

        const size_t columnStart = localCount(first, g.column, g.columns);
        for (int source = 0; source < g.rows; ++source) {
            size_t count = 0;
            for (size_t s = columnStart; s < matrix.localColumns; ++s) {
                const size_t j = g.column + s * g.columns;
                if (static_cast<int>(j % g.rows) != source) continue;
                if (g.row == source) {
                    const double* input = rowTile(j);
                    double* output = exchange.data() + count * elements;
                    for (int r = 0; r < b; ++r) {
                        for (int c = 0; c < b; ++c) {
                            output[c * b + r] = input[r * b + c];
                        }
                    }
                }
                ++count;
            }
            broadcast(exchange.data(), count * elements, source, g.columnComm);
            count = 0;
            for (size_t s = columnStart; s < matrix.localColumns; ++s) {
                const size_t j = g.column + s * g.columns;
                if (static_cast<int>(j % g.rows) == source) {
                    std::copy_n(exchange.data() + count * elements, elements,
                                columnData.data() + s * elements);
                    ++count;
                }
            }
        }
    }
};

template <bool subtract>
void updateTrailing(DistributedMatrix& a, const Panels& panels, size_t first, int width) {
    const auto& g = a.grid;
    const size_t rowStart = localCount(first, g.row, g.rows);
    const size_t columnStart = localCount(first, g.column, g.columns);
    for (size_t r = rowStart; r < a.localRows; ++r) {
        const size_t i = g.row + r * g.rows;
        for (size_t s = columnStart; s < a.localColumns; ++s) {
            const size_t j = g.column + s * g.columns;
            if (j > i) break;
            multiplyTile<subtract>(panels.rowTile(i), panels.columnTile(j),
                                   a.tile(i, j).data(), a.blockSize, width);
        }
    }
}

void generatePositiveDefiniteMatrix(DistributedMatrix& a) {
    const auto& g = a.grid;
    DistributedMatrix random(g, a.n, a.blockSize, false);
    std::vector<double> rows, packed;
    if (g.rank == 0) {
        rows.resize(static_cast<size_t>(a.blockSize) * a.n);
        packed.resize(a.tileElements);
    }
    unsigned int seed = 42;
    // Stream B in the original rand_r order. Only O(n * blockSize) temporary
    // storage is needed at the root; B itself is distributed across all ranks.
    for (size_t i = 0; i < a.blocks; ++i) {
        const int height = a.extent(i);
        if (g.rank == 0) {
            for (size_t e = 0; e < static_cast<size_t>(height) * a.n; ++e) {
                rows[e] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            }
        }
        for (size_t j = 0; j < a.blocks; ++j) {
            const int destination = g.owner(i, j);
            if (g.rank == 0) {
                std::fill(packed.begin(), packed.end(), 0.0);
                for (int r = 0; r < height; ++r) {
                    std::copy_n(rows.data() + r * a.n + j * a.blockSize, a.extent(j),
                                packed.data() + r * a.blockSize);
                }
                if (destination == 0) {
                    random.tile(i, j) = packed;
                } else {
                    MPI_Send(packed.data(), static_cast<int>(a.tileElements), MPI_DOUBLE,
                             destination, 0, MPI_COMM_WORLD);
                }
            } else if (g.rank == destination) {
                MPI_Recv(random.tile(i, j).data(), static_cast<int>(a.tileElements), MPI_DOUBLE,
                         0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    Panels panels(random);
    // Distributed matrix multiplication A = B B^T, preserving k order.
    for (size_t k = 0; k < a.blocks; ++k) {
        panels.distribute(k, 0);
        updateTrailing<false>(a, panels, 0, a.extent(k));
    }
    for (size_t i = g.row; i < a.blocks; i += g.rows) {
        if (static_cast<int>(i % g.columns) == g.column) {
            double* diagonal = a.tile(i, i).data();
            for (int r = 0; r < a.extent(i); ++r) {
                diagonal[r * a.blockSize + r] += a.n;
            }
        }
    }
}

int factorDiagonal(double* a, int b, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += a[i * b + k] * a[j * b + k];
            }
            const double value = a[i * b + j] - sum;
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) return j;
                a[i * b + j] = std::sqrt(value);
            } else {
                a[i * b + j] = value / a[j * b + j];
            }
        }
        std::fill(a + i * b + i + 1, a + (i + 1) * b, 0.0);
    }
    return -1;
}

void solvePanel(double* a, const double* diagonal, int b, int height, int width) {
    // Transpose the small tile so each triangular-solve update works on
    // independent, contiguous rows and can use SIMD rather than scalar dots.
    alignas(64) double transposed[64 * 64];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) transposed[j * b + i] = a[i * b + j];
    }
    for (int j = 0; j < width; ++j) {
        double* current = transposed + j * b;
        const double reciprocal = 1.0 / diagonal[j * b + j];
        for (int i = 0; i < height; ++i) current[i] *= reciprocal;
        for (int k = j + 1; k < width; ++k) {
            double* target = transposed + k * b;
            const double value = diagonal[k * b + j];
            for (int i = 0; i < height; ++i) target[i] -= current[i] * value;
        }
    }
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) a[i * b + j] = transposed[j * b + i];
    }
}

void prepareDiagonal(DistributedMatrix& a, size_t k, std::vector<double>& diagonal) {
    if (a.grid.rank == a.grid.owner(k, k)) {
        auto& tile = a.tile(k, k);
        const int failure = factorDiagonal(tile.data(), a.blockSize, a.extent(k));
        std::copy(tile.begin(), tile.end(), diagonal.begin());
        diagonal[a.tileElements] = failure + 1;
    }
}

bool choleskyDecomposition(DistributedMatrix& a, Panels& panels,
                           std::vector<double>& diagonal) {
    const auto& g = a.grid;
    if (a.blocks == 0) return true;
    prepareDiagonal(a, 0, diagonal);
    broadcast(diagonal.data(), diagonal.size(), g.owner(0, 0), MPI_COMM_WORLD);
    for (size_t k = 0; k < a.blocks; ++k) {
        // The diagonal broadcast includes the failure status so every rank
        // takes the same exit path without a separate reduction at each step.
        if (diagonal[a.tileElements] != 0.0) {
            if (g.rank == 0) {
                const size_t failed = k * a.blockSize
                    + static_cast<size_t>(diagonal[a.tileElements] - 1);
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", failed);
            }
            return false;
        }
        if (k + 1 == a.blocks) break;
        if (g.column == static_cast<int>(k % g.columns)) {
            const size_t start = localCount(k + 1, g.row, g.rows);
            for (size_t r = start; r < a.localRows; ++r) {
                const size_t i = g.row + r * g.rows;
                solvePanel(a.tile(i, k).data(), diagonal.data(), a.blockSize,
                           a.extent(i), a.extent(k));
            }
        }
        panels.distribute(k, k + 1);

        // Update the next panel first. Its owner can factor and broadcast the
        // next diagonal while all ranks compute the remaining trailing update.
        const size_t next = k + 1;
        if (g.column == static_cast<int>(next % g.columns)) {
            const size_t start = localCount(next, g.row, g.rows);
            for (size_t r = start; r < a.localRows; ++r) {
                const size_t i = g.row + r * g.rows;
                multiplyTile<true>(panels.rowTile(i), panels.columnTile(next),
                                   a.tile(i, next).data(), a.blockSize, a.extent(k));
            }
        }
        prepareDiagonal(a, next, diagonal);
        MPI_Request request;
        MPI_Ibcast(diagonal.data(), static_cast<int>(diagonal.size()), MPI_DOUBLE,
                   g.owner(next, next), MPI_COMM_WORLD, &request);
        updateTrailing<true>(a, panels, next + 1, a.extent(k));
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    return true;
}

bool validateCholesky(const DistributedMatrix& l, const DistributedMatrix& original) {
    DistributedMatrix reconstructed(l.grid, l.n, l.blockSize, true);
    Panels panels(l);
    for (size_t k = 0; k < l.blocks; ++k) {
        panels.distribute(k, k);
        updateTrailing<false>(reconstructed, panels, k, l.extent(k));
    }
    double errors[2] = {0.0, 0.0};
    for (size_t i = l.grid.row; i < l.blocks; i += l.grid.rows) {
        for (size_t j = l.grid.column; j <= i; j += l.grid.columns) {
            const auto& result = reconstructed.tile(i, j);
            const auto& input = original.tile(i, j);
            for (int r = 0; r < l.extent(i); ++r) {
                const int columns = i == j ? r + 1 : l.extent(j);
                for (int c = 0; c < columns; ++c) {
                    const size_t offset = r * l.blockSize + c;
                    const double error = std::abs(result[offset] - input[offset]);
                    if (!std::isfinite(error)) {
                        errors[0] = errors[1] = std::numeric_limits<double>::infinity();
                    } else {
                        errors[0] = std::max(errors[0], error);
                        errors[1] = std::max(errors[1], error / (std::abs(input[offset]) + 1e-10));
                    }
                }
            }
        }
    }
    double maximum[2];
    MPI_Allreduce(errors, maximum, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (l.grid.rank == 0) {
        printf("Max absolute error: %.10e\n", maximum[0]);
        printf("Max relative error: %.10e\n", maximum[1]);
        if (maximum[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return maximum[1] <= 1e-6;
}

void printDistributedResults(const DistributedMatrix& a) {
    std::vector<double> result, received;
    if (a.grid.rank == 0) {
        result.resize(a.n * a.n, 0.0);
        received.resize(a.tileElements);
    }
    // Gather only when explicitly requested, retaining the original row-major
    // statistics, sample positions, and hash, including zero upper entries.
    for (size_t i = 0; i < a.blocks; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            const int owner = a.grid.owner(i, j);
            if (a.grid.rank == 0) {
                const double* values;
                if (owner == 0) {
                    values = a.tile(i, j).data();
                } else {
                    MPI_Recv(received.data(), static_cast<int>(a.tileElements), MPI_DOUBLE,
                             owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    values = received.data();
                }
                for (int r = 0; r < a.extent(i); ++r) {
                    std::copy_n(values + r * a.blockSize, a.extent(j),
                                result.data() + (i * a.blockSize + r) * a.n + j * a.blockSize);
                }
            } else if (a.grid.rank == owner) {
                MPI_Send(a.tile(i, j).data(), static_cast<int>(a.tileElements), MPI_DOUBLE,
                         0, 1, MPI_COMM_WORLD);
            }
        }
    }
    if (a.grid.rank == 0) print_results(result, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(size_t n, bool validate, bool printResults, int processes) {
    ProcessGrid grid(processes);
    int blockSize = 64;
    // Keep enough tiles for small matrices and large process grids, while
    // retaining larger, efficient matrix multiplication kernels for large n.
    while (blockSize > 8 && n / blockSize + (n % blockSize != 0)
           < static_cast<size_t>(2 * std::max(grid.rows, grid.columns))) {
        blockSize /= 2;
    }
    if (grid.rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    DistributedMatrix a(grid, n, blockSize, true);
    generatePositiveDefiniteMatrix(a);
    std::optional<DistributedMatrix> original;
    if (validate) original.emplace(a);
    Panels panels(a);
    std::vector<double> diagonal(a.tileElements + 1);
    if (grid.rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a, panels, diagonal);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (grid.rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (grid.rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(a);
    if (validate) {
        if (grid.rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(a, *original);
        if (grid.rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    unsigned long long matrixSize = 512;
    // status: 0 = run, 1 = help, 2 = argument error.
    int options[3] = {0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* value = argv[++i];
                char* end = nullptr;
                errno = 0;
                matrixSize = std::strtoull(value, &end, 10);
                if (value[0] == '-' || end == value || *end != '\0' || errno == ERANGE
                    || matrixSize > std::numeric_limits<size_t>::max()
                    || (matrixSize != 0 && matrixSize >
                        std::numeric_limits<size_t>::max() / sizeof(double) / matrixSize)) {
                    printf("Invalid matrix size: %s\n", value);
                    options[0] = 2;
                    break;
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[2] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                options[0] = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                options[0] = 2;
                break;
            }
        }
        if (options[0] != 0) printUsage(argv[0]);
    }
    MPI_Bcast(options, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrixSize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    int status = options[0] == 2 ? 1 : 0;
    if (options[0] == 0) {
        try {
            status = runBenchmark(static_cast<size_t>(matrixSize), options[1] != 0,
                                  options[2] != 0, processes);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
            status = 1;
        }
    }
    MPI_Finalize();
    return status;
}
