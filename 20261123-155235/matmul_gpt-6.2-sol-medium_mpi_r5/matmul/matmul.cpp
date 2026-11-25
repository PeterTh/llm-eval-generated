#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

struct Tile {
    size_t row, col, rows, cols;
};

class TileLayout {
  public:
    TileLayout(size_t n, int ranks)
        : active(std::min<size_t>(ranks, n * n)), n(n),
          bands(static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(active))))) {}

    Tile tile(size_t rank) const {
        if (rank >= active) {
            return {};
        }

        // Distribute ranks over near-square tiles, including for prime rank counts.
        const size_t ranksPerBand = active / bands;
        const size_t extraBands = active % bands;
        const size_t extraRanks = extraBands * (ranksPerBand + 1);
        const size_t band = rank < extraRanks
            ? rank / (ranksPerBand + 1)
            : extraBands + (rank - extraRanks) / ranksPerBand;
        const size_t firstRank = band * ranksPerBand + std::min(band, extraBands);
        const size_t bandRanks = ranksPerBand + (band < extraBands);
        const size_t col = rank - firstRank;

        // Give each band at least one row, then weight the rest by its rank count.
        const std::uint64_t extraRows = n - bands;
        const size_t rowBegin = band + extraRows * firstRank / active;
        const size_t rowEnd = band + 1 + extraRows * (firstRank + bandRanks) / active;
        const size_t colBegin = n * col / bandRanks;
        const size_t colEnd = n * (col + 1) / bandRanks;
        return {rowBegin, colBegin, rowEnd - rowBegin, colEnd - colBegin};
    }

    const size_t active;

  private:
    size_t n, bands;
};

constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

void initMatrix(double* mat, size_t stride, size_t n, const Tile& tile) {
    for (size_t i = 0; i < tile.rows; ++i) {
        for (size_t j = 0; j < tile.cols; ++j) {
            mat[i * stride + j] = getPseudoRndValue(n, tile.row + i, tile.col + j);
        }
    }
}

#if defined(__AVX512F__)
constexpr size_t simdBytes = 64;
#elif defined(__AVX__)
constexpr size_t simdBytes = 32;
#elif defined(__SSE2__) || defined(__aarch64__)
constexpr size_t simdBytes = 16;
#else
constexpr size_t simdBytes = sizeof(double);
#endif
using Simd = double __attribute__((vector_size(simdBytes)));
constexpr size_t lanes = sizeof(Simd) / sizeof(double);
constexpr size_t panelCols = 3 * lanes;

// GCC vectorizes the serial dot product in pairs and contracts its scalar remainder.
#if defined(__GNUC__) && !defined(__clang__) && (defined(__FMA__) || defined(__aarch64__))
constexpr bool fusedRemainder = true;
#else
constexpr bool fusedRemainder = false;
#endif

size_t paddedCols(size_t cols) {
    return (cols + lanes - 1) / lanes * lanes;
}

void initRightMatrix(std::vector<double>& b, size_t stride, size_t n, const Tile& tile) {
    // Pack the columns used by a SIMD tile so successive k steps read contiguous data.
    for (size_t j = 0; j < tile.cols; j += panelCols) {
        const size_t width = std::min(panelCols, stride - j);
        const size_t cols = std::min(panelCols, tile.cols - j);
        initMatrix(b.data() + j * n, width, n, {0, tile.col + j, n, cols});
    }
}

Simd loadVector(const double* src) {
    Simd value;
    std::memcpy(&value, src, sizeof(value));
    return value;
}

template <size_t Rows, size_t Vectors>
void multiplyTile(const double* a, const double* b, double* c,
                  size_t n, size_t stride, size_t depth) {
    Simd accum[Rows][Vectors];
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t v = 0; v < Vectors; ++v) {
            accum[i][v] = loadVector(c + i * stride + v * lanes);
        }
    }
    // Each SIMD lane owns a distinct output element and visits k in ascending order.
    for (size_t k = 0; k < depth; ++k) {
        Simd right[Vectors];
        for (size_t v = 0; v < Vectors; ++v) {
            right[v] = loadVector(b + v * lanes);
        }
        for (size_t i = 0; i < Rows; ++i) {
            const double left = a[i * n + k];
            for (size_t v = 0; v < Vectors; ++v) {
                accum[i][v] += left * right[v];
            }
        }
        b += Vectors * lanes;
    }
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t v = 0; v < Vectors; ++v) {
            std::memcpy(c + i * stride + v * lanes, &accum[i][v], sizeof(Simd));
        }
    }
}

template <size_t Rows>
void multiplyColumns(const double* a, const double* b, double* c,
                     size_t n, size_t stride, size_t kk, size_t depth, size_t cols) {
    size_t j = 0;
    for (; j + panelCols <= cols; j += panelCols) {
        multiplyTile<Rows, 3>(a, b + j * n + kk * panelCols, c + j, n, stride, depth);
    }
    if (j + 2 * lanes == cols) {
        multiplyTile<Rows, 2>(a, b + j * n + kk * 2 * lanes, c + j, n, stride, depth);
    } else if (j < cols) {
        multiplyTile<Rows, 1>(a, b + j * n + kk * lanes, c + j, n, stride, depth);
    }
}

void matrixMultiply(const std::vector<double>& a, const std::vector<double>& b,
                    std::vector<double>& c, size_t n, size_t rows, size_t stride) {
    constexpr size_t rowBlock = 64;
    constexpr size_t colBlock = 192;
    constexpr size_t depthBlock = 128;
    static_assert(colBlock % panelCols == 0);
    const size_t depth = n - (fusedRemainder ? n % 2 : 0);

    for (size_t ii = 0; ii < rows; ii += rowBlock) {
        const size_t rowEnd = std::min(ii + rowBlock, rows);
        for (size_t jj = 0; jj < stride; jj += colBlock) {
            const size_t cols = std::min(colBlock, stride - jj);
            for (size_t kk = 0; kk < depth; kk += depthBlock) {
                const size_t blockDepth = std::min(depthBlock, depth - kk);
                const double* right = b.data() + jj * n;
                size_t i = ii;
                for (; i + 4 <= rowEnd; i += 4) {
                    multiplyColumns<4>(a.data() + i * n + kk, right,
                                       c.data() + i * stride + jj, n, stride, kk, blockDepth, cols);
                }
                for (; i < rowEnd; ++i) {
                    multiplyColumns<1>(a.data() + i * n + kk, right,
                                       c.data() + i * stride + jj, n, stride, kk, blockDepth, cols);
                }
            }
        }
    }
    if (depth < n) {
        for (size_t j = 0; j < stride; j += panelCols) {
            const size_t width = std::min(panelCols, stride - j);
            const double* right = b.data() + j * n + depth * width;
            for (size_t i = 0; i < rows; ++i) {
                for (size_t v = 0; v < width; ++v) {
                    double& value = c[i * stride + j + v];
                    value = std::fma(a[i * n + depth], right[v], value);
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, size_t n, size_t stride, const Tile& tile) {
    for (size_t i = tile.row; i < std::min(size_t{5}, tile.row + tile.rows); ++i) {
        for (size_t j = tile.col; j < std::min(size_t{5}, tile.col + tile.cols); ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k) {
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            }
            const double actual = c[(i - tile.row) * stride + j - tile.col];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

MPI_Datatype tileType(const Tile& tile, size_t stride) {
    MPI_Datatype type;
    MPI_Type_create_hvector(static_cast<int>(tile.rows), static_cast<int>(tile.cols),
                            static_cast<MPI_Aint>(stride * sizeof(double)), MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

void reportResults(const std::vector<double>& c, size_t n, size_t stride,
                   const TileLayout& layout, const Tile& tile, int rank) {
    if (rank != 0) {
        if (tile.rows != 0) {
            MPI_Datatype type = tileType(tile, stride);
            MPI_Send(c.data(), 1, type, 0, 0, MPI_COMM_WORLD);
            MPI_Type_free(&type);
        }
        return;
    }
    if (layout.active <= 1 && stride == n) {
        print_results(c, "MatrixC");
        return;
    }

    // Receive tiles directly into their row-major positions for the original result report.
    std::vector<double> result(n * n);
    std::vector<MPI_Request> requests(layout.active - 1);
    for (size_t source = 1; source < layout.active; ++source) {
        const Tile other = layout.tile(source);
        MPI_Datatype type = tileType(other, n);
        MPI_Irecv(result.data() + other.row * n + other.col, 1, type,
                  static_cast<int>(source), 0, MPI_COMM_WORLD, &requests[source - 1]);
        MPI_Type_free(&type);
    }
    for (size_t i = 0; i < tile.rows; ++i) {
        std::copy_n(c.data() + i * stride, tile.cols, result.data() + i * n);
    }
    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
    print_results(result, "MatrixC");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            const char* value = argv[++i];
            errno = 0;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || errno == ERANGE || parsed < 0 || parsed > INT_MAX
                || (parsed > 0 && static_cast<size_t>(parsed) > SIZE_MAX / static_cast<size_t>(parsed))) {
                if (rank == 0) {
                    printf("Invalid matrix size: %s\n", value);
                    printUsage(argv[0]);
                }
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    const TileLayout layout(n, ranks);
    const Tile tile = layout.tile(rank);
    const size_t stride = paddedCols(tile.cols);
    std::vector<double> a(tile.rows * n);
    std::vector<double> b(n * stride);
    std::vector<double> c(tile.rows * stride);

    // The inputs are deterministic, so ranks generate their own row and column slices.
    if (tile.rows != 0) {
        initMatrix(a.data(), n, n, {tile.row, 0, tile.rows, n});
        initRightMatrix(b, stride, n, tile);
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double localSeconds = 0.0;
    if (tile.rows != 0) {
        const double start = MPI_Wtime();
        matrixMultiply(a, b, c, n, tile.rows, stride);
        localSeconds = MPI_Wtime() - start;
    }
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double gflops = seconds > 0.0 ? (2.0 * n * n * n) / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) {
        reportResults(c, n, stride, layout, tile, rank);
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const int localValid = validateResult(c, n, stride, tile);
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, status);
    }
    MPI_Finalize();
    return status;
}
