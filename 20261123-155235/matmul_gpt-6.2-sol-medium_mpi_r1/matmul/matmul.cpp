#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr size_t columnWidth = 8;
constexpr size_t rowHeight = 4;
constexpr size_t depthBlock = 128;
constexpr size_t rowBlock = 64;

#if defined(__AVX__)
using SimdVector = double __attribute__((vector_size(32)));
#else
using SimdVector = double __attribute__((vector_size(16)));
#endif
constexpr size_t vectorWidth = sizeof(SimdVector) / sizeof(double);
constexpr size_t vectorsPerPanel = columnWidth / vectorWidth;

// GCC's original dot product vectorized pairs of products and contracted the
// final scalar multiply-add on FMA targets.
#if defined(__GNUC__) && !defined(__clang__) && defined(FP_FAST_FMA)
constexpr bool fusedTail = true;
#else
constexpr bool fusedTail = false;
#endif

struct Tile {
    size_t row = 0;
    size_t col = 0;
    size_t rows = 0;
    size_t cols = 0;

    bool empty() const { return rows == 0 || cols == 0; }
    size_t stride() const { return (cols + columnWidth - 1) / columnWidth * columnWidth; }
};

class TileLayout {
public:
    TileLayout(size_t n, int ranks)
        : n(n), active(std::min(n * n, static_cast<size_t>(ranks))) {
        if (active != 0) {
            // Keep the grid close to square even for prime rank counts.
            const size_t bands = std::max(static_cast<size_t>(std::lround(std::sqrt(active))),
                                          (active + n - 1) / n);
            columns = active / bands;
            wideBands = active % bands;
        }
    }

    Tile operator[](size_t rank) const {
        if (rank >= active) {
            return {};
        }
        const size_t wideRanks = wideBands * (columns + 1);
        const bool wide = rank < wideRanks;
        const size_t width = columns + wide;
        const size_t bandStart = wide ? rank / width * width
                                      : wideRanks + (rank - wideRanks) / width * width;
        const size_t colIndex = rank - bandStart;

        // Assign taller bands to groups with more ranks when the size allows.
        const size_t row = std::uint64_t(n) * bandStart / active;
        const size_t rowEnd = std::uint64_t(n) * (bandStart + width) / active;
        const size_t col = std::uint64_t(n) * colIndex / width;
        const size_t colEnd = std::uint64_t(n) * (colIndex + 1) / width;
        return {row, col, rowEnd - row, colEnd - col};
    }

    size_t activeRanks() const { return active; }

private:
    size_t n;
    size_t active;
    size_t columns = 0;
    size_t wideBands = 0;
};

double getPseudoRndValue(const size_t nSquared, const size_t i, const size_t j) {
    return ((i + 1) * (i + j + 1) * 1299709 % nSquared) / static_cast<double>(nSquared);
}

void initInputs(double* a, double* packedB, size_t n, const Tile& tile) {
    // Global indices let each rank generate its own rows of A and columns of B.
    const size_t nSquared = n * n;
    for (size_t i = 0; i < tile.rows; ++i) {
        for (size_t k = 0; k < n; ++k) {
            a[i * n + k] = getPseudoRndValue(nSquared, tile.row + i, k);
        }
    }

    // Each panel stores consecutive columns at each k for the inner kernel.
    for (size_t j = 0; j < tile.cols; j += columnWidth) {
        double* panel = packedB + j * n;
        const size_t width = std::min(columnWidth, tile.cols - j);
        for (size_t k = 0; k < n; ++k) {
            for (size_t col = 0; col < width; ++col) {
                panel[k * columnWidth + col] = getPseudoRndValue(nSquared, k, tile.col + j + col);
            }
            std::fill_n(panel + k * columnWidth + width, columnWidth - width, 0.0);
        }
    }
}

template <size_t Rows>
inline void multiplyBlock(const double* __restrict a, const double* __restrict b,
                          double* __restrict c, size_t n, size_t stride, size_t depth) {
    // Vectorize across output columns while preserving each dot product's k order.
    SimdVector sums[Rows][vectorsPerPanel];
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < vectorsPerPanel; ++j) {
            std::memcpy(&sums[i][j], c + i * stride + j * vectorWidth, sizeof(SimdVector));
        }
    }
    for (size_t k = 0; k < depth; ++k) {
        for (size_t j = 0; j < vectorsPerPanel; ++j) {
            SimdVector right;
            std::memcpy(&right, b + k * columnWidth + j * vectorWidth, sizeof(right));
            for (size_t i = 0; i < Rows; ++i) {
                sums[i][j] += a[i * n + k] * right;
            }
        }
    }
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < vectorsPerPanel; ++j) {
            std::memcpy(c + i * stride + j * vectorWidth, &sums[i][j], sizeof(SimdVector));
        }
    }
}

void matrixMultiply(const double* a, const double* packedB, double* c, size_t n,
                    const Tile& tile) {
    const size_t stride = tile.stride();
    const size_t depth = n - (fusedTail ? n % 2 : 0);
    for (size_t row = 0; row < tile.rows; row += rowBlock) {
        const size_t rowEnd = std::min(row + rowBlock, tile.rows);
        for (size_t k = 0; k < depth; k += depthBlock) {
            const size_t count = std::min(depthBlock, depth - k);
            for (size_t j = 0; j < tile.cols; j += columnWidth) {
                const double* b = packedB + j * n + k * columnWidth;
                size_t i = row;
                for (; i + rowHeight <= rowEnd; i += rowHeight) {
                    multiplyBlock<rowHeight>(a + i * n + k, b, c + i * stride + j,
                                             n, stride, count);
                }
                for (; i < rowEnd; ++i) {
                    multiplyBlock<1>(a + i * n + k, b, c + i * stride + j, n, stride, count);
                }
            }
        }
    }

    if (depth != n) {
        for (size_t j = 0; j < tile.cols; j += columnWidth) {
            const double* b = packedB + j * n + depth * columnWidth;
            for (size_t i = 0; i < tile.rows; ++i) {
                double* out = c + i * stride + j;
                const double left = a[i * n + depth];
                for (size_t col = 0; col < columnWidth; ++col) {
                    out[col] = std::fma(left, b[col], out[col]);
                }
            }
        }
    }
}

MPI_Datatype tileType(const Tile& tile, size_t stride) {
    MPI_Datatype type;
    MPI_Type_vector(static_cast<int>(tile.rows), static_cast<int>(tile.cols),
                    static_cast<int>(stride), MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

void printDistributedResults(const std::vector<double>& local, const TileLayout& layout,
                             const Tile& tile, size_t n, int rank) {
    constexpr int resultTag = 0;
    if (rank != 0) {
        if (!tile.empty()) {
            MPI_Datatype type = tileType(tile, tile.stride());
            MPI_Send(local.data(), 1, type, 0, resultTag, MPI_COMM_WORLD);
            MPI_Type_free(&type);
        }
        return;
    }

    if (tile.rows == n && tile.cols == n && tile.stride() == n) {
        print_results(local, "MatrixC");
        return;
    }

    // Receive directly into the row-major layout expected by print_results.
    std::vector<double> result(n * n);
    std::vector<MPI_Request> requests;
    requests.reserve(layout.activeRanks());
    for (size_t source = 1; source < layout.activeRanks(); ++source) {
        const Tile remote = layout[source];
        if (!remote.empty()) {
            MPI_Datatype type = tileType(remote, n);
            MPI_Irecv(result.data() + remote.row * n + remote.col, 1, type,
                      static_cast<int>(source), resultTag, MPI_COMM_WORLD, &requests.emplace_back());
            MPI_Type_free(&type);
        }
    }
    for (size_t i = 0; i < tile.rows; ++i) {
        std::copy_n(local.data() + i * tile.stride(), tile.cols,
                    result.data() + (tile.row + i) * n + tile.col);
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    print_results(result, "MatrixC");
}

bool validateResult(const std::vector<double>& local, const Tile& tile, size_t n) {
    if (tile.empty()) {
        return true;
    }

    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = pi % n;
        if (i < tile.row || i >= tile.row + tile.rows) {
            continue;
        }
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = pj % n;
            if (j < tile.col || j >= tile.col + tile.cols) {
                continue;
            }

            double expected = 0.0;
            for (size_t k = 0; k < n; ++k) {
                expected += getPseudoRndValue(n * n, i, k) * getPseudoRndValue(n * n, k, j);
            }
            const double actual = local[(i - tile.row) * tile.stride() + j - tile.col];
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    // A negative size signals an early exit on every rank.
    enum { invalidArguments = -1, helpRequested = -2 };
    int options[] = {512, 0, 0};
    auto& [matrixSize, validate, printResults] = options;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* value = argv[++i];
                char* end;
                errno = 0;
                const long parsed = std::strtol(value, &end, 10);
                if (errno == ERANGE || end == value || *end || parsed < 0 ||
                    parsed > INT_MAX - static_cast<long>(columnWidth)) {
                    printf("Invalid matrix size: %s\n", value);
                    matrixSize = invalidArguments;
                    break;
                }
                matrixSize = static_cast<int>(parsed);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                matrixSize = helpRequested;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                matrixSize = invalidArguments;
                break;
            }
        }
    }
    MPI_Bcast(options, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (matrixSize < 0) {
        return matrixSize == invalidArguments;
    }
    const size_t n = static_cast<size_t>(matrixSize);
    if (n != 0 && n + columnWidth - 1 > std::vector<double>().max_size() / n) {
        if (rank == 0) {
            printf("Matrix size is too large: %zu\n", n);
        }
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    const TileLayout layout(n, ranks);
    const Tile tile = layout[rank];
    std::vector<double> local(tile.rows * tile.stride(), 0.0);
    double elapsed;
    {
        std::unique_ptr<double[]> a, packedB;
        if (!tile.empty()) {
            a = std::make_unique_for_overwrite<double[]>(tile.rows * n);
            packedB = std::make_unique_for_overwrite<double[]>(n * tile.stride());
            initInputs(a.get(), packedB.get(), n, tile);
        }
        if (rank == 0) {
            printf("Computing matrix multiplication...\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        if (!tile.empty()) {
            matrixMultiply(a.get(), packedB.get(), local.data(), n, tile);
        }
        elapsed = MPI_Wtime() - start;
    }
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(duration * 1000.0));
        const double gflops = duration > 0.0 ? (2.0 * n * n * n) / duration / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        printDistributedResults(local, layout, tile, n, rank);
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const int localValid = validateResult(local, tile, n);
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return !valid;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
