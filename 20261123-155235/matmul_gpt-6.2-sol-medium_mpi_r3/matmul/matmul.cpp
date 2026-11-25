#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Tile {
    size_t firstRow = 0;
    size_t firstColumn = 0;
    size_t rows = 0;
    size_t columns = 0;
};

Tile rankTile(const size_t N, const size_t ranks, const size_t rank) {
    if (rank >= ranks) {
        return {};
    }

    // Row bands can have different numbers of column tiles, including for prime rank counts.
    const size_t bands = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(ranks))));
    const size_t base = ranks / bands;
    const size_t extra = ranks % bands;
    const size_t wideRanks = extra * (base + 1);
    const size_t band = rank < wideRanks ? rank / (base + 1) : extra + (rank - wideRanks) / base;
    const size_t bandSize = base + (band < extra);
    const size_t bandStart = band * base + std::min(band, extra);
    const size_t column = rank - bandStart;

    // Give each band a row, then distribute the rest in proportion to its rank count.
    const size_t firstRow = band + uint64_t(N - bands) * bandStart / ranks;
    const size_t lastRow = band + 1 + uint64_t(N - bands) * (bandStart + bandSize) / ranks;
    const size_t firstColumn = column * (N / bandSize) + std::min(column, N % bandSize);
    const size_t columns = N / bandSize + (column < N % bandSize);
    return {firstRow, firstColumn, lastRow - firstRow, columns};
}

constexpr size_t kernelRows = 4;
constexpr size_t kernelColumns = 8;

void initInputs(std::vector<double>& A, std::vector<double>& B, const Tile& tile, const size_t N) {
    // Recreate the input slices for this tile from their global indices without MPI transfers.
    for (size_t i = 0; i < tile.rows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            A[i * N + k] = getPseudoRndValue(N, tile.firstRow + i, k);
        }
    }
    // Pack B in contiguous micro-panels so the inner kernel can stream along k.
    const size_t fullColumns = tile.columns / kernelColumns * kernelColumns;
    for (size_t j = 0; j < fullColumns; j += kernelColumns) {
        for (size_t k = 0; k < N; ++k) {
            for (size_t lane = 0; lane < kernelColumns; ++lane) {
                B[j * N + k * kernelColumns + lane] =
                    getPseudoRndValue(N, k, tile.firstColumn + j + lane);
            }
        }
    }
    for (size_t j = fullColumns; j < tile.columns; ++j) {
        for (size_t k = 0; k < N; ++k) {
            B[j * N + k] = getPseudoRndValue(N, k, tile.firstColumn + j);
        }
    }
}

// Preserve the serial dot product's accumulation order and native compiler rounding.
// On FMA targets Clang contracts the dot product, while GCC vectorizes pairs of
// products separately from their additions and contracts an odd scalar tail.
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA) || defined(__FP_FAST_FMA) || defined(FP_FAST_FMA)
constexpr bool nativeFma = true;
#else
constexpr bool nativeFma = false;
#endif
#if defined(__clang__)
constexpr bool fuseProducts = nativeFma;
#else
constexpr bool fuseProducts = false;
#endif
constexpr bool fuseOddTail = nativeFma && !fuseProducts;

#if defined(__AVX__)
struct VectorAccumulator {
    __m256d low;
    __m256d high;

    explicit VectorAccumulator(const double* c)
        : low(_mm256_loadu_pd(c)), high(_mm256_loadu_pd(c + 4)) {}

    void add(const double* aValue, const __m256d bLow, const __m256d bHigh) {
        const __m256d a = _mm256_broadcast_sd(aValue);
        if constexpr (fuseProducts) {
            low = _mm256_fmadd_pd(a, bLow, low);
            high = _mm256_fmadd_pd(a, bHigh, high);
        } else {
            low = _mm256_add_pd(low, _mm256_mul_pd(a, bLow));
            high = _mm256_add_pd(high, _mm256_mul_pd(a, bHigh));
        }
    }

    void store(double* c) const {
        _mm256_storeu_pd(c, low);
        _mm256_storeu_pd(c + 4, high);
    }
};

template <size_t Rows>
void multiplyVectorBlock(const double* A, const double* B, double* C,
                         const size_t aStride, const size_t cStride, const size_t depth) {
    static_assert(Rows == 1 || Rows == 4);
    VectorAccumulator row0(C);
    if constexpr (Rows == 4) {
        VectorAccumulator row1(C + cStride);
        VectorAccumulator row2(C + 2 * cStride);
        VectorAccumulator row3(C + 3 * cStride);
        for (size_t k = 0; k < depth; ++k) {
            const __m256d bLow = _mm256_loadu_pd(B + k * kernelColumns);
            const __m256d bHigh = _mm256_loadu_pd(B + k * kernelColumns + 4);
            row0.add(A + k, bLow, bHigh);
            row1.add(A + aStride + k, bLow, bHigh);
            row2.add(A + 2 * aStride + k, bLow, bHigh);
            row3.add(A + 3 * aStride + k, bLow, bHigh);
        }
        row1.store(C + cStride);
        row2.store(C + 2 * cStride);
        row3.store(C + 3 * cStride);
    } else {
        for (size_t k = 0; k < depth; ++k) {
            row0.add(A + k, _mm256_loadu_pd(B + k * kernelColumns),
                     _mm256_loadu_pd(B + k * kernelColumns + 4));
        }
    }
    row0.store(C);
}
#endif

template <size_t Rows, size_t Columns>
void multiplyBlock(const double* A, const double* B, double* C,
                   const size_t aStride, const size_t cStride, const size_t depth) {
#if defined(__AVX__)
    if constexpr (Columns == kernelColumns) {
        multiplyVectorBlock<Rows>(A, B, C, aStride, cStride, depth);
        return;
    }
#endif
    double sums[Rows][Columns];
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Columns; ++j) {
            sums[i][j] = C[i * cStride + j];
        }
    }
    for (size_t k = 0; k < depth; ++k) {
        for (size_t i = 0; i < Rows; ++i) {
            const double a = A[i * aStride + k];
            for (size_t j = 0; j < Columns; ++j) {
                const double b = B[k * Columns + j];
                if constexpr (fuseProducts) {
                    sums[i][j] = std::fma(a, b, sums[i][j]);
                } else {
                    sums[i][j] += a * b;
                }
            }
        }
    }
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Columns; ++j) {
            C[i * cStride + j] = sums[i][j];
        }
    }
}

template <size_t Rows>
void multiplyRowGroup(const double* A, const double* B, double* C, const size_t N,
                      const size_t columns, const size_t firstColumn, const size_t lastColumn,
                      const size_t firstK, const size_t depth) {
    size_t j = firstColumn;
    for (; j + kernelColumns <= lastColumn; j += kernelColumns) {
        multiplyBlock<Rows, kernelColumns>(A + firstK, B + j * N + firstK * kernelColumns,
                                           C + j, N, columns, depth);
    }
    for (; j < lastColumn; ++j) {
        multiplyBlock<Rows, 1>(A + firstK, B + j * N + firstK, C + j, N, columns, depth);
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const Tile& tile, const size_t N) {
    constexpr size_t columnBlock = 64;
    constexpr size_t depthBlock = 128;
    const size_t depth = fuseOddTail ? N - N % 2 : N;
    for (size_t j = 0; j < tile.columns; j += columnBlock) {
        const size_t end = std::min(j + columnBlock, tile.columns);
        for (size_t k = 0; k < depth; k += depthBlock) {
            const size_t blockDepth = std::min(depthBlock, depth - k);
            size_t i = 0;
            for (; i + kernelRows <= tile.rows; i += kernelRows) {
                multiplyRowGroup<kernelRows>(A.data() + i * N, B.data(), C.data() + i * tile.columns,
                                             N, tile.columns, j, end, k, blockDepth);
            }
            for (; i < tile.rows; ++i) {
                multiplyRowGroup<1>(A.data() + i * N, B.data(), C.data() + i * tile.columns,
                                    N, tile.columns, j, end, k, blockDepth);
            }
        }
    }

    if (depth < N) {
        const size_t fullColumns = tile.columns / kernelColumns * kernelColumns;
        for (size_t i = 0; i < tile.rows; ++i) {
            const double a = A[i * N + depth];
            size_t j = 0;
            for (; j < fullColumns; j += kernelColumns) {
                for (size_t lane = 0; lane < kernelColumns; ++lane) {
                    double& c = C[i * tile.columns + j + lane];
                    c = std::fma(a, B[j * N + depth * kernelColumns + lane], c);
                }
            }
            for (; j < tile.columns; ++j) {
                double& c = C[i * tile.columns + j];
                c = std::fma(a, B[j * N + depth], c);
            }
        }
    }
}

void printDistributedResults(const std::vector<double>& local, const size_t N,
                             const size_t activeRanks, const int rank) {
    if (rank == 0) {
        std::vector<double> result(N * N);
        const Tile own = rankTile(N, activeRanks, 0);
        for (size_t i = 0; i < own.rows; ++i) {
            std::copy_n(local.data() + i * own.columns, own.columns, result.data() + i * N);
        }
        // Receive directly into the final row-major layout, keeping MPI counts in range.
        for (size_t source = 1; source < activeRanks; ++source) {
            const Tile tile = rankTile(N, activeRanks, source);
            const size_t rowsPerMessage = INT_MAX / tile.columns;
            for (size_t i = 0; i < tile.rows; i += rowsPerMessage) {
                const int rows = static_cast<int>(std::min(rowsPerMessage, tile.rows - i));
                MPI_Datatype layout;
                MPI_Type_vector(rows, static_cast<int>(tile.columns), static_cast<int>(N),
                                MPI_DOUBLE, &layout);
                MPI_Type_commit(&layout);
                MPI_Recv(result.data() + (tile.firstRow + i) * N + tile.firstColumn, 1, layout,
                         static_cast<int>(source), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&layout);
            }
        }
        print_results(result, "MatrixC");
    } else if (static_cast<size_t>(rank) < activeRanks) {
        const Tile tile = rankTile(N, activeRanks, rank);
        const size_t rowsPerMessage = INT_MAX / tile.columns;
        for (size_t i = 0; i < tile.rows; i += rowsPerMessage) {
            const size_t rows = std::min(rowsPerMessage, tile.rows - i);
            MPI_Send(local.data() + i * tile.columns, static_cast<int>(rows * tile.columns),
                     MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

bool validateResult(const std::vector<double>& C, const Tile& tile, const size_t N, const int rank) {
    constexpr size_t checks = 5;
    std::array<double, checks * checks> local{};
    std::array<double, checks * checks> actual{};
    if (N == 0) {
        return true;
    }

    for (size_t pi = 0; pi < checks; ++pi) {
        const size_t i = pi % N;
        for (size_t pj = 0; pj < checks; ++pj) {
            const size_t j = pj % N;
            if (i >= tile.firstRow && i < tile.firstRow + tile.rows &&
                j >= tile.firstColumn && j < tile.firstColumn + tile.columns) {
                local[pi * checks + pj] = C[(i - tile.firstRow) * tile.columns + j - tile.firstColumn];
            }
        }
    }
    MPI_Reduce(local.data(), actual.data(), actual.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        for (size_t point = 0; point < actual.size(); ++point) {
            const size_t i = (point / checks) % N;
            const size_t j = (point % checks) % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double error = std::abs((actual[point] - expected) / (expected + 1e-10));
            if (error > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual[point], error);
                valid = 0;
                break;
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(const int argc, char** argv, const int rank, const int ranks) {
    std::array<int, 4> options{512, 0, 0, -1};
    auto& [size, validate, printResults, exitStatus] = options;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* value = argv[++i];
                char* end = nullptr;
                errno = 0;
                const long parsed = strtol(value, &end, 10);
                if (errno == ERANGE || end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX ||
                    (parsed != 0 && static_cast<size_t>(parsed) >
                        std::vector<double>().max_size() / static_cast<size_t>(parsed))) {
                    printf("Invalid matrix size: %s\n", value);
                    exitStatus = 1;
                    break;
                }
                size = static_cast<int>(parsed);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitStatus = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitStatus = 1;
                break;
            }
        }
    }
    MPI_Bcast(options.data(), options.size(), MPI_INT, 0, MPI_COMM_WORLD);
    if (exitStatus >= 0) {
        return exitStatus;
    }

    const size_t N = size;
    const size_t activeRanks = std::min(N * N, static_cast<size_t>(ranks));
    const Tile tile = rankTile(N, activeRanks, rank);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> C(tile.rows * tile.columns);
    double localTime;
    {
        std::vector<double> A(tile.rows * N);
        std::vector<double> B(tile.columns * N);
        initInputs(A, B, tile, N);
        if (rank == 0) {
            printf("Computing matrix multiplication...\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        matrixMultiply(A, B, C, tile, N);
        localTime = MPI_Wtime() - start;
    }
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        const double gflops = elapsed > 0.0 ? (2.0 * N * N * N) / elapsed / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) {
        printDistributedResults(C, N, activeRanks, rank);
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(C, tile, N, rank);
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
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
