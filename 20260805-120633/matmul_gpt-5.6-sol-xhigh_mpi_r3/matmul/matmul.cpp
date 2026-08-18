#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct RowPartition {
    size_t first;
    size_t count;
};

// Give at most one more row to the lower ranks.  In addition to balancing the
// work, contiguous rows let Gatherv reconstruct the original row-major matrix.
RowPartition partitionRows(const size_t N, const int rank,
                           const int ranks) noexcept {
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rowsPerRank = N / rankCount;
    const size_t extraRows = N % rankCount;
    const size_t localRows = rowsPerRank + (rankIndex < extraRows ? 1 : 0);
    const size_t firstRow =
        rankIndex * rowsPerRank + std::min(rankIndex, extraRows);
    return {firstRow, localRows};
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const RowPartition rows) {
    for (size_t localI = 0; localI < rows.count; ++localI) {
        const size_t globalI = rows.first + localI;
        for (size_t j = 0; j < N; ++j) {
            mat[localI * N + j] = getPseudoRndValue(N, globalI, j);
        }
    }
}

// Cache-blocked local GEMM.  k is visited in increasing order for every output
// element, just as in the original implementation.  On AVX2 machines the 4x8
// microkernel keeps 32 output values in registers and shares B across four rows.
void matrixMultiply(const std::vector<double>& A,
                    const std::vector<double>& B, std::vector<double>& C,
                    const size_t localRows, const size_t N) {
    constexpr size_t rowBlock = 32;
    constexpr size_t columnBlock = 256;
    constexpr size_t depthBlock = 128;

    std::fill(C.begin(), C.end(), 0.0);

    const double* __restrict aData = A.data();
    const double* __restrict bData = B.data();
    double* __restrict cData = C.data();

    for (size_t jj = 0; jj < N; jj += columnBlock) {
        const size_t columnEnd = std::min(jj + columnBlock, N);

        for (size_t kk = 0; kk < N; kk += depthBlock) {
            const size_t depthEnd = std::min(kk + depthBlock, N);

            for (size_t ii = 0; ii < localRows; ii += rowBlock) {
                const size_t rowEnd = std::min(ii + rowBlock, localRows);
#if defined(__AVX2__) && defined(__FMA__)
                size_t i = ii;
                for (; i + 3 < rowEnd; i += 4) {
                    const double* __restrict a0 = aData + i * N;
                    const double* __restrict a1 = a0 + N;
                    const double* __restrict a2 = a1 + N;
                    const double* __restrict a3 = a2 + N;
                    double* __restrict c0 = cData + i * N;
                    double* __restrict c1 = c0 + N;
                    double* __restrict c2 = c1 + N;
                    double* __restrict c3 = c2 + N;

                    size_t j = jj;
                    for (; j + 7 < columnEnd; j += 8) {
                        __m256d c00 = _mm256_loadu_pd(c0 + j);
                        __m256d c01 = _mm256_loadu_pd(c0 + j + 4);
                        __m256d c10 = _mm256_loadu_pd(c1 + j);
                        __m256d c11 = _mm256_loadu_pd(c1 + j + 4);
                        __m256d c20 = _mm256_loadu_pd(c2 + j);
                        __m256d c21 = _mm256_loadu_pd(c2 + j + 4);
                        __m256d c30 = _mm256_loadu_pd(c3 + j);
                        __m256d c31 = _mm256_loadu_pd(c3 + j + 4);

                        auto accumulateProducts = [&](const size_t k) {
                            const double* __restrict b = bData + k * N + j;
                            const __m256d b0 = _mm256_loadu_pd(b);
                            const __m256d b1 = _mm256_loadu_pd(b + 4);
                            const __m256d av0 = _mm256_set1_pd(a0[k]);
                            const __m256d av1 = _mm256_set1_pd(a1[k]);
                            const __m256d av2 = _mm256_set1_pd(a2[k]);
                            const __m256d av3 = _mm256_set1_pd(a3[k]);
                            c00 = _mm256_add_pd(c00, _mm256_mul_pd(av0, b0));
                            c01 = _mm256_add_pd(c01, _mm256_mul_pd(av0, b1));
                            c10 = _mm256_add_pd(c10, _mm256_mul_pd(av1, b0));
                            c11 = _mm256_add_pd(c11, _mm256_mul_pd(av1, b1));
                            c20 = _mm256_add_pd(c20, _mm256_mul_pd(av2, b0));
                            c21 = _mm256_add_pd(c21, _mm256_mul_pd(av2, b1));
                            c30 = _mm256_add_pd(c30, _mm256_mul_pd(av3, b0));
                            c31 = _mm256_add_pd(c31, _mm256_mul_pd(av3, b1));
                        };

                        size_t k = kk;
                        for (; k + 1 < depthEnd; k += 2) {
                            accumulateProducts(k);
                            accumulateProducts(k + 1);
                        }
                        if (k < depthEnd) {
                            const double* __restrict b = bData + k * N + j;
                            const __m256d b0 = _mm256_loadu_pd(b);
                            const __m256d b1 = _mm256_loadu_pd(b + 4);
                            const __m256d av0 = _mm256_set1_pd(a0[k]);
                            const __m256d av1 = _mm256_set1_pd(a1[k]);
                            const __m256d av2 = _mm256_set1_pd(a2[k]);
                            const __m256d av3 = _mm256_set1_pd(a3[k]);
                            c00 = _mm256_fmadd_pd(av0, b0, c00);
                            c01 = _mm256_fmadd_pd(av0, b1, c01);
                            c10 = _mm256_fmadd_pd(av1, b0, c10);
                            c11 = _mm256_fmadd_pd(av1, b1, c11);
                            c20 = _mm256_fmadd_pd(av2, b0, c20);
                            c21 = _mm256_fmadd_pd(av2, b1, c21);
                            c30 = _mm256_fmadd_pd(av3, b0, c30);
                            c31 = _mm256_fmadd_pd(av3, b1, c31);
                        }

                        _mm256_storeu_pd(c0 + j, c00);
                        _mm256_storeu_pd(c0 + j + 4, c01);
                        _mm256_storeu_pd(c1 + j, c10);
                        _mm256_storeu_pd(c1 + j + 4, c11);
                        _mm256_storeu_pd(c2 + j, c20);
                        _mm256_storeu_pd(c2 + j + 4, c21);
                        _mm256_storeu_pd(c3 + j, c30);
                        _mm256_storeu_pd(c3 + j + 4, c31);
                    }

                    for (; j < columnEnd; ++j) {
                        double sum0 = c0[j];
                        double sum1 = c1[j];
                        double sum2 = c2[j];
                        double sum3 = c3[j];
                        size_t k = kk;
                        for (; k + 1 < depthEnd; k += 2) {
                            const double bValue = bData[k * N + j];
                            const double product0 = a0[k] * bValue;
                            const double product1 = a1[k] * bValue;
                            const double product2 = a2[k] * bValue;
                            const double product3 = a3[k] * bValue;
                            sum0 += product0;
                            sum1 += product1;
                            sum2 += product2;
                            sum3 += product3;

                            const double nextBValue = bData[(k + 1) * N + j];
                            const double nextProduct0 = a0[k + 1] * nextBValue;
                            const double nextProduct1 = a1[k + 1] * nextBValue;
                            const double nextProduct2 = a2[k + 1] * nextBValue;
                            const double nextProduct3 = a3[k + 1] * nextBValue;
                            sum0 += nextProduct0;
                            sum1 += nextProduct1;
                            sum2 += nextProduct2;
                            sum3 += nextProduct3;
                        }
                        if (k < depthEnd) {
                            const double bValue = bData[k * N + j];
                            sum0 = std::fma(a0[k], bValue, sum0);
                            sum1 = std::fma(a1[k], bValue, sum1);
                            sum2 = std::fma(a2[k], bValue, sum2);
                            sum3 = std::fma(a3[k], bValue, sum3);
                        }
                        c0[j] = sum0;
                        c1[j] = sum1;
                        c2[j] = sum2;
                        c3[j] = sum3;
                    }
                }

                for (; i < rowEnd; ++i) {
                    const double* __restrict a = aData + i * N;
                    double* __restrict c = cData + i * N;
                    size_t j = jj;
                    for (; j + 7 < columnEnd; j += 8) {
                        __m256d c0 = _mm256_loadu_pd(c + j);
                        __m256d c1 = _mm256_loadu_pd(c + j + 4);
                        size_t k = kk;
                        for (; k + 1 < depthEnd; k += 2) {
                            const double* __restrict b = bData + k * N + j;
                            const __m256d av = _mm256_set1_pd(a[k]);
                            c0 = _mm256_add_pd(
                                c0, _mm256_mul_pd(av, _mm256_loadu_pd(b)));
                            c1 = _mm256_add_pd(
                                c1,
                                _mm256_mul_pd(av, _mm256_loadu_pd(b + 4)));

                            b = bData + (k + 1) * N + j;
                            const __m256d nextAv = _mm256_set1_pd(a[k + 1]);
                            c0 = _mm256_add_pd(
                                c0,
                                _mm256_mul_pd(nextAv, _mm256_loadu_pd(b)));
                            c1 = _mm256_add_pd(
                                c1, _mm256_mul_pd(nextAv,
                                                _mm256_loadu_pd(b + 4)));
                        }
                        if (k < depthEnd) {
                            const double* __restrict b = bData + k * N + j;
                            const __m256d av = _mm256_set1_pd(a[k]);
                            c0 = _mm256_fmadd_pd(
                                av, _mm256_loadu_pd(b), c0);
                            c1 = _mm256_fmadd_pd(
                                av, _mm256_loadu_pd(b + 4), c1);
                        }
                        _mm256_storeu_pd(c + j, c0);
                        _mm256_storeu_pd(c + j + 4, c1);
                    }
                    for (; j < columnEnd; ++j) {
                        double sum = c[j];
                        size_t k = kk;
                        for (; k + 1 < depthEnd; k += 2) {
                            const double product = a[k] * bData[k * N + j];
                            sum += product;
                            const double nextProduct =
                                a[k + 1] * bData[(k + 1) * N + j];
                            sum += nextProduct;
                        }
                        if (k < depthEnd) {
                            sum = std::fma(a[k], bData[k * N + j], sum);
                        }
                        c[j] = sum;
                    }
                }
#else
                for (size_t i = ii; i < rowEnd; ++i) {
                    const double* __restrict a = aData + i * N;
                    double* __restrict c = cData + i * N;
                    for (size_t j = jj; j < columnEnd; ++j) {
                        double sum = c[j];
                        size_t k = kk;
                        for (; k + 1 < depthEnd; k += 2) {
                            const double product = a[k] * bData[k * N + j];
                            sum += product;
                            const double nextProduct =
                                a[k + 1] * bData[(k + 1) * N + j];
                            sum += nextProduct;
                        }
                        if (k < depthEnd) {
                            sum = std::fma(a[k], bData[k * N + j], sum);
                        }
                        c[j] = sum;
                    }
                }
#endif
            }
        }
    }
}

// Validate checkpoint rows on the ranks that own them, avoiding a gather when
// validation is requested without result printing.
bool validateResult(const std::vector<double>& B,
                    const std::vector<double>& localC, const size_t N,
                    const RowPartition rows, const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool localValid = true;

    for (const size_t rowPoint : checkPoints) {
        const size_t i = rowPoint % N;
        if (i < rows.first || i >= rows.first + rows.count) {
            continue;
        }

        const size_t localI = i - rows.first;
        for (const size_t columnPoint : checkPoints) {
            const size_t j = columnPoint % N;
            double expected = 0.0;
            size_t k = 0;
            for (; k + 1 < N; k += 2) {
                const double product =
                    getPseudoRndValue(N, i, k) * B[k * N + j];
                expected += product;
                const double nextProduct =
                    getPseudoRndValue(N, i, k + 1) * B[(k + 1) * N + j];
                expected += nextProduct;
            }
            if (k < N) {
                expected = std::fma(getPseudoRndValue(N, i, k),
                                    B[k * N + j], expected);
            }

            const double actual = localC[localI * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf(
                    "Validation failed on rank %d at (%zu, %zu): expected "
                    "%.10f, got %.10f (error: %.10e)\n",
                    rank, i, j, expected, actual, relError);
                localValid = false;
            }
        }
    }

    int valid = localValid ? 1 : 0;
    int globallyValid = 0;
    MPI_Allreduce(&valid, &globallyValid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    return globallyValid != 0;
}

// The normal benchmark does not communicate C.  Gather it only for -r, using
// the optimized collective whenever MPI's int-sized counts permit it.
void gatherResult(const std::vector<double>& localC,
                  std::vector<double>& globalC, const size_t N,
                  const RowPartition rows, const int rank, const int ranks) {
    constexpr int resultTag = 1701;
    const size_t maxMpiCount =
        static_cast<size_t>(std::numeric_limits<int>::max());
    const bool canUseGatherv = N * N <= maxMpiCount;

    if (canUseGatherv) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
            for (int r = 0; r < ranks; ++r) {
                const RowPartition part = partitionRows(N, r, ranks);
                counts[static_cast<size_t>(r)] =
                    static_cast<int>(part.count * N);
                displacements[static_cast<size_t>(r)] =
                    static_cast<int>(part.first * N);
            }
        }

        MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        return;
    }

    // Large MPI messages are split because MPI-3 collectives use int counts.
    if (rank == 0) {
        std::copy(localC.begin(), localC.end(), globalC.begin());
        for (int source = 1; source < ranks; ++source) {
            const RowPartition part = partitionRows(N, source, ranks);
            size_t remaining = part.count * N;
            size_t offset = part.first * N;
            while (remaining != 0) {
                const int chunk = static_cast<int>(
                    std::min(remaining, maxMpiCount));
                MPI_Recv(globalC.data() + offset, chunk, MPI_DOUBLE, source,
                         resultTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += static_cast<size_t>(chunk);
                remaining -= static_cast<size_t>(chunk);
            }
        }
    } else {
        size_t remaining = rows.count * N;
        size_t offset = 0;
        while (remaining != 0) {
            const int chunk =
                static_cast<int>(std::min(remaining, maxMpiCount));
            MPI_Send(localC.data() + offset, chunk, MPI_DOUBLE, 0, resultTag,
                     MPI_COMM_WORLD);
            offset += static_cast<size_t>(chunk);
            remaining -= static_cast<size_t>(chunk);
        }
    }
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf(
        "  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseMatrixSize(const char* text, size_t& N) {
    if (text[0] == '\0' || text[0] == '-') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || value == 0 ||
        value > std::numeric_limits<size_t>::max()) {
        return false;
    }
    N = static_cast<size_t>(value);
    return N <= std::numeric_limits<size_t>::max() / N;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseMatrixSize(argv[++i], N)) {
                if (rank == 0) {
                    std::printf("Invalid matrix size: %s\n", argv[i]);
                }
                argumentsValid = false;
                break;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
            break;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
            break;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    const size_t matrixElements = N * N;
    const RowPartition rows = partitionRows(N, rank, ranks);
    std::vector<double> localA;
    std::vector<double> B;
    std::vector<double> localC;

    if (matrixElements > B.max_size()) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size N=%zu is too large\n", N);
        }
        MPI_Finalize();
        return 2;
    }

    try {
        localA.resize(rows.count * N);
        B.resize(matrixElements);
        localC.resize(rows.count * N);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr,
                     "Rank %d could not allocate matrices for N=%zu\n", rank,
                     N);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }

    initMatrixRows(localA, N, rows);
    initMatrix(B, N);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(localA, B, localC, rows.count, N);
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long long durationMs =
            static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", durationMs);
        const double n = static_cast<double>(N);
        const double gflops = 2.0 * n * n * n / elapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> globalC;
        if (rank == 0) {
            try {
                globalC.resize(matrixElements);
            } catch (const std::bad_alloc&) {
                std::fprintf(stderr,
                             "Rank 0 could not allocate gathered result for "
                             "N=%zu\n",
                             N);
                MPI_Abort(MPI_COMM_WORLD, 2);
                return 2;
            }
        }
        gatherResult(localC, globalC, N, rows, rank, ranks);
        if (rank == 0) {
            print_results(globalC, "MatrixC");
        }
    }

    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const bool valid = validateResult(B, localC, N, rows, rank);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
