#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

namespace {

constexpr size_t packedColumnTile = 12;

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct RowPartition {
    size_t firstRow;
    size_t rowCount;
};

RowPartition partitionRows(const size_t N, const int rank,
                           const int processCount) noexcept {
    const size_t processes = static_cast<size_t>(processCount);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t rowsPerRank = N / processes;
    const size_t remainder = N % processes;

    return {rankIndex * rowsPerRank + std::min(rankIndex, remainder),
            rowsPerRank + (rankIndex < remainder ? 1U : 0U)};
}

void initMatrixRows(std::vector<double>& matrix, const size_t N,
                    const size_t firstRow, const size_t rowCount) {
    for (size_t localRow = 0; localRow < rowCount; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        double* const row = matrix.data() + localRow * N;
        for (size_t column = 0; column < N; ++column) {
            row[column] = getPseudoRndValue(N, globalRow, column);
        }
    }
}

size_t packedColumnExtent(const size_t N) noexcept {
    return ((N + packedColumnTile - 1) / packedColumnTile) *
           packedColumnTile;
}

// Store B in column panels. A microkernel consumes one complete panel while
// walking k, turning what would be large-stride accesses into a linear stream.
void initPackedMatrixB(std::vector<double>& matrix, const size_t N) {
    for (size_t columnBegin = 0; columnBegin < N;
         columnBegin += packedColumnTile) {
        double* const panel = matrix.data() + columnBegin * N;
        const size_t panelWidth =
            std::min(packedColumnTile, N - columnBegin);

        for (size_t depth = 0; depth < N; ++depth) {
            double* const panelRow = panel + depth * packedColumnTile;
            for (size_t lane = 0; lane < panelWidth; ++lane) {
                panelRow[lane] =
                    getPseudoRndValue(N, depth, columnBegin + lane);
            }
            std::fill(panelRow + panelWidth,
                      panelRow + packedColumnTile, 0.0);
        }
    }
}

double packedBValue(const std::vector<double>& matrix, const size_t N,
                    const size_t row, const size_t column) noexcept {
    const size_t columnBegin =
        (column / packedColumnTile) * packedColumnTile;
    return matrix[columnBegin * N + row * packedColumnTile +
                  column - columnBegin];
}

// MPI collectives use an int element count before MPI-4. Split very large
// matrices into identical collective calls so the implementation is not
// limited to INT_MAX elements.
void broadcastMatrix(std::vector<double>& matrix, const int root,
                     MPI_Comm communicator) {
    constexpr size_t maxChunk =
        static_cast<size_t>(std::numeric_limits<int>::max());

    for (size_t offset = 0; offset < matrix.size();) {
        const int chunk = static_cast<int>(
            std::min(maxChunk, matrix.size() - offset));
        MPI_Bcast(matrix.data() + offset, chunk, MPI_DOUBLE, root,
                  communicator);
        offset += static_cast<size_t>(chunk);
    }
}

// A register-blocked kernel for AVX2/FMA targets. A 4x12 output tile remains in
// registers throughout the entire dot product, and every B vector is reused by
// four rows. Each accumulator still visits k in increasing order, preserving
// the arithmetic ordering of the original implementation.
void matrixMultiply(const std::vector<double>& localA,
                    const std::vector<double>& B,
                    std::vector<double>& localC, const size_t N,
                    const size_t localRows) {
#if defined(__AVX2__) && defined(__FMA__)
    constexpr size_t vectorWidth = 4;
    constexpr size_t columnTile = 3 * vectorWidth;
    constexpr size_t rowTile = 4;

    size_t row = 0;
    for (; row + rowTile <= localRows; row += rowTile) {
        const double* const a0 = localA.data() + (row + 0) * N;
        const double* const a1 = localA.data() + (row + 1) * N;
        const double* const a2 = localA.data() + (row + 2) * N;
        const double* const a3 = localA.data() + (row + 3) * N;
        double* const c0 = localC.data() + (row + 0) * N;
        double* const c1 = localC.data() + (row + 1) * N;
        double* const c2 = localC.data() + (row + 2) * N;
        double* const c3 = localC.data() + (row + 3) * N;

        size_t column = 0;
        for (; column + columnTile <= N; column += columnTile) {
            __m256d c00 = _mm256_setzero_pd();
            __m256d c01 = _mm256_setzero_pd();
            __m256d c02 = _mm256_setzero_pd();
            __m256d c10 = _mm256_setzero_pd();
            __m256d c11 = _mm256_setzero_pd();
            __m256d c12 = _mm256_setzero_pd();
            __m256d c20 = _mm256_setzero_pd();
            __m256d c21 = _mm256_setzero_pd();
            __m256d c22 = _mm256_setzero_pd();
            __m256d c30 = _mm256_setzero_pd();
            __m256d c31 = _mm256_setzero_pd();
            __m256d c32 = _mm256_setzero_pd();

            for (size_t depth = 0; depth < N; ++depth) {
                const double* const b =
                    B.data() + column * N + depth * packedColumnTile;
                const __m256d b0 = _mm256_loadu_pd(b + 0 * vectorWidth);
                const __m256d b1 = _mm256_loadu_pd(b + 1 * vectorWidth);
                const __m256d b2 = _mm256_loadu_pd(b + 2 * vectorWidth);

                const __m256d av0 = _mm256_set1_pd(a0[depth]);
                c00 = _mm256_fmadd_pd(av0, b0, c00);
                c01 = _mm256_fmadd_pd(av0, b1, c01);
                c02 = _mm256_fmadd_pd(av0, b2, c02);
                const __m256d av1 = _mm256_set1_pd(a1[depth]);
                c10 = _mm256_fmadd_pd(av1, b0, c10);
                c11 = _mm256_fmadd_pd(av1, b1, c11);
                c12 = _mm256_fmadd_pd(av1, b2, c12);
                const __m256d av2 = _mm256_set1_pd(a2[depth]);
                c20 = _mm256_fmadd_pd(av2, b0, c20);
                c21 = _mm256_fmadd_pd(av2, b1, c21);
                c22 = _mm256_fmadd_pd(av2, b2, c22);
                const __m256d av3 = _mm256_set1_pd(a3[depth]);
                c30 = _mm256_fmadd_pd(av3, b0, c30);
                c31 = _mm256_fmadd_pd(av3, b1, c31);
                c32 = _mm256_fmadd_pd(av3, b2, c32);
            }

            _mm256_storeu_pd(c0 + column + 0 * vectorWidth, c00);
            _mm256_storeu_pd(c0 + column + 1 * vectorWidth, c01);
            _mm256_storeu_pd(c0 + column + 2 * vectorWidth, c02);
            _mm256_storeu_pd(c1 + column + 0 * vectorWidth, c10);
            _mm256_storeu_pd(c1 + column + 1 * vectorWidth, c11);
            _mm256_storeu_pd(c1 + column + 2 * vectorWidth, c12);
            _mm256_storeu_pd(c2 + column + 0 * vectorWidth, c20);
            _mm256_storeu_pd(c2 + column + 1 * vectorWidth, c21);
            _mm256_storeu_pd(c2 + column + 2 * vectorWidth, c22);
            _mm256_storeu_pd(c3 + column + 0 * vectorWidth, c30);
            _mm256_storeu_pd(c3 + column + 1 * vectorWidth, c31);
            _mm256_storeu_pd(c3 + column + 2 * vectorWidth, c32);
        }

        const double* const finalPanel = B.data() + column * N;
        size_t lane = 0;
        for (; column + vectorWidth <= N;
             column += vectorWidth, lane += vectorWidth) {
            __m256d cv0 = _mm256_setzero_pd();
            __m256d cv1 = _mm256_setzero_pd();
            __m256d cv2 = _mm256_setzero_pd();
            __m256d cv3 = _mm256_setzero_pd();
            for (size_t depth = 0; depth < N; ++depth) {
                const __m256d bv =
                    _mm256_loadu_pd(finalPanel +
                                    depth * packedColumnTile + lane);
                cv0 = _mm256_fmadd_pd(_mm256_set1_pd(a0[depth]), bv, cv0);
                cv1 = _mm256_fmadd_pd(_mm256_set1_pd(a1[depth]), bv, cv1);
                cv2 = _mm256_fmadd_pd(_mm256_set1_pd(a2[depth]), bv, cv2);
                cv3 = _mm256_fmadd_pd(_mm256_set1_pd(a3[depth]), bv, cv3);
            }
            _mm256_storeu_pd(c0 + column, cv0);
            _mm256_storeu_pd(c1 + column, cv1);
            _mm256_storeu_pd(c2 + column, cv2);
            _mm256_storeu_pd(c3 + column, cv3);
        }

        for (; column < N; ++column, ++lane) {
            double cv0 = 0.0;
            double cv1 = 0.0;
            double cv2 = 0.0;
            double cv3 = 0.0;
            for (size_t depth = 0; depth < N; ++depth) {
                const double b =
                    finalPanel[depth * packedColumnTile + lane];
                cv0 += a0[depth] * b;
                cv1 += a1[depth] * b;
                cv2 += a2[depth] * b;
                cv3 += a3[depth] * b;
            }
            c0[column] = cv0;
            c1[column] = cv1;
            c2[column] = cv2;
            c3[column] = cv3;
        }
    }

    // The same register accumulation keeps the final zero-to-three rows fast.
    for (; row < localRows; ++row) {
        const double* const a = localA.data() + row * N;
        double* const c = localC.data() + row * N;
        size_t column = 0;

        for (; column + columnTile <= N; column += columnTile) {
            __m256d cv0 = _mm256_setzero_pd();
            __m256d cv1 = _mm256_setzero_pd();
            __m256d cv2 = _mm256_setzero_pd();
            for (size_t depth = 0; depth < N; ++depth) {
                const double* const b =
                    B.data() + column * N + depth * packedColumnTile;
                const __m256d av = _mm256_set1_pd(a[depth]);
                cv0 = _mm256_fmadd_pd(
                    av, _mm256_loadu_pd(b + 0 * vectorWidth), cv0);
                cv1 = _mm256_fmadd_pd(
                    av, _mm256_loadu_pd(b + 1 * vectorWidth), cv1);
                cv2 = _mm256_fmadd_pd(
                    av, _mm256_loadu_pd(b + 2 * vectorWidth), cv2);
            }
            _mm256_storeu_pd(c + column + 0 * vectorWidth, cv0);
            _mm256_storeu_pd(c + column + 1 * vectorWidth, cv1);
            _mm256_storeu_pd(c + column + 2 * vectorWidth, cv2);
        }

        const double* const finalPanel = B.data() + column * N;
        size_t lane = 0;
        for (; column + vectorWidth <= N;
             column += vectorWidth, lane += vectorWidth) {
            __m256d cv = _mm256_setzero_pd();
            for (size_t depth = 0; depth < N; ++depth) {
                cv = _mm256_fmadd_pd(
                    _mm256_set1_pd(a[depth]),
                    _mm256_loadu_pd(finalPanel +
                                    depth * packedColumnTile + lane),
                    cv);
            }
            _mm256_storeu_pd(c + column, cv);
        }

        for (; column < N; ++column, ++lane) {
            double cv = 0.0;
            for (size_t depth = 0; depth < N; ++depth) {
                cv += a[depth] *
                      finalPanel[depth * packedColumnTile + lane];
            }
            c[column] = cv;
        }
    }
#else
    // Portable fallback: consume the same packed panels and expose independent
    // columns to the compiler's vectorizer.
    std::fill(localC.begin(), localC.end(), 0.0);

    for (size_t columnBegin = 0; columnBegin < N;
         columnBegin += packedColumnTile) {
        const size_t panelWidth =
            std::min(packedColumnTile, N - columnBegin);
        const double* const panel = B.data() + columnBegin * N;
        for (size_t fallbackRow = 0; fallbackRow < localRows;
             ++fallbackRow) {
            double* const c = localC.data() + fallbackRow * N;
            const double* const a = localA.data() + fallbackRow * N;
            for (size_t depth = 0; depth < N; ++depth) {
                const double* const b =
                    panel + depth * packedColumnTile;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
                for (size_t lane = 0; lane < panelWidth; ++lane) {
                    c[columnBegin + lane] += a[depth] * b[lane];
                }
            }
        }
    }
#endif
}

bool validateLocalResult(const std::vector<double>& localA,
                         const std::vector<double>& B,
                         const std::vector<double>& localC, const size_t N,
                         const RowPartition partition) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t rowPoint : checkPoints) {
        const size_t globalRow = rowPoint % N;
        if (globalRow < partition.firstRow ||
            globalRow >= partition.firstRow + partition.rowCount) {
            continue;
        }

        const size_t localRow = globalRow - partition.firstRow;
        for (const size_t columnPoint : checkPoints) {
            const size_t column = columnPoint % N;
            double expected = 0.0;
            for (size_t depth = 0; depth < N; ++depth) {
                expected += localA[localRow * N + depth] *
                            packedBValue(B, N, depth, column);
            }

            const double actual = localC[localRow * N + column];
            const double relativeError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relativeError > 1e-6) {
                return false;
            }
        }
    }

    return true;
}

void gatherResult(const std::vector<double>& localC,
                  std::vector<double>& fullC, const size_t N, const int rank,
                  const int processCount, MPI_Comm communicator) {
    constexpr int gatherTag = 101;
    constexpr size_t maxChunk =
        static_cast<size_t>(std::numeric_limits<int>::max());

    // Gatherv is substantially faster for normal-sized benchmark matrices.
    // Fall back to chunked point-to-point transfers only when its int counts or
    // displacements could overflow.
    if (N * N <= maxChunk) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            receiveCounts.resize(static_cast<size_t>(processCount));
            displacements.resize(static_cast<size_t>(processCount));
            for (int source = 0; source < processCount; ++source) {
                const RowPartition sourcePartition =
                    partitionRows(N, source, processCount);
                receiveCounts[static_cast<size_t>(source)] = static_cast<int>(
                    sourcePartition.rowCount * N);
                displacements[static_cast<size_t>(source)] = static_cast<int>(
                    sourcePartition.firstRow * N);
            }
        }

        MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    communicator);
        return;
    }

    if (rank == 0) {
        std::copy(localC.begin(), localC.end(), fullC.begin());
        for (int source = 1; source < processCount; ++source) {
            const RowPartition sourcePartition =
                partitionRows(N, source, processCount);
            const size_t resultOffset = sourcePartition.firstRow * N;
            const size_t elementCount = sourcePartition.rowCount * N;

            for (size_t offset = 0; offset < elementCount;) {
                const int chunk = static_cast<int>(
                    std::min(maxChunk, elementCount - offset));
                MPI_Recv(fullC.data() + resultOffset + offset, chunk,
                         MPI_DOUBLE, source, gatherTag, communicator,
                         MPI_STATUS_IGNORE);
                offset += static_cast<size_t>(chunk);
            }
        }
    } else {
        for (size_t offset = 0; offset < localC.size();) {
            const int chunk = static_cast<int>(
                std::min(maxChunk, localC.size() - offset));
            MPI_Send(localC.data() + offset, chunk, MPI_DOUBLE, 0, gatherTag,
                     communicator);
            offset += static_cast<size_t>(chunk);
        }
    }
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf(
        "  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-n") == 0 &&
            argument + 1 < argc) {
            const char* const value = argv[++argument];
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '\0' || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                argumentsValid = false;
            } else {
                N = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[argument]);
            }
            argumentsValid = false;
        }
    }

    if (N > std::numeric_limits<size_t>::max() -
                (packedColumnTile - 1) ||
        N > std::numeric_limits<size_t>::max() / N ||
        N > std::numeric_limits<size_t>::max() / packedColumnExtent(N)) {
        argumentsValid = false;
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
        std::printf("Validation: %s\n",
                    validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    const RowPartition partition = partitionRows(N, rank, processCount);
    std::vector<double> localA(partition.rowCount * N);
    std::vector<double> B(N * packedColumnExtent(N));
    std::vector<double> localC(partition.rowCount * N);

    initMatrixRows(localA, N, partition.firstRow, partition.rowCount);
    if (rank == 0) {
        initPackedMatrixB(B, N);
    }
    broadcastMatrix(B, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
        std::fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(localA, B, localC, N, partition.rowCount);
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMilliseconds =
            static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMilliseconds);

        const double operations = 2.0 * static_cast<double>(N) *
                                  static_cast<double>(N) *
                                  static_cast<double>(N);
        const double gflops = operations / elapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> fullC;
        if (rank == 0) {
            fullC.resize(N * N);
        }
        gatherResult(localC, fullC, N, rank, processCount, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(fullC, "MatrixC");
        }
    }

    int returnCode = 0;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }

        const int localValid =
            validateLocalResult(localA, B, localC, N, partition) ? 1 : 0;
        int globallyValid = 0;
        MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);

        if (rank == 0) {
            if (globallyValid != 0) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
            }
        }
        returnCode = globallyValid != 0 ? 0 : 1;
    }

    MPI_Finalize();
    return returnCode;
}
