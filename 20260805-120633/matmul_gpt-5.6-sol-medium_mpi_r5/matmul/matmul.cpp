#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct RowRange {
    size_t first;
    size_t count;
};

// Assign contiguous rows.  The first ranks receive one extra row when N is not
// divisible by the number of ranks.
RowRange rowsForRank(const size_t N, const int rank, const int ranks) noexcept {
    const size_t base = N / static_cast<size_t>(ranks);
    const size_t extra = N % static_cast<size_t>(ranks);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, extra), base + (r < extra ? 1U : 0U)};
}

void initLocalA(std::vector<double>& A, const size_t N,
                const RowRange rows) {
    for (size_t localI = 0; localI < rows.count; ++localI) {
        const size_t globalI = rows.first + localI;
        for (size_t j = 0; j < N; ++j) {
            A[localI * N + j] = getPseudoRndValue(N, globalI, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// B and C are traversed in contiguous column tiles.  This both exposes a wide
// SIMD loop and retains a tile of C in cache while k is traversed in the same
// order as the original implementation.
void matrixMultiply(const std::vector<double>& A,
                    const std::vector<double>& B,
                    std::vector<double>& C, const size_t N,
                    const size_t localRows) {
    constexpr size_t columnTile = 128;
    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double* __restrict c = C.data();

    std::fill(C.begin(), C.end(), 0.0);
    for (size_t jj = 0; jj < N; jj += columnTile) {
        const size_t jEnd = std::min(jj + columnTile, N);
        for (size_t i = 0; i < localRows; ++i) {
            double* __restrict cRow = c + i * N;
            const double* __restrict aRow = a + i * N;
            for (size_t k = 0; k < N; ++k) {
                const double aik = aRow[k];
                const double* __restrict bRow = b + k * N;
                for (size_t j = jj; j < jEnd; ++j) {
                    cRow[j] += aik * bRow[j];
                }
            }
        }
    }
}

bool validateLocalResult(const std::vector<double>& localC, const size_t N,
                         const RowRange rows, const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;

    for (const size_t pi : checkPoints) {
        const size_t i = pi % N;
        if (i < rows.first || i >= rows.first + rows.count) {
            continue;
        }
        for (const size_t pj : checkPoints) {
            const size_t j = pj % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) *
                            getPseudoRndValue(N, k, j);
            }

            const double actual = localC[(i - rows.first) * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::fprintf(stderr,
                             "Rank %d: validation failed at (%zu, %zu): "
                             "expected %.10f, got %.10f (error: %.10e)\n",
                             rank, i, j, expected, actual, relError);
                valid = false;
            }
        }
    }
    return valid;
}

// MPI's traditional point-to-point count is an int.  Chunk transfers so the
// result path also remains correct for matrices whose element count exceeds
// INT_MAX.
void transferDoubles(const double* sendBuffer, double* receiveBuffer,
                     size_t count, const int peer, const bool sending) {
    constexpr size_t maxChunk =
        static_cast<size_t>(std::numeric_limits<int>::max());
    size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
        if (sending) {
            MPI_Send(sendBuffer + offset, chunk, MPI_DOUBLE, peer, 0,
                     MPI_COMM_WORLD);
        } else {
            MPI_Recv(receiveBuffer + offset, chunk, MPI_DOUBLE, peer, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        offset += static_cast<size_t>(chunk);
    }
}

std::vector<double> gatherResult(const std::vector<double>& localC,
                                 const size_t N, const RowRange localRows,
                                 const int rank, const int ranks) {
    if (rank != 0) {
        transferDoubles(localC.data(), nullptr, localC.size(), 0, true);
        return {};
    }

    std::vector<double> result(N * N);
    std::copy(localC.begin(), localC.end(),
              result.begin() + localRows.first * N);
    for (int source = 1; source < ranks; ++source) {
        const RowRange sourceRows = rowsForRank(N, source, ranks);
        transferDoubles(nullptr, result.data() + sourceRows.first * N,
                        sourceRows.count * N, source, false);
    }
    return result;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
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
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            argumentsValid = end != argv[i] && *end == '\0' && value > 0 &&
                             value <= std::numeric_limits<size_t>::max();
            if (argumentsValid) {
                N = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (rank == 0 && (showHelp || !argumentsValid)) {
        printUsage(argv[0]);
    }
    if (showHelp || !argumentsValid) {
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    const RowRange localRows = rowsForRank(N, rank, ranks);
    std::vector<double> A(localRows.count * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows.count * N);
    initLocalA(A, N, localRows);
    initMatrix(B, N);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, localRows.count);
    const double localSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        const long elapsedMilliseconds =
            static_cast<long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", elapsedMilliseconds);
        const long double n = static_cast<long double>(N);
        const long double operations = 2.0L * n * n * n;
        const double gflops = elapsedSeconds > 0.0
                                  ? static_cast<double>(operations /
                                                        elapsedSeconds / 1.0e9L)
                                  : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> result =
            gatherResult(C, N, localRows, rank, ranks);
        if (rank == 0) {
            print_results(result, "MatrixC");
        }
    }

    int localValid = 1;
    if (validate) {
        localValid = validateLocalResult(C, N, localRows, rank) ? 1 : 0;
    }
    int globallyValid = 1;
    MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");
    }

    MPI_Finalize();
    return globallyValid ? 0 : 1;
}
