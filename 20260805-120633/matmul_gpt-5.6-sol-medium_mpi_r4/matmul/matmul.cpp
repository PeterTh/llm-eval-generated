#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
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

RowRange rowsForRank(const size_t N, const int rank, const int processes) {
    const size_t base = N / static_cast<size_t>(processes);
    const size_t extra = N % static_cast<size_t>(processes);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, extra), base + (r < extra ? 1 : 0)};
}

void initRows(std::vector<double>& mat, const size_t N,
              const RowRange rows) {
    for (size_t local_i = 0; local_i < rows.count; ++local_i) {
        const size_t global_i = rows.first + local_i;
        for (size_t j = 0; j < N; ++j) {
            mat[local_i * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initRows(mat, N, {0, N});
}

// MPI collectives traditionally take an int element count.  Chunking keeps the
// implementation correct for matrices whose element count is larger than that.
void broadcastMatrix(std::vector<double>& matrix, const int root,
                     MPI_Comm comm) {
    size_t offset = 0;
    while (offset < matrix.size()) {
        const int count = static_cast<int>(
            std::min(matrix.size() - offset, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(matrix.data() + offset, count, MPI_DOUBLE, root, comm);
        offset += static_cast<size_t>(count);
    }
}

// A cache-blocked GEMM.  Rows of A and C are local to this MPI rank, while B
// is replicated.  The innermost loop is contiguous and readily vectorized.
void matrixMultiply(const std::vector<double>& A,
                    const std::vector<double>& B,
                    std::vector<double>& C, const size_t localRows,
                    const size_t N) {
    constexpr size_t rowBlock = 32;
    constexpr size_t columnBlock = 128;
    constexpr size_t innerBlock = 128;

    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double* __restrict c = C.data();

    std::fill(C.begin(), C.end(), 0.0);
    for (size_t ii = 0; ii < localRows; ii += rowBlock) {
        const size_t iEnd = std::min(ii + rowBlock, localRows);
        for (size_t jj = 0; jj < N; jj += columnBlock) {
            const size_t jEnd = std::min(jj + columnBlock, N);
            for (size_t kk = 0; kk < N; kk += innerBlock) {
                const size_t kEnd = std::min(kk + innerBlock, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    double* const cRow = c + i * N;
                    const double* const aRow = a + i * N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double aik = aRow[k];
                        const double* const bRow = b + k * N;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += aik * bRow[j];
                        }
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& localA,
                    const std::vector<double>& B,
                    const std::vector<double>& localC, const size_t N,
                    const RowRange rows, MPI_Comm comm) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    int localValid = 1;

    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = checkPoints[pi] % N;
        if (i < rows.first || i >= rows.first + rows.count) {
            continue;
        }
        const size_t local_i = i - rows.first;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += localA[local_i * N + k] * B[k * N + j];
            }
            const double actual = localC[local_i * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                localValid = 0;
            }
        }
    }

    int globallyValid = 0;
    MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_MIN, comm);
    return globallyValid != 0;
}

// Gathering is deliberately deferred until -r is requested, so normal runs
// retain distributed memory use and do not pay an unnecessary communication
// cost.  Row chunks avoid MPI_Gatherv's int displacement limit.
void gatherResult(const std::vector<double>& localC,
                  std::vector<double>& fullC, const size_t N,
                  const RowRange localRows, const int rank,
                  const int processes, MPI_Comm comm) {
    const size_t totalElements = N * N;
    const size_t localFirst = localRows.first * N;
    const size_t localEnd = localFirst + localRows.count * N;
    for (size_t chunkFirst = 0; chunkFirst < totalElements;) {
        const size_t chunkElements = std::min(
            totalElements - chunkFirst, static_cast<size_t>(INT_MAX));
        const size_t chunkEnd = chunkFirst + chunkElements;
        const size_t sendFirst = std::max(chunkFirst, localFirst);
        const size_t sendEnd = std::min(chunkEnd, localEnd);
        const size_t sendElements =
            sendEnd > sendFirst ? sendEnd - sendFirst : 0;
        const int sendCount = static_cast<int>(sendElements);
        const double* sendBuffer = localC.data();
        if (sendElements != 0) {
            sendBuffer += sendFirst - localFirst;
        }

        std::vector<int> counts(static_cast<size_t>(processes));
        std::vector<int> displacements(static_cast<size_t>(processes));
        for (int p = 0; p < processes; ++p) {
            const RowRange other = rowsForRank(N, p, processes);
            const size_t otherFirst = other.first * N;
            const size_t otherEnd = otherFirst + other.count * N;
            const size_t first = std::max(chunkFirst, otherFirst);
            const size_t end = std::min(chunkEnd, otherEnd);
            const size_t count = end > first ? end - first : 0;
            counts[static_cast<size_t>(p)] =
                static_cast<int>(count);
            displacements[static_cast<size_t>(p)] =
                count == 0 ? 0 : static_cast<int>(first - chunkFirst);
        }

        double* receiveBuffer = rank == 0
                                    ? fullC.data() + chunkFirst
                                    : nullptr;
        MPI_Gatherv(sendBuffer, sendCount, MPI_DOUBLE, receiveBuffer,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, comm);
        chunkFirst = chunkEnd;
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int processes = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &processes);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                errno = 0;
                char* end = nullptr;
                const unsigned long long value = strtoull(argv[++i], &end, 10);
                if (errno != 0 || end == argv[i] || *end != '\0' || value == 0 ||
                    value > std::numeric_limits<size_t>::max()) {
                    fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                    argumentsValid = false;
                } else {
                    N = static_cast<size_t>(value);
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                argumentsValid = false;
            }
        }
        if (showHelp || !argumentsValid) {
            printUsage(argv[0]);
        }
    }

    unsigned long long config[] = {
        static_cast<unsigned long long>(N),
        static_cast<unsigned long long>(validate),
        static_cast<unsigned long long>(printResults),
        static_cast<unsigned long long>(showHelp),
        static_cast<unsigned long long>(argumentsValid)};
    MPI_Bcast(config, 5, MPI_UNSIGNED_LONG_LONG, 0, comm);
    N = static_cast<size_t>(config[0]);
    validate = config[1] != 0;
    printResults = config[2] != 0;
    showHelp = config[3] != 0;
    argumentsValid = config[4] != 0;
    if (showHelp || !argumentsValid) {
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size is too large.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const RowRange rows = rowsForRank(N, rank, processes);
    std::vector<double> localA(rows.count * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(rows.count * N);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", processes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
        initMatrix(B, N);
    }
    initRows(localA, N, rows);
    broadcastMatrix(B, 0, comm);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    matrixMultiply(localA, B, localC, rows.count, N);
    const double localDuration = MPI_Wtime() - start;

    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double n = static_cast<double>(N);
        const double gflops = (2.0 * n * n * n) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> fullC;
        if (rank == 0) {
            fullC.resize(N * N);
        }
        gatherResult(localC, fullC, N, rows, rank, processes, comm);
        if (rank == 0) {
            print_results(fullC, "MatrixC");
        }
    }

    int returnCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(localA, B, localC, N, rows, comm);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        returnCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return returnCode;
}
