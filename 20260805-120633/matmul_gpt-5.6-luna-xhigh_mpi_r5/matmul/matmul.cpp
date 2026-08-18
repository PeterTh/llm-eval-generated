#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.  The global row
// index is passed to initMatrixRows so every rank creates the same matrix as
// the original single-process program without communicating A.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrixRows(std::vector<double>& mat, const size_t rows, const size_t N,
                    const size_t firstRow) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

// Multiply local A rows by the complete B matrix.  The k order is identical
// to the original i-j-k kernel, so the floating-point result is unchanged.
// With k as the middle loop, B and each C row are accessed contiguously and
// the innermost loop can be vectorized by the compiler.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t rows, const size_t N) {
    for (size_t i = 0; i < rows; ++i) {
        const double* const aRow = A.data() + i * N;
        double* const cRow = C.data() + i * N;
        std::fill(cRow, cRow + N, 0.0);

        for (size_t k = 0; k < N; ++k) {
            const double aValue = aRow[k];
            const double* const bRow = B.data() + k * N;
            for (size_t j = 0; j < N; ++j) {
                cRow[j] += aValue * bRow[j];
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
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

// MPI point-to-point calls use an int element count on all commonly deployed
// MPI versions.  Chunking keeps the code correct for matrices whose flat
// storage exceeds that limit, while normal benchmark-sized matrices use one
// collective call.
constexpr size_t kMpiChunkElements = 1U << 20;

void broadcastMatrix(std::vector<double>& matrix, MPI_Comm communicator) {
    const size_t elements = matrix.size();
    for (size_t offset = 0; offset < elements; offset += kMpiChunkElements) {
        const size_t remaining = elements - offset;
        const int count = static_cast<int>(std::min(remaining, kMpiChunkElements));
        MPI_Bcast(matrix.data() + offset, count, MPI_DOUBLE, 0, communicator);
    }
}

void gatherMatrix(const std::vector<double>& localC, std::vector<double>& globalC,
                  const size_t N, const size_t firstRow,
                  const int rank, const int worldSize, MPI_Comm communicator) {
    const size_t globalElements = N * N;
    if (rank == 0) {
        globalC.resize(globalElements);
    }

    // Gatherv is the fast path and covers all practical benchmark sizes.  Its
    // counts and displacements are int-sized, so use ordered chunked sends as
    // a fallback for very large allocations.
    const bool canUseGatherv = globalElements <= static_cast<size_t>(std::numeric_limits<int>::max());
    if (canUseGatherv) {
        std::vector<int> receiveCounts(static_cast<size_t>(worldSize));
        std::vector<int> displacements(static_cast<size_t>(worldSize));
        const size_t baseRows = N / static_cast<size_t>(worldSize);
        const size_t remainder = N % static_cast<size_t>(worldSize);

        for (int process = 0; process < worldSize; ++process) {
            const size_t rows = baseRows + (static_cast<size_t>(process) < remainder ? 1 : 0);
            const size_t processFirstRow = static_cast<size_t>(process) * baseRows +
                                           std::min(static_cast<size_t>(process), remainder);
            receiveCounts[static_cast<size_t>(process)] = static_cast<int>(rows * N);
            displacements[static_cast<size_t>(process)] = static_cast<int>(processFirstRow * N);
        }

        MPI_Gatherv(localC.empty() ? nullptr : localC.data(),
                    static_cast<int>(localC.size()), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr, receiveCounts.data(),
                    displacements.data(), MPI_DOUBLE, 0, communicator);
        return;
    }

    if (rank == 0) {
        std::copy(localC.begin(), localC.end(), globalC.begin() + firstRow * N);
        for (int process = 1; process < worldSize; ++process) {
            const size_t rows = N / static_cast<size_t>(worldSize) +
                                (static_cast<size_t>(process) < N % static_cast<size_t>(worldSize) ? 1 : 0);
            const size_t processFirstRow = static_cast<size_t>(process) * (N / static_cast<size_t>(worldSize)) +
                                           std::min(static_cast<size_t>(process), N % static_cast<size_t>(worldSize));
            const size_t elements = rows * N;
            for (size_t offset = 0; offset < elements; offset += kMpiChunkElements) {
                const int count = static_cast<int>(std::min(elements - offset, kMpiChunkElements));
                MPI_Recv(globalC.data() + processFirstRow * N + offset, count, MPI_DOUBLE,
                         process, 0, communicator, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t offset = 0; offset < localC.size(); offset += kMpiChunkElements) {
            const int count = static_cast<int>(std::min(localC.size() - offset, kMpiChunkElements));
            MPI_Send(localC.data() + offset, count, MPI_DOUBLE, 0, 0, communicator);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseResult = 0;

    // Parse once and broadcast the configuration so all ranks follow exactly
    // the same collective communication path.
    unsigned long long nValue = static_cast<unsigned long long>(N);
    int validateValue = 0;
    int printResultsValue = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const long parsedN = std::strtol(argv[++i], nullptr, 10);
                if (parsedN <= 0) {
                    printf("Matrix size must be a positive integer.\n");
                    parseResult = 1;
                    break;
                }
                N = static_cast<size_t>(parsedN);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseResult = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseResult = 1;
                break;
            }
        }

        if (parseResult == 0) {
            if (N > std::numeric_limits<size_t>::max() / N) {
                printf("Matrix size is too large.\n");
                parseResult = 1;
            } else {
                nValue = static_cast<unsigned long long>(N);
                validateValue = validate ? 1 : 0;
                printResultsValue = printResults ? 1 : 0;
            }
        }
    }

    MPI_Bcast(&parseResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseResult != 0) {
        MPI_Finalize();
        return parseResult == 2 ? 0 : 1;
    }

    MPI_Bcast(&nValue, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateValue, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsValue, 1, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(nValue);
    validate = validateValue != 0;
    printResults = printResultsValue != 0;

    const size_t worldSizeAsSize = static_cast<size_t>(worldSize);
    const size_t baseRows = N / worldSizeAsSize;
    const size_t remainder = N % worldSizeAsSize;
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), remainder);
    const size_t matrixElements = N * N;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", worldSize);
        printf("Initializing matrices...\n");
    }

    // A is generated locally from its global row number.  Only B is
    // replicated, which avoids broadcasting or scattering the input A matrix.
    std::vector<double> localA(localRows * N);
    std::vector<double> B(matrixElements);
    std::vector<double> localC(localRows * N);
    initMatrixRows(localA, localRows, N, firstRow);
    if (rank == 0) {
        initMatrix(B, N);
    }
    broadcastMatrix(B, MPI_COMM_WORLD);

    // Retain full A only on rank zero when validation is requested, matching
    // the original validation algorithm without imposing that memory cost on
    // normal distributed benchmark runs.
    std::vector<double> A;
    std::vector<double> C;
    if (rank == 0 && validate) {
        A.resize(matrixElements);
        initMatrix(A, N);
    }
    if (rank == 0 && (validate || printResults)) {
        C.resize(matrixElements);
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(localA, B, localC, localRows, N);
    const double localDuration = MPI_Wtime() - start;

    double durationSeconds = 0.0;
    MPI_Reduce(&localDuration, &durationSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(durationSeconds * 1000.0);
        printf("Computation time: %ld ms\n", durationMilliseconds);

        const double gflops = durationSeconds > 0.0
                                  ? (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                                     static_cast<double>(N)) /
                                        durationSeconds / 1e9
                                  : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    const bool needGlobalResult = validate || printResults;
    if (needGlobalResult) {
        gatherMatrix(localC, C, N, firstRow, rank, worldSize, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
