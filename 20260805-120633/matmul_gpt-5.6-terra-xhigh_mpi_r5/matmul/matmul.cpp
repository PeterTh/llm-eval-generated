#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct RowPartition {
    size_t firstRow;
    size_t rowCount;
};

RowPartition partitionRows(const size_t N, const int rank, const int ranks) noexcept {
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t rowsPerRank = N / rankCount;
    const size_t extraRows = N % rankCount;
    const size_t rowCount = rowsPerRank + (rankIndex < extraRows ? 1 : 0);
    const size_t firstRow = rankIndex * rowsPerRank + std::min(rankIndex, extraRows);
    return {firstRow, rowCount};
}

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t firstRow,
                    const size_t rowCount) {
    for (size_t localRow = 0; localRow < rowCount; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        double* const row = mat.data() + localRow * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixRows(mat, N, 0, N);
}

// Each output element receives its terms in increasing-k order, as in the
// original implementation.  The i/j tiles improve cache reuse of B without
// changing the floating-point accumulation order for any C(i,j).
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t localRows, const size_t N) {
    constexpr size_t I_BLOCK = 32;
    constexpr size_t J_BLOCK = 128;

    const double* const a = A.data();
    const double* const b = B.data();
    double* const c = C.data();

    for (size_t ii = 0; ii < localRows; ii += I_BLOCK) {
        const size_t iEnd = std::min(ii + I_BLOCK, localRows);
        for (size_t jj = 0; jj < N; jj += J_BLOCK) {
            const size_t jEnd = std::min(jj + J_BLOCK, N);
            for (size_t k = 0; k < N; ++k) {
                const double* const bRow = b + k * N + jj;
                for (size_t i = ii; i < iEnd; ++i) {
                    const double aik = a[i * N + k];
                    double* const cRow = c + i * N + jj;
                    for (size_t j = jj; j < jEnd; ++j) {
                        cRow[j - jj] += aik * bRow[j - jj];
                    }
                }
            }
        }
    }
}

// Validate only rows owned by this rank. Together the ranks cover exactly the
// same check points used by the original validation routine.
bool validateLocalResult(const std::vector<double>& A, const std::vector<double>& B,
                         const std::vector<double>& C, const size_t N,
                         const RowPartition partition, const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t pointI : checkPoints) {
        const size_t globalRow = pointI % N;
        if (globalRow < partition.firstRow ||
            globalRow >= partition.firstRow + partition.rowCount) {
            continue;
        }

        const size_t localRow = globalRow - partition.firstRow;
        for (const size_t pointJ : checkPoints) {
            const size_t j = pointJ % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[localRow * N + k] * B[k * N + j];
            }

            const double actual = C[localRow * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed on rank %d at (%zu, %zu): expected %.10f, got %.10f "
                            "(error: %.10e)\n",
                            rank, globalRow, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
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
    int exitCode = 0;

    // Every MPI process receives the same command line, so parsing is local
    // and does not add a startup communication round trip.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
                if (rank == 0) {
                    std::printf("Invalid matrix size: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                exitCode = 1;
                break;
            }
            N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }

    if (exitCode == 0 &&
        (N == 0 || N > std::numeric_limits<size_t>::max() / N ||
         N * N > static_cast<size_t>(std::numeric_limits<int>::max()))) {
        if (rank == 0) {
            std::printf("Matrix size is unsupported: %zu\n", N);
        }
        exitCode = 1;
    }

    int globalExitCode = 0;
    MPI_Allreduce(&exitCode, &globalExitCode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalExitCode != 0) {
        MPI_Finalize();
        return globalExitCode;
    }

    const RowPartition partition = partitionRows(N, rank, ranks);
    const size_t localElements = partition.rowCount * N;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    // A and C are distributed by contiguous row ranges. B is replicated so
    // each rank can complete its rows independently after initialization.
    std::vector<double> A(localElements);
    std::vector<double> B(N * N);
    std::vector<double> C(localElements, 0.0);
    initMatrixRows(A, N, partition.firstRow, partition.rowCount);
    initMatrix(B, N);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, partition.rowCount, N);
    const double localSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const auto durationMs = static_cast<long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMs);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              elapsedSeconds / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        std::vector<double> globalC;
        if (rank == 0) {
            receiveCounts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
            for (int process = 0; process < ranks; ++process) {
                const RowPartition processPartition = partitionRows(N, process, ranks);
                receiveCounts[static_cast<size_t>(process)] =
                    static_cast<int>(processPartition.rowCount * N);
                displacements[static_cast<size_t>(process)] =
                    static_cast<int>(processPartition.firstRow * N);
            }
            globalC.resize(N * N);
        }

        MPI_Gatherv(C.data(), static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(globalC, "MatrixC");
        }
    }

    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const int localValid = validateLocalResult(A, B, C, N, partition, rank) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
        }
        exitCode = valid != 0 ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
