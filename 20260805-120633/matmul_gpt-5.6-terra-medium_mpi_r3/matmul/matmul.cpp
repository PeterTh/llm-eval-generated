#include <algorithm>
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
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initRows(std::vector<double>& mat, const size_t N, const size_t firstRow,
              const size_t rowCount) {
    for (size_t localRow = 0; localRow < rowCount; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        for (size_t j = 0; j < N; ++j) {
            mat[localRow * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

// B is transposed once, so the innermost loop reads both operands contiguously.
// The summation order remains increasing k, matching the original computation.
void matrixMultiply(const std::vector<double>& localA, const std::vector<double>& transposedB,
                    std::vector<double>& localC, const size_t N, const size_t localRows) {
    for (size_t i = 0; i < localRows; ++i) {
        const double* const aRow = localA.data() + i * N;
        double* const cRow = localC.data() + i * N;
        for (size_t j = 0; j < N; ++j) {
            const double* const bRow = transposedB.data() + j * N;
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += aRow[k] * bRow[k];
            }
            cRow[j] = sum;
        }
    }
}

bool validateLocalResult(const std::vector<double>& localA, const std::vector<double>& transposedB,
                         const std::vector<double>& localC, const size_t N,
                         const size_t firstRow, const size_t localRows) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t checkedRow : checkPoints) {
        const size_t globalRow = checkedRow % N;
        if (globalRow < firstRow || globalRow >= firstRow + localRows) {
            continue;
        }
        const size_t localRow = globalRow - firstRow;
        for (const size_t checkedColumn : checkPoints) {
            const size_t j = checkedColumn % N;
            const double* const aRow = localA.data() + localRow * N;
            const double* const bRow = transposedB.data() + j * N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += aRow[k] * bRow[k];
            }

            const double actual = localC[localRow * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            globalRow, j, expected, actual, relError);
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = std::strtoull(argv[++i], nullptr, 10);
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
            MPI_Finalize();
            return 1;
        }
    }

    // MPI counts are int; reject sizes that cannot be represented safely.
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0) {
            std::printf("Matrix size is unsupported: %zu\n", N);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t rowsPerRank = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = rowsPerRank + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * rowsPerRank +
                            std::min(static_cast<size_t>(rank), extraRows);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("MPI processes: %d\n", worldSize);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localRows * N);
    std::vector<double> transposedB(N * N);
    std::vector<double> localC(localRows * N);
    initRows(localA, N, firstRow, localRows);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            transposedB[j * N + i] = getPseudoRndValue(N, i, j);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    const double start = MPI_Wtime();
    matrixMultiply(localA, transposedB, localC, N, localRows);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> C;
    if (printResults) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            C.resize(N * N);
            receiveCounts.resize(worldSize);
            displacements.resize(worldSize);
            for (int process = 0; process < worldSize; ++process) {
                const size_t processRows = rowsPerRank +
                    (static_cast<size_t>(process) < extraRows ? 1 : 0);
                const size_t processFirstRow = static_cast<size_t>(process) * rowsPerRank +
                    std::min(static_cast<size_t>(process), extraRows);
                receiveCounts[process] = static_cast<int>(processRows * N);
                displacements[process] = static_cast<int>(processFirstRow * N);
            }
        }
        MPI_Gatherv(localC.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    rank == 0 ? C.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(duration * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double gflops = duration > 0.0 ? (2.0 * N * N * N) / duration / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(C, "MatrixC");
        }
    }

    if (validate) {
        const int localValid = validateLocalResult(localA, transposedB, localC, N, firstRow, localRows) ? 1 : 0;
        int allValid = 0;
        MPI_Reduce(&localValid, &allValid, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
            exitCode = allValid ? 0 : 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
