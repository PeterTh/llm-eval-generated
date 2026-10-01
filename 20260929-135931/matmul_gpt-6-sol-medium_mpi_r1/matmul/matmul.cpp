#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initRows(std::vector<double>& A, size_t N, size_t firstRow, size_t rows) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
}

void initMatrix(std::vector<double>& B, size_t N) {
    // Transpose B so each dot product reads it contiguously.
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            B[j * N + i] = getPseudoRndValue(N, i, j);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += A[i * N + k] * B[j * N + k];
            C[i * N + j] = sum;
        }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N, size_t firstRow, size_t rows) {
    for (size_t i = 0; i < std::min(size_t{5}, N); ++i) {
        if (i < firstRow || i - firstRow >= rows) continue;
        const size_t localRow = i - firstRow;
        for (size_t j = 0; j < std::min(size_t{5}, N); ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[localRow * N + k] * B[j * N + k];
            const double actual = C[localRow * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6 || !std::isfinite(actual)) {
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

// MPI counts are int, so transfer large partitions in bounded chunks.
void gatherResult(const std::vector<double>& localC, std::vector<double>& result,
                  size_t N, int rank, int ranks) {
    if (rank == 0) {
        std::copy(localC.begin(), localC.end(), result.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t first = size_t(source) * (N / ranks) +
                                 std::min(size_t(source), N % ranks);
            const size_t rows = N / ranks + (size_t(source) < N % ranks);
            size_t remaining = rows * N;
            double* dst = result.data() + first * N;
            while (remaining) {
                const int count = static_cast<int>(std::min(remaining, size_t{INT_MAX}));
                MPI_Recv(dst, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                dst += count;
                remaining -= count;
            }
        }
    } else {
        size_t remaining = localC.size();
        const double* src = localC.data();
        while (remaining) {
            const int count = static_cast<int>(std::min(remaining, size_t{INT_MAX}));
            MPI_Send(src, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            src += count;
            remaining -= count;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || end == value || *end || value[0] == '-' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                exitCode = 1;
                break;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            exitCode = 1;
            break;
        }
    }
    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double)) exitCode = 1;
    if (exitCode) {
        if (rank == 0) {
            printf("Invalid option or matrix size.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    const size_t firstRow = size_t(rank) * (N / ranks) +
                            std::min(size_t(rank), N % ranks);
    const size_t rows = N / ranks + (size_t(rank) < N % ranks);
    std::vector<double> A(rows * N), B(rows ? N * N : 0), C(rows * N, 0.0);
    initRows(A, N, firstRow, rows);
    if (rows) initMatrix(B, N);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }

    if (printResults) {
        std::vector<double> result;
        if (rank == 0) result.resize(N * N);
        gatherResult(C, result, N, rank, ranks);
        if (rank == 0) print_results(result, "MatrixC");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(A, B, C, N, firstRow, rows) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        if (!valid) exitCode = 1;
    }
    MPI_Finalize();
    return exitCode;
}
