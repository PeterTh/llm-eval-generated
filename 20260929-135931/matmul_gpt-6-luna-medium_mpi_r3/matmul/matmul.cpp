#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, i, j);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[i * N + k] * B[k * N + j];
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long parsed = strtol(argv[++i], nullptr, 10);
            if (parsed <= 0) parseStatus = 1;
            else N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
        }
    }
    int anyParseError = 0;
    MPI_Allreduce(&parseStatus, &anyParseError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyParseError) { MPI_Finalize(); return 1; }

    if (N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0) fprintf(stderr, "Matrix is too large for MPI collective counts\n");
        MPI_Finalize();
        return 1;
    }
    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t remainder = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), remainder);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", worldSize);
    }

    std::vector<double> B(N * N), localA(localRows * N), localC(localRows * N, 0.0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            B[i * N + j] = getPseudoRndValue(N, i, j);
    for (size_t i = 0; i < localRows; ++i)
        for (size_t j = 0; j < N; ++j)
            localA[i * N + j] = getPseudoRndValue(N, firstRow + i, j);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    // i-k-j ordering reuses each A value and streams contiguous rows of B and C.
    for (size_t i = 0; i < localRows; ++i) {
        double* cRow = localC.data() + i * N;
        const double* aRow = localA.data() + i * N;
        for (size_t k = 0; k < N; ++k) {
            const double aik = aRow[k];
            const double* bRow = B.data() + k * N;
            for (size_t j = 0; j < N; ++j) cRow[j] += aik * bRow[j];
        }
    }

    std::vector<double> C;
    std::vector<int> counts(static_cast<size_t>(worldSize)), displacements(static_cast<size_t>(worldSize));
    for (int p = 0; p < worldSize; ++p) {
        const size_t pRows = baseRows + (static_cast<size_t>(p) < remainder ? 1 : 0);
        const size_t pFirst = static_cast<size_t>(p) * baseRows + std::min(static_cast<size_t>(p), remainder);
        counts[static_cast<size_t>(p)] = static_cast<int>(pRows * N);
        displacements[static_cast<size_t>(p)] = static_cast<int>(pFirst * N);
    }
    if (rank == 0) C.resize(N * N);
    MPI_Gatherv(localC.data(), counts[static_cast<size_t>(rank)], MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        const double gflops = (2.0 * static_cast<double>(N) * N * N) / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A(N * N);
            initMatrix(A, N);
            const bool valid = validateResult(A, B, C, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
