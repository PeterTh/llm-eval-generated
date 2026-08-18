#include <algorithm>
#include <chrono>
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

void initRows(std::vector<double>& mat, const size_t N, const size_t firstRow,
              const size_t rowCount) {
    for (size_t i = 0; i < rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initRows(mat, N, 0, N);
}

// C owns consecutive rows of the product.  Keeping B replicated avoids
// communication in the O(N^3) kernel and gives every rank contiguous B rows.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const size_t localRows) {
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t i = 0; i < localRows; ++i) {
        double* const cRow = C.data() + i * N;
        const double* const aRow = A.data() + i * N;
        for (size_t k = 0; k < N; ++k) {
            const double a = aRow[k];
            const double* const bRow = B.data() + k * N;
            for (size_t j = 0; j < N; ++j) {
                cRow[j] += a * bRow[j];
            }
        }
    }
}

bool validateLocalRows(const std::vector<double>& A, const std::vector<double>& B,
                       const std::vector<double>& C, const size_t N,
                       const size_t firstRow, const size_t localRows, int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t globalI = checkPoints[pi] % N;
        if (globalI < firstRow || globalI >= firstRow + localRows) continue;
        const size_t i = globalI - firstRow;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                printf("Validation failed on rank %d at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       rank, globalI, j, expected, actual, relError);
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } exitCode = 1; }
    }
    if (exitCode || N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0 && N == 0) printf("Matrix size must be positive.\n");
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t remainder = N % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows + std::min(static_cast<size_t>(rank), remainder);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nInitializing matrices...\n",
               N, N, validate ? "enabled" : "disabled");
    }
    std::vector<double> A(localRows * N), B(N * N), C(localRows * N);
    initRows(A, N, firstRow, localRows);
    initMatrix(B, N);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, localRows);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long milliseconds = static_cast<long>(seconds * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double gflops = seconds > 0.0 ? (2.0 * N * N * N) / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < remainder);
            const size_t offset = static_cast<size_t>(r) * baseRows + std::min(static_cast<size_t>(r), remainder);
            counts[r] = static_cast<int>(rows * N);
            displacements[r] = static_cast<int>(offset * N);
        }
        std::vector<double> fullC(rank == 0 ? N * N : 0);
        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE, fullC.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(fullC, "MatrixC");
    }

    if (validate) {
        const int localValid = validateLocalRows(A, B, C, N, firstRow, localRows, rank) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exitCode = valid ? 0 : 1;
    }
    MPI_Finalize();
    return exitCode;
}
