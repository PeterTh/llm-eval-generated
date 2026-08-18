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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t firstRow) {
    const size_t localRows = mat.size() / N;
    for (size_t i = 0; i < localRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

// B is kept row-major.  The i-k-j ordering streams B and C contiguously while
// retaining the original increasing-k accumulation order for every C element.
void matrixMultiplyLocal(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N) {
    const size_t localRows = A.size() / N;
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

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } status = 1; }
    }
    if (status || N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0 && !status) printf("Matrix size must be non-zero and fit MPI counts.\n");
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t extraRows = N % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows + std::min(static_cast<size_t>(rank), extraRows);

    std::vector<int> recvCounts, displacements;
    if (rank == 0) {
        recvCounts.resize(ranks); displacements.resize(ranks);
        size_t offset = 0;
        for (int r = 0; r < ranks; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
            recvCounts[r] = static_cast<int>(rows * N);
            displacements[r] = static_cast<int>(offset);
            offset += rows * N;
        }
        printf("Matrix Multiplication Benchmark (MPI ranks: %d)\nMatrix size: %zu x %zu\nValidation: %s\n",
               ranks, N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(localRows * N), B(N * N), C(localRows * N);
    initMatrixRows(A, N, firstRow);
    initMatrix(B, N);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyLocal(A, B, C, N);
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> fullC;
    if (rank == 0 && (validate || printResults)) fullC.resize(N * N);
    if (validate || printResults) {
        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr, rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double gflops = elapsedSeconds > 0.0 ? (2.0 * N * N * N) / elapsedSeconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(fullC, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> fullA(N * N), fullB(N * N);
            initMatrix(fullA, N); initMatrix(fullB, N);
            status = validateResult(fullA, fullB, fullC, N) ? 0 : 1;
            printf("Validation: %s\n", status == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
