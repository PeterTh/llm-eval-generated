#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

static void initRows(std::vector<double>& mat, size_t N, size_t firstRow,
                     size_t rows) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
}

static void initMatrix(std::vector<double>& mat, size_t N) {
    initRows(mat, N, 0, N);
}

// C += A*B.  The i-k-j ordering streams through B and C, while blocking keeps
// their active portions in cache and exposes the innermost loop to SIMD.
static void matrixMultiply(const std::vector<double>& A,
                           const std::vector<double>& B,
                           std::vector<double>& C, size_t rows, size_t N) {
    constexpr size_t BI = 32;
    constexpr size_t BK = 64;
    constexpr size_t BJ = 128;
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t ii = 0; ii < rows; ii += BI) {
        const size_t iEnd = std::min(ii + BI, rows);
        for (size_t kk = 0; kk < N; kk += BK) {
            const size_t kEnd = std::min(kk + BK, N);
            for (size_t jj = 0; jj < N; jj += BJ) {
                const size_t jEnd = std::min(jj + BJ, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    double* const c = C.data() + i * N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double a = A[i * N + k];
                        const double* const b = B.data() + k * N;
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
                        for (size_t j = jj; j < jEnd; ++j)
                            c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

static bool validateLocal(const std::vector<double>& A,
                          const std::vector<double>& B,
                          const std::vector<double>& C, size_t firstRow,
                          size_t rows, size_t N, int rank) {
    const size_t checks = std::min<size_t>(5, N);
    bool valid = true;
    for (size_t i = 0; i < checks; ++i) {
        if (i < firstRow || i >= firstRow + rows) continue;
        const size_t localI = i - firstRow;
        for (size_t j = 0; j < checks; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[localI * N + k] * B[k * N + j];
            const double actual = C[localI * N + j];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (error > 1e-6) {
                std::fprintf(stderr,
                    "Rank %d: validation failed at (%zu, %zu): expected %.10f, "
                    "got %.10f (error: %.10e)\n",
                    rank, i, j, expected, actual, error);
                valid = false;
            }
        }
    }
    return valid;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) parseStatus = 1;
            else N = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else parseStatus = 1;
    }
    if (parseStatus != 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    if (N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is too large.\n");
        MPI_Finalize();
        return 1;
    }

    const size_t base = N / static_cast<size_t>(ranks);
    const size_t extra = N % static_cast<size_t>(ranks);
    const size_t rows = base + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), extra);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("MPI processes: %d\nValidation: %s\n", ranks,
                    validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(N * N), C(rows * N);
    initRows(A, N, firstRow, rows);
    if (rank == 0) initMatrix(B, N);

    // MPI requires int counts; broadcast by rows to support matrices whose
    // total element count exceeds INT_MAX.
    for (size_t i = 0; i < N; ++i)
        MPI_Bcast(B.data() + i * N, static_cast<int>(N), MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, rows, N);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double gflops = elapsed > 0.0 ? 2.0 * static_cast<double>(N) * N * N /
                                                  elapsed / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> fullC;
        if (rank == 0) fullC.resize(N * N);
        if (rank == 0) {
            std::copy(C.begin(), C.end(), fullC.begin());
            for (int source = 1; source < ranks; ++source) {
                const size_t sourceRows = base + (static_cast<size_t>(source) < extra);
                const size_t sourceFirst = static_cast<size_t>(source) * base +
                                           std::min(static_cast<size_t>(source), extra);
                for (size_t i = 0; i < sourceRows; ++i)
                    MPI_Recv(fullC.data() + (sourceFirst + i) * N,
                             static_cast<int>(N), MPI_DOUBLE, source, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            print_results(fullC, "MatrixC");
        } else {
            for (size_t i = 0; i < rows; ++i)
                MPI_Send(C.data() + i * N, static_cast<int>(N), MPI_DOUBLE, 0, 0,
                         MPI_COMM_WORLD);
        }
    }

    int localValid = !validate || validateLocal(A, B, C, firstRow, rows, N, rank);
    int globallyValid = 0;
    MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    if (validate && rank == 0)
        std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");

    MPI_Finalize();
    return globallyValid ? 0 : 1;
}
