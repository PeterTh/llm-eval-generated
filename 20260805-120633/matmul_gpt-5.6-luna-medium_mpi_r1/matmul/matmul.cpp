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

void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t firstRow, const size_t rowCount) {
    for (size_t i = 0; i < rowCount; ++i) {
        const size_t globalRow = firstRow + i;
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixRows(mat, N, 0, N);
}

// B is transposed before this routine so each dot product is contiguous.
// The k loop order is unchanged from the original implementation.
void matrixMultiply(const std::vector<double>& A,
                    const std::vector<double>& transposedB,
                    std::vector<double>& C, const size_t rows,
                    const size_t N) {
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t i = 0; i < rows; ++i) {
        const double* a = A.data() + i * N;
        double* c = C.data() + i * N;
        for (size_t j = 0; j < N; ++j) {
            const double* b = transposedB.data() + j * N;
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += a[k] * b[k];
            c[j] = sum;
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
            for (size_t k = 0; k < N; ++k)
                expected += A[i * N + k] * B[k * N + j];
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) /
                                              (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
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

[[noreturn]] void abortAll(const char* message, const int rank) {
    if (rank == 0) std::fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseError = 1;
        }
    }
    if (parseError) {
        MPI_Finalize();
        return 1;
    }

    if (N == 0 || N > std::numeric_limits<size_t>::max() / N)
        abortAll("Matrix size is invalid.", rank);
    const size_t elements = N * N;
    if (elements > static_cast<size_t>(std::numeric_limits<int>::max()))
        abortAll("Matrix is too large for this MPI implementation.", rank);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localElements = localRows * N;
    if (localElements > static_cast<size_t>(std::numeric_limits<int>::max()))
        abortAll("A local matrix block is too large for this MPI implementation.", rank);

    std::vector<double> localA(localElements);
    std::vector<double> B(elements);
    std::vector<double> localC(localElements);
    std::vector<double> fullA;
    std::vector<double> fullC;
    initMatrixRows(localA, N, firstRow, localRows);
    if (rank == 0) {
        initMatrix(B, N);
        if (validate) {
            fullA.resize(elements);
            initMatrix(fullA, N);
        }
        if (printResults || validate) fullC.resize(elements);
    }
    MPI_Bcast(B.data(), static_cast<int>(elements), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    std::vector<double> transposedB(elements);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            transposedB[j * N + i] = B[i * N + j];

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(localA, transposedB, localC, localRows, N);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<int> counts, displacements;
    if (rank == 0) {
        counts.resize(worldSize);
        displacements.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows);
            const size_t offset = (static_cast<size_t>(r) * baseRows +
                                   std::min(static_cast<size_t>(r), extraRows)) * N;
            counts[r] = static_cast<int>(rows * N);
            displacements[r] = static_cast<int>(offset);
        }
    }
    if (printResults || validate) {
        MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const long durationMs = static_cast<long>(seconds * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMs);
        const double gflops = seconds > 0.0
                                  ? (2.0 * static_cast<double>(N) * N * N) /
                                        seconds / 1e9
                                  : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(fullC, "MatrixC");
    }

    int valid = 1;
    if (validate && rank == 0) {
        std::printf("Validating result...\n");
        valid = validateResult(fullA, B, fullC, N) ? 1 : 0;
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
