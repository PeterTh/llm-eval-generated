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
        double* const row = mat.data() + localRow * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initRows(mat, N, 0, N);
}

// Each process owns a contiguous set of result rows.  The i-k-j order gives
// contiguous accesses to B and C; blocking keeps their working set in cache.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const size_t localRows) {
    constexpr size_t blockSize = 64;
    std::fill(C.begin(), C.end(), 0.0);

    for (size_t ii = 0; ii < localRows; ii += blockSize) {
        const size_t iEnd = std::min(ii + blockSize, localRows);
        for (size_t kk = 0; kk < N; kk += blockSize) {
            const size_t kEnd = std::min(kk + blockSize, N);
            for (size_t jj = 0; jj < N; jj += blockSize) {
                const size_t jEnd = std::min(jj + blockSize, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    const double* const aRow = A.data() + i * N;
                    double* const cRow = C.data() + i * N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double a = aRow[k];
                        const double* const bRow = B.data() + k * N;
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += a * bRow[j];
                        }
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (const size_t iPoint : checkPoints) {
        for (const size_t jPoint : checkPoints) {
            const size_t i = iPoint % N;
            const size_t j = jPoint % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
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

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
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
            MPI_Finalize();
            return 1;
        }
    }

    if (N == 0 || N > std::numeric_limits<size_t>::max() / N ||
        N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Matrix size is invalid or too large for MPI counts: %zu\n", N);
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(processCount);
    const size_t extraRows = N % static_cast<size_t>(processCount);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = baseRows * static_cast<size_t>(rank) +
                            std::min(static_cast<size_t>(rank), extraRows);

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    if (rank == 0 && (validate || printResults)) {
        receiveCounts.resize(processCount);
        displacements.resize(processCount);
        size_t offset = 0;
        for (int p = 0; p < processCount; ++p) {
            const size_t rows = baseRows + (static_cast<size_t>(p) < extraRows ? 1 : 0);
            receiveCounts[p] = static_cast<int>(rows * N);
            displacements[p] = static_cast<int>(offset);
            offset += rows * N;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", processCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);
    initRows(A, N, firstRow, localRows);
    if (rank == 0) initMatrix(B, N);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, localRows);
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> fullC;
    if (rank == 0 && (validate || printResults)) fullC.resize(N * N);
    if (validate || printResults) {
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsedSeconds * 1000.0));
        const double gflops = elapsedSeconds > 0.0 ?
            (2.0 * static_cast<double>(N) * N * N) / elapsedSeconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(fullC, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            if (validateResult(B, fullC, N)) {
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
