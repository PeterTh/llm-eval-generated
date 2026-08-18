#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N, const size_t firstRow = 0) {
    const size_t localRows = mat.size() / N;
    for (size_t localI = 0; localI < localRows; ++localI) {
        const size_t i = firstRow + localI;
        for (size_t j = 0; j < N; ++j) {
            mat[localI * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// A and C contain a contiguous row block owned by this rank; B is replicated.
// The blocking keeps an A/C tile and the reused B tile in cache, while the
// innermost loop is a contiguous SAXPY-style update that compilers vectorize.
// kk increases monotonically, so every C element receives its products in the
// same k order as the original scalar dot product.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    constexpr size_t rowBlock = 32;
    constexpr size_t kBlock = 64;
    constexpr size_t columnBlock = 128;

    const size_t localRows = C.size() / N;
    const double* const a = A.data();
    const double* const b = B.data();
    double* const c = C.data();

    for (size_t ii = 0; ii < localRows; ii += rowBlock) {
        const size_t iEnd = std::min(ii + rowBlock, localRows);
        for (size_t kk = 0; kk < N; kk += kBlock) {
            const size_t kEnd = std::min(kk + kBlock, N);
            for (size_t jj = 0; jj < N; jj += columnBlock) {
                const size_t jEnd = std::min(jj + columnBlock, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    const double* const aRow = a + i * N;
                    double* const cRow = c + i * N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double aValue = aRow[k];
                        const double* const bRow = b + k * N;
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += aValue * bRow[j];
                        }
                    }
                }
            }
        }
    }
}

struct ValidationFailure {
    int order = 25;
    double expected = 0.0;
    double actual = 0.0;
    double relError = 0.0;
};

// Each rank validates only checkpoints in the rows it owns.  The caller uses
// the checkpoint order to select and report the first failure globally.
ValidationFailure validateLocalResult(const std::vector<double>& A,
                                      const std::vector<double>& B,
                                      const std::vector<double>& C,
                                      const size_t N,
                                      const size_t firstRow) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            if (i < firstRow || i >= firstRow + C.size() / N) {
                continue;
            }

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[(i - firstRow) * N + k] * B[k * N + j];
            }

            const double actual = C[(i - firstRow) * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                return {static_cast<int>(pi * 5 + pj), expected, actual, relError};
            }
        }
    }

    return {};
}

size_t firstRowForRank(const int rank, const int worldSize, const size_t N) {
    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    return static_cast<size_t>(rank) * baseRows +
           std::min(static_cast<size_t>(rank), extraRows);
}

int ownerOfRow(const size_t row, const int worldSize, const size_t N) {
    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t extendedRows = extraRows * (baseRows + 1);
    if (row < extendedRows) {
        return static_cast<int>(row / (baseRows + 1));
    }
    return static_cast<int>(extraRows + (row - extendedRows) / baseRows);
}

bool gatherMatrix(const std::vector<double>& localC, std::vector<double>& globalC,
                  const size_t N, const int rank, const int worldSize) {
    if (N != 0 && N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0) {
            printf("Matrix is too large for MPI_Gatherv result output\n");
        }
        return false;
    }

    std::vector<int> receiveCounts(worldSize);
    std::vector<int> displacements(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        const size_t firstRow = firstRowForRank(process, worldSize, N);
        const size_t nextFirstRow = firstRowForRank(process + 1, worldSize, N);
        receiveCounts[process] = static_cast<int>((nextFirstRow - firstRow) * N);
        displacements[process] = static_cast<int>(firstRow * N);
    }

    if (rank == 0) {
        globalC.resize(N * N);
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                rank == 0 ? globalC.data() : nullptr,
                receiveCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (N == 0) {
        if (rank == 0) {
            printf("Matrix size must be greater than zero\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t firstRow = firstRowForRank(rank, worldSize, N);
    const size_t nextFirstRow = firstRowForRank(rank + 1, worldSize, N);
    const size_t localRows = nextFirstRow - firstRow;

    // A and C are distributed by contiguous rows. B is replicated because it
    // is read-only and this removes communication from the timed hot path.
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N, 0.0);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N, firstRow);
    initMatrix(B, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double gflops = (2.0 * N * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> globalC;
        if (gatherMatrix(C, globalC, N, rank, worldSize)) {
            if (rank == 0) {
                print_results(globalC, "MatrixC");
            }
        } else {
            exitCode = 1;
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const ValidationFailure localFailure = validateLocalResult(A, B, C, N, firstRow);
        int firstFailure = 25;
        MPI_Allreduce(&localFailure.order, &firstFailure, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (firstFailure == 25) {
            if (rank == 0) {
                printf("Validation: PASSED\n");
            }
        } else {
            constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
            const size_t i = checkPoints[firstFailure / 5] % N;
            const size_t j = checkPoints[firstFailure % 5] % N;
            const int owner = ownerOfRow(i, worldSize, N);
            double details[] = {localFailure.expected, localFailure.actual, localFailure.relError};
            MPI_Bcast(details, 3, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, details[0], details[1], details[2]);
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
