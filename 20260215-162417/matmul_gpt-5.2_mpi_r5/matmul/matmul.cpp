#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static inline void initMatrixRows(std::vector<double>& matRows, const size_t N, const size_t rowStart,
                                  const size_t rowCount) {
    for (size_t ii = 0; ii < rowCount; ++ii) {
        const size_t i = rowStart + ii;
        double* row = matRows.data() + ii * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        double* row = mat.data() + i * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void initMatrixTranspose(std::vector<double>& matT, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            matT[j * N + i] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void matrixMultiplyLocalRows(const std::vector<double>& Arows, const std::vector<double>& BT,
                                           std::vector<double>& Crows, const size_t N, const size_t localRows) {
    const double* __restrict__ bT = BT.data();
    const double* __restrict__ a = Arows.data();
    double* __restrict__ c = Crows.data();

    for (size_t ii = 0; ii < localRows; ++ii) {
        const double* __restrict__ arow = a + ii * N;
        double* __restrict__ crow = c + ii * N;
        for (size_t j = 0; j < N; ++j) {
            const double* __restrict__ brow = bT + j * N;
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += arow[k] * brow[k];
            }
            crow[j] = sum;
        }
    }
}

// Simple validation: compute a single element and compare
static bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                           const std::vector<double>& C, const size_t N) {
    // Check a few random positions
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
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j,
                       expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

static void printUsage(const char* progName) {
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
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t baseRows = N / static_cast<size_t>(world);
    const size_t rem = N % static_cast<size_t>(world);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t rowStart = baseRows * static_cast<size_t>(rank) +
                            (static_cast<size_t>(rank) < rem ? static_cast<size_t>(rank) : rem);

    // Allocate local A and C, and full B^T (replicated) for fast access
    std::vector<double> Arows(localRows * N);
    std::vector<double> BT(N * N);
    std::vector<double> Crows(localRows * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    initMatrixRows(Arows, N, rowStart, localRows);
    initMatrixTranspose(BT, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    matrixMultiplyLocalRows(Arows, BT, Crows, N, localRows);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localSec = t1 - t0;
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxSec * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate GFLOPS
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) /
                              (maxSec) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    const bool needFullC = validate || printResults;
    std::vector<double> C;
    std::vector<int> recvcounts;
    std::vector<int> displs;

    if (needFullC) {
        if (rank == 0) {
            C.resize(N * N);
            recvcounts.resize(static_cast<size_t>(world));
            displs.resize(static_cast<size_t>(world));
            int disp = 0;
            for (int r = 0; r < world; ++r) {
                const size_t rRows = baseRows + (static_cast<size_t>(r) < rem ? 1 : 0);
                const int cnt = static_cast<int>(rRows * N);
                recvcounts[static_cast<size_t>(r)] = cnt;
                displs[static_cast<size_t>(r)] = disp;
                disp += cnt;
            }
        }

        MPI_Gatherv(Crows.data(), static_cast<int>(localRows * N), MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    int exitCode = 0;

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(C, "MatrixC");
    }

    // Validation
    if (validate) {
        int validInt = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            std::vector<double> Afull(N * N);
            std::vector<double> Bfull(N * N);
            initMatrix(Afull, N);
            initMatrix(Bfull, N);
            const bool valid = validateResult(Afull, Bfull, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
                validInt = 1;
            } else {
                printf("Validation: FAILED\n");
                validInt = 0;
            }
        }
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exitCode = validInt ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
