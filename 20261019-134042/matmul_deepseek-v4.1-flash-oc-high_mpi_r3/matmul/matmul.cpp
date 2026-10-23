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

static size_t partitionCount(const size_t total, const size_t parts, const size_t idx) noexcept {
    return total / parts + (idx < (total % parts) ? 1 : 0);
}

static size_t partitionStart(const size_t total, const size_t parts, const size_t idx) noexcept {
    const size_t base = total / parts;
    const size_t rem = total % parts;
    return idx * base + (idx < rem ? idx : rem);
}

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowStart,
                    const size_t numRows) {
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowStart + i, j);
        }
    }
}

void initMatrixColumns(std::vector<double>& mat, const size_t N, const size_t colStart,
                       const size_t numCols) {
    for (size_t k = 0; k < N; ++k) {
        for (size_t j = 0; j < numCols; ++j) {
            mat[k * numCols + j] = getPseudoRndValue(N, k, colStart + j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t localRows,
                    const size_t localCols, const size_t N) {
    const double* __restrict__ aPtr = A.data();
    const double* __restrict__ bPtr = B.data();
    double* __restrict__ cPtr = C.data();

    for (size_t i = 0; i < localRows; ++i) {
        const double* __restrict__ aRow = aPtr + i * N;
        double* __restrict__ cRow = cPtr + i * localCols;
        size_t k = 0;
        for (; k + 1 < N; k += 2) {
            const double a0 = aRow[k];
            const double a1 = aRow[k + 1];
            const double* __restrict__ bRow0 = bPtr + k * localCols;
            const double* __restrict__ bRow1 = bPtr + (k + 1) * localCols;
            for (size_t j = 0; j < localCols; ++j) {
                cRow[j] += a0 * bRow0[j];
            }
            for (size_t j = 0; j < localCols; ++j) {
                cRow[j] += a1 * bRow1[j];
            }
        }
        if (k < N) {
            const double a = aRow[k];
            const double* __restrict__ bRow = bPtr + k * localCols;
            for (size_t j = 0; j < localCols; ++j) {
                cRow[j] = std::fma(a, bRow[j], cRow[j]);
            }
        }
    }
}

// Simple validation: recompute a few elements directly from the deterministic inputs
bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    int validate = 0;
    int printResults = 0;
    int stop = 0;
    int status = 0;

    // Parse command line arguments on rank 0 only
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                stop = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                status = 1;
                stop = 1;
                break;
            }
        }
    }

    unsigned long long nBroadcast = static_cast<unsigned long long>(N);
    MPI_Bcast(&nBroadcast, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(nBroadcast);

    int control[4] = {validate, printResults, stop, status};
    MPI_Bcast(control, 4, MPI_INT, 0, MPI_COMM_WORLD);
    validate = control[0];
    printResults = control[1];
    stop = control[2];
    status = control[3];

    if (stop) {
        MPI_Finalize();
        return status;
    }

    // Choose a nearly square 2D process grid
    size_t gridCols = static_cast<size_t>(std::sqrt(static_cast<double>(numRanks)));
    if (gridCols == 0) {
        gridCols = 1;
    }
    while (gridCols > 1 && (static_cast<size_t>(numRanks) % gridCols) != 0) {
        --gridCols;
    }
    const size_t gridRows = static_cast<size_t>(numRanks) / gridCols;

    const size_t gridRow = static_cast<size_t>(rank) / gridCols;
    const size_t gridCol = static_cast<size_t>(rank) % gridCols;

    const size_t rowStart = partitionStart(N, gridRows, gridRow);
    const size_t localRows = partitionCount(N, gridRows, gridRow);
    const size_t colStart = partitionStart(N, gridCols, gridCol);
    const size_t localCols = partitionCount(N, gridCols, gridCol);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate local blocks of the matrices
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * localCols);
    std::vector<double> C(localRows * localCols, 0.0);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    initMatrixRows(A, N, rowStart, localRows);
    initMatrixColumns(B, N, colStart, localCols);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    matrixMultiply(A, B, C, localRows, localCols, N);

    const double end = MPI_Wtime();
    double elapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        const double gflops = (2.0 * N * N * N) / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> Cfull;
    if (printResults || validate) {
        std::vector<int> counts(static_cast<size_t>(numRanks), 0);
        std::vector<int> displs(static_cast<size_t>(numRanks), 0);
        int total = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t rRow = static_cast<size_t>(r) / gridCols;
            const size_t rCol = static_cast<size_t>(r) % gridCols;
            const size_t rows = partitionCount(N, gridRows, rRow);
            const size_t cols = partitionCount(N, gridCols, rCol);
            counts[r] = static_cast<int>(rows * cols);
            displs[r] = total;
            total += counts[r];
        }

        std::vector<double> packed;
        if (rank == 0) {
            packed.resize(static_cast<size_t>(total));
        }
        double dummy = 0.0;
        void* recvBuf = (rank == 0) ? static_cast<void*>(packed.data())
                                    : static_cast<void*>(&dummy);
        MPI_Gatherv(C.data(), static_cast<int>(localRows * localCols), MPI_DOUBLE, recvBuf,
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            Cfull.assign(N * N, 0.0);
            for (int r = 0; r < numRanks; ++r) {
                const size_t rRow = static_cast<size_t>(r) / gridCols;
                const size_t rCol = static_cast<size_t>(r) % gridCols;
                const size_t rows = partitionCount(N, gridRows, rRow);
                const size_t cols = partitionCount(N, gridCols, rCol);
                const size_t rStart = partitionStart(N, gridRows, rRow);
                const size_t cStart = partitionStart(N, gridCols, rCol);
                const double* src = packed.data() + displs[r];
                double* dst = Cfull.data() + rStart * N + cStart;
                for (size_t rr = 0; rr < rows; ++rr) {
                    std::memcpy(dst + rr * N, src + rr * cols, cols * sizeof(double));
                }
            }
        }
    }

    if (rank == 0) {
        if (printResults) {
            print_results(Cfull, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(Cfull, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    int finalStatus = status;
    MPI_Bcast(&finalStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return finalStatus;
}
