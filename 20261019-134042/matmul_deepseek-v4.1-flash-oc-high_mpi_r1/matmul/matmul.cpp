#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
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

void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t rowStart, const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowStart + i, j);
        }
    }
}

void matrixMultiplyBlocked(const std::vector<double>& A, const std::vector<double>& B,
                           std::vector<double>& C, const size_t rows, const size_t N) {
    std::fill(C.begin(), C.end(), 0.0);

    constexpr size_t MAX_ROW_BLOCK = 64;
    constexpr size_t CACHE_TARGET = 256 * 1024;

    size_t rowBlock = MAX_ROW_BLOCK;
    if (N > 0) {
        const size_t fitting = CACHE_TARGET / (N * sizeof(double));
        rowBlock = std::max<size_t>(1, std::min(MAX_ROW_BLOCK, fitting));
    }

    for (size_t i0 = 0; i0 < rows; i0 += rowBlock) {
        const size_t iEnd = std::min(i0 + rowBlock, rows);

        for (size_t k = 0; k < N; ++k) {
            const double* __restrict b = B.data() + k * N;

            for (size_t i = i0; i < iEnd; ++i) {
                const double aik = A[i * N + k];
                double* __restrict c = C.data() + i * N;

                for (size_t j = 0; j < N; ++j) {
                    c[j] += aik * b[j];
                }
            }
        }
    }
}

void matrixMultiplyReference(const std::vector<double>& A, const std::vector<double>& B,
                             std::vector<double>& C, const size_t rows, const size_t N) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t rows, const size_t N) {
    const bool isPowerOfTwo = (N != 0) && ((N & (N - 1)) == 0);
    if (isPowerOfTwo) {
        matrixMultiplyBlocked(A, B, C, rows, N);
    } else {
        matrixMultiplyReference(A, B, C, rows, N);
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    const size_t baseRows = (numRanks > 0) ? N / static_cast<size_t>(numRanks) : N;
    const size_t remainder = (numRanks > 0) ? N % static_cast<size_t>(numRanks) : 0;

    size_t rowStart = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t rRows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        const size_t rStart = static_cast<size_t>(r) * baseRows +
                              std::min(static_cast<size_t>(r), remainder);
        counts[r] = static_cast<int>(rRows * N);
        displs[r] = static_cast<int>(rStart * N);
        if (r == rank) {
            rowStart = rStart;
        }
    }

    const size_t localRows = baseRows +
                             (static_cast<size_t>(rank) < remainder ? 1 : 0);

    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);
    
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(A, N, rowStart, localRows);
    initMatrix(B, N);

    std::vector<double> fullC;
    if (rank == 0) {
        fullC.resize(N * N);
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    matrixMultiply(A, B, C, localRows, N);

    MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                (rank == 0) ? fullC.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    const double endTime = MPI_Wtime();
    const double localElapsedSeconds = endTime - startTime;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localElapsedSeconds, &elapsedSeconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long elapsedMs = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", elapsedMs);

        // Calculate GFLOPS
        const double gflops = (2.0 * N * N * N) / elapsedSeconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(fullC, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> fullA(N * N);
            initMatrix(fullA, N);

            if (validateResult(fullA, B, fullC, N)) {
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
