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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t startRow,
                     const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        const size_t globalRow = startRow + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t localRows, const size_t N) {
    for (size_t i = 0; i < localRows; ++i) {
        double* cRow = &C[i * N];
        for (size_t k = 0; k < N; ++k) {
            const double aVal = A[i * N + k];
            const double* bRow = &B[k * N];
            for (size_t j = 0; j < N; ++j) {
                cRow[j] += aVal * bRow[j];
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    int exitCode = -1;
    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitCode != -1) {
        MPI_Finalize();
        return exitCode;
    }

    unsigned long long N64 = static_cast<unsigned long long>(N);
    int validateFlag = validate ? 1 : 0;
    int printFlag = printResults ? 1 : 0;
    MPI_Bcast(&N64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N64);
    validate = validateFlag != 0;
    printResults = printFlag != 0;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t baseRows = (size > 0) ? (N / static_cast<size_t>(size)) : 0;
    const size_t remainder = (size > 0) ? (N % static_cast<size_t>(size)) : 0;
    const size_t rankSize = static_cast<size_t>(rank);
    const size_t localRows = baseRows + (rankSize < remainder ? 1 : 0);
    const size_t startRow = rankSize * baseRows + (rankSize < remainder ? rankSize : remainder);

    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N, 0.0);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    if (N > 0) {
        initMatrix(B, N);
        initMatrixBlock(A, N, startRow, localRows);
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    matrixMultiply(A, B, C, localRows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double localTime = end - start;

    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = maxTime * 1000.0;
        printf("Computation time: %.3f ms\n", durationMs);
        if (durationMs > 0.0) {
            const double gflops = (2.0 * N * N * N) / maxTime / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        }
    }

    std::vector<double> gatheredC;
    if (printResults || validate) {
        if (rank == 0) {
            gatheredC.resize(N * N);
        }
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t rSize = static_cast<size_t>(r);
            const size_t rows = baseRows + (rSize < remainder ? 1 : 0);
            const size_t offset = rSize * baseRows + (rSize < remainder ? rSize : remainder);
            counts[r] = static_cast<int>(rows * N);
            displs[r] = static_cast<int>(offset * N);
        }
        const int sendCount = static_cast<int>(localRows * N);
        MPI_Gatherv(C.data(), sendCount, MPI_DOUBLE,
                    rank == 0 ? gatheredC.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (size == 1) {
            gatheredC.swap(C);
        }
    }

    if (rank == 0 && printResults) {
        print_results(gatheredC, "MatrixC");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(gatheredC, N);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
