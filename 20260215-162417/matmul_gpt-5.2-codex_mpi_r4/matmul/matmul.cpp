#include <algorithm>
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

void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t rowOffset, const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        const size_t globalRow = rowOffset + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& localA, const std::vector<double>& B,
                    std::vector<double>& localC, const size_t N, const size_t localRows) {
    for (size_t i = 0; i < localRows; ++i) {
        double* cRow = localC.data() + i * N;
        std::fill(cRow, cRow + N, 0.0);
        const double* aRow = localA.data() + i * N;
        for (size_t k = 0; k < N; ++k) {
            const double aVal = aRow[k];
            const double* bRow = B.data() + k * N;
            for (size_t j = 0; j < N; ++j) {
                cRow[j] += aVal * bRow[j];
            }
        }
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    int showHelp = 0;
    
    // Parse command line arguments
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
                parseStatus = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                showHelp = 1;
                parseStatus = 1;
                break;
            }
        }
    }

    uint64_t nValue = static_cast<uint64_t>(N);
    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&nValue, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(nValue);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;

    if (showHelp != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    const size_t baseRows = N / static_cast<size_t>(size);
    const size_t remainder = N % static_cast<size_t>(size);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t rowOffset = baseRows * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);

    std::vector<double> B(N * N);
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(B, N);
    }
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> localA(localRows * N);
    std::vector<double> localC(localRows * N);
    initMatrixBlock(localA, N, rowOffset, localRows);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(localA, B, localC, N, localRows);
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        size_t offset = 0;
        for (int r = 0; r < size; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
            counts[r] = static_cast<int>(rows * N);
            displs[r] = static_cast<int>(offset * N);
            offset += rows;
        }
    }

    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int finalStatus = 0;
    if (rank == 0) {
        const long durationMs = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        const double gflops = (2.0 * N * N * N) / maxTime / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A(N * N);
            initMatrix(A, N);
            const bool valid = validateResult(A, B, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
                finalStatus = 0;
            } else {
                printf("Validation: FAILED\n");
                finalStatus = 1;
            }
        }
    }

    MPI_Bcast(&finalStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return finalStatus;
}
