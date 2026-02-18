#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
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

void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t rowOffset, const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        const size_t globalRow = rowOffset + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void matrixMultiplyBlocked(const std::vector<double>& A, const std::vector<double>& B,
                           std::vector<double>& C, const size_t N, const size_t localRows) {
    constexpr size_t blockSize = 64;
    for (size_t ii = 0; ii < localRows; ii += blockSize) {
        const size_t iMax = std::min(localRows, ii + blockSize);
        for (size_t kk = 0; kk < N; kk += blockSize) {
            const size_t kMax = std::min(N, kk + blockSize);
            for (size_t jj = 0; jj < N; jj += blockSize) {
                const size_t jMax = std::min(N, jj + blockSize);
                for (size_t i = ii; i < iMax; ++i) {
                    for (size_t k = kk; k < kMax; ++k) {
                        const double aVal = A[i * N + k];
                        const size_t bBase = k * N;
                        const size_t cBase = i * N;
                        for (size_t j = jj; j < jMax; ++j) {
                            C[cBase + j] += aVal * B[bBase + j];
                        }
                    }
                }
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
    bool showHelp = false;
    int exitCode = 0;
    
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
                showHelp = true;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                exitCode = 1;
                showHelp = true;
                break;
            }
        }
    }

    int flags[4] = {static_cast<int>(validate), static_cast<int>(printResults),
                    static_cast<int>(showHelp), exitCode};
    MPI_Bcast(flags, 4, MPI_INT, 0, MPI_COMM_WORLD);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;
    showHelp = flags[2] != 0;
    exitCode = flags[3];

    unsigned long long nBroadcast = static_cast<unsigned long long>(N);
    MPI_Bcast(&nBroadcast, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(nBroadcast);

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return exitCode;
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }
    
    const size_t baseRows = N / static_cast<size_t>(size);
    const size_t extraRows = N % static_cast<size_t>(size);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t rowOffset = baseRows * static_cast<size_t>(rank) +
                             std::min(static_cast<size_t>(rank), extraRows);
    
    // Allocate matrices
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N, 0.0);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixBlock(A, N, rowOffset, localRows);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiplyBlocked(A, B, C, N, localRows);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double durationSeconds = end - start;
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", durationSeconds * 1000.0);
    }
    
    // Calculate GFLOPS
    if (rank == 0) {
        double gflops = (2.0 * N * N * N) / durationSeconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    std::vector<double> Cfull;
    if (printResults || validate) {
        std::vector<int> recvCounts;
        std::vector<int> displs;
        if (rank == 0) {
            recvCounts.resize(size);
            displs.resize(size);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
                recvCounts[r] = static_cast<int>(rows * N);
                displs[r] = static_cast<int>(offset * N);
                offset += rows;
            }
            Cfull.resize(N * N);
        }
        const int* recvCountsPtr = rank == 0 ? recvCounts.data() : nullptr;
        const int* displsPtr = rank == 0 ? displs.data() : nullptr;
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    rank == 0 ? Cfull.data() : nullptr, recvCountsPtr, displsPtr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(Cfull, "MatrixC");
    }
    
    // Validation
    int validationOk = 1;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        validationOk = validateResult(Cfull, N) ? 1 : 0;
        printf("Validation: %s\n", validationOk ? "PASSED" : "FAILED");
    }
    if (validate) {
        MPI_Bcast(&validationOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return validationOk ? 0 : 1;
}
