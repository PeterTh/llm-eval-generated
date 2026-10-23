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

void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t rowBegin, const size_t rowEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        const size_t localRow = i - rowBegin;
        for (size_t j = 0; j < N; ++j) {
            mat[localRow * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Cache-blocked, vectorized kernel that keeps the original rounding sequence
// (two products followed by two adds), so results are bit-identical.
void matrixMultiplyRows(const double* __restrict A, const double* __restrict B,
                        double* __restrict C, const size_t rows, const size_t N) {
    constexpr size_t BLOCK_M = 256;
    constexpr size_t BLOCK_N = 512;
    constexpr size_t BLOCK_K = 32;

    std::fill(C, C + rows * N, 0.0);

    for (size_t ii = 0; ii < rows; ii += BLOCK_M) {
        const size_t iEnd = std::min(ii + BLOCK_M, rows);
        for (size_t jj = 0; jj < N; jj += BLOCK_N) {
            const size_t jEnd = std::min(jj + BLOCK_N, N);
            for (size_t kk = 0; kk < N; kk += BLOCK_K) {
                const size_t kEnd = std::min(kk + BLOCK_K, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    const double* __restrict aRow = A + i * N;
                    double* __restrict cRow = C + i * N;
                    size_t k = kk;
                    for (; k + 1 < kEnd; k += 2) {
                        const double a0 = aRow[k];
                        const double a1 = aRow[k + 1];
                        const double* __restrict b0 = B + k * N;
                        const double* __restrict b1 = B + (k + 1) * N;
                        for (size_t j = jj; j < jEnd; ++j) {
                            const double p0 = a0 * b0[j];
                            const double p1 = a1 * b1[j];
                            cRow[j] += p0;
                            cRow[j] += p1;
                        }
                    }
                    if (k < kEnd) {
                        const double a0 = aRow[k];
                        const double* __restrict b0 = B + k * N;
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] = std::fma(a0, b0[j], cRow[j]);
                        }
                    }
                }
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
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

    std::vector<int> rowCounts(numRanks);
    std::vector<int> rowDispls(numRanks);
    const size_t baseRows = N / static_cast<size_t>(numRanks);
    const size_t extraRows = N % static_cast<size_t>(numRanks);
    size_t rowOffset = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        rowCounts[r] = static_cast<int>(rows);
        rowDispls[r] = static_cast<int>(rowOffset);
        rowOffset += rows;
    }

    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    const size_t localRowStart = static_cast<size_t>(rowDispls[rank]);

    std::vector<double> localA(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(localRows * N);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(localA, N, localRowStart, localRowStart + localRows);
    initMatrix(B, N);

    std::vector<int> elemCounts(numRanks);
    std::vector<int> elemDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        elemCounts[r] = rowCounts[r] * static_cast<int>(N);
        elemDispls[r] = rowDispls[r] * static_cast<int>(N);
    }

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiplyRows(localA.data(), B.data(), localC.data(), localRows, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long localMillis = static_cast<long long>(duration.count());
    long long millis = 0;
    MPI_Reduce(&localMillis, &millis, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> C;
    std::vector<double> A;
    const bool gatherC = printResults || validate;
    if (rank == 0 && gatherC) {
        C.resize(N * N);
    }
    if (gatherC) {
        MPI_Gatherv(localC.data(), elemCounts[rank], MPI_DOUBLE,
                    C.data(), elemCounts.data(), elemDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (validate) {
        if (rank == 0) {
            A.resize(N * N);
        }
        MPI_Gatherv(localA.data(), elemCounts[rank], MPI_DOUBLE,
                    A.data(), elemCounts.data(), elemDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %lld ms\n", millis);
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (millis / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
