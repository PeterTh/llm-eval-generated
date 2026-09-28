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

// Initialize rows [rowBegin, rowBegin + numRows) of an NxN matrix
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin,
                    const size_t numRows) {
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowBegin + i, j);
        }
    }
}

// Multiply a block of rows of A (numRows x N) with B (N x N) into C (numRows x N).
// Cache-tiled i-k-j ordering: for every element C[i][j] the accumulation over k
// still happens in strictly increasing k order (kk blocks ascend, k ascends within
// a block), so the result is identical to the naive triple loop while tiles of B
// are reused from cache and the innermost j loop vectorizes.
void matrixMultiplyRows(const std::vector<double>& A, const std::vector<double>& B,
                        std::vector<double>& C, const size_t N, const size_t numRows) {
    constexpr size_t KB = 256;  // k tile
    constexpr size_t JB = 512;  // j tile

    std::fill(C.begin(), C.end(), 0.0);

    for (size_t kk = 0; kk < N; kk += KB) {
        const size_t kEnd = std::min(kk + KB, N);
        for (size_t jj = 0; jj < N; jj += JB) {
            const size_t jEnd = std::min(jj + JB, N);
            for (size_t i = 0; i < numRows; ++i) {
                double* Ci = &C[i * N];
                const double* Ai = &A[i * N];
                for (size_t k = kk; k < kEnd; ++k) {
                    const double aik = Ai[k];
                    const double* Bk = &B[k * N];
                    for (size_t j = jj; j < jEnd; ++j) {
                        Ci[j] += aik * Bk[j];
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
        printf("MPI ranks: %d\n", numRanks);
    }

    // Row-block distribution: rank r owns rows [rowBegin, rowBegin + localRows)
    const size_t baseRows = N / static_cast<size_t>(numRanks);
    const size_t remainder = N % static_cast<size_t>(numRanks);
    const size_t urank = static_cast<size_t>(rank);
    const size_t localRows = baseRows + (urank < remainder ? 1 : 0);
    const size_t rowBegin = urank * baseRows + std::min(urank, remainder);

    // Allocate matrices: each rank holds its rows of A and C, plus all of B
    std::vector<double> localA(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(localRows * N);
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    // Initialize matrices: the generator is a pure function of the indices,
    // so every rank initializes its own data without communication
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(localA, N, rowBegin, localRows);
    initMatrixRows(B, N, 0, N);

    // Gather layout for assembling C on rank 0
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t ur = static_cast<size_t>(r);
        const size_t rows = baseRows + (ur < remainder ? 1 : 0);
        const size_t begin = ur * baseRows + std::min(ur, remainder);
        counts[r] = static_cast<int>(rows * N);
        displs[r] = static_cast<int>(begin * N);
    }

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyRows(localA, B, localC, N, localRows);

    MPI_Gatherv(localC.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A(N * N);
            initMatrixRows(A, N, 0, N);
            bool valid = validateResult(A, B, C, N);

            if (valid) {
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
