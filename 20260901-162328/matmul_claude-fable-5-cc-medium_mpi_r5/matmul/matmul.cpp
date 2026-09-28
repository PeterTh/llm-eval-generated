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

// Initialize a block of rows [rowBegin, rowEnd) of the full matrix
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin,
                    const size_t rowEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Multiply a block of rows of A (localRows x N) with B (N x N) into C (localRows x N).
// Uses an i-k-j loop order with blocking over k and j for cache efficiency and
// vectorization. Per element, additions occur in increasing k order, matching the
// naive i-j-k kernel bit-for-bit.
void matrixMultiplyRows(const std::vector<double>& A, const std::vector<double>& B,
                        std::vector<double>& C, const size_t N, const size_t localRows) {
    constexpr size_t KB = 64;
    constexpr size_t JB = 256;

    std::fill(C.begin(), C.begin() + localRows * N, 0.0);

    for (size_t kk = 0; kk < N; kk += KB) {
        const size_t kEnd = std::min(kk + KB, N);
        for (size_t jj = 0; jj < N; jj += JB) {
            const size_t jEnd = std::min(jj + JB, N);
            for (size_t i = 0; i < localRows; ++i) {
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
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block-row distribution: rank r owns rows [rowBegin, rowEnd)
    const size_t rowsPerRank = N / numRanks;
    const size_t remainder = N % numRanks;
    const size_t urank = static_cast<size_t>(rank);
    const size_t rowBegin = urank * rowsPerRank + std::min(urank, remainder);
    const size_t localRows = rowsPerRank + (urank < remainder ? 1 : 0);
    const size_t rowEnd = rowBegin + localRows;

    // Allocate matrices: each rank holds its rows of A and C, and all of B.
    // Rank 0 additionally holds the gathered full C (and full A for validation).
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(localRows * N);
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    // Initialize matrices; initialization is deterministic, so every rank
    // generates its portion of A and the full B locally without communication.
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(A, N, rowBegin, rowEnd);
    initMatrix(B, N);

    // Gather layout for assembling C on rank 0
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t ur = static_cast<size_t>(r);
        const size_t rBegin = ur * rowsPerRank + std::min(ur, remainder);
        const size_t rRows = rowsPerRank + (ur < remainder ? 1 : 0);
        recvCounts[r] = static_cast<int>(rRows * N);
        displs[r] = static_cast<int>(rBegin * N);
    }

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    const double start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    matrixMultiplyRows(A, B, localC, N, localRows);

    MPI_Gatherv(localC.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double end = MPI_Wtime();
    const long durationMs = static_cast<long>((end - start) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> fullA(N * N);
            initMatrix(fullA, N);
            bool valid = validateResult(fullA, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
