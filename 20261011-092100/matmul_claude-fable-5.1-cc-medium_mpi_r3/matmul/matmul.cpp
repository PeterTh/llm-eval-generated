#include <algorithm>
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

// Initialize rows [rowBegin, rowBegin + numRows) of an NxN matrix into a
// row-major buffer of numRows*N elements (global row index drives the values).
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t numRows) {
    for (size_t r = 0; r < numRows; ++r) {
        const size_t i = rowBegin + r;
        double* row = mat + r * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Computes C_local = A_local * B where A_local holds numRows rows of A (row-major,
// leading dimension N), B is the full NxN matrix and C_local receives numRows rows.
//
// Cache-blocked i-k-j ordering. For every C element the k index is always visited
// in increasing order 0..N-1 with a single running sum, so the summation order is
// identical to the naive triple loop of the original program.
//
// Floating-point contraction is pinned down explicitly so the result is
// bit-identical to the original build on this toolchain. There the compiler
// vectorizes the k reduction with an in-order fold (products via vector multiply,
// then sequential adds, i.e. no fusion) and, for odd N, handles the final k step
// with a scalar fused multiply-add. This file is compiled with -ffp-contract=off,
// so the main k steps use plain mul+add and the last k step of an odd N uses
// std::fma.
void matrixMultiplyRows(const double* __restrict A, const double* __restrict B,
                        double* __restrict C, const size_t N, const size_t numRows) {
    constexpr size_t BI = 32;    // rows of C/A per block
    constexpr size_t BK = 256;   // k extent per block
    constexpr size_t BJ = 2048;  // columns of C/B per block

    const size_t NMainK = N - (N % 2);  // k steps computed with separate mul/add

    std::fill(C, C + numRows * N, 0.0);

    for (size_t jj = 0; jj < N; jj += BJ) {
        const size_t jEnd = std::min(N, jj + BJ);
        for (size_t kk = 0; kk < N; kk += BK) {
            const size_t kEnd = std::min(N, kk + BK);
            const size_t kEndMain = std::min(NMainK, kEnd);
            for (size_t ii = 0; ii < numRows; ii += BI) {
                const size_t iEnd = std::min(numRows, ii + BI);
                for (size_t i = ii; i < iEnd; ++i) {
                    const double* __restrict aRow = A + i * N;
                    double* __restrict cRow = C + i * N;
                    for (size_t k = kk; k < kEndMain; ++k) {
                        const double a = aRow[k];
                        const double* __restrict bRow = B + k * N;
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += a * bRow[j];
                        }
                    }
                    for (size_t k = std::max(kk, kEndMain); k < kEnd; ++k) {
                        const double a = aRow[k];
                        const double* __restrict bRow = B + k * N;
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] = std::fma(a, bRow[j], cRow[j]);
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
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool isRoot = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on all ranks, so all ranks agree)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (isRoot) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
    }

    // 1D block row decomposition of A and C over ranks (B is replicated).
    // Rank r owns rows [rowOffset[r], rowOffset[r] + rowCount[r]).
    std::vector<int> rowCount(nprocs), rowOffset(nprocs);
    std::vector<int> elemCount(nprocs), elemOffset(nprocs);
    {
        const size_t base = N / static_cast<size_t>(nprocs);
        const size_t rem = N % static_cast<size_t>(nprocs);
        size_t off = 0;
        for (int r = 0; r < nprocs; ++r) {
            const size_t cnt = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            rowCount[r] = static_cast<int>(cnt);
            rowOffset[r] = static_cast<int>(off);
            elemCount[r] = static_cast<int>(cnt * N);
            elemOffset[r] = static_cast<int>(off * N);
            off += cnt;
        }
    }
    const size_t myRows = static_cast<size_t>(rowCount[rank]);
    const size_t myRowBegin = static_cast<size_t>(rowOffset[rank]);
    
    // Allocate matrices.
    // Root keeps the full A (its own rows are the leading prefix) so that
    // validation can run exactly as in the original program; other ranks only
    // hold their row block of A. B is replicated. C is gathered on root.
    std::vector<double> A(isRoot ? N * N : myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> Clocal(myRows * N);
    std::vector<double> C(isRoot ? N * N : 0);
    
    // Initialize matrices (deterministic, so every rank generates what it needs
    // locally and no initial data distribution is required)
    if (isRoot) printf("Initializing matrices...\n");
    if (isRoot) {
        initMatrixRows(A.data(), N, 0, N);
    } else {
        initMatrixRows(A.data(), N, myRowBegin, myRows);
    }
    initMatrixRows(B.data(), N, 0, N);
    
    // Perform matrix multiplication
    if (isRoot) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiplyRows(A.data(), B.data(), Clocal.data(), N, myRows);

    // Collect the distributed rows of C on root
    MPI_Gatherv(Clocal.data(), elemCount[rank], MPI_DOUBLE,
                C.data(), elemCount.data(), elemOffset.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    const double end = MPI_Wtime();
    const long durationMs = static_cast<long>((end - start) * 1000.0);

    int exitCode = 0;
    if (isRoot) {
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
            bool valid = validateResult(A, B, C, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
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
