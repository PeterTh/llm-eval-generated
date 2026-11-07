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

// Initialize rows [rowBegin, rowBegin + rows) of an NxN matrix into a local buffer.
// Initialization is deterministic, so every rank generates its own part without
// any communication.
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin,
                    const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowBegin + i, j);
        }
    }
}

// Local block multiply: C_local (rows x N) = A_local (rows x N) * B (N x N).
// Uses an i-k-j loop order with cache blocking over k and j. For every element
// C[i][j] the products A[i][k]*B[k][j] are accumulated in ascending k order,
// exactly as in the original triple loop. The original build (GCC, -O3
// -march=native) evaluates the reduction as a separately rounded product plus
// add for all k, and as a single fused multiply-add for the last k only when N
// is odd. That rounding pattern is reproduced here (the build disables FP
// contraction and the odd tail uses std::fma explicitly) so the result is
// bitwise identical to the original program.
void matrixMultiplyLocal(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N, const size_t rows) {
    constexpr size_t KB = 256;   // k-block
    constexpr size_t JB = 2048;  // j-block
    constexpr size_t IB = 32;    // i-block

    std::fill(C.begin(), C.begin() + rows * N, 0.0);
    const size_t kFma = (N & 1) ? N - 1 : N;  // index of the fused tail (none if N even)

    for (size_t jj = 0; jj < N; jj += JB) {
        const size_t jEnd = std::min(jj + JB, N);
        for (size_t kk = 0; kk < N; kk += KB) {
            const size_t kEnd = std::min(kk + KB, N);
            for (size_t ii = 0; ii < rows; ii += IB) {
                const size_t iEnd = std::min(ii + IB, rows);
                for (size_t i = ii; i < iEnd; ++i) {
                    double* __restrict cRow = C.data() + i * N;
                    const double* __restrict aRow = A.data() + i * N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double a = aRow[k];
                        const double* __restrict bRow = B.data() + k * N;
                        if (k != kFma) {
                            for (size_t j = jj; j < jEnd; ++j) {
                                cRow[j] += a * bRow[j];
                            }
                        } else {
                            for (size_t j = jj; j < jEnd; ++j) {
                                cRow[j] = std::fma(a, bRow[j], cRow[j]);
                            }
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int earlyExit = -1;  // >= 0: all ranks exit with this code after argument parsing
    
    // Parse command line arguments (identical argv on all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            earlyExit = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            earlyExit = 1;
            break;
        }
    }
    if (earlyExit >= 0) {
        MPI_Finalize();
        return earlyExit;
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", nprocs);
    }

    // 1D row-block decomposition of A and C. Rank r owns rows [rowBegin, rowBegin + rows).
    std::vector<int> counts(nprocs), displs(nprocs);
    {
        size_t off = 0;
        for (int r = 0; r < nprocs; ++r) {
            const size_t rows = N / nprocs + (static_cast<size_t>(r) < N % nprocs ? 1 : 0);
            counts[r] = static_cast<int>(rows * N);
            displs[r] = static_cast<int>(off * N);
            off += rows;
        }
    }
    const size_t localRows = static_cast<size_t>(counts[rank]) / N;
    const size_t rowBegin = static_cast<size_t>(displs[rank]) / N;
    
    // Allocate matrices: each rank holds its block of A and C and a full copy of B.
    // Rank 0 additionally holds the full result C (and full A for validation).
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);
    std::vector<double> Cfull(rank == 0 ? N * N : 0);
    
    // Initialize matrices (deterministic; no communication required)
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(A, N, rowBegin, localRows);
    initMatrixRows(B, N, 0, N);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiplyLocal(A, B, C, N, localRows);
    
    const double localCompute = MPI_Wtime() - start;

    // Gather the distributed result rows into the full matrix on rank 0.
    const double gatherStart = MPI_Wtime();
    MPI_Gatherv(C.data(), counts[rank], MPI_DOUBLE, Cfull.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double gatherTime = MPI_Wtime() - gatherStart;

    double computeTime = 0.0;
    MPI_Reduce(&localCompute, &computeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        const long ms = static_cast<long>(computeTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        printf("Result gather time: %ld ms\n", static_cast<long>(gatherTime * 1000.0));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(Cfull, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> Afull(N * N);
            initMatrixRows(Afull, N, 0, N);
            bool valid = validateResult(Afull, B, Cfull, N);

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
