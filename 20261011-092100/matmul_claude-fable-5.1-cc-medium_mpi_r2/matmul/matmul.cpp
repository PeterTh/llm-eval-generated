#include <mpi.h>

#include <algorithm>
#include <chrono>
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

// Initialize rows [rowBegin, rowEnd) of an NxN matrix into a local buffer
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin,
                    const size_t rowEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = mat.data() + (i - rowBegin) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Compute C_local (localRows x N) = A_local (localRows x N) * B (N x N).
// Cache-blocked i-k-j ordering: the inner loop over j is contiguous in both B and C
// and vectorizes well; each C[i][j] still accumulates k in order 0..N-1, so the
// result matches the naive triple loop.
void matrixMultiplyLocal(const double* __restrict A, const double* __restrict B,
                         double* __restrict C, const size_t localRows, const size_t N) {
    constexpr size_t BI = 64;   // rows of A/C per block
    constexpr size_t BK = 256;  // k-extent per block (rows of B)
    constexpr size_t BJ = 1024; // columns per block

    std::fill(C, C + localRows * N, 0.0);

    for (size_t jj = 0; jj < N; jj += BJ) {
        const size_t jEnd = std::min(jj + BJ, N);
        for (size_t kk = 0; kk < N; kk += BK) {
            const size_t kEnd = std::min(kk + BK, N);
            for (size_t ii = 0; ii < localRows; ii += BI) {
                const size_t iEnd = std::min(ii + BI, localRows);
                for (size_t i = ii; i < iEnd; ++i) {
                    const double* __restrict Ai = A + i * N;
                    double* __restrict Ci = C + i * N;
                    size_t k = kk;
                    // Unroll by 4 over k to reduce C load/store traffic
                    for (; k + 4 <= kEnd; k += 4) {
                        const double a0 = Ai[k], a1 = Ai[k + 1], a2 = Ai[k + 2], a3 = Ai[k + 3];
                        const double* __restrict B0 = B + k * N;
                        const double* __restrict B1 = B0 + N;
                        const double* __restrict B2 = B1 + N;
                        const double* __restrict B3 = B2 + N;
                        for (size_t j = jj; j < jEnd; ++j) {
                            double c = Ci[j];
                            c += a0 * B0[j];
                            c += a1 * B1[j];
                            c += a2 * B2[j];
                            c += a3 * B3[j];
                            Ci[j] = c;
                        }
                    }
                    for (; k < kEnd; ++k) {
                        const double a = Ai[k];
                        const double* __restrict Bk = B + k * N;
                        for (size_t j = jj; j < jEnd; ++j) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }
            }
        }
    }
}

// Simple validation: recompute a few elements from the generator and compare.
// Only called on the root rank with the fully gathered C.
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
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
    const bool isRoot = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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

    // 1D row-block decomposition of A and C; B is replicated on every rank.
    // Row ranges are computed identically on all ranks.
    std::vector<int> rowCounts(nprocs), rowOffsets(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        const size_t base = N / nprocs, rem = N % nprocs;
        rowCounts[r] = static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
        rowOffsets[r] = (r == 0) ? 0 : rowOffsets[r - 1] + rowCounts[r - 1];
    }
    const size_t rowBegin = rowOffsets[rank];
    const size_t localRows = rowCounts[rank];
    const size_t rowEnd = rowBegin + localRows;

    // Allocate matrices
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);

    // Initialize matrices: the generator is deterministic, so every rank builds its
    // own A rows and the full B locally with no communication.
    if (isRoot) printf("Initializing matrices...\n");
    initMatrixRows(A, N, rowBegin, rowEnd);
    initMatrixRows(B, N, 0, N);

    // Perform matrix multiplication
    if (isRoot) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A.data(), B.data(), C.data(), localRows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (isRoot) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;

    // Gather the full C on the root only when it is needed for output or validation.
    if (printResults || validate) {
        std::vector<double> Cfull;
        std::vector<int> recvCounts(nprocs), recvDispls(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            recvCounts[r] = static_cast<int>(rowCounts[r] * N);
            recvDispls[r] = static_cast<int>(rowOffsets[r] * N);
        }
        if (isRoot) Cfull.resize(N * N);
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    isRoot ? Cfull.data() : nullptr, recvCounts.data(), recvDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (isRoot) {
            // Print results for external validation
            if (printResults) {
                print_results(Cfull, "MatrixC");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(B, Cfull, N);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
