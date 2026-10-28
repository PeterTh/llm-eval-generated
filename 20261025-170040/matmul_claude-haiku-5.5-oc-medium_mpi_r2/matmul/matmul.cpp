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

// Fill 'rowCount' rows of an N-column matrix starting at global row 'rowBegin'.
// Each value depends only on its global (i, j), so ranks can initialize their own rows independently.
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowCount) {
    for (size_t r = 0; r < rowCount; ++r) {
        const size_t i = rowBegin + r;
        for (size_t j = 0; j < N; ++j) {
            mat[r * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Computes C = A * B for a block of 'rows' rows (A: rows x N, B: N x N, C: rows x N).
// Cache-blocked i-k-j kernel with 4-row register reuse of B. For every C(i,j) the
// products are accumulated in increasing k order, starting from 0.0, exactly as the
// original triple loop does, so the results are bit-identical to the serial version.
void matrixMultiplyRows(const double* A, const double* B, double* C, const size_t rows, const size_t N) {
    constexpr size_t KB = 128;
    constexpr size_t JB = 512;

    std::fill(C, C + rows * N, 0.0);

    for (size_t jb = 0; jb < N; jb += JB) {
        const size_t jend = std::min(jb + JB, N);
        for (size_t kb = 0; kb < N; kb += KB) {
            const size_t kend = std::min(kb + KB, N);

            size_t i = 0;
            for (; i + 4 <= rows; i += 4) {
                const double* a0 = A + (i + 0) * N;
                const double* a1 = A + (i + 1) * N;
                const double* a2 = A + (i + 2) * N;
                const double* a3 = A + (i + 3) * N;
                double* __restrict c0 = C + (i + 0) * N;
                double* __restrict c1 = C + (i + 1) * N;
                double* __restrict c2 = C + (i + 2) * N;
                double* __restrict c3 = C + (i + 3) * N;
                for (size_t k = kb; k < kend; ++k) {
                    const double x0 = a0[k];
                    const double x1 = a1[k];
                    const double x2 = a2[k];
                    const double x3 = a3[k];
                    const double* __restrict b = B + k * N;
                    for (size_t j = jb; j < jend; ++j) {
                        const double bj = b[j];
                        c0[j] += x0 * bj;
                        c1[j] += x1 * bj;
                        c2[j] += x2 * bj;
                        c3[j] += x3 * bj;
                    }
                }
            }
            for (; i < rows; ++i) {
                const double* a = A + i * N;
                double* __restrict c = C + i * N;
                for (size_t k = kb; k < kend; ++k) {
                    const double x = a[k];
                    const double* __restrict b = B + k * N;
                    for (size_t j = jb; j < jend; ++j) {
                        c[j] += x * b[j];
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare.
// A is regenerated on the fly (identical values to initMatrixRows), so only B and C are needed.
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

    int rankI = 0;
    int sizeI = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankI);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeI);
    const bool root = (rankI == 0);
    const size_t rank = static_cast<size_t>(rankI);
    const size_t size = static_cast<size_t>(sizeI);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks exit consistently)
    int exitEarly = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            exitEarly = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitEarly = 1;
            break;
        }
    }
    if (exitEarly >= 0) {
        MPI_Finalize();
        return exitEarly;
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %zu\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1-D row-block distribution: the first (N % size) ranks receive one extra row
    const size_t baseRows = N / size;
    const size_t extraRows = N % size;
    const size_t myRows = baseRows + (rank < extraRows ? 1 : 0);
    const size_t myStart = rank * baseRows + std::min(rank, extraRows);

    // Each rank holds its A rows, its C rows, and a full copy of B
    std::vector<double> A(myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(myRows * N);

    if (root) printf("Initializing matrices...\n");
    initMatrixRows(A, N, myStart, myRows);
    initMatrixRows(B, N, 0, N);

    if (root) printf("Computing matrix multiplication...\n");

    // Synchronize, then time the local computation; report the slowest rank
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    matrixMultiplyRows(A.data(), B.data(), C.data(), myRows, N);

    const double localSec = MPI_Wtime() - t0;
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int status = 0;
    if (root) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(maxSec));
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / maxSec / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full C matrix on the root only when it is needed for output or validation
    if (printResults || validate) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(size), displs(size);
        for (size_t r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(baseRows + (r < extraRows ? 1 : 0));
            displs[r] = static_cast<int>(r * baseRows + std::min(r, extraRows));
        }

        std::vector<double> fullC;
        if (root) fullC.resize(N * N);

        MPI_Gatherv(C.data(), static_cast<int>(myRows), rowType,
                    fullC.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);

        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(fullC, "MatrixC");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(B, fullC, N);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        }
    }

    // Propagate the root's exit status to all ranks
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
