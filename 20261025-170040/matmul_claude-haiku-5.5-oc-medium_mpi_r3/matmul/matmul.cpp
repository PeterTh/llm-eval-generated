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

// Row-block distribution of N rows across `size` ranks. The first (N % size)
// ranks receive one extra row.
struct RowPartition {
    size_t firstRow;
    size_t numRows;
};

RowPartition partitionRows(const size_t N, const int rank, const int size) {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = N / p;
    const size_t extra = N % p;
    return {r * base + std::min(r, extra), base + (r < extra ? 1 : 0)};
}

// Fill `numRows` rows of a row-major matrix, starting at global row `firstRow`.
// `mat` must hold numRows * N elements.
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t firstRow, const size_t numRows) {
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

// Local block multiply: C (numRows x N) = A (numRows x N) * B (N x N).
// Cache-blocked i-k-j loop order with a contiguous, vectorizable inner loop.
// For every C(i,j) the products are accumulated in ascending k order starting
// from 0.0, which is the same summation order as the naive triple loop. The
// build disables FP contraction (-ffp-contract=off) so that each product and
// sum rounds exactly as written in the original code.
void matrixMultiplyRows(const double* A, const double* B, double* C, const size_t numRows, const size_t N) {
    constexpr size_t KB = 64;   // k-block: B panel of KB x JB doubles stays in L2
    constexpr size_t JB = 256;  // j-block: C row segment stays in L1

    std::fill(C, C + numRows * N, 0.0);

    for (size_t jj = 0; jj < N; jj += JB) {
        const size_t jEnd = std::min(jj + JB, N);
        for (size_t kk = 0; kk < N; kk += KB) {
            const size_t kEnd = std::min(kk + KB, N);
            for (size_t i = 0; i < numRows; ++i) {
                double* c = C + i * N;
                const double* a = A + i * N;
                for (size_t k = kk; k < kEnd; ++k) {
                    const double aik = a[k];
                    const double* b = B + k * N;
                    for (size_t j = jj; j < jEnd; ++j) {
                        c[j] += aik * b[j];
                    }
                }
            }
        }
    }
}

// Simple validation: compute a few elements of C from the generator and compare.
// A(i,k) and B(k,j) are pure functions of their indices, so they are regenerated
// here rather than requiring the full A and B on the validating rank.
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
    const bool root = (rank == 0);

    auto finish = [](int code) {
        MPI_Finalize();
        return code;
    };

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            return finish(0);
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Each rank owns a contiguous block of rows of A and C. B is replicated.
    const RowPartition part = partitionRows(N, rank, size);
    std::vector<double> A(part.numRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(part.numRows * N);

    // Initialize matrices (generated locally; no communication required)
    if (root) printf("Initializing matrices...\n");
    initMatrixRows(A, N, part.firstRow, part.numRows);
    initMatrixRows(B, N, 0, N);

    // Perform distributed matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    matrixMultiplyRows(A.data(), B.data(), C.data(), part.numRows, N);

    const double localSec = MPI_Wtime() - t0;

    // The slowest rank determines the wall-clock time of the parallel run
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(maxSec));
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / maxSec / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full result on rank 0 only when it is needed for output or validation
    const bool needFull = printResults || validate;
    std::vector<double> fullC;
    if (needFull) {
        if (root) fullC.resize(N * N);

        // Row-typed gather: counts and displacements are in rows, avoiding int overflow
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const RowPartition p = partitionRows(N, r, size);
            counts[r] = static_cast<int>(p.numRows);
            displs[r] = static_cast<int>(p.firstRow);
        }

        MPI_Gatherv(C.data(), static_cast<int>(part.numRows), rowType,
                    fullC.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);

        MPI_Type_free(&rowType);
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(fullC, "MatrixC");
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (root) {
            printf("Validating result...\n");
            valid = validateResult(fullC, N) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        // Broadcast so every rank exits with the same status
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return finish(valid ? 0 : 1);
    }

    return finish(0);
}
