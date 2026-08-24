#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Parallel Cholesky decomposition using MPI with 1D block column distribution
// Right-looking algorithm:
//   A = L * L^T, L lower triangular.
//   For each column j:
//     1. Owner of column j computes L[j][j] = sqrt(A[j][j])
//        and L[i][j] = A[i][j] / L[j][j] for i > j.
//     2. Owner broadcasts column j to all ranks.
//     3. Each rank updates its owned columns l > j via rank-1 update:
//        A[i][l] -= L[i][j] * L[l][j]   for i >= l > j.
// ---------------------------------------------------------------------------

// Return the owning rank of global column j given columns-per-rank.
static inline int colOwner(const size_t j, const size_t colsPerRank) {
    return static_cast<int>(j / colsPerRank);
}

// ---------------------------------------------------------------------------
// Distributed Cholesky decomposition.
//  local_data  : n * local_count doubles, column-major storage of owned columns
//  n           : global matrix dimension
//  localStart  : first global column owned by this rank
//  localCount  : number of columns owned by this rank
//  colsPerRank : ceil(n / numProcs)
// ---------------------------------------------------------------------------
static bool choleskyDecompositionMPI(std::vector<double>& localData,
                                     const size_t n,
                                     const size_t localStart,
                                     const size_t localCount,
                                     const size_t colsPerRank,
                                     const int rank,
                                     const int /*numProcs*/) {

    // Broadcast buffer – large enough for the longest column (n elements)
    std::vector<double> buf(n);

    for (size_t j = 0; j < n; ++j) {
        const int owner = colOwner(j, colsPerRank);

        if (rank == owner) {
            const size_t lcol = j - owner * colsPerRank;      // local column index
            double *col = &localData[lcol * n];

            // Diagonal: L[j][j] = sqrt(A[j][j])
            const double diag = sqrt(col[j]);
            col[j] = diag;
            const double invDiag = 1.0 / diag;

            // Off-diagonal: L[i][j] = A[i][j] / L[j][j]  for i > j
            for (size_t i = j + 1; i < n; ++i) {
                col[i] *= invDiag;
            }

            // Prepare broadcast buffer: elements j … n-1 of column j
            std::copy(col + j, col + n, buf.begin());
        }

        // Broadcast entire column j (diagonal + off-diagonal) to every rank
        MPI_Bcast(buf.data(), static_cast<int>(n - j), MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);

        // Each rank updates its OWNED columns l > j  (rank-1 update)
        for (size_t c = 0; c < localCount; ++c) {
            const size_t l = localStart + c;
            if (l <= j) continue;

            const double L_lj = buf[l - j];          // L[l][j]
            double *colL = &localData[c * n];

            // A[i][l] -= L[i][j] * L[l][j]  for i >= l
            for (size_t i = l; i < n; ++i) {
                colL[i] -= buf[i - j] * L_lj;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Generate local columns of A = B * B^T + n*I  (deterministic, replicated B)
// ---------------------------------------------------------------------------
static void generateLocalMatrix(std::vector<double> &localData,
                                const size_t n,
                                const size_t localStart,
                                const size_t localCount) {

    unsigned int seed = 42;
    std::vector<double> B(n * n);

    // Every rank creates the *same* random B (same seed)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute owned columns of A = B * B^T
    for (size_t c = 0; c < localCount; ++c) {
        const size_t j = localStart + c;               // global column
        double *col = &localData[c * n];

        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            col[i] = sum;
        }

        // Diagonal dominance shift
        col[j] += static_cast<double>(n);
    }
}

// ---------------------------------------------------------------------------
// Generate the *whole* matrix A = B * B^T + n*I  (row-major, rank 0 only)
// ---------------------------------------------------------------------------
static void generateFullMatrix(std::vector<double> &A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += static_cast<double>(n);
    }
}

// ---------------------------------------------------------------------------
// Validation:  compute L * L^T and compare with the original A (rank 0)
// ---------------------------------------------------------------------------
static bool validateCholesky(const std::vector<double> &L,
                             const std::vector<double> &A_orig,
                             const size_t n) {

    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double err = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, err);
        const double rel = err / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Gather the distributed L matrix onto rank 0 in row-major order.
// The caller supplies a (possibly empty) vector; on return rank=0 holds the
// full n×n row-major matrix, other ranks get an empty vector.
// ---------------------------------------------------------------------------
static void gatherLMatrix(const std::vector<double> &localData,
                          const size_t n,
                          const size_t /*localStart*/,
                          const size_t /*localCount*/,
                          const size_t colsPerRank,
                          const int numProcs,
                          std::vector<double> &fullL) {

    if (fullL.size() != n * n) fullL.resize(n * n, 0.0);

    // recv_counts[p] = columns owned by rank p  *  n
    // displs[p]     = offset (in elements) in the column-major full matrix
    std::vector<int> recvCounts(numProcs);
    std::vector<int> displs(numProcs);

    for (int p = 0; p < numProcs; ++p) {
        const size_t start = static_cast<size_t>(p) * colsPerRank;
        const size_t cnt   = std::min(colsPerRank, n - start);
        recvCounts[p] = static_cast<int>(cnt * n);
        displs[p]     = static_cast<int>(start * n);
    }

    // The local data is already in column-major order (all rows for each owned
    // column).  Gather it into fullL which, after the gather, holds the matrix
    // in column-major order (column j → offset j*n, each column has n rows).
    MPI_Gatherv(localData.data(), static_cast<int>(localData.size()), MPI_DOUBLE,
                fullL.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// Convert a column-major matrix to row-major order.
// Only the lower triangle (including diagonal) is meaningful; the upper part
// is zeroed to match the original code's output semantics.
// ---------------------------------------------------------------------------
static void colMajorToRowMajor(const std::vector<double> &colMajor,
                               const size_t n,
                               std::vector<double> &rowMajor) {

    if (rowMajor.size() != n * n) rowMajor.resize(n * n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            // colMajor[j*n + i] = element at row i, column j (col-major storage)
            rowMajor[i * n + j] = colMajor[j * n + i];
        }
        // Upper triangular part → zero
        for (size_t j = i + 1; j < n; ++j) {
            rowMajor[i * n + j] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
static void printUsage(const char *progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command-line arguments (every rank parses identically)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
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

    // ---- 1-D block column distribution -----------------------------------
    const size_t colsPerRank = (n + static_cast<size_t>(numProcs) - 1)
                               / static_cast<size_t>(numProcs);
    const size_t localStart  = static_cast<size_t>(rank) * colsPerRank;
    const size_t localCount  = std::min(colsPerRank, n - localStart);

    // Local storage:  n rows per owned column, column-major
    std::vector<double> localData(localCount * n);

    // Rank 0 keeps the original matrix for optional validation
    std::vector<double> A_orig;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Processes:   %d\n", numProcs);
        printf("Validation:  %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);

        if (validate) A_orig.resize(n * n);
    }

    // ---- Generate the positive-definite matrix ---------------------------
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        fflush(stdout);
    }

    generateLocalMatrix(localData, n, localStart, localCount);

    // Rank 0 (re-)generates the full matrix for validation
    if (validate && rank == 0) {
        generateFullMatrix(A_orig, n);
    }

    // ---- Cholesky decomposition (timed) ----------------------------------
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    const bool success = choleskyDecompositionMPI(
        localData, n, localStart, localCount, colsPerRank, rank, numProcs);

    const auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // ~ n³/3 flops for Cholesky
        const double ops   = static_cast<double>(n) * n * n / 3.0;
        const double secs  = maxDurationMs / 1000.0;
        const double gflops = ops / secs / 1.0e9;
        printf("Performance:     %.3f GFLOPS\n", gflops);
        fflush(stdout);
    }

    // ---- Gather result to rank 0 for output / validation -----------------
    std::vector<double> fullMatrixColMajor;   // column-major on rank 0
    if (rank == 0) fullMatrixColMajor.resize(n * n, 0.0);

    gatherLMatrix(localData, n, localStart, localCount,
                  colsPerRank, numProcs, fullMatrixColMajor);

    // ---- Optional results output (rank 0) --------------------------------
    if (printResults && rank == 0) {
        std::vector<double> fullRowMajor;
        colMajorToRowMajor(fullMatrixColMajor, n, fullRowMajor);
        print_results(fullRowMajor, "CholeskyL");
    }

    // ---- Optional validation (rank 0) ------------------------------------
    if (validate && rank == 0) {
        printf("Validating result...\n");

        std::vector<double> LRowMajor;
        colMajorToRowMajor(fullMatrixColMajor, n, LRowMajor);

        const bool valid = validateCholesky(LRowMajor, A_orig, n);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
