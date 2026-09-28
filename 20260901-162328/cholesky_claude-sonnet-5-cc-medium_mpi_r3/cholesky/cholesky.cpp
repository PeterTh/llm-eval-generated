#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition (MPI, blocked right-looking algorithm).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is partitioned into block-columns of width B. Block-column j
// (holding all rows i >= j*B, i.e. the lower-triangular part) is owned by
// rank (j % size) in a 1D block-cyclic distribution. Because a block-column
// is owned in full (all rows), the diagonal-block factorization and the
// panel triangular-solve below it are purely local to the owner. After
// factoring block-column k, the owner broadcasts that panel to every rank,
// which then applies the corresponding rank-B trailing update to whichever
// block-columns j > k it owns.

namespace {

size_t blockWidth(size_t j, size_t n, size_t B) { return std::min(B, n - j * B); }
size_t blockRows(size_t j, size_t n, size_t B) { return n - j * B; }

// Factor block-column k in place (diagonal Cholesky block + triangular solve
// for the panel below it). Only touches entries with global row >= global
// column, matching the lower-triangular-only semantics of the original
// sequential algorithm.
bool factorPanel(std::vector<double>& slab, size_t kBase, size_t kb, size_t rows) {
    for (size_t jj = 0; jj < kb; ++jj) {
        double sum = 0.0;
        for (size_t p = 0; p < jj; ++p) {
            sum += slab[jj * kb + p] * slab[jj * kb + p];
        }
        const double val = slab[jj * kb + jj] - sum;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", kBase + jj);
            return false;
        }
        slab[jj * kb + jj] = sqrt(val);

        for (size_t ii = jj + 1; ii < rows; ++ii) {
            double sum2 = 0.0;
            for (size_t p = 0; p < jj; ++p) {
                sum2 += slab[ii * kb + p] * slab[jj * kb + p];
            }
            slab[ii * kb + jj] = (slab[ii * kb + jj] - sum2) / slab[jj * kb + jj];
        }
    }
    return true;
}

// Apply the rank-kb trailing update contributed by panel k to owned
// block-column j (only for entries with global row >= global column).
void applyTrailingUpdate(std::vector<double>& slabJ, size_t jBase, size_t jb, size_t rowsJ,
                          const double* panel, size_t kBase, size_t kb) {
    for (size_t jj2 = 0; jj2 < jb; ++jj2) {
        const size_t gc = jBase + jj2;
        const double* panelJRow = panel + (gc - kBase) * kb;
        for (size_t ii = jj2; ii < rowsJ; ++ii) {
            const size_t gi = jBase + ii;
            const double* panelIRow = panel + (gi - kBase) * kb;
            double sum = 0.0;
            for (size_t p = 0; p < kb; ++p) {
                sum += panelIRow[p] * panelJRow[p];
            }
            slabJ[ii * jb + jj2] -= sum;
        }
    }
}

} // namespace

bool choleskyDecompositionMPI(std::vector<std::vector<double>>& colBlocks, const size_t n, const size_t B,
                               const size_t nb, int rank, int size, MPI_Comm comm) {
    std::vector<double> panelBuf;

    for (size_t k = 0; k < nb; ++k) {
        const int owner = static_cast<int>(k % static_cast<size_t>(size));
        const size_t kBase = k * B;
        const size_t kb = blockWidth(k, n, B);
        const size_t rowsK = blockRows(k, n, B);

        int localOk = 1;
        if (rank == owner) {
            if (!factorPanel(colBlocks[k], kBase, kb, rowsK)) {
                localOk = 0;
            }
        }
        MPI_Bcast(&localOk, 1, MPI_INT, owner, comm);
        if (!localOk) {
            return false;
        }

        const double* panel = nullptr;
        if (rank == owner) {
            panel = colBlocks[k].data();
            MPI_Bcast(colBlocks[k].data(), static_cast<int>(rowsK * kb), MPI_DOUBLE, owner, comm);
        } else {
            panelBuf.resize(rowsK * kb);
            MPI_Bcast(panelBuf.data(), static_cast<int>(rowsK * kb), MPI_DOUBLE, owner, comm);
            panel = panelBuf.data();
        }

        for (size_t j = k + 1; j < nb; ++j) {
            if (static_cast<int>(j % static_cast<size_t>(size)) != rank) continue;
            const size_t jBase = j * B;
            const size_t jb = blockWidth(j, n, B);
            const size_t rowsJ = blockRows(j, n, B);
            applyTrailingUpdate(colBlocks[j], jBase, jb, rowsJ, panel, kBase, kb);
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix, distributed across ranks.
// Method: Create A = B * B^T where B is random (identical on every rank,
// since it is derived from a fixed seed), then add identity to make it
// strictly positive definite. Each rank only materializes the block-columns
// it owns, distributing the O(n^3) A = B*B^T computation across the cluster.
void generatePositiveDefiniteMatrixDistributed(std::vector<std::vector<double>>& colBlocks, const size_t n,
                                                const size_t B, const size_t nb, int rank, int size) {
    std::vector<double> Bmat(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        Bmat[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t j = 0; j < nb; ++j) {
        if (static_cast<int>(j % static_cast<size_t>(size)) != rank) continue;
        const size_t jBase = j * B;
        const size_t jb = blockWidth(j, n, B);
        const size_t rowsJ = blockRows(j, n, B);

        colBlocks[j].assign(rowsJ * jb, 0.0);
        for (size_t jj2 = 0; jj2 < jb; ++jj2) {
            const size_t gc = jBase + jj2;
            for (size_t ii = jj2; ii < rowsJ; ++ii) {
                const size_t gi = jBase + ii;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += Bmat[gi * n + k] * Bmat[gc * n + k];
                }
                if (gi == gc) {
                    sum += static_cast<double>(n);
                }
                colBlocks[j][ii * jb + jj2] = sum;
            }
        }
    }
}

// Generate a symmetric positive definite matrix (sequential, full matrix).
// Kept for the optional validation path, where rank 0 needs an ordinary
// dense reconstruction target compatible with validateCholesky().
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
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
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Gather the distributed block-columns into a full dense n x n matrix
// (only meaningful on rank 0; other ranks' output vector is left empty).
// When mirrorSymmetric is true, the upper triangle is filled by mirroring
// the lower triangle (reconstructing the original symmetric matrix); when
// false, the upper triangle is left zero (matching the stored lower-
// triangular-only representation of the L factor).
std::vector<double> gatherFullMatrix(std::vector<std::vector<double>>& colBlocks, const size_t n, const size_t B,
                                      const size_t nb, int rank, int size, MPI_Comm comm,
                                      bool mirrorSymmetric = false) {
    std::vector<double> full;
    if (rank == 0) {
        full.assign(n * n, 0.0);
    }

    for (size_t j = 0; j < nb; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(size));
        const size_t jBase = j * B;
        const size_t jb = blockWidth(j, n, B);
        const size_t rowsJ = blockRows(j, n, B);
        const int count = static_cast<int>(rowsJ * jb);

        const double* src = nullptr;
        std::vector<double> recvBuf;
        if (rank == 0) {
            if (owner == 0) {
                src = colBlocks[j].data();
            } else {
                recvBuf.resize(rowsJ * jb);
                MPI_Recv(recvBuf.data(), count, MPI_DOUBLE, owner, static_cast<int>(j), comm, MPI_STATUS_IGNORE);
                src = recvBuf.data();
            }
            for (size_t jj2 = 0; jj2 < jb; ++jj2) {
                const size_t gc = jBase + jj2;
                for (size_t ii = jj2; ii < rowsJ; ++ii) {
                    const size_t gi = jBase + ii;
                    full[gi * n + gc] = src[ii * jb + jj2];
                    if (mirrorSymmetric && gi != gc) {
                        full[gc * n + gi] = src[ii * jb + jj2];
                    }
                }
            }
        } else if (rank == owner) {
            MPI_Send(colBlocks[j].data(), count, MPI_DOUBLE, 0, static_cast<int>(j), comm);
        }
    }

    return full;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Choose a block width that gives each rank several block-columns for
    // reasonable load balance, while keeping blocks large enough to amortize
    // broadcast/synchronization overhead.
    size_t B = n / std::max<size_t>(1, static_cast<size_t>(size) * 4);
    B = std::clamp<size_t>(B, 32, 256);
    if (n == 0) {
        B = 1;
    } else {
        B = std::min(B, n);
    }
    const size_t nb = (n == 0) ? 0 : (n + B - 1) / B;

    std::vector<std::vector<double>> colBlocks(nb);

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrixDistributed(colBlocks, n, B, nb, rank, size);

    std::vector<double> A_orig;
    if (validate) {
        A_orig = gatherFullMatrix(colBlocks, n, B, nb, rank, size, MPI_COMM_WORLD, /*mirrorSymmetric=*/true);
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    bool success = choleskyDecompositionMPI(colBlocks, n, B, nb, rank, size, MPI_COMM_WORLD);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full result if needed for external validation or printing
    std::vector<double> A;
    if (printResults || validate) {
        A = gatherFullMatrix(colBlocks, n, B, nb, rank, size, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    // Validation
    int validationResult = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
                validationResult = 0;
            } else {
                printf("Validation: FAILED\n");
                validationResult = 1;
            }
        }
        MPI_Bcast(&validationResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validationResult;
    }

    MPI_Finalize();
    return 0;
}
