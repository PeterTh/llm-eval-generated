#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition (MPI, C API).
//
// Decomposes a symmetric positive definite matrix A into L * L^T, L lower triangular.
//
// Data distribution: rows are distributed cyclically (global row i is owned by
// rank i % P), which balances the triangular workload. Each rank stores its rows
// of the matrix M (which is overwritten in place by L) and the running partial sums
// S[i][j] = sum_{k<c0} L[i][k] * L[j][k] for all processed column panels.
//
// Algorithm: column panels of width kPanelWidth are processed left to right.
//   1. Panel rows of (A, S) are gathered; the small diagonal block is then computed
//      redundantly on every rank (deterministically, so all ranks agree on failure).
//   2. Owners compute their panel-column entries of L below the diagonal block.
//   3. The panel of L is allgathered and every rank adds its panel contribution
//      to the partial sums of its rows.
//
// Every element accumulates its dot product in the same order (k = 0, 1, ...) as
// the sequential algorithm, so the numerical results match the sequential code.

namespace {

// Number of columns processed per communication step.
constexpr size_t kPanelWidth = 64;

// Cyclic row distribution: global row i is owned by rank i % P.
struct RowDist {
    size_t n;     // matrix dimension
    size_t rank;  // this process
    size_t P;     // number of processes
    size_t nloc;  // number of rows owned by this process

    size_t global(size_t t) const { return rank + t * P; }
    // Number of rows in [0, x) owned by process q.
    size_t countBelow(size_t x, size_t q) const { return x > q ? (x - q + P - 1) / P : 0; }
    // Number of rows in [lo, hi) owned by process q.
    size_t countOwned(size_t lo, size_t hi, size_t q) const {
        return countBelow(hi, q) - countBelow(lo, q);
    }
    // First row >= lo owned by process q (may be >= the upper bound of the range).
    size_t firstOwnedFrom(size_t lo, size_t q) const { return lo + (q + P - lo % P) % P; }
};

RowDist makeRowDist(size_t n, int rank, int nprocs) {
    RowDist d;
    d.n = n;
    d.rank = static_cast<size_t>(rank);
    d.P = static_cast<size_t>(nprocs);
    d.nloc = d.rank < n ? (n - d.rank - 1) / d.P + 1 : 0;
    return d;
}

// Decomposes the distributed matrix in place (M holds this rank's rows, row-major
// with full length n). S must be zero-initialized with the same shape as M.
// Returns false if the matrix is not positive definite (identically on all ranks).
bool choleskyDecomposition(std::vector<double>& M, std::vector<double>& S, const RowDist& d,
                           MPI_Comm comm) {
    const size_t n = d.n;
    const size_t P = d.P;
    const size_t b = kPanelWidth;

    std::vector<int> counts(P), displs(P);
    std::vector<double> sendBuf(2 * b * b), recvBuf(2 * b * b);
    std::vector<double> Ab(b * b), Sb(b * b), Lb(b * b);
    std::vector<double> gSend(d.nloc * b), gRecv(n * b);
    std::vector<double> LT(b * n);  // panel of L, transposed: LT[jj * n + i] = L[i][c0 + jj]

    for (size_t c0 = 0; c0 < n; c0 += b) {
        const size_t c1 = std::min(n, c0 + b);
        const size_t w = c1 - c0;

        // ---- Step 1: gather (A, S) values of the panel rows [c0, c1) on all ranks.
        {
            size_t pos = 0;
            for (size_t i = d.firstOwnedFrom(c0, d.rank); i < c1; i += P) {
                const size_t t = (i - d.rank) / P;
                std::copy_n(&M[t * n + c0], w, &sendBuf[pos]);
                std::copy_n(&S[t * n + c0], w, &sendBuf[pos + w]);
                pos += 2 * w;
            }
            for (size_t q = 0; q < P; ++q) {
                counts[q] = static_cast<int>(2 * w * d.countOwned(c0, c1, q));
            }
            displs[0] = 0;
            for (size_t q = 1; q < P; ++q) displs[q] = displs[q - 1] + counts[q - 1];

            MPI_Allgatherv(sendBuf.data(), counts[d.rank], MPI_DOUBLE, recvBuf.data(),
                           counts.data(), displs.data(), MPI_DOUBLE, comm);

            for (size_t q = 0; q < P; ++q) {
                size_t p = static_cast<size_t>(displs[q]);
                for (size_t i = d.firstOwnedFrom(c0, q); i < c1; i += P) {
                    const size_t r = i - c0;
                    std::copy_n(&recvBuf[p], w, &Ab[r * w]);
                    std::copy_n(&recvBuf[p + w], w, &Sb[r * w]);
                    p += 2 * w;
                }
            }
        }

        // ---- Step 2: diagonal block (rows and columns in [c0, c1)), computed on every rank.
        for (size_t i = c0; i < c1; ++i) {
            const size_t ri = (i - c0) * w;
            for (size_t j = c0; j <= i; ++j) {
                const size_t rj = (j - c0) * w;
                double sum = Sb[ri + (j - c0)];
                for (size_t k = c0; k < j; ++k) {
                    sum += Lb[ri + (k - c0)] * Lb[rj + (k - c0)];
                }
                if (i == j) {
                    const double val = Ab[ri + (j - c0)] - sum;
                    if (val <= 0.0) {
                        if (d.rank == 0) {
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        }
                        return false;
                    }
                    Lb[ri + (j - c0)] = sqrt(val);
                } else {
                    Lb[ri + (j - c0)] = (Ab[ri + (j - c0)] - sum) / Lb[rj + (j - c0)];
                }
            }
        }

        // ---- Step 3: compute panel columns of the locally owned rows i >= c0.
        for (size_t t = 0; t < d.nloc; ++t) {
            const size_t i = d.global(t);
            if (i < c0) continue;
            double* row = &M[t * n];
            if (i < c1) {
                // Diagonal-block row: take L from the block, zero the upper part.
                for (size_t j = c0; j <= i; ++j) row[j] = Lb[(i - c0) * w + (j - c0)];
                for (size_t j = i + 1; j < n; ++j) row[j] = 0.0;
            } else {
                // Row below the panel: left-looking update of panel columns.
                const double* srow = &S[t * n];
                for (size_t j = c0; j < c1; ++j) {
                    const size_t rj = (j - c0) * w;
                    double sum = srow[j];
                    for (size_t k = c0; k < j; ++k) {
                        sum += row[k] * Lb[rj + (k - c0)];
                    }
                    row[j] = (row[j] - sum) / Lb[rj + (j - c0)];
                }
            }
        }

        // ---- Step 4: allgather the panel of L (rows i >= c0) to all ranks.
        {
            size_t pos = 0;
            for (size_t t = 0; t < d.nloc; ++t) {
                const size_t i = d.global(t);
                if (i < c0) continue;
                std::copy_n(&M[t * n + c0], w, &gSend[pos]);
                pos += w;
            }
            for (size_t q = 0; q < P; ++q) {
                counts[q] = static_cast<int>(w * d.countOwned(c0, n, q));
            }
            displs[0] = 0;
            for (size_t q = 1; q < P; ++q) displs[q] = displs[q - 1] + counts[q - 1];

            MPI_Allgatherv(gSend.data(), counts[d.rank], MPI_DOUBLE, gRecv.data(), counts.data(),
                           displs.data(), MPI_DOUBLE, comm);

            for (size_t q = 0; q < P; ++q) {
                size_t p = static_cast<size_t>(displs[q]);
                for (size_t i = d.firstOwnedFrom(c0, q); i < n; i += P) {
                    for (size_t jj = 0; jj < w; ++jj) LT[jj * n + i] = gRecv[p + jj];
                    p += w;
                }
            }
        }

        // ---- Step 5: add this panel's contribution to the partial sums of rows i >= c1.
        // Per element (i, k) the terms are added in increasing j, as in the sequential code.
        for (size_t t = 0; t < d.nloc; ++t) {
            const size_t i = d.global(t);
            if (i < c1) continue;
            double* srow = &S[t * n];
            for (size_t jj = 0; jj < w; ++jj) {
                const double lij = LT[jj * n + i];
                const double* lt = &LT[jj * n];
                for (size_t k = c1; k <= i; ++k) {
                    srow[k] += lij * lt[k];
                }
            }
        }
    }

    return true;
}

// Generates the same matrix as the sequential generator (A = B * B^T + n*I, B from
// rand_r with seed 42 filled in row-major order), storing only the locally owned rows.
void generatePositiveDefiniteMatrix(std::vector<double>& M, const RowDist& d) {
    const size_t n = d.n;
    const size_t P = d.P;

    std::vector<double> Bown(d.nloc * n);  // locally owned rows of B
    std::vector<double> row(n);
    unsigned int seed = 42;
    auto fillRow = [&]() {
        for (size_t k = 0; k < n; ++k) {
            row[k] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
    };

    // Pass 1: keep the locally owned rows of B.
    for (size_t j = 0; j < n; ++j) {
        fillRow();
        if (j % P == d.rank) {
            std::copy(row.begin(), row.end(), &Bown[(j / P) * n]);
        }
    }

    // Pass 2: regenerate the stream and compute column j of A for the owned rows.
    seed = 42;
    for (size_t j = 0; j < n; ++j) {
        fillRow();
        for (size_t t = 0; t < d.nloc; ++t) {
            const double* bi = &Bown[t * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += bi[k] * row[k];
            }
            M[t * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness.
    for (size_t t = 0; t < d.nloc; ++t) {
        M[t * n + d.global(t)] += n;
    }
}

// Gathers the full n x n matrix onto the root rank (other ranks get an empty vector).
std::vector<double> gatherFullMatrix(const std::vector<double>& M, const RowDist& d,
                                     MPI_Comm comm) {
    const size_t n = d.n;
    const size_t P = d.P;
    const bool isRoot = d.rank == 0;

    std::vector<int> counts(P), displs(P);
    for (size_t q = 0; q < P; ++q) {
        counts[q] = static_cast<int>(d.countOwned(0, n, q) * n);
    }
    displs[0] = 0;
    for (size_t q = 1; q < P; ++q) displs[q] = displs[q - 1] + counts[q - 1];

    std::vector<double> recv(isRoot ? n * n : 0);
    MPI_Gatherv(M.data(), static_cast<int>(M.size()), MPI_DOUBLE, recv.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, comm);

    if (!isRoot) return {};

    std::vector<double> full(n * n);
    for (size_t q = 0; q < P; ++q) {
        size_t p = static_cast<size_t>(displs[q]);
        for (size_t i = q; i < n; i += P) {
            std::copy_n(&recv[p], n, &full[i * n]);
            p += n;
        }
    }
    return full;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const RowDist& d, MPI_Comm comm) {
    // Validate by computing L * L^T and comparing with original matrix.
    // Columns of L are streamed by broadcasting one row of L at a time.
    const size_t n = d.n;
    const size_t P = d.P;

    std::vector<double> Lrow(n);
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t j = 0; j < n; ++j) {
        const size_t root = j % P;
        if (root == d.rank) {
            std::copy_n(&L[(j / P) * n], n, Lrow.begin());
        }
        MPI_Bcast(Lrow.data(), static_cast<int>(n), MPI_DOUBLE, static_cast<int>(root), comm);

        for (size_t t = 0; t < d.nloc; ++t) {
            const double* li = &L[t * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += li[k] * Lrow[k];
            }
            const double a = A_orig[t * n + j];
            const double error = fabs(sum - a);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(a) + 1e-10));
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (d.rank == 0) printf("Validation failed: relative error too large\n");
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool isRoot = rank == 0;
    const MPI_Comm comm = MPI_COMM_WORLD;

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
    }

    const RowDist d = makeRowDist(n, rank, nprocs);

    // Allocate local rows of the matrix
    std::vector<double> A(d.nloc * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (isRoot) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, d);

    if (validate) {
        A_orig = A;  // Save original local rows for validation
    }

    // Perform Cholesky decomposition
    if (isRoot) printf("Computing Cholesky decomposition...\n");
    std::vector<double> S(d.nloc * n, 0.0);

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, S, d, comm);
    const double end = MPI_Wtime();

    // Report the slowest rank's time
    double elapsed = end - start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (!success) {
        if (isRoot) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    const long durationMs = static_cast<long>(elapsed * 1000.0);
    if (isRoot) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full = gatherFullMatrix(A, d, comm);
        if (isRoot) print_results(full, "CholeskyL");
    }

    int exitCode = 0;
    // Validation
    if (validate) {
        if (isRoot) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, d, comm);

        if (valid) {
            if (isRoot) printf("Validation: PASSED\n");
        } else {
            if (isRoot) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
