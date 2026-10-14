#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Rows are distributed cyclically: global row i is owned by rank i % size and is
// stored locally as row i / size. Cyclic ownership spreads the triangular workload
// evenly across ranks. Each rank stores its rows contiguously in row-major order.

struct RowLayout {
    int rank;
    int size;
    size_t n;

    // Number of rows owned by rank r
    size_t rowsOn(int r) const {
        return n > static_cast<size_t>(r) ? (n - r - 1) / size + 1 : 0;
    }
    size_t localRows() const { return rowsOn(rank); }
    // Global index of local row t
    size_t globalRow(size_t t) const { return static_cast<size_t>(rank) + t * size; }
};

// Calls visit(block, src, count) once for the rows of every rank. The block holds the
// `count` rows owned by rank `src`, row-major. Blocks circulate around a ring, so each
// rank only ever holds two blocks at a time.
template <typename Visit>
void forEachRowBlock(const std::vector<double>& mine, const RowLayout& layout, MPI_Comm comm, Visit visit) {
    const int P = layout.size;
    const int next = (layout.rank + 1) % P;
    const int prev = (layout.rank + P - 1) % P;
    const size_t n = layout.n;

    std::vector<double> cur(layout.rowsOn(0) * n);  // rank 0 owns the most rows
    std::vector<double> incoming(cur.size());
    std::copy(mine.begin(), mine.end(), cur.begin());

    int src = layout.rank;
    for (int step = 0; step < P; ++step) {
        visit(cur.data(), src, layout.rowsOn(src));
        if (step + 1 == P) break;

        const int nextSrc = (src + P - 1) % P;
        MPI_Sendrecv(cur.data(), static_cast<int>(layout.rowsOn(src) * n), MPI_DOUBLE, next, 0,
                     incoming.data(), static_cast<int>(layout.rowsOn(nextSrc) * n), MPI_DOUBLE, prev, 0,
                     comm, MPI_STATUS_IGNORE);
        std::swap(cur, incoming);
        src = nextSrc;
    }
}

// Column-oriented distributed factorization. For column j, the owner of row j finishes
// L[j][j] and broadcasts row j of L; every rank then updates its own rows below the
// diagonal. Each entry is computed with the same operations in the same order as the
// sequential row-wise algorithm, so the factor is bit-identical for any number of ranks.
bool choleskyDecomposition(std::vector<double>& local, const RowLayout& layout, MPI_Comm comm) {
    const size_t n = layout.n;
    const size_t P = layout.size;
    const size_t me = layout.rank;
    const size_t nLocal = layout.localRows();
    std::vector<double> pivot(n);  // row j of L, up to column j

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % P);
        if (me == static_cast<size_t>(owner)) {
            double* rowj = &local[(j / P) * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += rowj[k] * rowj[k];
            }
            const double val = rowj[j] - sum;
            if (val <= 0.0) {
                // Non-positive pivot is broadcast as-is and signals failure to every rank
                pivot[j] = val;
            } else {
                pivot[j] = sqrt(val);
                rowj[j] = pivot[j];
            }
            std::copy(rowj, rowj + j, pivot.begin());
        }

        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, comm);
        if (pivot[j] <= 0.0) {
            if (me == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        // Update owned rows below the diagonal:
        // L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]
        const double ljj = pivot[j];
        // First local row whose global index exceeds j
        for (size_t t = j >= me ? (j - me) / P + 1 : 0; t < nLocal; ++t) {
            double* row = &local[t * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * pivot[k];
            }
            row[j] = (row[j] - sum) / ljj;
        }
    }

    // Zero out upper triangular part
    for (size_t t = 0; t < nLocal; ++t) {
        double* row = &local[t * n];
        for (size_t j = layout.globalRow(t) + 1; j < n; ++j) {
            row[j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix, distributed by rows
void generatePositiveDefiniteMatrix(std::vector<double>& local, const RowLayout& layout, MPI_Comm comm) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    const size_t n = layout.n;
    const size_t P = layout.size;
    const size_t me = layout.rank;
    const size_t nLocal = layout.localRows();

    // Every rank walks the same random sequence but keeps only the rows of B it owns
    std::vector<double> B(nLocal * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        const double value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        const size_t row = i / n;
        if (row % P == me) {
            B[(row / P) * n + i % n] = value;
        }
    }

    // Compute A = B * B^T for owned rows, with every block of B passing by this rank
    local.assign(nLocal * n, 0.0);
    forEachRowBlock(B, layout, comm, [&](const double* block, int src, size_t count) {
        for (size_t t = 0; t < nLocal; ++t) {
            const double* bi = &B[t * n];
            double* ai = &local[t * n];
            for (size_t u = 0; u < count; ++u) {
                const double* bj = block + u * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += bi[k] * bj[k];
                }
                ai[static_cast<size_t>(src) + u * P] = sum;
            }
        }
    });

    // Add diagonal dominance to ensure positive definiteness
    for (size_t t = 0; t < nLocal; ++t) {
        local[t * n + layout.globalRow(t)] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const RowLayout& layout, MPI_Comm comm) {
    // Validate by computing L * L^T and comparing with original matrix
    const size_t n = layout.n;
    const size_t P = layout.size;
    const size_t nLocal = layout.localRows();

    double maxError = 0.0;
    double relError = 0.0;

    // Compare owned rows of L * L^T with the original matrix, with every block of L passing by
    forEachRowBlock(L, layout, comm, [&](const double* block, int src, size_t count) {
        for (size_t t = 0; t < nLocal; ++t) {
            const double* li = &L[t * n];
            for (size_t u = 0; u < count; ++u) {
                const size_t j = static_cast<size_t>(src) + u * P;
                const double* lj = block + u * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += li[k] * lj[k];
                }

                const double orig = A_orig[t * n + j];
                const double error = fabs(sum - orig);
                maxError = std::max(maxError, error);

                const double rel = error / (fabs(orig) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
    });

    double localErrors[2] = {maxError, relError};
    double globalErrors[2];
    MPI_Allreduce(localErrors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, comm);
    maxError = globalErrors[0];
    relError = globalErrors[1];

    if (layout.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (layout.rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
        return false;
    }

    return true;
}

// Gather all rows into one row-major matrix on rank 0 (empty on other ranks)
std::vector<double> gatherRows(const std::vector<double>& local, const RowLayout& layout, MPI_Comm comm) {
    const size_t n = layout.n;
    const int P = layout.size;

    std::vector<int> counts(P), displs(P);
    int total = 0;
    for (int r = 0; r < P; ++r) {
        counts[r] = static_cast<int>(layout.rowsOn(r) * n);
        displs[r] = total;
        total += counts[r];
    }

    std::vector<double> packed(layout.rank == 0 ? static_cast<size_t>(total) : 0);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    if (layout.rank != 0) {
        return {};
    }

    std::vector<double> full(n * n);
    for (size_t i = 0; i < n; ++i) {
        const size_t owner = i % P;
        const size_t t = i / P;
        std::copy_n(packed.begin() + displs[owner] + t * n, n, full.begin() + i * n);
    }
    return full;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const MPI_Comm comm = MPI_COMM_WORLD;

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const RowLayout layout{rank, size, n};

    // Allocate local rows
    std::vector<double> A(layout.localRows() * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, layout, comm);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    const bool success = choleskyDecomposition(A, layout, comm);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;  // The slowest rank determines the time
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    const long ms = static_cast<long>(elapsed * 1000.0);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", ms);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        const std::vector<double> L = gatherRows(A, layout, comm);
        if (rank == 0) {
            print_results(L, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateCholesky(A, A_orig, layout, comm);

        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int status = run(argc, argv);
    MPI_Finalize();
    return status;
}
