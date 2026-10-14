#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition (MPI)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Rows are distributed cyclically: global row i lives on rank i % P, at local
// index i / P. The factorization is left-looking by column. At step j the owner
// of row j holds L[j][0..j-1] and the pivot value A[j][j] - sum_k L[j][k]^2 and
// broadcasts them. Every rank then finishes column j for its own rows. The
// broadcast of row j+1 is posted before the bulk of column j is updated, so the
// communication overlaps with computation.
//
// Every element is computed with the same floating-point operations in the same
// order as the sequential algorithm, so results are bit-identical for any number
// of ranks. Multiply-accumulates are written as std::fma and the build sets
// -ffp-contract=off, so the compiler cannot fuse different operations depending on
// how the loops are split.

namespace {

// Rows owned by `rank` when n rows are dealt cyclically over `size` ranks
size_t localRowsFor(size_t n, size_t rank, size_t size) {
    return n > rank ? (n - rank - 1) / size + 1 : 0;
}

struct Distribution {
    size_t n;
    size_t rank;
    size_t size;

    size_t localRows() const { return localRowsFor(n, rank, size); }
    size_t owner(size_t i) const { return i % size; }
    size_t localIndex(size_t i) const { return i / size; }   // valid on the owner of i
    size_t globalRow(size_t m) const { return rank + m * size; }

    // First local row whose global index is greater than i
    size_t firstLocalRowAbove(size_t i) const {
        return i < rank ? 0 : (i - rank) / size + 1;
    }
};

// For local rows [m0, m1): L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]
// `piv` holds row j of L. Four rows are interleaved so that independent dot products
// hide floating-point latency. Each sum still accumulates in ascending k.
void updateColumn(double* A, size_t n, size_t m0, size_t m1, size_t j, const double* piv) {
    size_t m = m0;
    for (; m + 4 <= m1; m += 4) {
        double* r0 = A + m * n;
        double* r1 = r0 + n;
        double* r2 = r1 + n;
        double* r3 = r2 + n;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (size_t k = 0; k < j; ++k) {
            const double p = piv[k];
            s0 = std::fma(r0[k], p, s0);
            s1 = std::fma(r1[k], p, s1);
            s2 = std::fma(r2[k], p, s2);
            s3 = std::fma(r3[k], p, s3);
        }
        r0[j] = (r0[j] - s0) / piv[j];
        r1[j] = (r1[j] - s1) / piv[j];
        r2[j] = (r2[j] - s2) / piv[j];
        r3[j] = (r3[j] - s3) / piv[j];
    }
    for (; m < m1; ++m) {
        double* r = A + m * n;
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum = std::fma(r[k], piv[k], sum);
        }
        r[j] = (r[j] - sum) / piv[j];
    }
}

// Returns false if the matrix is not positive definite. The decision is identical on every rank.
bool choleskyDecomposition(std::vector<double>& Aloc, const Distribution& d, MPI_Comm comm) {
    const size_t n = d.n;
    std::vector<double> buf[2] = {std::vector<double>(n), std::vector<double>(n)};
    MPI_Request req = MPI_REQUEST_NULL;

    // The owner of row r copies L[r][0..r-1] and its pivot value into buf[r & 1] and
    // broadcasts r + 1 values. Non-owners receive into the same buffer.
    auto postRow = [&](size_t r) {
        double* b = buf[r & 1].data();
        if (d.owner(r) == d.rank) {
            const double* row = &Aloc[d.localIndex(r) * n];
            std::copy(row, row + r, b);
            double sum = 0.0;
            for (size_t k = 0; k < r; ++k) {
                sum = std::fma(b[k], b[k], sum);
            }
            b[r] = row[r] - sum;
        }
        MPI_Ibcast(b, static_cast<int>(r + 1), MPI_DOUBLE, static_cast<int>(d.owner(r)), comm, &req);
    };

    if (n > 0) {
        postRow(0);
    }

    for (size_t j = 0; j < n; ++j) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        double* piv = buf[j & 1].data();

        // Matrix is not positive definite
        if (piv[j] <= 0.0) {
            if (d.rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }
        piv[j] = sqrt(piv[j]);

        if (d.owner(j) == d.rank) {
            double* row = &Aloc[d.localIndex(j) * n];
            row[j] = piv[j];
            // Zero out upper triangular part
            std::fill(row + j + 1, row + n, 0.0);
        }

        // The owner of row j+1 computes its column j first, so its broadcast can start early
        if (j + 1 < n) {
            const size_t r = j + 1;
            if (d.owner(r) == d.rank) {
                const size_t m = d.localIndex(r);
                updateColumn(Aloc.data(), n, m, m + 1, j, piv);
            }
            postRow(r);
        }

        // All remaining local rows below j+1
        updateColumn(Aloc.data(), n, d.firstLocalRowAbove(j + 1), d.localRows(), j, piv);
    }

    return true;
}

// Generates the same matrix as the sequential code: A = B * B^T, then add n to the
// diagonal, where B is filled row-major from rand_r with seed 42. Every rank replays
// the random stream but stores only its own rows. Only the lower triangle is
// computed unless fullRows is set, since the decomposition never reads the upper part.
void generatePositiveDefiniteMatrix(std::vector<double>& Aloc, const Distribution& d, bool fullRows) {
    const size_t n = d.n;
    const size_t rows = d.localRows();
    unsigned int seed = 42;

    auto drawRow = [&seed, n](double* row) {
        for (size_t k = 0; k < n; ++k) {
            row[k] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
    };

    // Pass 1: keep the locally owned rows of B
    std::vector<double> Bloc(rows * n);
    std::vector<double> scratch(n);
    for (size_t r = 0; r < n; ++r) {
        drawRow(d.owner(r) == d.rank ? &Bloc[d.localIndex(r) * n] : scratch.data());
    }

    // Pass 2: stream all rows of B in blocks. Each local row is reused across a block.
    Aloc.assign(rows * n, 0.0);
    seed = 42;
    constexpr size_t kBlock = 16;
    std::vector<double> block(kBlock * n);
    for (size_t j0 = 0; j0 < n; j0 += kBlock) {
        const size_t jn = std::min(kBlock, n - j0);
        for (size_t jj = 0; jj < jn; ++jj) {
            drawRow(&block[jj * n]);
        }
        for (size_t m = 0; m < rows; ++m) {
            const size_t i = d.globalRow(m);
            const double* bi = &Bloc[m * n];
            for (size_t jj = 0; jj < jn; ++jj) {
                const size_t j = j0 + jj;
                if (!fullRows && j > i) {
                    continue;
                }
                const double* bj = &block[jj * n];
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum = std::fma(bi[k], bj[k], sum);
                }
                Aloc[m * n + j] = sum;
                if (i == j) {
                    Aloc[m * n + j] += n;
                }
            }
        }
    }
}

// Gathers the distributed matrix into a full row-major matrix on rank 0. Other ranks get an empty vector.
std::vector<double> gatherMatrix(const std::vector<double>& Aloc, const Distribution& d, MPI_Comm comm) {
    const size_t n = d.n;
    std::vector<int> counts(d.size), displs(d.size);
    int offset = 0;
    for (size_t r = 0; r < d.size; ++r) {
        counts[r] = static_cast<int>(localRowsFor(n, r, d.size) * n);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> packed(d.rank == 0 ? n * n : 0);
    MPI_Gatherv(Aloc.data(), static_cast<int>(Aloc.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    if (d.rank != 0) {
        return {};
    }

    std::vector<double> full(n * n);
    for (size_t r = 0; r < d.size; ++r) {
        const size_t rows = localRowsFor(n, r, d.size);
        for (size_t m = 0; m < rows; ++m) {
            const size_t i = r + m * d.size;
            std::copy(packed.begin() + displs[r] + m * n, packed.begin() + displs[r] + (m + 1) * n,
                      full.begin() + i * n);
        }
    }
    return full;
}

// Validates L against the original matrix. Each row of L is broadcast in turn, and every rank
// reconstructs its own rows of L * L^T. Returns the same verdict on every rank.
bool validateCholesky(const std::vector<double>& Lloc, const std::vector<double>& Aorig_loc,
                      const Distribution& d, MPI_Comm comm) {
    const size_t n = d.n;
    const size_t rows = d.localRows();
    std::vector<double> Lrow(n);

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t j = 0; j < n; ++j) {
        if (d.owner(j) == d.rank) {
            const double* row = &Lloc[d.localIndex(j) * n];
            std::copy(row, row + n, Lrow.begin());
        }
        MPI_Bcast(Lrow.data(), static_cast<int>(n), MPI_DOUBLE, static_cast<int>(d.owner(j)), comm);

        for (size_t m = 0; m < rows; ++m) {
            const double* li = &Lloc[m * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum = std::fma(li[k], Lrow[k], sum);
            }

            const size_t idx = m * n + j;
            const double error = fabs(sum - Aorig_loc[idx]);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(Aorig_loc[idx]) + 1e-10);
            relError = std::max(relError, rel);
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
        if (d.rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
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

int run(int argc, char** argv, int rank, int size) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const bool root = rank == 0;

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
            if (root) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    const Distribution d{n, static_cast<size_t>(rank), static_cast<size_t>(size)};

    // Generate positive definite matrix (local rows only)
    std::vector<double> A;
    generatePositiveDefiniteMatrix(A, d, validate);

    std::vector<double> A_orig;
    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (root) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, d, comm);

    auto end = std::chrono::high_resolution_clock::now();

    // The slowest rank determines the time
    double seconds = std::chrono::duration<double>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(seconds));

    if (!success) {
        if (root) {
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> L = gatherMatrix(A, d, comm);
        if (root) {
            print_results(L, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, d, comm);

        if (root) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = run(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
