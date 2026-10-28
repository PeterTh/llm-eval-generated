#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed-memory data layout.
// Rows are dealt out cyclically: global row i lives on rank i % P as local row i / P.
// Cyclic dealing keeps the shrinking trailing work of the factorization balanced.
struct RowLayout {
    MPI_Comm comm;
    int rank;
    int size;
    size_t n;

    // Number of rows owned by rank r.
    size_t rowsOn(size_t r) const {
        const size_t P = static_cast<size_t>(size);
        return n > r ? (n - r - 1) / P + 1 : 0;
    }
    size_t localRows() const { return rowsOn(static_cast<size_t>(rank)); }
    int owner(size_t i) const { return static_cast<int>(i % static_cast<size_t>(size)); }
    size_t localIndex(size_t i) const { return i / static_cast<size_t>(size); }
    size_t globalRow(size_t m) const { return m * static_cast<size_t>(size) + static_cast<size_t>(rank); }
    // First local row index whose global row index is greater than j.
    size_t firstLocalAbove(size_t j) const {
        const size_t r = static_cast<size_t>(rank);
        return j < r ? 0 : (j - r) / static_cast<size_t>(size) + 1;
    }
};

// Dot product over k in [0, len), accumulated in ascending k as in the original code.
// Multiply and add are kept separate (the build uses -ffp-contract=off), so the result does not
// depend on whether the target CPU has fused multiply-add.
static inline double dot(const double* a, const double* b, size_t len) {
    double sum = 0.0;
    for (size_t k = 0; k < len; ++k) {
        sum += a[k] * b[k];
    }
    return sum;
}

// Four independent dot products evaluated together to hide floating-point add latency.
// Each individual sum is accumulated in exactly the same order as dot().
static inline void dot4(const double* const x[4], const double* const y[4], size_t len, double out[4]) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (size_t k = 0; k < len; ++k) {
        s0 += x[0][k] * y[0][k];
        s1 += x[1][k] * y[1][k];
        s2 += x[2][k] * y[2][k];
        s3 += x[3][k] * y[3][k];
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

// Cholesky decomposition, distributed by rows (left-looking, column order).
// Column j of L depends only on columns < j, and every entry is accumulated over k = 0..j-1
// in the same order as the original row-oriented code. The arithmetic is therefore the same for
// any number of ranks, and it matches the original algorithm up to compiler rounding choices.
//
// Step j:
//   1. The owner of row j has L[j][0..j-1] final; it finalizes L[j][j] and broadcasts
//      the row prefix L[j][0..j] to all ranks.
//   2. Every rank updates its own rows i > j: L[i][j] = (A[i][j] - sum_k L[i][k] L[j][k]) / L[j][j].
// Lookahead: the owner of row j+1 computes its column-j entry first and starts the nonblocking
// broadcast of row j+1, so that communication overlaps the remaining column-j work.
//
// Aloc holds this rank's rows (full length n, row-major); the factor overwrites the lower triangle
// in place and the upper triangle is left at zero.
static bool choleskyDecomposition(std::vector<double>& Aloc, const RowLayout& lay) {
    const size_t n = lay.n;
    if (n == 0) {
        return true;
    }
    const size_t rows = lay.localRows();

    // Row-prefix broadcast buffers (double-buffered for the lookahead).
    std::vector<double> buf[2] = {std::vector<double>(n), std::vector<double>(n)};

    // Owner-only: compute L[jn][jn] and write the row prefix L[jn][0..jn] into out.
    // On failure the (non-positive) value is passed along and checked by all ranks.
    auto finishRow = [&](size_t jn, double* out) {
        double* Lr = &Aloc[lay.localIndex(jn) * n];
        const double sum = dot(Lr, Lr, jn);
        const double val = Lr[jn] - sum;
        const double d = (val <= 0.0) ? val : std::sqrt(val);
        for (size_t k = 0; k < jn; ++k) {
            out[k] = Lr[k];
        }
        out[jn] = d;
        Lr[jn] = d;
    };

    // Update L[i][j] for local rows m in [mBegin, mEnd); cur is the broadcast row j prefix.
    auto updateRows = [&](size_t mBegin, size_t mEnd, size_t j, const double* cur) {
        size_t m = mBegin;
        for (; m + 4 <= mEnd; m += 4) {
            const double* x[4] = {&Aloc[m * n], &Aloc[(m + 1) * n], &Aloc[(m + 2) * n], &Aloc[(m + 3) * n]};
            const double* y[4] = {cur, cur, cur, cur};
            double o[4];
            dot4(x, y, j, o);
            for (size_t q = 0; q < 4; ++q) {
                double* Lr = &Aloc[(m + q) * n];
                Lr[j] = (Lr[j] - o[q]) / cur[j];
            }
        }
        for (; m < mEnd; ++m) {
            double* Lr = &Aloc[m * n];
            Lr[j] = (Lr[j] - dot(Lr, cur, j)) / cur[j];
        }
    };

    // Row 0 has no dependencies.
    if (lay.owner(0) == lay.rank) {
        finishRow(0, buf[0].data());
    }
    MPI_Bcast(buf[0].data(), 1, MPI_DOUBLE, lay.owner(0), lay.comm);

    MPI_Request req = MPI_REQUEST_NULL;
    for (size_t j = 0; j < n; ++j) {
        const double* cur = buf[j & 1].data();

        // All ranks hold the identical broadcast value, so they all abort at the same step.
        if (cur[j] <= 0.0) {
            if (lay.rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        const size_t jn = j + 1;
        const bool haveNext = jn < n;
        const bool ownNext = haveNext && lay.owner(jn) == lay.rank;
        size_t skipM = rows;  // local row already handled by the lookahead (if any)

        if (haveNext) {
            const int own = lay.owner(jn);
            double* nxt = buf[jn & 1].data();
            if (ownNext) {
                double* Lr = &Aloc[lay.localIndex(jn) * n];
                Lr[j] = (Lr[j] - dot(Lr, cur, j)) / cur[j];
                finishRow(jn, nxt);
                skipM = lay.localIndex(jn);
            }
            MPI_Ibcast(nxt, static_cast<int>(jn + 1), MPI_DOUBLE, own, lay.comm, &req);
        }

        const size_t mBegin = lay.firstLocalAbove(j);
        updateRows(mBegin, skipM, j, cur);
        updateRows(skipM + 1, rows, j, cur);

        if (haveNext) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
    }
    return true;
}

// Generate a symmetric positive definite matrix, distributed by rows.
// Method: A = B * B^T with B random, then add n to the diagonal.
// Only the lower triangle (j <= i) is computed, since A is exactly symmetric in floating point
// (the products and summation order for A[i][j] and A[j][i] are identical).
// B rows are circulated around a ring so no rank ever holds the full matrix.
static void generatePositiveDefiniteMatrix(std::vector<double>& Aloc, const RowLayout& lay) {
    const size_t n = lay.n;
    const size_t P = static_cast<size_t>(lay.size);
    const size_t rows = lay.localRows();
    const size_t maxRows = lay.rowsOn(0);

    // Local rows of B. The random stream is identical on every rank and is consumed in
    // the original row-major order; each rank keeps only the rows it owns.
    std::vector<double> Bloc(rows * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const bool mine = lay.owner(i) == lay.rank;
        for (size_t k = 0; k < n; ++k) {
            const double v = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            if (mine) {
                Bloc[lay.localIndex(i) * n + k] = v;
            }
        }
    }

    Aloc.assign(rows * n, 0.0);

    // Ring: at step t, the block currently held belongs to rank s = (rank - t) mod P.
    std::vector<double> cur(maxRows * n), nxt(maxRows * n);
    std::copy(Bloc.begin(), Bloc.end(), cur.begin());
    const int right = (lay.rank + 1) % lay.size;
    const int left = (lay.rank + lay.size - 1) % lay.size;
    const int blockCount = static_cast<int>(maxRows * n);

    for (int t = 0; t < lay.size; ++t) {
        const size_t s = static_cast<size_t>((lay.rank + lay.size - t) % lay.size);
        const size_t cntS = lay.rowsOn(s);

        for (size_t m = 0; m < rows; ++m) {
            const size_t i = lay.globalRow(m);
            if (i < s) {
                continue;
            }
            // Block rows are j = s + mm * P; keep those with j <= i.
            const size_t mmax = std::min(cntS, (i - s) / P + 1);
            const double* a = &Bloc[m * n];
            double* out = &Aloc[m * n];

            size_t mm = 0;
            for (; mm + 4 <= mmax; mm += 4) {
                const double* x[4] = {a, a, a, a};
                const double* y[4] = {&cur[mm * n], &cur[(mm + 1) * n], &cur[(mm + 2) * n], &cur[(mm + 3) * n]};
                double o[4];
                dot4(x, y, n, o);
                for (size_t q = 0; q < 4; ++q) {
                    out[s + (mm + q) * P] = o[q];
                }
            }
            for (; mm < mmax; ++mm) {
                out[s + mm * P] = dot(a, &cur[mm * n], n);
            }
        }

        if (t + 1 < lay.size) {
            MPI_Sendrecv(cur.data(), blockCount, MPI_DOUBLE, right, 0,
                         nxt.data(), blockCount, MPI_DOUBLE, left, 0,
                         lay.comm, MPI_STATUS_IGNORE);
            cur.swap(nxt);
        }
    }

    for (size_t m = 0; m < rows; ++m) {
        Aloc[m * n + lay.globalRow(m)] += n;
    }
}

// Gather the row-distributed matrix into a full row-major n x n matrix on rank 0.
// Other ranks receive an empty vector.
static std::vector<double> gatherMatrix(const std::vector<double>& Aloc, const RowLayout& lay) {
    const size_t n = lay.n;
    const size_t P = static_cast<size_t>(lay.size);
    std::vector<int> counts(lay.size), displs(lay.size);
    int off = 0;
    for (int s = 0; s < lay.size; ++s) {
        counts[s] = static_cast<int>(lay.rowsOn(static_cast<size_t>(s)) * n);
        displs[s] = off;
        off += counts[s];
    }

    std::vector<double> tmp;
    if (lay.rank == 0) {
        tmp.resize(n * n);
    }
    MPI_Gatherv(Aloc.data(), static_cast<int>(Aloc.size()), MPI_DOUBLE,
                tmp.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, lay.comm);

    if (lay.rank != 0) {
        return {};
    }
    std::vector<double> full(n * n);
    for (size_t s = 0; s < P; ++s) {
        const size_t cnt = lay.rowsOn(s);
        for (size_t mm = 0; mm < cnt; ++mm) {
            const size_t i = s + mm * P;
            std::copy(tmp.begin() + displs[s] + mm * n, tmp.begin() + displs[s] + (mm + 1) * n, full.begin() + i * n);
        }
    }
    return full;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    RowLayout lay{};
    lay.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(lay.comm, &lay.rank);
    MPI_Comm_size(lay.comm, &lay.size);
    const bool root = lay.rank == 0;

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks exit consistently)
    int earlyExit = -1;
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
            earlyExit = 0;
            break;
        } else {
            if (root) {
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

    lay.n = n;

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    // Each rank holds only its rows of A
    std::vector<double> Aloc;
    generatePositiveDefiniteMatrix(Aloc, lay);

    std::vector<double> A_orig_loc;
    if (validate) {
        A_orig_loc = Aloc; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (root) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(lay.comm);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(Aloc, lay);

    MPI_Barrier(lay.comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (root) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    int code = 0;
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> L;
    if (printResults || validate) {
        L = gatherMatrix(Aloc, lay);
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(L, "CholeskyL");
    }

    // Validation
    if (validate) {
        std::vector<double> A_orig = gatherMatrix(A_orig_loc, lay);
        if (root) {
            // Restore the (exactly symmetric) upper triangle of the original matrix
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    A_orig[i * n + j] = A_orig[j * n + i];
                }
            }
            printf("Validating result...\n");
            bool valid = validateCholesky(L, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                code = 1;
            }
        }
    }

    MPI_Bcast(&code, 1, MPI_INT, 0, lay.comm);
    MPI_Finalize();
    return code;
}
