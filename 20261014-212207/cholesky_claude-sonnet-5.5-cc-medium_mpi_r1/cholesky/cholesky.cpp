#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, left-looking, row-cyclic distribution).
// Row i of the matrix is owned by rank i % P. Column j of L is computed after the
// owner of row j broadcasts L[j][0..j]. The dot products keep the sequential
// k-order of the reference implementation, so results are bit-identical to it.
// The broadcast of row j+1 is overlapped with the remaining work of column j.

static int g_rank = 0, g_size = 1;

static inline size_t localRows(size_t n) { return (n + g_size - 1 - g_rank) / g_size; }

// dot products of 4 rows against one row, k in [0, len), sequential order per row
static inline void dot4(const double* r0, const double* r1, const double* r2, const double* r3,
                        const double* x, size_t len, double* out) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (size_t k = 0; k < len; ++k) {
        const double xv = x[k];
        s0 += r0[k] * xv;
        s1 += r1[k] * xv;
        s2 += r2[k] * xv;
        s3 += r3[k] * xv;
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

static inline double dot1(const double* r, const double* x, size_t len) {
    double s = 0.0;
    for (size_t k = 0; k < len; ++k) s += r[k] * x[k];
    return s;
}

// A: local rows (localRows(n) x n, row-major). Returns false on failure (same on all ranks).
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int P = g_size, r = g_rank;
    std::vector<double> bufs[2] = {std::vector<double>(n), std::vector<double>(n)};
    const size_t nloc = localRows(n);
    auto ownedRowPtr = [&](size_t i) { return A.data() + (i / P) * n; };

    // diagonal of owned row j; returns false if not positive definite
    auto diagonal = [&](size_t j) {
        double* row = ownedRowPtr(j);
        const double val = row[j] - dot1(row, row, j);
        if (val <= 0.0) {
            row[j] = 0.0;  // failure marker for receivers
            return false;
        }
        row[j] = sqrt(val);
        return true;
    };

    MPI_Request req = MPI_REQUEST_NULL;
    bool failed = false;
    size_t failIdx = 0;

    if (n == 0) return true;
    // row 0
    if (r == 0) {
        if (!diagonal(0)) { failed = true; }
    }
    {
        double* buf = (r == 0) ? ownedRowPtr(0) : bufs[0].data();
        MPI_Ibcast(buf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD, &req);
    }

    for (size_t j = 0; j < n; ++j) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        const int owner = (int)(j % P);
        const double* Lj = (r == owner) ? ownedRowPtr(j) : bufs[j & 1].data();
        if (!(Lj[j] > 0.0)) { failed = true; failIdx = j; break; }
        const double d = Lj[j];

        // first local row with global index > j
        size_t li0 = 0;
        if ((size_t)r <= j) li0 = (j - r) / P + 1;

        const bool last = (j + 1 == n);
        auto computeOne = [&](size_t li) {
            double* row = A.data() + li * n;
            row[j] = (row[j] - dot1(row, Lj, j)) / d;
        };

        size_t li = li0;
        if (!last) {
            const size_t nxt = j + 1;
            const int nowner = (int)(nxt % P);
            if (r == nowner) {
                // compute needed element of next row first, then post its broadcast
                computeOne(nxt / P);
                li = nxt / P + 1;
                bool ok = diagonal(nxt);
                (void)ok;
                MPI_Ibcast(ownedRowPtr(nxt), (int)(nxt + 1), MPI_DOUBLE, nowner, MPI_COMM_WORLD, &req);
            } else {
                MPI_Ibcast(bufs[nxt & 1].data(), (int)(nxt + 1), MPI_DOUBLE, nowner, MPI_COMM_WORLD, &req);
            }
        }

        for (; li + 4 <= nloc; li += 4) {
            double* r0 = A.data() + li * n;
            double out[4];
            dot4(r0, r0 + n, r0 + 2 * n, r0 + 3 * n, Lj, j, out);
            r0[j] = (r0[j] - out[0]) / d;
            r0[n + j] = (r0[n + j] - out[1]) / d;
            r0[2 * n + j] = (r0[2 * n + j] - out[2]) / d;
            r0[3 * n + j] = (r0[3 * n + j] - out[3]) / d;
        }
        for (; li < nloc; ++li) computeOne(li);
    }

    if (failed) {
        if (req != MPI_REQUEST_NULL) {
            // the pending broadcast was posted by every rank; complete it
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
        if (r == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", failIdx);
        return false;
    }

    // Zero out upper triangular part
    for (size_t li = 0; li < nloc; ++li) {
        const size_t i = li * P + r;
        for (size_t j = i + 1; j < n; ++j) A[li * n + j] = 0.0;
    }
    return true;
}

// Generate own rows of a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // A = B * B^T + n*I, B random (generated identically on every rank)
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const size_t nloc = localRows(n);
    A.assign(nloc * n, 0.0);
    for (size_t li = 0; li < nloc; ++li) {
        const size_t i = li * g_size + g_rank;
        const double* bi = &B[i * n];
        double* a = &A[li * n];
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            dot4(&B[j * n], &B[(j + 1) * n], &B[(j + 2) * n], &B[(j + 3) * n], bi, n, a + j);
        }
        for (; j < n; ++j) a[j] = dot1(&B[j * n], bi, n);
        a[i] += n;
    }
}

// Gather distributed rows into a full row-major matrix on all ranks (all=true) or on rank 0
void gatherMatrix(const std::vector<double>& Aloc, std::vector<double>& full, const size_t n, bool all) {
    const size_t maxRows = (n + g_size - 1) / g_size;
    std::vector<double> pad(maxRows * n, 0.0);
    std::copy(Aloc.begin(), Aloc.end(), pad.begin());
    std::vector<double> recv;
    if (all || g_rank == 0) recv.resize(maxRows * n * g_size);
    const int cnt = (int)(maxRows * n);
    if (all) {
        MPI_Allgather(pad.data(), cnt, MPI_DOUBLE, recv.data(), cnt, MPI_DOUBLE, MPI_COMM_WORLD);
    } else {
        MPI_Gather(pad.data(), cnt, MPI_DOUBLE, recv.data(), cnt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (all || g_rank == 0) {
        full.resize(n * n);
        for (size_t i = 0; i < n; ++i) {
            const double* src = &recv[((i % g_size) * maxRows + i / g_size) * n];
            std::copy(src, src + n, &full[i * n]);
        }
    }
}

bool validateCholesky(const std::vector<double>& Lloc, const std::vector<double>& A_orig_loc, const size_t n) {
    // Validate by computing L * L^T and comparing with original (distributed by rows)
    std::vector<double> L;
    gatherMatrix(Lloc, L, n, true);

    const size_t nloc = localRows(n);
    double maxError = 0.0, relError = 0.0;
    for (size_t li = 0; li < nloc; ++li) {
        const size_t i = li * g_size + g_rank;
        for (size_t j = 0; j < n; ++j) {
            // L is lower triangular: terms beyond min(i,j) are exactly zero
            const size_t len = std::min(i, j) + 1;
            const double sum = dot1(&L[i * n], &L[j * n], len);
            const double orig = A_orig_loc[li * n + j];
            const double error = fabs(sum - orig);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(orig) + 1e-10));
        }
    }
    double gm, gr;
    MPI_Allreduce(&maxError, &gm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&relError, &gr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", gm);
        printf("Max relative error: %.10e\n", gr);
    }
    if (gr > 1e-6) {
        if (g_rank == 0) printf("Validation failed: relative error too large\n");
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

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
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> A;
    std::vector<double> A_orig;
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A;  // Save original (local rows) for validation
    }

    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full;
        gatherMatrix(A, full, n, false);
        if (g_rank == 0) print_results(full, "CholeskyL");
    }

    int rc = 0;
    if (validate) {
        if (g_rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        if (g_rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        rc = valid ? 0 : 1;
    }

    MPI_Finalize();
    return rc;
}
