#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI).
// Rows are distributed cyclically (row i lives on rank i % P). For each column j the
// owner of row j finalizes it and broadcasts it; every rank then computes column j of
// its own rows below. Dot products keep the sequential k order of the original code.
// The broadcast of row j+1 is started before the bulk update of column j (look-ahead).

static int g_rank = 0, g_size = 1;

// Local rows are stored compactly: local row r <-> global row r * P + rank. Width n.
// Returns false on a non-positive-definite matrix (consistently on all ranks).
// If Lfull != nullptr, every final row is also copied into it.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, double* Lfull) {
    const size_t P = g_size, me = g_rank;
    const size_t nl = A.size() / (n ? n : 1);
    std::vector<double> bufStore(2 * (n + 1));
    double* buf[2] = {bufStore.data(), bufStore.data() + (n + 1)};
    std::vector<MPI_Request> reqs(n, MPI_REQUEST_NULL);

    // Compute the diagonal of row j (owner only) and fill the broadcast buffer
    auto finalizeRow = [&](size_t j) {
        double* row = &A[(j / P) * n];
        double* out = buf[j & 1];
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) sum += row[k] * row[k];
        const double val = row[j] - sum;
        const bool ok = !(val <= 0.0);
        if (ok) row[j] = sqrt(val);
        memcpy(out, row, (j + 1) * sizeof(double));
        out[j + 1] = ok ? 1.0 : 0.0;  // status flag travels with the row
    };

    if (n == 0) return true;
    if (me == 0) finalizeRow(0);
    MPI_Ibcast(buf[0], 2, MPI_DOUBLE, 0, MPI_COMM_WORLD, &reqs[0]);

    for (size_t j = 0; j < n; ++j) {
        MPI_Wait(&reqs[j], MPI_STATUS_IGNORE);
        const double* rj = buf[j & 1];
        if (rj[j + 1] == 0.0) {
            if (me == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        if (Lfull) memcpy(Lfull + j * n, rj, (j + 1) * sizeof(double));
        const double d = rj[j];

        size_t r0 = 0;  // first local row with global index > j + 1
        if (j + 1 < n) {
            const size_t nx = j + 1;
            const size_t root = nx % P;
            if (root == me) {
                double* row = &A[(nx / P) * n];
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) sum += row[k] * rj[k];
                row[j] = (row[j] - sum) / d;
                finalizeRow(nx);
            }
            r0 = nx >= me ? (nx - me) / P + 1 : 0;
            MPI_Ibcast(buf[nx & 1], (int)(nx + 2), MPI_DOUBLE, (int)root, MPI_COMM_WORLD, &reqs[nx]);
        } else {
            r0 = nl;
        }

        size_t r = r0;
        for (; r + 4 <= nl; r += 4) {
            double* a0 = &A[r * n];
            double* a1 = a0 + n;
            double* a2 = a1 + n;
            double* a3 = a2 + n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (size_t k = 0; k < j; ++k) {
                const double b = rj[k];
                s0 += a0[k] * b;
                s1 += a1[k] * b;
                s2 += a2[k] * b;
                s3 += a3[k] * b;
            }
            a0[j] = (a0[j] - s0) / d;
            a1[j] = (a1[j] - s1) / d;
            a2[j] = (a2[j] - s2) / d;
            a3[j] = (a3[j] - s3) / d;
        }
        for (; r < nl; ++r) {
            double* a0 = &A[r * n];
            double s0 = 0.0;
            for (size_t k = 0; k < j; ++k) s0 += a0[k] * rj[k];
            a0[j] = (a0[j] - s0) / d;
        }
    }
    return true;
}

// Generate the local rows of a symmetric positive definite matrix A = B*B^T + n*I
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    const size_t P = g_size, me = g_rank;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    const size_t nl = A.size() / (n ? n : 1);
    for (size_t r = 0; r < nl; ++r) {
        const size_t i = r * P + me;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[r * n + j] = sum;
        }
        A[r * n + i] += n;
    }
}

// Validate: every rank checks its own rows of L*L^T against A_orig; Lfull is on all ranks
bool validateCholesky(const double* L, const std::vector<double>& A_orig, const size_t n) {
    const size_t P = g_size, me = g_rank;
    const size_t nl = A_orig.size() / (n ? n : 1);
    double err[2] = {0.0, 0.0};
    for (size_t r = 0; r < nl; ++r) {
        const size_t i = r * P + me;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t m = std::min(i, j);
            for (size_t k = 0; k <= m; ++k) sum += L[i * n + k] * L[j * n + k];
            const double a = A_orig[r * n + j];
            const double error = fabs(sum - a);
            err[0] = std::max(err[0], error);
            err[1] = std::max(err[1], error / (fabs(a) + 1e-10));
        }
    }
    double gerr[2];
    MPI_Allreduce(err, gerr, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", gerr[0]);
        printf("Max relative error: %.10e\n", gerr[1]);
    }
    if (gerr[1] > 1e-6) {
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
    const bool root = (g_rank == 0);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", g_size);
        printf("Generating positive definite matrix...\n");
    }

    const size_t nl = (n > (size_t)g_rank) ? (n - g_rank + g_size - 1) / g_size : 0;
    std::vector<double> A(nl * n);
    std::vector<double> A_orig;
    generatePositiveDefiniteMatrix(A, n);
    if (validate) A_orig = A;

    // Full L is needed for validation (all ranks) and for printing (rank 0)
    std::vector<double> Lfull;
    if (validate || (printResults && root)) Lfull.assign(n * n, 0.0);

    if (root) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, Lfull.empty() ? nullptr : Lfull.data());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults && root) {
        print_results(Lfull, "CholeskyL");
    }

    int rc = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateCholesky(Lfull.data(), A_orig, n);
        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        rc = valid ? 0 : 1;
    }

    MPI_Finalize();
    return rc;
}
