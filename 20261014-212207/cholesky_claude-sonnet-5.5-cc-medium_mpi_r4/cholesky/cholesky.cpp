#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI). Rows are distributed cyclically:
// row i lives on rank i % P, stored locally as row i / P. For every column j the
// owner of row j broadcasts the finished row; every rank then updates column j of
// its own rows. The next row is produced and broadcast first (look-ahead) so the
// broadcast overlaps with the bulk of the column update. Each element is
// accumulated in the same order as in the sequential algorithm.

static int g_rank = 0, g_size = 1;

static inline size_t localRows(size_t n, int r) {
    return (n > (size_t)r) ? (n - r + g_size - 1) / g_size : 0;
}

// Dot product of rows x and y over [0, len), sequential order
static inline double dotSeq(const double* x, const double* y, size_t len) {
    double sum = 0.0;
    for (size_t k = 0; k < len; ++k) sum += x[k] * y[k];
    return sum;
}

// Returns false (on all ranks) if matrix is not positive definite
bool choleskyDecomposition(std::vector<double>& Al, const size_t n) {
    const size_t P = g_size, R = g_rank;
    std::vector<double> bufs[2] = {std::vector<double>(n + 2), std::vector<double>(n + 2)};
    MPI_Request req = MPI_REQUEST_NULL;

    // Computes diagonal of global row j (owned locally) and fills buf (row + status)
    auto finishRow = [&](size_t j, double* buf) {
        double* row = &Al[(j / P) * n];
        const double val = row[j] - dotSeq(row, row, j);
        if (val <= 0.0) {
            buf[j + 1] = 1.0;
        } else {
            row[j] = sqrt(val);
            buf[j + 1] = 0.0;
        }
        memcpy(buf, row, (j + 1) * sizeof(double));
    };

    // Row 0
    if (R == 0) finishRow(0, bufs[0].data());
    MPI_Bcast(bufs[0].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (size_t j = 0; j < n; ++j) {
        double* cur = bufs[j & 1].data();
        if (cur[j + 1] != 0.0) {
            if (R == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        size_t first = j + 1;  // first row to update in the bulk loop
        if (j + 1 < n) {
            double* nxt = bufs[(j + 1) & 1].data();
            const size_t o = (j + 1) % P;
            if (R == o) {
                double* row = &Al[((j + 1) / P) * n];
                row[j] = (row[j] - dotSeq(row, cur, j)) / cur[j];
                finishRow(j + 1, nxt);
            }
            MPI_Ibcast(nxt, (int)(j + 3), MPI_DOUBLE, (int)o, MPI_COMM_WORLD, &req);
            first = j + 2;
        }
        // Local rows i = R + l*P with i >= first
        size_t l = (first > R) ? (first - R + P - 1) / P : 0;
        const size_t nl = localRows(n, (int)R);
        const double cjj = cur[j];
        for (; l + 4 <= nl; l += 4) {
            double* r0 = &Al[l * n];
            double* r1 = r0 + n;
            double* r2 = r1 + n;
            double* r3 = r2 + n;
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t k = 0; k < j; ++k) {
                const double c = cur[k];
                s0 += r0[k] * c;
                s1 += r1[k] * c;
                s2 += r2[k] * c;
                s3 += r3[k] * c;
            }
            r0[j] = (r0[j] - s0) / cjj;
            r1[j] = (r1[j] - s1) / cjj;
            r2[j] = (r2[j] - s2) / cjj;
            r3[j] = (r3[j] - s3) / cjj;
        }
        for (; l < nl; ++l) {
            double* r = &Al[l * n];
            r[j] = (r[j] - dotSeq(r, cur, j)) / cjj;
        }
        if (j + 1 < n) MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    // Zero out upper triangular part
    for (size_t l = 0; l < localRows(n, (int)R); ++l) {
        const size_t i = R + l * P;
        for (size_t j = i + 1; j < n; ++j) Al[l * n + j] = 0.0;
    }
    return true;
}

// Generate local rows of a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& Al, const size_t n) {
    // A = B * B^T + n*I, B random (replicated on every rank, same sequence)
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const size_t nl = localRows(n, g_rank);
    for (size_t l = 0; l < nl; ++l) {
        const size_t i = g_rank + l * g_size;
        const double* bi = &B[i * n];
        double* a = &Al[l * n];
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            const double* b0 = &B[j * n];
            const double* b1 = b0 + n;
            const double* b2 = b1 + n;
            const double* b3 = b2 + n;
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t k = 0; k < n; ++k) {
                const double x = bi[k];
                s0 += x * b0[k];
                s1 += x * b1[k];
                s2 += x * b2[k];
                s3 += x * b3[k];
            }
            a[j] = s0; a[j + 1] = s1; a[j + 2] = s2; a[j + 3] = s3;
        }
        for (; j < n; ++j) a[j] = dotSeq(bi, &B[j * n], n);
        a[i] += n;
    }
}

// Gather distributed rows into the full matrix on every rank (or only root if all==false)
static void gatherFull(const std::vector<double>& Al, std::vector<double>& full, const size_t n, bool all) {
    const size_t P = g_size;
    const size_t maxl = (n + P - 1) / P;
    std::vector<double> send(maxl * n, 0.0);
    memcpy(send.data(), Al.data(), Al.size() * sizeof(double));
    std::vector<double> recv;
    if (all || g_rank == 0) recv.resize(maxl * n * P);
    const int cnt = (int)(maxl * n);
    if (all) MPI_Allgather(send.data(), cnt, MPI_DOUBLE, recv.data(), cnt, MPI_DOUBLE, MPI_COMM_WORLD);
    else MPI_Gather(send.data(), cnt, MPI_DOUBLE, recv.data(), cnt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (all || g_rank == 0) {
        full.assign(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const size_t r = i % P, l = i / P;
            memcpy(&full[i * n], &recv[(r * maxl + l) * n], n * sizeof(double));
        }
    }
}

bool validateCholesky(const std::vector<double>& Lfull, const std::vector<double>& Al_orig, const size_t n) {
    // Each rank checks its own rows of L * L^T against the original
    double maxError = 0.0, relError = 0.0;
    const size_t nl = localRows(n, g_rank);
    std::vector<double> rec(n);
    for (size_t l = 0; l < nl; ++l) {
        const size_t i = g_rank + l * g_size;
        for (size_t j = 0; j < n; ++j) rec[j] = dotSeq(&Lfull[i * n], &Lfull[j * n], n);
        for (size_t j = 0; j < n; ++j) {
            const double o = Al_orig[l * n + j];
            const double error = fabs(rec[j] - o);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(o) + 1e-10));
        }
    }
    double loc[2] = {maxError, relError}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = glob[0];
    relError = glob[1];

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }
    if (relError > 1e-6) {
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
    int rc = 0;

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

    const bool root = g_rank == 0;
    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> A(localRows(n, g_rank) * n);
    std::vector<double> A_orig;
    generatePositiveDefiniteMatrix(A, n);
    if (validate) A_orig = A;

    if (root) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", (long)duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> full;
        gatherFull(A, full, n, false);
        if (root) print_results(full, "CholeskyL");
    }

    if (validate) {
        if (root) printf("Validating result...\n");
        std::vector<double> full;
        gatherFull(A, full, n, true);
        bool valid = validateCholesky(full, A_orig, n);
        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        rc = valid ? 0 : 1;
    }

    MPI_Finalize();
    return rc;
}
