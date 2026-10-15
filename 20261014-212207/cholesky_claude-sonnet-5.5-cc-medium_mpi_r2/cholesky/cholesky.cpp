#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (Cholesky-Crout, row-oriented).
// Rows are distributed cyclically over the MPI ranks. The lower triangle is
// stored packed (row i starts at i*(i+1)/2) and every row is broadcast with a
// non-blocking collective as soon as it is final. Element (i,j) only needs rows
// <= j, so rows are computed as a pipelined wavefront across ranks. Dot
// products keep the sequential k order of the original code (bitwise equal).

static int g_rank = 0, g_size = 1;

static inline size_t rowOff(size_t i) { return i * (i + 1) / 2; }

// Compute C off-diagonal elements j0..j0+C-1 of row i (all < i) with C
// independent accumulation chains.
template <int C>
static inline void offDiagBlock(double* L, const size_t i, const size_t j0) {
    double* Li = L + rowOff(i);
    const double* Lj[C];
    double s[C];
    for (int b = 0; b < C; ++b) {
        Lj[b] = L + rowOff(j0 + b);
        s[b] = 0.0;
    }
    for (size_t k = 0; k < j0; ++k) {
        const double x = Li[k];
        for (int b = 0; b < C; ++b) s[b] += x * Lj[b][k];
    }
    for (int b = 0; b < C; ++b) {
        double sb = s[b];
        for (size_t k = j0; k < j0 + b; ++k) sb += Li[k] * Lj[b][k];
        Li[j0 + b] = (Li[j0 + b] - sb) / Lj[b][j0 + b];
    }
}

static inline void offDiag(double* L, size_t i, size_t j0, int cnt) {
    switch (cnt) {
        case 8: offDiagBlock<8>(L, i, j0); break;
        case 7: offDiagBlock<7>(L, i, j0); break;
        case 6: offDiagBlock<6>(L, i, j0); break;
        case 5: offDiagBlock<5>(L, i, j0); break;
        case 4: offDiagBlock<4>(L, i, j0); break;
        case 3: offDiagBlock<3>(L, i, j0); break;
        case 2: offDiagBlock<2>(L, i, j0); break;
        default: offDiagBlock<1>(L, i, j0); break;
    }
}

// L holds (packed) the lower triangle of A on the rows owned by this rank.
// On return every rank holds the complete packed L.
bool choleskyDecomposition(std::vector<double>& L, const size_t n) {
    std::vector<MPI_Request> reqs(n, MPI_REQUEST_NULL);
    size_t posted = 0;  // rows whose Ibcast has been posted
    size_t waited = 0;  // rows known to be complete locally
    double* base = L.data();

    auto postUpTo = [&](size_t end) {  // post rows [posted, end)
        for (; posted < end; ++posted)
            MPI_Ibcast(base + rowOff(posted), (int)(posted + 1), MPI_DOUBLE,
                       (int)(posted % g_size), MPI_COMM_WORLD, &reqs[posted]);
    };
    auto ensure = [&](size_t m) {  // make rows <= m available
        while (waited <= m) {
            if ((int)(waited % g_size) != g_rank) MPI_Wait(&reqs[waited], MPI_STATUS_IGNORE);
            ++waited;
        }
    };

    int ok = 1;
    for (size_t i = g_rank; i < n && ok; i += g_size) {
        postUpTo(i);  // receives for all lower rows
        double* Li = base + rowOff(i);
        size_t j0 = 0;
        while (j0 < i) {
            size_t cnt = std::min<size_t>(8, i - j0);
            ensure(j0 + cnt - 1);
            offDiag(base, i, j0, (int)cnt);
            j0 += cnt;
            if (waited < i && (int)(waited % g_size) != g_rank) {
                int flag;
                MPI_Test(&reqs[waited], &flag, MPI_STATUS_IGNORE);  // drive progress
            }
        }
        double sum = 0.0;
        for (size_t k = 0; k < i; ++k) sum += Li[k] * Li[k];
        const double val = Li[i] - sum;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return false;
        }
        Li[i] = sqrt(val);
        postUpTo(i + 1);  // send own row
    }
    postUpTo(n);
    MPI_Waitall((int)n, reqs.data(), MPI_STATUSES_IGNORE);
    return true;
}

// Generate the lower triangle of a symmetric positive definite matrix
// A = B*B^T + n*I, only for the rows owned by this rank (packed storage).
void generatePositiveDefiniteMatrix(std::vector<double>& L, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = g_rank; i < n; i += g_size) {
        double* Ai = L.data() + rowOff(i);
        const double* Bi = B.data() + i * n;
        size_t j = 0;
        for (; j + 4 <= i + 1; j += 4) {
            const double* b0 = B.data() + j * n;
            const double* b1 = b0 + n;
            const double* b2 = b1 + n;
            const double* b3 = b2 + n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double x = Bi[k];
                s0 += x * b0[k];
                s1 += x * b1[k];
                s2 += x * b2[k];
                s3 += x * b3[k];
            }
            Ai[j] = s0; Ai[j + 1] = s1; Ai[j + 2] = s2; Ai[j + 3] = s3;
        }
        for (; j <= i; ++j) {
            const double* b0 = B.data() + j * n;
            double s0 = 0.0;
            for (size_t k = 0; k < n; ++k) s0 += Bi[k] * b0[k];
            Ai[j] = s0;
        }
        Ai[i] += n;
    }
}

// Distributed validation: each rank checks its own rows (lower triangle; the
// matrices are symmetric so this covers the full comparison).
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = g_rank; i < n; i += g_size) {
        const double* Li = L.data() + rowOff(i);
        const double* Ai = A_orig.data() + rowOff(i);
        for (size_t j = 0; j <= i; ++j) {
            const double* Lj = L.data() + rowOff(j);
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) sum += Li[k] * Lj[k];
            const double error = fabs(sum - Ai[j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(Ai[j]) + 1e-10));
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
        printf("Generating positive definite matrix...\n");
    }

    // Packed lower triangle; only owned rows are filled initially
    std::vector<double> A(n * (n + 1) / 2);
    std::vector<double> A_orig;

    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A;  // own rows are valid
    }

    if (root) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    long ms = (long)((t1 - t0) * 1000.0);
    MPI_Bcast(&ms, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults && root) {
        std::vector<double> Lfull(n * n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <= i; ++j) Lfull[i * n + j] = A[rowOff(i) + j];
        print_results(Lfull, "CholeskyL");
    }

    int rc = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        rc = valid ? 0 : 1;
    }

    MPI_Finalize();
    return rc;
}
