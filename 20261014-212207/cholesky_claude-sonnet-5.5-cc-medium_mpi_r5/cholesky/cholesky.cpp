#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, right-looking blocked algorithm).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is distributed by block rows in a 1D block-cyclic layout: block row I
// (nb rows) lives on rank I % P. Each rank stores its rows at full width (row-major).

namespace {

struct Dist {
    size_t n, nb, nblk;
    int rank, P;
    size_t bs(size_t I) const { return std::min(nb, n - I * nb); }
    int owner(size_t I) const { return (int)(I % P); }
    size_t localRow(size_t g) const { return ((g / nb) / P) * nb + g % nb; }
    size_t localBlocks() const {
        return nblk > (size_t)rank ? (nblk - rank + P - 1) / P : 0;
    }
};

constexpr size_t MR = 4, NR = 8;

// A[rows r0..r0+nr) -= P[rows] * PT, for columns [jbeg, jend) (jend <= n).
// P rows are the panel columns of A itself (A[i*n + k0 .. k0+b)).
// PT is b x ld, column index j - jbeg.
inline void updateRows(double* A, size_t n, size_t r0, size_t nr, size_t k0, size_t b,
                       const double* PT, size_t ld, size_t jbeg, size_t jend) {
    size_t j = jbeg;
    if (nr == MR) {
        double* a0 = A + (r0 + 0) * n;
        double* a1 = A + (r0 + 1) * n;
        double* a2 = A + (r0 + 2) * n;
        double* a3 = A + (r0 + 3) * n;
        for (; j + NR <= jend; j += NR) {
            double acc[MR][NR] = {};
            const double* pt = PT + (j - jbeg);
            for (size_t t = 0; t < b; ++t, pt += ld) {
                const double x0 = a0[k0 + t], x1 = a1[k0 + t], x2 = a2[k0 + t], x3 = a3[k0 + t];
#pragma GCC unroll 8
                for (size_t c = 0; c < NR; ++c) {
                    const double y = pt[c];
                    acc[0][c] += x0 * y;
                    acc[1][c] += x1 * y;
                    acc[2][c] += x2 * y;
                    acc[3][c] += x3 * y;
                }
            }
            for (size_t c = 0; c < NR; ++c) {
                a0[j + c] -= acc[0][c];
                a1[j + c] -= acc[1][c];
                a2[j + c] -= acc[2][c];
                a3[j + c] -= acc[3][c];
            }
        }
    }
    for (size_t r = 0; r < nr; ++r) {
        double* a = A + (r0 + r) * n;
        size_t jj = (nr == MR) ? j : jbeg;
        for (; jj < jend; ++jj) {
            double s = 0.0;
            for (size_t t = 0; t < b; ++t) s += a[k0 + t] * PT[t * ld + (jj - jbeg)];
            a[jj] -= s;
        }
    }
}

// Returns failing global diagonal index, or -1 on success. Leaves upper part of local rows zeroed.
long choleskyDecomposition(std::vector<double>& A, const Dist& d) {
    const size_t n = d.n, nb = d.nb;
    std::vector<double> diag(nb * nb + 1);
    std::vector<double> panel, recv, PT;
    std::vector<int> counts(d.P), displs(d.P);

    for (size_t K = 0; K < d.nblk; ++K) {
        const size_t k0 = K * nb, b = d.bs(K);
        const int own = d.owner(K);

        // 1. Factor the diagonal block on its owner
        if (d.rank == own) {
            double* base = &A[d.localRow(k0) * n];
            long fail = -1;
            for (size_t r = 0; r < b && fail < 0; ++r) {
                double* row = base + r * n + k0;
                for (size_t c = 0; c <= r; ++c) {
                    const double* rc = base + c * n + k0;
                    double sum = 0.0;
                    for (size_t t = 0; t < c; ++t) sum += row[t] * rc[t];
                    if (c == r) {
                        const double val = row[r] - sum;
                        if (val <= 0.0) { fail = (long)(k0 + r); break; }
                        row[r] = sqrt(val);
                    } else {
                        row[c] = (row[c] - sum) / rc[c];
                    }
                }
            }
            diag[0] = (double)fail;
            for (size_t r = 0; r < b; ++r)
                for (size_t c = 0; c < b; ++c)
                    diag[1 + r * b + c] = (c <= r) ? base[r * n + k0 + c] : 0.0;
        }
        MPI_Bcast(diag.data(), (int)(b * b + 1), MPI_DOUBLE, own, MPI_COMM_WORLD);
        if (diag[0] >= 0) return (long)diag[0];
        const double* Lkk = diag.data() + 1;

        if (K + 1 == d.nblk) break;

        // 2. Panel solve for local block rows below K; collect into contiguous panel
        size_t myBlocks = 0;
        for (size_t I = K + 1; I < d.nblk; ++I) if (d.owner(I) == d.rank) ++myBlocks;
        panel.assign(myBlocks * nb * b, 0.0);
        size_t pi = 0;
        for (size_t I = K + 1; I < d.nblk; ++I) {
            if (d.owner(I) != d.rank) continue;
            const size_t bi = d.bs(I);
            for (size_t r = 0; r < bi; ++r) {
                double* row = &A[(d.localRow(I * nb) + r) * n + k0];
                for (size_t c = 0; c < b; ++c) {
                    double sum = 0.0;
                    for (size_t t = 0; t < c; ++t) sum += row[t] * Lkk[c * b + t];
                    row[c] = (row[c] - sum) / Lkk[c * b + c];
                }
                std::memcpy(&panel[(pi * nb + r) * b], row, b * sizeof(double));
            }
            ++pi;
        }

        // 3. Allgather the panel
        for (int q = 0; q < d.P; ++q) {
            size_t cnt = 0;
            for (size_t I = K + 1 + ((q - (long)(K + 1)) % d.P + d.P) % d.P; I < d.nblk; I += d.P) ++cnt;
            counts[q] = (int)(cnt * nb * b);
            displs[q] = q ? displs[q - 1] + counts[q - 1] : 0;
        }
        // Last block row may be short; its padding rows are zeros and ignored below.
        recv.resize((size_t)displs[d.P - 1] + counts[d.P - 1]);
        MPI_Allgatherv(panel.data(), counts[d.rank], MPI_DOUBLE, recv.data(), counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // 4. Build transposed panel PT (b x m), column = global row - jbeg
        const size_t jbeg = k0 + b, m = n - jbeg;
        PT.resize(b * m);
        for (size_t I = K + 1; I < d.nblk; ++I) {
            const int q = d.owner(I);
            const size_t idx = (I - (K + 1) - ((q - (long)(K + 1)) % d.P + d.P) % d.P) / d.P;
            const double* src = &recv[(size_t)displs[q] + idx * nb * b];
            const size_t bi = d.bs(I), col0 = I * nb - jbeg;
            for (size_t r = 0; r < bi; ++r)
                for (size_t t = 0; t < b; ++t) PT[t * m + col0 + r] = src[r * b + t];
        }

        // 5. Trailing update of local rows
        for (size_t I = K + 1; I < d.nblk; ++I) {
            if (d.owner(I) != d.rank) continue;
            const size_t bi = d.bs(I), lr = d.localRow(I * nb);
            for (size_t r = 0; r < bi; r += MR) {
                const size_t nr = std::min(MR, bi - r);
                const size_t jend = std::min(n, I * nb + r + nr);  // up to diagonal of last row
                // process tile-by-tile so the PT tile stays in L1 across groups: groups are
                // handled one at a time here, the PT slice (b x NR) is small and cached.
                updateRows(A.data(), n, lr + r, nr, k0, b, PT.data(), m, jbeg, jend);
            }
        }
    }

    // Zero out upper triangular part of local rows
    for (size_t I = (size_t)d.rank; I < d.nblk; I += d.P) {
        for (size_t r = 0; r < d.bs(I); ++r) {
            const size_t g = I * nb + r;
            double* row = &A[(d.localRow(I * nb) + r) * n];
            for (size_t j = g + 1; j < n; ++j) row[j] = 0.0;
        }
    }
    return -1;
}

// Generate local rows of a symmetric positive definite matrix A = B*B^T + n*I
void generatePositiveDefiniteMatrix(std::vector<double>& A, const Dist& d) {
    const size_t n = d.n;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    for (size_t I = (size_t)d.rank; I < d.nblk; I += d.P) {
        for (size_t r = 0; r < d.bs(I); ++r) {
            const size_t i = I * d.nb + r;
            const double* bi = &B[i * n];
            double* out = &A[(d.localRow(I * d.nb) + r) * n];
            size_t j = 0;
            for (; j + 4 <= n; j += 4) {
                const double *b0 = &B[j * n], *b1 = b0 + n, *b2 = b1 + n, *b3 = b2 + n;
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (size_t k = 0; k < n; ++k) {
                    const double x = bi[k];
                    s0 += x * b0[k]; s1 += x * b1[k]; s2 += x * b2[k]; s3 += x * b3[k];
                }
                out[j] = s0; out[j + 1] = s1; out[j + 2] = s2; out[j + 3] = s3;
            }
            for (; j < n; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < n; ++k) s += bi[k] * B[j * n + k];
                out[j] = s;
            }
            out[i] += n;
        }
    }
}

// Distributed validation: each rank checks its rows of L*L^T against A_orig.
// Lfull is the complete L (row-major) on every rank.
bool validateCholesky(const std::vector<double>& Lfull, const std::vector<double>& A_orig,
                      const Dist& d) {
    const size_t n = d.n;
    double maxError = 0.0, relError = 0.0;
    for (size_t I = (size_t)d.rank; I < d.nblk; I += d.P) {
        for (size_t r = 0; r < d.bs(I); ++r) {
            const size_t i = I * d.nb + r;
            const double* li = &Lfull[i * n];
            const double* ao = &A_orig[(d.localRow(I * d.nb) + r) * n];
            for (size_t j = 0; j < n; ++j) {
                const double* lj = &Lfull[j * n];
                double sum = 0.0;
                const size_t kmax = std::min(i, j);
                for (size_t k = 0; k <= kmax; ++k) sum += li[k] * lj[k];
                const double error = fabs(sum - ao[j]);
                maxError = std::max(maxError, error);
                relError = std::max(relError, error / (fabs(ao[j]) + 1e-10));
            }
        }
    }
    double loc[2] = {maxError, relError}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", glob[0]);
        printf("Max relative error: %.10e\n", glob[1]);
        if (glob[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return glob[1] <= 1e-6;
}

}  // namespace

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
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Dist d;
    d.n = n;
    d.rank = rank;
    d.P = P;
    d.nb = std::max<size_t>(4, std::min<size_t>(64, n / (2 * (size_t)P)));
    d.nblk = (n + d.nb - 1) / d.nb;

    // Allocate local rows
    std::vector<double> A(d.localBlocks() * d.nb * n);
    std::vector<double> A_orig;

    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, d);

    if (validate) {
        A_orig = A;  // Save original (local rows) for validation
    }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    long fail = choleskyDecomposition(A, d);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (fail >= 0) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %ld\n", fail);
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int rc = 0;
    if (printResults || validate) {
        // Assemble the full L on every rank
        std::vector<double> L(n * n);
        for (size_t I = 0; I < d.nblk; ++I) {
            const int o = d.owner(I);
            double* dst = &L[I * d.nb * n];
            const size_t cnt = d.bs(I) * n;
            if (rank == o) std::memcpy(dst, &A[d.localRow(I * d.nb) * n], cnt * sizeof(double));
            MPI_Bcast(dst, (int)cnt, MPI_DOUBLE, o, MPI_COMM_WORLD);
        }

        if (printResults && rank == 0) {
            print_results(L, "CholeskyL");
        }

        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            bool valid = validateCholesky(L, A_orig, d);
            if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            rc = valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return rc;
}
