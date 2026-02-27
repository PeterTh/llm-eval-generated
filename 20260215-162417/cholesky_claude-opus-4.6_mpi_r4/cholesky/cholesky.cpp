#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallelized blocked Cholesky decomposition using 1D block-cyclic
// column distribution for distributed memory cluster parallelism.

static const int NB = 64;

static inline int blk_width(int blk, size_t n) {
    return (int)std::min((size_t)NB, n - (size_t)blk * NB);
}

// Cholesky factorization of a small kb×kb block stored with leading dimension lda
static bool small_potrf(double* A, int kb, int lda) {
    for (int j = 0; j < kb; j++) {
        double s = A[j * lda + j];
        for (int k = 0; k < j; k++)
            s -= A[j * lda + k] * A[j * lda + k];
        if (s <= 0.0) return false;
        A[j * lda + j] = sqrt(s);
        double inv = 1.0 / A[j * lda + j];
        for (int i = j + 1; i < kb; i++) {
            double t = A[i * lda + j];
            for (int k = 0; k < j; k++)
                t -= A[i * lda + k] * A[j * lda + k];
            A[i * lda + j] = t * inv;
        }
    }
    return true;
}

// Panel solve: solve X * L^T = B for X, overwriting B in place
// X/B is m rows × kb cols, L is kb×kb lower triangular
static void panel_trsm(double* X, int m, const double* L, int kb, int ldx, int ldl) {
    for (int j = 0; j < kb; j++) {
        for (int k = 0; k < j; k++) {
            double Ljk = L[j * ldl + k];
            for (int i = 0; i < m; i++)
                X[i * ldx + j] -= X[i * ldx + k] * Ljk;
        }
        double inv = 1.0 / L[j * ldl + j];
        for (int i = 0; i < m; i++)
            X[i * ldx + j] *= inv;
    }
}

// GEMM: C -= A * B^T where C is m×nn, A is m×kk, B is nn×kk
static void gemm_nt_sub(double* __restrict__ C, const double* __restrict__ A,
                        const double* __restrict__ B,
                        int m, int nn, int kk, int ldc, int lda, int ldb) {
    // Process 4 rows at a time for register reuse
    int i = 0;
    for (; i + 3 < m; i += 4) {
        for (int j = 0; j < nn; j++) {
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (int l = 0; l < kk; l++) {
                double bv = B[j * ldb + l];
                s0 += A[(i    ) * lda + l] * bv;
                s1 += A[(i + 1) * lda + l] * bv;
                s2 += A[(i + 2) * lda + l] * bv;
                s3 += A[(i + 3) * lda + l] * bv;
            }
            C[(i    ) * ldc + j] -= s0;
            C[(i + 1) * ldc + j] -= s1;
            C[(i + 2) * ldc + j] -= s2;
            C[(i + 3) * ldc + j] -= s3;
        }
    }
    for (; i < m; i++) {
        for (int j = 0; j < nn; j++) {
            double s = 0.0;
            for (int l = 0; l < kk; l++)
                s += A[i * lda + l] * B[j * ldb + l];
            C[i * ldc + j] -= s;
        }
    }
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    }
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = sum;
        }
    }
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("Generating positive definite matrix...\n");
    }

    // Block-cyclic column distribution
    int nblks = (int)((n + NB - 1) / NB);
    std::vector<int> blk_lc(nblks, -1);
    int local_ncols = 0;
    for (int b = rank; b < nblks; b += nprocs) {
        blk_lc[b] = local_ncols;
        local_ncols += blk_width(b, n);
    }

    // Generate full matrix (deterministic, same on all ranks) and extract local columns
    std::vector<double> local_A((size_t)n * local_ncols);
    {
        std::vector<double> A(n * n);
        generatePositiveDefiniteMatrix(A, n);
        for (int b = rank; b < nblks; b += nprocs) {
            int gc = b * NB, bw = blk_width(b, n), lc = blk_lc[b];
            for (size_t i = 0; i < n; i++)
                for (int j = 0; j < bw; j++)
                    local_A[i * local_ncols + lc + j] = A[i * n + gc + j];
        }
    }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    // Right-looking blocked Cholesky with 1D block-cyclic column distribution
    bool success = true;
    std::vector<double> panel((size_t)n * NB);

    for (int k = 0; k < nblks; k++) {
        int kb = blk_width(k, n);
        int owner = k % nprocs;
        int prows = (int)(n - (size_t)k * NB);

        if (rank == owner) {
            int lk = blk_lc[k];
            // POTRF: factor diagonal block
            if (!small_potrf(&local_A[(size_t)k * NB * local_ncols + lk], kb, local_ncols))
                success = false;
            // TRSM: solve panel below diagonal
            if (success && (size_t)k * NB + kb < n)
                panel_trsm(&local_A[((size_t)k * NB + kb) * local_ncols + lk],
                           prows - kb,
                           &local_A[(size_t)k * NB * local_ncols + lk],
                           kb, local_ncols, local_ncols);
            // Pack panel into contiguous buffer for broadcast
            for (int i = 0; i < prows; i++)
                for (int j = 0; j < kb; j++)
                    panel[i * kb + j] = local_A[((size_t)k * NB + i) * local_ncols + lk + j];
        }

        MPI_Bcast(panel.data(), prows * kb, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // GEMM: update all locally-owned trailing column blocks
        for (int b = rank; b < nblks; b += nprocs) {
            if (b <= k) continue;
            int jb = blk_width(b, n), lj = blk_lc[b];
            int urows = (int)(n - (size_t)b * NB);
            int poff = b * NB - k * NB;
            gemm_nt_sub(&local_A[(size_t)b * NB * local_ncols + lj],
                        &panel[(size_t)poff * kb],
                        &panel[(size_t)poff * kb],
                        urows, jb, kb, local_ncols, kb, kb);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Check success across all ranks
    int lsuc = success ? 1 : 0, gsuc;
    MPI_Allreduce(&lsuc, &gsuc, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!gsuc) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Zero upper triangle in local data
    for (int b = rank; b < nblks; b += nprocs) {
        int gc = b * NB, bw = blk_width(b, n), lc = blk_lc[b];
        for (int j = 0; j < bw; j++)
            for (int i = 0; i < gc + j; i++)
                local_A[(size_t)i * local_ncols + lc + j] = 0.0;
    }

    // Gather full result matrix on rank 0 for validation/output
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(n * n, 0.0);
        // Copy own local columns
        for (int b = 0; b < nblks; b += nprocs) {
            int gc = b * NB, bw = blk_width(b, n), lc = blk_lc[b];
            for (size_t i = 0; i < n; i++)
                for (int j = 0; j < bw; j++)
                    A_full[i * n + gc + j] = local_A[i * local_ncols + lc + j];
        }
        // Receive from other ranks
        for (int src = 1; src < nprocs; src++) {
            int src_nc = 0;
            for (int b = src; b < nblks; b += nprocs)
                src_nc += blk_width(b, n);
            if (src_nc == 0) continue;
            std::vector<double> rbuf((size_t)n * src_nc);
            MPI_Recv(rbuf.data(), (int)((size_t)n * src_nc), MPI_DOUBLE,
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int slc = 0;
            for (int b = src; b < nblks; b += nprocs) {
                int gc = b * NB, bw = blk_width(b, n);
                for (size_t i = 0; i < n; i++)
                    for (int j = 0; j < bw; j++)
                        A_full[i * n + gc + j] = rbuf[i * src_nc + slc + j];
                slc += bw;
            }
        }
    } else if (local_ncols > 0) {
        MPI_Send(local_A.data(), (int)((size_t)n * local_ncols), MPI_DOUBLE,
                 0, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(A_full, "CholeskyL");
    }

    // Validation
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        std::vector<double> A_orig(n * n);
        generatePositiveDefiniteMatrix(A_orig, n);
        bool valid = validateCholesky(A_full, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
