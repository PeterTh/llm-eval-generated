#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Block size for blocked right-looking Cholesky
static constexpr size_t BS = 64;

static size_t num_blocks(size_t n) { return (n + BS - 1) / BS; }
static size_t block_cols(size_t k, size_t n) { return std::min(BS, n - k * BS); }

// Distribution of block columns across MPI ranks (blocked)
struct Dist {
    size_t start_blk, num_blks, col_start, col_end, local_cols;
    Dist() : start_blk(0), num_blks(0), col_start(0), col_end(0), local_cols(0) {}
    Dist(int rank, int nprocs, size_t n) {
        size_t nb = num_blocks(n);
        size_t base = nb / nprocs, extra = nb % nprocs;
        start_blk = (size_t)rank * base + std::min((size_t)rank, extra);
        num_blks = base + ((size_t)rank < extra ? 1 : 0);
        col_start = start_blk * BS;
        col_end = std::min((start_blk + num_blks) * BS, n);
        local_cols = col_end - col_start;
    }
    static int owner_of(int k, int nprocs, size_t n) {
        size_t nb = num_blocks(n);
        size_t base = nb / nprocs, extra = nb % nprocs;
        if (base == 0) return k;
        size_t threshold = (base + 1) * extra;
        if ((size_t)k < threshold) return (int)((size_t)k / (base + 1));
        return (int)(extra + ((size_t)k - threshold) / base);
    }
};

// Cholesky of m×m submatrix with row-major stride lda (in-place, lower triangular)
static bool cholesky_blk(double* __restrict__ A, size_t m, size_t lda) {
    for (size_t j = 0; j < m; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k)
            sum += A[j * lda + k] * A[j * lda + k];
        double val = A[j * lda + j] - sum;
        if (val <= 0.0) return false;
        A[j * lda + j] = sqrt(val);
        double inv_diag = 1.0 / A[j * lda + j];
        for (size_t i = j + 1; i < m; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < j; ++k)
                s += A[i * lda + k] * A[j * lda + k];
            A[i * lda + j] = (A[i * lda + j] - s) * inv_diag;
        }
    }
    return true;
}

// Right triangular solve: B := B * L^{-T}
// B is m×k with stride ldb, L is k×k lower triangular with stride ldl
static void trsm_bnrt(double* __restrict__ B, size_t ldb, size_t m, size_t k,
                       const double* __restrict__ L, size_t ldl) {
    for (size_t c = 0; c < k; ++c) {
        for (size_t j = 0; j < c; ++j) {
            double Lcj = L[c * ldl + j];
            for (size_t r = 0; r < m; ++r)
                B[r * ldb + c] -= B[r * ldb + j] * Lcj;
        }
        double inv = 1.0 / L[c * ldl + c];
        for (size_t r = 0; r < m; ++r)
            B[r * ldb + c] *= inv;
    }
}

// C -= A * B^T  (C: m×n stride ldc, A: m×k stride lda, B: n×k stride ldb)
static void gemm_sub_abt(double* __restrict__ C, size_t ldc,
                          const double* __restrict__ A, size_t lda,
                          const double* __restrict__ B, size_t ldb,
                          size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p)
                sum += A[i * lda + p] * B[j * ldb + p];
            C[i * ldc + j] -= sum;
        }
    }
}

// Generate a symmetric positive definite matrix (identical to original)
static void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += n;
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = sum;
        }
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
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
    int validate_flag = 0, print_flag = 0, help_flag = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                print_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                help_flag = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                help_flag = 1;
            }
        }
        if (help_flag) printUsage(argv[0]);
    }

    MPI_Bcast(&n, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&help_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (help_flag) { MPI_Finalize(); return 0; }
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool do_validate = validate_flag != 0;
    bool do_print = print_flag != 0;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", do_validate ? "enabled" : "disabled");
    }

    Dist dist(rank, nprocs, n);
    size_t nb = num_blocks(n);
    size_t lda = dist.local_cols;

    // Allocate local storage: n rows × local_cols columns (row-major)
    std::vector<double> Aloc(n * lda, 0.0);

    // Generate matrix on rank 0 and scatter by column blocks
    {
        std::vector<int> scounts(nprocs), sdispls(nprocs);
        std::vector<double> sbuf;
        if (rank == 0) {
            sbuf.resize(n * n);
            printf("Generating positive definite matrix...\n");
            std::vector<double> Afull(n * n);
            generatePositiveDefiniteMatrix(Afull, n);
            int off = 0;
            for (int p = 0; p < nprocs; ++p) {
                Dist dp(p, nprocs, n);
                sdispls[p] = off;
                scounts[p] = (int)(n * dp.local_cols);
                for (size_t i = 0; i < n; ++i)
                    for (size_t j = 0; j < dp.local_cols; ++j)
                        sbuf[off + i * dp.local_cols + j] = Afull[i * n + dp.col_start + j];
                off += (int)(n * dp.local_cols);
            }
        }
        MPI_Scatterv(sbuf.data(), scounts.data(), sdispls.data(), MPI_DOUBLE,
                     Aloc.data(), (int)(n * lda), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Blocked right-looking Cholesky decomposition with MPI
    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    // Allocate panel buffer (max panel size across all block columns)
    size_t max_panel = 0;
    for (size_t k = 0; k < nb; ++k) {
        size_t ps = (n - k * BS) * block_cols(k, n);
        max_panel = std::max(max_panel, ps);
    }
    std::vector<double> panel(max_panel > 0 ? max_panel : 1);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    bool success = true;

    for (size_t k = 0; k < nb; ++k) {
        size_t bsk = block_cols(k, n);
        int own_k = Dist::owner_of((int)k, nprocs, n);
        size_t panel_rows = n - k * BS;

        if (rank == own_k) {
            size_t lk = k * BS - dist.col_start;

            if (success) {
                // Panel factorization: Cholesky of diagonal block
                double* Akk = &Aloc[k * BS * lda + lk];
                if (!cholesky_blk(Akk, bsk, lda)) {
                    printf("Error: Matrix is not positive definite at block %zu\n", k);
                    success = false;
                }
                // Zero upper triangle of diagonal block
                for (size_t i = 0; i < bsk; ++i)
                    for (size_t j = i + 1; j < bsk; ++j)
                        Akk[i * lda + j] = 0.0;

                // Triangular solve for off-diagonal blocks in this panel
                if (success) {
                    for (size_t i = k + 1; i < nb; ++i) {
                        size_t bsi = block_cols(i, n);
                        double* Aik = &Aloc[i * BS * lda + lk];
                        trsm_bnrt(Aik, lda, bsi, bsk, Akk, lda);
                    }
                }
            }

            // Pack panel contiguously: rows [k*BS, n), columns of block k
            for (size_t r = 0; r < panel_rows; ++r)
                std::memcpy(&panel[r * bsk], &Aloc[(k * BS + r) * lda + lk],
                            bsk * sizeof(double));
        }

        // Broadcast factored panel to all processes
        int psz = (int)(panel_rows * bsk);
        MPI_Bcast(&psz, 1, MPI_INT, own_k, MPI_COMM_WORLD);
        MPI_Bcast(panel.data(), psz, MPI_DOUBLE, own_k, MPI_COMM_WORLD);

        // Check for failure across all processes
        int lok = success ? 1 : 0, gok;
        MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!gok) { success = false; break; }

        // Update trailing submatrix using local block columns
        if (lda > 0) {
            for (size_t jj = 0; jj < dist.num_blks; ++jj) {
                size_t j = dist.start_blk + jj;
                if (j <= k) continue;

                size_t bsj = block_cols(j, n);
                size_t lj = j * BS - dist.col_start;
                const double* Ljk = &panel[(j * BS - k * BS) * bsk];

                // Update diagonal block A[j,j] -= L[j,k] * L[j,k]^T
                double* Ajj = &Aloc[j * BS * lda + lj];
                gemm_sub_abt(Ajj, lda, Ljk, bsk, Ljk, bsk, bsj, bsj, bsk);

                // Update off-diagonal blocks A[i,j] -= L[i,k] * L[j,k]^T
                for (size_t i = j + 1; i < nb; ++i) {
                    size_t bsi = block_cols(i, n);
                    const double* Lik = &panel[(i * BS - k * BS) * bsk];
                    double* Aij = &Aloc[i * BS * lda + lj];
                    gemm_sub_abt(Aij, lda, Lik, bsk, Ljk, bsk, bsi, bsj, bsk);
                }
            }
        }
    }

    // Zero out upper triangular part of local matrix
    for (size_t jloc = 0; jloc < lda; ++jloc) {
        size_t jglob = dist.col_start + jloc;
        for (size_t i = 0; i < jglob; ++i)
            Aloc[i * lda + jloc] = 0.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Gather results on rank 0
    std::vector<double> Afull;
    if (rank == 0 && (do_print || do_validate))
        Afull.resize(n * n);

    {
        std::vector<int> rcounts(nprocs), rdispls(nprocs);
        std::vector<double> rbuf;
        int off = 0;
        for (int p = 0; p < nprocs; ++p) {
            Dist dp(p, nprocs, n);
            rcounts[p] = (int)(n * dp.local_cols);
            if (rank == 0) { rdispls[p] = off; rbuf.resize(n * n); }
            off += (int)(n * dp.local_cols);
        }

        MPI_Gatherv(Aloc.data(), (int)(n * lda), MPI_DOUBLE,
                    rank == 0 ? rbuf.data() : nullptr,
                    rcounts.data(), rdispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Unpack on rank 0
        if (rank == 0 && (do_print || do_validate)) {
            off = 0;
            for (int p = 0; p < nprocs; ++p) {
                Dist dp(p, nprocs, n);
                for (size_t i = 0; i < n; ++i)
                    for (size_t j = 0; j < dp.local_cols; ++j)
                        Afull[i * n + dp.col_start + j] = rbuf[off + i * dp.local_cols + j];
                off += (int)(n * dp.local_cols);
            }
        }
    }

    if (rank == 0) {
        if (!success) {
            printf("Cholesky decomposition failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Computation time: %lld ms\n", global_duration_ms);

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (do_print) {
            print_results(Afull, "CholeskyL");
        }

        if (do_validate) {
            std::vector<double> A_orig(n * n);
            generatePositiveDefiniteMatrix(A_orig, n);
            printf("Validating result...\n");
            bool valid = validateCholesky(Afull, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
