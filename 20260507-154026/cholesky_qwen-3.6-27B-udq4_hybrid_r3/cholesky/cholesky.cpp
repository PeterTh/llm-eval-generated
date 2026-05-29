#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Block size for blocked Cholesky
static const int NB = 128;

// ---------- 1-D row distribution helper ----------
struct RowDist {
    int n;
    int mpi_size;
    int mpi_rank;
    int n_local;
    int row_start;

    RowDist() : n(0), mpi_size(0), mpi_rank(0), n_local(0), row_start(0) {}

    explicit RowDist(int n_, int sz_, int rk_)
        : n(n_), mpi_size(sz_), mpi_rank(rk_)
    {
        int base  = n / mpi_size;
        int extra = n % mpi_size;
        n_local   = base + (mpi_rank < extra ? 1 : 0);
        row_start = mpi_rank * base + std::min(mpi_rank, extra);
    }

    int n_local_of(int r) const {
        int base = n / mpi_size, extra = n % mpi_size;
        return base + (r < extra ? 1 : 0);
    }
    int row_start_of(int r) const {
        int base = n / mpi_size, extra = n % mpi_size;
        return r * base + std::min(r, extra);
    }
    int rank_of_row(int row) const {
        for (int r = 0; r < mpi_size; ++r) {
            int rs = row_start_of(r), rl = n_local_of(r);
            if (row >= rs && row < rs + rl) return r;
        }
        return mpi_size - 1;
    }
};

// ---------- matrix generation (OpenMP for matmul) ----------
void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i)
        B[i] = (static_cast<double>(rand_r(&seed)) /
                static_cast<double>(RAND_MAX)) - 0.5;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double* Bi = B.data() + i * n;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const double* Bj = B.data() + j * n;
            for (size_t k = 0; k < n; ++k)
                sum += Bi[k] * Bj[k];
            A[i * n + j] = sum;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += static_cast<double>(n);
}

// ---------- hybrid MPI + CUDA + OpenMP blocked Cholesky ----------
// A is the *local* row block (row-major, n_local x n) on every rank.
// Matrix stored column-major on GPU: A_col[j * n_local + i] = A[i][j]
//
// Blocked algorithm (column-cycling):
//   for k = 0; k < n; k += NB:
//     1. POTRF: factor A11 -> L11  (rank owning rows k..k+ps-1)
//     2. Bcast L11 to all ranks
//     3. TRSM:  L21 = A21 * L11^{-T}  (all ranks with rows below diagonal)
//     4. All-gather L21
//     5. GEMM:  A22 = A22 - L21 * L21^T
bool choleskyDecomposition(std::vector<double>& A, size_t n,
                           int mpi_size, int mpi_rank)
{
    RowDist dist(static_cast<int>(n), mpi_size, mpi_rank);
    const int n_local   = dist.n_local;
    const int row_start = dist.row_start;
    const int n_int     = static_cast<int>(n);

    // ---- ranks with no rows still participate in MPI calls ----
    if (n_local == 0) {
        for (int k = 0; k < n_int; k += NB) {
            int ps = std::min(NB, n_int - k);
            MPI_Barrier(MPI_COMM_WORLD);
            std::vector<double> dbuf(ps * ps);
            MPI_Bcast(dbuf.data(), ps * ps, MPI_DOUBLE,
                      dist.rank_of_row(k), MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            int diag_end = k + ps;
            if (diag_end < n_int) {
                std::vector<int> sc(mpi_size), sd(mpi_size);
                for (int r = 0; r < mpi_size; ++r) {
                    int rs = dist.row_start_of(r), rl = dist.n_local_of(r);
                    int os = std::max(rs, diag_end);
                    int oe = std::min(rs + rl, n_int);
                    int oc = std::max(0, oe - os);
                    sc[r] = oc * ps;
                    sd[r] = (os - diag_end) * ps;
                }
                MPI_Allgatherv(nullptr, 0, MPI_DOUBLE,
                               nullptr, sc.data(), sd.data(),
                               MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
        return true;
    }

    // ---- convert row-major -> column-major for cuBLAS ----
    std::vector<double> A_col(n_local * n);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_int; ++j)
        for (int i = 0; i < n_local; ++i)
            A_col[j * n_local + i] = A[i * n + j];

    // ---- GPU memory ----
    // dA: n_local x n (column-major, ld=n_local)
    double *dA = nullptr;
    // dL11: ps x ps (column-major, ld=ps)
    double *dL11 = nullptr;
    // dL21: (n-diag_end) x ps (column-major, ld=n-diag_end)
    double *dL21 = nullptr;

    cudaError_t ce;
    if ((ce = cudaMalloc(&dA, n_local * n * sizeof(double))) != cudaSuccess ||
        (ce = cudaMemcpy(dA, A_col.data(), n_local * n * sizeof(double),
                          cudaMemcpyHostToDevice)) != cudaSuccess) {
        if (mpi_rank == 0) printf("CUDA error: %s\n", cudaGetErrorString(ce));
        return false;
    }
    cudaMalloc(&dL11, NB * NB * sizeof(double));
    // dL21: worst case is (n-1) x NB, use n x NB with ld=n
    // dL21: ld=n, max rows = n-NB, so need n*n
    cudaMalloc(&dL21, n * n * sizeof(double));

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cusolverDnHandle_t solver = nullptr;
    cusolverDnCreate(&solver);

    const double alpha     =  1.0;
    const double beta_val  =  1.0;
    const double alpha_neg = -1.0;

    std::vector<double> diag_buf(NB * NB);
    std::vector<double> send_buf(n_local * NB);
    // recv_buf: packed L21 with ld = l21_rows (varies)
    // recv_buf: worst case l21_rows*n = n*n, but we pack with ld=l21_rows
    std::vector<double> recv_buf(n * NB);
    std::vector<double> dwork;

    bool success = true;

    // ---- blocked Cholesky loop ----
    for (int k = 0; k < n_int && success; k += NB) {
        const int ps        = std::min(NB, n_int - k);
        const int rank_diag = dist.rank_of_row(k);
        const int diag_end  = k + ps;

        // === 1. Factor diagonal block (POTRF via cuSOLVER) ===
        if (mpi_rank == rank_diag) {
            const int lro = k - row_start;
            // d_diag points to column k of dA, row lro
            double* d_diag = dA + k * n_local + lro;

            int buf_size = 0;
            cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                                        ps, d_diag, n_local, &buf_size);
            if (static_cast<int>(dwork.size()) < buf_size)
                dwork.resize(buf_size);

            int *d_dinfo = nullptr;
            cudaMalloc(&d_dinfo, sizeof(int));
            cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER,
                             ps, d_diag, n_local,
                             dwork.data(), buf_size, d_dinfo);
            cudaDeviceSynchronize();

            int h_dinfo = 0;
            cudaMemcpy(&h_dinfo, d_dinfo, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_dinfo);

            if (h_dinfo > 0) {
                printf("Error: Matrix is not positive definite "
                       "at diagonal element %d\n", k + h_dinfo - 1);
                success = false;
            } else {
                // Extract ps x ps L11 block from d_diag (ld=n_local) to diag_buf (ld=ps)
                for (int p = 0; p < ps; ++p)
                    cudaMemcpy(diag_buf.data() + p * ps,
                               d_diag + p * n_local,
                               ps * sizeof(double),
                               cudaMemcpyDeviceToHost);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (!success) break;

        // === 2. Broadcast diagonal block ===
        MPI_Bcast(diag_buf.data(), ps * ps, MPI_DOUBLE,
                  rank_diag, MPI_COMM_WORLD);
        cudaMemcpy(dL11, diag_buf.data(),
                   ps * ps * sizeof(double),
                   cudaMemcpyHostToDevice);

        // === 3. Solve columns below diagonal (TRSM) ===
        // L21 = A21 * L11^{-T}
        // Only for rows >= diag_end
        if (diag_end < n_int) {
            const int s0 = std::max(row_start, diag_end);
            const int s1 = row_start + n_local;
            if (s0 < s1) {
                const int m  = s1 - s0;
                const int lo = s0 - row_start;
                // dS points to column k, row lo of dA
                double* dS = dA + k * n_local + lo;

                // TRSM: dS = alpha * dS * inv(L11^T)
                // dS is m x ps (ld=n_local), dL11 is ps x ps (ld=ps)
                cublasDtrsm(handle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    m, ps, &alpha, dL11, ps, dS, n_local);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // === 4. All-gather solved L21 ===
        if (diag_end < n_int) {
            const int l21_rows = n_int - diag_end;

            std::vector<int> sc(mpi_size), sd(mpi_size);
            for (int r = 0; r < mpi_size; ++r) {
                int rs = dist.row_start_of(r), rl = dist.n_local_of(r);
                int os = std::max(rs, diag_end);
                int oe = std::min(rs + rl, n_int);
                int oc = std::max(0, oe - os);
                sc[r] = oc * ps;
                sd[r] = (os - diag_end) * ps;
            }

            // Copy local L21 slice -> send buffer
            // Local rows from diag_end onward
            const int ll0 = std::max(0, diag_end - row_start);
            int llc = n_local - ll0;
            if (llc < 0) llc = 0;

            if (llc > 0) {
                double* dL = dA + k * n_local + ll0;
                // dL has ld=n_local, send_buf is packed with ld=llc
                cudaMemcpy2D(send_buf.data(), llc * sizeof(double),
                             dL, n_local * sizeof(double),
                             ps * sizeof(double), llc,
                             cudaMemcpyDeviceToHost);
            }

            MPI_Allgatherv(send_buf.data(), llc * ps, MPI_DOUBLE,
                           recv_buf.data(), sc.data(), sd.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);

            // Copy recv_buf (ld=l21_rows) -> dL21 (ld=n)
            cudaMemcpy2D(dL21, n * sizeof(double),
                         recv_buf.data(), l21_rows * sizeof(double),
                         ps * sizeof(double), l21_rows,
                         cudaMemcpyHostToDevice);

            // === 5. Update trailing submatrix (GEMM) ===
            // A22 = A22 - L21_local * L21_full^T
            // L21_local: mu x ps (ld=n_local)
            // L21_full:  l21_rows x ps (ld=n)
            // A22:       mu x l21_rows (ld=n_local)
            const int u0 = std::max(row_start, diag_end);
            const int u1 = row_start + n_local;
            if (u0 < u1) {
                const int mu = u1 - u0;
                const int lo = u0 - row_start;

                double* dA_L21 = dA + k * n_local + lo;
                double* dC     = dA + diag_end * n_local + lo;

                // C(m,n) = alpha * A(m,k) * B(n,k)^T + beta * C(m,n)
                //   = alpha * L21_local(mu,ps) * L21_full(l21_rows,ps)^T + beta * A22
                //   = alpha * L21_local * L21_full^T + beta * A22
                //   = -1 * L21_local * L21_full^T + 1 * A22
                cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    mu, l21_rows, ps,
                    &alpha_neg,
                    dA_L21, n_local,
                    dL21,   n,
                    &beta_val,
                    dC,     n_local);

                // Debug: verify A22 diagonal is positive
                if (mpi_rank == 0) {
                    int dlen = std::min(mu, l21_rows);
                    std::vector<double> diag_check(dlen);
                    for (int i = 0; i < dlen; ++i)
                        cudaMemcpy(&diag_check[i], dC + i * n_local + i,
                                   sizeof(double), cudaMemcpyDeviceToHost);
                    double min_d = diag_check[0], max_d = diag_check[0];
                    for (int i = 1; i < dlen; ++i) {
                        min_d = std::min(min_d, diag_check[i]);
                        max_d = std::max(max_d, diag_check[i]);
                    }
                    printf("DEBUG block k=%d: A22 diag min=%.6f max=%.6f\n", k, min_d, max_d);
                }
            }
        }
    }

    // ---- copy result GPU -> CPU ----
    cudaMemcpy(A_col.data(), dA,
               n_local * n * sizeof(double),
               cudaMemcpyDeviceToHost);

    // ---- column-major -> row-major ----
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_int; ++j)
        for (int i = 0; i < n_local; ++i)
            A[i * n + j] = A_col[j * n_local + i];

    // ---- zero upper triangle ----
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        int gi = row_start + i;
        for (int j = gi + 1; j < n_int; ++j)
            A[i * n + j] = 0.0;
    }

    cudaFree(dA);
    cudaFree(dL11);
    cudaFree(dL21);
    cublasDestroy(handle);
    cusolverDnDestroy(solver);
    return success;
}

// ---------- validation (OpenMP) ----------
bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, size_t n)
{
    std::vector<double> reconstructed(n * n);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        double e = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, e);
        relError = std::max(relError,
            e / (fabs(A_orig[i]) + 1e-10));
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
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
    printf("Parallel: MPI + CUDA + OpenMP (hybrid)\n");
}

int main(int argc, char** argv) {
    int mpi_size = 1, mpi_rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("CUDA device: %s\n", prop.name);
    }

    RowDist dist(static_cast<int>(n), mpi_size, mpi_rank);
    const int n_local = dist.n_local;

    std::vector<double> A_local(n_local * n);
    std::vector<double> A_orig_local(n_local * n);

    // Generate full matrix on rank 0, then scatter
    std::vector<double> A_full;
    if (mpi_rank == 0) {
        A_full.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
    }

    // Scatter rows to all ranks
    {
        std::vector<int> sc(mpi_size), sd(mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            int rs = dist.row_start_of(r), rl = dist.n_local_of(r);
            sc[r] = rl * static_cast<int>(n);
            sd[r] = rs * static_cast<int>(n);
        }
        if (mpi_rank == 0) {
            MPI_Scatterv(A_full.data(), sc.data(), sd.data(),
                         MPI_DOUBLE,
                         A_local.data(), n_local * static_cast<int>(n),
                         MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(nullptr, sc.data(), sd.data(),
                         MPI_DOUBLE,
                         A_local.data(), n_local * static_cast<int>(n),
                         MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    if (validate)
        A_orig_local = A_local;

    if (mpi_rank == 0)
        printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    bool success = choleskyDecomposition(A_local, n, mpi_size, mpi_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (mpi_rank == 0)
            printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full result for print/validation
    std::vector<double> A_full_result;
    if (printResults || validate) {
        std::vector<int> sc(mpi_size), sd(mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            int rs = dist.row_start_of(r), rl = dist.n_local_of(r);
            sc[r] = rl * static_cast<int>(n);
            sd[r] = rs * static_cast<int>(n);
        }
        A_full_result.resize(n * n);
        MPI_Gatherv(A_local.data(), n_local * static_cast<int>(n), MPI_DOUBLE,
                    A_full_result.data(), sc.data(), sd.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (printResults && mpi_rank == 0)
        print_results(A_full_result, "CholeskyL");

    if (validate && mpi_rank == 0) {
        std::vector<double> A_full_orig;
        std::vector<int> sc(mpi_size), sd(mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            int rs = dist.row_start_of(r), rl = dist.n_local_of(r);
            sc[r] = rl * static_cast<int>(n);
            sd[r] = rs * static_cast<int>(n);
        }
        A_full_orig.resize(n * n);
        MPI_Gatherv(A_orig_local.data(), n_local * static_cast<int>(n), MPI_DOUBLE,
                    A_full_orig.data(), sc.data(), sd.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        printf("Validating result...\n");
        bool valid = validateCholesky(A_full_result, A_full_orig, n);

        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
