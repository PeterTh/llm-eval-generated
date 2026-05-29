#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define BLOCK_SIZE 128

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = call;                                               \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(_err));                                 \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _st = call;                                             \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    static_cast<int>(_st));                                    \
            return false;                                                      \
        }                                                                      \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Blocked Cholesky  L = chol(A)  with hybrid MPI + OpenMP + CUDA    */
/*                                                                  */
/*  MPI    – each rank owns a contiguous block of rows.  Panels       */
/*           are gathered with MPI_Allgatherv so every rank has       */
/*           the data it needs for the GEMM update.                   */
/*  CUDA   – cuBLAS TRSM (panel solve) and GEMM (rank-k update).     */
/*  OpenMP – panel factorisation, host-side transposes, matrix       */
/*           generation & validation.                                 */
/* ------------------------------------------------------------------ */

/* Unblocked Cholesky on a pc x pc lower-triangular block (column-major,
 * lda = pc).  M[col*pc + row] = M_{row,col}.                        */
static bool chol_panel_cpu(double* M, int pc) {
    for (int j = 0; j < pc; ++j) {
        /* Subtract dot products from column j */
        #pragma omp parallel for
        for (int i = j; i < pc; ++i) {
            double s = 0.0;
            for (int k = 0; k < j; ++k)
                s += M[k * pc + i] * M[k * pc + j];
            M[j * pc + i] -= s;
        }
        double d = M[j * pc + j];
        if (d <= 0.0) {
            fprintf(stderr, "Not positive definite at (%d,%d)\n", j, j);
            return false;
        }
        d = sqrt(d);
        M[j * pc + j] = d;
        #pragma omp parallel for
        for (int i = j + 1; i < pc; ++i)
            M[j * pc + i] /= d;
    }
    return true;
}

/* Copy a rectangular block from d_A (column-major, lda) to host
 * row-major buffer.  Block covers rows [r0, r0+nr) and columns
 * [c0, c0+nc).                                                       */
static void gpu_to_host_rm(const double* d_A, int lda,
                           int r0, int c0, int nr, int nc,
                           double* h_rm) {
    std::vector<double> col_tmp(nr);
    for (int c = 0; c < nc; ++c) {
        cudaMemcpy(col_tmp.data(),
                   d_A + static_cast<ptrdiff_t>(c0 + c) * lda + r0,
                   nr * sizeof(double), cudaMemcpyDeviceToHost);
        #pragma omp parallel for
        for (int r = 0; r < nr; ++r)
            h_rm[r * nc + c] = col_tmp[r];
    }
    cudaDeviceSynchronize();
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           int mpi_rank, int mpi_size) {
    const int bs = BLOCK_SIZE;
    const int N  = static_cast<int>(n);

    /* Assign each MPI rank a GPU device */
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    int my_device = mpi_rank % std::max(num_devices, 1);
    CUDA_CHECK(cudaSetDevice(my_device));

    /* Row assignment: rank r owns rows [r*rrp, min((r+1)*rrp, N)) */
    const int rrp      = (N + mpi_size - 1) / mpi_size;
    const int my_start = mpi_rank * rrp;
    const int my_end   = std::min((mpi_rank + 1) * rrp, N);
    const int my_rows  = my_end - my_start;

    /* GPU memory: full N x N column-major matrix, lda = N.
     * Each rank holds the entire matrix on its GPU.                       */
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(double)));

    /* Convert from row-major host to column-major GPU */
    std::vector<double> h_col(N * N);
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            h_col[c * N + r] = A[r * N + c];
    CUDA_CHECK(cudaMemcpy(d_A, h_col.data(),
                          N * N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double alpha_m1 = -1.0, alpha_1 = 1.0, beta_1 = 1.0;

    /* Pre-allocate GPU buffers for panel ops */
    double* d_panel = nullptr;   /* pc x pc, lda=pc */
    double* d_o     = nullptr;   /* up to (N) x pc */
    CUDA_CHECK(cudaMalloc(&d_panel, bs * bs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_o, N * bs * sizeof(double)));

    std::vector<int> recv_counts(mpi_size), recv_displs(mpi_size);
    std::vector<double> dummy_buf(1);

    for (int k = 0; k < N; k += bs) {
        const int pc = std::min(bs, N - k);

        /* ================================================================ */
        /* Step 1 – gather diagonal block A(k:k+pc, k:k+pc)                */
        /* ================================================================ */
        const int d_start = std::max(my_start, k);
        const int d_end   = std::min(my_end, k + pc);
        const int d_rows  = std::max(0, d_end - d_start);

        std::vector<double> d_send(d_rows * pc);
        if (d_rows > 0)
            gpu_to_host_rm(d_A, N, d_start, k, d_rows, pc, d_send.data());

        /* recv_counts/displs in ELEMENTS (each row contributes pc elements) */
        int total_recv = 0;
        for (int r = 0; r < mpi_size; ++r) {
            const int rs  = r * rrp;
            const int re  = std::min((r + 1) * rrp, N);
            const int rss = std::max(rs, k);
            const int ree = std::min(re, k + pc);
            recv_counts[r] = std::max(0, ree - rss) * pc;
            recv_displs[r] = total_recv;
            total_recv    += recv_counts[r];
        }
        std::vector<double> d_recv(total_recv, 0.0);
        MPI_Allgatherv(d_rows > 0 ? d_send.data() : dummy_buf.data(),
                       d_rows * pc, MPI_DOUBLE,
                       d_recv.data(), recv_counts.data(),
                       recv_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        /* Transpose to column-major */
        std::vector<double> d_col(pc * pc);
        #pragma omp parallel for collapse(2)
        for (int r = 0; r < pc; ++r)
            for (int c = 0; c < pc; ++c)
                d_col[c * pc + r] = d_recv[r * pc + c];

        /* ================================================================ */
        /* Step 2 – POTRF on diagonal block (CPU + OpenMP)                 */
        /* ================================================================ */
        if (!chol_panel_cpu(d_col.data(), pc)) {
            CUDA_CHECK(cudaFree(d_panel));
            CUDA_CHECK(cudaFree(d_o));
            CUDA_CHECK(cudaFree(d_A));
            CUBLAS_CHECK(cublasDestroy(handle));
            return false;
        }

        /* Broadcast factored panel to GPU buffer (lda=pc) */
        CUDA_CHECK(cudaMemcpy(d_panel, d_col.data(),
                              pc * pc * sizeof(double),
                              cudaMemcpyHostToDevice));

        /* Copy factored panel back to d_A for the rows this rank owns */
        if (d_rows > 0) {
            for (int c = 0; c < pc; ++c) {
                CUDA_CHECK(cudaMemcpy(
                    d_A + static_cast<ptrdiff_t>(k + c) * N + d_start,
                    d_panel + static_cast<ptrdiff_t>(c) * pc + (d_start - k),
                    d_rows * sizeof(double),
                    cudaMemcpyDeviceToDevice));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        /* ================================================================ */
        /* Step 3 – TRSM for off-diagonal panel rows (GPU)                 */
        /*    Solve:  B = B * inv(L^T)  (SIDE_RIGHT, OP_T)                 */
        /* ================================================================ */
        const int tr_start = std::max(my_start, k + pc);
        const int tr_rows  = std::max(0, my_end - tr_start);

        if (tr_rows > 0) {
            CUBLAS_CHECK(cublasDtrsm(handle, CUBLAS_SIDE_RIGHT,
                                     CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T,
                                     CUBLAS_DIAG_NON_UNIT,
                                     tr_rows, pc, &alpha_1,
                                     d_panel, pc,
                                     d_A + static_cast<ptrdiff_t>(k) * N +
                                         tr_start,
                                     N));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        /* ================================================================ */
        /* Step 4 – gather off-diag panel A(k+pc:N, k:k+pc)                */
        /* ================================================================ */
        const int o_start = std::max(my_start, k + pc);
        const int o_rows  = std::max(0, my_end - o_start);

        std::vector<double> o_send(o_rows * pc);
        if (o_rows > 0)
            gpu_to_host_rm(d_A, N, o_start, k, o_rows, pc, o_send.data());

        /* recv_counts/displs in ELEMENTS */
        int total_o_recv = 0;
        for (int r = 0; r < mpi_size; ++r) {
            const int rs  = r * rrp;
            const int re  = std::min((r + 1) * rrp, N);
            const int rss = std::max(rs, k + pc);
            recv_counts[r] = std::max(0, re - rss) * pc;
            recv_displs[r] = total_o_recv;
            total_o_recv  += recv_counts[r];
        }
        std::vector<double> o_recv(total_o_recv);
        MPI_Allgatherv(o_rows > 0 ? o_send.data() : dummy_buf.data(),
                       o_rows * pc, MPI_DOUBLE,
                       o_recv.data(), recv_counts.data(),
                       recv_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        /* ================================================================ */
        /* Step 5 – GEMM trailing-submatrix update (GPU)                   */
        /*    C = C - L * L^T                                              */
        /* ================================================================ */
        const int u_start = k + pc;
        const int u_rows  = N - u_start;

        if (u_rows > 0) {
            const int eff = std::max(my_start, u_start);
            if (eff < my_end) {
                const int nr = my_end - eff;
                const int total_o_rows = total_o_recv / pc;

                /* Transpose off-diagonal panel to column-major */
                std::vector<double> o_col(total_o_recv);
                #pragma omp parallel for collapse(2)
                for (int r = 0; r < total_o_rows; ++r)
                    for (int c = 0; c < pc; ++c)
                        o_col[c * total_o_rows + r] = o_recv[r * pc + c];

                CUDA_CHECK(cudaMemcpy(d_o, o_col.data(),
                                      total_o_recv * sizeof(double),
                                      cudaMemcpyHostToDevice));

                /* C = C - A * B^T
                 * A = L(eff:my_end, k:k+pc),  nr x pc
                 * B = L(k+pc:N, k:k+pc),      total_o_rows x pc
                 * C = L(eff:my_end, u_start:N), nr x u_rows   */
                CUBLAS_CHECK(cublasDgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    nr, u_rows, pc, &alpha_m1,
                    d_A + static_cast<ptrdiff_t>(k) * N + eff, N,
                    d_o, total_o_rows,
                    &beta_1,
                    d_A + static_cast<ptrdiff_t>(u_start) * N + eff, N));
            }
        }
    }

    /* ---- Copy results back to host (only local rows) ---- */
    if (my_rows > 0) {
        std::vector<double> h_back(N * my_rows);
        /* Extract rows [my_start, my_end) from d_A (column-major, lda=N) */
        for (int c = 0; c < N; ++c) {
            cudaMemcpy(h_back.data() + c * my_rows,
                       d_A + static_cast<ptrdiff_t>(c) * N + my_start,
                       my_rows * sizeof(double), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();

        #pragma omp parallel for collapse(2)
        for (int r = 0; r < my_rows; ++r)
            for (int c = 0; c < N; ++c)
                A[(my_start + r) * N + c] = h_back[c * my_rows + r];
    }

    CUDA_CHECK(cudaFree(d_panel));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_A));
    CUBLAS_CHECK(cublasDestroy(handle));

    /* Gather all rows back to rank 0 */
    std::vector<int> g_counts(mpi_size), g_displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        const int rs = r * rrp;
        const int re = std::min((r + 1) * rrp, N);
        g_counts[r] = (re - rs) * N;
        g_displs[r] = rs * N;
    }
    MPI_Gatherv(A.data(), my_rows * N, MPI_DOUBLE,
                A.data(), g_counts.data(), g_displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Zero upper triangular part on host (rank 0) */
    if (mpi_rank == 0) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                A[i * N + j] = 0.0;
    }

    return true;
}

/* Generate A = B*B^T + n*I  (OpenMP parallelised) */
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    #pragma omp parallel for
    for (size_t i = 0; i < n * n; ++i) {
        unsigned int s = seed + static_cast<unsigned int>(i);
        B[i] = (rand_r(&s) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += B[i * n + k] * B[j * n + k];
            A[i * n + j] = s;
        }
    }

    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += static_cast<double>(n);
}

/* Validate L*L^T ≈ A_orig  (OpenMP parallelised) */
bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> R(n * n);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += L[i * n + k] * L[j * n + k];
            R[i * n + j] = s;
        }
    }

    double maxErr = 0.0, relErr = 0.0;
    #pragma omp parallel for reduction(max : maxErr, relErr)
    for (size_t i = 0; i < n * n; ++i) {
        const double e = fabs(R[i] - A_orig[i]);
        if (e > maxErr)
            maxErr = e;
        const double r = e / (fabs(A_orig[i]) + 1e-10);
        if (r > relErr)
            relErr = r;
    }

    printf("Max absolute error: %.10e\n", maxErr);
    printf("Max relative error: %.10e\n", relErr);

    if (relErr > 1e-6) {
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

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    size_t n = 512;
    int validate = 0, printResults = 0;

    if (mpi_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    /* broadcast parameters */
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    /* generate on rank 0, broadcast */
    if (mpi_rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }
    MPI_Bcast(A.data(), static_cast<int>(n * n), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    if (validate && mpi_rank == 0)
        A_orig = A;

    /* ---- Cholesky (timed) ---- */
    if (mpi_rank == 0)
        printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    bool ok = choleskyDecomposition(A, n, mpi_rank, mpi_size);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    if (!ok) {
        if (mpi_rank == 0)
            printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                       .count();
        printf("Computation time: %ld ms\n", ms);

        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults)
            print_results(A, "CholeskyL");

        if (validate) {
            printf("Validating result...\n");
            if (validateCholesky(A, A_orig, n)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
