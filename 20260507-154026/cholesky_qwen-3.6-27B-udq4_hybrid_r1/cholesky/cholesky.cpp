#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = call;                                                 \
        if (_err != cudaSuccess) {                                               \
            if (rank == 0)                                                       \
                printf("CUDA error %d: %s at %s:%d\n", _err,                     \
                       cudaGetErrorString(_err), __FILE__, __LINE__);            \
            return false;                                                        \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t _st = call;                                               \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                       \
            if (rank == 0)                                                       \
                printf("cuBLAS error %d at %s:%d\n", _st, __FILE__, __LINE__);   \
            return false;                                                        \
        }                                                                        \
    } while (0)

#define CUSOLVER_CHECK(call)                                                     \
    do {                                                                         \
        cusolverStatus_t _st = call;                                             \
        if (_st != CUSOLVER_STATUS_SUCCESS) {                                     \
            if (rank == 0)                                                       \
                printf("cuSOLVER error %d at %s:%d\n", _st, __FILE__, __LINE__);  \
            return false;                                                        \
        }                                                                        \
    } while (0)

/* ------------------------------------------------------------------ */
/*  CUDA kernels for strided sub-matrix <-> contiguous-buffer copies   */
/*  d_local is column-major with lda = n.  Sub-blocks starting at     */
/*  arbitrary row offsets are NOT contiguous, so cudaMemcpy cannot     */
/*  be used.  These kernels copy nb x nb or m x nb blocks between      */
/*  strided (lda=n) and contiguous (lda=nb) layouts.                   */
/* ------------------------------------------------------------------ */

__global__ void copyBlockD2H(const double* __restrict__ src,
                              double* __restrict__ dst,
                              int n, int nb, int row_off) {
    /* Copy nb x nb block from d_local (lda=n) at row_off to contiguous (lda=nb) */
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < nb && c < nb)
        dst[c * nb + r] = src[c * n + row_off + r];
}

__global__ void copyBlockH2D(const double* __restrict__ src,
                              double* __restrict__ dst,
                              int n, int nb, int row_off) {
    /* Copy nb x nb block from contiguous (lda=nb) to d_local (lda=n) at row_off */
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < nb && c < nb)
        dst[c * n + row_off + r] = src[c * nb + r];
}

__global__ void copyWD2H(const double* __restrict__ src,
                          double* __restrict__ dst,
                          int n, int m, int nb, int row_off) {
    /* Copy m x nb block from d_local (lda=n) at row_off to contiguous (lda=nb) */
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < m && c < nb)
        dst[c * nb + r] = src[c * n + row_off + r];
}

/* ------------------------------------------------------------------ */
/*  Matrix generation - OpenMP parallelised                            */
/* ------------------------------------------------------------------ */
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    /* A = B * B^T */
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += n;
}

/* ------------------------------------------------------------------ */
/*  Blocked Cholesky  -  1-D column distribution + CUDA + MPI         */
/*  Each rank owns nb = n/num_ranks columns, stored column-major on   */
/*  GPU with lda = n.                                                 */
/*                                                                    */
/*  Algorithm (per block step kc):                                     */
/*    1. POTRF on diagonal block (owned by rank kc)                    */
/*    2. Broadcast L(kc,kc) to all ranks                              */
/*    3. Rank kc computes W = A21 * L(kc,kc)^{-T} via TRSM            */
/*       (W is m x nb, m = n - (kc+1)*nb)                             */
/*    4. Broadcast W to all ranks                                     */
/*    5. Each rank r > kc updates its trailing submatrix:              */
/*       A_rc -= W * W_r^T  where W_r is the nb x nb slice of W       */
/*       corresponding to rank r's rows in the trailing submatrix     */
/* ------------------------------------------------------------------ */
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    extern int rank;
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const size_t nb = n / num_ranks;          /* columns per rank */
    const size_t local_elems = n * nb;

    /* ---- device memory (column-major, lda = n) ---------------------- */
    double *d_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_local, local_elems * sizeof(double)));

    /* row-major host -> column-major device */
    std::vector<double> h_local(local_elems);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < nb; ++j)
            h_local[j * n + i] = A[i * n + (rank * nb + j)];
    CUDA_CHECK(cudaMemcpy(d_local, h_local.data(),
                          local_elems * sizeof(double), cudaMemcpyHostToDevice));

    /* ---- cuBLAS / cuSOLVER handles ---------------------------------- */
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    /* ---- pinned host buffer for MPI communication ------------------- */
    double *h_buf = nullptr;
    /* worst case: m*nb where m can be up to n-nb, so allocate n*nb */
    cudaMallocHost(&h_buf, n * nb * sizeof(double));

    /* ---- device buffer for W (m x nb, contiguous, lda = nb) --------- */
    double *d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, n * nb * sizeof(double)));

    /* ---- device scalars (alpha / beta) ------------------------------- */
    double *d_alpha = nullptr, *d_beta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta, sizeof(double)));
    double h_one  =  1.0;
    double h_mone = -1.0;
    CUDA_CHECK(cudaMemcpy(d_alpha, &h_one,  sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  &h_one,  sizeof(double), cudaMemcpyHostToDevice));

    /* ---- cuSOLVER POTRF workspace ------------------------------------ */
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                                               (int)nb, h_buf, (int)n, &lwork));
    double *d_work = nullptr;
    if (lwork > 0) CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
    int *devInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    /* ---- kernel block size ------------------------------------------ */
    dim3 bsz(16, 16);

    /* ================================================================== */
    /*  Main blocked loop                                                 */
    /* ================================================================== */
    for (size_t kc = 0; kc < (size_t)num_ranks; ++kc) {
        const size_t m = n - (kc + 1) * nb;   /* trailing rows below diagonal block */

        /* 1. POTRF on diagonal block (owner = rank kc)                   */
        /*    Diagonal block is at rows kc*nb .. (kc+1)*nb-1,             */
        /*    cols 0 .. nb-1 of d_local (lda=n).                         */
        if (rank == (int)kc) {
            double *d_blk = d_local + kc * nb;
            CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER,
                                            (int)nb, d_blk, (int)n,
                                            d_work, lwork, devInfo));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (m == 0) break;   /* nothing left to update */

        /* 2. Broadcast L(kc,kc) to every rank                           */
        {
            dim3 blk((nb + bsz.x - 1) / bsz.x,
                     (nb + bsz.y - 1) / bsz.y);

            if (rank == (int)kc) {
                /* Copy nb x nb diagonal block from strided d_local to contiguous h_buf */
                copyBlockD2H<<<blk, bsz>>>(d_local, h_buf,
                                           (int)n, (int)nb, (int)(kc * nb));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Bcast(h_buf, (int)(nb * nb), MPI_DOUBLE, (int)kc, MPI_COMM_WORLD);

            /* Write back to strided d_local */
            copyBlockH2D<<<blk, bsz>>>(h_buf, d_local,
                                       (int)n, (int)nb, (int)(kc * nb));
        }

        /* 3. TRSM  W = A21 * L^{-T}   (rank kc only)                    */
        /*    A21 is at rows (kc+1)*nb .. n-1, cols 0 .. nb-1 of d_local */
        /*    This is m x nb with lda=n.                                  */
        /*    TRSM: B = alpha * B * inv(L^T)                             */
        if (rank == (int)kc) {
            double *d_A21 = d_local + (kc + 1) * nb;
            double *d_L   = d_local + kc * nb;

            CUDA_CHECK(cudaMemcpy(d_alpha, &h_one, sizeof(double),
                                  cudaMemcpyHostToDevice));
            CUBLAS_CHECK(cublasDtrsm(handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                (int)m, (int)nb, d_alpha,
                d_L, (int)n,
                d_A21, (int)n));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        /* 4. Broadcast W to all ranks                                   */
        /*    W is m x nb.  Copy from strided d_local to contiguous h_buf, */
        /*    broadcast via MPI, then upload to d_W (lda=nb).             */
        {
            dim3 blk((nb + bsz.x - 1) / bsz.x,
                     (m + bsz.y - 1) / bsz.y);

            if (rank == (int)kc) {
                copyWD2H<<<blk, bsz>>>(d_local, h_buf,
                                       (int)n, (int)m, (int)nb,
                                       (int)((kc + 1) * nb));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Bcast(h_buf, (int)(m * nb), MPI_DOUBLE, (int)kc, MPI_COMM_WORLD);

            /* Upload to device (both contiguous, lda=nb) */
            CUDA_CHECK(cudaMemcpy(d_W, h_buf,
                                  m * nb * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }

        /* 5. GEMM  A_rc -= W * W_r^T   (ranks kc+1 .. p-1)              */
        /*    W is m x nb (lda=nb).                                       */
        /*    W_r is nb x nb slice of W for rank r's rows:               */
        /*      W_r = W[(r-kc-1)*nb .. (r-kc)*nb-1, :]                   */
        /*    A_rc is m x nb at rows (kc+1)*nb .. n-1, cols 0 .. nb-1    */
        /*    of d_local (lda=n).                                         */
        /*    GEMM: C(m x nb) = -1 * A(m x nb) * B^T(nb x nb) + 1 * C   */
        if (rank > (int)kc) {
            double *d_Wr = d_W + (rank - (int)kc - 1) * nb;
            double *d_rc = d_local + (kc + 1) * nb;

            CUDA_CHECK(cudaMemcpy(d_alpha, &h_mone, sizeof(double),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_beta,  &h_one,  sizeof(double),
                                  cudaMemcpyHostToDevice));

            CUBLAS_CHECK(cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                (int)m, (int)nb, (int)nb,
                d_alpha, d_W,      (int)nb,
                        d_Wr,      (int)nb,
                d_beta,  d_rc,     (int)n));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ---- copy result back (column-major -> row-major) ---------------- */
    CUDA_CHECK(cudaMemcpy(h_local.data(), d_local,
                          local_elems * sizeof(double),
                          cudaMemcpyDeviceToHost));
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < nb; ++j)
            A[i * n + (rank * nb + j)] = h_local[j * n + i];

    /* ---- cleanup ---------------------------------------------------- */
    CUDA_CHECK(cudaFree(d_local));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_beta));
    if (d_work)  CUDA_CHECK(cudaFree(d_work));
    if (devInfo) CUDA_CHECK(cudaFree(devInfo));
    cudaFreeHost(h_buf);
    cublasDestroy(handle);
    cusolverDnDestroy(solver);
    return true;
}

/* ------------------------------------------------------------------ */
/*  Validation - OpenMP parallelised                                  */
/* ------------------------------------------------------------------ */
bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = sum;
        }

    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        const double err = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, err);
        relError = std::max(relError,
                            err / (fabs(A_orig[i]) + 1e-10));
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    return relError <= 1e-6;
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */
int rank;  /* global so choleskyDecomposition can reference it */

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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = 512;
    int validate = 0, printResults = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = (size_t)atoi(argv[++i]);
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
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d, OpenMP threads: %d\n",
               num_ranks, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    if (validate) A_orig = A;

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    double t0 = MPI_Wtime();
    bool success = choleskyDecomposition(A, n);
    double t1 = MPI_Wtime();

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    /* Zero upper triangle on each rank's local columns (OpenMP) */
    {
        const size_t nb = n / num_ranks;
        const size_t col_start = rank * nb;
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = col_start; j < col_start + nb; ++j)
                if (j > i)
                    A[i * n + j] = 0.0;
    }

    /* Gather all columns from all ranks so every rank has the full matrix */
    {
        const size_t nb = n / num_ranks;
        std::vector<double> sendbuf(n * nb);
        std::vector<double> recvbuf(n * n);
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < nb; ++j)
                sendbuf[j * n + i] = A[i * n + (rank * nb + j)];
        MPI_Allgather(sendbuf.data(), (int)(n * nb), MPI_DOUBLE,
                      recvbuf.data(),  (int)(n * nb), MPI_DOUBLE,
                      MPI_COMM_WORLD);
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                A[i * n + j] = recvbuf[j * n + i];
    }

    if (rank == 0) {
        double dur = t1 - t0;
        double ops = (double)n * n * n / 3.0;
        printf("Computation time: %.3f s\n", dur);
        printf("Performance: %.3f GFLOPS\n", ops / dur / 1e9);
    }

    if (printResults && rank == 0)
        print_results(A, "CholeskyL");

    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        if (!valid) { MPI_Finalize(); return 1; }
    }

    MPI_Finalize();
    return 0;
}
