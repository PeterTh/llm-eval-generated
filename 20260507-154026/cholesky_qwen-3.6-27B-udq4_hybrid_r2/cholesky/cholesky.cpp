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

/* ------------------------------------------------------------------ */
/*  Global state                                                       */
/* ------------------------------------------------------------------ */
static int           mpi_rank       = 0;
static int           mpi_size       = 1;
static cublasHandle_t cublas_handle = nullptr;
static cusolverDnHandle_t cusolver_handle = nullptr;

/*
 * local_A is row-major: local_A[local_row * n + col]
 * Each rank owns rows [local_offset, local_offset+local_rows).
 */
static size_t        local_rows    = 0;
static size_t        local_offset  = 0;
static std::vector<double> local_A;

/* ------------------------------------------------------------------ */
/*  Error-checking helpers                                             */
/* ------------------------------------------------------------------ */
static void check_cuda(cudaError_t e, const char* m) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[rank %d] CUDA: %s (%s)\n", mpi_rank, m,
                cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CHK_CU(e) check_cuda(e, #e)

static void check_cublas(cublasStatus_t s, const char* m) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[rank %d] cuBLAS: %s (%d)\n", mpi_rank, m,
                (int)s);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CHK_CB(s) check_cublas(s, #s)

static void check_cusolver(cusolverStatus_t s, const char* m) {
    if (s != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[rank %d] cuSOLVER: %s (%d)\n", mpi_rank, m,
                (int)s);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CHK_CS(s) check_cusolver(s, #s)

/* ------------------------------------------------------------------ */
/*  Row-distribution: rank r owns rows [off, off+cnt)                  */
/* ------------------------------------------------------------------ */
static void row_info(size_t n, int r, size_t& off, size_t& cnt) {
    size_t base = n / mpi_size, rem = n % mpi_size;
    off = base * r + std::min((size_t)r, rem);
    cnt = base + (r < (int)rem ? 1 : 0);
}

/* ------------------------------------------------------------------ */
/*  Matrix generation  (row-major, identical to original)              */
/* ------------------------------------------------------------------ */
static void generateMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    std::fill(A.begin(), A.end(), 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += B[i * n + k] * B[j * n + k];
            A[i * n + j] = s;
        }
    }
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += (double)n;
}

/* ------------------------------------------------------------------ */
/*  Blocked Cholesky  (MPI row-dist  +  CUDA cuBLAS/cuSOLVER)          */
/*
 * Row-major local_A[lr * n + c].
 * GPU: column-major d_A[c * local_rows + lr].
 *
 * For each block k = 0, nb, 2nb, ...
 *   1. Gather A(k:k+an, k:k+an) to diagonal rank dr
 *   2. POTRF on dr (cuSOLVER GPU)
 *   3. Broadcast L(k:k+an, k:k+an)
 *   4. TRSM: A(k+an:,k) = A(k+an:,k) * inv(L(k,k)^T)  (cuBLAS GPU)
 *   5. Allgather block column A(k+an:,k)
 *   6. GEMM: A(k+an:,k+an:) -= A(k+an:,k) * A(k+an:,k)^T  (cuBLAS GPU)
 * ------------------------------------------------------------------ */
static bool blockedCholesky(size_t n) {
    if (local_rows == 0) return true;

    const size_t nb = 128;

    /* GPU buffer: column-major copy of local_A */
    /* d_A[c * local_rows + lr] for c in [0,n), lr in [0,local_rows) */
    double *d_A = nullptr;
    CHK_CU(cudaMalloc(&d_A, n * local_rows * sizeof(double)));
    {
        std::vector<double> tmp(n * local_rows);
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t lr = 0; lr < local_rows; ++lr)
            for (size_t c = 0; c < n; ++c)
                tmp[c * local_rows + lr] = local_A[lr * n + c];
        CHK_CU(cudaMemcpy(d_A, tmp.data(),
                          n * local_rows * sizeof(double),
                          cudaMemcpyHostToDevice));
    }

    /* GPU buffers for block ops */
    double *d_diag = nullptr;
    CHK_CU(cudaMalloc(&d_diag, nb * nb * sizeof(double)));

    int lwork = 0;
    CHK_CS(cusolverDnDpotrf_bufferSize(cusolver_handle,
            CUBLAS_FILL_MODE_LOWER, (int)nb, d_diag, (int)nb, &lwork));
    double *d_work = nullptr;
    if (lwork > 0) CHK_CU(cudaMalloc(&d_work, lwork * sizeof(double)));

    bool ok = true;

    for (size_t k = 0; k < n && ok; k += nb) {
        size_t an = std::min(nb, n - k);

        /* ---- find diagonal rank ---- */
        int dr = 0;
        for (int r = 0; r < mpi_size; ++r) {
            size_t ro, rc; row_info(n, r, ro, rc);
            if (ro <= k && ro + rc > k) { dr = r; break; }
        }

        /* ======== 1. Gather diagonal block to dr ======== */
        /* Diagonal block: rows [k, k+an), cols [k, k+an) */
        /* Each rank contributes rows in [k, k+an) intersect [local_offset, local_offset+local_rows) */
        size_t ms = std::max(local_offset, k);
        size_t me = std::min(local_offset + local_rows, k + an);
        size_t mc = (me > ms) ? (me - ms) : 0;

        /* Extract from GPU col-major: d_A[(k+c)*local_rows + lr_start]
         * gives column c, rows ms..me-1. Col-major [an x mc]. */
        std::vector<double> sbuf(mc * an);
        if (mc > 0) {
            size_t lr_start = ms - local_offset;
            for (size_t c = 0; c < an; ++c)
                CHK_CU(cudaMemcpy(sbuf.data() + c * mc,
                                  d_A + (k + c) * local_rows + lr_start,
                                  mc * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> gc(mpi_size), gd(mpi_size);
        size_t disp = 0;
        for (int r = 0; r < mpi_size; ++r) {
            size_t ro, rc; row_info(n, r, ro, rc);
            size_t rs = std::max(ro, k), re = std::min(ro + rc, k + an);
            size_t c = (re > rs) ? (re - rs) : 0;
            gc[r] = (int)(c * an);
            gd[r] = (int)disp;
            disp += gc[r];
        }

        std::vector<double> rdiag(an * an);
        MPI_Gatherv(sbuf.data(), (int)(mc * an), MPI_DOUBLE,
                    rdiag.data(), gc.data(), gd.data(), MPI_DOUBLE,
                    dr, MPI_COMM_WORLD);

        /* ======== 2. POTRF on diagonal rank (GPU) ======== */
        if (mpi_rank == dr) {
            CHK_CU(cudaMemcpy(d_diag, rdiag.data(),
                              an * an * sizeof(double),
                              cudaMemcpyHostToDevice));
            int *d_info = nullptr;
            CHK_CU(cudaMalloc(&d_info, sizeof(int)));
            CHK_CS(cusolverDnDpotrf(cusolver_handle,
                    CUBLAS_FILL_MODE_LOWER, (int)an, d_diag, (int)an,
                    d_work, lwork, d_info));
            CHK_CU(cudaMemcpy(rdiag.data(), d_diag,
                              an * an * sizeof(double),
                              cudaMemcpyDeviceToHost));
            int h_info = 0;
            CHK_CU(cudaMemcpy(&h_info, d_info, sizeof(int),
                              cudaMemcpyDeviceToHost));
            CHK_CU(cudaFree(d_info));
            if (h_info > 0) {
                fprintf(stderr, "[rank %d] POTRF failed at %d\n", mpi_rank, h_info);
                ok = false;
            }
            fprintf(stderr, "[rank %d] POTRF k=%zu: rdiag[0]=%.6f\n", mpi_rank, k, rdiag[0]);
        }
        if (!ok) break;

        /* ======== 3. Broadcast diagonal block ======== */
        MPI_Bcast(rdiag.data(), (int)(an * an), MPI_DOUBLE,
                  dr, MPI_COMM_WORLD);

        /* Write back to GPU */
        if (mc > 0) {
            size_t lr_start = ms - local_offset;
            for (size_t c = 0; c < an; ++c)
                CHK_CU(cudaMemcpy(d_A + (k + c) * local_rows + lr_start,
                                  rdiag.data() + c * an,
                                  mc * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
        /* Debug: verify d_A[0] after write-back */
        {
            double dval;
            CHK_CU(cudaMemcpy(&dval, d_A, sizeof(double), cudaMemcpyDeviceToHost));
            fprintf(stderr, "[rank %d] After write-back k=%zu: d_A[0]=%.6f, rdiag[0]=%.6f\n",
                    mpi_rank, k, dval, rdiag[0]);
        }

        /* ======== 4. TRSM: rows below diagonal ======== */
        /* A(tr:,k) = A(tr:,k) * inv(L(k,k)^T)
         * Only ranks with rows > k+an participate */
        size_t tr_start = std::max(local_offset, k + an) - local_offset;
        size_t tr_count = (local_offset + local_rows) -
                           std::max(local_offset, k + an);
        if (tr_count > 0) {
            CHK_CU(cudaMemcpy(d_diag, rdiag.data(),
                              an * an * sizeof(double),
                              cudaMemcpyHostToDevice));

            /* GPU col-major: C[an x tr_count] at d_A[k*local_rows + tr_start]
             *                A[an x an] at d_diag
             * cublasDtrsm(SIDE_RIGHT, LOWER, TRANS, NON_UNIT,
             *             m=tr_count, n=an, alpha, A, lda=an, C, ldc=local_rows)
             * C := alpha * C * inv(op(A))
             * op(A) = A^T (since TRANS)
             * C(tr_count x an) := C * inv(A^T)
             * In col-major: C is [an x tr_count], A is [an x an]
             */
            double alpha = 1.0;
            CHK_CB(cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                (int)tr_count, (int)an, &alpha,
                d_diag, (int)an,
                d_A + k * local_rows + tr_start, (int)local_rows));
        }

        /* ======== 5. Allgather block column + GEMM ======== */
        if (k + an < n) {
            size_t bcr = n - (k + an);

            /* Extract block column from GPU: rows [k+an, n), cols [k, k+an) */
            size_t mbs = std::max(local_offset, k + an);
            size_t mbe = std::min(local_offset + local_rows, n);
            size_t mbc = (mbe > mbs) ? (mbe - mbs) : 0;

            std::vector<double> bsend(mbc * an);
            if (mbc > 0) {
                size_t lr_start = mbs - local_offset;
                for (size_t c = 0; c < an; ++c)
                    CHK_CU(cudaMemcpy(bsend.data() + c * mbc,
                        d_A + (k + c) * local_rows + lr_start,
                        mbc * sizeof(double), cudaMemcpyDeviceToHost));
            }

            std::vector<int> bc(mpi_size), bd(mpi_size);
            disp = 0;
            for (int r = 0; r < mpi_size; ++r) {
                size_t ro, rc; row_info(n, r, ro, rc);
                size_t rs = std::max(ro, k + an), re = std::min(ro + rc, n);
                size_t c = (re > rs) ? (re - rs) : 0;
                bc[r] = (int)(c * an);
                bd[r] = (int)disp;
                disp += bc[r];
            }

            std::vector<double> fbc(bcr * an);
            MPI_Allgatherv(bsend.data(), (int)(mbc * an), MPI_DOUBLE,
                           fbc.data(), bc.data(), bd.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);

            /* GEMM on GPU: C -= L_tr * L_bc^T
             * C is tr_count x gcol (row x col)
             * L_tr is tr_count x an (TRSM result, in d_A)
             * L_bc is bcr x an (full block column)
             *
             * We need: C(tr_count x gcol) -= L_tr(tr_count x an) * L_bc_part^T(an x gcol)
             * where L_bc_part is rows [lrf, lrf+gcol) of L_bc (gcol x an).
             *
             * cublasDgemm(OPA, OPB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
             * C(m x n) = alpha * op(A) * op(B) + beta * C(m x n)
             *
             * m = tr_count, n = gcol, k = an
             * op(A) = L_tr (tr_count x an)
             * op(B) = L_bc_part^T (an x gcol)
             *
             * For op(A) = L_tr (tr_count x an):
             *   OPA=T: A is an x tr_count, lda >= an
             *   L_tr in d_A is [an x tr_count] at d_A[k*local_rows + tr_start], lda=local_rows
             *   local_rows >= an? Not always. Use temp buffer.
             *
             * For op(B) = L_bc_part^T (an x gcol):
             *   OPB=N: B is an x gcol, ldb >= an
             *   L_bc_part^T in col-major is [gcol x an] with ldb=gcol.
             *   We need B as [an x gcol] with ldb >= an.
             *   So we need to transpose L_bc_part^T.
             *   L_bc_part^T is an x gcol. In col-major it's [gcol x an].
             *   So B = [gcol x an] col-major, OPB=N gives op(B) = B = [gcol x an] matrix.
             *   But we need op(B) = [an x gcol].
             *   So OPB=T: op(B) = B^T = [an x gcol] ✓
             *   B is [gcol x an] col-major, ldb >= gcol.
             *
             * So:
             * cublasDgemm(OPA=T, OPB=T, m=tr_count, n=gcol, k=an, alpha=-1,
             *             A[an x tr_count] lda=an,
             *             B[gcol x an] ldb=gcol,
             *             beta=1, C[gcol x tr_count] ldc=gcol)
             *
             * But C in d_A is at d_A[(k+an)*local_rows+tr_start] with ldc=local_rows.
             * C is [gcol x tr_count] in d_A. ldc=local_rows.
             * We need ldc >= m = tr_count. local_rows >= tr_count ✓.
             * But we also need the C buffer to be [gcol x tr_count] with ldc = gcol for the formula.
             * Wait no, ldc can be anything >= m.
             *
             * Actually, let me re-examine:
             * C(m x n) = C(tr_count x gcol)
             * In col-major, C is [n x m] = [gcol x tr_count] with ldc >= m = tr_count.
             * In d_A: C is at d_A[(k+an)*local_rows + tr_start].
             * d_A is [n x local_rows] col-major.
             * C submatrix: columns (k+an) to (k+an+gcol-1), rows tr_start to (tr_start+tr_count-1).
             * In d_A: C[r + c*local_rows] for r in [tr_start, tr_start+tr_count), c in [k+an, k+an+gcol).
             * This is [tr_count x gcol] with leading dimension local_rows.
             * For cublas: C is [gcol x tr_count] with ldc = local_rows.
             * We need ldc >= m = tr_count. local_rows >= tr_count ✓.
             * But is the layout [gcol x tr_count]? Let me check.
             * d_A[(k+an)*local_rows + tr_start] is the first element.
             * Moving along the leading dimension (local_rows) gives us the next row of C.
             * Moving along the base gives us the next column of C.
             * So C[r + c*local_rows] where r is the row index and c is the column index.
             * In cublas column-major: C[c*ldc + r] where r in [0,m), c in [0,n).
             * So m = tr_count, n = gcol, ldc = local_rows. ✓
             *
             * Now for A:
             * L_tr is tr_count x an. In col-major: [an x tr_count].
             * In d_A: L_tr is at d_A[k*local_rows + tr_start].
             * L_tr[r + c*local_rows] for r in [tr_start, tr_start+tr_count), c in [k, k+an).
             * This is [tr_count x an] with leading dimension local_rows.
             * For cublas: A is [an x tr_count] with lda = local_rows.
             * OPA=T: op(A) = A^T = [tr_count x an] = L_tr ✓
             * We need lda >= k = an. local_rows >= an? Not always.
             *
             * Hmm, if local_rows < an, we have a problem. Let me use temp buffers.
             */

            size_t gcol = n - (k + an);
            if (tr_count > 0 && gcol > 0) {
                size_t lrf = (local_offset >= k + an)
                               ? (local_offset - (k + an)) : 0;

                /* Prepare L_tr: [an x tr_count] col-major, lda=an */
                std::vector<double> tmp_tr(an * tr_count);
                #pragma omp parallel for collapse(2) schedule(static)
                for (size_t r = 0; r < tr_count; ++r)
                    for (size_t c = 0; c < an; ++c)
                        tmp_tr[c * tr_count + r] = local_A[(tr_start + r) * n + (k + c)];

                double *d_Ltr = nullptr;
                CHK_CU(cudaMalloc(&d_Ltr, an * tr_count * sizeof(double)));
                CHK_CU(cudaMemcpy(d_Ltr, tmp_tr.data(),
                                  an * tr_count * sizeof(double),
                                  cudaMemcpyHostToDevice));

                /* Prepare L_bc_part: [an x gcol] col-major, lda=an */
                /* fbc is row-major [bcr x an]: fbc[r*an+c] */
                /* L_bc_part is rows lrf..lrf+gcol-1 of fbc (gcol x an) */
                /* L_bc_part^T is an x gcol */
                /* In col-major, L_bc_part^T is [gcol x an] with ldb=gcol */
                std::vector<double> tmp_bc(gcol * an);
                #pragma omp parallel for collapse(2) schedule(static)
                for (size_t r = 0; r < gcol; ++r)
                    for (size_t c = 0; c < an; ++c)
                        tmp_bc[c * gcol + r] = fbc[(lrf + r) * an + c];

                double *d_Lbc = nullptr;
                CHK_CU(cudaMalloc(&d_Lbc, gcol * an * sizeof(double)));
                CHK_CU(cudaMemcpy(d_Lbc, tmp_bc.data(),
                                  gcol * an * sizeof(double),
                                  cudaMemcpyHostToDevice));

                /* Prepare C: [tr_count x gcol] col-major, ldc=tr_count */
                std::vector<double> tmp_C(tr_count * gcol);
                #pragma omp parallel for collapse(2) schedule(static)
                for (size_t r = 0; r < tr_count; ++r)
                    for (size_t c = 0; c < gcol; ++c)
                        tmp_C[c * tr_count + r] = local_A[(tr_start + r) * n + (k + an + c)];

                double *d_C = nullptr;
                CHK_CU(cudaMalloc(&d_C, tr_count * gcol * sizeof(double)));
                CHK_CU(cudaMemcpy(d_C, tmp_C.data(),
                                  tr_count * gcol * sizeof(double),
                                  cudaMemcpyHostToDevice));

                /* C(tr_count x gcol) -= L_tr(tr_count x an) * L_bc_part^T(an x gcol)
                 *
                 * cublasDgemm(OPA=T, OPB=N, m=tr_count, n=gcol, k=an, alpha=-1,
                 *             A[an x tr_count] lda=an,
                 *             B[an x gcol] ldb=?,
                 *             beta=1, C[tr_count x gcol] ldc=tr_count)
                 *
                 * op(A) = A^T = L_tr (tr_count x an) ✓
                 * op(B) = B = L_bc_part^T (an x gcol)
                 * B is an x gcol, ldb >= an.
                 * But d_Lbc is [gcol x an] with ldb=gcol.
                 * We need B as [an x gcol] with ldb >= an.
                 *
                 * Hmm, d_Lbc stores L_bc_part^T in col-major as [gcol x an].
                 * If I use OPB=T: op(B) = B^T. B is [gcol x an], so B^T is [an x gcol].
                 * That's exactly L_bc_part^T! ✓
                 *
                 * cublasDgemm(OPA=T, OPB=T, m=tr_count, n=gcol, k=an, alpha=-1,
                 *             A[an x tr_count] lda=an,
                 *             B[gcol x an] ldb=gcol,
                 *             beta=1, C[tr_count x gcol] ldc=tr_count)
                 *
                 * op(A) = A^T = L_tr (tr_count x an) ✓
                 * op(B) = B^T = L_bc_part^T (an x gcol) ✓
                 * C = -1 * L_tr * L_bc_part^T + C ✓
                 */
                double alpha = -1.0, beta = 1.0;
                CHK_CB(cublasDgemm(cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_T,
                    (int)tr_count, (int)gcol, (int)an,
                    &alpha,
                    d_Ltr, (int)an,
                    d_Lbc, (int)gcol,
                    &beta,
                    d_C, (int)tr_count));

                /* Copy C back to local_A and d_A */
                CHK_CU(cudaMemcpy(tmp_C.data(), d_C,
                                  tr_count * gcol * sizeof(double),
                                  cudaMemcpyDeviceToHost));
                #pragma omp parallel for collapse(2) schedule(static)
                for (size_t r = 0; r < tr_count; ++r)
                    for (size_t c = 0; c < gcol; ++c)
                        local_A[(tr_start + r) * n + (k + an + c)] = tmp_C[c * tr_count + r];

                /* Update d_A for future blocks: copy C from d_C to d_A */
                /* d_C is [tr_count x gcol] col-major with ldc=tr_count */
                /* d_A is [n x local_rows] col-major with ldc=local_rows */
                /* Copy using cudaMemcpy2D: gcol rows, each of tr_count doubles */
                CHK_CU(cudaMemcpy2D(
                    d_A + (k + an) * local_rows + tr_start,
                    local_rows * sizeof(double),
                    d_C,
                    tr_count * sizeof(double),
                    tr_count * sizeof(double),
                    (size_t)gcol,
                    cudaMemcpyDeviceToDevice));

                CHK_CU(cudaFree(d_Lbc));
                CHK_CU(cudaFree(d_Ltr));
                CHK_CU(cudaFree(d_C));
            }
        }
    }

    /* Copy back: GPU col-major -> CPU row-major */
    {
        std::vector<double> tmp(n * local_rows);
        CHK_CU(cudaMemcpy(tmp.data(), d_A,
                          n * local_rows * sizeof(double),
                          cudaMemcpyDeviceToHost));
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t lr = 0; lr < local_rows; ++lr)
            for (size_t c = 0; c < n; ++c)
                local_A[lr * n + c] = tmp[c * local_rows + lr];
    }

    /* Zero out upper triangle */
    for (size_t lr = 0; lr < local_rows; ++lr) {
        size_t global_row = local_offset + lr;
        for (size_t c = global_row + 1; c < n; ++c)
            local_A[lr * n + c] = 0.0;
    }

    CHK_CU(cudaFree(d_A));
    CHK_CU(cudaFree(d_diag));
    if (d_work) CHK_CU(cudaFree(d_work));

    return ok;
}

/* ------------------------------------------------------------------ */
/*  Validation  (OpenMP-parallelised matrix multiply)                  */
/* ------------------------------------------------------------------ */
static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& A_orig,
                             size_t n) {
    std::vector<double> recon(n * n);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += L[i * n + k] * L[j * n + k];
            recon[i * n + j] = s;
        }
    }

    double maxErr = 0.0, relErr = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        double e  = fabs(recon[i] - A_orig[i]);
        maxErr    = std::max(maxErr, e);
        double r  = e / (fabs(A_orig[i]) + 1e-10);
        relErr    = std::max(relErr, r);
    }
    printf("Max absolute error: %.10e\n", maxErr);
    printf("Max relative error: %.10e\n", relErr);
    if (relErr > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    cublasCreate(&cublas_handle);
    cusolverDnCreate(&cusolver_handle);

    size_t n = 512;
    bool   validate     = false;
    bool   printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n = (size_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (strcmp(argv[i], "-r") == 0)
            printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!mpi_rank) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Matrix size (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
            }
            cublasDestroy(cublas_handle);
            cusolverDnDestroy(cusolver_handle);
            MPI_Finalize(); return 0;
        } else {
            if (!mpi_rank) printf("Unknown option: %s\n", argv[i]);
            cublasDestroy(cublas_handle);
            cusolverDnDestroy(cusolver_handle);
            MPI_Finalize(); return 1;
        }
    }

    if (!mpi_rank) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    /* ---- generate matrix (rank 0) ---- */
    std::vector<double> A(n * n), A_orig;
    if (!mpi_rank) {
        printf("Generating positive definite matrix...\n");
        generateMatrix(A, n);
        if (validate) A_orig = A;
    }

    /* ---- distribute rows (row-major Scatterv) ---- */
    row_info(n, mpi_rank, local_offset, local_rows);
    if (local_rows > 0) local_A.resize(n * local_rows);

    std::vector<int> sc(mpi_size), sd(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        size_t ro, rc; row_info(n, r, ro, rc);
        sc[r] = (int)(rc * n);
        sd[r] = (int)(ro * n);
    }
    MPI_Scatterv(A.data(), sc.data(), sd.data(), MPI_DOUBLE,
                 local_A.data(), (int)(n * local_rows), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* ---- blocked Cholesky (MPI + CUDA) ---- */
    if (!mpi_rank) printf("Computing Cholesky decomposition...\n");

    auto t0 = std::chrono::high_resolution_clock::now();
    bool success = blockedCholesky(n);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (!success) {
        if (!mpi_rank) printf("Cholesky decomposition failed\n");
        cublasDestroy(cublas_handle);
        cusolverDnDestroy(cusolver_handle);
        MPI_Finalize(); return 1;
    }

    if (!mpi_rank) {
        printf("Computation time: %ld ms\n", (long)ms);
        double gflops = (double)n * n * n / 3.0 / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    /* ---- gather results (row-major Gatherv) ---- */
    std::vector<double> result(n * n);
    MPI_Gatherv(local_A.data(), (int)(n * local_rows), MPI_DOUBLE,
                result.data(), sc.data(), sd.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (!mpi_rank) {
        if (printResults) print_results(result, "CholeskyL");

        if (validate) {
            printf("Validating result...\n");
            bool v = validateCholesky(result, A_orig, n);
            printf("Validation: %s\n", v ? "PASSED" : "FAILED");
            if (!v) {
                cublasDestroy(cublas_handle);
                cusolverDnDestroy(cusolver_handle);
                MPI_Finalize(); return 1;
            }
        }
    }

    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);
    MPI_Finalize();
    return 0;
}
