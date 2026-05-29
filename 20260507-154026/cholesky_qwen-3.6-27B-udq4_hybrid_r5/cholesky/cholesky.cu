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

static constexpr int BLOCK_SIZE = 128;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = call;                                                   \
        if (_e != cudaSuccess) {                                                 \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(_e));                                      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t _s = call;                                                \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                       \
            printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, _s);       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

#define CUSOLVER_CHECK(call)                                                     \
    do {                                                                         \
        cusolverStatus_t _s = call;                                              \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                     \
            printf("cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, _s);     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

#define MPI_CHECK(call)                                                          \
    do {                                                                         \
        int _e = call;                                                           \
        if (_e != MPI_SUCCESS) {                                                 \
            char _b[MPI_MAX_ERROR_STRING];                                       \
            int _l;                                                              \
            MPI_Error_string(_e, _b, &_l);                                       \
            printf("MPI error at %s:%d: %s\n", __FILE__, __LINE__, _b);          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Matrix generation – OpenMP-parallelised                            */
/* ------------------------------------------------------------------ */
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        unsigned int s = seed + static_cast<unsigned int>(i);
        B[i] = (rand_r(&s) / static_cast<double>(RAND_MAX)) - 0.5;
    }

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

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += static_cast<double>(n);
}

/* ------------------------------------------------------------------ */
/*  Blocked Cholesky – 1-D row-block MPI + cuBLAS/cuSOLVER on GPU     */
/*  GPU stores data in column-major: d_A[i + j*ld] = A[i][j]          */
/* ------------------------------------------------------------------ */
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, nranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    // ---- row distribution ----
    const size_t rpr  = n / nranks;
    const size_t rem  = n % nranks;
    const size_t my_s = rank * rpr + std::min(static_cast<size_t>(rank), rem);
    const size_t my_e = (rank + 1) * rpr + std::min(static_cast<size_t>(rank + 1), rem);
    const size_t lrows = my_e - my_s;

    // Pre-compute rank boundaries
    std::vector<size_t> rs(nranks), re(nranks);
    for (int r = 0; r < nranks; ++r) {
        rs[r] = r * rpr + std::min(static_cast<size_t>(r), rem);
        re[r] = (r + 1) * rpr + std::min(static_cast<size_t>(r + 1), rem);
    }

    // ---- cuBLAS + cuSOLVER handles + stream ----
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // ---- pinned host buffers (row-major for MPI) ----
    double* h_panel_rm = nullptr;  // [nb x nb]
    double* h_L21_rm   = nullptr;  // [remaining x nb]

    if (lrows > 0) {
        CUDA_CHECK(cudaMallocHost(&h_panel_rm, BLOCK_SIZE * BLOCK_SIZE * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_L21_rm,   n * BLOCK_SIZE * sizeof(double)));
    }

    // ---- device memory (column-major) ----
    // d_A[i + j*lrows] = A[my_s+i][j]   for i in [0,lrows), j in [0,n)
    double *d_A = nullptr, *d_panel = nullptr, *d_L21 = nullptr, *d_work = nullptr;
    int* d_info = nullptr;
    int max_work = 0;

    if (lrows > 0) {
        CUDA_CHECK(cudaMalloc(&d_A,     lrows * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel, BLOCK_SIZE * BLOCK_SIZE * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_L21,   n * BLOCK_SIZE * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info,  sizeof(int)));

        // Convert local rows from row-major → column-major and transfer
        std::vector<double> lA_cm(lrows * n);
        for (size_t i = 0; i < lrows; ++i)
            for (size_t j = 0; j < n; ++j)
                lA_cm[i + j * lrows] = A[(my_s + i) * n + j];
        CUDA_CHECK(cudaMemcpy(d_A, lA_cm.data(), lrows * n * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    const double alpha_one    =  1.0;
    const double alpha_minus1 = -1.0;

    // ---- send / recv count arrays ----
    std::vector<int> sendcounts(nranks), displs(nranks);

    // ================================================================
    //  Blocked Cholesky loop
    // ================================================================
    for (size_t k = 0; k < n; k += BLOCK_SIZE) {
        // Ensure previous iteration's GPU work is done
        if (lrows > 0) CUDA_CHECK(cudaStreamSynchronize(stream));
        const int nb = static_cast<int>(std::min(static_cast<size_t>(BLOCK_SIZE), n - k));
        const int remaining = static_cast<int>(n - k - nb);

        // ----------------------------------------------------------------
        //  1. Gather diagonal panel A(k:k+nb, k:k+nb)
        // ----------------------------------------------------------------
        const size_t ps = std::max(k, my_s);
        const size_t pe = std::min(k + static_cast<size_t>(nb), my_e);
        const int mpr = (ps < pe) ? static_cast<int>(pe - ps) : 0;
        const int lprs = (ps >= my_s) ? static_cast<int>(ps - my_s) : 0;

        // Compute sendcounts / displs
        {
            size_t goff = 0;
            for (int r = 0; r < nranks; ++r) {
                const size_t rps = std::max(k, rs[r]);
                const size_t rpe = std::min(k + static_cast<size_t>(nb), re[r]);
                const int rnr = (rps < rpe) ? static_cast<int>(rpe - rps) : 0;
                sendcounts[r] = rnr * nb;
                displs[r]     = static_cast<int>(goff * nb);
                goff += rnr;
            }
        }

        // Copy my panel rows from GPU to host (row-major for MPI)
        // GPU column-major: d_A[i + j*ld] = A[my_s+i][j]
        // Panel: rows [lprs, lprs+mpr), cols [k, k+nb)
        // Need h_panel_rm[i*nb + j] = d_A[(lprs+i) + (k+j)*lrows]
        if (mpr > 0 && lrows > 0) {
            // Copy columns to temp, then transpose to row-major
            std::vector<double> tmp(mpr * nb);
            for (int j = 0; j < nb; ++j) {
                CUDA_CHECK(cudaMemcpyAsync(
                    tmp.data() + j * mpr,
                    d_A + lprs + (k + j) * lrows,
                    mpr * sizeof(double), cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < mpr; ++i)
                for (int j = 0; j < nb; ++j)
                    h_panel_rm[i * nb + j] = tmp[j * mpr + i];
        }

        // All-gather the full panel
        MPI_CHECK(MPI_Allgatherv(
            h_panel_rm, mpr * nb, MPI_DOUBLE,
            h_panel_rm, sendcounts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD));

        // ----------------------------------------------------------------
        //  2. POTRF on the panel (GPU) via cuSOLVER
        // ----------------------------------------------------------------
        if (lrows > 0) {
            // Copy panel to GPU (column-major): d_panel[i + j*nb] = h_panel_rm[i*nb + j]
            // h_panel_rm is row-major [nb x nb], need to transpose to column-major
            std::vector<double> tmp3(nb * nb);
            for (int i = 0; i < nb; ++i)
                for (int j = 0; j < nb; ++j)
                    tmp3[i + j * nb] = h_panel_rm[i * nb + j];
            CUDA_CHECK(cudaMemcpyAsync(d_panel, tmp3.data(),
                                       nb * nb * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                                                        nb, d_panel, nb, &lwork));
            if (lwork > max_work) {
                CUDA_CHECK(cudaFreeAsync(d_work, stream));
                CUDA_CHECK(cudaMallocAsync(&d_work, lwork * sizeof(double), stream));
                max_work = lwork;
            }
            CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER,
                                             nb, d_panel, nb, d_work, lwork, d_info));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int info = 0;
            CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (info > 0) {
                printf("POTRF failed: matrix is not positive definite at factor %d\n", info);
                return false;
            }

            // Copy factorised panel back to d_A
            // d_A[lprs+i + (k+j)*lrows] = d_panel[i + j*nb]
            if (mpr > 0) {
                for (int j = 0; j < nb; ++j) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        d_A + lprs + (k + j) * lrows,
                        d_panel + j * nb,
                        mpr * sizeof(double), cudaMemcpyDeviceToDevice, stream));
                }
            }
        }

        // ----------------------------------------------------------------
        //  3. TRSM  –  L21 = A21 * inv(L11^T)   (GPU)
        // ----------------------------------------------------------------
        if (remaining > 0 && lrows > 0) {
            const int a21s = (k + nb > my_s) ? static_cast<int>(k + nb - my_s) : 0;
            const int a21r = static_cast<int>(lrows) - a21s;
            if (a21r > 0) {
                CUBLAS_CHECK(cublasDtrsm(
                    handle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    a21r, nb,
                    &alpha_one,
                    d_panel, nb,
                    d_A + k * lrows + a21s, lrows));
            }
        }

        // ----------------------------------------------------------------
        //  4. Gather full L21 + GEMM update  (GPU)
        //     A22 = A22 - L21_local * L21_full^T
        // ----------------------------------------------------------------
        if (remaining > 0 && lrows > 0) {
            const size_t ls  = std::max(k + static_cast<size_t>(nb), my_s);
            const size_t le  = std::min(n, my_e);
            const int mlr = (ls < le) ? static_cast<int>(le - ls) : 0;
            const int llrs = (ls >= my_s) ? static_cast<int>(ls - my_s) : 0;

            // Compute sendcounts / displs for L21
            {
                size_t goff = 0;
                for (int r = 0; r < nranks; ++r) {
                    const size_t rps = std::max(k + static_cast<size_t>(nb), rs[r]);
                    const size_t rpe = std::min(n, re[r]);
                    const int rnr = (rps < rpe) ? static_cast<int>(rpe - rps) : 0;
                    sendcounts[r] = rnr * nb;
                    displs[r]     = static_cast<int>(goff * nb);
                    goff += rnr;
                }
            }

            // Copy L21 contribution GPU → host (row-major)
            if (mlr > 0) {
                std::vector<double> tmp2(mlr * nb);
                for (int j = 0; j < nb; ++j) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        tmp2.data() + j * mlr,
                        d_A + llrs + (k + j) * lrows,
                        mlr * sizeof(double), cudaMemcpyDeviceToHost, stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int i = 0; i < mlr; ++i)
                    for (int j = 0; j < nb; ++j)
                        h_L21_rm[i * nb + j] = tmp2[j * mlr + i];
            }

            // All-gather full L21
            MPI_CHECK(MPI_Allgatherv(
                h_L21_rm, mlr * nb, MPI_DOUBLE,
                h_L21_rm, sendcounts.data(), displs.data(), MPI_DOUBLE,
                MPI_COMM_WORLD));

            // Copy full L21 to GPU (column-major)
            // h_L21_rm is row-major [remaining x nb], transpose to column-major
            std::vector<double> tmp4(remaining * nb);
            for (int i = 0; i < remaining; ++i)
                for (int j = 0; j < nb; ++j)
                    tmp4[i + j * remaining] = h_L21_rm[i * nb + j];
            CUDA_CHECK(cudaMemcpyAsync(d_L21, tmp4.data(),
                                       remaining * nb * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

            // GEMM:  C = C - A * B^T
            const int a22s = (k + nb > my_s) ? static_cast<int>(k + nb - my_s) : 0;
            const int a22r = static_cast<int>(lrows) - a22s;
            if (a22r > 0) {
                CUBLAS_CHECK(cublasDgemm(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    a22r, remaining, nb,
                    &alpha_minus1,
                    d_A + k * lrows + a22s,     lrows,
                    d_L21,                       remaining,
                    &alpha_one,
                    d_A + (k + nb) * lrows + a22s, lrows));
            }
        }
    }

    // ================================================================
    //  Gather result to rank 0
    // ================================================================
    if (lrows > 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy full local matrix from GPU → host (column-major)
        std::vector<double> lA_cm(lrows * n);
        CUDA_CHECK(cudaMemcpy(lA_cm.data(), d_A,
                              lrows * n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        // Convert column-major → row-major, zero upper triangle
        std::vector<double> lA_rm(lrows * n);
        for (size_t i = 0; i < lrows; ++i) {
            const size_t gr = my_s + i;
            for (size_t j = 0; j < n; ++j)
                lA_rm[i * n + j] = (j <= gr)
                    ? lA_cm[i + j * lrows]
                    : 0.0;
        }

        // Gatherv to rank 0
        std::vector<int> rc(nranks), gd(nranks);
        for (int r = 0; r < nranks; ++r) {
            const size_t nr = re[r] - rs[r];
            rc[r] = static_cast<int>(nr * n);
            gd[r] = static_cast<int>(rs[r] * n);
        }

        if (rank == 0) {
            std::vector<double> fA(n * n);
            MPI_CHECK(MPI_Gatherv(
                lA_rm.data(), static_cast<int>(lrows * n), MPI_DOUBLE,
                fA.data(), rc.data(), gd.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD));
            A = std::move(fA);
        } else {
            MPI_CHECK(MPI_Gatherv(
                lA_rm.data(), static_cast<int>(lrows * n), MPI_DOUBLE,
                nullptr, nullptr, nullptr, MPI_DOUBLE,
                0, MPI_COMM_WORLD));
        }
    } else {
        std::vector<int> rc(nranks), gd(nranks);
        for (int r = 0; r < nranks; ++r) {
            const size_t nr = re[r] - rs[r];
            rc[r] = static_cast<int>(nr * n);
            gd[r] = static_cast<int>(rs[r] * n);
        }

        if (rank == 0) {
            std::vector<double> fA(n * n);
            MPI_CHECK(MPI_Gatherv(
                nullptr, 0, MPI_DOUBLE,
                fA.data(), rc.data(), gd.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD));
            A = std::move(fA);
        } else {
            MPI_CHECK(MPI_Gatherv(
                nullptr, 0, MPI_DOUBLE,
                nullptr, nullptr, nullptr, MPI_DOUBLE,
                0, MPI_COMM_WORLD));
        }
    }

    // ---- cleanup ----
    if (lrows > 0) {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_panel));
        CUDA_CHECK(cudaFree(d_L21));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFreeHost(h_panel_rm));
        CUDA_CHECK(cudaFreeHost(h_L21_rm));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    CUSOLVER_CHECK(cusolverDnDestroy(solver));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return true;
}

/* ------------------------------------------------------------------ */
/*  Validation – OpenMP-parallelised L·L^T reconstruction             */
/* ------------------------------------------------------------------ */
bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig,
                      const size_t n) {
    std::vector<double> reconstructed(n * n);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = s;
        }
    }

    double maxErr = 0.0, relErr = 0.0;
    #pragma omp parallel for reduction(max:maxErr) reduction(max:relErr) schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        const double e = std::fabs(reconstructed[i] - A_orig[i]);
        maxErr = std::max(maxErr, e);
        relErr = std::max(relErr, e / (std::fabs(A_orig[i]) + 1e-10));
    }

    printf("Max absolute error: %.10e\n", maxErr);
    printf("Max relative error: %.10e\n", relErr);
    return relErr <= 1e-6;
}

/* ------------------------------------------------------------------ */
/*  CLI                                                               */
/* ------------------------------------------------------------------ */
void printUsage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    size_t n = 512;
    bool validate = false, printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                n = static_cast<size_t>(atoi(argv[++i]));
            else if (strcmp(argv[i], "-v") == 0)
                validate = true;
            else if (strcmp(argv[i], "-r") == 0)
                printResults = true;
            else if (strcmp(argv[i], "-h") == 0) {
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

    MPI_CHECK(MPI_Bcast(&n,            1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&validate,     1, MPI_C_BOOL,        0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&printResults, 1, MPI_C_BOOL,        0, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA enabled\n",
               nranks, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each rank generates the full matrix independently (same seed → same data)
    std::vector<double> A(n * n);
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    const std::vector<double> A_orig = validate ? A : std::vector<double>();

    // ---- timed Cholesky ----
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    const auto t0 = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    const auto t1 = std::chrono::high_resolution_clock::now();
    const long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    // ---- results (rank 0 only) ----
    if (rank == 0) {
        printf("Computation time: %ld ms\n", ms);
        const double gflops = (static_cast<double>(n) * n * n / 3.0)
                             / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults)
            print_results(A, "CholeskyL");

        if (validate) {
            printf("Validating result...\n");
            const bool v = validateCholesky(A, A_orig, n);
            printf("Validation: %s\n", v ? "PASSED" : "FAILED");
            if (!v) { MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
