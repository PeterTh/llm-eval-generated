#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include "../common/results_output.hpp"

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define BLAS(call) do { if ((call) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS error %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define SOLVER(call) do { if ((call) != CUSOLVER_STATUS_SUCCESS) { fprintf(stderr, "cuSOLVER error %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512; bool validate = false, print = false; int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end; long x = strtol(argv[++i], &end, 10);
            if (*end || x <= 0 || x > INT_MAX) { if (!rank) fprintf(stderr, "Invalid matrix size\n"); MPI_Finalize(); return 1; }
            n = (int)x;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    // MPI counts and cuBLAS leading dimensions are integers.
    if ((size_t)n * n > INT_MAX) { if (!rank) fprintf(stderr, "Matrix too large for MPI counts\n"); MPI_Finalize(); return 1; }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices; MPI_Comm_rank(local, &localRank); CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % devices)); MPI_Comm_free(&local);
    cublasHandle_t blas; cusolverDnHandle_t solver; BLAS(cublasCreate(&blas)); SOLVER(cusolverDnCreate(&solver));
    constexpr int block = 128;
    std::vector<int> rows;
    for (int i = 0; i < n; ++i) if ((i / block) % ranks == rank) rows.push_back(i);
    const int m = (int)rows.size(), ld = std::max(1, m);
    if (!rank) { printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled"); }
    // Preserve the original rand_r sequence on every rank; each rank needs all rows of B.
    std::vector<double> B((size_t)n * n), localB((size_t)ld * n);
    unsigned int seed = 42;
    for (double &x : B) x = rand_r(&seed) / (double)RAND_MAX - 0.5;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) localB[(size_t)j * ld + i] = B[(size_t)rows[i] * n + j];
    double *dB, *dLocalB, *dA, *dPanel, *dDiag, *work; int *info;
    CUDA(cudaMalloc(&dB, (size_t)n*n*sizeof(double))); CUDA(cudaMalloc(&dLocalB, (size_t)ld*n*sizeof(double)));
    CUDA(cudaMalloc(&dA, (size_t)ld*n*sizeof(double))); CUDA(cudaMalloc(&dPanel, (size_t)n*block*sizeof(double)));
    CUDA(cudaMalloc(&dDiag, (size_t)block*block*sizeof(double))); CUDA(cudaMalloc(&info, sizeof(int)));
    CUDA(cudaMemcpy(dB, B.data(), (size_t)n*n*sizeof(double), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dLocalB, localB.data(), (size_t)ld*n*sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0, minus = -1.0;
    // B's row-major storage is the column-major transpose.
    if (m) BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one, dLocalB, ld, dB, n, &zero, dA, ld));
    std::vector<double> diagonal(m, (double)n);
    if (m) {
        CUDA(cudaMemcpy(dLocalB, diagonal.data(), (size_t)m*sizeof(double), cudaMemcpyHostToDevice));
        for (int i = 0; i < m; ++i) BLAS(cublasDaxpy(blas, 1, &one, dLocalB+i, 1, dA + (size_t)rows[i]*ld+i, 1));
    }
    CUDA(cudaFree(dB)); CUDA(cudaFree(dLocalB)); B.clear(); localB.clear();
    std::vector<double> localMatrix((size_t)ld*n), full, original;
    auto collect = [&](std::vector<double>& result, bool lower) {
        CUDA(cudaMemcpy(localMatrix.data(), dA, (size_t)ld*n*sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> assembled((size_t)n*n, 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j)
            if (!lower || j <= rows[i]) assembled[(size_t)rows[i]*n+j] = localMatrix[(size_t)j*ld+i];
        if (!rank) result.resize((size_t)n*n);
        MPI_Reduce(assembled.data(), rank == 0 ? result.data() : nullptr, n*n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    };
    if (validate) collect(original, false);
    int lwork; SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, block, dDiag, block, &lwork));
    CUDA(cudaMalloc(&work, (size_t)lwork*sizeof(double)));
    std::vector<double> panel((size_t)n*block), panelLocal((size_t)ld*block);
    if (!rank) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    for (int k = 0; k < n; k += block) {
        int b = std::min(block, n-k), owner = (k / block) % ranks;
        int first = (int)(std::lower_bound(rows.begin(), rows.end(), k) - rows.begin());
        int below = (int)(std::lower_bound(rows.begin(), rows.end(), k+b) - rows.begin());
        int ok = 0;
        if (rank == owner) {
            CUDA(cudaMemcpy2D(dDiag, block*sizeof(double), dA+(size_t)k*ld+first, ld*sizeof(double), b*sizeof(double), b, cudaMemcpyDeviceToDevice));
            SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, b, dDiag, block, work, lwork, info));
            CUDA(cudaMemcpy(&ok, info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!ok) CUDA(cudaMemcpy2D(dA+(size_t)k*ld+first, ld*sizeof(double), dDiag, block*sizeof(double), b*sizeof(double), b, cudaMemcpyDeviceToDevice));
        }
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (ok) { if (!rank) printf("Error: Matrix is not positive definite at diagonal element %d\n", k+ok-1); status = 1; break; }
        // Share the factored diagonal before solving all local portions of the panel.
        std::vector<double> diag((size_t)block*block);
        if (rank == owner) CUDA(cudaMemcpy(diag.data(), dDiag, diag.size()*sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Bcast(diag.data(), block*block, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank != owner) CUDA(cudaMemcpy(dDiag, diag.data(), diag.size()*sizeof(double), cudaMemcpyHostToDevice));
        if (m > below) BLAS(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m-below, b, &one, dDiag, block, dA+(size_t)k*ld+below, ld));
        if (m) CUDA(cudaMemcpy2D(panelLocal.data(), ld*sizeof(double), dA+(size_t)k*ld, ld*sizeof(double), (size_t)m*sizeof(double), b, cudaMemcpyDeviceToHost));
        std::fill(panel.begin(), panel.begin()+(size_t)n*b, 0.0);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < b; ++j) for (int i = first; i < m; ++i)
            if (rows[i] >= k+j) panel[(size_t)j*n+rows[i]] = panelLocal[(size_t)j*ld+i];
        MPI_Allreduce(MPI_IN_PLACE, panel.data(), n*b, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        CUDA(cudaMemcpy(dPanel, panel.data(), (size_t)n*b*sizeof(double), cudaMemcpyHostToDevice));
        if (m > below && k+b < n) BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m-below, n-k-b, b, &minus, dA+(size_t)k*ld+below, ld, dPanel+k+b, n, &one, dA+(size_t)(k+b)*ld+below, ld));
    }
    CUDA(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD); double elapsed = MPI_Wtime()-start;
    if (!status && (print || validate)) collect(full, true);
    if (!rank && !status) {
        long ms = (long)(elapsed*1000.0);
        printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", ms, (double)n*n*n/3.0/elapsed/1e9);
        if (print) print_results(full, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            double maxError = 0.0, relError = 0.0;
            #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
            for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k <= std::min(i,j); ++k) sum += full[(size_t)i*n+k]*full[(size_t)j*n+k];
                double error = fabs(sum-original[(size_t)i*n+j]);
                maxError = std::max(maxError, error); relError = std::max(relError, error/(fabs(original[(size_t)i*n+j])+1e-10));
            }
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
            if (relError > 1e-6) { printf("Validation failed: relative error too large\nValidation: FAILED\n"); status = 1; }
            else printf("Validation: PASSED\n");
        }
    } else if (!rank && status) printf("Cholesky decomposition failed\n");
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA(cudaFree(work)); CUDA(cudaFree(info)); CUDA(cudaFree(dDiag)); CUDA(cudaFree(dPanel)); CUDA(cudaFree(dA));
    SOLVER(cusolverDnDestroy(solver)); BLAS(cublasDestroy(blas)); MPI_Finalize(); return status;
}
