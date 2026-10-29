#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../common/results_output.hpp"

static int rank_id;
#define CHECK(call) do { auto status_ = (call); if (status_ != 0) { \
    fprintf(stderr, "Rank %d: %s failed (%d) at line %d\n", rank_id, #call, (int)status_, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank_id) fprintf(stderr, "MPI thread support is insufficient\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int n = 512; bool validate = false, print = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end; long value = strtol(argv[++i], &end, 10);
            if (*end || value < 1 || value > INT_MAX) { if (!rank_id) fprintf(stderr, "Invalid matrix size\n"); MPI_Finalize(); return 1; }
            n = (int)value;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank_id) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank_id) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    // MPI's counts and displacements below are expressed in doubles.
    if ((size_t)n * n > INT_MAX) { if (!rank_id) fprintf(stderr, "Matrix is too large for MPI counts\n"); MPI_Finalize(); return 1; }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local);
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank_id); MPI_Abort(MPI_COMM_WORLD, 1); }
    CHECK(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local);
    cublasHandle_t blas; cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas)); CHECK(cusolverDnCreate(&solver));
    constexpr int block = 128;
    int blocks = (n + block - 1) / block;
    std::vector<int> starts(ranks + 1), counts(ranks), displs(ranks);
    for (int r = 0; r <= ranks; ++r) starts[r] = std::min(n, (int)((long long)blocks * r / ranks) * block);
    for (int r = 0; r < ranks; ++r) { counts[r] = (starts[r+1] - starts[r]) * n; displs[r] = starts[r] * n; }
    int first = starts[rank_id], cols = starts[rank_id+1] - first;
    if (!rank_id) printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled");
    // Keep the original rand_r sequence; B's rows are columns in column-major storage.
    std::vector<double> B((size_t)n*n); unsigned int seed = 42;
    for (double& x : B) x = rand_r(&seed) / (double)RAND_MAX - 0.5;
    double *dB, *dA, *dPanel, *work; int *info;
    CHECK(cudaMalloc(&dB, (size_t)n*n*sizeof(double)));
    CHECK(cudaMalloc(&dA, std::max((size_t)1, (size_t)n*cols)*sizeof(double)));
    CHECK(cudaMalloc(&dPanel, (size_t)n*block*sizeof(double)));
    CHECK(cudaMalloc(&info, sizeof(int)));
    CHECK(cudaMemcpy(dB, B.data(), (size_t)n*n*sizeof(double), cudaMemcpyHostToDevice));
    double one = 1.0, zero = 0.0, minus = -1.0;
    if (cols) CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, cols, n, &one, dB, n, dB+(size_t)first*n, n, &zero, dA, n));
    // Add n to the distributed diagonal with one strided device operation.
    std::vector<double> diag(cols, (double)n);
    double* dDiag;
    CHECK(cudaMalloc(&dDiag, std::max(1, cols)*sizeof(double)));
    if (cols) {
        CHECK(cudaMemcpy(dDiag, diag.data(), (size_t)cols*sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cublasDaxpy(blas, cols, &one, dDiag, 1, dA+first, n+1));
    }
    CHECK(cudaFree(dDiag));
    CHECK(cudaFree(dB)); B.clear(); B.shrink_to_fit();
    std::vector<double> original, gathered;
    std::vector<double> hostLocal((size_t)n*cols);
    if (validate) {
        CHECK(cudaMemcpy(hostLocal.data(), dA, hostLocal.size()*sizeof(double), cudaMemcpyDeviceToHost));
        if (!rank_id) original.resize((size_t)n*n);
        MPI_Gatherv(hostLocal.data(), counts[rank_id], MPI_DOUBLE, original.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int lwork; CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, std::min(block,n), dA, n, &lwork));
    CHECK(cudaMalloc(&work, (size_t)lwork*sizeof(double)));
    std::vector<double> panel((size_t)n*block);
    if (!rank_id) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    bool success = true;
    for (int k = 0; k < n; k += block) {
        int b = std::min(block, n-k), owner = (int)(std::upper_bound(starts.begin(), starts.end(), k) - starts.begin()) - 1;
        int failure = 0;
        if (rank_id == owner) {
            double* a = dA+(size_t)(k-first)*n+k;
            CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, b, a, n, work, lwork, info));
            CHECK(cudaMemcpy(&failure, info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!failure && k+b < n) CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n-k-b, b, &one, a, n, a+b, n));
            if (!failure) CHECK(cudaMemcpy2D(panel.data(), (size_t)(n-k)*sizeof(double), a, (size_t)n*sizeof(double), (size_t)(n-k)*sizeof(double), b, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(&failure, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failure) { if (!rank_id) printf("Error: Matrix is not positive definite at diagonal element %d\n", k+failure-1); success = false; break; }
        MPI_Bcast(panel.data(), (n-k)*b, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        int j = std::max(first, k+b), remaining = starts[rank_id+1]-j;
        if (remaining > 0) {
            CHECK(cudaMemcpy(dPanel, panel.data(), (size_t)(n-k)*b*sizeof(double), cudaMemcpyHostToDevice));
            CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, n-k-b, remaining, b, &minus, dPanel+b, n-k, dPanel+(j-k), n-k, &one, dA+(size_t)(j-first)*n+k+b, n));
        }
    }
    CHECK(cudaDeviceSynchronize()); double elapsed = MPI_Wtime()-start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { if (!rank_id) printf("Cholesky decomposition failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (!rank_id) printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", (long)(max_elapsed*1000), (double)n*n*n/3.0/max_elapsed/1e9);
    if (print || validate) {
        CHECK(cudaMemcpy(hostLocal.data(), dA, hostLocal.size()*sizeof(double), cudaMemcpyDeviceToHost));
        if (!rank_id) gathered.resize((size_t)n*n);
        MPI_Gatherv(hostLocal.data(), counts[rank_id], MPI_DOUBLE, gathered.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int valid = 1;
    if (!rank_id && (print || validate)) {
        std::vector<double> L((size_t)n*n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) L[(size_t)i*n+j] = gathered[(size_t)j*n+i];
        if (print) print_results(L, "CholeskyL");
        if (validate) {
            printf("Validating result...\n"); double maxError = 0, relError = 0;
            #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
            for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
                double sum = 0; for (int k = 0; k <= std::min(i,j); ++k) sum += L[(size_t)i*n+k]*L[(size_t)j*n+k];
                double ref = original[(size_t)j*n+i], error = fabs(sum-ref);
                maxError = std::max(maxError,error); relError = std::max(relError,error/(fabs(ref)+1e-10));
            }
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
            if (relError > 1e-6) { printf("Validation failed: relative error too large\nValidation: FAILED\n"); valid = 0; }
            else printf("Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CHECK(cudaFree(work)); CHECK(cudaFree(info)); CHECK(cudaFree(dPanel)); CHECK(cudaFree(dA));
    CHECK(cusolverDnDestroy(solver)); CHECK(cublasDestroy(blas)); MPI_Finalize(); return valid ? 0 : 1;
}
