#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include "../common/results_output.hpp"

static int rank_id;
#define CHECK(call) do { if ((call) != 0) { fprintf(stderr, "Rank %d: %s failed at line %d\n", rank_id, #call, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank_id) fprintf(stderr, "MPI does not provide thread support\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int n = 512; bool validate = false, print = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end; long v = strtol(argv[++i], &end, 10);
            if (*end || v < 1 || v > INT_MAX) { if (!rank_id) fprintf(stderr, "Invalid matrix size\n"); MPI_Finalize(); return 1; }
            n = (int)v;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank_id) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank_id) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local);
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank_id); MPI_Abort(MPI_COMM_WORLD, 1); }
    CHECK(cudaSetDevice(local_rank % devices));
    cublasHandle_t blas; cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas)); CHECK(cusolverDnCreate(&solver));
    size_t count = (size_t)n * n;
    if (count > SIZE_MAX / sizeof(double)) { if (!rank_id) fprintf(stderr, "Matrix too large\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (!rank_id) { printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled"); }
    // Preserve the original rand_r sequence. Column-major B is the transpose of
    // the original row-major B, so B^T B produces the same symmetric matrix.
    std::vector<double> host(count);
    unsigned int seed = 42;
    for (size_t i = 0; i < count; ++i) host[i] = rand_r(&seed) / (double)RAND_MAX - 0.5;
    double *dB, *dA;
    CHECK(cudaMalloc(&dB, count * sizeof(double))); CHECK(cudaMalloc(&dA, count * sizeof(double)));
    CHECK(cudaMemcpy(dB, host.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0, minus = -1.0;
    CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, dB, n, dB, n, &zero, dA, n));
    CHECK(cudaFree(dB));
    // Add n to the diagonal on the host while retaining a copy for validation.
    CHECK(cudaMemcpy(host.data(), dA, count * sizeof(double), cudaMemcpyDeviceToHost));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) host[(size_t)i * n + i] += n;
    CHECK(cudaMemcpy(dA, host.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> original;
    if (validate && !rank_id) original = host;
    constexpr int block = 256;
    std::vector<double> panel((size_t)n * std::min(n, block));
    int lwork = 0, *info; double *work;
    CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, std::min(n, block), dA, n, &lwork));
    CHECK(cudaMalloc(&work, (size_t)std::max(lwork, 1) * sizeof(double))); CHECK(cudaMalloc(&info, sizeof(int)));
    if (!rank_id) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    int failed = 0;
    for (int k = 0, tile = 0; k < n; k += block, ++tile) {
        int b = std::min(block, n - k), end = k + b, owner = tile % ranks;
        if (rank_id == owner) {
            CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, b, dA + (size_t)k*n + k, n, work, lwork, info));
            CHECK(cudaMemcpy(&failed, info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!failed && end < n) {
                CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                  n-end, b, &one, dA + (size_t)k*n+k, n, dA + (size_t)k*n+end, n));
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) { if (!rank_id) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + failed - 1); break; }
        // Stage through host memory so this also works with non-CUDA-aware MPI.
        size_t elems = (size_t)n * b;
        if (rank_id == owner) CHECK(cudaMemcpy(panel.data(), dA + (size_t)k*n, elems*sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t off = 0; off < elems;) {
            int chunk = (int)std::min(elems-off, (size_t)INT_MAX);
            MPI_Bcast(panel.data()+off, chunk, MPI_DOUBLE, owner, MPI_COMM_WORLD); off += chunk;
        }
        if (rank_id != owner) CHECK(cudaMemcpy(dA + (size_t)k*n, panel.data(), elems*sizeof(double), cudaMemcpyHostToDevice));
        // Each rank owns complete trailing block columns. Only their lower parts
        // are updated; later owners broadcast the resulting factor panels.
        for (int j = end, t = tile + 1; j < n; j += block, ++t) if (t % ranks == rank_id) {
            int w = std::min(block, n-j);
            CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, n-j, w, b, &minus,
                              dA + (size_t)k*n+j, n, dA + (size_t)k*n+j, n,
                              &one, dA + (size_t)j*n+j, n));
        }
    }
    CHECK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;
    if (failed) { if (!rank_id) printf("Cholesky decomposition failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (!rank_id) {
        printf("Computation time: %ld ms\n", (long)(elapsed * 1000));
        printf("Performance: %.3f GFLOPS\n", (double)n*n*n / (3.0 * elapsed * 1e9));
    }
    if (!rank_id && (validate || print)) {
        CHECK(cudaMemcpy(host.data(), dA, count*sizeof(double), cudaMemcpyDeviceToHost));
        // Convert column-major factor to the original row-major result.
        std::vector<double> L(count, 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) L[(size_t)i*n+j] = host[(size_t)j*n+i];
        if (print) print_results(L, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            double maxError = 0, relError = 0;
            #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
            for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
                double sum = 0; for (int k = 0; k <= std::min(i,j); ++k) sum += L[(size_t)i*n+k]*L[(size_t)j*n+k];
                double err = fabs(sum-original[(size_t)j*n+i]);
                maxError = std::max(maxError, err); relError = std::max(relError, err/(fabs(original[(size_t)j*n+i])+1e-10));
            }
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
            if (relError > 1e-6) { printf("Validation failed: relative error too large\nValidation: FAILED\n"); failed = 1; }
            else printf("Validation: PASSED\n");
        }
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(info); cudaFree(work); cudaFree(dA); cusolverDnDestroy(solver); cublasDestroy(blas);
    MPI_Finalize(); return failed ? 1 : 0;
}
