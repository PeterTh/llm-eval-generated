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
#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "Rank %d: CUDA: %s (%s:%d)\n", rank_id, cudaGetErrorString(e_), __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CHECK_BLAS(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "Rank %d: cuBLAS failure at line %d\n", rank_id, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CHECK_SOLVER(x) do { if ((x) != CUSOLVER_STATUS_SUCCESS) { fprintf(stderr, "Rank %d: cuSOLVER failure at line %d\n", rank_id, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}
int main(int argc, char** argv) {
    int provided, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank_id) fprintf(stderr, "MPI does not provide required thread support\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int n = 512; bool validate = false, output = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end; long v = strtol(argv[++i], &end, 10);
            if (*end || v <= 0 || v > INT_MAX || v > INT_MAX / v) { if (!rank_id) fprintf(stderr, "Invalid matrix size\n"); MPI_Finalize(); return 1; }
            n = static_cast<int>(v);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) output = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank_id) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank_id) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local);
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
    CHECK_CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank_id); MPI_Abort(MPI_COMM_WORLD, 1); }
    CHECK_CUDA(cudaSetDevice(local_rank % devices));
    cublasHandle_t blas; cusolverDnHandle_t solver;
    CHECK_BLAS(cublasCreate(&blas)); CHECK_SOLVER(cusolverDnCreate(&solver));
    const size_t nn = size_t(n) * n;
    std::vector<double> host(nn), original;
    unsigned int seed = 42;
    for (size_t i = 0; i < nn; ++i) host[i] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    if (!rank_id) { printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled"); }
    double *a, *b; CHECK_CUDA(cudaMalloc(&a, nn * sizeof(double))); CHECK_CUDA(cudaMalloc(&b, nn * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(b, host.data(), nn * sizeof(double), cudaMemcpyHostToDevice));
    // The row-major random matrix is the transpose of the column-major device view.
    const double one = 1.0, zero = 0.0, minus = -1.0;
    CHECK_BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, b, n, b, n, &zero, a, n));
    CHECK_CUDA(cudaFree(b));
    // Add n to the diagonal using a strided vector operation.
    std::vector<double> diagonal(n, double(n)); double* d_diag;
    CHECK_CUDA(cudaMalloc(&d_diag, size_t(n) * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_diag, diagonal.data(), size_t(n) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_BLAS(cublasDaxpy(blas, n, &one, d_diag, 1, a, n + 1)); CHECK_CUDA(cudaFree(d_diag));
    if (validate && !rank_id) { original.resize(nn); CHECK_CUDA(cudaMemcpy(original.data(), a, nn * sizeof(double), cudaMemcpyDeviceToHost)); }
    constexpr int block = 256;
    int lwork = 0; CHECK_SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, std::min(n, block), a, n, &lwork));
    double* work; int* info; CHECK_CUDA(cudaMalloc(&work, size_t(lwork) * sizeof(double))); CHECK_CUDA(cudaMalloc(&info, sizeof(int)));
    std::vector<double> panel(size_t(n) * block);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank_id) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::steady_clock::now();
    int failed = 0;
    for (int k = 0, kb = 0; k < n; k += block, ++kb) {
        int w = std::min(block, n-k), owner = kb % ranks;
        if (rank_id == owner) {
            CHECK_SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, w, a + size_t(k)*n+k, n, work, lwork, info));
            CHECK_CUDA(cudaMemcpy(&failed, info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!failed) CHECK_CUDA(cudaMemcpy2D(panel.data(), size_t(w)*sizeof(double), a+size_t(k)*n+k, size_t(n)*sizeof(double), size_t(w)*sizeof(double), w, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) { if (!rank_id) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + failed - 1); break; }
        MPI_Bcast(panel.data(), w*w, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank_id != owner) CHECK_CUDA(cudaMemcpy2D(a+size_t(k)*n+k, size_t(n)*sizeof(double), panel.data(), size_t(w)*sizeof(double), size_t(w)*sizeof(double), w, cudaMemcpyHostToDevice));
        // Each rank solves and broadcasts its owned block rows of the panel.
        for (int i = k + w, ib = kb + 1; i < n; i += block, ++ib) {
            int h = std::min(block, n-i), root = ib % ranks;
            if (rank_id == root) {
                CHECK_BLAS(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, h, w, &one, a+size_t(k)*n+k, n, a+size_t(k)*n+i, n));
                CHECK_CUDA(cudaMemcpy2D(panel.data()+size_t(i)*w, size_t(h)*sizeof(double), a+size_t(k)*n+i, size_t(n)*sizeof(double), size_t(h)*sizeof(double), w, cudaMemcpyDeviceToHost));
            }
            MPI_Bcast(panel.data()+size_t(i)*w, h*w, MPI_DOUBLE, root, MPI_COMM_WORLD);
            if (rank_id != root) CHECK_CUDA(cudaMemcpy2D(a+size_t(k)*n+i, size_t(n)*sizeof(double), panel.data()+size_t(i)*w, size_t(h)*sizeof(double), size_t(h)*sizeof(double), w, cudaMemcpyHostToDevice));
        }
        // Update only the lower-triangular blocks owned by this rank.
        for (int i = k+w, ib = kb+1; i < n; i += block, ++ib) if (ib % ranks == rank_id) {
            int h = std::min(block, n-i);
            for (int j = k+w; j <= i; j += block) {
                int q = std::min(block, n-j);
                CHECK_BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, h, q, w, &minus, a+size_t(k)*n+i, n, a+size_t(k)*n+j, n, &one, a+size_t(j)*n+i, n));
            }
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    if (!rank_id && !failed) { double seconds = std::chrono::duration<double>(end-start).count(); printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", long(seconds*1000), double(n)*n*n/3.0/seconds/1e9); }
    if (!failed && (output || validate)) {
        // Gather lower block rows, converting from column-major to the original row-major layout.
        std::vector<double> tile(size_t(block)*block);
        if (!rank_id) std::fill(host.begin(), host.end(), 0.0);
        for (int i=0, ib=0; i<n; i+=block, ++ib) for (int j=0; j<=i; j+=block) {
            int h=std::min(block,n-i), w=std::min(block,n-j), root=ib%ranks;
            if (rank_id == root) CHECK_CUDA(cudaMemcpy2D(tile.data(), size_t(h)*sizeof(double), a+size_t(j)*n+i, size_t(n)*sizeof(double), size_t(h)*sizeof(double), w, cudaMemcpyDeviceToHost));
            if (root != 0) { if (rank_id == root) MPI_Send(tile.data(), h*w, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); else if (!rank_id) MPI_Recv(tile.data(), h*w, MPI_DOUBLE, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
            if (!rank_id) for (int col=0; col<w; ++col) for (int row=0; row<h; ++row) if (i+row >= j+col) host[size_t(i+row)*n+j+col]=tile[size_t(col)*h+row];
        }
        if (!rank_id && output) print_results(host, "CholeskyL");
        if (!rank_id && validate) {
            printf("Validating result...\n"); double maxerr=0, relerr=0;
            #pragma omp parallel for reduction(max:maxerr,relerr) schedule(static)
            for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) {
                double sum=0; for (int k=0; k<=std::min(i,j); ++k) sum += host[size_t(i)*n+k]*host[size_t(j)*n+k];
                double expected=original[size_t(j)*n+i], err=std::fabs(sum-expected);
                maxerr=std::max(maxerr,err); relerr=std::max(relerr,err/(std::fabs(expected)+1e-10));
            }
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxerr, relerr);
            if (relerr > 1e-6) { printf("Validation failed: relative error too large\nValidation: FAILED\n"); failed=1; } else printf("Validation: PASSED\n");
        }
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(info); cudaFree(work); cudaFree(a); cusolverDnDestroy(solver); cublasDestroy(blas); MPI_Finalize(); return failed ? 1 : 0;
}
