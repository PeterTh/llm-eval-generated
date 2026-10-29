#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../common/results_output.hpp"

#define CHECK(call) do { auto err = (call); if (err != 0) { \
    fprintf(stderr, "Rank %d: %s failed (%d) at line %d\n", rank, #call, (int)err, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    int n = 512; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end; long v = strtol(argv[++i], &end, 10);
            if (*end || v < 0 || v > INT_MAX) { if (!rank) fprintf(stderr, "Invalid matrix size\n"); MPI_Finalize(); return 1; }
            n = static_cast<int>(v);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!rank) printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled");
    int devices = 0; CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank; MPI_Comm_rank(local, &local_rank); CHECK(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local);
    cublasHandle_t blas; cusolverDnHandle_t solver; CHECK(cublasCreate(&blas)); CHECK(cusolverDnCreate(&solver));
    std::vector<int> counts(nranks), offsets(nranks);
    for (int p = 0; p < nranks; ++p) { offsets[p] = (long long)n * p / nranks; counts[p] = (long long)n * (p + 1) / nranks - offsets[p]; }
    int start = offsets[rank], rows = counts[rank], ld = std::max(rows, 1);
    size_t nn = (size_t)n * n;
    std::vector<double> B(nn); unsigned int seed = 42;
    for (size_t i = 0; i < nn; ++i) B[i] = rand_r(&seed) / (double)RAND_MAX - 0.5;
    // B is row-major on the host; the device copy is its column-major transpose.
    double *db, *da, *panel, *diag, *work; int *info;
    CHECK(cudaMalloc(&db, std::max<size_t>(nn, 1) * sizeof(double)));
    CHECK(cudaMalloc(&da, std::max<size_t>((size_t)ld * n, 1) * sizeof(double)));
    constexpr int block = 128;
    CHECK(cudaMalloc(&panel, std::max<size_t>((size_t)n * block, 1) * sizeof(double)));
    CHECK(cudaMalloc(&diag, block * block * sizeof(double)));
    CHECK(cudaMalloc(&info, sizeof(int)));
    if (nn) CHECK(cudaMemcpy(db, B.data(), nn * sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0, minus = -1.0;
    if (rows && n) {
        // A_local = B_local * B^T, with B rows contiguous in host layout.
        CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, n, &one, db + (size_t)start * n, n, db, n, &zero, da, ld));
    }
    // Add n to the diagonal on the host after copying the distributed slabs.
    std::vector<double> initial((size_t)rows * n);
    if (rows && n) CHECK(cudaMemcpy(initial.data(), da, initial.size() * sizeof(double), cudaMemcpyDeviceToHost));
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) initial[(size_t)(start + i) * rows + i] += n;
    if (rows && n) CHECK(cudaMemcpy(da, initial.data(), initial.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaFree(db)); B.clear(); B.shrink_to_fit();
    int lwork = 0;
    if (rows && n) CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, std::min(block, rows), da, ld, &lwork));
    CHECK(cudaMalloc(&work, std::max(lwork, 1) * sizeof(double)));
    std::vector<double> hostdiag(block * block), hostpanel((size_t)n * block);
    if (!rank) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); double begin = MPI_Wtime();
    bool success = true;
    for (int k = 0; k < n;) {
        int owner = 0; while (owner + 1 < nranks && offsets[owner] + counts[owner] <= k) ++owner;
        int b = std::min({block, n - k, offsets[owner] + counts[owner] - k});
        int status = 0;
        if (rank == owner) {
            double* d = da + (size_t)k * ld + k - start;
            CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, b, d, ld, work, lwork, info));
            CHECK(cudaMemcpy(&status, info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!status) CHECK(cudaMemcpy2D(hostdiag.data(), b * sizeof(double), d, ld * sizeof(double), b * sizeof(double), b, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (status) { if (!rank) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + status - 1); success = false; break; }
        MPI_Bcast(hostdiag.data(), b * b, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CHECK(cudaMemcpy(diag, hostdiag.data(), (size_t)b * b * sizeof(double), cudaMemcpyHostToDevice));
        int first = std::max(start, k + b), active = std::max(0, start + rows - first);
        if (active) CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, active, b, &one, diag, b, da + (size_t)k * ld + first - start, ld));
        // Gather each column of the panel; all ranks then update their trailing rows.
        std::vector<double> send((size_t)active * b);
        if (active) CHECK(cudaMemcpy2D(send.data(), active * sizeof(double), da + (size_t)k * ld + first - start, ld * sizeof(double), active * sizeof(double), b, cudaMemcpyDeviceToHost));
        std::vector<int> recvcounts(nranks), displs(nranks), rowcounts(nranks);
        int trailing = n - k - b, total = 0;
        for (int p = 0; p < nranks; ++p) {
            int f = std::max(offsets[p], k + b);
            rowcounts[p] = std::max(0, offsets[p] + counts[p] - f);
            if ((long long)rowcounts[p] * b > INT_MAX || (long long)total + (long long)rowcounts[p] * b > INT_MAX) {
                fprintf(stderr, "Rank %d: panel exceeds MPI count limit\n", rank); MPI_Abort(MPI_COMM_WORLD, 1);
            }
            recvcounts[p] = rowcounts[p] * b; displs[p] = total; total += recvcounts[p];
        }
        std::vector<double> gathered(total);
        MPI_Allgatherv(send.data(), active * b, MPI_DOUBLE, gathered.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < b; ++j) {
            int pos = 0;
            for (int p = 0; p < nranks; ++p) {
                if (rowcounts[p]) std::memcpy(hostpanel.data() + (size_t)j * trailing + pos,
                    gathered.data() + displs[p] + (size_t)j * rowcounts[p], rowcounts[p] * sizeof(double));
                pos += rowcounts[p];
            }
        }
        if (trailing) CHECK(cudaMemcpy(panel, hostpanel.data(), (size_t)trailing * b * sizeof(double), cudaMemcpyHostToDevice));
        if (active && trailing) CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, active, trailing, b, &minus, panel + first - (k + b), trailing, panel, trailing, &one, da + (size_t)(k + b) * ld + first - start, ld));
        k += b;
    }
    CHECK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD); double elapsed = MPI_Wtime() - begin;
    if (!success) { if (!rank) printf("Cholesky decomposition failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (!rank) printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", (long)(elapsed * 1000), (double)n * n * n / 3.0 / elapsed / 1e9);
    if (validate || results) {
        std::vector<double> slab((size_t)rows * n), factor(rank == 0 ? nn : 0), orig(rank == 0 && validate ? nn : 0);
        if (rows && n) CHECK(cudaMemcpy(slab.data(), da, slab.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> column(n), original_column(n);
        for (int j = 0; j < n; ++j) {
            MPI_Gatherv(rows ? slab.data() + (size_t)j * rows : nullptr, rows, MPI_DOUBLE, column.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (validate) MPI_Gatherv(rows ? initial.data() + (size_t)j * rows : nullptr, rows, MPI_DOUBLE, original_column.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (!rank) for (int i = 0; i < n; ++i) { factor[(size_t)i * n + j] = i >= j ? column[i] : 0.0; if (validate) orig[(size_t)i * n + j] = original_column[i]; }
        }
        if (!rank && results) print_results(factor, "CholeskyL");
        int valid = 1;
        if (!rank && validate) {
            printf("Validating result...\n"); double maxerr = 0, relerr = 0;
            #pragma omp parallel for reduction(max:maxerr,relerr) schedule(static)
            for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
                double sum = 0; for (int k = 0; k <= std::min(i,j); ++k) sum += factor[(size_t)i*n+k] * factor[(size_t)j*n+k];
                double e = fabs(sum - orig[(size_t)i*n+j]); maxerr = std::max(maxerr,e); relerr = std::max(relerr,e / (fabs(orig[(size_t)i*n+j]) + 1e-10));
            }
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxerr, relerr);
            valid = relerr <= 1e-6; if (!valid) printf("Validation failed: relative error too large\n"); printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid) { MPI_Finalize(); return 1; }
    }
    CHECK(cudaFree(work)); CHECK(cudaFree(info)); CHECK(cudaFree(diag)); CHECK(cudaFree(panel)); CHECK(cudaFree(da)); CHECK(cusolverDnDestroy(solver)); CHECK(cublasDestroy(blas)); MPI_Finalize(); return 0;
}
