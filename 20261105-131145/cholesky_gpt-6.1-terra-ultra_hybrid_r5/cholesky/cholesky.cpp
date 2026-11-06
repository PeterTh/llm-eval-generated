#include <algorithm>
#include <climits>
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

static int rank_id;

static void fail(const char* what, int code) {
    fprintf(stderr, "Rank %d: %s failed (%d)\n", rank_id, what, code);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) fail(#call, (int)e); } while (0)
#define BLAS(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) fail(#call, (int)e); } while (0)
#define SOLVER(call) do { cusolverStatus_t e = (call); if (e != CUSOLVER_STATUS_SUCCESS) fail(#call, (int)e); } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) fail(#call, e); } while (0)

template <typename T> static T* device_alloc(size_t count) {
    T* p = nullptr;
    if (count) CUDA(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)));
    return p;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id));
    int nranks;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI thread support", provided);

    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            long x = strtol(argv[++i], &end, 10);
            // MPI counts and displacements below are expressed as int.
            if (!*argv[i] || *end || x < 0 || x > 46340) {
                if (!rank_id) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = static_cast<int>(x);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank_id) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank_id) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!rank_id) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local_comm));
    int local_rank, devices;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) fail("no CUDA devices", 0);
    CUDA(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    BLAS(cublasCreate(&blas));
    SOLVER(cusolverDnCreate(&solver));

    std::vector<int> starts(nranks + 1), counts(nranks), displs(nranks);
    for (int p = 0; p <= nranks; ++p) starts[p] = static_cast<int>((static_cast<long long>(n) * p) / nranks);
    for (int p = 0; p < nranks; ++p) {
        counts[p] = (starts[p + 1] - starts[p]) * n;
        displs[p] = starts[p] * n;
    }
    int first = starts[rank_id], rows = starts[rank_id + 1] - first;
    size_t local_size = static_cast<size_t>(rows) * n;
    double* dA = device_alloc<double>(local_size);

    // Every rank reproduces the original rand_r stream exactly.  B is reused
    // by the GPU to construct only the rows owned by this rank.
    std::vector<double> B(static_cast<size_t>(n) * n);
    unsigned int seed = 42;
    for (double& x : B) x = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    double* dB = device_alloc<double>(B.size());
    if (!B.empty()) CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double>().swap(B);
    const double one = 1.0, zero = 0.0, minus_one = -1.0;
    if (rows && n) {
        // Row-major row blocks are column-major blocks of A^T.
        BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, n, &one,
                         dB, n, dB + static_cast<size_t>(first) * n, n, &zero, dA, n));
    }
    CUDA(cudaFree(dB));
    std::vector<double> hostA(local_size);
    if (local_size) CUDA(cudaMemcpy(hostA.data(), dA, local_size * sizeof(double), cudaMemcpyDeviceToHost));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) hostA[static_cast<size_t>(i) * n + first + i] += n;
    if (local_size) CUDA(cudaMemcpy(dA, hostA.data(), local_size * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> original;
    if (validate) original = hostA;
    std::vector<double>().swap(hostA);

    constexpr int tile = 256;
    const int maxb = std::min(tile, n);
    double* diag = device_alloc<double>(static_cast<size_t>(maxb) * maxb);
    double* panel = device_alloc<double>(static_cast<size_t>(maxb) * n);
    int* devinfo = device_alloc<int>(1);
    int lwork = 0;
    bool queried[tile + 1] = {};
    for (int p = 0; p < nranks; ++p) {
        int length = starts[p + 1] - starts[p];
        int sizes[2] = {length >= tile ? tile : 0, length % tile};
        for (int b : sizes) if (b && !queried[b]) {
            int required = 0;
            SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, b, diag, b, &required));
            lwork = std::max(lwork, required);
            queried[b] = true;
        }
    }
    double* work = device_alloc<double>(lwork);
    std::vector<double> hdiag(static_cast<size_t>(maxb) * maxb);
    std::vector<double> hlocal(static_cast<size_t>(maxb) * rows);
    std::vector<double> hpanel(static_cast<size_t>(maxb) * n);
    std::vector<int> pc(nranks), pd(nranks);

    if (!rank_id) printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    bool success = true;
    for (int j = 0; j < n;) {
        int owner = static_cast<int>(std::upper_bound(starts.begin(), starts.end(), j) - starts.begin()) - 1;
        int b = std::min(tile, starts[owner + 1] - j);
        int end = j + b;
        int info = 0;
        if (rank_id == owner) {
            const double* block = dA + j + static_cast<size_t>(j - first) * n;
            CUDA(cudaMemcpy2D(diag, b * sizeof(double), block, n * sizeof(double), b * sizeof(double), b, cudaMemcpyDeviceToDevice));
            SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, b, diag, b, work, lwork, devinfo));
            CUDA(cudaMemcpy(&info, devinfo, sizeof(info), cudaMemcpyDeviceToHost));
            if (!info) CUDA(cudaMemcpy(hdiag.data(), diag, static_cast<size_t>(b) * b * sizeof(double), cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info) {
            if (!rank_id) printf("Error: Matrix is not positive definite at diagonal element %d\n", j + std::max(0, info - 1));
            success = false; break;
        }
        MPI_CHECK(MPI_Bcast(hdiag.data(), b * b, MPI_DOUBLE, owner, MPI_COMM_WORLD));
        if (rank_id != owner) CUDA(cudaMemcpy(diag, hdiag.data(), static_cast<size_t>(b) * b * sizeof(double), cudaMemcpyHostToDevice));
        // Store the factored diagonal back into its original row block.
        if (rank_id == owner) {
            double* block = dA + j + static_cast<size_t>(j - first) * n;
            CUDA(cudaMemcpy2D(block, n * sizeof(double), diag, b * sizeof(double), b * sizeof(double), b, cudaMemcpyDeviceToDevice));
        }
        int lo = std::max(first, end), local_trailing = std::max(0, starts[rank_id + 1] - lo);
        if (local_trailing) {
            double* local_panel = dA + j + static_cast<size_t>(lo - first) * n;
            BLAS(cublasDtrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                             CUBLAS_DIAG_NON_UNIT, b, local_trailing, &one, diag, b, local_panel, n));
            CUDA(cudaMemcpy2D(hlocal.data(), b * sizeof(double), local_panel, n * sizeof(double),
                              b * sizeof(double), local_trailing, cudaMemcpyDeviceToHost));
        }
        for (int p = 0; p < nranks; ++p) {
            int s = std::max(starts[p], end), e = std::max(starts[p + 1], end);
            pc[p] = (e - s) * b;
            pd[p] = (s - end) * b;
        }
        MPI_CHECK(MPI_Allgatherv(hlocal.data(), local_trailing * b, MPI_DOUBLE, hpanel.data(), pc.data(), pd.data(), MPI_DOUBLE, MPI_COMM_WORLD));
        int trailing = n - end;
        if (trailing && local_trailing) {
            CUDA(cudaMemcpy(panel, hpanel.data(), static_cast<size_t>(trailing) * b * sizeof(double), cudaMemcpyHostToDevice));
            const double* local_panel = dA + j + static_cast<size_t>(lo - first) * n;
            double* target = dA + end + static_cast<size_t>(lo - first) * n;
            BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, trailing, local_trailing, b,
                             &minus_one, panel, b, local_panel, n, &one, target, n));
        }
        j = end;
    }
    CUDA(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (!rank_id) printf("Cholesky decomposition failed\n");
        MPI_Finalize(); return 1;
    }
    if (!rank_id) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));
        double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / max_elapsed / 1e9);
    }

    if (validate || printResults) {
        hostA.resize(local_size);
        if (local_size) CUDA(cudaMemcpy(hostA.data(), dA, local_size * sizeof(double), cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i)
            std::fill(hostA.begin() + static_cast<size_t>(i) * n + first + i + 1,
                      hostA.begin() + static_cast<size_t>(i + 1) * n, 0.0);
        std::vector<double> full;
        if (validate || !rank_id) full.resize(static_cast<size_t>(n) * n);
        if (validate) MPI_CHECK(MPI_Allgatherv(hostA.data(), counts[rank_id], MPI_DOUBLE, full.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD));
        else MPI_CHECK(MPI_Gatherv(hostA.data(), counts[rank_id], MPI_DOUBLE, full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (printResults && !rank_id) print_results(full, "CholeskyL");
        if (validate) {
            if (!rank_id) printf("Validating result...\n");
            double* dL = device_alloc<double>(full.size());
            if (!full.empty()) CUDA(cudaMemcpy(dL, full.data(), full.size() * sizeof(double), cudaMemcpyHostToDevice));
            if (local_size) {
                BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, n, &one,
                                 dL, n, dL + static_cast<size_t>(first) * n, n, &zero, dA, n));
                CUDA(cudaMemcpy(hostA.data(), dA, local_size * sizeof(double), cudaMemcpyDeviceToHost));
            }
            CUDA(cudaFree(dL));
            double maxerr = 0.0, relerr = 0.0;
            #pragma omp parallel for reduction(max:maxerr,relerr) schedule(static)
            for (size_t i = 0; i < local_size; ++i) {
                double error = std::fabs(hostA[i] - original[i]);
                maxerr = std::max(maxerr, error);
                relerr = std::max(relerr, error / (std::fabs(original[i]) + 1e-10));
            }
            double local_errors[2] = {maxerr, relerr}, errors[2];
            MPI_CHECK(MPI_Allreduce(local_errors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
            if (!rank_id) {
                printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
                if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
                printf("Validation: %s\n", errors[1] > 1e-6 ? "FAILED" : "PASSED");
            }
            if (errors[1] > 1e-6) success = false;
        }
    }
    CUDA(cudaFree(work)); CUDA(cudaFree(devinfo)); CUDA(cudaFree(panel)); CUDA(cudaFree(diag)); CUDA(cudaFree(dA));
    SOLVER(cusolverDnDestroy(solver)); BLAS(cublasDestroy(blas));
    MPI_CHECK(MPI_Finalize());
    return success ? 0 : 1;
}
