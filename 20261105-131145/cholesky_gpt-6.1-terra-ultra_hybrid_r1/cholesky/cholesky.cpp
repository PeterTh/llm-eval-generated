#include <algorithm>
#include <chrono>
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

namespace {
int rank_id = 0;

[[noreturn]] void fail(const char* what, int code) {
    fprintf(stderr, "Rank %d: %s (error %d)\n", rank_id, what, code);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) fail(cudaGetErrorString(e_), int(e_)); } while (0)
#define BLAS(call) do { cublasStatus_t e_ = (call); if (e_ != CUBLAS_STATUS_SUCCESS) fail(#call, int(e_)); } while (0)
#define SOLVER(call) do { cusolverStatus_t e_ = (call); if (e_ != CUSOLVER_STATUS_SUCCESS) fail(#call, int(e_)); } while (0)
#define MPI_CHECK(call) do { int e_ = (call); if (e_ != MPI_SUCCESS) fail(#call, e_); } while (0)

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A, int n) {
    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k <= std::min(i, j); ++k)
                sum += L[size_t(i) * n + k] * L[size_t(j) * n + k];
            double error = std::fabs(sum - A[size_t(i) * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(A[size_t(i) * n + j]) + 1e-10));
        }
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// Gather column-major local row blocks and convert to the original layout.
std::vector<double> gatherMatrix(const double* device, int rows, int n,
                                 const std::vector<std::vector<int>>& globalRows,
                                 int block, bool triangular) {
    int ranks = int(globalRows.size());
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = int(globalRows[r].size()) * n;
        displs[r] = r ? displs[r - 1] + counts[r - 1] : 0;
    }
    std::vector<double> local(size_t(rows) * n);
    if (rows && n) CUDA(cudaMemcpy(local.data(), device, local.size() * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> packed, result;
    if (rank_id == 0) packed.resize(size_t(n) * n);
    MPI_CHECK(MPI_Gatherv(local.data(), counts[rank_id], MPI_DOUBLE, packed.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if (rank_id == 0) {
        result.resize(size_t(n) * n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            int r = (i / block) % ranks;
            int local = (i / block / ranks) * block + i % block;
            int length = int(globalRows[r].size());
            for (int j = 0; j < n; ++j)
                result[size_t(i) * n + j] = triangular && j > i ? 0.0 :
                    packed[size_t(displs[r]) + size_t(j) * length + local];
        }
    }
    return result;
}
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not provide MPI_THREAD_FUNNELED", provided);

    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            long value = std::strtol(argv[++i], &end, 10);
            if (*end || value < 0 || value > 46340) {
                if (!rank_id) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = int(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank_id) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank_id) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }

    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &localComm));
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) fail("No CUDA devices available", 0);
    CUDA(cudaSetDevice(localRank % devices));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    BLAS(cublasCreate(&blas));
    SOLVER(cusolverDnCreate(&solver));

    // Block-cyclic rows retain work on every rank as the factorization advances.
    int block = std::max(1, std::min(128, (n + ranks - 1) / ranks));
    int blocks = (n + block - 1) / block;
    std::vector<std::vector<int>> globalRows(ranks);
    for (int b = 0; b < blocks; ++b) {
        auto& owned = globalRows[b % ranks];
        for (int i = b * block; i < std::min(n, (b + 1) * block); ++i)
            owned.push_back(i);
    }
    const auto& myRows = globalRows[rank_id];
    int rows = int(myRows.size());
    if (!rank_id) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    // Preserve the original rand_r sequence, then form each row slab on the GPU.
    std::vector<double> B(size_t(n) * n);
    if (!rank_id) {
        unsigned int seed = 42;
        for (double& x : B) x = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
    MPI_CHECK(MPI_Bcast(B.data(), int(B.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    double *dB = nullptr, *dA = nullptr;
    if (n) {
        CUDA(cudaMalloc(&dB, size_t(n) * n * sizeof(double)));
        CUDA(cudaMemcpy(dB, B.data(), size_t(n) * n * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (rows) {
        CUDA(cudaMalloc(&dA, size_t(rows) * n * sizeof(double)));
        CUDA(cudaMemset(dA, 0, size_t(rows) * n * sizeof(double)));
        std::vector<double> diagonal(block, double(n));
        for (int offset = 0; offset < rows; offset += block) {
            int global = myRows[offset];
            int count = std::min(block, rows - offset);
            CUDA(cudaMemcpy2D(dA + offset + size_t(global) * rows,
                              size_t(rows + 1) * sizeof(double), diagonal.data(),
                              sizeof(double), sizeof(double), count, cudaMemcpyHostToDevice));
        }
        // Pack the owned rows into one operand so generation uses one large GEMM.
        double* dLocalB = dB;
        if (rows != n) {
            std::vector<double> localB(size_t(rows) * n);
            #pragma omp parallel for schedule(static) if(rows > 128)
            for (int i = 0; i < rows; ++i)
                std::memcpy(localB.data() + size_t(i) * n,
                            B.data() + size_t(myRows[i]) * n, size_t(n) * sizeof(double));
            CUDA(cudaMalloc(&dLocalB, size_t(rows) * n * sizeof(double)));
            CUDA(cudaMemcpy(dLocalB, localB.data(), localB.size() * sizeof(double), cudaMemcpyHostToDevice));
        }
        const double one = 1.0;
        BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, n, &one,
                         dLocalB, n, dB, n, &one, dA, rows));
        if (dLocalB != dB) CUDA(cudaFree(dLocalB));
    }
    std::vector<double>().swap(B);
    if (dB) CUDA(cudaFree(dB));
    std::vector<double> original;
    if (validate) original = gatherMatrix(dA, rows, n, globalRows, block, false);

    double *dDiag = nullptr, *dPanel = nullptr, *dWork = nullptr;
    int *dInfo = nullptr, lwork = 0;
    if (n) {
        CUDA(cudaMalloc(&dDiag, size_t(block) * block * sizeof(double)));
        CUDA(cudaMalloc(&dPanel, size_t(n) * block * sizeof(double)));
        CUDA(cudaMalloc(&dInfo, sizeof(int)));
        if (rows) {
            SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, std::min(block, rows), dA, rows, &lwork));
            CUDA(cudaMalloc(&dWork, size_t(std::max(1, lwork)) * sizeof(double)));
        }
    }
    std::vector<double> diag(size_t(block) * block), localPanel(size_t(rows) * block),
                        packedPanel(size_t(n) * block), panel(size_t(n) * block);
    std::vector<int> counts(ranks), displs(ranks);
    if (!rank_id) printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::steady_clock::now();
    bool success = true;
    for (int k = 0; k < n; k += block) {
        int width = std::min(block, n - k);
        int owner = (k / block) % ranks;
        int info = 0;
        if (rank_id == owner) {
            int localK = int(std::lower_bound(myRows.begin(), myRows.end(), k) - myRows.begin());
            double* tile = dA + localK + size_t(k) * rows;
            SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width, tile, rows, dWork, lwork, dInfo));
            CUDA(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (!info) CUDA(cudaMemcpy2D(diag.data(), size_t(width) * sizeof(double), tile, size_t(rows) * sizeof(double), size_t(width) * sizeof(double), width, cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info) {
            if (!rank_id) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + info - 1);
            success = false; break;
        }
        if (k + width == n) break;
        MPI_CHECK(MPI_Bcast(diag.data(), width * width, MPI_DOUBLE, owner, MPI_COMM_WORLD));
        CUDA(cudaMemcpy(dDiag, diag.data(), size_t(width) * width * sizeof(double), cudaMemcpyHostToDevice));
        int localStart = int(std::lower_bound(myRows.begin(), myRows.end(), k + width) - myRows.begin());
        int trailing = rows - localStart;
        if (trailing) {
            const double one = 1.0;
            BLAS(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                             CUBLAS_DIAG_NON_UNIT, trailing, width, &one, dDiag, width,
                             dA + localStart + size_t(k) * rows, rows));
        }
        // Stage through host memory so MPI does not require CUDA-aware support.
        if (trailing) CUDA(cudaMemcpy2D(localPanel.data(), size_t(trailing) * sizeof(double), dA + localStart + size_t(k) * rows,
                                        size_t(rows) * sizeof(double), size_t(trailing) * sizeof(double), width, cudaMemcpyDeviceToHost));
        for (int r = 0; r < ranks; ++r) {
            int firstTrailing = int(std::lower_bound(globalRows[r].begin(), globalRows[r].end(), k + width) - globalRows[r].begin());
            counts[r] = (int(globalRows[r].size()) - firstTrailing) * width;
            displs[r] = r ? displs[r - 1] + counts[r - 1] : 0;
        }
        MPI_CHECK(MPI_Allgatherv(localPanel.data(), trailing * width, MPI_DOUBLE, packedPanel.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD));
        #pragma omp parallel for schedule(static) if(n * width > 32768)
        for (int j = 0; j < width; ++j)
            for (int r = 0; r < ranks; ++r) {
                int firstTrailing = int(std::lower_bound(globalRows[r].begin(), globalRows[r].end(), k + width) - globalRows[r].begin());
                int length = int(globalRows[r].size()) - firstTrailing;
                for (int local = 0; local < length; ++local)
                    panel[size_t(j) * n + globalRows[r][firstTrailing + local]] = packedPanel[size_t(displs[r]) + size_t(j) * length + local];
            }
        if (trailing && k + width < n) {
            CUDA(cudaMemcpy2D(dPanel + k + width, size_t(n) * sizeof(double),
                              panel.data() + k + width, size_t(n) * sizeof(double),
                              size_t(n - k - width) * sizeof(double), width, cudaMemcpyHostToDevice));
            const double minusOne = -1.0, one = 1.0;
            BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, trailing, n - k - width, width,
                             &minusOne, dA + localStart + size_t(k) * rows, rows,
                             dPanel + k + width, n, &one,
                             dA + localStart + size_t(k + width) * rows, rows));
        }
    }
    CUDA(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::vector<double> result;
    if (success && (validate || printResults)) result = gatherMatrix(dA, rows, n, globalRows, block, true);
    int valid = success ? 1 : 0;
    if (!rank_id) {
        if (!success) printf("Cholesky decomposition failed\n");
        else {
            printf("Computation time: %ld ms\n", long(duration));
            double ops = double(n) * n * n / 3.0;
            printf("Performance: %.3f GFLOPS\n", ops / (duration / 1000.0) / 1e9);
            if (printResults) print_results(result, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                valid = validateCholesky(result, original, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (dWork) CUDA(cudaFree(dWork));
    if (dInfo) CUDA(cudaFree(dInfo));
    if (dPanel) CUDA(cudaFree(dPanel));
    if (dDiag) CUDA(cudaFree(dDiag));
    if (dA) CUDA(cudaFree(dA));
    SOLVER(cusolverDnDestroy(solver));
    BLAS(cublasDestroy(blas));
    MPI_Finalize();
    return valid ? 0 : 1;
}
