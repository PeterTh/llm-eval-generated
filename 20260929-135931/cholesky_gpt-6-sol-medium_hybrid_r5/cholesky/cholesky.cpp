#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../common/results_output.hpp"

// Cyclic block columns balance the trailing updates across MPI ranks.
static constexpr int BLOCK = 128;
static void cudaCheck(cudaError_t s, const char* op) {
    if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(s)); MPI_Abort(MPI_COMM_WORLD, 2); }
}
static void blasCheck(cublasStatus_t s, const char* op) {
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s: cuBLAS error %d\n", op, int(s)); MPI_Abort(MPI_COMM_WORLD, 2); }
}
static void solverCheck(cusolverStatus_t s, const char* op) {
    if (s != CUSOLVER_STATUS_SUCCESS) { fprintf(stderr, "%s: cuSOLVER error %d\n", op, int(s)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

static std::vector<double> makeB(int n) {
    std::vector<double> B(size_t(n) * n);
    unsigned int seed = 42;
    for (double& x : B) x = (rand_r(&seed) / double(RAND_MAX)) - 0.5;
    return B;
}

// Store the owned lower-triangular columns in column-major order. For a
// symmetric matrix the generated values match the original input bitwise.
static void generateColumns(std::vector<double>& A, const std::vector<double>& B,
                            int n, int rank, int ranks) {
    int blocks = (n + BLOCK - 1) / BLOCK;
#pragma omp parallel for schedule(static) if(n >= 256)
    for (int t = 0; t < blocks; ++t) {
        if (rank >= 0 && t % ranks != rank) continue;
        for (int col = t * BLOCK; col < std::min(n, (t + 1) * BLOCK); ++col)
            for (int row = col; row < n; ++row) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k)
                    sum += B[size_t(row) * n + k] * B[size_t(col) * n + k];
                A[size_t(col) * n + row] = sum + (row == col ? n : 0);
            }
    }
}

static bool factor(std::vector<double>& A, int n, int rank, int ranks) {
    double *dA = nullptr, *work = nullptr;
    int* infoDevice = nullptr;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    cudaCheck(cudaMalloc(&dA, sizeof(double) * size_t(n) * n), "allocate matrix");
    cudaCheck(cudaMemcpy(dA, A.data(), sizeof(double) * size_t(n) * n,
                         cudaMemcpyHostToDevice), "copy matrix");
    blasCheck(cublasCreate(&blas), "create cuBLAS");
    solverCheck(cusolverDnCreate(&solver), "create cuSOLVER");
    cudaCheck(cudaMalloc(&infoDevice, sizeof(int)), "allocate status");
    int workSize = 0;
    solverCheck(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                std::min(BLOCK, n), dA, n, &workSize), "query potrf workspace");
    cudaCheck(cudaMalloc(&work, sizeof(double) * size_t(workSize)), "allocate workspace");
    std::vector<double> panel(size_t(n) * std::min(BLOCK, n));
    const double one = 1.0, minusOne = -1.0;
    bool success = true;
    for (int k = 0, tile = 0; k < n; k += BLOCK, ++tile) {
        int width = std::min(BLOCK, n - k), owner = tile % ranks, info = 0;
        if (rank == owner) {
            solverCheck(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width,
                        dA + size_t(k) * n + k, n, work, workSize, infoDevice), "potrf");
            cudaCheck(cudaMemcpy(&info, infoDevice, sizeof(int), cudaMemcpyDeviceToHost),
                      "copy potrf status");
            if (info == 0 && k + width < n)
                blasCheck(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                          CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n - k - width, width,
                          &one, dA + size_t(k) * n + k, n,
                          dA + size_t(k) * n + k + width, n), "panel trsm");
            if (info == 0)
                cudaCheck(cudaMemcpy(panel.data(), dA + size_t(k) * n,
                          sizeof(double) * size_t(n) * width, cudaMemcpyDeviceToHost),
                          "copy panel from GPU");
        }
        MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (info != 0) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + std::max(0, info - 1));
            success = false;
            break;
        }
        MPI_Bcast(panel.data(), n * width, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank != owner)
            cudaCheck(cudaMemcpy(dA + size_t(k) * n, panel.data(),
                      sizeof(double) * size_t(n) * width, cudaMemcpyHostToDevice),
                      "copy broadcast panel to GPU");
        int trailing = k + width;
        for (int t = tile + 1; t * BLOCK < n; ++t) {
            if (t % ranks != rank) continue;
            int col = t * BLOCK, cols = std::min(BLOCK, n - col);
            // Active A(:,J) -= L(:,K) * L(J,K)^T.
            blasCheck(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                      n - trailing, cols, width, &minusOne,
                      dA + size_t(k) * n + trailing, n,
                      dA + size_t(k) * n + col, n, &one,
                      dA + size_t(col) * n + trailing, n), "trailing GEMM");
        }
    }
    if (success)
        for (int t = rank; t * BLOCK < n; t += ranks) {
            int col = t * BLOCK, cols = std::min(BLOCK, n - col);
            cudaCheck(cudaMemcpy(A.data() + size_t(col) * n, dA + size_t(col) * n,
                      sizeof(double) * size_t(n) * cols, cudaMemcpyDeviceToHost),
                      "copy result from GPU");
        }
    cudaFree(work);
    cudaFree(infoDevice);
    cublasDestroy(blas);
    cusolverDnDestroy(solver);
    cudaFree(dA);
    return success;
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& original, int n) {
    double maxError = 0.0, relError = 0.0;
#pragma omp parallel for reduction(max:maxError,relError) schedule(static) if(n >= 256)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k <= std::min(i, j); ++k)
                sum += L[size_t(i) * n + k] * L[size_t(j) * n + k];
            double error = std::fabs(sum - original[size_t(i) * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[size_t(i) * n + j]) + 1e-10));
        }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) printf("Validation failed: relative error too large\n");
    return relError <= 1e-6;
}

static void usage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks, localRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "query CUDA devices");
    if (devices == 0) { if (rank == 0) fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(localRank % devices), "select CUDA device");
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            long parsed = std::atol(argv[++i]);
            if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size\n");
                MPI_Finalize(); return 1;
            }
            n = int(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) usage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nGenerating positive definite matrix...\n", validate ? "enabled" : "disabled");
    }
    std::vector<double> B = makeB(n), A(size_t(n) * n, 0.0), original;
    generateColumns(A, B, n, rank, ranks);
    if (rank == 0 && validate) {
        original.resize(size_t(n) * n, 0.0);
        generateColumns(original, B, n, -1, ranks);
#pragma omp parallel for schedule(static) if(n >= 256)
        for (int row = 0; row < n; ++row)
            for (int col = 0; col < row; ++col)
                original[size_t(row) * n + col] = original[size_t(col) * n + row];
    }
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    bool success = factor(A, n, rank, ranks);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize(); return 1;
    }
    // Gather contiguous column blocks; each GPU holds only its own final columns.
    if (rank == 0) {
        for (int t = 0; t * BLOCK < n; ++t) {
            if (t % ranks == 0) continue;
            int col = t * BLOCK;
            MPI_Recv(A.data() + size_t(col) * n, n * std::min(BLOCK, n - col),
                     MPI_DOUBLE, t % ranks, t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        for (int t = rank; t * BLOCK < n; t += ranks) {
            int col = t * BLOCK;
            MPI_Send(A.data() + size_t(col) * n, n * std::min(BLOCK, n - col),
                     MPI_DOUBLE, 0, t, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double seconds = std::chrono::duration<double>(end - start).count();
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f GFLOPS\n", (double(n) * n * n / 3.0) / seconds / 1e9);
        if (printResults || validate) {
            std::vector<double> L(size_t(n) * n, 0.0);
#pragma omp parallel for schedule(static) if(n >= 256)
            for (int row = 0; row < n; ++row)
                for (int col = 0; col <= row; ++col)
                    L[size_t(row) * n + col] = A[size_t(col) * n + row];
            if (printResults) print_results(L, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(L, original, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                MPI_Finalize(); return valid ? 0 : 1;
            }
        }
    }
    MPI_Finalize();
    return 0;
}
