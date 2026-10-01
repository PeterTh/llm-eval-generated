#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, int(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// One independent row solves L(i,k:k+b) * L(k:k+b,k:k+b)^T = A(i,k:k+b).
__global__ void solvePanel(double* a, const double* diag, size_t n,
                           size_t localOffset, size_t pivot, int width, int rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    size_t localRow = size_t(r) + localOffset;
    double* row = a + localRow * n + pivot;
    for (int j = 0; j < width; ++j) {
        double value = row[j];
        for (int t = 0; t < j; ++t) value -= row[t] * diag[j * width + t];
        row[j] = value / diag[j * width + j];
    }
}

// Preserve the original random sequence, then divide independent matrix rows among ranks.
void generateRandomMatrix(std::vector<double>& B) {
    unsigned int seed = 42;
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = (rand_r(&seed) / double(RAND_MAX)) - 0.5;
}
void generatePositiveDefiniteRows(std::vector<double>& A, const std::vector<double>& B,
                                  size_t n, size_t first, size_t localRows) {
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < localRows; ++r) {
        size_t i = first + r;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[r * n + j] = sum + (i == j ? double(n) : 0.0);
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= std::min(i, j); ++k)
                sum += L[i * n + k] * L[j * n + k];
            double error = fabs(sum - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(original[i * n + j]) + 1e-10));
        }
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) printf("Validation failed: relative error too large\n");
    return relError <= 1e-6;
}

static size_t rowBegin(size_t n, int process, int ranks, size_t block) {
    size_t blocks = (n + block - 1) / block;
    return std::min(n, ((blocks * size_t(process)) / size_t(ranks)) * block);
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            unsigned long long parsed = strtoull(argv[++i], &end, 10);
            if (*end || parsed == 0 || parsed > size_t(INT_MAX / std::max(1, ranks))) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            n = size_t(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n > size_t(INT_MAX) / n) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds MPI count limit\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank;
    MPI_Comm_rank(shared, &localRank);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) { fprintf(stderr, "Rank %d: no CUDA device\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&shared);

    const int block = 64;
    const size_t first = rowBegin(n, rank, ranks, block);
    const size_t last = rowBegin(n, rank + 1, ranks, block);
    const size_t localRows = last - first;
    std::vector<int> counts(ranks), displacements(ranks), panelCounts(ranks), panelDisplacements(ranks);
    for (int p = 0; p < ranks; ++p) {
        size_t begin = rowBegin(n, p, ranks, block), end = rowBegin(n, p + 1, ranks, block);
        counts[p] = int((end - begin) * n);
        displacements[p] = int(begin * n);
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\nGenerating positive definite matrix...\n", validate ? "enabled" : "disabled");
    }
    std::vector<double> B(n * n);
    if (rank == 0) generateRandomMatrix(B);
    MPI_Bcast(B.data(), int(B.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> local(localRows * n);
    generatePositiveDefiniteRows(local, B, n, first, localRows);
    B.clear();
    B.shrink_to_fit();
    std::vector<double> A;
    std::vector<double> original;
    if (validate) {
        if (rank == 0) original.resize(n * n);
        MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE,
                    rank == 0 ? original.data() : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double* dA = nullptr;
    double* dPanel = nullptr;
    double* dDiag = nullptr;
    cudaCheck(cudaMalloc(&dA, std::max(size_t(1), localRows * n) * sizeof(double)), "cudaMalloc matrix");
    cudaCheck(cudaMalloc(&dPanel, n * size_t(block) * sizeof(double)), "cudaMalloc panel");
    cudaCheck(cudaMalloc(&dDiag, block * block * sizeof(double)), "cudaMalloc diagonal");
    cudaCheck(cudaMemcpy(dA, local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice), "copy matrix");
    cublasHandle_t blas;
    blasCheck(cublasCreate(&blas), "cublasCreate");
    const double minusOne = -1.0, one = 1.0;
    std::vector<double> diag(block * block), panel(n * size_t(block));
    std::vector<double> localPanel(localRows * size_t(block));

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = true;
    for (size_t pivot = 0; pivot < n; pivot += block) {
        int width = int(std::min(size_t(block), n - pivot));
        int owner = 0;
        while (owner + 1 < ranks && rowBegin(n, owner + 1, ranks, block) <= pivot) ++owner;
        if (rank == owner) {
            cudaCheck(cudaMemcpy2D(diag.data(), width * sizeof(double),
                     dA + (pivot - first) * n + pivot, n * sizeof(double),
                     width * sizeof(double), width, cudaMemcpyDeviceToHost), "copy diagonal");
            for (int j = 0; j < width; ++j) {
                double value = diag[j * width + j];
                for (int t = 0; t < j; ++t) value -= diag[j * width + t] * diag[j * width + t];
                if (!(value > 0.0)) { success = false; break; }
                diag[j * width + j] = sqrt(value);
                for (int i = j + 1; i < width; ++i) {
                    double x = diag[i * width + j];
                    for (int t = 0; t < j; ++t) x -= diag[i * width + t] * diag[j * width + t];
                    diag[i * width + j] = x / diag[j * width + j];
                }
            }
            if (success) cudaCheck(cudaMemcpy2D(dA + (pivot - first) * n + pivot, n * sizeof(double),
                    diag.data(), width * sizeof(double), width * sizeof(double), width,
                    cudaMemcpyHostToDevice), "store diagonal");
        }
        int good = success ? 1 : 0;
        MPI_Bcast(&good, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!good) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", pivot);
            success = false;
            break;
        }
        MPI_Bcast(diag.data(), width * width, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dDiag, diag.data(), width * width * sizeof(double), cudaMemcpyHostToDevice), "copy diagonal factor");
        size_t begin = std::max(first, pivot + width);
        if (begin < last) {
            int rows = int(last - begin);
            solvePanel<<<(rows + 127) / 128, 128>>>(dA, dDiag, n, begin - first, pivot, width, rows);
            cudaCheck(cudaGetLastError(), "solve panel");
        }
        // Gather each process's panel rows in global row order.
        for (int p = 0; p < ranks; ++p) {
            size_t b = rowBegin(n, p, ranks, block), e = rowBegin(n, p + 1, ranks, block);
            panelCounts[p] = int((e - b) * size_t(width));
            panelDisplacements[p] = int(b * size_t(width));
        }
        if (localRows) cudaCheck(cudaMemcpy2D(localPanel.data(), width * sizeof(double),
                dA + pivot, n * sizeof(double), width * sizeof(double), localRows,
                cudaMemcpyDeviceToHost), "copy local panel");
        MPI_Allgatherv(localPanel.data(), panelCounts[rank], MPI_DOUBLE,
                       panel.data(), panelCounts.data(), panelDisplacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dPanel, panel.data(), n * size_t(width) * sizeof(double), cudaMemcpyHostToDevice), "copy gathered panel");
        if (begin < last && pivot + width < n) {
            // Row-major C -= P_local * P_all^T is column-major C^T -= P_all * P_local^T.
            int m = int(last - begin), q = int(n - pivot - width);
            blasCheck(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, q, m, width,
                      &minusOne, dPanel + (pivot + width) * size_t(width), width,
                      dPanel + begin * size_t(width), width,
                      &one, dA + (begin - first) * n + pivot + width, int(n)), "trailing update");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "factorization sync");
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (success) {
        cudaCheck(cudaMemcpy(local.data(), dA, local.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy result");
        // The original output is a strictly lower triangular matrix.
        for (size_t i = 0; i < localRows; ++i)
            for (size_t j = first + i + 1; j < n; ++j) local[i * n + j] = 0.0;
        if (rank == 0) A.resize(n * n);
        MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE, rank == 0 ? A.data() : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    blasCheck(cublasDestroy(blas), "cublasDestroy");
    cudaCheck(cudaFree(dA), "cudaFree matrix");
    cudaCheck(cudaFree(dPanel), "cudaFree panel");
    cudaCheck(cudaFree(dDiag), "cudaFree diagonal");
    if (rank == 0) {
        if (!success) printf("Cholesky decomposition failed\n");
        else {
            printf("Computation time: %ld ms\n", long(maxElapsed * 1000));
            printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * maxElapsed));
            if (printResults) print_results(A, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(A, original, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                success = valid;
            }
        }
    }
    int result = success ? 0 : 1;
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
