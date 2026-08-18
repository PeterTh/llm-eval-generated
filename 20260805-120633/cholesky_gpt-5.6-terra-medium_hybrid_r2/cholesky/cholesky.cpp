#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// The matrix is distributed by contiguous, block-aligned row ranges.  A
// right-looking blocked algorithm keeps each rank's rows on its GPU.  MPI
// distributes completed panels; cuSOLVER/cublas perform the dense work.
constexpr int kBlockSize = 256;

static void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
static void checkBlas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + " failed");
}
static void checkSolver(cusolverStatus_t status, const char* what) {
    if (status != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + " failed");
}

__global__ void zeroUpper(double* a, int rows, int n, int globalFirstRow) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < n && col > globalFirstRow + row) a[static_cast<size_t>(row) * n + col] = 0.0;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, int n) {
    std::vector<double> b(static_cast<size_t>(n) * n);
    unsigned int seed = 42; // Retain the benchmark's deterministic input.
    for (size_t i = 0; i < b.size(); ++i) b[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;

    // Each output element is independent, making input construction a useful
    // OpenMP phase without affecting the deterministic B matrix above.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; ++k) sum += b[static_cast<size_t>(i) * n + k] * b[static_cast<size_t>(j) * n + k];
            a[static_cast<size_t>(i) * n + j] = sum + (i == j ? n : 0.0);
        }
    }
}

static bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, int n) {
    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k <= std::min(i, j); ++k) sum += l[static_cast<size_t>(i) * n + k] * l[static_cast<size_t>(j) * n + k];
            const double error = std::fabs(sum - original[static_cast<size_t>(i) * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[static_cast<size_t>(i) * n + j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    return relError <= 1e-6;
}

static void printUsage(const char* prog) {
    std::printf("Usage: %s [options]\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", prog);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int exitCode = 0;
    try {
        int n = 512;
        bool validate = false, printResults = false;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "-v")) validate = true;
            else if (!std::strcmp(argv[i], "-r")) printResults = true;
            else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
            else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
        }
        if (n <= 0 || static_cast<long long>(n) * n > std::numeric_limits<int>::max())
            throw std::runtime_error("matrix size must be positive and fit MPI's count type");

        // Contiguous ownership consists of whole panels, avoiding any panel
        // handoff during factorization and balancing the number of panels.
        const int panels = (n + kBlockSize - 1) / kBlockSize;
        const int basePanels = panels / ranks, extraPanels = panels % ranks;
        std::vector<int> rowCounts(ranks), rowDispls(ranks);
        int nextRow = 0;
        for (int r = 0; r < ranks; ++r) {
            const int ownedPanels = basePanels + (r < extraPanels ? 1 : 0);
            rowDispls[r] = nextRow;
            rowCounts[r] = std::min(n, nextRow + ownedPanels * kBlockSize) - nextRow;
            nextRow += rowCounts[r];
        }
        const int localRows = rowCounts[rank], firstRow = rowDispls[rank];
        std::vector<int> elementCounts(ranks), elementDispls(ranks);
        for (int r = 0; r < ranks; ++r) { elementCounts[r] = rowCounts[r] * n; elementDispls[r] = rowDispls[r] * n; }

        if (!rank) {
            std::printf("Cholesky Decomposition Benchmark (MPI + OpenMP + CUDA)\nMatrix size: %d x %d\nMPI ranks: %d, CUDA block size: %d\nValidation: %s\nGenerating positive definite matrix...\n",
                        n, n, ranks, kBlockSize, validate ? "enabled" : "disabled");
        }
        std::vector<double> fullA, original;
        if (!rank) { fullA.resize(static_cast<size_t>(n) * n); generatePositiveDefiniteMatrix(fullA, n); if (validate) original = fullA; }
        std::vector<double> localA(static_cast<size_t>(localRows) * n);
        MPI_Scatterv(rank ? nullptr : fullA.data(), elementCounts.data(), elementDispls.data(), MPI_DOUBLE,
                     localA.data(), localRows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        int localRank = 0; MPI_Comm_rank(localComm, &localRank); MPI_Comm_free(&localComm);
        int devices = 0; checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (!devices) throw std::runtime_error("no CUDA device available");
        checkCuda(cudaSetDevice(localRank % devices), "cudaSetDevice");
        cublasHandle_t blas; cusolverDnHandle_t solver;
        checkBlas(cublasCreate(&blas), "cublasCreate"); checkSolver(cusolverDnCreate(&solver), "cusolverDnCreate");
        double *dA = nullptr, *dPanel = nullptr, *dDiag = nullptr;
        checkCuda(cudaMalloc(&dA, localA.size() * sizeof(double)), "cudaMalloc matrix");
        checkCuda(cudaMemcpy(dA, localA.data(), localA.size() * sizeof(double), cudaMemcpyHostToDevice), "upload matrix");
        checkCuda(cudaMalloc(&dDiag, static_cast<size_t>(kBlockSize) * kBlockSize * sizeof(double)), "cudaMalloc diagonal");

        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        bool success = true;
        for (int k = 0; k < n; k += kBlockSize) {
            const int b = std::min(kBlockSize, n - k);
            int owner = 0;
            while (owner + 1 < ranks && k >= rowDispls[owner] + rowCounts[owner]) ++owner;
            if (rank == owner) {
                const int localK = k - firstRow;
                int workSize = 0, *dInfo = nullptr; double* work = nullptr;
                checkSolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, b, dA + static_cast<size_t>(localK) * n + k, n, &workSize), "potrf workspace");
                checkCuda(cudaMalloc(&work, static_cast<size_t>(workSize) * sizeof(double)), "cudaMalloc potrf workspace");
                checkCuda(cudaMalloc(&dInfo, sizeof(int)), "cudaMalloc potrf info");
                checkSolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, b, dA + static_cast<size_t>(localK) * n + k, n, work, workSize, dInfo), "cusolver potrf");
                int info = 0; checkCuda(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost), "download potrf info");
                success = info == 0;
                if (success) checkCuda(cudaMemcpy2D(dDiag, b * sizeof(double), dA + static_cast<size_t>(localK) * n + k, n * sizeof(double), b * sizeof(double), b, cudaMemcpyDeviceToDevice), "pack diagonal");
                cudaFree(work); cudaFree(dInfo);
            }
            int ok = success ? 1 : 0; MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
            if (!ok) { success = false; break; }
            // Portable MPI stacks may not be CUDA-aware, so synchronize a host copy for the panel factor.
            std::vector<double> diagonal(static_cast<size_t>(b) * b);
            if (rank == owner) checkCuda(cudaMemcpy(diagonal.data(), dDiag, diagonal.size() * sizeof(double), cudaMemcpyDeviceToHost), "download diagonal");
            MPI_Bcast(diagonal.data(), b * b, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(dDiag, diagonal.data(), diagonal.size() * sizeof(double), cudaMemcpyHostToDevice), "upload diagonal");

            const int solveFirst = std::max(firstRow, k + b), solveLast = firstRow + localRows;
            if (solveLast > solveFirst) {
                const int m = solveLast - solveFirst, localOffset = solveFirst - firstRow;
                const double one = 1.0;
                // Row-major B is viewed as B^T.  The diagonal is likewise
                // transposed in memory, hence op(T) exposes lower L.
                checkBlas(cublasDtrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                       b, m, &one, dDiag, b,
                                       dA + static_cast<size_t>(localOffset) * n + k, n), "cublas trsm");
            }
            const int tail = n - (k + b);
            if (!tail) continue;
            std::vector<double> panel(static_cast<size_t>(tail) * b);
            const int sendRows = std::max(0, std::min(firstRow + localRows, n) - std::max(firstRow, k + b));
            std::vector<double> send(static_cast<size_t>(sendRows) * b);
            if (sendRows) {
                const int offset = std::max(firstRow, k + b) - firstRow;
                checkCuda(cudaMemcpy2D(send.data(), b * sizeof(double), dA + static_cast<size_t>(offset) * n + k, n * sizeof(double), b * sizeof(double), sendRows, cudaMemcpyDeviceToHost), "download panel rows");
            }
            std::vector<int> panelCounts(ranks), panelDispls(ranks);
            for (int r = 0; r < ranks; ++r) {
                const int begin = std::max(rowDispls[r], k + b), end = rowDispls[r] + rowCounts[r];
                panelCounts[r] = std::max(0, end - begin) * b;
                panelDispls[r] = std::max(0, begin - (k + b)) * b;
            }
            MPI_Allgatherv(send.data(), sendRows * b, MPI_DOUBLE, panel.data(), panelCounts.data(), panelDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            cudaFree(dPanel); dPanel = nullptr;
            checkCuda(cudaMalloc(&dPanel, panel.size() * sizeof(double)), "cudaMalloc panel");
            checkCuda(cudaMemcpy(dPanel, panel.data(), panel.size() * sizeof(double), cudaMemcpyHostToDevice), "upload panel");
            if (solveLast > solveFirst) {
                const int m = solveLast - solveFirst, localOffset = solveFirst - firstRow;
                const double minusOne = -1.0, one = 1.0;
                // C(row-major)^T = panel^T * local-panel, so this updates the
                // trailing rectangle in one high-throughput GEMM.
                checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, tail, m, b, &minusOne,
                                      dPanel, b, dA + static_cast<size_t>(localOffset) * n + k, n,
                                      &one, dA + static_cast<size_t>(localOffset) * n + k + b, n), "cublas gemm");
            }
        }
        checkCuda(cudaDeviceSynchronize(), "CUDA synchronization");
        const double localSeconds = MPI_Wtime() - start;
        double seconds = 0.0; MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (success) {
            if (localRows) {
                const dim3 threads(32, 8), blocks((n + threads.x - 1) / threads.x, (localRows + threads.y - 1) / threads.y);
                zeroUpper<<<blocks, threads>>>(dA, localRows, n, firstRow);
                checkCuda(cudaGetLastError(), "zero upper kernel");
                checkCuda(cudaMemcpy(localA.data(), dA, localA.size() * sizeof(double), cudaMemcpyDeviceToHost), "download result");
            }
        }
        cudaFree(dPanel); cudaFree(dDiag); cudaFree(dA); cublasDestroy(blas); cusolverDnDestroy(solver);
        if (!success) { if (!rank) std::printf("Cholesky decomposition failed: matrix is not positive definite\n"); exitCode = 1; }
        else {
            std::vector<double> result;
            if (!rank) result.resize(static_cast<size_t>(n) * n);
            MPI_Gatherv(localA.data(), localRows * n, MPI_DOUBLE, rank ? nullptr : result.data(), elementCounts.data(), elementDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (!rank) {
                std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", seconds * 1000.0, (static_cast<double>(n) * n * n / 3.0) / seconds / 1e9);
                if (printResults) print_results(result, "CholeskyL");
                if (validate) { const bool valid = validateCholesky(result, original, n); std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); exitCode = valid ? 0 : 1; }
            }
            MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Rank %d error: %s\n", rank, e.what());
        exitCode = 1;
    }
    MPI_Finalize();
    return exitCode;
}
