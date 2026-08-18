#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void cublasCheck(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Rank %d: cuBLAS %s failed (status %d)\n", rank, operation,
                     static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static bool validateResult(const std::vector<double>& B, const std::vector<double>& C, size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Matrix size N (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > std::numeric_limits<size_t>::max() / N || N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "Matrix size must be non-zero and fit MPI count limits.\n");
        MPI_Finalize(); return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "device discovery", rank);
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device found.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "device selection", rank);
    MPI_Comm_free(&localComm);

    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t remainder = N % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder);
    const size_t rowStart = baseRows * rank + (static_cast<size_t>(rank) < remainder ? rank : remainder);
    const size_t localElements = localRows * N;
    const size_t fullElements = N * N;

    if (!rank) {
        std::printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\nMatrix size: %zu x %zu\nMPI ranks: %d\nValidation: %s\nInitializing matrices...\n",
                    N, N, ranks, validate ? "enabled" : "disabled");
    }
    std::vector<double> localA(localElements), B(fullElements), localC(localElements);
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < static_cast<long long>(localElements); ++idx) {
        const size_t localRow = static_cast<size_t>(idx) / N;
        const size_t col = static_cast<size_t>(idx) % N;
        localA[static_cast<size_t>(idx)] = getPseudoRndValue(N, rowStart + localRow, col);
    }
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < static_cast<long long>(fullElements); ++idx) {
        const size_t row = static_cast<size_t>(idx) / N;
        const size_t col = static_cast<size_t>(idx) % N;
        B[static_cast<size_t>(idx)] = getPseudoRndValue(N, row, col);
    }

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t allocationElements = localElements ? localElements : 1;
    cudaCheck(cudaMalloc(&dA, allocationElements * sizeof(double)), "allocation of A", rank);
    cudaCheck(cudaMalloc(&dB, fullElements * sizeof(double)), "allocation of B", rank);
    cudaCheck(cudaMalloc(&dC, allocationElements * sizeof(double)), "allocation of C", rank);
    if (localElements) cudaCheck(cudaMemcpy(dA, localA.data(), localElements * sizeof(double), cudaMemcpyHostToDevice), "copy of A", rank);
    cudaCheck(cudaMemcpy(dB, B.data(), fullElements * sizeof(double), cudaMemcpyHostToDevice), "copy of B", rank);

    cublasHandle_t cublas;
    cublasCheck(cublasCreate(&cublas), "handle creation", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) std::printf("Computing matrix multiplication...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    if (localRows) {
        // Row-major C=A*B is column-major C^T=B^T*A^T; this ordering avoids
        // transpose buffers and dispatches the vendor-tuned DGEMM kernel.
        const double alpha = 1.0, beta = 0.0;
        cublasCheck(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(N), static_cast<int>(localRows), static_cast<int>(N),
                                &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                &beta, dC, static_cast<int>(N)), "DGEMM", rank);
    }
    cudaCheck(cudaDeviceSynchronize(), "kernel synchronization", rank);
    const auto end = std::chrono::high_resolution_clock::now();
    const double localElapsed = std::chrono::duration<double>(end - start).count();
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localElements) cudaCheck(cudaMemcpy(localC.data(), dC, localElements * sizeof(double), cudaMemcpyDeviceToHost), "copy of C", rank);

    std::vector<int> counts, displacements;
    std::vector<double> C;
    if (!rank) {
        counts.resize(ranks); displacements.resize(ranks); C.resize(fullElements);
        for (int r = 0; r < ranks; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < remainder);
            counts[r] = static_cast<int>(rows * N);
            displacements[r] = r ? displacements[r - 1] + counts[r - 1] : 0;
        }
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE, rank ? nullptr : C.data(),
                rank ? nullptr : counts.data(), rank ? nullptr : displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cublasDestroy(cublas);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    int valid = 1;
    if (!rank && validate) valid = validateResult(B, C, N) ? 1 : 0;
    if (!rank) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", elapsed * 1000.0,
                    (2.0 * N * N * N) / elapsed / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
