#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void checkCuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void checkCublas(const cublasStatus_t status, const char* const operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error in %s: %d\n", operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for collapse(2) schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        for (long long j = 0; j < static_cast<long long>(N); ++j) {
            mat[static_cast<size_t>(i) * N + static_cast<size_t>(j)] =
                getPseudoRndValue(N, static_cast<size_t>(i), static_cast<size_t>(j));
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[i * N + k] * B[k * N + j];
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

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Matrix size N (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false, printResults = false;
    bool badArguments = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else badArguments = true;
    }
    if (badArguments || N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0) { std::printf("Invalid matrix size or option\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }

    // Assign a GPU locally on each node, independent of global MPI rank placement.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    std::vector<int> counts(worldSize), displacements(worldSize);
    size_t offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        counts[r] = static_cast<int>(rows * N);
        displacements[r] = static_cast<int>(offset * N);
        offset += rows;
    }

    std::vector<double> A, B(N * N), C;
    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nInitializing matrices...\n",
                    N, N, validate ? "enabled" : "disabled");
        A.resize(N * N); C.resize(N * N);
        initMatrix(A, N); initMatrix(B, N);
    }
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> localA(localRows * N), localC(localRows * N);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 localA.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    // A process may own no rows when more ranks than rows are requested.
    // Keep valid device pointers while simply skipping its zero-sized GEMM.
    checkCuda(cudaMalloc(&dA, std::max<size_t>(1, localA.size()) * sizeof(double)), "cudaMalloc(A)");
    checkCuda(cudaMalloc(&dB, B.size() * sizeof(double)), "cudaMalloc(B)");
    checkCuda(cudaMalloc(&dC, std::max<size_t>(1, localC.size()) * sizeof(double)), "cudaMalloc(C)");
    checkCuda(cudaMemcpy(dA, localA.data(), localA.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B");
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    const double alpha = 1.0, beta = 0.0;
    // cuBLAS is column-major: B^T * A_local^T produces (A_local * B)^T in dC.
    if (localRows != 0)
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N),
                                static_cast<int>(localRows), static_cast<int>(N), &alpha, dB,
                                static_cast<int>(N), dA, static_cast<int>(N), &beta, dC,
                                static_cast<int>(N)), "cublasDgemm");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count(), seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(localC.data(), dC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C");
    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    checkCublas(cublasDestroy(handle), "cublasDestroy");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    int result = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", seconds * 1000.0,
                    (2.0 * N * N * N) / seconds / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) { std::printf("Validating result...\n"); result = validateResult(A, B, C, N) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED"); }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
