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
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void blasCheck(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Rank %d: %s failed (cuBLAS status %d)\n", rank, operation, int(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static bool validateResult(const std::vector<double>& C, size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = strtoull(argv[++i], &end, 10);
            if (!end || *end || value == 0 || value > static_cast<unsigned long long>(INT_MAX) || argv[i][0] == '-') {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (N > std::numeric_limits<size_t>::max() / N || N * N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Matrix is too large for MPI_Gatherv counts\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);

    const size_t firstRow = N * static_cast<size_t>(rank) / static_cast<size_t>(ranks);
    const size_t lastRow = N * (static_cast<size_t>(rank) + 1) / static_cast<size_t>(ranks);
    const size_t rows = lastRow - firstRow;
    const size_t localElements = rows * N;
    const size_t allElements = N * N;
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(localElements), B(allElements), localC(localElements);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            B[i * N + j] = getPseudoRndValue(N, i, j);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i * N + j] = getPseudoRndValue(N, firstRow + i, j);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    const double start = MPI_Wtime();
    if (rows != 0) {
        double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
        cudaCheck(cudaMalloc(&deviceA, localElements * sizeof(double)), "cudaMalloc A", rank);
        cudaCheck(cudaMalloc(&deviceB, allElements * sizeof(double)), "cudaMalloc B", rank);
        cudaCheck(cudaMalloc(&deviceC, localElements * sizeof(double)), "cudaMalloc C", rank);
        cudaCheck(cudaMemcpy(deviceA, A.data(), localElements * sizeof(double), cudaMemcpyHostToDevice), "copy A", rank);
        cudaCheck(cudaMemcpy(deviceB, B.data(), allElements * sizeof(double), cudaMemcpyHostToDevice), "copy B", rank);
        cublasHandle_t handle;
        blasCheck(cublasCreate(&handle), "cublasCreate", rank);
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A B is column-major C^T = B^T A^T.
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                              &alpha, deviceB, static_cast<int>(N), deviceA, static_cast<int>(N),
                              &beta, deviceC, static_cast<int>(N)), "cublasDgemm", rank);
        cudaCheck(cudaMemcpy(localC.data(), deviceC, localElements * sizeof(double), cudaMemcpyDeviceToHost), "copy C", rank);
        blasCheck(cublasDestroy(handle), "cublasDestroy", rank);
        cudaCheck(cudaFree(deviceA), "cudaFree A", rank);
        cudaCheck(cudaFree(deviceB), "cudaFree B", rank);
        cudaCheck(cudaFree(deviceC), "cudaFree C", rank);
    }

    std::vector<double> C;
    std::vector<int> counts, displacements;
    if (rank == 0) {
        C.resize(allElements);
        counts.resize(ranks);
        displacements.resize(ranks);
        for (int p = 0; p < ranks; ++p) {
            const size_t begin = N * static_cast<size_t>(p) / static_cast<size_t>(ranks);
            const size_t end = N * (static_cast<size_t>(p) + 1) / static_cast<size_t>(ranks);
            counts[p] = static_cast<int>((end - begin) * N);
            displacements[p] = static_cast<int>(begin * N);
        }
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int success = 1;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            success = validateResult(C, N) ? 1 : 0;
            printf("Validation: %s\n", success ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return success ? 0 : 1;
}
