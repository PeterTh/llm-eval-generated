#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static int rank_id = 0;

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank_id, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Rank %d: %s failed (cuBLAS status %d)\n", rank_id, operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Keep the unsigned arithmetic of the original initializer, including wraparound.
static inline double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void initMatrix(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// MPI counts are int, so transfer large result blocks in chunks.
static void transferResult(double* data, size_t count, int peer, bool send) {
    while (count != 0) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank_id == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            unsigned long long parsed = strtoull(value, &end, 10);
            if (*value == '-' || *value == '\0' || *end != '\0' || parsed == 0 || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (rank_id == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank_id == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank_id == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const size_t base = N / static_cast<size_t>(ranks);
    const size_t remainder = N % static_cast<size_t>(ranks);
    const size_t rows = base + (static_cast<size_t>(rank_id) < remainder);
    const size_t firstRow = base * static_cast<size_t>(rank_id) + std::min(static_cast<size_t>(rank_id), remainder);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank_id);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    if (rank_id == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(N * N);
    initMatrix(A, N, firstRow, rows);
    initMatrix(B, N, 0, N);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cublasHandle_t handle = nullptr;
    if (rows != 0) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dA), rows * N * sizeof(double)), "cudaMalloc A");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dB), N * N * sizeof(double)), "cudaMalloc B");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dC), rows * N * sizeof(double)), "cudaMalloc C");
        cudaCheck(cudaMemcpy(dA, A.data(), rows * N * sizeof(double), cudaMemcpyHostToDevice), "copy A to GPU");
        cudaCheck(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU");
        blasCheck(cublasCreate(&handle), "cublasCreate");
        // Load the selected GEMM kernel and warm up the device before timing.
        const double alpha = 1.0, beta = 0.0;
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                              &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                              &beta, dC, static_cast<int>(N)), "warm-up cublasDgemm");
        cudaCheck(cudaDeviceSynchronize(), "warm-up matrix multiplication");
    }

    if (rank_id == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows != 0) {
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        const double alpha = 1.0, beta = 0.0;
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                              &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                              &beta, dC, static_cast<int>(N)), "cublasDgemm");
        cudaCheck(cudaDeviceSynchronize(), "matrix multiplication");
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank_id == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / duration / 1e9);
    }

    // Validation needs only the first five global rows; result printing needs all rows.
    size_t copyRows = printResults ? rows : (validate && firstRow < 5 ? std::min(rows, size_t(5) - firstRow) : 0);
    std::vector<double> localC(copyRows * N);
    if (copyRows != 0) cudaCheck(cudaMemcpy(localC.data(), dC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C to host");

    if (printResults) {
        if (rank_id == 0) {
            std::vector<double> C(N * N);
            std::copy(localC.begin(), localC.end(), C.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t rRows = base + (static_cast<size_t>(r) < remainder);
                size_t rFirst = base * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), remainder);
                transferResult(C.data() + rFirst * N, rRows * N, r, false);
            }
            print_results(C, "MatrixC");
        } else {
            transferResult(localC.data(), localC.size(), 0, true);
        }
    }

    int valid = 1;
    if (validate) {
        if (rank_id == 0) printf("Validating result...\n");
        for (size_t i = 0; i < std::min(copyRows, size_t(5)); ++i) {
            if (firstRow + i >= 5) break;
            for (size_t pj = 0; pj < 5; ++pj) {
                size_t j = pj % N;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
                double actual = localC[i * N + j];
                double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (!(relError <= 1e-6)) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", firstRow + i, j, expected, actual, relError);
                    valid = 0;
                }
            }
        }
        int allValid = 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        valid = allValid;
        if (rank_id == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    if (rows != 0) {
        blasCheck(cublasDestroy(handle), "cublasDestroy");
        cudaCheck(cudaFree(dA), "cudaFree A");
        cudaCheck(cudaFree(dB), "cudaFree B");
        cudaCheck(cudaFree(dC), "cudaFree C");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
