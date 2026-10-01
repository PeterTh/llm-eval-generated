#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

[[noreturn]] void fail(int rank, const char* operation, int code) {
    std::fprintf(stderr, "Rank %d: %s failed (%d)\n", rank, operation, code);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status, int rank, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void checkCublas(cublasStatus_t status, int rank, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) fail(rank, operation, static_cast<int>(status));
}

void initMatrix(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rows; ++row)
        for (size_t j = 0; j < N; ++j)
            mat[row * N + j] = getPseudoRndValue(N, firstRow + row, j);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N, size_t firstRow,
                    size_t rows) {
    int failures = 0;
#pragma omp parallel for collapse(2) reduction(+:failures) schedule(static)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            size_t i = static_cast<size_t>(pi) % N;
            size_t j = static_cast<size_t>(pj) % N;
            if (i < firstRow || i - firstRow >= rows) continue;
            size_t localRow = i - firstRow;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[localRow * N + k] * B[k * N + j];
            double actual = C[localRow * N + j];
            double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!(relError <= 1e-6)) {
#pragma omp critical
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                ++failures;
            }
        }
    }
    return failures == 0;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// MPI counts are int; transfer the optional complete result in bounded chunks.
void gatherResult(const std::vector<double>& local, std::vector<double>& complete,
                  size_t N, int rank, int worldSize) {
    constexpr size_t chunk = 1u << 24;
    if (rank == 0) std::copy(local.begin(), local.end(), complete.begin());
    for (int peer = 1; peer < worldSize; ++peer) {
        size_t first = (N / static_cast<size_t>(worldSize)) * peer +
                       std::min(static_cast<size_t>(peer), N % static_cast<size_t>(worldSize));
        size_t rows = N / static_cast<size_t>(worldSize) +
                      (static_cast<size_t>(peer) < N % static_cast<size_t>(worldSize));
        size_t elements = rows * N;
        for (size_t offset = 0; offset < elements; offset += chunk) {
            int count = static_cast<int>(std::min(chunk, elements - offset));
            if (rank == 0) {
                int rc = MPI_Recv(complete.data() + first * N + offset, count,
                                  MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (rc != MPI_SUCCESS) fail(rank, "MPI_Recv", rc);
            } else if (rank == peer) {
                int rc = MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 0,
                                  MPI_COMM_WORLD);
                if (rc != MPI_SUCCESS) fail(rank, "MPI_Send", rc);
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || value == end || *end || value[0] == '-' || parsed == 0 ||
                parsed > INT_MAX || parsed > std::numeric_limits<size_t>::max() / parsed ||
                parsed * parsed > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "cudaGetDeviceCount");
    if (deviceCount == 0) fail(rank, "no CUDA devices available", 0);
    checkCuda(cudaSetDevice(localRank % deviceCount), rank, "cudaSetDevice");

    size_t quotient = N / static_cast<size_t>(worldSize);
    size_t remainder = N % static_cast<size_t>(worldSize);
    size_t rows = quotient + (static_cast<size_t>(rank) < remainder);
    size_t firstRow = quotient * rank + std::min(static_cast<size_t>(rank), remainder);
    std::vector<double> A(rows * N), B(N * N), C(rows * N);
    initMatrix(A, N, firstRow, rows);
    initMatrix(B, N, 0, N);

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cublasHandle_t handle = nullptr;
    size_t aBytes = A.size() * sizeof(double);
    if (rows != 0) {
        size_t bBytes = B.size() * sizeof(double);
        checkCuda(cudaMalloc(&deviceA, aBytes), rank, "cudaMalloc(A)");
        checkCuda(cudaMalloc(&deviceB, bBytes), rank, "cudaMalloc(B)");
        checkCuda(cudaMalloc(&deviceC, aBytes), rank, "cudaMalloc(C)");
        checkCuda(cudaMemcpy(deviceA, A.data(), aBytes, cudaMemcpyHostToDevice), rank,
                  "copy A to GPU");
        checkCuda(cudaMemcpy(deviceB, B.data(), bBytes, cudaMemcpyHostToDevice), rank,
                  "copy B to GPU");

        checkCublas(cublasCreate(&handle), rank, "cublasCreate");
    }

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows != 0) {
        const double alpha = 1.0, beta = 0.0;
        // Row-major C=A*B is column-major C^T=B^T*A^T in the same buffers.
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(N), static_cast<int>(rows),
                                static_cast<int>(N), &alpha, deviceB,
                                static_cast<int>(N), deviceA, static_cast<int>(N),
                                &beta, deviceC, static_cast<int>(N)),
                    rank, "cublasDgemm");
        checkCuda(cudaDeviceSynchronize(), rank, "cudaDeviceSynchronize");
    }
    double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rows != 0) {
        checkCuda(cudaMemcpy(C.data(), deviceC, aBytes, cudaMemcpyDeviceToHost), rank,
                  "copy C from GPU");
        checkCublas(cublasDestroy(handle), rank, "cublasDestroy");
        checkCuda(cudaFree(deviceA), rank, "cudaFree(A)");
        checkCuda(cudaFree(deviceB), rank, "cudaFree(B)");
        checkCuda(cudaFree(deviceC), rank, "cudaFree(C)");
    }
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        std::printf("Performance: %.3f GFLOPS\n",
                    seconds > 0.0 ? 2.0 * static_cast<double>(N) * N * N / seconds / 1e9 : 0.0);
    }

    if (printResults) {
        std::vector<double> complete;
        if (rank == 0) complete.resize(N * N);
        gatherResult(C, complete, N, rank, worldSize);
        if (rank == 0) print_results(complete, "MatrixC");
    }
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        int localValid = validateResult(A, B, C, N, firstRow, rows) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    MPI_Finalize();
    return 0;
}
