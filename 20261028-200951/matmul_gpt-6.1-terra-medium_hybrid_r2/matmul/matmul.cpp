#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}
static size_t rowStart(size_t n, int ranks, int rank) {
    return (n / ranks) * rank + std::min(n % ranks, static_cast<size_t>(rank));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long value = strtoull(arg, &end, 10);
            if (errno || *arg == '-' || *arg == '\0' || *end || value == 0 || value > INT_MAX || value > std::numeric_limits<size_t>::max() / value / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    const size_t first = rowStart(N, ranks, rank);
    const size_t rows = rowStart(N, ranks, rank + 1) - first;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice");
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nInitializing matrices...\n", N, N, validate ? "enabled" : "disabled");
    }
    try {
        std::vector<double> A(rows * N), B(N * N), C;
        if (validate || printResults) C.resize(rows * N);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, first + i, j);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t handle = nullptr;
        if (rows) {
            cudaCheck(cudaMalloc(&dA, A.size() * sizeof(double)), "cudaMalloc A");
            cudaCheck(cudaMalloc(&dB, B.size() * sizeof(double)), "cudaMalloc B");
            cudaCheck(cudaMalloc(&dC, A.size() * sizeof(double)), "cudaMalloc C");
            blasCheck(cublasCreate(&handle), "cublasCreate");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("Computing matrix multiplication...\n");
        double start = MPI_Wtime();
        if (rows) {
            cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A to GPU");
            cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU");
            const double alpha = 1.0, beta = 0.0;
            // Row-major C = A*B is column-major C^T = B^T*A^T.
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N), &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N), &beta, dC, static_cast<int>(N)), "cublasDgemm");
            if (validate || printResults)
                cudaCheck(cudaMemcpy(C.data(), dC, C.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C from GPU");
            else cudaCheck(cudaDeviceSynchronize(), "matrix multiplication");
        }
        double elapsed = MPI_Wtime() - start, maxElapsed = 0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
            printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
        }
        if (printResults) {
            std::vector<double> result;
            if (rank == 0) {
                result.resize(N * N);
                std::copy(C.begin(), C.end(), result.begin());
                for (int source = 1; source < ranks; ++source) {
                    size_t offset = rowStart(N, ranks, source) * N;
                    size_t count = (rowStart(N, ranks, source + 1) - rowStart(N, ranks, source)) * N;
                    while (count) {
                        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(result.data() + offset, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        offset += chunk; count -= chunk;
                    }
                }
                print_results(result, "MatrixC");
            } else {
                size_t offset = 0;
                while (offset < C.size()) {
                    int chunk = static_cast<int>(std::min(C.size() - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Send(C.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    offset += chunk;
                }
            }
        }
        int valid = 1;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            for (size_t pi = 0; pi < 5; ++pi) {
                size_t i = pi % N;
                if (i < first || i >= first + rows) continue;
                for (size_t pj = 0; pj < 5; ++pj) {
                    size_t j = pj % N;
                    double expected = 0;
                    for (size_t k = 0; k < N; ++k) expected += A[(i - first) * N + k] * B[k * N + j];
                    double actual = C[(i - first) * N + j];
                    double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (!std::isfinite(actual) || error > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, error);
                        valid = 0;
                    }
                }
            }
            int allValid;
            MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            valid = allValid;
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        if (rows) {
            blasCheck(cublasDestroy(handle), "cublasDestroy");
            cudaCheck(cudaFree(dA), "cudaFree A");
            cudaCheck(cudaFree(dB), "cudaFree B");
            cudaCheck(cudaFree(dC), "cudaFree C");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
