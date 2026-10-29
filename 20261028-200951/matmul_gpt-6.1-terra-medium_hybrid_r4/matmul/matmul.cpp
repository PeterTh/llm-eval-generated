#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void fail(const char* message, int rank) {
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) fail(cudaGetErrorString(e), rank); } while (0)
#define BLAS_CHECK(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) fail("cuBLAS operation failed", rank); } while (0)

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not support MPI_THREAD_FUNNELED", rank);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / parsed ||
                parsed * parsed > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize(); return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) fail("No CUDA device available", rank);
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);

    const size_t rows = N / ranks + (static_cast<size_t>(rank) < N % ranks);
    const size_t first = (N / ranks) * rank + std::min(static_cast<size_t>(rank), N % ranks);
    if (!rank) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(N * N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, first + i, j);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cublasHandle_t handle;
    BLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (rows) {
        CUDA_CHECK(cudaMalloc(&dA, rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, rows * N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dA, A.data(), rows * N * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (!rank) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows) {
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A * B is column-major C^T = B^T * A^T.
        BLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N),
                               static_cast<int>(rows), static_cast<int>(N), &alpha, dB,
                               static_cast<int>(N), dA, static_cast<int>(N), &beta, dC, static_cast<int>(N)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
    }
    std::vector<double> C;
    if (validate || printResults) {
        std::vector<double> localC(rows * N);
        if (rows) CUDA_CHECK(cudaMemcpy(localC.data(), dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost));
        if (rank == 0) {
            C.resize(N * N);
            std::copy(localC.begin(), localC.end(), C.begin());
            for (int source = 1; source < ranks; ++source) {
                size_t sourceRows = N / ranks + (static_cast<size_t>(source) < N % ranks);
                size_t offset = ((N / ranks) * source + std::min(static_cast<size_t>(source), N % ranks)) * N;
                size_t count = sourceRows * N;
                while (count) {
                    int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(C.data() + offset, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += chunk; count -= chunk;
                }
            }
        } else {
            size_t offset = 0, count = rows * N;
            while (count) {
                int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                MPI_Send(localC.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                offset += chunk; count -= chunk;
            }
        }
    }
    int valid = 1;
    if (!rank) {
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            for (size_t pi = 0; pi < 5 && valid; ++pi) for (size_t pj = 0; pj < 5 && valid; ++pj) {
                size_t i = pi % N, j = pj % N;
                double expected = 0;
                for (size_t k = 0; k < N; ++k) expected += getPseudoRndValue(N, i, k) * B[k * N + j];
                double actual = C[i * N + j];
                double error = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(error) || error > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, error);
                    valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (dA) CUDA_CHECK(cudaFree(dA));
    if (dC) CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dB));
    BLAS_CHECK(cublasDestroy(handle));
    MPI_Finalize();
    return valid ? 0 : 1;
}
