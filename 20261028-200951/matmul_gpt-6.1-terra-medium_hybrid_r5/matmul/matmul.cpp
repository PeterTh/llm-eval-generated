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
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) fail(cudaGetErrorString(e)); } while (0)
#define CUBLAS_CHECK(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) fail("cuBLAS call failed: " #call); } while (0)

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not support MPI_THREAD_FUNNELED");
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long n = strtoull(value, &end, 10);
            if (errno || *value == '-' || *value == '\0' || *end || n == 0 || n > INT_MAX || n > std::numeric_limits<size_t>::max() / n || n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize(); return 1;
            }
            N = static_cast<size_t>(n);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) fail("No CUDA device available");
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    const size_t rows = N / ranks + (static_cast<size_t>(rank) < N % ranks);
    const size_t first = (N / ranks) * rank + std::min(static_cast<size_t>(rank), N % ranks);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    try {
        std::vector<double> A(rows * N), B(N * N), C(rows * N);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                B[i * N + j] = getPseudoRndValue(N, i, j);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < N; ++j)
                A[i * N + j] = getPseudoRndValue(N, first + i, j);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t handle = nullptr;
        if (rows) {
            CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUBLAS_CHECK(cublasCreate(&handle));
            // Initialize cuBLAS kernels before timing the multiplication.
            const double alpha = 1.0, beta = 0.0;
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                     &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                     &beta, dC, static_cast<int>(N)));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        if (rows) {
            const double alpha = 1.0, beta = 0.0;
            // Row-major C = A * B is column-major C^T = B^T * A^T.
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                     &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                     &beta, dC, static_cast<int>(N)));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        double elapsed = MPI_Wtime() - start, maxElapsed = 0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
            printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
        }
        if (rows && (validate || printResults))
            CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(double), cudaMemcpyDeviceToHost));
        if (rows) {
            CUBLAS_CHECK(cublasDestroy(handle));
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
        if (printResults) {
            std::vector<double> all;
            if (rank == 0) { all.resize(N * N); std::copy(C.begin(), C.end(), all.begin()); }
            // Chunk messages to avoid MPI's int count limit.
            for (int source = 1; source < ranks; ++source) {
                const size_t sourceRows = N / ranks + (static_cast<size_t>(source) < N % ranks);
                const size_t offset = ((N / ranks) * source + std::min(static_cast<size_t>(source), N % ranks)) * N;
                const size_t count = sourceRows * N;
                for (size_t pos = 0; pos < count; ) {
                    int chunk = static_cast<int>(std::min(count - pos, static_cast<size_t>(INT_MAX)));
                    if (rank == source) MPI_Send(C.data() + pos, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    if (rank == 0) MPI_Recv(all.data() + offset + pos, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += chunk;
                }
            }
            if (rank == 0) print_results(all, "MatrixC");
        }
        int valid = 1;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            #pragma omp parallel for reduction(&:valid) schedule(static)
            for (int p = 0; p < 25; ++p) {
                size_t i = static_cast<size_t>(p / 5) % N, j = static_cast<size_t>(p % 5) % N;
                if (i >= first && i < first + rows) {
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k) expected += A[(i - first) * N + k] * B[k * N + j];
                    double actual = C[(i - first) * N + j];
                    double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (!(error <= 1e-6)) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, error);
                        valid = 0;
                    }
                }
            }
            int globalValid;
            MPI_Allreduce(&valid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            valid = globalValid;
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::exception& e) { fail(e.what()); }
}
