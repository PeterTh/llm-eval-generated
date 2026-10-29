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

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void fail(const char* message, int rank) {
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) fail(cudaGetErrorString(e), rank); } while (0)
#define BLAS_CHECK(call) do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "Rank %d: cuBLAS error %d at %s:%d\n", rank, static_cast<int>(s), __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
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
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long value = strtoull(arg, &end, 10);
            if (errno || arg[0] == '-' || end == arg || *end || value == 0 || value > INT_MAX ||
                value > std::numeric_limits<size_t>::max() / value ||
                value * value > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(value);
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

    const size_t base = N / static_cast<size_t>(ranks), extra = N % static_cast<size_t>(ranks);
    const auto rowsFor = [&](int r) { return base + (static_cast<size_t>(r) < extra ? 1 : 0); };
    const auto startFor = [&](int r) { return base * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), extra); };
    const size_t rows = rowsFor(rank), first = startFor(rank);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // Local ranks select different visible GPUs; a scheduler may instead expose one GPU per rank.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("No CUDA device available", rank);
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    try {
        std::vector<double> A(rows * N), B(N * N), C(rows * N);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, first + i, j);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t handle = nullptr;
        if (rows) {
            CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
            BLAS_CHECK(cublasCreate(&handle));
        }
        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        if (rows) {
            const double alpha = 1.0, beta = 0.0;
            // Row-major C = A * B is column-major C^T = B^T * A^T.
            BLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                   &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                   &beta, dC, static_cast<int>(N)));
            CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        double elapsed = MPI_Wtime() - start, maxElapsed = 0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
            printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
        }
        if (rows) {
            BLAS_CHECK(cublasDestroy(handle));
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }

        if (printResults) {
            std::vector<double> result;
            if (rank == 0) {
                result.resize(N * N);
                std::copy(C.begin(), C.end(), result.begin());
                for (int r = 1; r < ranks; ++r) {
                    size_t offset = startFor(r) * N, count = rowsFor(r) * N;
                    while (count) {
                        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(result.data() + offset, chunk, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        offset += chunk; count -= chunk;
                    }
                }
                print_results(result, "MatrixC");
            } else {
                size_t offset = 0, count = C.size();
                while (count) {
                    int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                    MPI_Send(C.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    offset += chunk; count -= chunk;
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
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k) expected += A[(i - first) * N + k] * B[k * N + j];
                    double actual = C[(i - first) * N + j];
                    double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (!std::isfinite(error) || error > 1e-6) {
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
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::exception& e) {
        fail(e.what(), rank);
    }
}
