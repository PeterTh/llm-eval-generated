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
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = strtoull(arg, &end, 10);
            if (arg[0] == '-' || *arg == '\0' || *end != '\0' || value == 0 || value > INT_MAX ||
                value > std::numeric_limits<size_t>::max() / value / sizeof(double)) {
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
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    // Ranks on the same node select distinct visible devices when possible.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) { fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&local);

    const size_t base = N / static_cast<size_t>(ranks), extra = N % static_cast<size_t>(ranks);
    const size_t rows = base + (static_cast<size_t>(rank) < extra);
    const size_t first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    std::vector<double> A(rows * N), B(N * N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, first + i, j);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cublasHandle_t handle = nullptr;
    if (rows) {
        cudaCheck(cudaMalloc(&dA, A.size() * sizeof(double)), "allocate A");
        cudaCheck(cudaMalloc(&dB, B.size() * sizeof(double)), "allocate B");
        cudaCheck(cudaMalloc(&dC, A.size() * sizeof(double)), "allocate C");
        cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A");
        cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B");
        blasCheck(cublasCreate(&handle), "create cuBLAS handle");
        // Initialize cuBLAS kernels before measuring the multiplication.
        const double alpha = 1.0, beta = 0.0;
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(rows),
                              static_cast<int>(N), &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                              &beta, dC, static_cast<int>(N)), "warm up matrix multiplication");
        cudaCheck(cudaDeviceSynchronize(), "warm up synchronization");
    }
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows) {
        const double alpha = 1.0, beta = 0.0;
        // Row-major C=A*B is column-major C^T=B^T*A^T.
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(rows),
                              static_cast<int>(N), &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                              &beta, dC, static_cast<int>(N)), "matrix multiplication");
        cudaCheck(cudaDeviceSynchronize(), "matrix multiplication synchronization");
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
    }

    if (validate || printResults) {
        std::vector<double> localC(rows * N);
        if (rows) cudaCheck(cudaMemcpy(localC.data(), dC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C");
        std::vector<double> C;
        if (rank == 0) {
            C.resize(N * N);
            std::copy(localC.begin(), localC.end(), C.begin());
            for (int src = 1; src < ranks; ++src) {
                size_t srcRows = base + (static_cast<size_t>(src) < extra);
                size_t offset = (static_cast<size_t>(src) * base + std::min(static_cast<size_t>(src), extra)) * N;
                size_t count = srcRows * N;
                while (count) {
                    int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(C.data() + offset, chunk, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += chunk; count -= chunk;
                }
            }
            if (printResults) print_results(C, "MatrixC");
            if (validate) {
                printf("Validating result...\n");
                bool valid = true;
                for (size_t pi = 0; pi < 5 && valid; ++pi) for (size_t pj = 0; pj < 5 && valid; ++pj) {
                    size_t i = pi % N, j = pj % N;
                    double expected = 0;
                    for (size_t k = 0; k < N; ++k) expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    double actual = C[i * N + j];
                    double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (error > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, error);
                        valid = false;
                    }
                }
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                if (!valid) { MPI_Abort(MPI_COMM_WORLD, 1); }
            }
        } else {
            size_t offset = 0, count = localC.size();
            while (count) {
                int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                MPI_Send(localC.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                offset += chunk; count -= chunk;
            }
        }
    }
    if (rows) {
        blasCheck(cublasDestroy(handle), "destroy cuBLAS handle");
        cudaCheck(cudaFree(dA), "free A"); cudaCheck(cudaFree(dB), "free B"); cudaCheck(cudaFree(dC), "free C");
    }
    MPI_Finalize();
    return 0;
}
