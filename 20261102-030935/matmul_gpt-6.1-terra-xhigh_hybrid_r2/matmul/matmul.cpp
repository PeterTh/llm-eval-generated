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

namespace {
int rank_id = 0;

[[noreturn]] void fail(const char* operation, const char* detail) {
    fprintf(stderr, "Rank %d: %s: %s\n", rank_id, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) fail(operation, cudaGetErrorString(status));
}

void checkBlas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[64];
        snprintf(message, sizeof(message), "cuBLAS error %d", static_cast<int>(status));
        fail(operation, message);
    }
}

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// MPI counts are int, so transfer large result matrices in chunks.
void transferResult(double* data, size_t count, int peer, bool send) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        const int status = send ? MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD)
                                : MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (status != MPI_SUCCESS) fail("MPI result transfer", "communication failed");
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI initialization", "MPI_THREAD_FUNNELED is not supported");

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno || *arg == '-' || *arg == '\0' || *end != '\0' || value == 0 || value > INT_MAX ||
                value > std::numeric_limits<size_t>::max() / value / sizeof(double)) {
                if (rank_id == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
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
    if (rank_id == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
        fflush(stdout);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    std::vector<double> A, B, C;
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cublasHandle_t handle = nullptr;
    if (rows != 0) {
        int devices = 0;
        checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices == 0) fail("CUDA initialization", "no CUDA device available");
        checkCuda(cudaSetDevice(localRank % devices), "cudaSetDevice");
        A.resize(rows * N);
        B.resize(N * N);
        if (validate || printResults) C.resize(rows * N);
        initMatrix(A, N, firstRow, rows);
        initMatrix(B, N, 0, N);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), rows * N * sizeof(double)), "cudaMalloc A");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), N * N * sizeof(double)), "cudaMalloc B");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), rows * N * sizeof(double)), "cudaMalloc C");
        checkBlas(cublasCreate(&handle), "cublasCreate");
    }

    if (rank_id == 0) {
        printf("Computing matrix multiplication...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows != 0) {
        checkCuda(cudaMemcpy(deviceA, A.data(), rows * N * sizeof(double), cudaMemcpyHostToDevice), "copy A to GPU");
        checkCuda(cudaMemcpy(deviceB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU");
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A * B is column-major C^T = B^T * A^T.
        checkBlas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                              &alpha, deviceB, static_cast<int>(N), deviceA, static_cast<int>(N),
                              &beta, deviceC, static_cast<int>(N)), "cublasDgemm");
        if (validate || printResults) {
            checkCuda(cudaMemcpy(C.data(), deviceC, rows * N * sizeof(double), cudaMemcpyDeviceToHost), "copy C from GPU");
        } else {
            checkCuda(cudaDeviceSynchronize(), "matrix multiplication");
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank_id == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / duration / 1e9);
    }

    if (printResults) {
        if (rank_id == 0) {
            std::vector<double> result(N * N);
            std::copy(C.begin(), C.end(), result.begin());
            for (int r = 1; r < ranks; ++r) {
                const size_t rRows = base + (static_cast<size_t>(r) < remainder);
                const size_t rFirst = base * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), remainder);
                transferResult(result.data() + rFirst * N, rRows * N, r, false);
            }
            print_results(result, "MatrixC");
        } else {
            transferResult(C.data(), rows * N, 0, true);
        }
    }

    int valid = 1;
    if (validate) {
        if (rank_id == 0) printf("Validating result...\n");
        int failures = 0;
        #pragma omp parallel for reduction(+:failures) schedule(static)
        for (int point = 0; point < 25; ++point) {
            const size_t i = static_cast<size_t>(point / 5) % N;
            const size_t j = static_cast<size_t>(point % 5) % N;
            if (i >= firstRow && i - firstRow < rows) {
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) expected += A[(i - firstRow) * N + k] * B[k * N + j];
                const double actual = C[(i - firstRow) * N + j];
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (!(relError <= 1e-6)) {
                    #pragma omp critical
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
                    ++failures;
                }
            }
        }
        int localValid = failures == 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank_id == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    if (rows != 0) {
        checkBlas(cublasDestroy(handle), "cublasDestroy");
        checkCuda(cudaFree(deviceA), "cudaFree A");
        checkCuda(cudaFree(deviceB), "cudaFree B");
        checkCuda(cudaFree(deviceC), "cudaFree C");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
