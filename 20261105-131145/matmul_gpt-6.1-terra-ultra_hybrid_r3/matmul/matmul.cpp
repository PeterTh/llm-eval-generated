#include <algorithm>
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

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Divide a contiguous range as evenly as possible, including when it is shorter
// than the number of workers.
size_t rangeBegin(size_t n, int workers, int worker) {
    return (n / workers) * worker + std::min(n % workers, static_cast<size_t>(worker));
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, static_cast<int>(status));
        std::abort();
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

struct DeviceWork {
    int device;
    size_t firstRow;
    size_t rows;
    double *a = nullptr, *b = nullptr, *c = nullptr;
    cublasHandle_t handle = nullptr;
};

// MPI's portable count argument is int; transfer large arrays in chunks.
void sendDoubles(const double* data, size_t count, int peer, bool send) {
    while (count != 0) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send)
            MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else
            MPI_Recv(const_cast<double*>(data), chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
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
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = strtoull(arg, &end, 10);
            if (arg[0] == '-' || end == arg || *end != '\0' || value > INT_MAX ||
                (value != 0 && value > std::numeric_limits<size_t>::max() / value) ||
                value * value > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                parseStatus = 1;
                break;
            }
            N = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 2;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, localSize, visibleDevices;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    cudaCheck(cudaGetDeviceCount(&visibleDevices), "cudaGetDeviceCount");
    if (visibleDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t firstRow = rangeBegin(N, ranks, rank);
    const size_t localRows = rangeBegin(N, ranks, rank + 1) - firstRow;
    int firstDevice, deviceCount;
    if (localSize <= visibleDevices) {
        firstDevice = static_cast<int>(rangeBegin(visibleDevices, localSize, localRank));
        deviceCount = static_cast<int>(rangeBegin(visibleDevices, localSize, localRank + 1)) - firstDevice;
    } else {
        firstDevice = localRank % visibleDevices;
        deviceCount = 1;
    }
    deviceCount = static_cast<int>(std::min(localRows, static_cast<size_t>(deviceCount)));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(localRows * N), B;
    if (localRows) B.resize(N * N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (localRows ? N : 0); ++i)
        for (size_t j = 0; j < N; ++j)
            B[i * N + j] = getPseudoRndValue(N, i, j);

    std::vector<DeviceWork> work(deviceCount);
    #pragma omp parallel for num_threads(deviceCount > 0 ? deviceCount : 1) schedule(static)
    for (int d = 0; d < deviceCount; ++d) {
        DeviceWork& w = work[d];
        w.device = firstDevice + d;
        w.firstRow = rangeBegin(localRows, deviceCount, d);
        w.rows = rangeBegin(localRows, deviceCount, d + 1) - w.firstRow;
        cudaCheck(cudaSetDevice(w.device), "cudaSetDevice");
        cudaCheck(cudaMalloc(&w.a, w.rows * N * sizeof(double)), "cudaMalloc A");
        cudaCheck(cudaMalloc(&w.b, N * N * sizeof(double)), "cudaMalloc B");
        cudaCheck(cudaMalloc(&w.c, w.rows * N * sizeof(double)), "cudaMalloc C");
        cudaCheck(cudaMemcpy(w.a, A.data() + w.firstRow * N, w.rows * N * sizeof(double), cudaMemcpyHostToDevice), "copy A to GPU");
        cudaCheck(cudaMemcpy(w.b, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU");
        blasCheck(cublasCreate(&w.handle), "cublasCreate");
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    #pragma omp parallel for num_threads(deviceCount > 0 ? deviceCount : 1) schedule(static)
    for (int d = 0; d < deviceCount; ++d) {
        DeviceWork& w = work[d];
        cudaCheck(cudaSetDevice(w.device), "cudaSetDevice");
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        blasCheck(cublasDgemm(w.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(w.rows), static_cast<int>(N),
                              &alpha, w.b, static_cast<int>(N), w.a, static_cast<int>(N),
                              &beta, w.c, static_cast<int>(N)), "cublasDgemm");
        cudaCheck(cudaDeviceSynchronize(), "matrix multiplication");
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
    }

    std::vector<double> C;
    if (validate || printResults) C.resize(localRows * N);
    #pragma omp parallel for num_threads(deviceCount > 0 ? deviceCount : 1) schedule(static)
    for (int d = 0; d < deviceCount; ++d) {
        DeviceWork& w = work[d];
        cudaCheck(cudaSetDevice(w.device), "cudaSetDevice");
        if (validate || printResults)
            cudaCheck(cudaMemcpy(C.data() + w.firstRow * N, w.c, w.rows * N * sizeof(double), cudaMemcpyDeviceToHost), "copy C from GPU");
        blasCheck(cublasDestroy(w.handle), "cublasDestroy");
        cudaCheck(cudaFree(w.a), "cudaFree A");
        cudaCheck(cudaFree(w.b), "cudaFree B");
        cudaCheck(cudaFree(w.c), "cudaFree C");
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(N * N);
            std::copy(C.begin(), C.end(), result.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t begin = rangeBegin(N, ranks, r), end = rangeBegin(N, ranks, r + 1);
                if (end != begin)
                    sendDoubles(result.data() + begin * N, (end - begin) * N, r, false);
            }
            print_results(result, "MatrixC");
        } else {
            sendDoubles(C.data(), C.size(), 0, true);
        }
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        for (size_t pi = 0; pi < 5 && valid && N != 0; ++pi) {
            const size_t i = pi % N;
            if (i < firstRow || i >= firstRow + localRows) continue;
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t j = pj % N;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k)
                    expected += A[(i - firstRow) * N + k] * B[k * N + j];
                const double actual = C[(i - firstRow) * N + j];
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(actual) || relError > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
                    valid = 0;
                    break;
                }
            }
        }
        int allValid;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        valid = allValid;
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
