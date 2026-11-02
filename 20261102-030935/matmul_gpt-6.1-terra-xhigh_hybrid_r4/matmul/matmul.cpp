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

namespace {
int rankId = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rankId, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", rankId, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Rank %d: %s: cuBLAS error %d\n", rankId, operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// MPI counts are int, so move large result blocks in chunks.
void transferResult(double* data, size_t count, int peer, bool send) {
    while (count != 0) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        int status = send ? MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD)
                          : MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (status != MPI_SUCCESS) fail("MPI result transfer failed");
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

size_t rowCount(size_t n, int ranks, int rank) {
    return n / static_cast<size_t>(ranks) + (static_cast<size_t>(rank) < n % static_cast<size_t>(ranks));
}

size_t rowStart(size_t n, int ranks, int rank) {
    return static_cast<size_t>(rank) * (n / static_cast<size_t>(ranks)) +
           std::min(static_cast<size_t>(rank), n % static_cast<size_t>(ranks));
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not support MPI_THREAD_FUNNELED");

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(INT_MAX) ||
                parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (rankId == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rankId == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rankId == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) fail("No CUDA device is available");
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    const size_t rows = rowCount(N, ranks, rankId);
    const size_t firstRow = rowStart(N, ranks, rankId);
    if (rankId == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                    N, N, validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    try {
        std::vector<double> A(rows * N), B(rows ? N * N : 0), C(rows * N);
        if (rows) {
            #pragma omp parallel
            {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < N; ++j)
                        A[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
                #pragma omp for schedule(static)
                for (size_t i = 0; i < N; ++i)
                    for (size_t j = 0; j < N; ++j)
                        B[i * N + j] = getPseudoRndValue(N, i, j);
            }
        }

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t handle = nullptr;
        if (rows) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dA), A.size() * sizeof(double)), "cudaMalloc A");
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dB), B.size() * sizeof(double)), "cudaMalloc B");
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dC), C.size() * sizeof(double)), "cudaMalloc C");
            blasCheck(cublasCreate(&handle), "cublasCreate");
        }
        if (rankId == 0) std::printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        if (rows) {
            cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A to GPU");
            cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU");
            const double alpha = 1.0, beta = 0.0;
            // Row-major C = A*B is column-major C^T = B^T*A^T.
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                  &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                  &beta, dC, static_cast<int>(N)), "cublasDgemm");
            cudaCheck(cudaMemcpy(C.data(), dC, C.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C from GPU");
        }
        double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rankId == 0) {
            std::printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
            std::printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
        }
        if (rows) {
            blasCheck(cublasDestroy(handle), "cublasDestroy");
            cudaCheck(cudaFree(dA), "cudaFree A");
            cudaCheck(cudaFree(dB), "cudaFree B");
            cudaCheck(cudaFree(dC), "cudaFree C");
        }

        if (printResults) {
            if (rankId == 0) {
                std::vector<double> result(N * N);
                std::copy(C.begin(), C.end(), result.begin());
                for (int peer = 1; peer < ranks; ++peer)
                    transferResult(result.data() + rowStart(N, ranks, peer) * N,
                                   rowCount(N, ranks, peer) * N, peer, false);
                print_results(result, "MatrixC");
            } else {
                transferResult(C.data(), C.size(), 0, true);
            }
        }

        int failedRank = ranks;
        size_t failedI = 0, failedJ = 0;
        double failedExpected = 0.0, failedActual = 0.0, failedError = 0.0;
        if (validate) {
            for (size_t pi = 0; pi < 5 && failedRank == ranks; ++pi) {
                size_t i = pi % N;
                if (i < firstRow || i >= firstRow + rows) continue;
                for (size_t pj = 0; pj < 5; ++pj) {
                    size_t j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += A[(i - firstRow) * N + k] * B[k * N + j];
                    double actual = C[(i - firstRow) * N + j];
                    double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (error > 1e-6 || !std::isfinite(actual)) {
                        failedRank = rankId;
                        failedI = i; failedJ = j;
                        failedExpected = expected; failedActual = actual; failedError = error;
                        break;
                    }
                }
            }
            int firstFailedRank = ranks;
            MPI_Allreduce(&failedRank, &firstFailedRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (firstFailedRank != ranks) {
                unsigned long long position[2] = {failedI, failedJ};
                double details[3] = {failedExpected, failedActual, failedError};
                MPI_Bcast(position, 2, MPI_UNSIGNED_LONG_LONG, firstFailedRank, MPI_COMM_WORLD);
                MPI_Bcast(details, 3, MPI_DOUBLE, firstFailedRank, MPI_COMM_WORLD);
                if (rankId == 0) {
                    std::printf("Validating result...\n");
                    std::printf("Validation failed at (%llu, %llu): expected %.10f, got %.10f (error: %.10e)\n",
                                position[0], position[1], details[0], details[1], details[2]);
                    std::printf("Validation: FAILED\n");
                }
                MPI_Finalize();
                return 1;
            }
            if (rankId == 0) std::printf("Validating result...\nValidation: PASSED\n");
        }
    } catch (const std::exception& e) {
        fail(e.what());
    }
    MPI_Finalize();
    return 0;
}
