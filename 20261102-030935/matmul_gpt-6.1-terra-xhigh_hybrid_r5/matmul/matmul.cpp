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

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void fail(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void checkBlas(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Rank %d: %s failed (cuBLAS status %d)\n", rank, operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// The first N % ranks ranks receive one additional row.
static size_t rowsForRank(size_t n, int ranks, int rank) {
    return n / static_cast<size_t>(ranks) + (static_cast<size_t>(rank) < n % static_cast<size_t>(ranks));
}
static size_t firstRow(size_t n, int ranks, int rank) {
    return (n / static_cast<size_t>(ranks)) * static_cast<size_t>(rank)
         + std::min(static_cast<size_t>(rank), n % static_cast<size_t>(ranks));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not provide MPI_THREAD_FUNNELED", rank);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || *value == '-' || *value == '\0' || *end != '\0' || parsed == 0 || parsed > INT_MAX) {
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
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) fail("Matrix size is too large", rank);

    try {
        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        int localRank = 0;
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);
        int deviceCount = 0;
        checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
        if (deviceCount == 0) fail("No CUDA device available", rank);
        checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);

        const size_t localRows = rowsForRank(N, ranks, rank);
        const size_t rowStart = firstRow(N, ranks, rank);
        const size_t localElements = localRows * N;
        if (rank == 0) {
            std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                        N, N, validate ? "enabled" : "disabled");
            std::printf("Initializing matrices...\n");
        }

        std::vector<double> A(localElements), B(N * N);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localRows; ++i) {
            for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, rowStart + i, j);
        }

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t handle = nullptr;
        if (localRows != 0) {
            checkCuda(cudaMalloc(&dA, localElements * sizeof(double)), "cudaMalloc A", rank);
            checkCuda(cudaMalloc(&dB, N * N * sizeof(double)), "cudaMalloc B", rank);
            checkCuda(cudaMalloc(&dC, localElements * sizeof(double)), "cudaMalloc C", rank);
            checkCuda(cudaMemcpy(dA, A.data(), localElements * sizeof(double), cudaMemcpyHostToDevice), "copy A", rank);
            checkCuda(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B", rank);
            checkBlas(cublasCreate(&handle), "cublasCreate", rank);
            // Exclude CUDA's first-use kernel loading and setup from the benchmark.
            const double alpha = 1.0, beta = 0.0;
            checkBlas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  static_cast<int>(N), static_cast<int>(localRows), static_cast<int>(N),
                                  &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                  &beta, dC, static_cast<int>(N)), "warm-up cublasDgemm", rank);
            checkCuda(cudaDeviceSynchronize(), "warm-up matrix multiplication", rank);
        }
        if (rank == 0) std::printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        if (localRows != 0) {
            const double alpha = 1.0, beta = 0.0;
            // Row-major C = A*B is column-major C^T = B^T*A^T.
            checkBlas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  static_cast<int>(N), static_cast<int>(localRows), static_cast<int>(N),
                                  &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                  &beta, dC, static_cast<int>(N)), "cublasDgemm", rank);
            checkCuda(cudaDeviceSynchronize(), "matrix multiplication", rank);
        }
        const double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
            std::printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
        }

        std::vector<double> localC;
        if ((printResults || validate) && localRows != 0) {
            localC.resize(localElements);
            checkCuda(cudaMemcpy(localC.data(), dC, localElements * sizeof(double), cudaMemcpyDeviceToHost), "copy C", rank);
        }
        if (printResults) {
            std::vector<double> C;
            if (rank == 0) {
                C.resize(N * N);
                std::copy(localC.begin(), localC.end(), C.begin());
                for (int source = 1; source < ranks; ++source) {
                    size_t offset = firstRow(N, ranks, source) * N;
                    size_t remaining = rowsForRank(N, ranks, source) * N;
                    while (remaining != 0) {
                        int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(C.data() + offset, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        offset += count;
                        remaining -= count;
                    }
                }
                print_results(C, "MatrixC");
            } else {
                size_t offset = 0;
                while (offset < localElements) {
                    int count = static_cast<int>(std::min(localElements - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Send(localC.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    offset += count;
                }
            }
        }
        int valid = 1;
        if (validate) {
            if (rank == 0) std::printf("Validating result...\n");
            for (size_t pi = 0; pi < 5; ++pi) {
                const size_t i = pi % N;
                if (i < rowStart || i >= rowStart + localRows) continue;
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k) expected += A[(i - rowStart) * N + k] * B[k * N + j];
                    const double actual = localC[(i - rowStart) * N + j];
                    const double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (relError > 1e-6 || !std::isfinite(actual)) {
                        std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                                    i, j, expected, actual, relError);
                        valid = 0;
                    }
                }
            }
            int allValid = 0;
            MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            valid = allValid;
            if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        if (handle) checkBlas(cublasDestroy(handle), "cublasDestroy", rank);
        if (dA) checkCuda(cudaFree(dA), "cudaFree A", rank);
        if (dB) checkCuda(cudaFree(dB), "cudaFree B", rank);
        if (dC) checkCuda(cudaFree(dC), "cudaFree C", rank);
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::exception& e) {
        fail(e.what(), rank);
    }
}
