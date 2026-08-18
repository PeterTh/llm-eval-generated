#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

void fail(const char* operation, const char* detail, int rank) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        fail(operation, cudaGetErrorString(status), rank);
    }
}

void checkCublas(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[64];
        std::snprintf(message, sizeof(message), "cuBLAS status %d", static_cast<int>(status));
        fail(operation, message, rank);
    }
}

// Generate the same deterministic values as the original benchmark.
constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) /
           static_cast<double>(n * n);
}

size_t firstRowForRank(size_t n, int rank, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    return static_cast<size_t>(rank) * base +
           std::min(static_cast<size_t>(rank), extra);
}

size_t rowsForRank(size_t n, int rank, int ranks) {
    return n / static_cast<size_t>(ranks) +
           (static_cast<size_t>(rank) < n % static_cast<size_t>(ranks));
}

void initLocalA(std::vector<double>& a, size_t n, size_t firstRow,
                size_t localRows) {
#pragma omp parallel for schedule(static)
    for (size_t localI = 0; localI < localRows; ++localI) {
        const size_t globalI = firstRow + localI;
        for (size_t j = 0; j < n; ++j) {
            a[localI * n + j] = getPseudoRndValue(n, globalI, j);
        }
    }
}

void initMatrix(std::vector<double>& matrix, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix[i * n + j] = getPseudoRndValue(n, i, j);
        }
    }
}

bool validateResult(const std::vector<double>& b, const std::vector<double>& c,
                    size_t n) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % n;
            const size_t j = checkPoints[pj] % n;
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k) {
                expected += getPseudoRndValue(n, i, k) * b[k * n + j];
            }
            const double actual = c[i * n + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, "
                            "got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        fail("MPI_Init_thread", "MPI_THREAD_FUNNELED is unavailable", rank);
    }

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            const bool valueValid = end != argv[i] && *end == '\0' && value > 0 &&
                                    value <= std::numeric_limits<size_t>::max();
            argumentsValid = argumentsValid && valueValid;
            if (valueValid) n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (rank == 0) std::printf("Unknown or incomplete option: %s\n", argv[i]);
        }
    }

    if (n > std::numeric_limits<size_t>::max() / n ||
        n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        argumentsValid = false;
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const size_t localRows = rowsForRank(n, rank, ranks);
    const size_t firstRow = firstRowForRank(n, rank, ranks);
    if (localRows != 0 && n > static_cast<size_t>(std::numeric_limits<int>::max()) / localRows) {
        fail("MPI_Gatherv", "per-rank matrix partition exceeds MPI int count", rank);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) fail("CUDA setup", "no CUDA device is available", rank);
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);

    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties", rank);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP "
                    "thread(s)/rank, CUDA/cuBLAS\n",
                    ranks, omp_get_max_threads());
        std::printf("Rank 0 CUDA device: %s\n", properties.name);
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localRows * n);
    std::vector<double> b(n * n);
    std::vector<double> localC(localRows * n);
    std::vector<double> c;
    if (rank == 0) c.resize(n * n);
    initLocalA(localA, n, firstRow, localRows);
    initMatrix(b, n);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), b.size() * sizeof(double)),
              "cudaMalloc(B)", rank);
    if (localRows != 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), localA.size() * sizeof(double)),
                  "cudaMalloc(A)", rank);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), localC.size() * sizeof(double)),
                  "cudaMalloc(C)", rank);
        checkCuda(cudaHostRegister(localA.data(), localA.size() * sizeof(double),
                                   cudaHostRegisterDefault),
                  "cudaHostRegister(A)", rank);
        checkCuda(cudaHostRegister(localC.data(), localC.size() * sizeof(double),
                                   cudaHostRegisterDefault),
                  "cudaHostRegister(C)", rank);
    }
    checkCuda(cudaHostRegister(b.data(), b.size() * sizeof(double), cudaHostRegisterDefault),
              "cudaHostRegister(B)", rank);

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate", rank);
    checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode", rank);

    // Force CUDA context and cuBLAS kernel initialization out of the measured region.
    if (localRows != 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                                &alpha, deviceB, 1, deviceA, 1, &beta, deviceC, 1),
                    "cuBLAS warm-up", rank);
        checkCuda(cudaDeviceSynchronize(), "CUDA warm-up", rank);
    }

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        receiveCounts.resize(ranks);
        displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t rows = rowsForRank(n, r, ranks);
            const size_t offset = firstRowForRank(n, r, ranks) * n;
            if (offset > static_cast<size_t>(std::numeric_limits<int>::max())) {
                fail("MPI_Gatherv", "matrix displacement exceeds MPI int range", rank);
            }
            receiveCounts[r] = static_cast<int>(rows * n);
            displacements[r] = static_cast<int>(offset);
        }
    }

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (localRows != 0) {
        checkCuda(cudaMemcpyAsync(deviceA, localA.data(), localA.size() * sizeof(double),
                                  cudaMemcpyHostToDevice),
                  "copy A to GPU", rank);
        checkCuda(cudaMemcpyAsync(deviceB, b.data(), b.size() * sizeof(double),
                                  cudaMemcpyHostToDevice),
                  "copy B to GPU", rank);
        const double alpha = 1.0;
        const double beta = 0.0;
        // Row-major C=A*B is column-major C^T=B^T*A^T in the same storage.
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(n), static_cast<int>(localRows),
                                static_cast<int>(n), &alpha, deviceB,
                                static_cast<int>(n), deviceA, static_cast<int>(n),
                                &beta, deviceC, static_cast<int>(n)),
                    "cublasDgemm", rank);
        checkCuda(cudaMemcpyAsync(localC.data(), deviceC, localC.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                  "copy C from GPU", rank);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", rank);
    }

    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                rank == 0 ? c.data() : nullptr,
                rank == 0 ? receiveCounts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    checkCublas(cublasDestroy(handle), "cublasDestroy", rank);
    checkCuda(cudaHostUnregister(b.data()), "cudaHostUnregister(B)", rank);
    if (localRows != 0) {
        checkCuda(cudaHostUnregister(localA.data()), "cudaHostUnregister(A)", rank);
        checkCuda(cudaHostUnregister(localC.data()), "cudaHostUnregister(C)", rank);
    }
    checkCuda(cudaFree(deviceA), "cudaFree(A)", rank);
    checkCuda(cudaFree(deviceB), "cudaFree(B)", rank);
    checkCuda(cudaFree(deviceC), "cudaFree(C)", rank);

    int result = EXIT_SUCCESS;
    if (rank == 0) {
        const auto milliseconds = static_cast<long long>(std::llround(elapsed * 1000.0));
        const double operations = 2.0 * static_cast<double>(n) *
                                  static_cast<double>(n) * static_cast<double>(n);
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", operations / elapsed / 1.0e9);
        if (printResults) print_results(c, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(b, c, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
