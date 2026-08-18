#include <mpi.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t n, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) /
           static_cast<double>(n * n);
}

void abortWithMessage(const int rank, const char* message) {
    if (rank == 0) {
        std::fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error during %s: %s\n", rank,
                     operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void checkCublas(const cublasStatus_t status, const char* operation,
                 const int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Rank %d: cuBLAS error during %s (status %d)\n",
                     rank, operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void initMatrixRows(double* matrix, const size_t n, const size_t firstRow,
                    const size_t rows) {
#pragma omp parallel for schedule(static)
    for (long long localRow = 0; localRow < static_cast<long long>(rows);
         ++localRow) {
        const size_t globalRow = firstRow + static_cast<size_t>(localRow);
        for (size_t column = 0; column < n; ++column) {
            matrix[static_cast<size_t>(localRow) * n + column] =
                getPseudoRndValue(n, globalRow, column);
        }
    }
}

void initMatrix(double* matrix, const size_t n) {
    initMatrixRows(matrix, n, 0, n);
}

bool validateResult(const std::vector<double>& c, const size_t n) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    long long failedCheck = -1;
    double failedExpected = 0.0;
    double failedActual = 0.0;
    double failedError = 0.0;

#pragma omp parallel for collapse(2) schedule(static)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % n;
            const size_t j = checkPoints[pj] % n;
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k) {
                expected += getPseudoRndValue(n, i, k) *
                            getPseudoRndValue(n, k, j);
            }
            const double actual = c[i * n + j];
            const double relativeError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relativeError > 1e-6) {
#pragma omp critical
                {
                    const long long check = pi * 5LL + pj;
                    if (failedCheck < 0 || check < failedCheck) {
                        failedCheck = check;
                        failedExpected = expected;
                        failedActual = actual;
                        failedError = relativeError;
                    }
                }
            }
        }
    }

    if (failedCheck >= 0) {
        const size_t i = checkPoints[failedCheck / 5] % n;
        const size_t j = checkPoints[failedCheck % 5] % n;
        std::printf("Validation failed at (%zu, %zu): expected %.10f, got "
                    "%.10f (error: %.10e)\n",
                    i, j, failedExpected, failedActual, failedError);
        return false;
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int suppliedThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &suppliedThreadLevel);

    int rank = 0;
    int rankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
    if (suppliedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(INT_MAX)) {
                argumentsValid = false;
            } else {
                n = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (n > std::numeric_limits<size_t>::max() / n ||
        n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
        abortWithMessage(rank, "Matrix dimensions overflow addressable memory");
    }

    // Contiguous block-row distribution, balanced to within one row.
    const size_t baseRows = n / static_cast<size_t>(rankCount);
    const size_t extraRows = n % static_cast<size_t>(rankCount);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localElements = localRows * n;

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "device discovery", rank);
    if (deviceCount == 0) {
        abortWithMessage(rank, "No CUDA device is available");
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "device selection", rank);
    cublasHandle_t blasHandle = nullptr;
    checkCublas(cublasCreate(&blasHandle), "handle creation", rank);
    checkCublas(cublasSetMathMode(blasHandle, CUBLAS_DEFAULT_MATH),
                "math mode selection", rank);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", rankCount,
                    omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localElements);
    std::vector<double> b(n * n);
    initMatrixRows(localA.data(), n, firstRow, localRows);
    initMatrix(b.data(), n);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), n * n * sizeof(double)),
              "B allocation", rank);
    checkCuda(cudaMemcpy(deviceB, b.data(), n * n * sizeof(double),
                         cudaMemcpyHostToDevice),
              "B upload", rank);
    if (localElements != 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA),
                             localElements * sizeof(double)),
                  "A allocation", rank);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC),
                             localElements * sizeof(double)),
                  "C allocation", rank);
        checkCuda(cudaMemcpy(deviceA, localA.data(), localElements * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "A upload", rank);
    }

    const auto multiplyLocal = [&]() {
        const double alpha = 1.0;
        const double beta = 0.0;
        // Row-major C=A*B is column-major C^T=B^T*A^T in the same buffers.
        checkCublas(cublasDgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(n),
                                static_cast<int>(localRows),
                                static_cast<int>(n), &alpha, deviceB,
                                static_cast<int>(n), deviceA,
                                static_cast<int>(n), &beta, deviceC,
                                static_cast<int>(n)),
                    "matrix multiplication", rank);
    };

    // Force lazy CUDA/cuBLAS module setup out of the measured steady-state
    // operation. The timed call below overwrites this warm-up result.
    if (localRows != 0) {
        multiplyLocal();
        checkCuda(cudaDeviceSynchronize(), "matrix multiplication warm-up", rank);
    }

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows != 0) {
        multiplyLocal();
        checkCuda(cudaDeviceSynchronize(), "matrix multiplication", rank);
    }
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<double> fullC;
    if (validate || printResults) {
        std::vector<double> localC(localElements);
        if (localElements != 0) {
            checkCuda(cudaMemcpy(localC.data(), deviceC,
                                 localElements * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                      "result download", rank);
        }
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            fullC.resize(n * n);
            receiveCounts.resize(rankCount);
            displacements.resize(rankCount);
            for (int process = 0; process < rankCount; ++process) {
                const size_t processRows =
                    baseRows + (static_cast<size_t>(process) < extraRows);
                const size_t processStart = static_cast<size_t>(process) * baseRows +
                    std::min(static_cast<size_t>(process), extraRows);
                if (processRows * n > static_cast<size_t>(INT_MAX) ||
                    processStart * n > static_cast<size_t>(INT_MAX)) {
                    abortWithMessage(rank,
                        "Result is too large for this MPI implementation's Gatherv");
                }
                receiveCounts[process] = static_cast<int>(processRows * n);
                displacements[process] = static_cast<int>(processStart * n);
            }
        }
        if (localElements > static_cast<size_t>(INT_MAX)) {
            abortWithMessage(rank,
                "Local result is too large for this MPI implementation's Gatherv");
        }
        MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    checkCuda(cudaFree(deviceC), "C release", rank);
    checkCuda(cudaFree(deviceA), "A release", rank);
    checkCuda(cudaFree(deviceB), "B release", rank);
    checkCublas(cublasDestroy(blasHandle), "handle destruction", rank);
    MPI_Comm_free(&localCommunicator);

    int returnCode = EXIT_SUCCESS;
    if (rank == 0) {
        const long long milliseconds =
            static_cast<long long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = 2.0 * static_cast<double>(n) *
                                  static_cast<double>(n) *
                                  static_cast<double>(n);
        const double gflops = elapsedSeconds > 0.0
                                  ? operations / elapsedSeconds / 1.0e9
                                  : std::numeric_limits<double>::infinity();
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(fullC, "MatrixC");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(fullC, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                returnCode = EXIT_FAILURE;
            }
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
