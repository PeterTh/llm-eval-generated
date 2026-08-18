#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
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

namespace {

[[noreturn]] void fail(const char* what, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, what);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s: %s", operation,
                      cudaGetErrorString(status));
        fail(message, rank);
    }
}

void checkCublas(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s failed (cuBLAS status %d)",
                      operation, static_cast<int>(status));
        fail(message, rank);
    }
}

constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * size_t{1299709}) % (n * n)) /
           static_cast<double>(n * n);
}

void initMatrixRows(std::vector<double>& matrix, size_t n, size_t firstRow,
                    size_t rows) {
#pragma omp parallel for schedule(static)
    for (long long localI = 0; localI < static_cast<long long>(rows); ++localI) {
        const size_t i = firstRow + static_cast<size_t>(localI);
        for (size_t j = 0; j < n; ++j) {
            matrix[static_cast<size_t>(localI) * n + j] =
                getPseudoRndValue(n, i, j);
        }
    }
}

void gatherRows(const std::vector<double>& local, std::vector<double>& global,
                size_t n, int rank, int ranks) {
    constexpr int tag = 711;
    if (rank == 0) {
        std::copy(local.begin(), local.end(), global.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t beginRow = (n * static_cast<size_t>(source)) /
                                    static_cast<size_t>(ranks);
            const size_t endRow = (n * static_cast<size_t>(source + 1)) /
                                  static_cast<size_t>(ranks);
            const size_t count = (endRow - beginRow) * n;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(
                    std::min(count - offset, static_cast<size_t>(INT_MAX)));
                MPI_Recv(global.data() + beginRow * n + offset, chunk, MPI_DOUBLE,
                         source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += static_cast<size_t>(chunk);
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const int chunk = static_cast<int>(
                std::min(local.size() - offset, static_cast<size_t>(INT_MAX)));
            MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, tag,
                     MPI_COMM_WORLD);
            offset += static_cast<size_t>(chunk);
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
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));
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
        fail("MPI does not provide the required thread support", rank);
    }

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || value == 0 ||
                value > static_cast<unsigned long long>(INT_MAX)) {
                argumentsValid = false;
            } else {
                n = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }

    if (help || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (n > std::numeric_limits<size_t>::max() / n) {
        fail("matrix size overflows addressable memory", rank);
    }

    // Contiguous row ownership, balanced to within one row.
    const size_t firstRow = (n * static_cast<size_t>(rank)) /
                            static_cast<size_t>(ranks);
    const size_t endRow = (n * static_cast<size_t>(rank + 1)) /
                          static_cast<size_t>(ranks);
    const size_t localRows = endRow - firstRow;
    const size_t matrixElements = n * n;
    const size_t localElements = localRows * n;

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) fail("no CUDA device is available", rank);
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Parallelism: %d MPI rank(s), up to %d OpenMP thread(s) per "
                    "rank, CUDA/cuBLAS\n",
                    ranks, omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> b(matrixElements);
    std::vector<double> localA(localElements);
    std::vector<double> localC(localElements);
    // B is needed in full by every accelerator.  Its deterministic generation is
    // cheaper and more scalable than sending N^2 doubles across the network.
    initMatrixRows(b, n, 0, n);
    initMatrixRows(localA, n, firstRow, localRows);

    double* deviceB = nullptr;
    double* deviceA = nullptr;
    double* deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceB, matrixElements * sizeof(double)),
              "allocating device matrix B", rank);
    checkCuda(cudaMemcpy(deviceB, b.data(), matrixElements * sizeof(double),
                         cudaMemcpyHostToDevice),
              "copying matrix B to device", rank);
    if (localElements != 0) {
        checkCuda(cudaMalloc(&deviceA, localElements * sizeof(double)),
                  "allocating local device matrix A", rank);
        checkCuda(cudaMalloc(&deviceC, localElements * sizeof(double)),
                  "allocating local device matrix C", rank);
        checkCuda(cudaMemcpy(deviceA, localA.data(), localElements * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "copying local matrix A to device", rank);
    }

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate", rank);
    if (localRows != 0) {
        // Force lazy CUDA/cuBLAS initialization out of the measured region.
        const double alpha = 1.0;
        const double beta = 0.0;
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                                &alpha, deviceB, 1, deviceA, 1, &beta, deviceC, 1),
                    "cuBLAS warm-up", rank);
        checkCuda(cudaDeviceSynchronize(), "synchronizing cuBLAS warm-up", rank);
    }
    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows != 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        // Row-major C=A*B is column-major C^T=B^T*A^T in the same storage.
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(n), static_cast<int>(localRows),
                                static_cast<int>(n), &alpha, deviceB,
                                static_cast<int>(n), deviceA, static_cast<int>(n),
                                &beta, deviceC, static_cast<int>(n)),
                    "cublasDgemm", rank);
    }
    checkCuda(cudaDeviceSynchronize(), "synchronizing matrix multiplication", rank);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const bool needOutput = validate || printResults;
    if (needOutput && localElements != 0) {
        checkCuda(cudaMemcpy(localC.data(), deviceC, localElements * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "copying local matrix C to host", rank);
    }
    cublasDestroy(handle);
    cudaFree(deviceC);
    cudaFree(deviceA);
    cudaFree(deviceB);

    std::vector<double> c;
    if (rank == 0 && needOutput) c.resize(matrixElements);
    if (needOutput) gatherRows(localC, c, n, rank, ranks);

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        const double milliseconds = seconds * 1000.0;
        const double operations = 2.0 * static_cast<double>(n) *
                                  static_cast<double>(n) * static_cast<double>(n);
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", operations / seconds / 1e9);
        if (printResults) print_results(c, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(b, c, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
