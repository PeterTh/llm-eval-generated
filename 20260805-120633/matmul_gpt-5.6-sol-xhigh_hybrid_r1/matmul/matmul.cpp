#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// Generate the same pseudo-random values as the original benchmark.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

[[noreturn]] void abortRun(const int rank, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int error, const char* operation, const int rank) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char errorString[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, errorString, &length);
    std::fprintf(stderr, "Rank %d: %s failed: %.*s\n", rank, operation,
                 length, errorString);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void checkCuda(const cudaError_t error, const char* operation, const int rank) {
    if (error == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

const char* cublasStatusString(const cublasStatus_t status) noexcept {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "internal error";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "not supported";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "license error";
        default:
            return "unknown error";
    }
}

void checkCublas(const cublasStatus_t status, const char* operation,
                 const int rank) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }

    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                 cublasStatusString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

size_t rowsForRank(const size_t N, const int rank, const int ranks) noexcept {
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rankIndex = static_cast<size_t>(rank);
    return N / rankCount + (rankIndex < N % rankCount ? 1 : 0);
}

size_t rowBeginForRank(const size_t N, const int rank,
                       const int ranks) noexcept {
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rankIndex = static_cast<size_t>(rank);
    return rankIndex * (N / rankCount) + std::min(rankIndex, N % rankCount);
}

void initMatrixRows(double* matrix, const size_t N, const size_t globalRowBegin,
                    const size_t rows) {
    // OpenMP parallelizes the host-side generation of each rank's input slab.
#pragma omp parallel for schedule(static)
    for (std::int64_t localRow = 0;
         localRow < static_cast<std::int64_t>(rows); ++localRow) {
        const size_t i = globalRowBegin + static_cast<size_t>(localRow);
        double* const row = matrix + static_cast<size_t>(localRow) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

bool parseMatrixSize(const char* text, size_t& value) noexcept {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<size_t>::max())) {
        return false;
    }

    value = static_cast<size_t>(parsed);
    return true;
}

void gatherResult(const std::vector<double>& localC, std::vector<double>& C,
                  const size_t N, const int rank, const int ranks) {
    const size_t totalElements = N * N;
    const size_t localElements = localC.size();

    // MPI_Gatherv is the fast path. Its counts and displacements are int in
    // MPI-3, so retain a chunked point-to-point fallback for very large jobs.
    if (totalElements <= static_cast<size_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
            for (int source = 0; source < ranks; ++source) {
                counts[static_cast<size_t>(source)] = static_cast<int>(
                    rowsForRank(N, source, ranks) * N);
                displacements[static_cast<size_t>(source)] = static_cast<int>(
                    rowBeginForRank(N, source, ranks) * N);
            }
        }

        checkMpi(MPI_Gatherv(localC.data(), static_cast<int>(localElements),
                             MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                             rank == 0 ? counts.data() : nullptr,
                             rank == 0 ? displacements.data() : nullptr,
                             MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv", rank);
        return;
    }

    constexpr int resultTag = 1701;
    if (rank == 0) {
        std::copy(localC.begin(), localC.end(), C.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t offset = rowBeginForRank(N, source, ranks) * N;
            const size_t elements = rowsForRank(N, source, ranks) * N;
            size_t received = 0;
            while (received < elements) {
                const int chunk = static_cast<int>(std::min(
                    elements - received, static_cast<size_t>(INT_MAX)));
                checkMpi(MPI_Recv(C.data() + offset + received, chunk,
                                  MPI_DOUBLE, source, resultTag,
                                  MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                         "MPI_Recv(result)", rank);
                received += static_cast<size_t>(chunk);
            }
        }
    } else {
        size_t sent = 0;
        while (sent < localElements) {
            const int chunk = static_cast<int>(std::min(
                localElements - sent, static_cast<size_t>(INT_MAX)));
            checkMpi(MPI_Send(localC.data() + sent, chunk, MPI_DOUBLE, 0,
                              resultTag, MPI_COMM_WORLD),
                     "MPI_Send(result)", rank);
            sent += static_cast<size_t>(chunk);
        }
    }
}

bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    double expected[25] = {};
    double relativeError[25] = {};

    // Each output element still accumulates k in the original order, while
    // independent validation points are evaluated concurrently with OpenMP.
#pragma omp parallel for schedule(static)
    for (int point = 0; point < 25; ++point) {
        const size_t i = checkPoints[point / 5] % N;
        const size_t j = checkPoints[point % 5] % N;
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) {
            sum += getPseudoRndValue(N, i, k) *
                   getPseudoRndValue(N, k, j);
        }

        expected[point] = sum;
        const double actual = C[i * N + j];
        relativeError[point] =
            std::abs((actual - sum) / (sum + 1.0e-10));
    }

    for (int point = 0; point < 25; ++point) {
        if (relativeError[point] > 1.0e-6 ||
            !std::isfinite(relativeError[point])) {
            const size_t i = checkPoints[point / 5] % N;
            const size_t j = checkPoints[point % 5] % N;
            std::printf("Validation failed at (%zu, %zu): expected %.10f, "
                        "got %.10f (error: %.10e)\n",
                        i, j, expected[point], C[i * N + j],
                        relativeError[point]);
            return false;
        }
    }

    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) "
                "(default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int initError =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                        &providedThreadLevel);
    if (initError != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int ranks = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size", rank);
    checkMpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
             "MPI_Comm_set_errhandler", rank);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortRun(rank, "MPI does not provide the required FUNNELED thread level");
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            argumentsValid = parseMatrixSize(argv[++i], N) && argumentsValid;
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

    if (N > static_cast<size_t>(INT_MAX) ||
        N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double) ||
        N > static_cast<size_t>(std::numeric_limits<std::int64_t>::max())) {
        abortRun(rank, "matrix size exceeds implementation limits");
    }

    // Map node-local MPI ranks round-robin onto the visible CUDA devices.
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localCommunicator),
             "MPI_Comm_split_type", rank);
    int localRank = 0;
    checkMpi(MPI_Comm_rank(localCommunicator, &localRank),
             "MPI_Comm_rank(local)", rank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        abortRun(rank, "no CUDA device is visible");
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);
    checkCuda(cudaFree(nullptr), "CUDA context initialization", rank);

    cudaDeviceProp deviceProperties{};
    checkCuda(cudaGetDeviceProperties(&deviceProperties, device),
              "cudaGetDeviceProperties", rank);
    checkMpi(MPI_Comm_free(&localCommunicator), "MPI_Comm_free", rank);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA device for rank 0: %s\n", deviceProperties.name);
        std::printf("Initializing matrices...\n");
    }

    const size_t localRows = rowsForRank(N, rank, ranks);
    const size_t firstRow = rowBeginForRank(N, rank, ranks);
    const size_t matrixElements = N * N;
    const size_t localElements = localRows * N;
    const size_t matrixBytes = matrixElements * sizeof(double);
    const size_t localBytes = localElements * sizeof(double);

    std::vector<double> localA;
    std::vector<double> B;
    std::vector<double> localC;
    try {
        localA.resize(localElements);
        if (localRows != 0) {
            B.resize(matrixElements);
        }
    } catch (const std::bad_alloc&) {
        abortRun(rank, "host matrix allocation failed");
    }

    if (localRows != 0) {
        initMatrixRows(localA.data(), N, firstRow, localRows);
        initMatrixRows(B.data(), N, 0, N);
    }

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    cublasHandle_t cublasHandle = nullptr;
    checkCublas(cublasCreate(&cublasHandle), "cublasCreate", rank);

    if (localRows != 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), localBytes),
                  "cudaMalloc(A)", rank);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), matrixBytes),
                  "cudaMalloc(B)", rank);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), localBytes),
                  "cudaMalloc(C)", rank);
        checkCuda(cudaMemcpy(deviceA, localA.data(), localBytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(A)", rank);
        checkCuda(cudaMemcpy(deviceB, B.data(), matrixBytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(B)", rank);

        // Prime cuBLAS and CUDA's lazy module loading outside the measured
        // region. A small subproblem avoids doubling the benchmark's work.
        const int warmRows = static_cast<int>(std::min<size_t>(N, 128));
        const int warmColumns =
            static_cast<int>(std::min<size_t>(localRows, 128));
        const double alpha = 1.0;
        const double beta = 0.0;
        checkCublas(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                warmRows, warmColumns, warmRows, &alpha,
                                deviceB, static_cast<int>(N), deviceA,
                                static_cast<int>(N), &beta, deviceC,
                                static_cast<int>(N)),
                    "cublasDgemm(warm-up)", rank);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warm-up)",
                  rank);
    }

    // Inputs have reached device memory and validation regenerates its small
    // reference set, so release these large host copies before collection.
    std::vector<double>().swap(localA);
    std::vector<double>().swap(B);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(start)", rank);
    const double start = MPI_Wtime();

    if (localRows != 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        // Row-major C=A*B is column-major C^T=B^T*A^T. Swapping the
        // operands lets cuBLAS compute that view directly without transposes.
        checkCublas(
            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        static_cast<int>(N), static_cast<int>(localRows),
                        static_cast<int>(N), &alpha, deviceB,
                        static_cast<int>(N), deviceA, static_cast<int>(N),
                        &beta, deviceC, static_cast<int>(N)),
            "cublasDgemm", rank);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(DGEMM)",
                  rank);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    checkMpi(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD),
             "MPI_Reduce(time)", rank);

    std::vector<double> C;
    if (printResults || validate) {
        try {
            localC.resize(localElements);
        } catch (const std::bad_alloc&) {
            abortRun(rank, "local result allocation failed");
        }
        if (rank == 0) {
            try {
                C.resize(matrixElements);
            } catch (const std::bad_alloc&) {
                abortRun(rank, "result matrix allocation failed");
            }
        }
        if (localRows != 0) {
            checkCuda(cudaMemcpy(localC.data(), deviceC, localBytes,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(C)", rank);
        }
        gatherResult(localC, C, N, rank, ranks);
    }

    checkCublas(cublasDestroy(cublasHandle), "cublasDestroy", rank);
    if (deviceC != nullptr) {
        checkCuda(cudaFree(deviceC), "cudaFree(C)", rank);
        checkCuda(cudaFree(deviceB), "cudaFree(B)", rank);
        checkCuda(cudaFree(deviceA), "cudaFree(A)", rank);
    }

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double operations =
            2.0 * static_cast<double>(N) * static_cast<double>(N) *
            static_cast<double>(N);
        const double gflops = operations / elapsed / 1.0e9;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(C, N)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }

    checkMpi(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD),
             "MPI_Bcast(exit code)", rank);
    checkMpi(MPI_Finalize(), "MPI_Finalize", rank);
    return exitCode;
}
