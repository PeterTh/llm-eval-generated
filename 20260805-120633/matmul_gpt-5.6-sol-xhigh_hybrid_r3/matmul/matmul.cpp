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

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

struct Options {
    size_t matrixSize = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
};

[[noreturn]] void abortJob(const char* message, const char* expression,
                           const char* file, int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s (%s) at %s:%d\n", rank, message,
                 expression, file, line);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(int status, const char* expression, const char* file, int line) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char errorString[MPI_MAX_ERROR_STRING] = {};
    int errorLength = 0;
    MPI_Error_string(status, errorString, &errorLength);
    errorString[std::min(errorLength, MPI_MAX_ERROR_STRING - 1)] = '\0';
    abortJob(errorString, expression, file, line);
}

void checkCuda(cudaError_t status, const char* expression, const char* file,
               int line) {
    if (status != cudaSuccess) {
        abortJob(cudaGetErrorString(status), expression, file, line);
    }
}

const char* cublasErrorString(cublasStatus_t status) noexcept {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "unknown cuBLAS error";
    }
}

void checkCublas(cublasStatus_t status, const char* expression,
                 const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        abortJob(cublasErrorString(status), expression, file, line);
    }
}

#define MPI_CHECK(expression) \
    checkMpi((expression), #expression, __FILE__, __LINE__)
#define CUDA_CHECK(expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__)
#define CUBLAS_CHECK(expression) \
    checkCublas((expression), #expression, __FILE__, __LINE__)

class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t elements) {
        if (elements != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  elements * sizeof(double)));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    double* get() noexcept { return data_; }
    const double* get() const noexcept { return data_; }

  private:
    double* data_ = nullptr;
};

// Generate exactly the same deterministic values as the original benchmark.
constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) /
           static_cast<double>(n * n);
}

size_t rowsForRank(size_t n, int rank, int ranks) noexcept {
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rankIndex = static_cast<size_t>(rank);
    return n / rankCount + (rankIndex < n % rankCount ? 1 : 0);
}

size_t rowOffsetForRank(size_t n, int rank, int ranks) noexcept {
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rankIndex = static_cast<size_t>(rank);
    return rankIndex * (n / rankCount) + std::min(rankIndex, n % rankCount);
}

void initMatrix(std::vector<double>& matrix, size_t n) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix[static_cast<size_t>(i) * n + j] =
                getPseudoRndValue(n, static_cast<size_t>(i), j);
        }
    }
}

void initLocalMatrixA(std::vector<double>& matrix, size_t n,
                      size_t globalRowOffset, size_t localRows) {
#pragma omp parallel for schedule(static)
    for (long long localRow = 0;
         localRow < static_cast<long long>(localRows); ++localRow) {
        const size_t row = static_cast<size_t>(localRow);
        const size_t globalRow = globalRowOffset + row;
        for (size_t j = 0; j < n; ++j) {
            matrix[row * n + j] = getPseudoRndValue(n, globalRow, j);
        }
    }
}

bool validateLocalResult(const std::vector<double>& localA,
                         const std::vector<double>& matrixB,
                         const std::vector<double>& localC, size_t n,
                         size_t rowOffset, size_t localRows, int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t globalRow = checkPoints[pi] % n;
        if (globalRow < rowOffset || globalRow >= rowOffset + localRows) {
            continue;
        }
        const size_t localRow = globalRow - rowOffset;

        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t column = checkPoints[pj] % n;
            double expected = 0.0;
#pragma omp simd reduction(+ : expected)
            for (long long k = 0; k < static_cast<long long>(n); ++k) {
                const size_t index = static_cast<size_t>(k);
                expected += localA[localRow * n + index] *
                            matrixB[index * n + column];
            }

            const double actual = localC[localRow * n + column];
            const double relativeError =
                std::abs((actual - expected) / (expected + 1.0e-10));
            if (relativeError > 1.0e-6) {
                std::fprintf(
                    stderr,
                    "Rank %d: validation failed at (%zu, %zu): expected "
                    "%.10f, got %.10f (error: %.10e)\n",
                    rank, globalRow, column, expected, actual, relativeError);
                return false;
            }
        }
    }
    return true;
}

void gatherResult(const std::vector<double>& localC,
                  std::vector<double>& fullC, size_t n, int rank, int ranks) {
    const size_t totalElements = n * n;
    const size_t localElements = localC.size();

    if (totalElements <= static_cast<size_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
            for (int source = 0; source < ranks; ++source) {
                counts[static_cast<size_t>(source)] = static_cast<int>(
                    rowsForRank(n, source, ranks) * n);
                displacements[static_cast<size_t>(source)] = static_cast<int>(
                    rowOffsetForRank(n, source, ranks) * n);
            }
        }

        MPI_CHECK(MPI_Gatherv(
            localC.data(), static_cast<int>(localElements), MPI_DOUBLE,
            rank == 0 ? fullC.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
            MPI_COMM_WORLD));
        return;
    }

    // MPI counts are int-sized. Chunk very large matrices without restricting N.
    constexpr int resultTag = 7101;
    constexpr size_t maxChunk = static_cast<size_t>(INT_MAX);
    if (rank == 0) {
        std::copy(localC.begin(), localC.end(), fullC.begin());
        for (int source = 1; source < ranks; ++source) {
            size_t remaining = rowsForRank(n, source, ranks) * n;
            size_t destination = rowOffsetForRank(n, source, ranks) * n;
            while (remaining != 0) {
                const int chunk =
                    static_cast<int>(std::min(remaining, maxChunk));
                MPI_CHECK(MPI_Recv(fullC.data() + destination, chunk,
                                   MPI_DOUBLE, source, resultTag,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                destination += static_cast<size_t>(chunk);
                remaining -= static_cast<size_t>(chunk);
            }
        }
    } else {
        size_t remaining = localElements;
        size_t sourceOffset = 0;
        while (remaining != 0) {
            const int chunk = static_cast<int>(std::min(remaining, maxChunk));
            MPI_CHECK(MPI_Send(localC.data() + sourceOffset, chunk, MPI_DOUBLE,
                               0, resultTag, MPI_COMM_WORLD));
            sourceOffset += static_cast<size_t>(chunk);
            remaining -= static_cast<size_t>(chunk);
        }
    }
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf(
        "  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* valueString = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value =
                std::strtoull(valueString, &end, 10);
            if (errno != 0 || end == valueString || *end != '\0' ||
                valueString[0] == '-' || value == 0 ||
                value > static_cast<unsigned long long>(INT_MAX) ||
                value > std::numeric_limits<size_t>::max()) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", valueString);
                return false;
            }
            options.matrixSize = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.showHelp = true;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int initStatus =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (initStatus != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortJob("MPI does not provide MPI_THREAD_FUNNELED", "MPI_Init_thread",
                 __FILE__, __LINE__);
    }

    Options options;
    const bool argumentsValid = parseOptions(argc, argv, options);
    if (!argumentsValid || options.showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const size_t n = options.matrixSize;
    if (n > std::numeric_limits<size_t>::max() / n ||
        n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large for this platform\n");
        }
        MPI_CHECK(MPI_Finalize());
        return EXIT_FAILURE;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    int localRanks = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localRanks));

    // Avoid CPU oversubscription unless the launcher/user explicitly selected a
    // per-rank OpenMP thread count.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_dynamic(0);
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localRanks));
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortJob("no CUDA device is visible", "cudaGetDeviceCount", __FILE__,
                 __LINE__);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // Initialize the CUDA context before timing.

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    if (localRank == 0 && localRanks > deviceCount) {
        std::fprintf(stderr,
                     "Warning: %d local MPI ranks share %d visible CUDA "
                     "devices; one rank per GPU is recommended.\n",
                     localRanks, deviceCount);
    }

    const size_t localRows = rowsForRank(n, rank, ranks);
    const size_t rowOffset = rowOffsetForRank(n, rank, ranks);
    const size_t matrixElements = n * n;
    const size_t localElements = localRows * n;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n",
                    options.validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank%s, up to %d OpenMP "
                    "threads/rank, CUDA/cuBLAS\n",
                    ranks, ranks == 1 ? "" : "s", omp_get_max_threads());
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
        std::printf("Initializing matrices...\n");
    }

    int returnCode = EXIT_SUCCESS;
    try {
        std::vector<double> localA(localElements);
        std::vector<double> matrixB(matrixElements);
        initLocalMatrixA(localA, n, rowOffset, localRows);
        initMatrix(matrixB, n);

        DeviceBuffer deviceA(localElements);
        DeviceBuffer deviceB(matrixElements);
        DeviceBuffer deviceC(localElements);

        if (localElements != 0) {
            CUDA_CHECK(cudaMemcpy(deviceA.get(), localA.data(),
                                  localElements * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(deviceB.get(), matrixB.data(),
                              matrixElements * sizeof(double),
                              cudaMemcpyHostToDevice));

        cublasHandle_t cublasHandle = nullptr;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));
        CUBLAS_CHECK(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));
        CUBLAS_CHECK(
            cublasSetAtomicsMode(cublasHandle, CUBLAS_ATOMICS_ALLOWED));

        const double alpha = 1.0;
        const double beta = 0.0;

        // Trigger any lazy cuBLAS initialization outside the measured region.
        if (localRows != 0) {
            CUBLAS_CHECK(cublasDgemm(
                cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha,
                deviceB.get(), 1, deviceA.get(), 1, &beta, deviceC.get(), 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        cudaEvent_t startEvent = nullptr;
        cudaEvent_t stopEvent = nullptr;
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));

        if (rank == 0) {
            std::printf("Computing matrix multiplication...\n");
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        CUDA_CHECK(cudaEventRecord(startEvent));

        if (localRows != 0) {
            // Row-major C=A*B is column-major C^T=B^T*A^T. The dimensions
            // below therefore produce localRows complete row-major C rows.
            CUBLAS_CHECK(cublasDgemm(
                cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(localRows),
                static_cast<int>(n), &alpha, deviceB.get(),
                static_cast<int>(n), deviceA.get(), static_cast<int>(n), &beta,
                deviceC.get(), static_cast<int>(n)));
        }

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        float localElapsedMilliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&localElapsedMilliseconds, startEvent,
                                        stopEvent));

        const double localElapsed =
            static_cast<double>(localElapsedMilliseconds);
        double elapsedMilliseconds = 0.0;
        MPI_CHECK(MPI_Reduce(&localElapsed, &elapsedMilliseconds, 1, MPI_DOUBLE,
                             MPI_MAX, 0, MPI_COMM_WORLD));

        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));

        const bool needHostResult = options.validate || options.printResults;
        std::vector<double> localC;
        if (needHostResult) {
            localC.resize(localElements);
            if (localElements != 0) {
                CUDA_CHECK(cudaMemcpy(localC.data(), deviceC.get(),
                                      localElements * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            }
        }

        CUBLAS_CHECK(cublasDestroy(cublasHandle));

        if (rank == 0) {
            const double operations =
                2.0 * static_cast<double>(n) * static_cast<double>(n) *
                static_cast<double>(n);
            const double gflops =
                operations / (elapsedMilliseconds * 1.0e6);
            std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
            std::printf("Performance: %.3f GFLOPS\n", gflops);
        }

        if (options.printResults) {
            std::vector<double> fullC;
            if (rank == 0) {
                fullC.resize(matrixElements);
            }
            gatherResult(localC, fullC, n, rank, ranks);
            if (rank == 0) {
                print_results(fullC, "MatrixC");
            }
        }

        if (options.validate) {
            if (rank == 0) {
                std::printf("Validating result...\n");
            }
            const int localValid =
                validateLocalResult(localA, matrixB, localC, n, rowOffset,
                                    localRows, rank)
                    ? 1
                    : 0;
            int globallyValid = 0;
            MPI_CHECK(MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT,
                                    MPI_MIN, MPI_COMM_WORLD));
            if (rank == 0) {
                std::printf("Validation: %s\n",
                            globallyValid != 0 ? "PASSED" : "FAILED");
            }
            if (globallyValid == 0) {
                returnCode = EXIT_FAILURE;
            }
        }
    } catch (const std::exception& error) {
        abortJob(error.what(), "matrix allocation/execution", __FILE__,
                 __LINE__);
    }

    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_CHECK(MPI_Finalize());
    return returnCode;
}
