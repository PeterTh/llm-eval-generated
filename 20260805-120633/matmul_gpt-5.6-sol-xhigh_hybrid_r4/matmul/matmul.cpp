#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// Generate pseudo-random values for matrix initialization.  This is kept
// identical to the serial benchmark so that every rank can independently
// generate its part of A and the replicated B without communicating them.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct RowRange {
    size_t first;
    size_t count;
};

RowRange rowsForRank(const size_t N, const int rank, const int ranks) noexcept {
    const size_t base = N / static_cast<size_t>(ranks);
    const size_t extra = N % static_cast<size_t>(ranks);
    const size_t rankAsSize = static_cast<size_t>(rank);
    return {rankAsSize * base + std::min(rankAsSize, extra),
            base + (rankAsSize < extra ? 1U : 0U)};
}

void initMatrixRows(std::vector<double>& matrix, const size_t N,
                    const size_t firstRow, const size_t rows) {
#pragma omp parallel for schedule(static)
    for (size_t localRow = 0; localRow < rows; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        for (size_t column = 0; column < N; ++column) {
            matrix[localRow * N + column] =
                getPseudoRndValue(N, globalRow, column);
        }
    }
}

const char* cublasStatusName(const cublasStatus_t status) noexcept {
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
            return "unknown cuBLAS status";
    }
}

[[noreturn]] void abortCuda(const char* expression, const cudaError_t error,
                            const char* file, const int line, const int rank) {
    std::fprintf(stderr, "Rank %d: CUDA call %s failed at %s:%d: %s\n", rank,
                 expression, file, line, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

[[noreturn]] void abortCublas(const char* expression, const cublasStatus_t status,
                              const char* file, const int line, const int rank) {
    std::fprintf(stderr, "Rank %d: cuBLAS call %s failed at %s:%d: %s\n", rank,
                 expression, file, line, cublasStatusName(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call, rank)                                                   \
    do {                                                                          \
        const cudaError_t cudaCheckResult = (call);                               \
        if (cudaCheckResult != cudaSuccess) {                                     \
            abortCuda(#call, cudaCheckResult, __FILE__, __LINE__, (rank));        \
        }                                                                         \
    } while (false)

#define CUBLAS_CHECK(call, rank)                                                  \
    do {                                                                          \
        const cublasStatus_t cublasCheckResult = (call);                          \
        if (cublasCheckResult != CUBLAS_STATUS_SUCCESS) {                         \
            abortCublas(#call, cublasCheckResult, __FILE__, __LINE__, (rank));    \
        }                                                                         \
    } while (false)

struct DeviceResources {
    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
};

void destroyDeviceResources(DeviceResources& resources) noexcept {
    if (resources.handle != nullptr) {
        cublasDestroy(resources.handle);
    }
    if (resources.stream != nullptr) {
        cudaStreamDestroy(resources.stream);
    }
    if (resources.C != nullptr) {
        cudaFree(resources.C);
    }
    if (resources.B != nullptr) {
        cudaFree(resources.B);
    }
    if (resources.A != nullptr) {
        cudaFree(resources.A);
    }
}

// cuBLAS uses column-major matrices.  Row-major A*B is therefore evaluated as
// C^T = B^T*A^T, which also leaves C in the desired row-major byte layout.
void launchGemm(DeviceResources& resources, const size_t N,
                const size_t localRows, const int rank) {
    if (localRows == 0) {
        return;
    }

    const int dimension = static_cast<int>(N);
    const int columnsOfTransposedC = static_cast<int>(localRows);
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(resources.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            dimension, columnsOfTransposedC, dimension, &alpha,
                            resources.B, dimension, resources.A, dimension,
                            &beta, resources.C, dimension),
                 rank);
}

bool validateLocalResult(const std::vector<double>& localC, const size_t N,
                         const RowRange rows, const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t rawI : checkPoints) {
        const size_t i = rawI % N;
        if (i < rows.first || i >= rows.first + rows.count) {
            continue;
        }
        for (const size_t rawJ : checkPoints) {
            const size_t j = rawJ % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) *
                            getPseudoRndValue(N, k, j);
            }

            const double actual = localC[(i - rows.first) * N + j];
            const double relativeError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relativeError > 1e-6) {
                std::fprintf(stderr,
                             "Rank %d: validation failed at (%zu, %zu): "
                             "expected %.10f, got %.10f (error: %.10e)\n",
                             rank, i, j, expected, actual, relativeError);
                return false;
            }
        }
    }
    return true;
}

// MPI-3 collective counts are int-sized.  Chunked point-to-point transfers
// retain support for matrices whose total element count exceeds INT_MAX.
void gatherResult(const std::vector<double>& localC, std::vector<double>& fullC,
                  const size_t N, const int rank, const int ranks) {
    constexpr int resultTag = 1729;
    constexpr size_t maxChunk =
        static_cast<size_t>(std::numeric_limits<int>::max());

    if (rank == 0) {
        const RowRange rootRows = rowsForRank(N, 0, ranks);
        std::copy(localC.begin(), localC.end(),
                  fullC.begin() + static_cast<std::ptrdiff_t>(rootRows.first * N));

        for (int source = 1; source < ranks; ++source) {
            const RowRange sourceRows = rowsForRank(N, source, ranks);
            size_t remaining = sourceRows.count * N;
            size_t offset = sourceRows.first * N;
            while (remaining != 0) {
                const size_t chunk = std::min(remaining, maxChunk);
                MPI_Recv(fullC.data() + offset, static_cast<int>(chunk), MPI_DOUBLE,
                         source, resultTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
                remaining -= chunk;
            }
        }
    } else {
        size_t remaining = localC.size();
        size_t offset = 0;
        while (remaining != 0) {
            const size_t chunk = std::min(remaining, maxChunk);
            MPI_Send(localC.data() + offset, static_cast<int>(chunk), MPI_DOUBLE, 0,
                     resultTag, MPI_COMM_WORLD);
            offset += chunk;
            remaining -= chunk;
        }
    }
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, size_t& value) noexcept {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return value <= std::numeric_limits<size_t>::max() / value &&
           value * value <= std::numeric_limits<size_t>::max() / sizeof(double);
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                        &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide required thread support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], N)) {
                argumentsValid = false;
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
            if (!argumentsValid) {
                std::fprintf(stderr, "Matrix size must be a positive, supported integer.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const RowRange localRows = rowsForRank(N, rank, ranks);
    const size_t fullElements = N * N;
    const size_t localElements = localRows.count * N;
    const size_t fullBytes = fullElements * sizeof(double);
    const size_t localBytes = localElements * sizeof(double);

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), rank);
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA device is visible\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device), rank);

    int openmpThreads = 1;
#pragma omp parallel
    {
#pragma omp master
        openmpThreads = omp_get_num_threads();
    }
    int minimumOpenmpThreads = 1;
    int maximumOpenmpThreads = 1;
    MPI_Reduce(&openmpThreads, &minimumOpenmpThreads, 1, MPI_INT, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&openmpThreads, &maximumOpenmpThreads, 1, MPI_INT, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank(s), %d-%d OpenMP thread(s) "
                    "per rank, CUDA/cuBLAS\n",
                    ranks, minimumOpenmpThreads, maximumOpenmpThreads);
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> hostA;
    std::vector<double> hostB;
    try {
        hostA.resize(localElements);
        if (localRows.count != 0) {
            hostB.resize(fullElements);
        }
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "Rank %d: host matrix allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    initMatrixRows(hostA, N, localRows.first, localRows.count);
    if (localRows.count != 0) {
        initMatrixRows(hostB, N, 0, N);
    }

    DeviceResources resources;
    if (localRows.count != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&resources.A), localBytes), rank);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&resources.B), fullBytes), rank);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&resources.C), localBytes), rank);
        CUDA_CHECK(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
                   rank);
        CUBLAS_CHECK(cublasCreate(&resources.handle), rank);
        CUBLAS_CHECK(cublasSetStream(resources.handle, resources.stream), rank);
        CUBLAS_CHECK(cublasSetMathMode(resources.handle, CUBLAS_DEFAULT_MATH), rank);

        CUDA_CHECK(cudaMemcpyAsync(resources.A, hostA.data(), localBytes,
                                   cudaMemcpyHostToDevice, resources.stream),
                   rank);
        CUDA_CHECK(cudaMemcpyAsync(resources.B, hostB.data(), fullBytes,
                                   cudaMemcpyHostToDevice, resources.stream),
                   rank);
        CUDA_CHECK(cudaStreamSynchronize(resources.stream), rank);

        // Exclude one-time CUDA context and library initialization from the
        // benchmark, matching the original program's compute-only timing.
        launchGemm(resources, N, localRows.count, rank);
        CUDA_CHECK(cudaStreamSynchronize(resources.stream), rank);
    }

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    launchGemm(resources, N, localRows.count, rank);
    if (localRows.count != 0) {
        CUDA_CHECK(cudaStreamSynchronize(resources.stream), rank);
    }
    const double localSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const auto milliseconds = static_cast<long long>(elapsedSeconds * 1000.0);
        const double operations = 2.0 * static_cast<double>(N) *
                                  static_cast<double>(N) * static_cast<double>(N);
        const double gflops = operations / elapsedSeconds / 1.0e9;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> localC;
    if (validate || printResults) {
        try {
            localC.resize(localElements);
        } catch (const std::bad_alloc&) {
            std::fprintf(stderr, "Rank %d: host result allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (localRows.count != 0) {
            CUDA_CHECK(cudaMemcpyAsync(localC.data(), resources.C, localBytes,
                                       cudaMemcpyDeviceToHost, resources.stream),
                       rank);
            CUDA_CHECK(cudaStreamSynchronize(resources.stream), rank);
        }
    }

    if (printResults) {
        std::vector<double> fullC;
        if (rank == 0) {
            try {
                fullC.resize(fullElements);
            } catch (const std::bad_alloc&) {
                std::fprintf(stderr, "Rank 0: full result allocation failed\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        gatherResult(localC, fullC, N, rank, ranks);
        if (rank == 0) {
            print_results(fullC, "MatrixC");
        }
    }

    bool globallyValid = true;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const int localValid = validateLocalResult(localC, N, localRows, rank) ? 1 : 0;
        int allValid = 0;
        MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        globallyValid = allValid != 0;
        if (rank == 0) {
            std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");
        }
    }

    destroyDeviceResources(resources);
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return globallyValid ? EXIT_SUCCESS : EXIT_FAILURE;
}
