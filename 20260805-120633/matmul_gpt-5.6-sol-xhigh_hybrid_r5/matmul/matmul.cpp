#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#ifndef _OPENMP
#error "This benchmark requires OpenMP support"
#endif

namespace {

// Generate the same deterministic values as the original benchmark.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct RowBlock {
    size_t begin;
    size_t count;
};

RowBlock getRowBlock(const size_t N, const int rank, const int ranks) noexcept {
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rowsPerRank = N / rankCount;
    const size_t extraRows = N % rankCount;
    return {rankIndex * rowsPerRank + (rankIndex < extraRows ? rankIndex : extraRows),
            rowsPerRank + (rankIndex < extraRows ? 1U : 0U)};
}

[[noreturn]] void abortRank(const int rank, const char* operation,
                            const char* reason) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, reason);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        abortRank(rank, operation, cudaGetErrorString(status));
    }
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
            return "unknown cuBLAS error";
    }
}

void checkCublas(const cublasStatus_t status, const char* operation,
                 const int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        abortRank(rank, operation, cublasStatusString(status));
    }
}

bool checkedProduct(const size_t lhs, const size_t rhs, size_t& product) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    product = lhs * rhs;
    return true;
}

void initMatrices(std::vector<double>& localA, std::vector<double>& B,
                  const size_t N, const RowBlock rows) {
    // Both independent loops use all host cores assigned to this MPI rank.
#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (size_t localRow = 0; localRow < rows.count; ++localRow) {
            const size_t globalRow = rows.begin + localRow;
            for (size_t column = 0; column < N; ++column) {
                localA[localRow * N + column] =
                    getPseudoRndValue(N, globalRow, column);
            }
        }

#pragma omp for schedule(static)
        for (size_t row = 0; row < N; ++row) {
            for (size_t column = 0; column < N; ++column) {
                B[row * N + column] = getPseudoRndValue(N, row, column);
            }
        }
    }
}

bool validateLocalResult(const std::vector<double>& localC, const size_t N,
                         const RowBlock rows, const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    int localFailures = 0;
    double localMaxRelativeError = 0.0;

    // The selected output rows live on different ranks; OpenMP parallelizes
    // the independent reference dot products within each owning rank.
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : localFailures) \
    reduction(max : localMaxRelativeError)
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t row = checkPoints[pi] % N;
            const size_t column = checkPoints[pj] % N;
            if (row < rows.begin || row >= rows.begin + rows.count) {
                continue;
            }

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, row, k) *
                            getPseudoRndValue(N, k, column);
            }

            const double actual = localC[(row - rows.begin) * N + column];
            const double relativeError =
                std::abs((actual - expected) / (expected + 1.0e-10));
            localMaxRelativeError = std::max(localMaxRelativeError, relativeError);
            localFailures += relativeError > 1.0e-6 ? 1 : 0;
        }
    }

    int globalFailures = 0;
    double globalMaxRelativeError = 0.0;
    MPI_Allreduce(&localFailures, &globalFailures, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localMaxRelativeError, &globalMaxRelativeError, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0 && globalFailures != 0) {
        std::printf("Validation failed: %d checked values exceeded tolerance "
                    "(maximum relative error %.10e)\n",
                    globalFailures, globalMaxRelativeError);
    }
    return globalFailures == 0;
}

// MPI count/displacement parameters are int-sized in MPI-3. Gather in flat
// chunks so result output also works for matrices with more than INT_MAX items.
void gatherResult(const std::vector<double>& localC, std::vector<double>& globalC,
                  const size_t N, const RowBlock localRows, const int rank,
                  const int ranks) {
    const size_t totalElements = N * N;
    constexpr size_t maxChunk = static_cast<size_t>(INT_MAX);
    std::vector<int> receiveCounts(static_cast<size_t>(ranks));
    std::vector<int> displacements(static_cast<size_t>(ranks));
    double zeroCountBuffer = 0.0;

    const size_t localBegin = localRows.begin * N;
    const size_t localEnd = localBegin + localRows.count * N;
    for (size_t chunkBegin = 0; chunkBegin < totalElements;) {
        const size_t chunkSize = std::min(maxChunk, totalElements - chunkBegin);
        const size_t chunkEnd = chunkBegin + chunkSize;

        for (int sourceRank = 0; sourceRank < ranks; ++sourceRank) {
            const RowBlock sourceRows = getRowBlock(N, sourceRank, ranks);
            const size_t sourceBegin = sourceRows.begin * N;
            const size_t sourceEnd = sourceBegin + sourceRows.count * N;
            const size_t intersectionBegin = std::max(sourceBegin, chunkBegin);
            const size_t intersectionEnd = std::min(sourceEnd, chunkEnd);
            const size_t count = intersectionEnd > intersectionBegin
                                     ? intersectionEnd - intersectionBegin
                                     : 0;
            receiveCounts[static_cast<size_t>(sourceRank)] = static_cast<int>(count);
            displacements[static_cast<size_t>(sourceRank)] =
                count != 0 ? static_cast<int>(intersectionBegin - chunkBegin) : 0;
        }

        const size_t sendBegin = std::max(localBegin, chunkBegin);
        const size_t sendEnd = std::min(localEnd, chunkEnd);
        const size_t sendCount = sendEnd > sendBegin ? sendEnd - sendBegin : 0;
        const double* sendBuffer = sendCount != 0
                                       ? localC.data() + (sendBegin - localBegin)
                                       : &zeroCountBuffer;
        double* receiveBuffer = rank == 0 ? globalC.data() + chunkBegin : nullptr;

        MPI_Gatherv(sendBuffer, static_cast<int>(sendCount), MPI_DOUBLE,
                    receiveBuffer, receiveCounts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        chunkBegin = chunkEnd;
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

bool parseMatrixSize(const char* text, size_t& N) noexcept {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > static_cast<unsigned long long>(INT_MAX) ||
        value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    N = static_cast<size_t>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel) !=
        MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        abortRank(rank, "MPI_Init_thread", "MPI_THREAD_FUNNELED is unavailable");
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

    size_t matrixElements = 0;
    if (!checkedProduct(N, N, matrixElements) ||
        matrixElements > std::numeric_limits<size_t>::max() / sizeof(double)) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large for this platform\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &nodeCommunicator);
    int nodeRank = 0;
    int ranksOnNode = 1;
    MPI_Comm_rank(nodeCommunicator, &nodeRank);
    MPI_Comm_size(nodeCommunicator, &ranksOnNode);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        abortRank(rank, "CUDA device selection", "no CUDA-capable GPU was found");
    }
    const int device = nodeRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);

    cudaDeviceProp deviceProperties{};
    checkCuda(cudaGetDeviceProperties(&deviceProperties, device),
              "cudaGetDeviceProperties", rank);

    const RowBlock rows = getRowBlock(N, rank, ranks);
    size_t localElements = 0;
    if (!checkedProduct(rows.count, N, localElements)) {
        abortRank(rank, "local matrix sizing", "element count overflow");
    }

    std::vector<double> localA;
    std::vector<double> B;
    try {
        localA.resize(localElements);
        B.resize(matrixElements);
    } catch (const std::bad_alloc&) {
        abortRank(rank, "host allocation", "insufficient memory");
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank%s, up to %d OpenMP "
                    "thread%s/rank, CUDA/cuBLAS\n",
                    ranks, ranks == 1 ? "" : "s", omp_get_max_threads(),
                    omp_get_max_threads() == 1 ? "" : "s");
        std::printf("Rank 0 GPU: device %d (%s)\n", device, deviceProperties.name);
        if (ranksOnNode > deviceCount) {
            std::printf("Warning: %d ranks share %d GPU%s on rank 0's node; "
                        "one rank per GPU is recommended\n",
                        ranksOnNode, deviceCount, deviceCount == 1 ? "" : "s");
        }
        std::printf("Initializing matrices...\n");
    }
    initMatrices(localA, B, N, rows);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    const size_t localBytes = localElements * sizeof(double);
    const size_t matrixBytes = matrixElements * sizeof(double);
    // cudaMalloc(0) is invalid, so idle ranks retain a one-element allocation.
    const size_t allocatedLocalBytes = std::max(sizeof(double), localBytes);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), allocatedLocalBytes),
              "cudaMalloc(A)", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), matrixBytes),
              "cudaMalloc(B)", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), allocatedLocalBytes),
              "cudaMalloc(C)", rank);

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags", rank);
    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate", rank);
    checkCublas(cublasSetStream(handle, stream), "cublasSetStream", rank);
    checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH),
                "cublasSetMathMode", rank);

    if (localBytes != 0) {
        checkCuda(cudaMemcpyAsync(deviceA, localA.data(), localBytes,
                                  cudaMemcpyHostToDevice, stream),
                  "copying A to the GPU", rank);
    }
    checkCuda(cudaMemcpyAsync(deviceB, B.data(), matrixBytes,
                              cudaMemcpyHostToDevice, stream),
              "copying B to the GPU", rank);
    checkCuda(cudaStreamSynchronize(stream), "initial GPU transfer", rank);

    // Host inputs are no longer needed; release them before allocating optional
    // output storage. This is important for large multi-rank jobs.
    std::vector<double>().swap(localA);
    std::vector<double>().swap(B);

    const double alpha = 1.0;
    const double beta = 0.0;
    const auto launchDgemm = [&]() {
        if (rows.count == 0) {
            return;
        }
        // Row-major C=A*B is column-major C^T=B^T*A^T. This layout transform
        // avoids transposes and lets cuBLAS execute one large optimized DGEMM.
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(N), static_cast<int>(rows.count),
                                static_cast<int>(N), &alpha, deviceB,
                                static_cast<int>(N), deviceA, static_cast<int>(N),
                                &beta, deviceC, static_cast<int>(N)),
                    "cublasDgemm", rank);
    };

    // Prime lazy cuBLAS module/workspace initialization so the benchmark
    // measures steady-state multiplication rather than one-time setup.
    launchDgemm();
    checkCuda(cudaStreamSynchronize(stream), "cuBLAS warm-up", rank);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    launchDgemm();
    checkCuda(cudaStreamSynchronize(stream), "matrix multiplication", rank);
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const bool needHostResult = validate || printResults;
    std::vector<double> localC;
    if (needHostResult) {
        try {
            localC.resize(localElements);
        } catch (const std::bad_alloc&) {
            abortRank(rank, "host result allocation", "insufficient memory");
        }
        if (localBytes != 0) {
            checkCuda(cudaMemcpyAsync(localC.data(), deviceC, localBytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "copying C from the GPU", rank);
            checkCuda(cudaStreamSynchronize(stream), "result GPU transfer", rank);
        }
    }

    checkCublas(cublasDestroy(handle), "cublasDestroy", rank);
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy", rank);
    checkCuda(cudaFree(deviceC), "cudaFree(C)", rank);
    checkCuda(cudaFree(deviceB), "cudaFree(B)", rank);
    checkCuda(cudaFree(deviceA), "cudaFree(A)", rank);

    if (rank == 0) {
        const double operations = 2.0 * static_cast<double>(N) *
                                  static_cast<double>(N) * static_cast<double>(N);
        const double gflops = elapsed > 0.0 ? operations / elapsed / 1.0e9 : 0.0;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> C;
        if (rank == 0) {
            try {
                C.resize(matrixElements);
            } catch (const std::bad_alloc&) {
                abortRank(rank, "global result allocation", "insufficient memory");
            }
        }
        gatherResult(localC, C, N, rows, rank, ranks);
        if (rank == 0) {
            print_results(C, "MatrixC");
        }
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateLocalResult(localC, N, rows, rank);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    MPI_Comm_free(&nodeCommunicator);
    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
