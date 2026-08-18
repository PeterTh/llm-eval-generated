#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

[[noreturn]] void fail(const int rank, const char* const where, const char* const detail) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, where, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int status, const int rank, const char* const where) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char message[MPI_MAX_ERROR_STRING]{};
    int messageLength = 0;
    MPI_Error_string(status, message, &messageLength);
    fail(rank, where, message);
}

void checkCuda(const cudaError_t status, const int rank, const char* const where) {
    if (status != cudaSuccess) {
        fail(rank, where, cudaGetErrorString(status));
    }
}

void checkCublas(const cublasStatus_t status, const int rank, const char* const where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[64]{};
        std::snprintf(message, sizeof(message), "cuBLAS status %d", static_cast<int>(status));
        fail(rank, where, message);
    }
}

#define MPI_CHECK(call) checkMpi((call), rank, #call)
#define CUDA_CHECK(call) checkCuda((call), rank, #call)
#define CUBLAS_CHECK(call) checkCublas((call), rank, #call)

void initMatrixRows(std::vector<double>& matrix, const size_t N, const size_t firstRow,
                    const size_t rowCount) {
#pragma omp parallel for schedule(static)
    for (long long localRow = 0; localRow < static_cast<long long>(rowCount); ++localRow) {
        const size_t globalRow = firstRow + static_cast<size_t>(localRow);
        double* const row = matrix.data() + static_cast<size_t>(localRow) * N;
#pragma omp simd
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

// Recreate A directly from its deterministic initializer, avoiding a second full A allocation on
// rank zero solely for validation.
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t checkRow : checkPoints) {
        for (const size_t checkColumn : checkPoints) {
            const size_t i = checkRow % N;
            const size_t j = checkColumn % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relativeError = std::abs((actual - expected) / (expected + 1e-10));
            if (relativeError > 1e-6) {
                std::printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                    i, j, expected, actual, relativeError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* const progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseMatrixSize(const char* const value, size_t& N) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    N = static_cast<size_t>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        fail(rank, "MPI_Init_thread", "MPI_THREAD_FUNNELED support is required");
    }

    // Only rank zero handles user-visible argument diagnostics and then communicates the result.
    // This prevents every MPI rank from printing the same help or error message.
    enum class ParseResult : int { Run = 0, SuccessExit = 1, FailureExit = 2 };
    ParseResult parseResult = ParseResult::Run;
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                if (!parseMatrixSize(argv[++i], N)) {
                    std::printf("Invalid matrix size: %s\n", argv[i]);
                    parseResult = ParseResult::FailureExit;
                    break;
                }
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseResult = ParseResult::SuccessExit;
                break;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseResult = ParseResult::FailureExit;
                break;
            }
        }
    }

    int parseCode = static_cast<int>(parseResult);
    unsigned long long broadcastN = static_cast<unsigned long long>(N);
    int flags = (validate ? 1 : 0) | (printResults ? 2 : 0);
    MPI_CHECK(MPI_Bcast(&parseCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&broadcastN, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&flags, 1, MPI_INT, 0, MPI_COMM_WORLD));

    parseResult = static_cast<ParseResult>(parseCode);
    if (parseResult != ParseResult::Run) {
        MPI_CHECK(MPI_Finalize());
        return parseResult == ParseResult::SuccessExit ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    N = static_cast<size_t>(broadcastN);
    validate = (flags & 1) != 0;
    printResults = (flags & 2) != 0;

    // MPI and cuBLAS use int dimensions/counts. This condition also rules out allocations that
    // would be impractical on the target accelerators before a product can overflow below.
    if (N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::sqrt(static_cast<double>(std::numeric_limits<int>::max())))) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large for MPI/cuBLAS integer dimensions\n");
        }
        MPI_CHECK(MPI_Finalize());
        return EXIT_FAILURE;
    }

    const size_t matrixElements = N * N;
    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localElements = localRows * N;

    std::vector<int> receiveCounts(static_cast<size_t>(worldSize));
    std::vector<int> displacements(static_cast<size_t>(worldSize));
    for (int process = 0; process < worldSize; ++process) {
        const size_t processRows =
            baseRows + (static_cast<size_t>(process) < extraRows ? 1 : 0);
        const size_t processFirstRow = static_cast<size_t>(process) * baseRows +
                                       std::min(static_cast<size_t>(process), extraRows);
        receiveCounts[static_cast<size_t>(process)] = static_cast<int>(processRows * N);
        displacements[static_cast<size_t>(process)] = static_cast<int>(processFirstRow * N);
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    // A is distributed by row. B is shared by all row blocks and is initialized once, then
    // broadcast. Both initialization paths execute OpenMP work on every participating rank.
    std::vector<double> localA(localElements);
    std::vector<double> B(matrixElements);
    initMatrixRows(localA, N, firstRow, localRows);
    if (rank == 0) {
        initMatrixRows(B, N, 0, N);
    }
    MPI_CHECK(MPI_Bcast(B.data(), static_cast<int>(matrixElements), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // Choose one GPU per local MPI rank. This makes the executable usable unchanged on a
    // multi-node cluster where each node exposes the same local CUDA-device numbering.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fail(rank, "cudaGetDeviceCount", "no CUDA device is available");
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Allocate at least one element for ranks that own zero rows, which lets every rank follow
    // the same CUDA setup path even when there are more ranks than matrix rows.
    const size_t allocationElements = std::max<size_t>(localElements, 1);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceA), allocationElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceB), matrixElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceC), allocationElements * sizeof(double)));

    CUDA_CHECK(cudaMemcpyAsync(deviceB, B.data(), matrixElements * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceA, localA.data(), localElements * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto multiplyLocalBlock = [&] {
        if (localRows == 0) {
            return;
        }

        const double alpha = 1.0;
        const double beta = 0.0;
        const int n = static_cast<int>(N);
        const int rows = static_cast<int>(localRows);

        // Row-major C=A*B is the column-major transposed product C^T=B^T*A^T. Viewing the
        // row-major buffers as column-major therefore requires no transposition or repacking.
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, rows, n, &alpha, deviceB,
                                 n, deviceA, n, &beta, deviceC, n));
    };

    // cuBLAS initializes dispatch state lazily on its first GEMM. Prime it after all input copies
    // so that accelerator-library setup is part of initialization, not the measured operation.
    multiplyLocalBlock();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }

    // The barrier aligns ranks before the timed region; the maximum rank time is the distributed
    // step's elapsed time. Device transfers are deliberately outside this region, matching the
    // original benchmark's convention of timing only matrix multiplication after initialization.
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::steady_clock::now();
    multiplyLocalBlock();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double computationSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &computationSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    // A full host result is needed only for the optional interfaces that expose it. Otherwise
    // each rank retains its completed local C block and avoids an unnecessary global gather.
    const bool needGlobalResult = validate || printResults;
    std::vector<double> localC(localElements);
    std::vector<double> C;
    if (needGlobalResult) {
        if (localElements != 0) {
            CUDA_CHECK(cudaMemcpyAsync(localC.data(), deviceC, localElements * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (rank == 0) {
            C.resize(matrixElements);
        }
        MPI_CHECK(MPI_Gatherv(localElements == 0 ? nullptr : localC.data(),
                              static_cast<int>(localElements), MPI_DOUBLE,
                              rank == 0 ? C.data() : nullptr, receiveCounts.data(),
                              displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceC));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceA));

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(computationSeconds));
        std::printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              computationSeconds / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(B, C, N)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
