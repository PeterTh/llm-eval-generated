#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// Each MPI rank owns a contiguous block of rows.  Initializing the input from
// its deterministic formula eliminates input-distribution traffic while still
// producing exactly the same A and B as the single-process program.
void initMatrixRows(double* matrix, const size_t N, const size_t firstRow,
                    const size_t rowCount) {
#pragma omp parallel for schedule(static)
    for (std::int64_t localRow = 0;
         localRow < static_cast<std::int64_t>(rowCount); ++localRow) {
        const size_t globalRow = firstRow + static_cast<size_t>(localRow);
        double* const row = matrix + static_cast<size_t>(localRow) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void initFullMatrix(double* matrix, const size_t N) {
    initMatrixRows(matrix, N, 0, N);
}

const char* cublasErrorString(const cublasStatus_t status) {
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
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

[[noreturn]] void abortWithMessage(const int rank, const char* operation,
                                   const char* detail) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        abortWithMessage(rank, operation, cudaGetErrorString(status));
    }
}

void checkCublas(const cublasStatus_t status, const char* operation, const int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        abortWithMessage(rank, operation, cublasErrorString(status));
    }
}

// The original validation samples the first five rows and columns.  Regenerate
// those input values directly so the root rank need not retain a full copy of A.
bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t iPoint : checkPoints) {
        for (const size_t jPoint : checkPoints) {
            const size_t i = iPoint % N;
            const size_t j = jPoint % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f "
                            "(error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseMatrixSize(const char* text, size_t* const value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        return false;
    }

    *value = static_cast<size_t>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

    // Parse once, then explicitly share the benchmark configuration with all
    // ranks so every rank takes the same collective-control-flow path.
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                if (!parseMatrixSize(argv[++i], &N)) {
                    std::printf("Invalid matrix size: %s\n", argv[i]);
                    parseError = true;
                }
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                parseError = true;
            }
        }
    }

    std::uint64_t broadcastN = static_cast<std::uint64_t>(N);
    int options[4] = {validate ? 1 : 0, printResults ? 1 : 0, showHelp ? 1 : 0,
                      parseError ? 1 : 0};
    MPI_Bcast(&broadcastN, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(broadcastN);
    validate = options[0] != 0;
    printResults = options[1] != 0;
    showHelp = options[2] != 0;
    parseError = options[3] != 0;

    if (showHelp || parseError) {
        if (rank == 0) {
            if (showHelp || parseError) {
                printUsage(argv[0]);
            }
        }
        MPI_Finalize();
        return parseError ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    const size_t matrixElements = N * N;
    if (matrixElements / N != N ||
        matrixElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large for this MPI/cuBLAS implementation.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localElements = localRows * N;

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        abortWithMessage(rank, "CUDA device discovery", "no CUDA devices are visible");
    }
    // A scheduler normally exposes one GPU to each rank.  Modulo mapping also
    // keeps direct multi-rank launches correct when a node has fewer visible GPUs.
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localElements);
    std::vector<double> B(matrixElements);
    initMatrixRows(localA.data(), N, firstRow, localRows);
    initFullMatrix(B.data(), N);

    // cudaMalloc with a zero size is not portable across all supported CUDA
    // runtimes.  Ranks beyond N have no rows, but still participate safely in
    // the MPI collectives with a one-element scratch allocation.
    const size_t deviceLocalAllocationElements = std::max<size_t>(localElements, 1);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, deviceLocalAllocationElements * sizeof(double)),
              "cudaMalloc(A)", rank);
    checkCuda(cudaMalloc(&deviceB, matrixElements * sizeof(double)), "cudaMalloc(B)", rank);
    checkCuda(cudaMalloc(&deviceC, deviceLocalAllocationElements * sizeof(double)),
              "cudaMalloc(C)", rank);

    if (localElements != 0) {
        checkCuda(cudaMemcpy(deviceA, localA.data(), localElements * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(A)", rank);
    }
    checkCuda(cudaMemcpy(deviceB, B.data(), matrixElements * sizeof(double), cudaMemcpyHostToDevice),
              "cudaMemcpy(B)", rank);

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate", rank);
    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate", rank);
    checkCublas(cublasSetStream(handle, stream), "cublasSetStream", rank);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }

    // cuBLAS uses column-major storage.  Treating the row-major arrays as their
    // transposes computes C^T = B^T A^T, whose memory layout is exactly C = A B.
    const double alpha = 1.0;
    const double beta = 0.0;

    // cuBLAS initializes internal dispatch state on its first GEMM.  Warm it on
    // every rank before the synchronized measurement so the reported time is
    // the steady-state distributed GEMM time, rather than one-time setup cost.
    if (localRows != 0) {
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N),
                                static_cast<int>(localRows), static_cast<int>(N), &alpha, deviceB,
                                static_cast<int>(N), deviceA, static_cast<int>(N), &beta, deviceC,
                                static_cast<int>(N)),
                    "cublasDgemm warm-up", rank);
        checkCuda(cudaStreamSynchronize(stream), "CUDA warm-up synchronization", rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double localStart = MPI_Wtime();
    if (localRows != 0) {
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N),
                                static_cast<int>(localRows), static_cast<int>(N), &alpha, deviceB,
                                static_cast<int>(N), deviceA, static_cast<int>(N), &beta, deviceC,
                                static_cast<int>(N)),
                    "cublasDgemm", rank);
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", rank);
    const double localSeconds = MPI_Wtime() - localStart;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> localC;
    std::vector<double> globalC;
    if (validate || printResults) {
        localC.resize(localElements);
        if (localElements != 0) {
            checkCuda(cudaMemcpy(localC.data(), deviceC, localElements * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(C)", rank);
        }

        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            globalC.resize(matrixElements);
            receiveCounts.resize(static_cast<size_t>(worldSize));
            displacements.resize(static_cast<size_t>(worldSize));
            for (int process = 0; process < worldSize; ++process) {
                const size_t processRows = baseRows +
                                           (static_cast<size_t>(process) < extraRows ? 1 : 0);
                const size_t processFirstRow = static_cast<size_t>(process) * baseRows +
                                               std::min(static_cast<size_t>(process), extraRows);
                receiveCounts[static_cast<size_t>(process)] = static_cast<int>(processRows * N);
                displacements[static_cast<size_t>(process)] =
                    static_cast<int>(processFirstRow * N);
            }
        }

        MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    checkCublas(cublasDestroy(handle), "cublasDestroy", rank);
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy", rank);
    checkCuda(cudaFree(deviceC), "cudaFree(C)", rank);
    checkCuda(cudaFree(deviceB), "cudaFree(B)", rank);
    checkCuda(cudaFree(deviceA), "cudaFree(A)", rank);
    MPI_Comm_free(&nodeComm);

    int validationStatus = EXIT_SUCCESS;
    if (rank == 0) {
        const double elapsedMilliseconds = elapsedSeconds * 1000.0;
        const double gflops = elapsedSeconds > 0.0
                                  ? (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                                     static_cast<double>(N)) /
                                        elapsedSeconds / 1.0e9
                                  : 0.0;
        std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(globalC, "MatrixC");
        }

        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(globalC, N)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                validationStatus = EXIT_FAILURE;
            }
        }
    }

    MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validationStatus;
}
