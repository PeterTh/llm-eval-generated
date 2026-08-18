#include <algorithm>
#include <chrono>
#include <cstddef>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

[[noreturn]] void fail(const char* operation, const char* detail, const int rank) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        fail(operation, cudaGetErrorString(status), rank);
    }
}

void checkCublas(const cublasStatus_t status, const char* operation, const int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail(operation, "cuBLAS returned an error", rank);
    }
}

void checkMpi(const int status, const char* operation, const int rank) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, error, &length);
        fail(operation, error, rank);
    }
}

struct RowPartition {
    size_t rows;
    size_t firstRow;
};

RowPartition partitionRows(const size_t N, const int rank, const int ranks) {
    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t extraRows = N % static_cast<size_t>(ranks);
    const size_t rankAsSize = static_cast<size_t>(rank);
    return {
        baseRows + (rankAsSize < extraRows ? 1 : 0),
        rankAsSize * baseRows + std::min(rankAsSize, extraRows),
    };
}

// Each MPI rank initializes only its owned rows of A.  B is replicated on
// every rank, avoiding communication in the timed GEMM phase.
void initLocalMatrixA(std::vector<double>& matrix, const size_t N, const size_t firstRow) {
#pragma omp parallel for schedule(static)
    for (long long localRow = 0; localRow < static_cast<long long>(matrix.size() / N); ++localRow) {
        const size_t globalRow = firstRow + static_cast<size_t>(localRow);
        double* const row = matrix.data() + static_cast<size_t>(localRow) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void initMatrixB(std::vector<double>& matrix, const size_t N) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        double* const row = matrix.data() + static_cast<size_t>(i) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, static_cast<size_t>(i), j);
        }
    }
}

// Validate the same points as the original program, but only on the rank that
// owns each requested output row.  The collective reduction yields one result
// for the distributed matrix.
bool validateLocalResult(const std::vector<double>& localA, const std::vector<double>& B,
                         const std::vector<double>& localC, const size_t N,
                         const RowPartition partition) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t checkRow : checkPoints) {
        const size_t i = checkRow % N;
        if (i < partition.firstRow || i >= partition.firstRow + partition.rows) {
            continue;
        }

        const size_t localRow = i - partition.firstRow;
        const double* const aRow = localA.data() + localRow * N;
        const double* const cRow = localC.data() + localRow * N;
        for (const size_t checkColumn : checkPoints) {
            const size_t j = checkColumn % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += aRow[k] * B[k * N + j];
            }

            const double actual = cRow[j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
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

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int ranks = 0;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size", rank);
    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        fail("MPI_Init_thread", "MPI_THREAD_FUNNELED is not supported", rank);
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = EXIT_SUCCESS;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    // cuBLAS and MPI counts are 32-bit.  This also prevents overflow in N*N.
    if (N == 0 || N > static_cast<size_t>(INT_MAX) || N > SIZE_MAX / N ||
        N * N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size must be positive and have at most INT_MAX elements\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm nodeComm = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                 &nodeComm),
             "MPI_Comm_split_type", rank);
    int localRank = 0;
    checkMpi(MPI_Comm_rank(nodeComm, &localRank), "MPI_Comm_rank(node communicator)", rank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        fail("cudaGetDeviceCount", "no CUDA devices are visible", rank);
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);
    checkMpi(MPI_Comm_free(&nodeComm), "MPI_Comm_free(node communicator)", rank);

    const RowPartition partition = partitionRows(N, rank, ranks);
    const size_t localElements = partition.rows * N;
    const size_t matrixElements = N * N;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks, omp_get_max_threads());
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localElements);
    std::vector<double> B(matrixElements);
    std::vector<double> localC(localElements);
    initLocalMatrixA(localA, N, partition.firstRow);
    initMatrixB(B, N);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    cublasHandle_t handle = nullptr;

    if (localElements != 0) {
        checkCuda(cudaMalloc(&deviceA, localElements * sizeof(*deviceA)), "cudaMalloc(A)", rank);
        checkCuda(cudaMalloc(&deviceB, matrixElements * sizeof(*deviceB)), "cudaMalloc(B)", rank);
        checkCuda(cudaMalloc(&deviceC, localElements * sizeof(*deviceC)), "cudaMalloc(C)", rank);
        checkCuda(cudaMemcpy(deviceA, localA.data(), localElements * sizeof(*deviceA),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(A)", rank);
        checkCuda(cudaMemcpy(deviceB, B.data(), matrixElements * sizeof(*deviceB),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(B)", rank);
        checkCublas(cublasCreate(&handle), "cublasCreate", rank);
    }

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier", rank);
    const auto start = std::chrono::steady_clock::now();
    if (localElements != 0) {
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;
        // Row-major A*B is column-major B^T*A^T.  This form avoids explicit
        // transposes and maps the local output rows directly onto deviceC.
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N),
                                 static_cast<int>(partition.rows), static_cast<int>(N), &alpha,
                                 deviceB, static_cast<int>(N), deviceA, static_cast<int>(N), &beta,
                                 deviceC, static_cast<int>(N)),
                    "cublasDgemm", rank);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", rank);
    }
    const auto end = std::chrono::steady_clock::now();

    const double localMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMilliseconds = 0.0;
    checkMpi(MPI_Reduce(&localMilliseconds, &elapsedMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD),
             "MPI_Reduce(computation time)", rank);

    if (printResults || validate) {
        if (localElements != 0) {
            checkCuda(cudaMemcpy(localC.data(), deviceC, localElements * sizeof(*deviceC),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(C)", rank);
        }
    }

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              (elapsedMilliseconds / 1000.0) / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        std::vector<double> C;
        if (rank == 0) {
            receiveCounts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
            for (int mpiRank = 0; mpiRank < ranks; ++mpiRank) {
                const RowPartition peer = partitionRows(N, mpiRank, ranks);
                receiveCounts[static_cast<size_t>(mpiRank)] = static_cast<int>(peer.rows * N);
                displacements[static_cast<size_t>(mpiRank)] = static_cast<int>(peer.firstRow * N);
            }
            C.resize(matrixElements);
        }
        checkMpi(MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE, C.data(),
                             receiveCounts.data(), displacements.data(), MPI_DOUBLE, 0,
                             MPI_COMM_WORLD),
                 "MPI_Gatherv(C)", rank);
        if (rank == 0) {
            print_results(C, "MatrixC");
        }
    }

    if (validate) {
        int localValid = validateLocalResult(localA, B, localC, N, partition) ? 1 : 0;
        int globallyValid = 0;
        checkMpi(MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
                 "MPI_Allreduce(validation)", rank);
        if (rank == 0) {
            std::printf("Validating result...\n");
            std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");
            exitCode = globallyValid ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        checkMpi(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(exit code)", rank);
    }

    if (handle != nullptr) {
        checkCublas(cublasDestroy(handle), "cublasDestroy", rank);
    }
    if (deviceC != nullptr) {
        checkCuda(cudaFree(deviceC), "cudaFree(C)", rank);
    }
    if (deviceB != nullptr) {
        checkCuda(cudaFree(deviceB), "cudaFree(B)", rank);
    }
    if (deviceA != nullptr) {
        checkCuda(cudaFree(deviceA), "cudaFree(A)", rank);
    }
    checkMpi(MPI_Finalize(), "MPI_Finalize", rank);
    return exitCode;
}
