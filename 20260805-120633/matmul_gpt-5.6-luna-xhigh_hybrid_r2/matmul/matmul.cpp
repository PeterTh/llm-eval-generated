#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// Generate the same deterministic values as the original benchmark.
constexpr double getPseudoRndValue(const std::size_t N,
                                   const std::size_t i,
                                   const std::size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& matrix, const std::size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            matrix[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Initialize only the rows owned by one MPI rank.  The global row offset
// preserves exactly the same input matrix as the serial implementation.
void initMatrixRows(std::vector<double>& matrix,
                    const std::size_t N,
                    const std::size_t firstRow,
                    const std::size_t rowCount) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t localRow = 0; localRow < rowCount; ++localRow) {
        for (std::size_t j = 0; j < N; ++j) {
            matrix[localRow * N + j] =
                getPseudoRndValue(N, firstRow + localRow, j);
        }
    }
}

bool validateResult(const std::vector<double>& A,
                   const std::vector<double>& B,
                   const std::vector<double>& C,
                   const std::size_t N) {
    constexpr std::array<std::size_t, 5> checkPoints = {0, 1, 2, 3, 4};
    std::array<double, 25> expected{};
    std::array<double, 25> actual{};

    // Each check is independent.  Keep the inner k loop serial so that the
    // reference sum retains the original operation order.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const std::size_t i = checkPoints[pi] % N;
            const std::size_t j = checkPoints[pj] % N;
            double sum = 0.0;
            for (std::size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            const std::size_t index = static_cast<std::size_t>(pi * 5 + pj);
            expected[index] = sum;
            actual[index] = C[i * N + j];
        }
    }

    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const std::size_t index = static_cast<std::size_t>(pi * 5 + pj);
            const double relError = std::abs(
                (actual[index] - expected[index]) / (expected[index] + 1e-10));
            if (relError > 1e-6) {
                const std::size_t i = checkPoints[pi] % N;
                const std::size_t j = checkPoints[pj] % N;
                std::printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got %.10f "
                    "(error: %.10e)\n",
                    i, j, expected[index], actual[index], relError);
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

[[noreturn]] void abortMpi(const char* message,
                           const int rank,
                           const int errorCode = 1) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, errorCode);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* expression, const int rank) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA call %s failed: %s",
                      expression, cudaGetErrorString(status));
        abortMpi(message, rank);
    }
}

void checkCublas(const cublasStatus_t status,
                 const char* expression,
                 const int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[256];
        std::snprintf(message, sizeof(message),
                      "cuBLAS call %s failed (status %d)",
                      expression, static_cast<int>(status));
        abortMpi(message, rank);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, rank)
#define CUBLAS_CHECK(call) checkCublas((call), #call, rank)

std::size_t parseMatrixSize(const char* value, const int rank) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (value[0] == '\0' || end == value || *end != '\0' || parsed == 0) {
        abortMpi("Matrix size must be a positive integer", rank);
    }
    if (parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        abortMpi("Matrix size is too large for this platform", rank);
    }
    return static_cast<std::size_t>(parsed);
}

// MPI counts are int in the portable MPI-3 C API.  Chunking keeps the input
// broadcast correct for large matrices without imposing a small-size limit.
void broadcastDoubles(double* data,
                      const std::size_t count,
                      const int root,
                      const int rank) {
    constexpr std::size_t maxCount =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (std::size_t offset = 0; offset < count;) {
        const std::size_t chunk = std::min(maxCount, count - offset);
        const int mpiCount = static_cast<int>(chunk);
        if (MPI_Bcast(data + offset, mpiCount, MPI_DOUBLE, root,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
            abortMpi("MPI_Bcast failed", rank);
        }
        offset += chunk;
    }
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        abortMpi("MPI implementation does not provide MPI_THREAD_FUNNELED", rank);
    }

    std::size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = parseMatrixSize(argv[++i], rank);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (N > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        abortMpi("Matrix size exceeds the CUDA/cuBLAS dimension limit", rank);
    }
    if (N > std::numeric_limits<std::size_t>::max() / N) {
        abortMpi("Matrix allocation size overflows size_t", rank);
    }
    const std::size_t matrixElements = N * N;

    // Select a GPU by node-local MPI rank.  This remains correct when the
    // launcher assigns global ranks across multiple accelerator nodes.
    MPI_Comm localComm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &localComm) != MPI_SUCCESS) {
        abortMpi("MPI_Comm_split_type failed", rank);
    }
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        abortMpi("No CUDA devices are available", rank);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    const std::size_t firstRow =
        (N * static_cast<std::size_t>(rank)) /
        static_cast<std::size_t>(worldSize);
    const std::size_t lastRow =
        (N * static_cast<std::size_t>(rank + 1)) /
        static_cast<std::size_t>(worldSize);
    const std::size_t localRows = lastRow - firstRow;
    const std::size_t localElements = localRows * N;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // B is shared by every row tile.  Initialize it once and broadcast it;
    // A is generated independently for each rank's owned rows.
    std::vector<double> B(matrixElements);
    std::vector<double> localA;
    std::vector<double> A;
    const double* hostA = nullptr;

    if (rank == 0) {
        std::printf("Initializing matrices...\n");
        initMatrix(B, N);
        if (validate) {
            A.resize(matrixElements);
            initMatrix(A, N);
            hostA = A.data() + firstRow * N;
        } else {
            localA.resize(localElements);
            initMatrixRows(localA, N, firstRow, localRows);
            hostA = localA.data();
        }
    } else {
        localA.resize(localElements);
        initMatrixRows(localA, N, firstRow, localRows);
        hostA = localA.data();
    }
    broadcastDoubles(B.data(), matrixElements, 0, rank);

    std::vector<double> localC(localElements);
    std::vector<double> C;
    if (rank == 0 && (validate || printResults)) {
        C.resize(matrixElements);
    }

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    const std::size_t safeLocalElements = std::max<std::size_t>(localElements, 1);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceA),
                          safeLocalElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceB),
                          matrixElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceC),
                          safeLocalElements * sizeof(double)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (localElements > 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceA, hostA,
                                   localElements * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(deviceB, B.data(), matrixElements * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    if (localRows > 0) {
        // Row-major matrices are viewed by cuBLAS as transposed column-major
        // matrices: C = A*B becomes C^T = B^T*A^T.  This avoids a costly
        // explicit transpose while retaining the original matrix semantics.
        const int dimension = static_cast<int>(N);
        const int localDimension = static_cast<int>(localRows);
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 dimension, localDimension, dimension,
                                 &alpha,
                                 deviceB, dimension,
                                 deviceA, dimension,
                                 &beta,
                                 deviceC, dimension));
        CUDA_CHECK(cudaMemcpyAsync(localC.data(), deviceC,
                                   localElements * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // Gather only when requested.  The benchmark's normal path avoids moving
    // the full result back through MPI, while -r and -v retain their original
    // complete-result behavior.
    if (validate || printResults) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            receiveCounts.resize(static_cast<std::size_t>(worldSize));
            displacements.resize(static_cast<std::size_t>(worldSize));
            for (int r = 0; r < worldSize; ++r) {
                const std::size_t rFirst =
                    (N * static_cast<std::size_t>(r)) /
                    static_cast<std::size_t>(worldSize);
                const std::size_t rLast =
                    (N * static_cast<std::size_t>(r + 1)) /
                    static_cast<std::size_t>(worldSize);
                const std::size_t count = (rLast - rFirst) * N;
                const std::size_t displacement = rFirst * N;
                if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                    displacement > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    abortMpi("MPI_Gatherv count/displacement exceeds MPI int range", rank);
                }
                receiveCounts[static_cast<std::size_t>(r)] = static_cast<int>(count);
                displacements[static_cast<std::size_t>(r)] =
                    static_cast<int>(displacement);
            }
        }
        const int localCount = static_cast<int>(localElements);
        if (MPI_Gatherv(localC.data(), localCount, MPI_DOUBLE,
                        rank == 0 ? C.data() : nullptr,
                        rank == 0 ? receiveCounts.data() : nullptr,
                        rank == 0 ? displacements.data() : nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
            abortMpi("MPI_Gatherv failed", rank);
        }
    }

    if (rank == 0) {
        const auto duration = std::chrono::milliseconds(
            std::max<long long>(1, static_cast<long long>(elapsed * 1000.0)));
        std::printf("Computation time: %lld ms\n",
                    static_cast<long long>(duration.count()));
        const double seconds = static_cast<double>(duration.count()) / 1000.0;
        const double gflops = (2.0 * static_cast<double>(N) *
                               static_cast<double>(N) * static_cast<double>(N)) /
                              seconds / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
            valid = validateResult(A, B, C, N) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceC));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceA));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return valid == 0 ? 1 : 0;
}
