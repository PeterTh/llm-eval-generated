#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int TileSize = 32;

// Generate the same deterministic values as the original benchmark.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// The matrix is initialized in parallel on the host.  Collapsed row/column
// OpenMP work distribution keeps all ranks busy even when N is not large.
void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t firstRow, const size_t rows) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixRows(mat, N, 0, N);
}

// Each CUDA thread computes one C element.  The shared-memory tiles preserve
// the original left-to-right k accumulation order while substantially
// reducing global-memory traffic for B and A.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t rows, const size_t N) {
    __shared__ double tileA[TileSize][TileSize];
    __shared__ double tileB[TileSize][TileSize];

    const size_t localRow = static_cast<size_t>(blockIdx.y) * TileSize +
                            static_cast<size_t>(threadIdx.y);
    const size_t col = static_cast<size_t>(blockIdx.x) * TileSize +
                       static_cast<size_t>(threadIdx.x);
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    double sum = 0.0;
    for (size_t tileStart = 0; tileStart < N; tileStart += TileSize) {
        const size_t aColumn = tileStart + static_cast<size_t>(tx);
        const size_t bRow = tileStart + static_cast<size_t>(ty);

        tileA[ty][tx] = (localRow < rows && aColumn < N)
                             ? A[localRow * N + aColumn]
                             : 0.0;
        tileB[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        const size_t remaining = N - tileStart;
        const int tileWidth = remaining < TileSize
                                  ? static_cast<int>(remaining)
                                  : TileSize;
        #pragma unroll
        for (int k = 0; k < tileWidth; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }

    if (localRow < rows && col < N) {
        C[localRow * N + col] = sum;
    }
}

[[noreturn]] void abortMPI(const char* message, const int rank) {
    if (rank == 0) {
        std::fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkMPI(const int error, const char* operation, const int rank) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char errorString[MPI_MAX_ERROR_STRING] = {};
    int errorLength = 0;
    MPI_Error_string(error, errorString, &errorLength);
    char message[512] = {};
    std::snprintf(message, sizeof(message), "MPI error in %s: %.*s", operation,
                  errorLength, errorString);
    abortMPI(message, rank);
}

void checkCUDA(const cudaError_t error, const char* operation, const int rank) {
    if (error == cudaSuccess) {
        return;
    }

    char message[512] = {};
    std::snprintf(message, sizeof(message), "CUDA error in %s: %s", operation,
                  cudaGetErrorString(error));
    abortMPI(message, rank);
}

// MPI's classic interfaces use an int count.  Broadcasting in chunks keeps
// the benchmark usable for matrices whose element count exceeds INT_MAX.
void broadcastDoubles(double* data, const size_t count, const int root,
                      const int rank) {
    constexpr size_t maxMPIChunk =
        static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count;) {
        const size_t chunk = std::min(maxMPIChunk, count - offset);
        checkMPI(MPI_Bcast(data + offset, static_cast<int>(chunk), MPI_DOUBLE,
                           root, MPI_COMM_WORLD),
                 "MPI_Bcast", rank);
        offset += chunk;
    }
}

void selectCUDADevice(const int rank, int& device) {
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    checkMPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localCommunicator),
             "MPI_Comm_split_type", rank);

    int localRank = 0;
    checkMPI(MPI_Comm_rank(localCommunicator, &localRank), "MPI_Comm_rank",
             rank);
    checkMPI(MPI_Comm_free(&localCommunicator), "MPI_Comm_free", rank);

    int deviceCount = 0;
    checkCUDA(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount <= 0) {
        abortMPI("No CUDA devices are available to the MPI rank", rank);
    }

    device = localRank % deviceCount;
    checkCUDA(cudaSetDevice(device), "cudaSetDevice", rank);
}

void matrixMultiplyCUDA(const std::vector<double>& A,
                        const std::vector<double>& B,
                        std::vector<double>& C, const size_t rows,
                        const size_t N, const int rank) {
    if (rows == 0 || N == 0) {
        return;
    }

    const size_t localElements = rows * N;
    const size_t matrixElements = N * N;
    const size_t bytesA = localElements * sizeof(double);
    const size_t bytesB = matrixElements * sizeof(double);
    const size_t bytesC = localElements * sizeof(double);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    cudaStream_t stream = nullptr;

    checkCUDA(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytesA),
              "cudaMalloc(A)", rank);
    checkCUDA(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytesB),
              "cudaMalloc(B)", rank);
    checkCUDA(cudaMalloc(reinterpret_cast<void**>(&deviceC), bytesC),
              "cudaMalloc(C)", rank);
    checkCUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreate", rank);

    checkCUDA(cudaMemcpyAsync(deviceA, A.data(), bytesA,
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(A)", rank);
    checkCUDA(cudaMemcpyAsync(deviceB, B.data(), bytesB,
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(B)", rank);

    const dim3 block(TileSize, TileSize, 1);
    const dim3 grid(static_cast<unsigned int>((N + TileSize - 1) / TileSize),
                    static_cast<unsigned int>((rows + TileSize - 1) /
                                               TileSize),
                    1);
    matrixMultiplyKernel<<<grid, block, 0, stream>>>(deviceA, deviceB, deviceC,
                                                     rows, N);
    checkCUDA(cudaGetLastError(), "matrixMultiplyKernel launch", rank);

    checkCUDA(cudaMemcpyAsync(C.data(), deviceC, bytesC,
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(C)", rank);
    checkCUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize", rank);

    checkCUDA(cudaStreamDestroy(stream), "cudaStreamDestroy", rank);
    checkCUDA(cudaFree(deviceC), "cudaFree(C)", rank);
    checkCUDA(cudaFree(deviceB), "cudaFree(B)", rank);
    checkCUDA(cudaFree(deviceA), "cudaFree(A)", rank);
}

bool validateResult(const std::vector<double>& A,
                    const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return C.empty();
    }

    // Check a few deterministic positions, matching the original benchmark.
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got "
                    "%.10f (error: %.10e)\n",
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

bool parseSize(const char* value, size_t& result) {
    if (value == nullptr || *value == '\0' || *value == '-') {
        return false;
    }

    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (*end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    result = static_cast<size_t>(parsed);
    return true;
}

bool checkedSquare(const size_t N, size_t& result) {
    if (N != 0 && N > std::numeric_limits<size_t>::max() / N) {
        return false;
    }
    result = N * N;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    checkMPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                             &providedThreadLevel),
             "MPI_Init_thread", 0);

    int rank = 0;
    int worldSize = 1;
    checkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    checkMPI(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size", rank);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortMPI("MPI implementation does not provide MPI_THREAD_FUNNELED", rank);
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], N)) {
                parseError = true;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp && !parseError) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (parseError) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    size_t matrixElements = 0;
    if (!checkedSquare(N, matrixElements)) {
        abortMPI("Matrix size overflows the addressable element count", rank);
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Contiguous row partitioning gives every rank a complete C row range and
    // avoids communicating A: each rank regenerates its deterministic rows.
    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows);
    const size_t firstRow = baseRows * static_cast<size_t>(rank) +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localElements = localRows * N;

    std::vector<double> B(matrixElements);
    std::vector<double> localA(localElements);
    std::vector<double> localC(localElements);

    if (rank == 0) {
        std::printf("Initializing matrices...\n");
        initMatrix(B, N);
    }
    initMatrixRows(localA, N, firstRow, localRows);
    broadcastDoubles(B.data(), matrixElements, 0, rank);

    int device = 0;
    selectCUDADevice(rank, device);
    (void)device;

    const bool gatherResult = printResults || validate;
    std::vector<double> C;
    std::vector<double> A;
    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    if (rank == 0 && gatherResult) {
        if (matrixElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
            abortMPI("Matrix is too large for MPI_Gatherv counts", rank);
        }
        C.resize(matrixElements);
        receiveCounts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
        size_t displacement = 0;
        for (int process = 0; process < worldSize; ++process) {
            const size_t processRows =
                baseRows + (static_cast<size_t>(process) < extraRows);
            const size_t processElements = processRows * N;
            if (processElements >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
                abortMPI("Local matrix is too large for MPI_Gatherv counts", rank);
            }
            receiveCounts[static_cast<size_t>(process)] =
                static_cast<int>(processElements);
            displacements[static_cast<size_t>(process)] =
                static_cast<int>(displacement);
            displacement += processElements;
        }
        if (validate) {
            A.resize(matrixElements);
            initMatrix(A, N);
        }
    }

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    checkMPI(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier", rank);
    const auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyCUDA(localA, B, localC, localRows, N, rank);

    if (gatherResult) {
        checkMPI(MPI_Gatherv(
                     localC.empty() ? nullptr : localC.data(),
                     static_cast<int>(localElements), MPI_DOUBLE,
                     rank == 0 && !C.empty() ? C.data() : nullptr,
                     rank == 0 ? receiveCounts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE,
                     0, MPI_COMM_WORLD),
                 "MPI_Gatherv", rank);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const double localMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMilliseconds = 0.0;
    checkMPI(MPI_Reduce(&localMilliseconds, &elapsedMilliseconds, 1, MPI_DOUBLE,
                        MPI_MAX, 0, MPI_COMM_WORLD),
             "MPI_Reduce", rank);

    int exitCode = 0;
    if (rank == 0) {
        const long durationMilliseconds = std::max(
            1L, static_cast<long>(std::llround(elapsedMilliseconds)));
        std::printf("Computation time: %ld ms\n", durationMilliseconds);

        const double seconds = elapsedMilliseconds / 1000.0;
        const double gflops = seconds > 0.0
                                  ? (2.0 * static_cast<double>(N) *
                                     static_cast<double>(N) *
                                     static_cast<double>(N)) /
                                        seconds / 1e9
                                  : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(A, B, C, N)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    checkMPI(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD),
             "MPI_Bcast(exitCode)", rank);
    checkMPI(MPI_Finalize(), "MPI_Finalize", rank);
    return exitCode;
}
