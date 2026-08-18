#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
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

constexpr int kTileSize = 32;
constexpr int kRowsPerThread = 4;

// The row decomposition keeps A and C local to an MPI rank.  B is replicated
// once per rank so that each GPU can execute its local rows without a per-tile
// MPI synchronization.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const std::size_t rows,
                                     const std::size_t N) {
    __shared__ double tileA[kTileSize][kTileSize];
    __shared__ double tileB[kTileSize][kTileSize];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const std::size_t rowBase =
        static_cast<std::size_t>(blockIdx.y) * kTileSize +
        static_cast<std::size_t>(ty * kRowsPerThread);
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * kTileSize +
        static_cast<std::size_t>(tx);

    double accumulators[kRowsPerThread] = {0.0, 0.0, 0.0, 0.0};

    for (std::size_t tileStart = 0; tileStart < N; tileStart += kTileSize) {
        // There are 256 threads and 1024 values in each shared tile.  Each
        // thread loads four values, giving coalesced global accesses for both
        // matrices while keeping the tile size large enough for reuse.
        const unsigned int linearThread =
            threadIdx.y * blockDim.x + threadIdx.x;
        for (unsigned int index = linearThread;
             index < kTileSize * kTileSize;
             index += blockDim.x * blockDim.y) {
            const int tileRow = static_cast<int>(index / kTileSize);
            const int tileCol = static_cast<int>(index % kTileSize);
            const std::size_t aRow =
                static_cast<std::size_t>(blockIdx.y) * kTileSize + tileRow;
            const std::size_t aCol = tileStart + tileCol;
            const std::size_t bRow = tileStart + tileRow;
            const std::size_t bCol =
                static_cast<std::size_t>(blockIdx.x) * kTileSize + tileCol;

            tileA[tileRow][tileCol] =
                (aRow < rows && aCol < N) ? A[aRow * N + aCol] : 0.0;
            tileB[tileRow][tileCol] =
                (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            const double b = tileB[k][tx];
            #pragma unroll
            for (int outputRow = 0; outputRow < kRowsPerThread; ++outputRow) {
                accumulators[outputRow] +=
                    tileA[ty * kRowsPerThread + outputRow][k] * b;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int outputRow = 0; outputRow < kRowsPerThread; ++outputRow) {
        const std::size_t row = rowBase + outputRow;
        if (row < rows && col < N) {
            C[row * N + col] = accumulators[outputRow];
        }
    }
}

[[noreturn]] void abortMPI(const int rank, const char* message) {
    std::fprintf(stderr, "MPI rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const int rank, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA failure in %s: %s\n",
                     rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

// Generate pseudo-random values for matrix initialization.  This is kept
// identical to the original benchmark so that -r remains reproducible.
constexpr double getPseudoRndValue(const std::size_t N,
                                    const std::size_t i,
                                    const std::size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrixRows(std::vector<double>& matrix,
                    const std::size_t N,
                    const std::size_t firstRow,
                    const std::size_t rowCount) {
    #pragma omp parallel for schedule(static)
    for (std::int64_t localRow = 0;
         localRow < static_cast<std::int64_t>(rowCount);
         ++localRow) {
        const std::size_t globalRow =
            firstRow + static_cast<std::size_t>(localRow);
        for (std::size_t column = 0; column < N; ++column) {
            matrix[static_cast<std::size_t>(localRow) * N + column] =
                getPseudoRndValue(N, globalRow, column);
        }
    }
}

bool parseMatrixSize(const char* text, std::size_t& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
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

bool validateResult(const std::vector<double>& A,
                    const std::vector<double>& B,
                    const std::vector<double>& C,
                    const std::size_t N) {
    constexpr std::size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (std::size_t pi = 0; pi < 5; ++pi) {
        for (std::size_t pj = 0; pj < 5; ++pj) {
            const std::size_t i = checkPoints[pi] % N;
            const std::size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (std::size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relativeError =
                std::abs((actual - expected) / (expected + 1e-10));

            if (relativeError > 1e-6) {
                std::printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got %.10f "
                    "(error: %.10e)\n",
                    i, j, expected, actual, relativeError);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortMPI(rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    std::size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
            argumentsValid =
                parseMatrixSize(argv[++argument], N) && argumentsValid;
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
        }
    }

    if (!argumentsValid || showHelp) {
        if (rank == 0) {
            if (!argumentsValid) {
                std::printf("Invalid command line arguments.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (N > std::numeric_limits<std::size_t>::max() / N) {
        abortMPI(rank, "matrix size overflows the host address space");
    }
    const std::size_t matrixElements = N * N;
    if (matrixElements > std::numeric_limits<std::size_t>::max() /
                             sizeof(double)) {
        abortMPI(rank, "matrix allocation size overflows the host address space");
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        abortMPI(rank, "no CUDA devices are visible to this MPI rank");
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), rank, "cudaSetDevice");

    const std::size_t rankAsSize = static_cast<std::size_t>(rank);
    const std::size_t worldAsSize = static_cast<std::size_t>(worldSize);
    const std::size_t baseRows = N / worldAsSize;
    const std::size_t extraRows = N % worldAsSize;
    const std::size_t localRows = baseRows + (rankAsSize < extraRows ? 1 : 0);
    const std::size_t firstRow =
        rankAsSize * baseRows + std::min(rankAsSize, extraRows);
    const std::size_t localElements = localRows * N;

    std::vector<double> localA(localElements);
    std::vector<double> B(matrixElements);
    std::vector<double> localC(localElements);

    if (rank == 0) {
        std::printf("Initializing matrices...\n");
    }
    initMatrixRows(localA, N, firstRow, localRows);
    initMatrixRows(B, N, 0, N);

    cudaStream_t stream = nullptr;
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              rank, "cudaStreamCreateWithFlags");
    const std::size_t localBytes =
        std::max<std::size_t>(1, localElements) * sizeof(double);
    const std::size_t matrixBytes = matrixElements * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), localBytes),
              rank, "cudaMalloc(A)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), matrixBytes),
              rank, "cudaMalloc(B)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), localBytes),
              rank, "cudaMalloc(C)");

    if (localElements != 0) {
        checkCuda(cudaMemcpyAsync(deviceA, localA.data(), localBytes,
                                  cudaMemcpyHostToDevice, stream),
                  rank, "cudaMemcpyAsync(A)");
    }
    checkCuda(cudaMemcpyAsync(deviceB, B.data(), matrixBytes,
                              cudaMemcpyHostToDevice, stream),
              rank, "cudaMemcpyAsync(B)");
    checkCuda(cudaStreamSynchronize(stream), rank, "initial H2D copies");

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();

    if (localRows != 0) {
        const dim3 block(kTileSize, kTileSize / kRowsPerThread, 1);
        const dim3 grid(static_cast<unsigned int>((N + kTileSize - 1) /
                                                   kTileSize),
                        static_cast<unsigned int>((localRows + kTileSize - 1) /
                                                   kTileSize),
                        1);
        matrixMultiplyKernel<<<grid, block, 0, stream>>>(
            deviceA, deviceB, deviceC, localRows, N);
        checkCuda(cudaGetLastError(), rank, "matrixMultiplyKernel launch");
        checkCuda(cudaMemcpyAsync(localC.data(), deviceC, localBytes,
                                  cudaMemcpyDeviceToHost, stream),
                  rank, "cudaMemcpyAsync(C)");
    }
    checkCuda(cudaStreamSynchronize(stream), rank, "matrix multiplication");
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds =
        std::chrono::duration<double>(end - start).count();

    checkCuda(cudaFree(deviceA), rank, "cudaFree(A)");
    checkCuda(cudaFree(deviceB), rank, "cudaFree(B)");
    checkCuda(cudaFree(deviceC), rank, "cudaFree(C)");
    checkCuda(cudaStreamDestroy(stream), rank, "cudaStreamDestroy");
    MPI_Comm_free(&localCommunicator);

    double maximumSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maximumSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    std::vector<double> globalC;
    std::vector<double> globalA;
    if (validate || printResults) {
        receiveCounts.resize(static_cast<std::size_t>(worldSize));
        displacements.resize(static_cast<std::size_t>(worldSize));
        std::size_t displacement = 0;
        for (int receiver = 0; receiver < worldSize; ++receiver) {
            const std::size_t receiverRows =
                baseRows + (static_cast<std::size_t>(receiver) < extraRows ? 1 : 0);
            const std::size_t receiverElements = receiverRows * N;
            if (receiverElements > static_cast<std::size_t>(
                                      std::numeric_limits<int>::max()) ||
                displacement > static_cast<std::size_t>(
                                   std::numeric_limits<int>::max())) {
                abortMPI(rank,
                         "-v/-r matrix is too large for MPI_Gatherv counts");
            }
            receiveCounts[static_cast<std::size_t>(receiver)] =
                static_cast<int>(receiverElements);
            displacements[static_cast<std::size_t>(receiver)] =
                static_cast<int>(displacement);
            displacement += receiverElements;
        }
        if (rank == 0) {
            globalC.resize(matrixElements);
            if (validate) {
                globalA.resize(matrixElements);
            }
        }

        MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr,
                    receiveCounts.data(), displacements.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (validate) {
            MPI_Gatherv(localA.data(), static_cast<int>(localElements), MPI_DOUBLE,
                        rank == 0 ? globalA.data() : nullptr,
                        receiveCounts.data(), displacements.data(), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        const double gflops =
            maximumSeconds > 0.0
                ? (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                   static_cast<double>(N)) /
                      maximumSeconds / 1e9
                : 0.0;
        const long durationMilliseconds =
            static_cast<long>(maximumSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMilliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(globalC, "MatrixC");
        }
    }

    int validationPassed = 1;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
            validationPassed = validateResult(globalA, B, globalC, N) ? 1 : 0;
            std::printf("Validation: %s\n",
                        validationPassed == 1 ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validationPassed == 1 ? 0 : 1;
}
