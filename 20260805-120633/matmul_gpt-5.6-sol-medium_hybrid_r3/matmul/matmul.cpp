#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

static void initializeMatrices(std::vector<double>& localA,
                               std::vector<double>& B, size_t N,
                               size_t firstRow, size_t localRows) {
#pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(localRows); ++row) {
        const size_t globalRow = firstRow + static_cast<size_t>(row);
        for (size_t col = 0; col < N; ++col) {
            localA[static_cast<size_t>(row) * N + col] =
                getPseudoRndValue(N, globalRow, col);
        }
    }

#pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(N); ++row) {
        for (size_t col = 0; col < N; ++col) {
            B[static_cast<size_t>(row) * N + col] =
                getPseudoRndValue(N, static_cast<size_t>(row), col);
        }
    }
}

// A 32x32 output tile is produced by 256 threads. Each thread accumulates four
// output values, which gives good instruction-level parallelism while keeping
// the shared-memory loads fully coalesced.
constexpr int TILE = 32;
constexpr int THREAD_ROWS = 8;

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N,
                                     size_t localRows) {
    __shared__ double tileA[TILE][TILE + 1];
    __shared__ double tileB[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE;
    double sums[TILE / THREAD_ROWS] = {0.0, 0.0, 0.0, 0.0};

    for (size_t tile = 0; tile < N; tile += TILE) {
#pragma unroll
        for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
            const int localRow = ty + q * THREAD_ROWS;
            const size_t row = rowBase + localRow;
            const size_t k = tile + tx;
            tileA[localRow][tx] =
                (row < localRows && k < N) ? A[row * N + k] : 0.0;

            const int linear = (ty * TILE + tx) + q * (THREAD_ROWS * TILE);
            const int bRow = linear / TILE;
            const int bCol = linear % TILE;
            const size_t globalBRow = tile + bRow;
            const size_t globalBCol =
                static_cast<size_t>(blockIdx.x) * TILE + bCol;
            tileB[bRow][bCol] = (globalBRow < N && globalBCol < N)
                                    ? B[globalBRow * N + globalBCol]
                                    : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double b = tileB[k][tx];
#pragma unroll
            for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
                sums[q] += tileA[ty + q * THREAD_ROWS][k] * b;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        const size_t row = rowBase + ty + q * THREAD_ROWS;
        if (row < localRows && col < N) C[row * N + col] = sums[q];
    }
}

static void gatherResult(const std::vector<double>& localC,
                         std::vector<double>& C, size_t N, size_t firstRow,
                         size_t localRows, int rank, int ranks) {
    // MPI counts are int. Gather the matrix in row windows so large practical
    // matrices are not constrained by a single collective's count limit.
    const size_t rowsPerWindow = static_cast<size_t>(INT_MAX) / N;
    std::vector<int> counts(static_cast<size_t>(ranks));
    std::vector<int> displacements(static_cast<size_t>(ranks));

    for (size_t windowFirst = 0; windowFirst < N;
         windowFirst += rowsPerWindow) {
        const size_t windowRows = std::min(rowsPerWindow, N - windowFirst);
        const size_t windowLast = windowFirst + windowRows;
        const size_t localFirst = std::max(firstRow, windowFirst);
        const size_t localLast = std::min(firstRow + localRows, windowLast);
        const int sendCount = localLast > localFirst
                                  ? static_cast<int>((localLast - localFirst) * N)
                                  : 0;
        const double* sendBuffer = localC.data() +
                                   (localFirst > firstRow
                                        ? (localFirst - firstRow) * N
                                        : 0);

        if (rank == 0) {
            for (int r = 0; r < ranks; ++r) {
                const size_t rFirst = (N * static_cast<size_t>(r)) / ranks;
                const size_t rLast =
                    (N * static_cast<size_t>(r + 1)) / ranks;
                const size_t overlapFirst = std::max(rFirst, windowFirst);
                const size_t overlapLast = std::min(rLast, windowLast);
                counts[static_cast<size_t>(r)] =
                    overlapLast > overlapFirst
                        ? static_cast<int>((overlapLast - overlapFirst) * N)
                        : 0;
                displacements[static_cast<size_t>(r)] =
                    static_cast<int>((overlapFirst - windowFirst) * N);
            }
        }
        MPI_Gatherv(sendBuffer, sendCount, MPI_DOUBLE,
                    rank == 0 ? C.data() + windowFirst * N : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }
}

static bool validateResult(const std::vector<double>& C, size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) *
                            getPseudoRndValue(N, k, j);
            }
            const double actual = C[i * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, "
                            "got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            const bool sizeValid = errno == 0 && end != argv[i] && *end == '\0' &&
                                   value > 0 &&
                                   value <= std::numeric_limits<size_t>::max();
            argumentsValid = argumentsValid && sizeValid;
            if (sizeValid) N = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }
    if (N > std::numeric_limits<size_t>::max() / N ||
        N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int gpuCount = 0;
    cudaCheck(cudaGetDeviceCount(&gpuCount), "cudaGetDeviceCount", rank);
    if (gpuCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA-capable GPU is available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(localRank % gpuCount), "cudaSetDevice", rank);
    MPI_Comm_free(&localComm);

    const size_t firstRow = (N * static_cast<size_t>(rank)) / ranks;
    const size_t lastRow = (N * static_cast<size_t>(rank + 1)) / ranks;
    const size_t localRows = lastRow - firstRow;
    const size_t localElements = localRows * N;
    const size_t matrixElements = N * N;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks,
                    omp_get_max_threads());
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localElements);
    std::vector<double> B(matrixElements);
    std::vector<double> localC(localElements);
    initializeMatrices(localA, B, N, firstRow, localRows);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    if (localElements != 0) {
        cudaCheck(cudaMalloc(&deviceA, localElements * sizeof(double)), "cudaMalloc(A)", rank);
        cudaCheck(cudaMalloc(&deviceC, localElements * sizeof(double)), "cudaMalloc(C)", rank);
        cudaCheck(cudaMemcpy(deviceA, localA.data(), localElements * sizeof(double),
                             cudaMemcpyHostToDevice), "copy A to GPU", rank);
    }
    cudaCheck(cudaMalloc(&deviceB, matrixElements * sizeof(double)), "cudaMalloc(B)", rank);
    cudaCheck(cudaMemcpy(deviceB, B.data(), matrixElements * sizeof(double),
                         cudaMemcpyHostToDevice), "copy B to GPU", rank);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows != 0) {
        const dim3 threads(TILE, THREAD_ROWS);
        const dim3 blocks(static_cast<unsigned int>((N + TILE - 1) / TILE),
                          static_cast<unsigned int>((localRows + TILE - 1) / TILE));
        matrixMultiplyKernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, N,
                                                  localRows);
        cudaCheck(cudaGetLastError(), "matrixMultiplyKernel launch", rank);
    }
    cudaCheck(cudaDeviceSynchronize(), "matrixMultiplyKernel execution", rank);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (localElements != 0) {
        cudaCheck(cudaMemcpy(localC.data(), deviceC, localElements * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy C from GPU", rank);
    }
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        const double operations = 2.0 * static_cast<double>(N) * N * N;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", operations / seconds / 1e9);
    }

    std::vector<double> C;
    if (validate || printResults) {
        if (rank == 0) C.resize(matrixElements);
        gatherResult(localC, C, N, firstRow, localRows, rank, ranks);
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(C, N)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
