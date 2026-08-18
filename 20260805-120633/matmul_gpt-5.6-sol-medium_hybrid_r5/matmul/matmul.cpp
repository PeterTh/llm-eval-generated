#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t error, const char* operation, int rank) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 2);
}

// A 32x32 output tile is computed by 256 threads.  Each thread accumulates four
// rows, increasing arithmetic intensity without requiring excessive registers.
template <int TILE = 32, int ROWS_PER_THREAD = 4>
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t localRows,
                                     size_t N) {
    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;
    const size_t row0 = static_cast<size_t>(blockIdx.y) * TILE + ty;
    double sums[ROWS_PER_THREAD] = {};

    for (size_t kb = 0; kb < N; kb += TILE) {
#pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const int tileRow = ty + q * blockDim.y;
            const size_t row = row0 + static_cast<size_t>(q * blockDim.y);
            const size_t k = kb + tx;
            As[tileRow][tx] = (row < localRows && k < N) ? A[row * N + k] : 0.0;

            const size_t bk = kb + tileRow;
            Bs[tileRow][tx] = (bk < N && col < N) ? B[bk * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double b = Bs[k][tx];
#pragma unroll
            for (int q = 0; q < ROWS_PER_THREAD; ++q)
                sums[q] = fma(As[ty + q * blockDim.y][k], b, sums[q]);
        }
        __syncthreads();
    }

    if (col < N) {
#pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const size_t row = row0 + static_cast<size_t>(q * blockDim.y);
            if (row < localRows) C[row * N + col] = sums[q];
        }
    }
}

static void initRows(std::vector<double>& matrix, size_t firstRow,
                     size_t rows, size_t N) {
#pragma omp parallel for schedule(static)
    for (std::int64_t lr = 0; lr < static_cast<std::int64_t>(rows); ++lr) {
        const size_t globalRow = firstRow + static_cast<size_t>(lr);
        for (size_t j = 0; j < N; ++j)
            matrix[static_cast<size_t>(lr) * N + j] =
                getPseudoRndValue(N, globalRow, j);
    }
}

static bool validateLocal(const std::vector<double>& A,
                          const std::vector<double>& B,
                          const std::vector<double>& C, size_t firstRow,
                          size_t localRows, size_t N, int rank) {
    int failures = 0;
#pragma omp parallel for collapse(2) reduction(+ : failures) schedule(static)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = static_cast<size_t>(pi) % N;
            const size_t j = static_cast<size_t>(pj) % N;
            if (i < firstRow || i >= firstRow + localRows) continue;
            const size_t lr = i - firstRow;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[lr * N + k] * B[k * N + j];
            const double actual = C[lr * N + j];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (error > 1e-6 || !std::isfinite(actual)) {
#pragma omp critical
                std::fprintf(stderr,
                             "Rank %d validation failed at (%zu, %zu): "
                             "expected %.10f, got %.10f (error %.10e)\n",
                             rank, i, j, expected, actual, error);
                ++failures;
            }
        }
    }
    return failures == 0;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false, help = false;
    int parseOk = 1;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                char* end = nullptr;
                const unsigned long long value = std::strtoull(argv[++i], &end, 10);
                if (*end != '\0' || value == 0 || value > std::numeric_limits<size_t>::max())
                    parseOk = 0;
                else
                    N = static_cast<size_t>(value);
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                help = true;
            } else {
                std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
                parseOk = 0;
            }
        }
        if (!parseOk || help) printUsage(argv[0]);
    }
    std::uint64_t wireN = static_cast<std::uint64_t>(N);
    int flags[4] = {parseOk, validate, printResults, help};
    MPI_Bcast(&wireN, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 4, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(wireN);
    validate = flags[1] != 0;
    printResults = flags[2] != 0;
    if (!flags[0] || flags[3]) {
        MPI_Finalize();
        return flags[0] ? 0 : 1;
    }
    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double)) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is too large for addressable memory\n");
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t extra = N % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extra);

    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA accelerators available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks,
                    omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> A(localRows * N), B(N * N), C(localRows * N);
    initRows(A, firstRow, localRows, N);
    initRows(B, 0, N, N);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t localBytes = localRows * N * sizeof(double);
    const size_t matrixBytes = N * N * sizeof(double);
    if (localBytes) {
        cudaCheck(cudaMalloc(&dA, localBytes), "cudaMalloc(A)", rank);
        cudaCheck(cudaMalloc(&dC, localBytes), "cudaMalloc(C)", rank);
        cudaCheck(cudaMemcpy(dA, A.data(), localBytes, cudaMemcpyHostToDevice),
                  "copy A to device", rank);
    }
    cudaCheck(cudaMalloc(&dB, matrixBytes), "cudaMalloc(B)", rank);
    cudaCheck(cudaMemcpy(dB, B.data(), matrixBytes, cudaMemcpyHostToDevice),
              "copy B to device", rank);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    if (localRows) {
        const dim3 block(32, 8);
        const dim3 grid(static_cast<unsigned>((N + 31) / 32),
                        static_cast<unsigned>((localRows + 31) / 32));
        matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, localRows, N);
        cudaCheck(cudaGetLastError(), "matrixMultiplyKernel launch", rank);
    }
    cudaCheck(cudaDeviceSynchronize(), "matrixMultiplyKernel execution", rank);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localBytes)
        cudaCheck(cudaMemcpy(C.data(), dC, localBytes, cudaMemcpyDeviceToHost),
                  "copy C to host", rank);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double n = static_cast<double>(N);
        std::printf("Performance: %.3f GFLOPS\n", 2.0 * n * n * n / seconds / 1e9);
    }

    int locallyValid = !validate || validateLocal(A, B, C, firstRow, localRows, N, rank);
    int globallyValid = 0;
    MPI_Reduce(&locallyValid, &globallyValid, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

    if (printResults) {
        if (N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) std::fprintf(stderr, "Result gathering exceeds MPI count limits\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        std::vector<int> counts, displacements;
        std::vector<double> gathered;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            gathered.resize(N * N);
            size_t offset = 0;
            for (int r = 0; r < ranks; ++r) {
                const size_t rows = baseRows + (static_cast<size_t>(r) < extra);
                counts[r] = static_cast<int>(rows * N);
                displacements[r] = static_cast<int>(offset);
                offset += rows * N;
            }
        }
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) print_results(gathered, "MatrixC");
    }

    int result = 0;
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");
        result = globallyValid ? 0 : 1;
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
