#include <mpi.h>
#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr int TILE = 32;

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t error, const char* what, int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, what, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, rank)

// C's rows are distributed across MPI ranks.  This tiled kernel is deliberately
// independent of the global row offset: each rank supplies its local A rows.
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t localRows, size_t N) {
    __shared__ double aTile[TILE][TILE];
    __shared__ double bTile[TILE][TILE];

    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t base = 0; base < N; base += TILE) {
        const size_t aCol = base + threadIdx.x;
        const size_t bRow = base + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (row < localRows && aCol < N) ? A[row * N + aCol] : 0.0;
        bTile[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        __syncthreads();
    }
    if (row < localRows && col < N) C[row * N + col] = sum;
}

static void initLocalMatrix(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(rows); ++local) {
        const size_t row = firstRow + static_cast<size_t>(local);
        for (size_t j = 0; j < N; ++j) mat[static_cast<size_t>(local) * N + j] = getPseudoRndValue(N, row, j);
    }
}

static void initMatrix(std::vector<double>& mat, size_t N) { initLocalMatrix(mat, N, 0, N); }

static bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                           const std::vector<double>& C, size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) for (size_t pj = 0; pj < 5; ++pj) {
        const size_t i = checkPoints[pi] % N, j = checkPoints[pj] % N;
        double expected = 0.0;
        for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
        const double actual = C[i * N + j];
        const double relError = std::abs((actual - expected) / (expected + 1e-10));
        if (relError > 1e-6) {
            std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
            return false;
        }
    }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512; bool validate = false, printResults = false;
    int badArgs = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else badArgs = 1;
    }
    if (badArgs || N == 0 || N > static_cast<size_t>(INT_MAX) || N > static_cast<size_t>(INT_MAX) / N) {
        if (!rank) { std::printf("Invalid matrix size or option\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const size_t firstRow = N * static_cast<size_t>(rank) / ranks;
    const size_t endRow = N * static_cast<size_t>(rank + 1) / ranks;
    const size_t localRows = endRow - firstRow;
    const size_t localElements = localRows * N, totalElements = N * N;
    std::vector<double> localA(localElements), B(totalElements), localC(localElements);
    std::vector<double> C; if (!rank) C.resize(totalElements);

    if (!rank) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nInitializing matrices...\n", N, N, validate ? "enabled" : "disabled");
    }
    initLocalMatrix(localA, N, firstRow, localRows);
    initMatrix(B, N);

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    if (localElements) {
        CUDA_CHECK(cudaMalloc(&deviceA, localElements * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&deviceC, localElements * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(deviceA, localA.data(), localElements * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&deviceB, totalElements * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(deviceB, B.data(), totalElements * sizeof(double), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) std::printf("Computing matrix multiplication...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    if (localRows) {
        const dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);
        matmulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, localRows, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localC.data(), deviceC, localElements * sizeof(double), cudaMemcpyDeviceToHost));
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = N * static_cast<size_t>(r) / ranks;
        const size_t finish = N * static_cast<size_t>(r + 1) / ranks;
        counts[r] = static_cast<int>((finish - begin) * N); displs[r] = static_cast<int>(begin * N);
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE, rank ? nullptr : C.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(deviceA)); CUDA_CHECK(cudaFree(deviceB)); CUDA_CHECK(cudaFree(deviceC));

    int result = 0;
    if (!rank) {
        const long durationMs = static_cast<long>(std::llround(seconds * 1000.0));
        std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", durationMs, (2.0 * N * N * N) / seconds / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            std::vector<double> fullA(totalElements); initMatrix(fullA, N);
            if (validateResult(fullA, B, C, N)) std::printf("Validation: PASSED\n");
            else { std::printf("Validation: FAILED\n"); result = 1; }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
