#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr int TILE = 32;

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t N, size_t localRows) {
    __shared__ double aTile[TILE][TILE];
    __shared__ double bTile[TILE][TILE];

    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;

    for (size_t tile = 0; tile < N; tile += TILE) {
        const size_t aCol = tile + threadIdx.x;
        const size_t bRow = tile + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (row < localRows && aCol < N) ? A[row * N + aCol] : 0.0;
        bTile[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        __syncthreads();
    }
    if (row < localRows && col < N)
        C[row * N + col] = sum;
}

static void initLocalMatrix(std::vector<double>& matrix, size_t N, size_t firstRow, size_t rows) {
    #pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(rows); ++row) {
        const size_t globalRow = firstRow + static_cast<size_t>(row);
        for (size_t col = 0; col < N; ++col)
            matrix[static_cast<size_t>(row) * N + col] = getPseudoRndValue(N, globalRow, col);
    }
}

static bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                           const std::vector<double>& C, size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N, j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n  -n <num>     Matrix size N (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (N == 0 || N > std::numeric_limits<size_t>::max() / N || N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "Matrix size must be nonzero and fit MPI counts.\n");
        MPI_Finalize(); return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) { if (!rank) std::fprintf(stderr, "No CUDA devices available.\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
    MPI_Comm_free(&localComm);

    const size_t firstRow = (N * static_cast<size_t>(rank)) / ranks;
    const size_t localRows = (N * static_cast<size_t>(rank + 1)) / ranks - firstRow;
    std::vector<double> localA(localRows * N), B(N * N), localC(localRows * N);
    initLocalMatrix(localA, N, firstRow, localRows);
    initLocalMatrix(B, N, 0, N);

    const size_t localElements = localRows * N;
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    // Give inactive ranks a valid device pointer; this also permits ranks > matrix rows.
    checkCuda(cudaMalloc(&deviceA, (localElements ? localElements : 1) * sizeof(double)), "cudaMalloc(A)", rank);
    checkCuda(cudaMalloc(&deviceB, B.size() * sizeof(double)), "cudaMalloc(B)", rank);
    checkCuda(cudaMalloc(&deviceC, (localElements ? localElements : 1) * sizeof(double)), "cudaMalloc(C)", rank);
    if (localElements)
        checkCuda(cudaMemcpy(deviceA, localA.data(), localElements * sizeof(double), cudaMemcpyHostToDevice), "copy A", rank);
    checkCuda(cudaMemcpy(deviceB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B", rank);

    if (!rank) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA tile: %d\n", ranks, omp_get_max_threads(), TILE);
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const dim3 block(TILE, TILE);
    const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE), static_cast<unsigned>((localRows + TILE - 1) / TILE));
    if (localRows) {
        matmulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N, localRows);
        checkCuda(cudaGetLastError(), "kernel launch", rank);
        checkCuda(cudaMemcpy(localC.data(), deviceC, localElements * sizeof(double), cudaMemcpyDeviceToHost), "copy C", rank);
    }
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double elapsed = 0.0;
    MPI_Reduce(&localSeconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displacements;
    std::vector<double> C;
    if (!rank) { counts.resize(ranks); displacements.resize(ranks); C.resize(N * N); }
    for (int r = 0; r < ranks; ++r) {
        const size_t startRow = (N * static_cast<size_t>(r)) / ranks;
        const size_t endRow = (N * static_cast<size_t>(r + 1)) / ranks;
        const int count = static_cast<int>((endRow - startRow) * N);
        if (!rank) { counts[r] = count; displacements[r] = static_cast<int>(startRow * N); }
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE, rank ? nullptr : C.data(),
                rank ? nullptr : counts.data(), rank ? nullptr : displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    checkCuda(cudaFree(deviceA), "cudaFree(A)", rank);
    checkCuda(cudaFree(deviceB), "cudaFree(B)", rank);
    checkCuda(cudaFree(deviceC), "cudaFree(C)", rank);
    int result = 0;
    if (!rank) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / elapsed / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::vector<double> A(N * N);
            initLocalMatrix(A, N, 0, N);
            std::printf("Validating result...\n");
            if (validateResult(A, B, C, N)) std::printf("Validation: PASSED\n");
            else { std::printf("Validation: FAILED\n"); result = 1; }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
