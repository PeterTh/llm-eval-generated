#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

// Keep initialization bit-for-bit compatible with the original benchmark.
__host__ __device__ constexpr double getPseudoRndValue(const size_t N,
                                                        const size_t i,
                                                        const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t rows, const size_t N,
                const size_t rowOffset = 0) {
#pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(rows); ++ii) {
        const size_t i = rowOffset + static_cast<size_t>(ii);
        for (size_t j = 0; j < N; ++j)
            mat[static_cast<size_t>(ii) * N + j] = getPseudoRndValue(N, i, j);
    }
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const int N, const int rows) {
    constexpr int TILE = 16;
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;

    for (int base = 0; base < N; base += TILE) {
        const int aCol = base + threadIdx.x;
        const int bRow = base + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] =
            (row < rows && aCol < N) ? A[row * N + aCol] : 0.0;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < N)
        C[row * N + col] = sum;
}

[[noreturn]] void mpiCudaFailure(const char* what, cudaError_t error) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void cudaCheck(cudaError_t error, const char* what) {
    if (error != cudaSuccess) mpiCudaFailure(what, error);
}

void deviceMultiply(const std::vector<double>& localA, const std::vector<double>& B,
                    std::vector<double>& localC, const size_t rows, const size_t N,
                    const int device) {
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t aBytes = rows * N * sizeof(double);
    const size_t bBytes = N * N * sizeof(double);
    const size_t cBytes = aBytes;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dA), std::max(aBytes, sizeof(double))), "cudaMalloc(A)");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dB), bBytes), "cudaMalloc(B)");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dC), std::max(cBytes, sizeof(double))), "cudaMalloc(C)");
    if (aBytes != 0)
        cudaCheck(cudaMemcpy(dA, localA.data(), aBytes, cudaMemcpyHostToDevice), "copy A");
    cudaCheck(cudaMemcpy(dB, B.data(), bBytes, cudaMemcpyHostToDevice), "copy B");

    constexpr int TILE = 16;
    const dim3 block(TILE, TILE);
    if (rows != 0) {
        const dim3 grid((static_cast<unsigned>(N) + TILE - 1) / TILE,
                        (static_cast<unsigned>(rows) + TILE - 1) / TILE);
        matmulKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N), static_cast<int>(rows));
        cudaCheck(cudaGetLastError(), "matmulKernel launch");
        cudaCheck(cudaDeviceSynchronize(), "matmulKernel execution");
        cudaCheck(cudaMemcpy(localC.data(), dC, cBytes, cudaMemcpyDeviceToHost), "copy C");
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    bool valid = true;
#pragma omp parallel for collapse(2) reduction(&:valid)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = static_cast<size_t>(pi) % N;
            const size_t j = static_cast<size_t>(pj) % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
            const double error = std::abs((C[i * N + j] - expected) / (expected + 1e-10));
            if (error > 1e-6) valid = false;
        }
    }
    return valid;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Matrix size (default: 512)\n"
                "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Matrix size must be in [1, %d].\n", std::numeric_limits<int>::max());
        MPI_Finalize(); return 1;
    }
    if (N > std::numeric_limits<size_t>::max() / N || N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI counts.\n");
        MPI_Finalize(); return 1;
    }

    const size_t base = N / static_cast<size_t>(world), extra = N % static_cast<size_t>(world);
    const size_t localRows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < extra ? 1 : 0);
        counts[r] = static_cast<int>(rows * N);
        displacements[r] = static_cast<int>((base * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), extra)) * N);
    }
    std::vector<double> A(rank == 0 ? N * N : 0), B(N * N);
    std::vector<double> localA(localRows * N), localC(localRows * N);
    std::vector<double> C(rank == 0 ? N * N : 0);
    if (rank == 0) initMatrix(A, N, N);
    initMatrix(B, N, N);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 localA.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) mpiCudaFailure("cudaGetDeviceCount", cudaErrorNoDevice);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                    N, N, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n", world, omp_get_max_threads(), deviceCount);
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    deviceMultiply(localA, B, localC, localRows, N, localRank % deviceCount);
    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double localTime = MPI_Wtime() - start, duration = [&] { double t = 0; MPI_Reduce(&localTime, &t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); return t; }();
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", duration * 1e3,
                    (2.0 * N * N * N) / duration / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) std::printf("Validation: %s\n", validateResult(A, B, C, N) ? "PASSED" : "FAILED");
    }
    const int result = (rank == 0 && validate && !validateResult(A, B, C, N)) ? 1 : 0;
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return result;
}
