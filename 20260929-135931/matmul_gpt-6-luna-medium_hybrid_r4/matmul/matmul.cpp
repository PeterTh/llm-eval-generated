#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {
constexpr int TILE = 32;

__global__ void matmulKernel(const double* A, const double* B, double* C,
                             size_t n, size_t firstRow, size_t localRows) {
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    const size_t row = firstRow + static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t base = 0; base < n; base += TILE) {
        const size_t ak = base + threadIdx.x;
        const size_t bk = base + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < firstRow + localRows && ak < n) ? A[row * n + ak] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = (bk < n && col < n) ? B[bk * n + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < firstRow + localRows && col < n) C[(row - firstRow) * n + col] = sum;
}

void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < static_cast<long long>(mat.size()); ++idx) {
        const size_t i = static_cast<size_t>(idx) / N;
        const size_t j = static_cast<size_t>(idx) % N;
        mat[static_cast<size_t>(idx)] = getPseudoRndValue(N, i, j);
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
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

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size N (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t N = 512;
    bool validate = false, printResults = false;
    int badArgs = 0, showHelp = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (!end || *end || parsed == 0) badArgs = 1; else N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) showHelp = 1;
        else badArgs = 1;
    }
    int anyBad = 0;
    MPI_Allreduce(&badArgs, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (showHelp || anyBad) {
        if (rank == 0) { if (anyBad) std::printf("Invalid arguments.\n"); printUsage(argv[0]); }
        MPI_Finalize(); return anyBad ? 1 : 0;
    }
    unsigned long long nArg = static_cast<unsigned long long>(N);
    MPI_Bcast(&nArg, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(nArg);
    if (N > static_cast<size_t>(std::numeric_limits<int>::max()) || N > std::numeric_limits<size_t>::max() / N || N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Matrix size exceeds supported MPI count limits.\n");
        MPI_Finalize(); return 1;
    }

    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "A CUDA device is required on every MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads per rank: %d, CUDA devices per node: %d\n", world, omp_get_max_threads(), deviceCount);
        std::printf("Initializing matrices...\n");
    }
    const size_t total = N * N;
    std::vector<double> A(total), B(total);
    if (rank == 0) { initMatrix(A, N); initMatrix(B, N); }
    MPI_Bcast(A.data(), static_cast<int>(total), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), static_cast<int>(total), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const size_t firstRow = N * static_cast<size_t>(rank) / static_cast<size_t>(world);
    const size_t endRow = N * static_cast<size_t>(rank + 1) / static_cast<size_t>(world);
    const size_t localRows = endRow - firstRow;
    std::vector<double> localC(localRows * N);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dA, total * sizeof(double)), "cudaMalloc A");
    cudaCheck(cudaMalloc(&dB, total * sizeof(double)), "cudaMalloc B");
    cudaCheck(cudaMalloc(&dC, std::max<size_t>(1, localRows * N) * sizeof(double)), "cudaMalloc C");
    cudaCheck(cudaMemcpy(dA, A.data(), total * sizeof(double), cudaMemcpyHostToDevice), "copy A");
    cudaCheck(cudaMemcpy(dB, B.data(), total * sizeof(double), cudaMemcpyHostToDevice), "copy B");

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    if (localRows) {
        dim3 block(TILE, TILE);
        dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE), static_cast<unsigned>((localRows + TILE - 1) / TILE));
        matmulKernel<<<grid, block>>>(dA, dB, dC, N, firstRow, localRows);
        cudaCheck(cudaGetLastError(), "matmulKernel launch");
        cudaCheck(cudaMemcpy(localC.data(), dC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C");
    }
    std::vector<double> C;
    if (rank == 0) C.resize(total);
    std::vector<int> counts(static_cast<size_t>(world)), displs(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r) {
        const size_t rb = N * static_cast<size_t>(r) / static_cast<size_t>(world);
        const size_t re = N * static_cast<size_t>(r + 1) / static_cast<size_t>(world);
        counts[static_cast<size_t>(r)] = static_cast<int>((re - rb) * N);
        displs[static_cast<size_t>(r)] = static_cast<int>(rb * N);
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    int resultCode = 0;
    if (rank == 0) {
        const long durationMs = static_cast<long>(seconds * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMs);
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * static_cast<double>(N) * N * N) / seconds / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            resultCode = validateResult(A, B, C, N) ? 0 : 1;
            std::printf("Validation: %s\n", resultCode == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&resultCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return resultCode;
}
