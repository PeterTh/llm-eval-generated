#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, i, j);
}

// Eight rows per block keeps the block within the 1024-thread limit while
// allowing every thread to produce one result and reusing a 32-column tile.
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const size_t N, const size_t rowOffset,
                             const size_t rows) {
    constexpr int TILE = 32;
    __shared__ double tileA[8][TILE];
    __shared__ double tileB[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t row = rowOffset + blockIdx.y * 8 + ty;
    const size_t col = blockIdx.x * TILE + tx;
    double sum = 0.0;

    for (size_t k0 = 0; k0 < N; k0 += TILE) {
        const size_t ak = k0 + tx;
        tileA[ty][tx] = (row < rowOffset + rows && ak < N) ? A[row * N + ak] : 0.0;

        // The 32x32 B tile is loaded by the 32x8 block (four rows/thread).
        for (int br = ty; br < TILE; br += 8) {
            const size_t bk = k0 + static_cast<size_t>(br);
            tileB[br][tx] = (bk < N && col < N) ? B[bk * N + col] : 0.0;
        }
        __syncthreads();

        if (row < rowOffset + rows && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE; ++k)
                sum += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }
    if (row < rowOffset + rows && col < N)
        C[(row - rowOffset) * N + col] = sum;
}

void cudaCheck(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void matrixMultiplyDistributed(const std::vector<double>& A, const std::vector<double>& B,
                               std::vector<double>& localC, const size_t N,
                               const size_t rowOffset, const size_t rows, const int rank) {
    if (rows == 0) return;
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0)
        throw std::runtime_error("no CUDA device is available");
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t bytesA = N * N * sizeof(double);
    const size_t bytesC = rows * N * sizeof(double);
    cudaCheck(cudaMalloc(&dA, bytesA), "cudaMalloc(A)");
    cudaCheck(cudaMalloc(&dB, bytesA), "cudaMalloc(B)");
    cudaCheck(cudaMalloc(&dC, bytesC), "cudaMalloc(C)");

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");
    cudaCheck(cudaMemcpyAsync(dA, A.data(), bytesA, cudaMemcpyHostToDevice, stream), "copy A");
    cudaCheck(cudaMemcpyAsync(dB, B.data(), bytesA, cudaMemcpyHostToDevice, stream), "copy B");

    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((N + 31) / 32),
                    static_cast<unsigned>((rows + 7) / 8));
    matmulKernel<<<grid, block, 0, stream>>>(dA, dB, dC, N, rowOffset, rows);
    cudaCheck(cudaGetLastError(), "matmulKernel launch");
    cudaCheck(cudaMemcpyAsync(localC.data(), dC, bytesC, cudaMemcpyDeviceToHost, stream), "copy C");
    cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy");
    cudaCheck(cudaFree(dC), "cudaFree(C)");
    cudaCheck(cudaFree(dB), "cudaFree(B)");
    cudaCheck(cudaFree(dA), "cudaFree(A)");
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;
    #pragma omp parallel for collapse(2) reduction(&:valid)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N, j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                #pragma omp critical
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                valid = false;
            }
        }
    }
    return valid;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num> Matrix size N (default: 512)\n"
                "  -v       Enable validation\n  -r       Print results for external validation\n"
                "  -h       Show this help message\n", progName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { parseError = 1; }
    }
    if (N == 0 || N * N > static_cast<size_t>(INT_MAX)) parseError = 1;
    if (parseError) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }

    const size_t base = N / static_cast<size_t>(world), extra = N % static_cast<size_t>(world);
    const size_t rows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t rowOffset = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    std::vector<double> A(N * N), B(N * N), localC(rows * N);
    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                    N, N, validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
        initMatrix(A, N); initMatrix(B, N);
    }
    MPI_Bcast(A.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    try { matrixMultiplyDistributed(A, B, localC, N, rowOffset, rows, rank); }
    catch (const std::exception& e) { std::fprintf(stderr, "Rank %d: %s\n", rank, e.what()); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const long localMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxMs = 0; MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) {
        const size_t rRows = base + (static_cast<size_t>(r) < extra ? 1 : 0);
        const size_t rOffset = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra);
        counts[r] = static_cast<int>(rRows * N); displacements[r] = static_cast<int>(rOffset * N);
    }
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);
    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computing matrix multiplication...\nComputation time: %ld ms\nPerformance: %.3f GFLOPS\n",
                    maxMs, (maxMs > 0) ? (2.0 * N * N * N) / (maxMs / 1000.0) / 1e9 : 0.0);
        if (printResults) print_results(C, "MatrixC");
        bool valid = true;
        if (validate) {
            std::printf("Validating result...\n");
            valid = validateResult(A, B, C, N);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        const int result = (validate && !valid) ? 1 : 0;
        MPI_Finalize(); return result;
    }
    MPI_Finalize(); return 0;
}
