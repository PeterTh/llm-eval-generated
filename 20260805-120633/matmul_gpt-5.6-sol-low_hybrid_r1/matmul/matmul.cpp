#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

// A 32x32 tile minimizes global-memory traffic.  The 8-row thread block keeps
// occupancy high for double precision while each thread accumulates four rows.
constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     size_t rows, size_t N) {
    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE;
    double sums[TILE / BLOCK_ROWS] = {};

    for (size_t kBase = 0; kBase < N; kBase += TILE) {
#pragma unroll
        for (int r = 0; r < TILE; r += BLOCK_ROWS) {
            const size_t row = rowBase + ty + r;
            const size_t k = kBase + tx;
            As[ty + r][tx] = (row < rows && k < N) ? A[row * N + k] : 0.0;

            const size_t brow = kBase + ty + r;
            Bs[ty + r][tx] = (brow < N && col < N) ? B[brow * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double b = Bs[k][tx];
#pragma unroll
            for (int r = 0; r < TILE / BLOCK_ROWS; ++r)
                sums[r] = fma(As[ty + r * BLOCK_ROWS][k], b, sums[r]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < TILE / BLOCK_ROWS; ++r) {
        const size_t row = rowBase + ty + r * BLOCK_ROWS;
        if (row < rows && col < N) C[row * N + col] = sums[r];
    }
}

static bool validateResult(const std::vector<double>& C, size_t N) {
    int failed = 0;
#pragma omp parallel for collapse(2) reduction(| : failed)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = static_cast<size_t>(pi) % N;
            const size_t j = static_cast<size_t>(pj) % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
#pragma omp critical
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                failed = 1;
            }
        }
    }
    return failed == 0;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -n <num>     Matrix size N (default: 512)\n");
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
    bool validate = false, printResults = false;
    bool argsValid = true, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            argsValid = end && *end == '\0' && value > 0;
            N = static_cast<size_t>(value);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else argsValid = false;
    }
    if (help || !argsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argsValid ? 0 : 1;
    }

    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI_Gatherv counts\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", MPI_COMM_WORLD);

    const size_t firstRow = N * static_cast<size_t>(rank) / ranks;
    const size_t endRow = N * static_cast<size_t>(rank + 1) / ranks;
    const size_t localRows = endRow - firstRow;
    std::vector<double> A(localRows * N), B(N * N), localC(localRows * N);
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < localRows * N; ++p)
        A[p] = getPseudoRndValue(N, firstRow + p / N, p % N);
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < N * N; ++p)
        B[p] = getPseudoRndValue(N, p / N, p % N);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                    validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
        std::printf("Computing matrix multiplication...\n");
    }

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dA, std::max<size_t>(1, A.size()) * sizeof(double)), "cudaMalloc(A)", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dB, B.size() * sizeof(double)), "cudaMalloc(B)", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dC, std::max<size_t>(1, localC.size()) * sizeof(double)), "cudaMalloc(C)", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B", MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows) {
        const dim3 block(TILE, BLOCK_ROWS);
        const dim3 grid((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);
        matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, localRows, N);
    }
    cudaCheck(cudaGetLastError(), "matrixMultiplyKernel launch", MPI_COMM_WORLD);
    cudaCheck(cudaDeviceSynchronize(), "matrixMultiplyKernel", MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(localC.data(), dC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C", MPI_COMM_WORLD);

    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = N * static_cast<size_t>(r) / ranks;
        const size_t finish = N * static_cast<size_t>(r + 1) / ranks;
        counts[r] = static_cast<int>((finish - begin) * N);
        displacements[r] = static_cast<int>(begin * N);
    }
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);
    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        const double gflops = 2.0 * static_cast<double>(N) * N * N / maxElapsed / 1e9;
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", maxElapsed * 1e3, gflops);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(C, N);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return result;
}
