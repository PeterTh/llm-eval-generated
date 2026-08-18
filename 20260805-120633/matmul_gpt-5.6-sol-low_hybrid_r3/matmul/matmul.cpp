#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr int TILE = 16;

constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t error, const char *what, MPI_Comm comm) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, what,
                     cudaGetErrorString(error));
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

__global__ void matrixMultiplyKernel(const double *__restrict__ A,
                                     const double *__restrict__ B,
                                     double *__restrict__ C, size_t rows,
                                     size_t N) {
    __shared__ double as[TILE][TILE];
    __shared__ double bs[TILE][TILE];
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;

    for (size_t base = 0; base < N; base += TILE) {
        const size_t ak = base + threadIdx.x;
        const size_t bk = base + threadIdx.y;
        as[threadIdx.y][threadIdx.x] =
            (row < rows && ak < N) ? A[row * N + ak] : 0.0;
        bs[threadIdx.y][threadIdx.x] =
            (bk < N && col < N) ? B[bk * N + col] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum = fma(as[threadIdx.y][k], bs[k][threadIdx.x], sum);
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
}

static void initLocalA(std::vector<double> &A, size_t N, size_t firstRow,
                       size_t rows) {
#pragma omp parallel for schedule(static)
    for (long long li = 0; li < static_cast<long long>(rows); ++li) {
        const size_t i = firstRow + static_cast<size_t>(li);
        for (size_t j = 0; j < N; ++j)
            A[static_cast<size_t>(li) * N + j] = getPseudoRndValue(N, i, j);
    }
}

static void initB(std::vector<double> &B, size_t N) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i)
        for (size_t j = 0; j < N; ++j)
            B[static_cast<size_t>(i) * N + j] =
                getPseudoRndValue(N, static_cast<size_t>(i), j);
}

static bool validateResult(const std::vector<double> &C, size_t N) {
    int failed = 0;
#pragma omp parallel for collapse(2) reduction(| : failed)
    for (int ii = 0; ii < 5; ++ii) {
        for (int jj = 0; jj < 5; ++jj) {
            const size_t i = static_cast<size_t>(ii) % N;
            const size_t j = static_cast<size_t>(jj) % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) *
                            getPseudoRndValue(N, k, j);
            const double rel = std::abs((C[i * N + j] - expected) /
                                        (expected + 1e-10));
            if (rel > 1e-6) failed = 1;
        }
    }
    return failed == 0;
}

static void printUsage(const char *name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -n <num>     Matrix size N (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseResult = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char *end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end || value == 0 || value > std::numeric_limits<size_t>::max())
                parseResult = 1;
            else N = static_cast<size_t>(value);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) parseResult = 2;
        else parseResult = 1;
    }
    if (parseResult) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseResult == 2 ? 0 : 1;
    }

    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is too large.\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (devices == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice", MPI_COMM_WORLD);

    const size_t firstRow = (N * static_cast<size_t>(rank)) / ranks;
    const size_t endRow = (N * static_cast<size_t>(rank + 1)) / ranks;
    const size_t rows = endRow - firstRow;
    std::vector<double> A(rows * N), B(N * N), localC(rows * N), C;
    if (rank == 0) {
        C.resize(N * N);
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\nInitializing matrices...\n",
                    validate ? "enabled" : "disabled");
    }
    initLocalA(A, N, firstRow, rows);
    initB(B, N);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dA, std::max<size_t>(1, A.size()) * sizeof(double)), "cudaMalloc A", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dB, B.size() * sizeof(double)), "cudaMalloc B", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dC, std::max<size_t>(1, localC.size()) * sizeof(double)), "cudaMalloc C", MPI_COMM_WORLD);
    if (!A.empty()) cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B", MPI_COMM_WORLD);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows) {
        const dim3 block(TILE, TILE);
        const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                        static_cast<unsigned>((rows + TILE - 1) / TILE));
        matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, rows, N);
        cudaCheck(cudaGetLastError(), "kernel launch", MPI_COMM_WORLD);
        cudaCheck(cudaDeviceSynchronize(), "kernel execution", MPI_COMM_WORLD);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!localC.empty()) cudaCheck(cudaMemcpy(localC.data(), dC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C", MPI_COMM_WORLD);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = (N * static_cast<size_t>(r)) / ranks;
        const size_t finish = (N * static_cast<size_t>(r + 1)) / ranks;
        const size_t count = (finish - begin) * N;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) std::fprintf(stderr, "MPI gather count exceeds implementation limit.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        counts[r] = static_cast<int>(count);
        offsets[r] = static_cast<int>(begin * N);
    }
    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    MPI_Comm_free(&localComm);
    int result = 0;
    if (rank == 0) {
        const long long ms = static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", ms);
        const double gflops = elapsed > 0.0 ? 2.0 * static_cast<double>(N) * N * N / elapsed / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            result = validateResult(C, N) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
