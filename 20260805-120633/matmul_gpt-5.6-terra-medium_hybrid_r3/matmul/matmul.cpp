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

#define CUDA_CHECK(call) do {                                                        \
    const cudaError_t error_ = (call);                                               \
    if (error_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA failure at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(error_));                                    \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error_));                         \
    }                                                                                \
} while (0)

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Each rank owns consecutive rows.  This preserves the original matrix values without
// moving A through the network, while B is replicated once for efficient GPU reuse.
void initRows(std::vector<double>& matrix, size_t N, size_t firstRow, size_t rowCount) {
    #pragma omp parallel for schedule(static)
    for (size_t localRow = 0; localRow < rowCount; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        for (size_t j = 0; j < N; ++j)
            matrix[localRow * N + j] = getPseudoRndValue(N, globalRow, j);
    }
}

void initMatrix(std::vector<double>& matrix, size_t N) {
    initRows(matrix, N, 0, N);
}

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
        aTile[threadIdx.y][threadIdx.x] = (row < localRows && aCol < N)
            ? A[row * N + aCol] : 0.0;
        bTile[threadIdx.y][threadIdx.x] = (bRow < N && col < N)
            ? B[bRow * N + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        __syncthreads();
    }
    if (row < localRows && col < N)
        C[row * N + col] = sum;
}

bool validateLocal(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, size_t N, size_t firstRow,
                   size_t localRows) {
    int failed = 0;
    #pragma omp parallel for schedule(static) reduction(|:failed)
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        // Preserve the original validation's first five columns, distributed by row.
        if (globalRow < 5) {
            for (size_t col = 0; col < 5 && col < N; ++col) {
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k)
                    expected += A[localRow * N + k] * B[k * N + col];
                const double actual = C[localRow * N + col];
                if (std::abs((actual - expected) / (expected + 1e-10)) > 1e-6)
                    failed = 1;
            }
        }
    }
    return failed == 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    constexpr size_t maxMpiElements = static_cast<size_t>(std::numeric_limits<int>::max());
    if (N == 0 || N > maxMpiElements / N) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is outside the supported MPI count range\n");
        MPI_Finalize();
        return 1;
    }

    const size_t firstRow = N * static_cast<size_t>(rank) / static_cast<size_t>(ranks);
    const size_t endRow = N * static_cast<size_t>(rank + 1) / static_cast<size_t>(ranks);
    const size_t localRows = endRow - firstRow;
    const size_t localElements = localRows * N;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks, omp_get_max_threads());
        std::printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(localElements), B(N * N), C(localElements);
    initRows(A, N, firstRow, localRows);
    if (rank == 0) initMatrix(B, N);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, localElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, localElements * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), localElements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    const auto start = std::chrono::steady_clock::now();
    if (localRows != 0) {
        const dim3 block(TILE, TILE);
        const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                        static_cast<unsigned>((localRows + TILE - 1) / TILE));
        matmulKernel<<<grid, block>>>(dA, dB, dC, localRows, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(C.data(), dC, localElements * sizeof(double), cudaMemcpyDeviceToHost));

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }

    if (printResults) {
        std::vector<int> counts(ranks), offsets(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t rows = N * static_cast<size_t>(r + 1) / ranks - N * static_cast<size_t>(r) / ranks;
            counts[r] = static_cast<int>(rows * N);
            offsets[r] = static_cast<int>((N * static_cast<size_t>(r) / ranks) * N);
        }
        std::vector<double> globalC(rank == 0 ? N * N : 0);
        MPI_Gatherv(C.data(), static_cast<int>(localElements), MPI_DOUBLE, globalC.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(globalC, "MatrixC");
    }

    int localValid = !validate || validateLocal(A, B, C, N, firstRow, localRows);
    int valid = 0;
    MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (validate && rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    MPI_Finalize();
    return valid ? 0 : 1;
}
