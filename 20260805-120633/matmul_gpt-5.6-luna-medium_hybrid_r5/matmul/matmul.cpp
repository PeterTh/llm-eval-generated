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
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// This is deliberately a deterministic, integer-only generator: every MPI
// rank can reproduce the same inputs without communicating initialization data.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long index = 0; index < static_cast<long long>(N * N); ++index) {
        const size_t i = static_cast<size_t>(index) / N;
        const size_t j = static_cast<size_t>(index) % N;
        mat[static_cast<size_t>(index)] = getPseudoRndValue(N, i, j);
    }
}

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                      const double* __restrict__ B,
                                      double* __restrict__ C,
                                      const size_t N,
                                      const size_t rowOffset,
                                      const size_t rowCount) {
    // A 32x32 tile gives coalesced global accesses while keeping the inner
    // product in registers. The row offset lets each MPI rank own a slab.
    constexpr int TILE = 32;
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];

    const size_t localRow = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    const size_t globalRow = rowOffset + localRow;
    double sum = 0.0;

    for (size_t tile = 0; tile < N; tile += TILE) {
        const size_t aCol = tile + threadIdx.x;
        const size_t bRow = tile + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] =
            (localRow < rowCount && aCol < N) ? A[globalRow * N + aCol] : 0.0;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if (localRow < rowCount && col < N)
        C[localRow * N + col] = sum;
}

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t error,
                               const int rank) {
    if (rank == 0)
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t error, const char* operation, const int rank) {
    if (error != cudaSuccess)
        cudaFailure(operation, error, rank);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;

    #pragma omp parallel for collapse(2) reduction(&:valid)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[i * N + k] * B[k * N + j];
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6)
                valid = false;
        }
    }
    return valid;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            N = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
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
    // MPI_Gatherv/Bcast use int counts here; bound the dense matrix element
    // count before converting from size_t to preserve defined behavior.
    constexpr size_t maxNForMpiCounts = 46340;
    if (N == 0 || N > maxNForMpiCounts) {
        if (rank == 0) std::fprintf(stderr, "N must be in the range 1..%zu\n", maxNForMpiCounts);
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) cudaFailure("selecting a CUDA device", cudaErrorNoDevice, rank);
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t rowOffset = static_cast<size_t>(rank) * baseRows +
                             std::min(static_cast<size_t>(rank), extraRows);
    const size_t rowCount = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const int localElements = static_cast<int>(rowCount * N);

    std::vector<double> A(N * N), B(N * N), C(N * N);
    initMatrix(A, N);
    initMatrix(B, N);
    MPI_Bcast(A.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, N * N * sizeof(double)), "cudaMalloc(A)", rank);
    checkCuda(cudaMalloc(&deviceB, N * N * sizeof(double)), "cudaMalloc(B)", rank);
    checkCuda(cudaMalloc(&deviceC, std::max<size_t>(1, rowCount * N) * sizeof(double)), "cudaMalloc(C)", rank);
    checkCuda(cudaMemcpy(deviceA, A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy A", rank);
    checkCuda(cudaMemcpy(deviceB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B", rank);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                    N, N, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
                    worldSize, omp_get_max_threads(), deviceCount);
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    constexpr int TILE = 32;
    if (rowCount != 0) {
        dim3 block(TILE, TILE);
        dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                  static_cast<unsigned>((rowCount + TILE - 1) / TILE));
        matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N, rowOffset, rowCount);
        checkCuda(cudaGetLastError(), "matrixMultiplyKernel launch", rank);
        checkCuda(cudaDeviceSynchronize(), "matrixMultiplyKernel execution", rank);
        checkCuda(cudaMemcpy(C.data() + rowOffset * N, deviceC, rowCount * N * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy C", rank);
    }

    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        counts[r] = static_cast<int>(rows * N);
        displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseRows +
                                             std::min(static_cast<size_t>(r), extraRows)) * N);
    }
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : C.data() + rowOffset * N, rank == 0 ? 0 : localElements,
                MPI_DOUBLE, C.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    checkCuda(cudaFree(deviceA), "cudaFree(A)", rank);
    checkCuda(cudaFree(deviceB), "cudaFree(B)", rank);
    checkCuda(cudaFree(deviceC), "cudaFree(C)", rank);
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(A, B, C, N);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
