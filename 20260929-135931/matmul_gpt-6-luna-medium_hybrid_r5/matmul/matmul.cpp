#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error on rank %d: %s\n", rank, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N, size_t rowOffset = 0) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(mat.size() / N); ++i)
        for (size_t j = 0; j < N; ++j)
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, rowOffset + static_cast<size_t>(i), j);
}

// A/B tiles are shared across output elements. Each output accumulates k in the
// same order as the reference implementation, avoiding reduction-order changes.
__global__ void multiplyRows(const double* A, const double* B, double* C,
                             size_t N, size_t rows) {
    constexpr int TILE = 16;
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    const size_t localRow = blockIdx.y * TILE + threadIdx.y;
    const size_t col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t kt = 0; kt < N; kt += TILE) {
        const size_t ak = kt + threadIdx.x;
        const size_t bk = kt + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (localRow < rows && ak < N) ? A[localRow * N + ak] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = (bk < N && col < N) ? B[bk * N + col] : 0.0;
        __syncthreads();
        #pragma unroll 1
        for (int k = 0; k < TILE && kt + k < N; ++k)
            if (localRow < rows && col < N)
                sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (localRow < rows && col < N) C[localRow * N + col] = sum;
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
            printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size N (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = static_cast<size_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (N == 0 || N > static_cast<size_t>(INT_MAX) || N > static_cast<size_t>(std::sqrt(INT_MAX))) {
        if (rank == 0) fprintf(stderr, "Matrix size must be positive and fit MPI's count limits\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign ranks to GPUs locally, so multi-node runs start device numbering at 0 on each node.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d cannot access a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t baseRows = N / worldSize, extraRows = N % worldSize;
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t rowStart = static_cast<size_t>(rank) * baseRows + std::min(static_cast<size_t>(rank), extraRows);
    std::vector<int> counts(worldSize), displs(worldSize);
    for (int p = 0; p < worldSize; ++p) {
        const size_t rows = baseRows + (static_cast<size_t>(p) < extraRows ? 1 : 0);
        counts[p] = static_cast<int>(rows * N);
        displs[p] = static_cast<int>((static_cast<size_t>(p) * baseRows + std::min(static_cast<size_t>(p), extraRows)) * N);
    }
    std::vector<double> A(localRows * N), B(N * N), C(localRows * N);
    std::vector<double> fullA, fullC;
    if (rank == 0) { fullA.resize(N * N); fullC.resize(N * N); }
    if (rank == 0) { initMatrix(fullA, N); initMatrix(B, N); }
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 A.data(), static_cast<int>(localRows * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\nComputing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, std::max<size_t>(1, localRows * N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, std::max<size_t>(1, localRows * N) * sizeof(double)));
    if (localRows) CUDA_CHECK(cudaMemcpy(dA, A.data(), localRows * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (localRows) {
        dim3 block(16, 16), grid((N + 15) / 16, (localRows + 15) / 16);
        multiplyRows<<<grid, block>>>(dA, dB, dC, N, localRows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C.data(), dC, localRows * N * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? fullC.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Computation time: %ld ms\n", static_cast<long>(ms));
        const double seconds = std::chrono::duration<double>(end - start).count();
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? (2.0 * N * N * N) / seconds / 1e9 : 0.0);
        if (printResults) print_results(fullC, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullA, B, fullC, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
