#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i)
        for (size_t j = 0; j < N; ++j)
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, static_cast<size_t>(i), j);
}

__global__ void multiplyRows(const double* A, const double* B, double* C,
                             size_t N, size_t rows, size_t rowOffset) {
    constexpr int TILE = 32;
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    const size_t row = blockIdx.y * TILE + threadIdx.y;
    const size_t col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t t = 0; t < (N + TILE - 1) / TILE; ++t) {
        const size_t aCol = t * TILE + threadIdx.x;
        const size_t bRow = t * TILE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = row < rows && aCol < N
            ? A[(rowOffset + row) * N + aCol] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = bRow < N && col < N
            ? B[bRow * N + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
}

static void cudaCheck(cudaError_t err, const char* what, int rank) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
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
            printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                   i, j, expected, actual, relError);
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

    unsigned long long nArg = 512;
    bool validate = false, printResults = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            nArg = strtoull(argv[++i], &end, 10);
            if (*end != '\0' || nArg == 0) badArgs = true;
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else badArgs = true;
    }
    unsigned long long anyBad = badArgs;
    MPI_Allreduce(MPI_IN_PLACE, &anyBad, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (anyBad || nArg > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
        nArg > std::numeric_limits<size_t>::max() / nArg) {
        if (rank == 0) fprintf(stderr, "Invalid arguments or matrix size\n");
        MPI_Finalize();
        return 1;
    }
    const size_t N = static_cast<size_t>(nArg);
    const size_t baseRows = N / worldSize, extra = N % worldSize;
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extra);
    const size_t rowOffset = static_cast<size_t>(rank) * baseRows +
                             (static_cast<size_t>(rank) < extra ? static_cast<size_t>(rank) : extra);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "CUDA device unavailable\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "select device", rank);
    MPI_Comm_free(&localComm);

    std::vector<double> A(N * N), B(N * N), C(N * N);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\nInitializing matrices...\n", worldSize, omp_get_max_threads());
    }
    initMatrix(A, N);
    initMatrix(B, N);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dA, N * N * sizeof(double)), "allocate A", rank);
    cudaCheck(cudaMalloc(&dB, N * N * sizeof(double)), "allocate B", rank);
    if (localRows != 0) cudaCheck(cudaMalloc(&dC, localRows * N * sizeof(double)), "allocate C", rank);
    cudaCheck(cudaMemcpy(dA, A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy A", rank);
    cudaCheck(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B", rank);
    if (localRows != 0) {
        dim3 block(32, 32), grid((N + 31) / 32, (localRows + 31) / 32);
        multiplyRows<<<grid, block>>>(dA, dB, dC, N, localRows, rowOffset);
        cudaCheck(cudaGetLastError(), "launch multiplication", rank);
        cudaCheck(cudaMemcpy(C.data() + rowOffset * N, dC, localRows * N * sizeof(double), cudaMemcpyDeviceToHost), "copy C", rank);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extra);
        const size_t offset = static_cast<size_t>(r) * baseRows + (static_cast<size_t>(r) < extra ? static_cast<size_t>(r) : extra);
        if (rows * N > static_cast<size_t>(std::numeric_limits<int>::max()) || offset * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) fprintf(stderr, "Matrix exceeds MPI collective count limits\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[r] = static_cast<int>(rows * N);
        displacements[r] = static_cast<int>(offset * N);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, C.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();
    if (rank == 0) {
        const double seconds = std::chrono::duration<double>(end - start).count();
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(A, B, C, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
