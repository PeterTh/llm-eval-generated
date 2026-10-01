#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t rows, const size_t N,
                const size_t firstRow = 0) {
    #pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(rows); ++row) {
        const size_t i = firstRow + static_cast<size_t>(row);
        for (size_t j = 0; j < N; ++j)
            mat[static_cast<size_t>(row) * N + j] = getPseudoRndValue(N, i, j);
    }
}

constexpr int TILE = 32;

__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C,
                                    int rows, int N) {
    __shared__ double aTile[TILE][TILE];
    __shared__ double bTile[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * TILE + tx;
    const int row = blockIdx.y * TILE + ty;
    double sum = 0.0;
    for (int base = 0; base < N; base += TILE) {
        const int ak = base + tx, bk = base + ty;
        aTile[ty][tx] = (row < rows && ak < N) ? A[static_cast<size_t>(row) * N + ak] : 0.0;
        bTile[ty][tx] = (bk < N && col < N) ? B[static_cast<size_t>(bk) * N + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += aTile[ty][k] * bTile[k][tx];
        __syncthreads();
    }
    if (row < rows && col < N) C[static_cast<size_t>(row) * N + col] = sum;
}

inline void cudaCheck(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t rows, size_t N) {
    if (rows == 0) return;
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t aBytes = rows * N * sizeof(double), matrixBytes = N * N * sizeof(double);
    cudaCheck(cudaMalloc(&dA, aBytes), "allocate A");
    cudaCheck(cudaMalloc(&dB, matrixBytes), "allocate B");
    cudaCheck(cudaMalloc(&dC, aBytes), "allocate C");
    cudaCheck(cudaMemcpy(dA, A.data(), aBytes, cudaMemcpyHostToDevice), "copy A");
    cudaCheck(cudaMemcpy(dB, B.data(), matrixBytes, cudaMemcpyHostToDevice), "copy B");
    dim3 block(TILE, TILE);
    dim3 grid((static_cast<unsigned>(N) + TILE - 1) / TILE,
              (static_cast<unsigned>(rows) + TILE - 1) / TILE);
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(rows), static_cast<int>(N));
    cudaCheck(cudaGetLastError(), "launch matrix multiply");
    cudaCheck(cudaMemcpy(C.data(), dC, aBytes, cudaMemcpyDeviceToHost), "copy C");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N) {
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

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = strtoull(argv[++i], &end, 10);
            if (!end || *end || parsed == 0 || parsed > static_cast<unsigned long long>(INT_MAX)) parseError = 1;
            else N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parseError = 1; }
    }
    int anyError = 0;
    MPI_Allreduce(&parseError, &anyError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyError) { MPI_Finalize(); return 1; }
    if (N > static_cast<size_t>(INT_MAX) || N * N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Matrix size exceeds supported MPI/CUDA indexing limits\n");
        MPI_Finalize(); return 1;
    }

    const size_t rows = N / worldSize + (static_cast<size_t>(rank) < N % worldSize ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * (N / worldSize) + std::min(static_cast<size_t>(rank), N % worldSize);
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "Every MPI rank requires an available CUDA device\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "select device");

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(N * N), localC(rows * N);
    initMatrix(A, rows, N, firstRow);
    initMatrix(B, N, N);
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, localC, rows, N);
    const int localCount = static_cast<int>(rows * N);
    std::vector<int> counts, displacements;
    if (rank == 0) {
        counts.resize(worldSize); displacements.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const size_t rRows = N / worldSize + (static_cast<size_t>(r) < N % worldSize ? 1 : 0);
            const size_t rFirst = static_cast<size_t>(r) * (N / worldSize) + std::min(static_cast<size_t>(r), N % worldSize);
            counts[r] = static_cast<int>(rRows * N);
            displacements[r] = static_cast<int>(rFirst * N);
        }
    }
    MPI_Gatherv(localC.data(), localCount, MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double elapsedLocal = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&elapsedLocal, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / elapsed / 1e9);
        if (printResults) print_results(C, "MatrixC");
    }
    int validationFailed = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        std::vector<double> fullA(N * N);
        initMatrix(fullA, N, N);
        validationFailed = validateResult(fullA, B, C, N) ? 0 : 1;
        printf("Validation: %s\n", validationFailed ? "FAILED" : "PASSED");
    }
    MPI_Bcast(&validationFailed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validationFailed;
}
