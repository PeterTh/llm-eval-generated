#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr int TILE = 16;

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long q = 0; q < static_cast<long long>(N * N); ++q)
        mat[static_cast<size_t>(q)] = getPseudoRndValue(N, static_cast<size_t>(q) / N, static_cast<size_t>(q) % N);
}

__global__ void multiplyKernel(const double* A, const double* B, double* C,
                               int n, int firstRow, int rowCount) {
    __shared__ double aTile[TILE][TILE];
    __shared__ double bTile[TILE][TILE];
    const int localRow = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    const int globalRow = firstRow + localRow;
    double sum = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int ak = base + threadIdx.x;
        const int bk = base + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (localRow < rowCount && ak < n) ? A[globalRow * n + ak] : 0.0;
        bTile[threadIdx.y][threadIdx.x] = (bk < n && col < n) ? B[bk * n + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        __syncthreads();
    }
    if (localRow < rowCount && col < n) C[localRow * n + col] = sum;
}

bool cudaCheck(cudaError_t status, const char* what, int rank) {
    if (status == cudaSuccess) return true;
    if (rank == 0) std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
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
    std::printf("Usage: %s [options]\nOptions:\n", progName);
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = static_cast<size_t>(std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (N == 0 || N > static_cast<size_t>(INT_MAX) || N * N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is unsupported (must be positive and fit MPI counts).\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }
    std::vector<double> A(N * N), B(N * N), C(N * N);
    initMatrix(A, N); initMatrix(B, N);
    const int baseRows = static_cast<int>(N) / world, extra = static_cast<int>(N) % world;
    const int rows = baseRows + (rank < extra ? 1 : 0);
    const int firstRow = rank * baseRows + (rank < extra ? rank : extra);
    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) {
        const int rr = baseRows + (r < extra ? 1 : 0), offset = r * baseRows + (r < extra ? r : extra);
        counts[r] = rr * static_cast<int>(N); displacements[r] = offset * static_cast<int>(N);
    }
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    int localOk = deviceStatus == cudaSuccess && deviceCount > 0;
    if (localOk) localOk = cudaSetDevice(rank % deviceCount) == cudaSuccess;
    int allOk = 0; MPI_Allreduce(&localOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk) {
        if (rank == 0) std::fprintf(stderr, "Every MPI rank requires an available CUDA device.\n");
        MPI_Finalize(); return 1;
    }
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    bool ok = cudaCheck(cudaMalloc(&dA, N * N * sizeof(double)), "cudaMalloc(A)", rank) &&
              cudaCheck(cudaMalloc(&dB, N * N * sizeof(double)), "cudaMalloc(B)", rank) &&
              cudaCheck(cudaMalloc(&dC, static_cast<size_t>(rows) * N * sizeof(double)), "cudaMalloc(C)", rank);
    if (ok) ok = cudaCheck(cudaMemcpy(dA, A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy A", rank) &&
                 cudaCheck(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B", rank);
    int allocOk = ok, allAllocOk = 0;
    MPI_Allreduce(&allocOk, &allAllocOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allAllocOk) { cudaFree(dA); cudaFree(dB); cudaFree(dC); MPI_Finalize(); return 1; }
    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    if (rows > 0) {
        dim3 block(TILE, TILE), grid((static_cast<unsigned>(N) + TILE - 1) / TILE,
                                    (static_cast<unsigned>(rows) + TILE - 1) / TILE);
        multiplyKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N), firstRow, rows);
        ok = cudaCheck(cudaGetLastError(), "kernel launch", rank) && cudaCheck(cudaDeviceSynchronize(), "kernel", rank);
        if (ok) ok = cudaCheck(cudaMemcpy(C.data() + static_cast<size_t>(firstRow) * N, dC,
                         static_cast<size_t>(rows) * N * sizeof(double), cudaMemcpyDeviceToHost), "copy C", rank);
    }
    int computeOk = ok, allComputeOk = 0;
    MPI_Allreduce(&computeOk, &allComputeOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, C.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = allComputeOk && validateResult(A, B, C, N);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            MPI_Finalize(); return valid ? 0 : 1;
        }
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    MPI_Finalize(); return allComputeOk ? 0 : 1;
}
