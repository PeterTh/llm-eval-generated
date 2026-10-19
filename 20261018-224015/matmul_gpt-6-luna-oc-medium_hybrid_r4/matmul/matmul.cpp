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

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long ij = 0; ij < static_cast<long long>(N * N); ++ij) {
        const size_t i = static_cast<size_t>(ij) / N;
        const size_t j = static_cast<size_t>(ij) % N;
        mat[static_cast<size_t>(ij)] = getPseudoRndValue(N, i, j);
    }
}

__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C,
                                     size_t N, size_t rowStart, size_t rows) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t localRow = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (col >= N || localRow >= rows) return;
    const size_t row = rowStart + localRow;
    double sum = 0.0;
    for (size_t k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
    C[localRow * N + col] = sum;
}

static void cudaCheck(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(err));
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
        const double error = std::abs((C[i * N + j] - expected) / (expected + 1e-10));
        if (error > 1e-6) {
            std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                        i, j, expected, C[i * N + j], error);
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n  -n <num> Matrix size N (default: 512)\n  -v       Enable validation\n  -r       Print results for external validation\n  -h       Show this help message\n", progName);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t N = 512;
    bool validate = false, printResults = false;
    bool badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end || value == 0) badArgs = true;
            else N = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else badArgs = true;
    }
    if (N > static_cast<size_t>(INT_MAX) || N > SIZE_MAX / N || N * N > static_cast<size_t>(INT_MAX)) badArgs = true;
    int localBad = badArgs ? 1 : 0, anyBad = 0;
    MPI_Allreduce(&localBad, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyBad) {
        if (rank == 0) { std::fprintf(stderr, "Invalid arguments or matrix too large for MPI counts.\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    const size_t rowStart = N * static_cast<size_t>(rank) / static_cast<size_t>(world);
    const size_t rowEnd = N * static_cast<size_t>(rank + 1) / static_cast<size_t>(world);
    const size_t rows = rowEnd - rowStart;
    std::vector<double> A(N * N), B(N * N), localC(rows * N);
    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
    }
    initMatrix(A, N);
    initMatrix(B, N);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dA, A.size() * sizeof(double)), "cudaMalloc A");
    cudaCheck(cudaMalloc(&dB, B.size() * sizeof(double)), "cudaMalloc B");
    if (!localC.empty()) cudaCheck(cudaMalloc(&dC, localC.size() * sizeof(double)), "cudaMalloc C");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A");
    cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B");
    if (rows != 0) {
        dim3 block(32, 8);
        dim3 grid(static_cast<unsigned>((N + block.x - 1) / block.x),
                  static_cast<unsigned>((rows + block.y - 1) / block.y));
        matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, N, rowStart, rows);
        cudaCheck(cudaGetLastError(), "matmul kernel launch");
        cudaCheck(cudaMemcpy(localC.data(), dC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C");
    }
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);
    std::vector<int> counts(static_cast<size_t>(world)), displacements(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r) {
        const size_t begin = N * static_cast<size_t>(r) / static_cast<size_t>(world);
        const size_t end = N * static_cast<size_t>(r + 1) / static_cast<size_t>(world);
        counts[r] = static_cast<int>((end - begin) * N);
        displacements[r] = static_cast<int>(begin * N);
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaFree(dA); cudaFree(dB); if (dC) cudaFree(dC);

    int result = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            result = validateResult(A, B, C, N) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
