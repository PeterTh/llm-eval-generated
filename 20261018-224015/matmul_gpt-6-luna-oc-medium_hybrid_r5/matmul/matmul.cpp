#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <climits>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

__global__ void matmulKernel(const double* A, const double* B, double* C, size_t n, size_t rows) {
    __shared__ double tileA[16][16];
    __shared__ double tileB[16][16];
    const size_t j = blockIdx.x * 16 + threadIdx.x;
    const size_t i = blockIdx.y * 16 + threadIdx.y;
    double sum = 0.0;
    for (size_t base = 0; base < n; base += 16) {
        tileA[threadIdx.y][threadIdx.x] = i < rows && base + threadIdx.x < n ? A[i * n + base + threadIdx.x] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = base + threadIdx.y < n && j < n ? B[(base + threadIdx.y) * n + j] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (i < rows && j < n) C[i * n + j] = sum;
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t points[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) for (size_t pj = 0; pj < 5; ++pj) {
        const size_t i = points[pi] % N, j = points[pj] % N;
        double expected = 0.0;
        for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
        const double error = std::abs((C[i * N + j] - expected) / (expected + 1e-10));
        if (error > 1e-6) {
            printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, C[i * N + j], error);
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size N (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parseError = 1; }
    }
    if (N == 0 || N > static_cast<size_t>(INT_MAX) || N * N > static_cast<size_t>(INT_MAX)) parseError = 1;
    MPI_Allreduce(MPI_IN_PLACE, &parseError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (parseError) { MPI_Finalize(); return 1; }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(N * N), B(N * N), C;
    #pragma omp parallel for schedule(static)
    for (long long q = 0; q < static_cast<long long>(N * N); ++q) {
        const size_t i = static_cast<size_t>(q) / N, j = static_cast<size_t>(q) % N;
        A[static_cast<size_t>(q)] = getPseudoRndValue(N, i, j);
        B[static_cast<size_t>(q)] = getPseudoRndValue(N, i, j);
    }
    const size_t base = N / ranks, extra = N % ranks;
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < extra);
        counts[r] = static_cast<int>(rows * N);
        displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra)) * N);
    }
    const size_t localRows = base + (static_cast<size_t>(rank) < extra);
    std::vector<double> localA(localRows * N), localC(localRows * N);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 localA.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) { if (rank == 0) fprintf(stderr, "Each MPI rank requires an available CUDA device\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(rank % deviceCount));
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dA), std::max<size_t>(1, localRows * N) * sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dB), N * N * sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dC), std::max<size_t>(1, localRows * N) * sizeof(double)));
    if (localRows) cudaCheck(cudaMemcpy(dA, localA.data(), localRows * N * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (localRows) {
        dim3 block(16, 16), grid((N + 15) / 16, (localRows + 15) / 16);
        matmulKernel<<<grid, block>>>(dA, dB, dC, N, localRows);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(localC.data(), dC, localRows * N * sizeof(double), cudaMemcpyDeviceToHost));
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (rank == 0) C.resize(N * N);
    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        const double seconds = std::chrono::duration<double>(end - start).count();
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
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
