#include <mpi.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

constexpr int TILE = 32;

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, i, j);
}

void transpose(const std::vector<double>& B, std::vector<double>& BT, size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            BT[j * N + i] = B[i * N + j];
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ BT,
                             double* __restrict__ C, size_t rows, size_t N) {
    __shared__ double aTile[TILE][TILE];
    __shared__ double bTile[TILE][TILE];
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t base = 0; base < N; base += TILE) {
        const size_t ak = base + threadIdx.x;
        const size_t bk = base + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (row < rows && ak < N) ? A[row * N + ak] : 0.0;
        bTile[threadIdx.y][threadIdx.x] = (col < N && bk < N) ? BT[col * N + bk] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
}

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

void matrixMultiplyCuda(const std::vector<double>& A, const std::vector<double>& BT,
                        std::vector<double>& C, size_t rows, size_t N) {
    double *dA = nullptr, *dBT = nullptr, *dC = nullptr;
    const size_t aBytes = rows * N * sizeof(double), bBytes = N * N * sizeof(double);
    const size_t cBytes = rows * N * sizeof(double);
    try {
        cudaCheck(cudaMalloc(&dA, aBytes), "cudaMalloc(A)");
        cudaCheck(cudaMalloc(&dBT, bBytes), "cudaMalloc(BT)");
        cudaCheck(cudaMalloc(&dC, cBytes), "cudaMalloc(C)");
        cudaCheck(cudaMemcpy(dA, A.data(), aBytes, cudaMemcpyHostToDevice), "copy A");
        cudaCheck(cudaMemcpy(dBT, BT.data(), bBytes, cudaMemcpyHostToDevice), "copy BT");
        const dim3 block(TILE, TILE);
        const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                        static_cast<unsigned>((rows + TILE - 1) / TILE));
        matmulKernel<<<grid, block>>>(dA, dBT, dC, rows, N);
        cudaCheck(cudaGetLastError(), "matmul kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "matmul kernel");
        cudaCheck(cudaMemcpy(C.data(), dC, cBytes, cudaMemcpyDeviceToHost), "copy C");
        cudaFree(dC); cudaFree(dBT); cudaFree(dA);
    } catch (...) {
        cudaFree(dC); cudaFree(dBT); cudaFree(dA);
        throw;
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
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size N (default: 512)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            N = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0) printf("Matrix size is too large or invalid\n");
        MPI_Finalize(); return 1;
    }
    if (static_cast<size_t>(world) > N) {
        if (rank == 0) printf("MPI process count cannot exceed matrix size\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
               N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    const size_t begin = (N * static_cast<size_t>(rank)) / static_cast<size_t>(world);
    const size_t end = (N * static_cast<size_t>(rank + 1)) / static_cast<size_t>(world);
    const size_t rows = end - begin;
    std::vector<double> A(rank == 0 ? N * N : rows * N), B(N * N), BT(N * N), C(rows * N), gathered;
    if (rank == 0) { initMatrix(A, N); initMatrix(B, N); }
    else if (rows != 0) initMatrix(A, rows);

    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    transpose(B, BT, N);
    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) {
        const size_t rb = (N * static_cast<size_t>(r)) / static_cast<size_t>(world);
        const size_t re = (N * static_cast<size_t>(r + 1)) / static_cast<size_t>(world);
        counts[r] = static_cast<int>((re - rb) * N);
        displacements[r] = static_cast<int>(rb * N);
    }
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 A.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) throw std::runtime_error("no CUDA device available");
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiplyCuda(A, BT, C, rows, N);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) gathered.resize(N * N);
    MPI_Gatherv(C.data(), counts[rank], MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computing matrix multiplication...\n");
        printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", milliseconds,
               maxElapsed > 0.0 ? (2.0 * N * N * N) / maxElapsed / 1e9 : 0.0);
        if (printResults) print_results(gathered, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(A, B, gathered, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize(); return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
