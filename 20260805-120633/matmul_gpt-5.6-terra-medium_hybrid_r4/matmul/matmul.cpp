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

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double aTile[TILE][TILE];
    __shared__ double bTile[TILE][TILE];

    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;

    for (size_t tile = 0; tile < N; tile += TILE) {
        const size_t aCol = tile + threadIdx.x;
        const size_t bRow = tile + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (row < rows && aCol < N) ? A[row * N + aCol] : 0.0;
        bTile[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
}

void initLocalMatrix(std::vector<double>& matrix, const size_t N, const size_t firstRow) {
    const size_t rows = matrix.size() / N;
    #pragma omp parallel for schedule(static)
    for (size_t localRow = 0; localRow < rows; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        for (size_t j = 0; j < N; ++j) {
            matrix[localRow * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void initMatrix(std::vector<double>& matrix, const size_t N) {
    initLocalMatrix(matrix, N, 0);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Matrix size N (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > std::numeric_limits<size_t>::max() / N || N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is unsupported\n");
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            (static_cast<size_t>(rank) < extraRows ? rank : extraRows);
    const size_t localElements = localRows * N;
    // CUDA does not guarantee that a zero-byte allocation or a zero-sized grid is valid.
    const size_t allocatedLocalElements = localElements == 0 ? 1 : localElements;

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "device discovery", rank);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    checkCuda(cudaSetDevice(localRank % deviceCount), "device selection", rank);
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N,
                    validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\nInitializing matrices...\n", worldSize, omp_get_max_threads());
    }

    std::vector<double> localA(localElements), B(N * N), localC(localElements);
    initLocalMatrix(localA, N, firstRow);
    if (rank == 0) initMatrix(B, N);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, allocatedLocalElements * sizeof(double)), "allocation of A", rank);
    checkCuda(cudaMalloc(&deviceB, N * N * sizeof(double)), "allocation of B", rank);
    checkCuda(cudaMalloc(&deviceC, allocatedLocalElements * sizeof(double)), "allocation of C", rank);
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream creation", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpyAsync(deviceA, localA.data(), localElements * sizeof(double), cudaMemcpyHostToDevice, stream), "copy of A", rank);
    checkCuda(cudaMemcpyAsync(deviceB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice, stream), "copy of B", rank);
    if (localRows != 0) {
        const dim3 block(TILE, TILE);
        const dim3 grid((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);
        matmulKernel<<<grid, block, 0, stream>>>(deviceA, deviceB, deviceC, localRows, N);
        checkCuda(cudaGetLastError(), "kernel launch", rank);
    }
    checkCuda(cudaMemcpyAsync(localC.data(), deviceC, localElements * sizeof(double), cudaMemcpyDeviceToHost, stream), "copy of C", rank);
    checkCuda(cudaStreamSynchronize(stream), "stream synchronization", rank);
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        counts[r] = static_cast<int>(rows * N);
        displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseRows +
                           (static_cast<size_t>(r) < extraRows ? r : extraRows)) * N);
    }
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);
    MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE, C.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    checkCuda(cudaStreamDestroy(stream), "stream destruction", rank);
    checkCuda(cudaFree(deviceA), "free of A", rank);
    checkCuda(cudaFree(deviceB), "free of B", rank);
    checkCuda(cudaFree(deviceC), "free of C", rank);

    int result = 0;
    if (rank == 0) {
        const auto milliseconds = static_cast<long>(std::llround(maxSeconds * 1000.0));
        std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", milliseconds,
                    (2.0 * N * N * N) / maxSeconds / 1e9);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            std::vector<double> A(N * N);
            initMatrix(A, N);
            if (validateResult(A, B, C, N)) std::printf("Validation: PASSED\n");
            else { std::printf("Validation: FAILED\n"); result = 1; }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
