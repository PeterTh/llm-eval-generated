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

static void checkCuda(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, 1);
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        for (size_t j = 0; j < N; ++j)
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, static_cast<size_t>(i), j);
    }
}

// Each rank owns a contiguous set of rows.  The kernel is deliberately tiled
// so B is reused from shared memory by an entire CUDA thread block.
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t rows, size_t N) {
    __shared__ double aTile[TILE][TILE];
    __shared__ double bTile[TILE][TILE];

    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    double sum = 0.0;

    for (size_t base = 0; base < N; base += TILE) {
        const size_t aCol = base + threadIdx.x;
        const size_t bRow = base + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (row < rows && aCol < N) ? A[row * N + aCol] : 0.0;
        bTile[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < N)
        C[row * N + col] = sum;
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    int failed = 0;
    #pragma omp parallel for collapse(2) reduction(|:failed) schedule(static)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[i * N + k] * B[k * N + j];
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                #pragma omp critical
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                failed = 1;
            }
        }
    }
    return failed == 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Matrix size N (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
        if (rank == 0) std::fprintf(stderr, "Matrix size is invalid or exceeds MPI count limits.\n");
        MPI_Finalize(); return 1;
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "device discovery", MPI_COMM_WORLD);
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    checkCuda(cudaSetDevice(rank % deviceCount), "device selection", MPI_COMM_WORLD);

    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t extraRows = N % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows);
    const size_t localElements = localRows * N;
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows);
        counts[r] = static_cast<int>(rows * N);
        displacements[r] = r == 0 ? 0 : displacements[r - 1] + counts[r - 1];
    }

    std::vector<double> A, B(N * N), C;
    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark (MPI ranks: %d, CUDA GPUs: %d, OpenMP threads: %d)\n",
                    ranks, deviceCount, omp_get_max_threads());
        std::printf("Matrix size: %zu x %zu\nValidation: %s\nInitializing matrices...\n", N, N,
                    validate ? "enabled" : "disabled");
        A.resize(N * N); C.resize(N * N);
        initMatrix(A, N); initMatrix(B, N);
    }
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> localA(localElements), localC(localElements);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 localA.data(), static_cast<int>(localElements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    checkCuda(cudaMalloc(&dB, N * N * sizeof(double)), "allocation of B", MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy of B", MPI_COMM_WORLD);
    if (localElements != 0) {
        checkCuda(cudaMalloc(&dA, localElements * sizeof(double)), "allocation of A", MPI_COMM_WORLD);
        checkCuda(cudaMalloc(&dC, localElements * sizeof(double)), "allocation of C", MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(dA, localA.data(), localElements * sizeof(double), cudaMemcpyHostToDevice), "copy of A", MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    if (localElements != 0) {
        const dim3 block(TILE, TILE);
        const dim3 grid((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);
        matmulKernel<<<grid, block>>>(dA, dB, dC, localRows, N);
        checkCuda(cudaGetLastError(), "kernel launch", MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(localC.data(), dC, localElements * sizeof(double), cudaMemcpyDeviceToHost), "copy of C", MPI_COMM_WORLD);
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double globalSeconds = 0.0;
    MPI_Reduce(&localSeconds, &globalSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    int result = 0;
    if (rank == 0) {
        const auto millis = static_cast<long long>(globalSeconds * 1000.0);
        std::printf("Computation time: %lld ms\n", millis);
        std::printf("Performance: %.3f GFLOPS\n", globalSeconds > 0 ? (2.0 * N * N * N) / globalSeconds / 1e9 : 0.0);
        if (printResults) print_results(C, "MatrixC");
        if (validate) { std::printf("Validating result...\n"); result = validateResult(A, B, C, N) ? 0 : 1; std::printf("Validation: %s\n", result ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
