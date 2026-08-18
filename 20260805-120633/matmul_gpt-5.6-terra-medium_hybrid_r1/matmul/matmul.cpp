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

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void checkCuda(const cudaError_t error, const char* const operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// A 32x32 tile has good occupancy for double precision on current CUDA devices
// while using only 16 KiB of shared memory for the two input tiles.
constexpr int Tile = 32;

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const int localRows, const int N) {
    __shared__ double aTile[Tile][Tile];
    __shared__ double bTile[Tile][Tile];

    const int row = blockIdx.y * Tile + threadIdx.y;
    const int col = blockIdx.x * Tile + threadIdx.x;
    double sum = 0.0;

    for (int base = 0; base < N; base += Tile) {
        const int aCol = base + threadIdx.x;
        const int bRow = base + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] =
            (row < localRows && aCol < N) ? A[static_cast<size_t>(row) * N + aCol] : 0.0;
        bTile[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < Tile; ++k) {
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < localRows && col < N) {
        C[static_cast<size_t>(row) * N + col] = sum;
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, static_cast<size_t>(i), j);
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
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
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool badArguments = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            badArguments = true;
        }
    }
    const size_t maxMatrixElements = static_cast<size_t>(std::numeric_limits<int>::max());
    if (badArguments || N == 0 || N > maxMatrixElements / N) {
        if (rank == 0) {
            std::printf(badArguments ? "Unknown option supplied\n" : "Matrix is too large for MPI counts\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Assign one CUDA device per local MPI rank, wrapping only when a node has
    // fewer GPUs than ranks.  The node-local communicator avoids cross-node IDs.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "querying CUDA devices");
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "selecting CUDA device");

    const int n = static_cast<int>(N);
    const int baseRows = n / ranks;
    const int remainder = n % ranks;
    const int localRows = baseRows + (rank < remainder ? 1 : 0);
    std::vector<int> rowCounts(ranks), rowOffsets(ranks), elementCounts(ranks), elementOffsets(ranks);
    int rowOffset = 0;
    for (int r = 0; r < ranks; ++r) {
        rowCounts[r] = baseRows + (r < remainder ? 1 : 0);
        rowOffsets[r] = rowOffset;
        elementCounts[r] = rowCounts[r] * n;
        elementOffsets[r] = rowOffset * n;
        rowOffset += rowCounts[r];
    }

    std::vector<double> A;
    std::vector<double> B(static_cast<size_t>(n) * n);
    std::vector<double> C;
    std::vector<double> localA(static_cast<size_t>(localRows) * n);
    std::vector<double> localC(static_cast<size_t>(localRows) * n);
    if (rank == 0) {
        A.resize(static_cast<size_t>(n) * n);
        C.resize(static_cast<size_t>(n) * n);
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                    N, N, validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }

    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, elementCounts.data(), elementOffsets.data(), MPI_DOUBLE,
                 localA.data(), localRows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    const size_t localBytes = localA.size() * sizeof(double);
    const size_t matrixBytes = B.size() * sizeof(double);
    // Some ranks can own no rows when ranks > N.  CUDA allocations must still
    // receive a non-zero size on those ranks, although no transfer or launch is made.
    checkCuda(cudaMalloc(&deviceA, localBytes == 0 ? sizeof(double) : localBytes), "allocating A");
    checkCuda(cudaMalloc(&deviceB, matrixBytes), "allocating B");
    checkCuda(cudaMalloc(&deviceC, localBytes == 0 ? sizeof(double) : localBytes), "allocating C");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    checkCuda(cudaMemcpy(deviceB, B.data(), matrixBytes, cudaMemcpyHostToDevice), "copying B to device");
    if (localRows > 0) {
        checkCuda(cudaMemcpy(deviceA, localA.data(), localBytes, cudaMemcpyHostToDevice), "copying A to device");
        const dim3 block(Tile, Tile);
        const dim3 grid((n + Tile - 1) / Tile, (localRows + Tile - 1) / Tile);
        matmulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, localRows, n);
        checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
        checkCuda(cudaDeviceSynchronize(), "synchronizing matrix multiplication kernel");
        checkCuda(cudaMemcpy(localC.data(), deviceC, localBytes, cudaMemcpyDeviceToHost), "copying C from device");
    }
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(localC.data(), localRows * n, MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                elementCounts.data(), elementOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    checkCuda(cudaFree(deviceA), "freeing A");
    checkCuda(cudaFree(deviceB), "freeing B");
    checkCuda(cudaFree(deviceC), "freeing C");
    MPI_Comm_free(&localComm);

    int result = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double gflops = (2.0 * N * N * N) / elapsedSeconds / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
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
