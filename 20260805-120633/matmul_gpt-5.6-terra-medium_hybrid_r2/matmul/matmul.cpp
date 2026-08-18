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

namespace {

constexpr int kTile = 32;

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    }
}

__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C, int localRows, int n) {
    __shared__ double aTile[kTile][kTile];
    __shared__ double bTile[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    double sum = 0.0;

    for (int tile = 0; tile < n; tile += kTile) {
        const int aCol = tile + threadIdx.x;
        const int bRow = tile + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] =
            (row < localRows && aCol < n) ? A[row * n + aCol] : 0.0;
        bTile[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? B[bRow * n + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < localRows && col < n) {
        C[row * n + col] = sum;
    }
}

void initLocalA(std::vector<double>& matrix, size_t n, size_t firstRow) {
    const size_t rows = matrix.size() / n;
    #pragma omp parallel for schedule(static)
    for (long long localRow = 0; localRow < static_cast<long long>(rows); ++localRow) {
        const size_t globalRow = firstRow + static_cast<size_t>(localRow);
        for (size_t col = 0; col < n; ++col) {
            matrix[static_cast<size_t>(localRow) * n + col] =
                getPseudoRndValue(n, globalRow, col);
        }
    }
}

void initMatrix(std::vector<double>& matrix, size_t n) {
    #pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(n); ++row) {
        for (size_t col = 0; col < n; ++col) {
            matrix[static_cast<size_t>(row) * n + col] =
                getPseudoRndValue(n, static_cast<size_t>(row), col);
        }
    }
}

bool validateResult(const std::vector<double>& B, const std::vector<double>& C, size_t n) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t row = checkPoints[pi] % n;
            const size_t col = checkPoints[pj] % n;
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k) {
                expected += getPseudoRndValue(n, row, k) * B[k * n + col];
            }
            const double actual = C[row * n + col];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            row, col, expected, actual, relError);
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

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::sqrt(std::numeric_limits<int>::max()))) {
        if (rank == 0) std::fprintf(stderr, "Matrix size must be between 1 and 46340.\n");
        MPI_Finalize();
        return 1;
    }
    const int nInt = static_cast<int>(n);
    const size_t baseRows = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            (static_cast<size_t>(rank) < remainder ? rank : remainder);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "querying CUDA devices");
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are available.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Map ranks to devices on their own node, rather than using global MPI rank.
    // This remains correct when each cluster node has a different rank offset.
    checkCuda(cudaSetDevice(localRank % deviceCount), "selecting CUDA device");

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", n, n);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA tile: %d\n",
                    ranks, omp_get_max_threads(), kTile);
        std::printf("Validation: %s\nInitializing matrices...\n",
                    validate ? "enabled" : "disabled");
    }

    std::vector<double> localA(localRows * n);
    std::vector<double> B(n * n);
    initLocalA(localA, n, firstRow);
    if (rank == 0) initMatrix(B, n);
    MPI_Bcast(B.data(), nInt * nInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> recvCounts;
    std::vector<int> displacements;
    std::vector<double> C;
    if (rank == 0) {
        recvCounts.resize(ranks);
        displacements.resize(ranks);
        int displacement = 0;
        for (int process = 0; process < ranks; ++process) {
            const size_t processRows = baseRows + (static_cast<size_t>(process) < remainder ? 1 : 0);
            recvCounts[process] = static_cast<int>(processRows * n);
            displacements[process] = displacement;
            displacement += recvCounts[process];
        }
        C.resize(n * n);
    }

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceB, n * n * sizeof(double)), "allocating B");
    if (localRows != 0) {
        checkCuda(cudaMalloc(&deviceA, localRows * n * sizeof(double)), "allocating A");
        checkCuda(cudaMalloc(&deviceC, localRows * n * sizeof(double)), "allocating C");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    checkCuda(cudaMemcpy(deviceB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice),
              "copying B to device");
    if (localRows != 0) {
        checkCuda(cudaMemcpy(deviceA, localA.data(), localRows * n * sizeof(double), cudaMemcpyHostToDevice),
                  "copying A to device");
        const dim3 block(kTile, kTile);
        const dim3 grid((nInt + kTile - 1) / kTile,
                        (static_cast<int>(localRows) + kTile - 1) / kTile);
        matmulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, static_cast<int>(localRows), nInt);
        checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
        checkCuda(cudaMemcpy(localA.data(), deviceC, localRows * n * sizeof(double), cudaMemcpyDeviceToHost),
                  "copying C from device");
    }
    const double localElapsed = MPI_Wtime() - start;

    MPI_Gatherv(localRows == 0 ? nullptr : localA.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const auto milliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double gflops = elapsed > 0.0 ? (2.0 * n * n * n) / elapsed / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(B, C, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    if (deviceA) checkCuda(cudaFree(deviceA), "freeing A");
    if (deviceB) checkCuda(cudaFree(deviceB), "freeing B");
    if (deviceC) checkCuda(cudaFree(deviceC), "freeing C");
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
