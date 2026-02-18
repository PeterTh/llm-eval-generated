#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr int kTileSize = 16;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t localRows,
                    const size_t rowOffset) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t li = 0; li < localRows; ++li) {
        for (size_t j = 0; j < N; ++j) {
            const size_t i = rowOffset + li;
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrixFull(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

__global__ void matmulKernel(const double* A, const double* B, double* C, int N, int localRows) {
    __shared__ double As[kTileSize][kTileSize];
    __shared__ double Bs[kTileSize][kTileSize];

    const int row = blockIdx.y * kTileSize + threadIdx.y;
    const int col = blockIdx.x * kTileSize + threadIdx.x;
    double sum = 0.0;

    const int tiles = (N + kTileSize - 1) / kTileSize;
    for (int t = 0; t < tiles; ++t) {
        const int aCol = t * kTileSize + threadIdx.x;
        const int bRow = t * kTileSize + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < localRows && aCol < N) ? A[row * N + aCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < localRows && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMultiplyCuda(const std::vector<double>& A, const std::vector<double>& B,
                        std::vector<double>& C, const size_t N, const size_t localRows) {
    if (localRows == 0 || N == 0) {
        return;
    }

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    const size_t bytesA = localRows * N * sizeof(double);
    const size_t bytesB = N * N * sizeof(double);
    const size_t bytesC = localRows * N * sizeof(double);

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 block(kTileSize, kTileSize);
    dim3 grid((static_cast<unsigned int>(N) + kTileSize - 1) / kTileSize,
              (static_cast<unsigned int>(localRows) + kTileSize - 1) / kTileSize);

    matmulKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N), static_cast<int>(localRows));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t nValue = 512;
    int validateInt = 0;
    int printResultsInt = 0;
    int parseStatus = 0; // 0 = ok, 1 = error, 2 = help

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                nValue = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
            } else if (strcmp(argv[i], "-v") == 0) {
                validateInt = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResultsInt = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 1 : 0;
    }

    MPI_Bcast(&nValue, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t N = static_cast<size_t>(nValue);
    const bool validate = validateInt != 0;
    const bool printResults = printResultsInt != 0;

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t rowsPerRank = N / static_cast<size_t>(size);
    const size_t remainder = N % static_cast<size_t>(size);
    const size_t localRows = rowsPerRank + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t rowOffset = rowsPerRank * static_cast<size_t>(rank)
        + (static_cast<size_t>(rank) < remainder ? static_cast<size_t>(rank) : remainder);

    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(A, N, localRows, rowOffset);
    initMatrixFull(B, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    matrixMultiplyCuda(A, B, C, N, localRows);

    const double localElapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long durationMs = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> Cfull;
    if (printResults || validate) {
        if (rank == 0) {
            Cfull.resize(N * N);
        }

        std::vector<int> recvCounts(size, 0);
        std::vector<int> displs(size, 0);
        size_t offset = 0;
        for (int r = 0; r < size; ++r) {
            const size_t rows = rowsPerRank + (static_cast<size_t>(r) < remainder ? 1 : 0);
            recvCounts[r] = static_cast<int>(rows * N);
            displs[r] = static_cast<int>(offset * N);
            offset += rows;
        }

        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    rank == 0 ? Cfull.data() : nullptr, recvCounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(Cfull, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(Cfull, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
