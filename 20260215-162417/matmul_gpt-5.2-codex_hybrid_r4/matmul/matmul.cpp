#include <algorithm>
#include <chrono>
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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

constexpr int kTileSize = 16;

inline void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
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

void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t rowOffset) {
    const size_t rows = mat.size() / N;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowOffset + i, j);
        }
    }
}

__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const size_t N,
                             const size_t rows) {
    __shared__ double As[kTileSize][kTileSize];
    __shared__ double Bs[kTileSize][kTileSize];

    const size_t row = blockIdx.y * kTileSize + threadIdx.y;
    const size_t col = blockIdx.x * kTileSize + threadIdx.x;
    double sum = 0.0;

    const size_t tiles = (N + kTileSize - 1) / kTileSize;
    for (size_t t = 0; t < tiles; ++t) {
        const size_t tiledCol = t * kTileSize + threadIdx.x;
        const size_t tiledRow = t * kTileSize + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < rows && tiledCol < N) ? A[row * N + tiledCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] =
            (tiledRow < N && col < N) ? B[tiledRow * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMultiplyCUDA(const std::vector<double>& A,
                        const std::vector<double>& B,
                        std::vector<double>& C,
                        const size_t N,
                        const size_t rows) {
    if (rows == 0) {
        return;
    }

    const size_t aBytes = rows * N * sizeof(double);
    const size_t bBytes = N * N * sizeof(double);
    const size_t cBytes = rows * N * sizeof(double);

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    checkCuda(cudaMalloc(&dA, aBytes), "cudaMalloc A");
    checkCuda(cudaMalloc(&dB, bBytes), "cudaMalloc B");
    checkCuda(cudaMalloc(&dC, cBytes), "cudaMalloc C");

    checkCuda(cudaMemcpy(dA, A.data(), aBytes, cudaMemcpyHostToDevice), "copy A to device");
    checkCuda(cudaMemcpy(dB, B.data(), bBytes, cudaMemcpyHostToDevice), "copy B to device");

    dim3 block(kTileSize, kTileSize);
    dim3 grid((N + kTileSize - 1) / kTileSize,
              (rows + kTileSize - 1) / kTileSize);
    matmulKernel<<<grid, block>>>(dA, dB, dC, N, rows);
    checkCuda(cudaGetLastError(), "matmul kernel launch");
    checkCuda(cudaDeviceSynchronize(), "matmul kernel sync");

    checkCuda(cudaMemcpy(C.data(), dC, cBytes, cudaMemcpyDeviceToHost), "copy C to host");

    checkCuda(cudaFree(dA), "cudaFree A");
    checkCuda(cudaFree(dB), "cudaFree B");
    checkCuda(cudaFree(dC), "cudaFree C");
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseOk = 1;
    int exitCode = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseOk = 0;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseOk = 0;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&parseOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOk) {
        MPI_Finalize();
        return exitCode;
    }

    uint64_t nValue = static_cast<uint64_t>(N);
    MPI_Bcast(&nValue, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(nValue);

    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");

    const size_t baseRows = N / static_cast<size_t>(size);
    const size_t remainder = N % static_cast<size_t>(size);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t rowOffset =
        baseRows * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);

    std::vector<double> A_local(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(localRows * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixFull(B, N);
    if (localRows > 0) {
        initMatrixBlock(A_local, N, rowOffset);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    const double start = MPI_Wtime();
    matrixMultiplyCUDA(A_local, B, C_local, N, localRows);
    const double end = MPI_Wtime();
    const double localTime = end - start;

    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> recvCounts;
    std::vector<int> displs;
    std::vector<double> C;
    if (rank == 0) {
        recvCounts.resize(size);
        displs.resize(size);
        size_t offset = 0;
        for (int r = 0; r < size; ++r) {
            const size_t rows =
                baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
            recvCounts[r] = static_cast<int>(rows * N);
            displs[r] = static_cast<int>(offset * N);
            offset += rows;
        }
        C.resize(N * N);
    }

    MPI_Gatherv(C_local.data(),
                static_cast<int>(localRows * N),
                MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              maxTime / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(C, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
