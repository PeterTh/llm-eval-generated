#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr int TILE = 32;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixFull(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowOffset, const size_t rows) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowOffset + i, j);
        }
    }
}

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

inline void checkMpi(int err, const char* msg) {
    if (err != MPI_SUCCESS) {
        char errStr[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(err, errStr, &len);
        fprintf(stderr, "MPI error (%s): %s\n", msg, errStr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, int M, int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;
    const int tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        const int tiledCol = t * TILE + threadIdx.x;
        const int tiledRow = t * TILE + threadIdx.y;

        if (row < M && tiledCol < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (tiledRow < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMultiplyGPU(const std::vector<double>& A, const std::vector<double>& B,
                       std::vector<double>& C, const size_t M, const size_t N, int device) {
    if (M == 0 || N == 0) {
        return;
    }

    checkCuda(cudaSetDevice(device), "cudaSetDevice");

    const size_t bytesA = M * N * sizeof(double);
    const size_t bytesB = N * N * sizeof(double);
    const size_t bytesC = M * N * sizeof(double);

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t stream;

    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    checkCuda(cudaMalloc(&dA, bytesA), "cudaMalloc A");
    checkCuda(cudaMalloc(&dB, bytesB), "cudaMalloc B");
    checkCuda(cudaMalloc(&dC, bytesC), "cudaMalloc C");

    checkCuda(cudaMemcpyAsync(dA, A.data(), bytesA, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync A");
    checkCuda(cudaMemcpyAsync(dB, B.data(), bytesB, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync B");

    dim3 block(TILE, TILE);
    dim3 grid((static_cast<int>(N) + TILE - 1) / TILE, (static_cast<int>(M) + TILE - 1) / TILE);

    matmulKernel<<<grid, block, 0, stream>>>(dA, dB, dC, static_cast<int>(M), static_cast<int>(N));
    checkCuda(cudaGetLastError(), "matmulKernel");

    checkCuda(cudaMemcpyAsync(C.data(), dC, bytesC, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync C");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    checkCuda(cudaFree(dA), "cudaFree A");
    checkCuda(cudaFree(dB), "cudaFree B");
    checkCuda(cudaFree(dC), "cudaFree C");
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                const double a = getPseudoRndValue(N, i, k);
                expected += a * B[k * N + j];
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
    checkMpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");

    int worldRank = 0;
    int worldSize = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");
    const bool isRoot = (worldRank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argError = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argError = true;
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp || argError) {
        if (isRoot) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argError ? 1 : 0;
    }

    if (N == 0) {
        if (isRoot) {
            printf("Matrix size must be greater than zero.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (isRoot) {
            printf("Matrix size too large for CUDA indexing.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t totalElements = N * N;
    if (totalElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (isRoot) {
            printf("Matrix size too large for MPI counts.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (isRoot) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
        if (provided < MPI_THREAD_FUNNELED) {
            printf("Warning: MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
    }

    const size_t rowsPerRank = N / static_cast<size_t>(worldSize);
    const size_t remainder = N % static_cast<size_t>(worldSize);
    const size_t localRows = rowsPerRank + ((static_cast<size_t>(worldRank) < remainder) ? 1 : 0);
    const size_t rowOffset = rowsPerRank * static_cast<size_t>(worldRank) +
                             std::min(static_cast<size_t>(worldRank), remainder);

    std::vector<double> B(totalElements);
    if (isRoot) {
        printf("Initializing matrices...\n");
        initMatrixFull(B, N);
    }
    checkMpi(MPI_Bcast(B.data(), static_cast<int>(totalElements), MPI_DOUBLE, 0, MPI_COMM_WORLD),
             "MPI_Bcast B");

    std::vector<double> A_local(localRows * N);
    std::vector<double> C_local(localRows * N);
    initMatrixRows(A_local, N, rowOffset, localRows);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (isRoot) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = worldRank % deviceCount;

    if (isRoot) {
        printf("Computing matrix multiplication...\n");
    }

    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    const double start = MPI_Wtime();

    if (localRows > 0) {
        matrixMultiplyGPU(A_local, B, C_local, localRows, N, device);
    }

    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    checkMpi(MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             "MPI_Reduce time");

    if (isRoot) {
        if (maxTime <= 0.0) {
            maxTime = 1e-9;
        }
        printf("Computation time: %.3f ms\n", maxTime * 1000.0);
        const double n = static_cast<double>(N);
        const double gflops = (2.0 * n * n * n) / (maxTime * 1e9);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults || validate) {
        std::vector<double> C;
        if (isRoot) {
            C.resize(totalElements);
        }

        std::vector<int> recvCounts;
        std::vector<int> displs;
        if (isRoot) {
            recvCounts.resize(worldSize);
            displs.resize(worldSize);
            for (int r = 0; r < worldSize; ++r) {
                const size_t rRows = rowsPerRank + ((static_cast<size_t>(r) < remainder) ? 1 : 0);
                const size_t rOffset = rowsPerRank * static_cast<size_t>(r) +
                                       std::min(static_cast<size_t>(r), remainder);
                recvCounts[r] = static_cast<int>(rRows * N);
                displs[r] = static_cast<int>(rOffset * N);
            }
        }

        const int localCount = static_cast<int>(localRows * N);
        checkMpi(MPI_Gatherv(C_local.data(), localCount, MPI_DOUBLE,
                             isRoot ? C.data() : nullptr,
                             isRoot ? recvCounts.data() : nullptr,
                             isRoot ? displs.data() : nullptr,
                             MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv C");

        if (printResults && isRoot) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            int valid = 1;
            if (isRoot) {
                printf("Validating result...\n");
                valid = validateResult(B, C, N) ? 1 : 0;
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
            checkMpi(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast validation");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
