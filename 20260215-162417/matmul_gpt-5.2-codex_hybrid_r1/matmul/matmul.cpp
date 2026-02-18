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

constexpr int kTileSize = 16;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

#define CHECK_CUDA(call)                                                                      \
    do {                                                                                      \
        const cudaError_t err = (call);                                                       \
        if (err != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                     \
        }                                                                                     \
    } while (0)

__global__ void matmulKernel(const double* A, const double* B, double* C, size_t N,
                             size_t localRows) {
    __shared__ double As[kTileSize][kTileSize];
    __shared__ double Bs[kTileSize][kTileSize];

    const size_t row = static_cast<size_t>(blockIdx.y) * kTileSize + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * kTileSize + threadIdx.x;

    double sum = 0.0;
    for (size_t tile = 0; tile < N; tile += kTileSize) {
        const size_t tiledCol = tile + threadIdx.x;
        const size_t tiledRow = tile + threadIdx.y;

        if (row < localRows && tiledCol < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < N && tiledRow < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

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

void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t startRow,
                     const size_t rows) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, startRow + i, j);
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& C, const size_t N) {
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI does not provide required thread support.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseError = true;
                break;
            }
        }
    }

    uint64_t n64 = static_cast<uint64_t>(N);
    int validateFlag = validate ? 1 : 0;
    int printFlag = printResults ? 1 : 0;
    int helpFlag = showHelp ? 1 : 0;
    int errorFlag = parseError ? 1 : 0;

    MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&helpFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&errorFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    N = static_cast<size_t>(n64);
    validate = validateFlag != 0;
    printResults = printFlag != 0;
    showHelp = helpFlag != 0;
    parseError = errorFlag != 0;

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (parseError) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t remainder = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t rowOffset =
        baseRows * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK_CUDA(cudaSetDevice(rank % deviceCount));

    std::vector<double> A_local(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(localRows * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixBlock(A_local, N, rowOffset, localRows);
    initMatrixBlock(B, N, 0, N);
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    const size_t aBytes = localRows * N * sizeof(double);
    const size_t bBytes = N * N * sizeof(double);
    const size_t cBytes = localRows * N * sizeof(double);

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    if (localRows > 0) {
        CHECK_CUDA(cudaMalloc(&dA, aBytes));
        CHECK_CUDA(cudaMalloc(&dC, cBytes));
        CHECK_CUDA(cudaMemcpy(dA, A_local.data(), aBytes, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMalloc(&dB, bBytes));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), bBytes, cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double elapsedMs = 0.0;
    if (localRows > 0) {
        dim3 block(kTileSize, kTileSize);
        dim3 grid((N + kTileSize - 1) / kTileSize, (localRows + kTileSize - 1) / kTileSize);

        CHECK_CUDA(cudaEventRecord(start));
        matmulKernel<<<grid, block>>>(dA, dB, dC, N, localRows);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));

        CHECK_CUDA(cudaMemcpy(C_local.data(), dC, cBytes, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    if (dA != nullptr) {
        CHECK_CUDA(cudaFree(dA));
    }
    if (dB != nullptr) {
        CHECK_CUDA(cudaFree(dB));
    }
    if (dC != nullptr) {
        CHECK_CUDA(cudaFree(dC));
    }

    std::vector<double> C;
    std::vector<int> recvcounts;
    std::vector<int> displs;

    if (rank == 0) {
        C.resize(N * N);
        recvcounts.resize(worldSize);
        displs.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
            recvcounts[r] = static_cast<int>(rows * N);
            displs[r] = static_cast<int>((baseRows * static_cast<size_t>(r) +
                                          std::min(static_cast<size_t>(r), remainder)) *
                                         N);
        }
    }

    const int sendcount = static_cast<int>(localRows * N);
    MPI_Gatherv(C_local.data(), sendcount, MPI_DOUBLE,
                C.empty() ? nullptr : C.data(),
                recvcounts.empty() ? nullptr : recvcounts.data(),
                displs.empty() ? nullptr : displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double localTimeSec = elapsedMs / 1000.0;
    double maxTimeSec = 0.0;
    MPI_Reduce(&localTimeSec, &maxTimeSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxTimeSec * 1000.0);
        if (maxTimeSec > 0.0) {
            const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                                   static_cast<double>(N)) /
                                  maxTimeSec / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        } else {
            printf("Performance: 0.000 GFLOPS\n");
        }
    }

    if (printResults && rank == 0) {
        print_results(C, "MatrixC");
    }

    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = validateResult(C, N);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
