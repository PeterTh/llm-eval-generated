#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
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

bool parseSize(const char* arg, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(arg, &end, 10);
    if (errno != 0 || end == arg || *end != '\0') {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

#define CUDA_CHECK(call) \
    do { \
        const cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

constexpr int kTileSize = 16;

__global__ void matmulKernel(const double* A, const double* B, double* C, const int N, const int rows) {
    __shared__ double As[kTileSize][kTileSize];
    __shared__ double Bs[kTileSize][kTileSize];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTileSize + ty;
    const int col = blockIdx.x * kTileSize + tx;

    double sum = 0.0;
    for (int t = 0; t < N; t += kTileSize) {
        const int tiledCol = t + tx;
        const int tiledRow = t + ty;

        As[ty][tx] = (row < rows && tiledCol < N) ? A[row * N + tiledCol] : 0.0;
        Bs[ty][tx] = (col < N && tiledRow < N) ? B[tiledRow * N + col] : 0.0;
        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < rows && col < N) {
        C[row * N + col] = sum;
    }
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
            fprintf(stderr, "MPI implementation does not provide required threading support.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseOk = true;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                if (!parseSize(argv[++i], N)) {
                    fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                    parseOk = false;
                    break;
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
                break;
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                parseOk = false;
                break;
            }
        }

        if (showHelp || !parseOk) {
            printUsage(argv[0]);
        }
    }

    uint64_t N64 = static_cast<uint64_t>(N);
    int flags[4] = {parseOk ? 1 : 0, showHelp ? 1 : 0, validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(flags, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    parseOk = flags[0] != 0;
    showHelp = flags[1] != 0;
    validate = flags[2] != 0;
    printResults = flags[3] != 0;
    N = static_cast<size_t>(N64);

    if (!parseOk || showHelp) {
        MPI_Finalize();
        return parseOk ? 0 : 1;
    }

    if (N == 0) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size must be greater than zero.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
    if (N > maxInt || N > maxInt / N) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size is too large for MPI count limits.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", worldSize);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t totalElems = N * N;
    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t remainder = N % static_cast<size_t>(worldSize);

    std::vector<int> sendcounts(worldSize, 0);
    std::vector<int> displs(worldSize, 0);
    size_t offsetRows = 0;
    size_t localRows = 0;

    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        sendcounts[r] = static_cast<int>(rows * N);
        displs[r] = static_cast<int>(offsetRows * N);
        if (r == rank) {
            localRows = rows;
        }
        offsetRows += rows;
    }

    const size_t localElems = localRows * N;
    const int localCount = static_cast<int>(localElems);

    std::vector<double> A_full;
    std::vector<double> B_full(totalElems);
    std::vector<double> C_full;

    if (rank == 0) {
        A_full.resize(totalElems);
        initMatrix(A_full, N);
        initMatrix(B_full, N);
        if (printResults || validate) {
            C_full.resize(totalElems);
        }
    }

    MPI_Bcast(B_full.data(), static_cast<int>(totalElems), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> A_local(localElems);
    std::vector<double> C_local(localElems);

    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 localElems > 0 ? A_local.data() : nullptr, localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, localElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, totalElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, localElems * sizeof(double)));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(d_A, A_local.data(), localElems * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B_full.data(), totalElems * sizeof(double), cudaMemcpyHostToDevice));

        dim3 block(kTileSize, kTileSize);
        dim3 grid((static_cast<int>(N) + kTileSize - 1) / kTileSize,
                  (static_cast<int>(localRows) + kTileSize - 1) / kTileSize);

        matmulKernel<<<grid, block>>>(d_A, d_B, d_C, static_cast<int>(N), static_cast<int>(localRows));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C_local.data(), d_C, localElems * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Allreduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (localElems > 0) {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    if (printResults || validate) {
        MPI_Gatherv(localElems > 0 ? C_local.data() : nullptr, localCount, MPI_DOUBLE,
                    rank == 0 ? C_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int validationFlag = 1;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxTime * 1000.0);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              (maxTime * 1e9);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            validationFlag = validateResult(A_full, B_full, C_full, N) ? 1 : 0;
            printf("Validation: %s\n", validationFlag ? "PASSED" : "FAILED");
        }
    }

    if (validate) {
        MPI_Bcast(&validationFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return (validate && !validationFlag) ? 1 : 0;
}
