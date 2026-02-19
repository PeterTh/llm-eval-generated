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

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                          \
        cudaError_t _e = (call);                                                                  \
        if (_e != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            MPI_Abort(MPI_COMM_WORLD, 2);                                                         \
        }                                                                                         \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Simple validation: compute a single element and compare
static bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                           const std::vector<double>& C, const size_t N) {
    // Check a few fixed positions (mod N for small N)
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

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const double* __restrict__ A, const double* __restrict__ B,
                                    double* __restrict__ C, int M, int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;
    const int tilesK = (N + TILE - 1) / TILE;

    for (int t = 0; t < tilesK; ++t) {
        const int kA = t * TILE + threadIdx.x;
        const int kB = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && kA < N) ? A[row * N + kA] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (kB < N && col < N) ? B[kB * N + col] : 0.0;

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

static double runCudaMatmul(const std::vector<double>& A_local, const std::vector<double>& B,
                            std::vector<double>& C_local, size_t localRows, size_t N) {
    const size_t bytesA = localRows * N * sizeof(double);
    const size_t bytesB = N * N * sizeof(double);
    const size_t bytesC = localRows * N * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, A_local.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((static_cast<unsigned>(N) + TILE - 1) / TILE,
              (static_cast<unsigned>(localRows) + TILE - 1) / TILE);

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled_kernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(localRows), static_cast<int>(N));
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C_local.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return static_cast<double>(ms);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
    if (N == 0 || N > maxInt || (N > 0 && N > maxInt / N)) {
        if (rank == 0) {
            fprintf(stderr, "Invalid/too-large N for MPI counts: %zu\n", N);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    const size_t localRows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    std::vector<int> sendcounts;
    std::vector<int> displs;

    std::vector<double> A_full;
    std::vector<double> B;
    std::vector<double> C_full;

    if (rank == 0) {
        sendcounts.resize(size);
        displs.resize(size);
        size_t disp = 0;
        for (int r = 0; r < size; ++r) {
            const size_t rows = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t cnt = rows * N;
            if (cnt > static_cast<size_t>(std::numeric_limits<int>::max())) {
                fprintf(stderr, "Matrix too large for MPI int counts.\n");
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            sendcounts[r] = static_cast<int>(cnt);
            displs[r] = static_cast<int>(disp);
            disp += cnt;
        }

        A_full.resize(N * N);
        B.resize(N * N);
        C_full.resize(N * N);

        printf("Initializing matrices...\n");
        initMatrix(A_full, N);
        initMatrix(B, N);
    } else {
        B.resize(N * N);
    }

    std::vector<double> A_local(localRows * N);
    std::vector<double> C_local(localRows * N);

    // Distribute A rows and replicate B to all ranks.
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, A_local.data(),
                 static_cast<int>(localRows * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    const double localMs = runCudaMatmul(A_local, B, C_local, localRows, N);

    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(C_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C_full.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMs);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResult(A_full, B, C_full, N);
            if (ok) {
                printf("Validation: PASSED\n");
                rc = 0;
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
