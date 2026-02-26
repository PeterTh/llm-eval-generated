#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// Generate pseudo-random values for matrix initialization (deterministic)
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

static void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowOffset,
                           const size_t numRows) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t li = 0; li < numRows; ++li) {
        for (size_t j = 0; j < N; ++j) {
            const size_t i = rowOffset + li;
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static bool validateResultDeterministic(const std::vector<double>& C, const size_t N) {
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

static constexpr int TILE = 16;

__global__ void matmul_tiled(const double* __restrict__ A_local, const double* __restrict__ B,
                             double* __restrict__ C_local, int N, int localRows) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const int col = blockIdx.x * TILE + threadIdx.x;
    const int localRow = blockIdx.y * TILE + threadIdx.y;

    double sum = 0.0;

    for (int k0 = 0; k0 < N; k0 += TILE) {
        const int aCol = k0 + threadIdx.x;
        const int bRow = k0 + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (localRow < localRows && aCol < N)
                                           ? A_local[localRow * N + aCol]
                                           : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;

        __syncthreads();

#pragma unroll
        for (int t = 0; t < TILE; ++t) {
            sum += As[threadIdx.y][t] * Bs[t][threadIdx.x];
        }

        __syncthreads();
    }

    if (localRow < localRows && col < N) {
        C_local[localRow * N + col] = sum;
    }
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    uint64_t N64 = 512;
    int validate = 0;
    int printResults = 0;
    int shouldExit = 0;
    int exitCode = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N64 = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&N64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    const size_t N = static_cast<size_t>(N64);
    if (N == 0) {
        if (rank == 0) {
            printf("Matrix size must be > 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Determine local rank on the node for GPU selection
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree((void*)0)); // initialize context

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", world);
        printf("OpenMP threads (rank 0): %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block distribution of rows of C across ranks
    const size_t baseRows = N / static_cast<size_t>(world);
    const size_t rem = N % static_cast<size_t>(world);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t rowStart = baseRows * static_cast<size_t>(rank) +
                            std::min(static_cast<size_t>(rank), rem);

    // Guard MPI_Gatherv int counts
    if (static_cast<unsigned long long>(N) * static_cast<unsigned long long>(N) >
        static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            fprintf(stderr, "N is too large for MPI_Gatherv counts (N*N overflows int).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate and initialize inputs (deterministic)
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    std::vector<double> B(N * N);
    initMatrix(B, N);

    std::vector<double> A_local(localRows * N);
    initMatrixRows(A_local, N, rowStart, localRows);

    std::vector<double> C_local(localRows * N);

    // Device buffers (only allocate what we need)
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), A_local.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), B.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), C_local.size() * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(dA, A_local.data(), A_local.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    if (localRows > 0) {
        const dim3 block(TILE, TILE);
        const dim3 grid((static_cast<unsigned int>(N) + TILE - 1) / TILE,
                        (static_cast<unsigned int>(localRows) + TILE - 1) / TILE);
        matmul_tiled<<<grid, block>>>(dA, dB, dC, static_cast<int>(N), static_cast<int>(localRows));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C_local.data(), dC, C_local.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localTime = (t1 - t0);
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather C on rank 0 for printing/validation
    std::vector<double> C;
    std::vector<int> recvCounts, displs;
    if (rank == 0) {
        C.resize(N * N);
        recvCounts.resize(world);
        displs.resize(world);
        for (int r = 0; r < world; ++r) {
            const size_t lr = baseRows + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t rs = baseRows * static_cast<size_t>(r) +
                              std::min(static_cast<size_t>(r), rem);
            recvCounts[r] = static_cast<int>(lr * N);
            displs[r] = static_cast<int>(rs * N);
        }
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Cleanup device memory
    if (dA) CUDA_CHECK(cudaFree(dA));
    if (dB) CUDA_CHECK(cudaFree(dB));
    if (dC) CUDA_CHECK(cudaFree(dC));

    if (rank == 0) {
        const long ms = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double time_s = std::max(maxTime, 1e-12);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              time_s / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResultDeterministic(C, N);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            MPI_Finalize();
            return ok ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
