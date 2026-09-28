#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),        \
                    __FILE__, __LINE__);                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a block of rows [rowStart, rowStart + rowCount) of an NxN matrix,
// using OpenMP threads on the host
void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t rowStart,
                     const size_t rowCount) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowStart + i, j);
        }
    }
}

constexpr int TILE = 32;

// Tiled shared-memory DGEMM: C (rows x N) = A (rows x N) * B (N x N)
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;

    double sum = 0.0;
    const size_t numTiles = (N + TILE - 1) / TILE;

    for (size_t t = 0; t < numTiles; ++t) {
        const size_t aCol = t * TILE + threadIdx.x;
        const size_t bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < rows && aCol < N) ? A[row * N + aCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && col < N) {
        C[row * N + col] = sum;
    }
}

// Distributed matrix multiplication: each rank computes its block of C rows on the
// GPU, then the blocks are gathered on rank 0
void matrixMultiply(const std::vector<double>& localA, const std::vector<double>& B,
                    std::vector<double>& localC, std::vector<double>& C, const size_t N,
                    const size_t rowCount, const std::vector<int>& counts,
                    const std::vector<int>& displs) {
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;

    if (rowCount > 0) {
        CUDA_CHECK(cudaMalloc(&dA, rowCount * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, rowCount * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(dA, localA.data(), rowCount * N * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

        const dim3 block(TILE, TILE);
        const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                        static_cast<unsigned>((rowCount + TILE - 1) / TILE));
        matmulKernel<<<grid, block>>>(dA, dB, dC, rowCount, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(localC.data(), dC, rowCount * N * sizeof(double),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    MPI_Gatherv(localC.data(), static_cast<int>(rowCount * N), MPI_DOUBLE, C.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
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

    // Bind each rank to a GPU by node-local rank
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // establish context up front

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", nranks,
               omp_get_max_threads(), deviceCount);
    }

    // Row-block decomposition of A and C across ranks
    const size_t rowsBase = N / nranks;
    const size_t rowsRem = N % nranks;
    const size_t rowStart = rank * rowsBase + std::min<size_t>(rank, rowsRem);
    const size_t rowCount = rowsBase + (static_cast<size_t>(rank) < rowsRem ? 1 : 0);

    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t rc = rowsBase + (static_cast<size_t>(r) < rowsRem ? 1 : 0);
        const size_t rs = r * rowsBase + std::min<size_t>(r, rowsRem);
        counts[r] = static_cast<int>(rc * N);
        displs[r] = static_cast<int>(rs * N);
    }

    // Allocate matrices: every rank holds full B and its row block of A and C;
    // rank 0 additionally holds full A (for validation) and the gathered C
    std::vector<double> A(rank == 0 ? N * N : rowCount * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(rowCount * N);
    std::vector<double> C(rank == 0 ? N * N : 0);

    // Initialize matrices (deterministic formula, so no communication needed)
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixBlock(A, N, rank == 0 ? 0 : rowStart, rank == 0 ? N : rowCount);
    initMatrixBlock(B, N, 0, N);

    // Rank 0 computes on its own leading rows of the full A
    const std::vector<double>& localA = A;

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(localA, B, localC, C, N, rowCount, counts, displs);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);

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
