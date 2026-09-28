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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// OpenMP-parallel initialization of a row block [rowStart, rowEnd) of the matrix
void initMatrixRows(double* mat, const size_t N, const size_t rowStart, const size_t rowEnd) {
#pragma omp parallel for schedule(static)
    for (size_t i = rowStart; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowStart) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Tiled shared-memory matrix multiplication kernel.
// Computes C[localRows x N] = A[localRows x N] * B[N x N], where A holds the
// row block of the global matrix assigned to this rank.
constexpr int TILE = 32;

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const int localRows, const int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    double sum = 0.0;
    const int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        As[ty][tx] = (row < localRows && aCol < N) ? A[static_cast<size_t>(row) * N + aCol] : 0.0;
        Bs[ty][tx] = (bRow < N && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < localRows && col < N) {
        C[static_cast<size_t>(row) * N + col] = sum;
    }
}

// Multiply this rank's row block of A with B on the local GPU
void matrixMultiplyGPU(const std::vector<double>& Ablock, const std::vector<double>& B,
                       std::vector<double>& Cblock, const size_t N, const size_t localRows) {
    if (localRows == 0) return;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t blockBytes = localRows * N * sizeof(double);
    const size_t fullBytes = N * N * sizeof(double);

    CUDA_CHECK(cudaMalloc(&dA, blockBytes));
    CUDA_CHECK(cudaMalloc(&dB, fullBytes));
    CUDA_CHECK(cudaMalloc(&dC, blockBytes));

    CUDA_CHECK(cudaMemcpy(dA, Ablock.data(), blockBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), fullBytes, cudaMemcpyHostToDevice));

    const dim3 threads(TILE, TILE);
    const dim3 blocks((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);
    matmulKernel<<<blocks, threads>>>(dA, dB, dC, static_cast<int>(localRows),
                                      static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(Cblock.data(), dC, blockBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
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

    // Bind each rank to a local GPU (round-robin over visible devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // establish context before timing

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per node: %d, OpenMP threads: %d\n",
               nranks, deviceCount, omp_get_max_threads());
    }

    // Row-block decomposition of C (and A) across ranks
    const size_t base = N / nranks;
    const size_t rem = N % nranks;
    const size_t rowStart = rank * base + std::min<size_t>(rank, rem);
    const size_t localRows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t rowEnd = rowStart + localRows;

    // Allocate matrices: each rank holds its row block of A and C, plus full B.
    // Rank 0 additionally holds full A (for validation) and full C (gather target).
    std::vector<double> Ablock(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> Cblock(localRows * N);
    std::vector<double> A, C;
    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
    }

    // Initialize matrices (deterministic, so each rank initializes locally
    // without communication; OpenMP-parallel)
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(Ablock.data(), N, rowStart, rowEnd);
    initMatrixRows(B.data(), N, 0, N);
    if (rank == 0) initMatrixRows(A.data(), N, 0, N);

    // Gather layout for assembling C on rank 0
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t rRows = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        const size_t rStart = r * base + std::min<size_t>(r, rem);
        counts[r] = static_cast<int>(rRows * N);
        displs[r] = static_cast<int>(rStart * N);
    }

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyGPU(Ablock, B, Cblock, N, localRows);

    MPI_Gatherv(Cblock.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDurationMs / 1000.0) / 1e9;
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
