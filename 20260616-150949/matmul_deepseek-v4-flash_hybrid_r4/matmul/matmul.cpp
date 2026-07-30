#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Tile size for CUDA shared memory tiling
constexpr int TILE_SIZE = 32;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                          \
        }                                                                     \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize matrix with OpenMP parallel for
void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// CUDA kernel: tiled matrix multiplication using shared memory
// C (M x N) = A (M x K) * B (K x N)
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                              double* __restrict__ C, size_t M, size_t K, size_t N) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    size_t row = by * TILE_SIZE + ty;
    size_t col = bx * TILE_SIZE + tx;

    double sum = 0.0;
    size_t numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (size_t t = 0; t < numTiles; ++t) {
        size_t tiledK = t * TILE_SIZE;

        if (row < M && tiledK + tx < K)
            As[ty][tx] = A[row * K + tiledK + tx];
        else
            As[ty][tx] = 0.0;

        if (col < N && tiledK + ty < K)
            Bs[ty][tx] = B[(tiledK + ty) * N + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Hybrid MPI+OpenMP+CUDA matrix multiplication.
// Distributes rows across MPI ranks; each rank uses its GPU via CUDA
// with tiled shared-memory kernels. OpenMP used for host-side init.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // --- Determine row distribution per rank (contiguous blocks) ---
    size_t baseRows = N / numRanks;
    size_t rem = N % numRanks;

    // Helper: starting row for a given rank
    auto rankOffset = [&](int r) -> size_t {
        size_t rRem  = static_cast<size_t>(std::min(r, static_cast<int>(rem)));
        size_t rBase = static_cast<size_t>(std::max(0, r - static_cast<int>(rem)));
        return rRem * (baseRows + 1) + rBase * baseRows;
    };
    // Helper: number of rows for a given rank
    auto rankRows = [&](int r) -> size_t {
        return baseRows + (r < static_cast<int>(rem) ? 1 : 0);
    };

    size_t localRows = rankRows(rank);
    size_t offset    = rankOffset(rank);

    // --- Set CUDA device (round-robin across MPI ranks) ---
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: No CUDA-capable device found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    // --- Allocate device memory ---
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, localRows * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, localRows * N * sizeof(double)));

    // --- Copy data to GPU ---
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, A.data() + offset * N, localRows * N * sizeof(double),
                          cudaMemcpyHostToDevice));

    // --- Launch CUDA tiled matmul kernel ---
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (localRows + TILE_SIZE - 1) / TILE_SIZE);

    matmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, localRows, N, N);
    CUDA_CHECK(cudaGetLastError());

    // --- Copy result back from GPU ---
    CUDA_CHECK(cudaMemcpy(C.data() + offset * N, d_C, localRows * N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Clean up device memory ---
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // --- MPI gather: send all local results to rank 0 ---
    if (rank == 0) {
        for (int r = 1; r < numRanks; ++r) {
            size_t rRows   = rankRows(r);
            size_t rOffset = rankOffset(r);
            MPI_Recv(C.data() + rOffset * N, rRows * N, MPI_DOUBLE,
                     r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(C.data() + offset * N, localRows * N, MPI_DOUBLE,
                 0, 0, MPI_COMM_WORLD);
    }
}

// Validate result on a small set of positions
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
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
                printf("Validation failed at (%zu, %zu): expected %.10f, "
                       "got %.10f (error: %.10e)\n",
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    int validate = 0;
    int printResults = 0;

    // Rank 0 parses command-line args, then broadcasts
    int exitFlag = 0;  // 0=continue, 1=help/exited
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitFlag = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitFlag = 2;
            }
        }
    }

    // Broadcast exit flag so all ranks agree
    MPI_Bcast(&exitFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitFlag) {
        MPI_Finalize();
        return (exitFlag == 1) ? 0 : 1;
    }

    // Broadcast parsed arguments to all ranks
    uint64_t nVal = static_cast<uint64_t>(N);
    MPI_Bcast(&nVal, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(nVal);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Verify CUDA devices are available
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: No CUDA-capable device found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI+OpenMP+CUDA Hybrid)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA devices detected: %d\n", numDevices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate full matrices on each rank (deterministic init allows this)
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    if (rank == 0)
        printf("Initializing matrices (OpenMP parallel)...\n");

    // Parallel initialization using OpenMP
    initMatrix(A, N);
    initMatrix(B, N);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("Computing matrix multiplication (CUDA on GPU)...\n");

    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N);

    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS: 2*N^3 operations divided by time in seconds, converted to GFLOPS
        double elapsedSec = static_cast<double>(duration.count()) / 1000.0;
        double gflops = (2.0 * static_cast<double>(N) * N * N) / elapsedSec / 1.0e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation (rank 0 only)
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation (rank 0 only -- other ranks do not have full C)
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
