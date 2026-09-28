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
        const cudaError_t err_ = (call);                                        \
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

// Initialize rows [rowBegin, rowEnd) of an NxN matrix; OpenMP-parallel on the host
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
#pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

constexpr int TILE = 32;

// Tiled shared-memory kernel: computes C = A * B for a block of rows of A/C.
// A is localRows x N, B is N x N, C is localRows x N.
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t localRows, const size_t N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE + 1];

    const size_t row = blockIdx.y * TILE + threadIdx.y;
    const size_t col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;
    const size_t numTiles = (N + TILE - 1) / TILE;

    for (size_t t = 0; t < numTiles; ++t) {
        const size_t aCol = t * TILE + threadIdx.x;
        const size_t bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < localRows && aCol < N) ? A[row * N + aCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < localRows && col < N) {
        C[row * N + col] = sum;
    }
}

// Multiply the local row block of A by the full B on this rank's GPU
void matrixMultiplyLocal(std::vector<double>& C_local, const size_t N, const size_t localRows,
                         const double* dA, const double* dB, double* dC) {
    if (localRows == 0) return;

    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);
    matmulKernel<<<grid, block>>>(dA, dB, dC, localRows, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C_local.data(), dC, localRows * N * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    bool valid = true;
#pragma omp parallel for collapse(2) schedule(static)
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
                valid = false;
            }
        }
    }

    return valid;
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    // Select a GPU based on the rank's position on its node (round-robin)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per node: %d, OpenMP threads: %d\n",
               numRanks, deviceCount, omp_get_max_threads());
    }

    // Row-block decomposition of C (and A) across ranks
    const size_t rowBegin = (N * static_cast<size_t>(rank)) / numRanks;
    const size_t rowEnd = (N * static_cast<size_t>(rank + 1)) / numRanks;
    const size_t localRows = rowEnd - rowBegin;

    // Allocate matrices: each rank holds its row block of A/C and the full B
    std::vector<double> A_local(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(localRows * N);
    std::vector<double> C;

    // Initialize matrices (deterministic formula, so no communication needed)
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(A_local.data(), N, rowBegin, rowEnd);
    initMatrixRows(B.data(), N, 0, N);

    // Stage local inputs on the GPU before timing (setup, like initialization)
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&dA, localRows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, localRows * N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dA, A_local.data(), localRows * N * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Gather layout for assembling C on rank 0
    std::vector<int> counts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t rb = (N * static_cast<size_t>(r)) / numRanks;
        const size_t re = (N * static_cast<size_t>(r + 1)) / numRanks;
        counts[r] = static_cast<int>((re - rb) * N);
        displs[r] = static_cast<int>(rb * N);
    }
    if (rank == 0) {
        C.resize(N * N);
    }

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(C_local, N, localRows, dA, dB, dC);
    MPI_Gatherv(C_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (dA) CUDA_CHECK(cudaFree(dA));
    if (dB) CUDA_CHECK(cudaFree(dB));
    if (dC) CUDA_CHECK(cudaFree(dC));

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

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
            std::vector<double> A(N * N);
            initMatrixRows(A.data(), N, 0, N);
            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
