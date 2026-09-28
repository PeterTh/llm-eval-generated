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
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a block of rows [rowBegin, rowBegin + numRows) of an NxN matrix,
// stored at mat (numRows * N elements), using OpenMP across rows.
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t numRows) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < numRows; ++r) {
        const size_t i = rowBegin + r;
        for (size_t j = 0; j < N; ++j) {
            mat[r * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Tiled shared-memory matrix multiply kernel with 2x2 register blocking.
// Computes C[rows x N] = A[rows x N] * B[N x N] for a block of rows of C.
constexpr int TILE = 32;
constexpr int BLK = 16;  // thread block is BLK x BLK, each thread computes 2x2 outputs

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t N, const size_t rows) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE + 1];

    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y * 2;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x * 2;

    double c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;

    const size_t numTiles = (N + TILE - 1) / TILE;
    for (size_t t = 0; t < numTiles; ++t) {
        const size_t kBase = t * TILE;
        // Each thread loads a 2x2 patch of the A and B tiles
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const size_t ar = rowBase + dy;
                const size_t ak = kBase + threadIdx.x * 2 + dx;
                As[threadIdx.y * 2 + dy][threadIdx.x * 2 + dx] =
                    (ar < rows && ak < N) ? A[ar * N + ak] : 0.0;
                const size_t bk = kBase + threadIdx.y * 2 + dy;
                const size_t bc = colBase + dx;
                Bs[threadIdx.y * 2 + dy][threadIdx.x * 2 + dx] =
                    (bk < N && bc < N) ? B[bk * N + bc] : 0.0;
            }
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double a0 = As[threadIdx.y * 2][k];
            const double a1 = As[threadIdx.y * 2 + 1][k];
            const double b0 = Bs[k][threadIdx.x * 2];
            const double b1 = Bs[k][threadIdx.x * 2 + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    if (rowBase < rows && colBase < N) C[rowBase * N + colBase] = c00;
    if (rowBase < rows && colBase + 1 < N) C[rowBase * N + colBase + 1] = c01;
    if (rowBase + 1 < rows && colBase < N) C[(rowBase + 1) * N + colBase] = c10;
    if (rowBase + 1 < rows && colBase + 1 < N) C[(rowBase + 1) * N + colBase + 1] = c11;
}

// Multiply this rank's block of rows of A with B on the GPU.
void matrixMultiplyLocal(const double* A_local, const double* B, double* C_local,
                         const size_t N, const size_t localRows) {
    if (localRows == 0) return;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, localRows * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, localRows * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dA, A_local, localRows * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice));

    const dim3 block(BLK, BLK);
    const dim3 grid((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);
    matmulKernel<<<grid, block>>>(dA, dB, dC, N, localRows);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C_local, dC, localRows * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
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
#pragma omp critical
                {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                    valid = false;
                }
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
    MPI_Init(&argc, &argv);

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

    // Bind each rank to a GPU based on its rank within the node
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
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n",
               nranks, omp_get_max_threads(), deviceCount);
    }

    // Row-block decomposition of C (and A) across ranks
    std::vector<int> rowCounts(nranks), rowStarts(nranks);
    {
        const size_t base = N / nranks;
        const size_t rem = N % nranks;
        size_t start = 0;
        for (int r = 0; r < nranks; ++r) {
            rowCounts[r] = static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
            rowStarts[r] = static_cast<int>(start);
            start += rowCounts[r];
        }
    }
    const size_t localRows = rowCounts[rank];
    const size_t rowOffset = rowStarts[rank];

    // Allocate matrices: rank 0 keeps full A and C (for validation/output),
    // other ranks only their row block of A and C. B is replicated.
    std::vector<double> A(rank == 0 ? N * N : localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rank == 0 ? N * N : localRows * N);
    double* A_local = rank == 0 ? A.data() + rowOffset * N : A.data();
    double* C_local = rank == 0 ? C.data() + rowOffset * N : C.data();

    // Initialize matrices (deterministic formula, so no communication needed)
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrixRows(A.data(), N, 0, N);
    } else {
        initMatrixRows(A_local, N, rowOffset, localRows);
    }
    initMatrixRows(B.data(), N, 0, N);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A_local, B.data(), C_local, N, localRows);

    // Gather row blocks of C to rank 0
    std::vector<int> recvCounts(nranks), recvDispls(nranks);
    for (int r = 0; r < nranks; ++r) {
        recvCounts[r] = rowCounts[r] * static_cast<int>(N);
        recvDispls[r] = rowStarts[r] * static_cast<int>(N);
    }
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : C_local, static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalDuration);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (globalDuration / 1000.0) / 1e9;
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
