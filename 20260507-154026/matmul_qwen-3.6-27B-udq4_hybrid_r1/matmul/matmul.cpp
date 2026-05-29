#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA kernel – tiled GEMM with shared memory (row-major, C = A * B)
// ---------------------------------------------------------------------------
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const size_t N) {
    constexpr int BLOCK = 32;
    __shared__ double sA[BLOCK][BLOCK];
    __shared__ double sB[BLOCK][BLOCK];

    const int r = blockIdx.y * BLOCK + threadIdx.y;
    const int c = blockIdx.x * BLOCK + threadIdx.x;
    double sum = 0.0;

    const int numTiles = (N + BLOCK - 1) / BLOCK;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile into shared memory
        if (r < N && t * BLOCK + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[r * N + t * BLOCK + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        if (c < N && t * BLOCK + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * BLOCK + threadIdx.y) * N + c];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < BLOCK; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (r < N && c < N)
        C[r * N + c] = sum;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// OpenMP-parallelised initialisation
void initMatrix(double* mat, const size_t N) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, i, j);
}

// ---------------------------------------------------------------------------
// OpenMP-parallelised CPU GEMM (used by non-GPU ranks)
// ---------------------------------------------------------------------------
void matmulCPU(double* localC, const double* localA, const double* B,
               const size_t localRows, const size_t N) {
    std::memset(localC, 0, localRows * N * sizeof(double));

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < localRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += localA[i * N + k] * B[k * N + j];
            localC[i * N + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Hybrid matmul: MPI row-block distribution
//   • Rows of C are block-distributed across MPI ranks
//   • Rank 0 uses CUDA for its local GEMM (GPU)
//   • Other ranks use OpenMP for their local GEMM (CPU)
//   • Results are gathered back to rank 0
// ---------------------------------------------------------------------------
void matrixMultiply(double* A, const double* B, double* C, const size_t N,
                    const int mpi_rank, const int mpi_size) {
    // Distribute rows of C (and corresponding rows of A) across ranks
    const size_t totalRows = N;
    size_t localRows = totalRows / mpi_size;
    size_t remainder = totalRows % mpi_size;
    if (mpi_rank < static_cast<int>(remainder)) ++localRows;

    // Compute the starting row for this rank
    size_t myStartRow = 0;
    for (int r = 0; r < mpi_rank; ++r) {
        size_t rRows = totalRows / mpi_size;
        if (r < static_cast<int>(remainder)) ++rRows;
        myStartRow += rRows;
    }

    if (localRows == 0) return;  // rank has no work

    // Allocate local buffers
    const size_t localASize = localRows * N;
    const size_t localCSize = localRows * N;
    double* localA = new double[localASize];
    double* localC = new double[localCSize];

    // Copy my rows of A into localA
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows; ++i)
        std::memcpy(localA + i * N, A + (myStartRow + i) * N, N * sizeof(double));

    if (mpi_rank == 0) {
        // ---- Rank 0: CUDA GEMM ----
        cudaSetDevice(0);

        double *dA, *dB, *dC;
        cudaMalloc(&dA, localASize * sizeof(double));
        cudaMalloc(&dB, N * N * sizeof(double));
        cudaMalloc(&dC, localCSize * sizeof(double));

        cudaMemcpy(dA, localA, localASize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B,    N * N * sizeof(double),    cudaMemcpyHostToDevice);

        constexpr int BLOCK = 32;
        dim3 blockDim(BLOCK, BLOCK);
        dim3 gridDim((N + BLOCK - 1) / BLOCK, (localRows + BLOCK - 1) / BLOCK);

        matmulKernel<<<gridDim, blockDim>>>(dA, dB, dC, N);
        cudaDeviceSynchronize();

        cudaMemcpy(localC, dC, localCSize * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    } else {
        // ---- Other ranks: OpenMP CPU GEMM ----
        matmulCPU(localC, localA, B, localRows, N);
    }

    // Copy local result into the global C buffer
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows; ++i)
        std::memcpy(C + (myStartRow + i) * N, localC + i * N, N * sizeof(double));

    delete[] localA;
    delete[] localC;
}

// ---------------------------------------------------------------------------
// Validation (serial check on rank 0)
// ---------------------------------------------------------------------------
bool validateResult(const double* A, const double* B, const double* C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[i * N + k] * B[k * N + j];

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
    printf("Usage: mpirun %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    // ---- MPI init ----
    int mpi_size, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (mpi_rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks:   %d\n", mpi_size);
        printf("OMP threads: %d\n", omp_get_max_threads());
        printf("Validation:  %s\n", validate ? "enabled" : "disabled");
    }

    const size_t total = N * N;

    // ---- Allocate & initialise (all ranks hold full A, B, C) ----
    double* A = new double[total];
    double* B = new double[total];
    double* C = new double[total];
    std::memset(C, 0, total * sizeof(double));

    printf("Rank %d: Initializing matrices...\n", mpi_rank);
    initMatrix(A, N);
    initMatrix(B, N);
    fflush(stdout);

    // Synchronise before the timed region
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Timed matrix multiplication ----
    printf("Rank %d: Computing matrix multiplication...\n", mpi_rank);
    fflush(stdout);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(A, B, C, N, mpi_rank, mpi_size);
    auto end   = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double secs   = duration.count() / 1000.0;

    // Each rank only does its local share, but we report per-rank time.
    // The wall-clock time is the max across ranks.
    if (secs > 0.0) {
        // GFLOPS for the local work of this rank
        size_t localRows = N / mpi_size;
        size_t rem = N % mpi_size;
        if (mpi_rank < static_cast<int>(rem)) ++localRows;
        double localGFlops = (2.0 * localRows * N * N) / secs / 1e9;
        printf("Rank %d: Computation time: %ld ms  (%.3f GFLOPS local)\n",
               mpi_rank, duration.count(), localGFlops);
    } else {
        printf("Rank %d: Computation time: %ld ms\n", mpi_rank, duration.count());
    }
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Gather partial C from all ranks to rank 0 ----
    {
        // Re-compute localRows / myStartRow for the gather
        size_t localRows = N / mpi_size;
        size_t remainder = N % mpi_size;
        if (mpi_rank < static_cast<int>(remainder)) ++localRows;

        size_t myStartRow = 0;
        for (int r = 0; r < mpi_rank; ++r) {
            size_t rRows = N / mpi_size;
            if (r < static_cast<int>(remainder)) ++rRows;
            myStartRow += rRows;
        }

        // Build counts and displacements (in elements, not bytes)
        std::vector<int> counts(mpi_size);
        std::vector<int> displs(mpi_size);
        int offset = 0;
        for (int r = 0; r < mpi_size; ++r) {
            size_t rRows = N / mpi_size;
            if (r < static_cast<int>(remainder)) ++rRows;
            counts[r] = static_cast<int>(rRows * N);
            displs[r] = offset;
            offset += counts[r];
        }

        // Use temporary buffers to avoid any in-place gather issues
        int sendCount = static_cast<int>(localRows * N);
        double* sendbuf = new double[sendCount];
        std::memcpy(sendbuf, C + myStartRow * N, sendCount * sizeof(double));

        double* recvbuf = nullptr;
        if (mpi_rank == 0) recvbuf = new double[total];

        MPI_Gatherv(sendbuf, sendCount, MPI_DOUBLE,
                    recvbuf, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        delete[] sendbuf;

        if (mpi_rank == 0) {
            std::memcpy(C, recvbuf, total * sizeof(double));
            delete[] recvbuf;
        }
    }

    // ---- Print results / validate (rank 0 only) ----
    if (mpi_rank == 0) {
        if (printResults) {
            std::vector<double> Cvec(C, C + total);
            print_results(Cvec, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            delete[] A; delete[] B; delete[] C;
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;

    MPI_Finalize();
    return 0;
}
