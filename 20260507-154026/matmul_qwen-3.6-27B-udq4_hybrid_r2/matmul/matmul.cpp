#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Pseudo-random matrix initialiser (unchanged semantics)
// ---------------------------------------------------------------------------
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// ---------------------------------------------------------------------------
// CUDA kernel – tiled GEMM with shared memory
//   C[M][Nc] += A[M][K] * B[K][Nc]
// ---------------------------------------------------------------------------
__global__ void matmulKernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M, int Nc, int K)
{
    constexpr int TILE = 32;
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;
    const int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0;

        int bRow = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (col < Nc && bRow < K) ? B[bRow * Nc + col] : 0.0;

        __syncthreads();

        for (int n = 0; n < TILE; ++n)
            sum += sA[threadIdx.y][n] * sB[n][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < Nc)
        C[row * Nc + col] = sum;
}

// ---------------------------------------------------------------------------
// OpenMP-parallelised block initialisation
// ---------------------------------------------------------------------------
static void initBlock(double* block, const size_t N, const int rowOffset,
                      const int localRows, const int cols)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            block[i * cols + j] =
                getPseudoRndValue(N, static_cast<size_t>(rowOffset + i),
                                  static_cast<size_t>(j));
        }
    }
}

// ---------------------------------------------------------------------------
// Validation (unchanged semantics)
// ---------------------------------------------------------------------------
static bool validateResult(const std::vector<double>& A,
                           const std::vector<double>& B,
                           const std::vector<double>& C, const size_t N)
{
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[i * N + k] * B[k * N + j];

            const double actual   = C[i * N + j];
            const double relError = std::abs((actual - expected) /
                                             (expected + 1e-10));

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

// ---------------------------------------------------------------------------
// main – Hybrid MPI (row-block distribution) + OpenMP + CUDA
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    /* ---- argument parsing (rank 0 only) ---- */
    size_t N = 512;
    bool validate     = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                N = static_cast<size_t>(atoi(argv[++i]));
            else if (strcmp(argv[i], "-v") == 0)
                validate = true;
            else if (strcmp(argv[i], "-r") == 0)
                printResults = true;
            else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Matrix size N (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results\n");
                printf("  -h           Show help\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    /* broadcast parameters to every rank */
    MPI_Bcast(&N,            1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_C_BOOL,        0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL,        0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    /* ---- 1-D row-block distribution across MPI ranks ---- */
    const int baseRows  = static_cast<int>(N) / numRanks;
    const int remainder = static_cast<int>(N) % numRanks;
    const int localRows = baseRows + (rank < remainder ? 1 : 0);
    const int myRowStart = rank * baseRows + std::min(rank, remainder);

    const size_t localSize = static_cast<size_t>(localRows) * N;
    const size_t fullSize  = static_cast<size_t>(N) * N;

    /* ---- allocate host buffers ---- */
    std::vector<double> A_local(localSize);
    std::vector<double> B_full(fullSize);
    std::vector<double> C_local(localSize, 0.0);

    /* ---- initialise (OpenMP) ---- */
    if (rank == 0) printf("Initializing matrices...\n");

    initBlock(A_local.data(), N, myRowStart, localRows, N);
    if (rank == 0)
        initBlock(B_full.data(), N, 0, N, N);

    /* broadcast B (needed by every rank for the row-block scheme) */
    MPI_Bcast(B_full.data(), fullSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ---- CUDA: select GPU, allocate device memory ---- */
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    cudaSetDevice((numGPUs > 0) ? (rank % numGPUs) : 0);

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, localSize * sizeof(double));
    cudaMalloc(&d_B, fullSize  * sizeof(double));
    cudaMalloc(&d_C, localSize * sizeof(double));

    cudaMemcpy(d_A, A_local.data(), localSize * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_full.data(),  fullSize  * sizeof(double),
               cudaMemcpyHostToDevice);

    /* ---- GPU computation ---- */
    if (rank == 0) printf("Computing matrix multiplication...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    constexpr int TILE = 32;
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE,
                (localRows + TILE - 1) / TILE);

    matmulKernel<<<blocks, threads>>>(d_A, d_B, d_C, localRows, N, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C_local.data(), d_C, localSize * sizeof(double),
               cudaMemcpyDeviceToHost);

    const double t1 = MPI_Wtime();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /* ---- gather C to rank 0 ---- */
    std::vector<double> C_full;
    std::vector<int> recvCounts(numRanks), displs(numRanks);
    int disp = 0;
    for (int r = 0; r < numRanks; ++r) {
        const int rows = baseRows + (r < remainder ? 1 : 0);
        recvCounts[r] = rows * N;
        displs[r]     = disp;
        disp += recvCounts[r];
    }
    if (rank == 0) C_full.resize(fullSize);

    MPI_Gatherv(C_local.data(), static_cast<int>(localSize), MPI_DOUBLE,
                C_full.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* ---- gather A to rank 0 (only when validation is requested) ---- */
    std::vector<double> A_full;
    if (validate) {
        if (rank == 0) A_full.resize(fullSize);
        MPI_Gatherv(A_local.data(), static_cast<int>(localSize), MPI_DOUBLE,
                    A_full.data(), recvCounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    /* ---- rank 0: report, print, validate ---- */
    int exitCode = 0;
    if (rank == 0) {
        const double tSecs = t1 - t0;
        const double gflops = (2.0 * N * N * N) / tSecs / 1e9;

        printf("Computation time: %ld ms\n",
               static_cast<long>(tSecs * 1000.0));
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults)
            print_results(C_full, "MatrixC");

        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResult(A_full, B_full, C_full, N);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            exitCode = ok ? 0 : 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
