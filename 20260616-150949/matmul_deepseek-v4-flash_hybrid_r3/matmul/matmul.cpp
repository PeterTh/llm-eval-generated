#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA tiled matrix multiplication kernel
// ---------------------------------------------------------------------------
#define TILE_SIZE 32

// Error checking helper for CUDA calls
#define CUDA_CHECK(call, rank)                                                \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n",            \
                    (rank), __FILE__, __LINE__, cudaGetErrorString(err));     \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                         \
        }                                                                     \
    } while (0)

/**
 * Tiled matrix multiplication kernel.
 * Computes  C[M][N] = A[M][K] * B[K][N]
 *
 * Each thread block handles a TILE_SIZE x TILE_SIZE output tile.
 * Shared memory tiles of A and B are reused across the K dimension.
 */
__global__ void matmulTiled(const double* __restrict__ A,
                            const double* __restrict__ B,
                            double* __restrict__ C,
                            size_t M,
                            size_t N,
                            size_t K) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    size_t numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (size_t t = 0; t < numTiles; ++t) {
        // Load tile of A
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;

        // Load tile of B
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] =
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        // Compute partial dot-product over the tile
        for (size_t k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// Host helper functions (unchanged semantics, OpenMP-accelerated)
// ---------------------------------------------------------------------------

constexpr double getPseudoRndValue(const size_t N,
                                   const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

bool validateResult(const std::vector<double>& A,
                    const std::vector<double>& B,
                    const std::vector<double>& C,
                    const size_t N) {
    // Check a few positions in parallel
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    int failCount = 0;

#pragma omp parallel for collapse(2) reduction(+ : failCount)
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
#pragma omp critical
                printf("Validation failed at (%zu, %zu): expected %.10f, "
                       "got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                ++failCount;
            }
        }
    }

    return failCount == 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main – hybrid MPI + OpenMP + CUDA
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Pin each MPI rank to a GPU (round-robin across available devices)
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: No CUDA-capable device found\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(world_rank % numDevices), world_rank);

    // -----------------------------------------------------------------------
    // Parse arguments on rank 0, then broadcast
    // -----------------------------------------------------------------------
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    if (world_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atol(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast configuration
    unsigned long long N_ull = static_cast<unsigned long long>(N);
    MPI_Bcast(&N_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N_ull);
    int validate_int = validate ? 1 : 0;
    int print_int = printResults ? 1 : 0;
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validate_int != 0;
    printResults = print_int != 0;

    // -----------------------------------------------------------------------
    // Work distribution: contiguous blocks of rows
    // -----------------------------------------------------------------------
    size_t baseRows = N / static_cast<size_t>(world_size);
    size_t remainder = N % static_cast<size_t>(world_size);
    size_t localRows =
        baseRows + (static_cast<size_t>(world_rank) < remainder ? 1 : 0);
    size_t localCount = localRows * N;

    // Build scatter / gather maps (element counts and displacements)
    std::vector<int> sendcounts(static_cast<size_t>(world_size));
    std::vector<int> displs(static_cast<size_t>(world_size));
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
        size_t rows = baseRows + (static_cast<size_t>(i) < remainder ? 1 : 0);
        sendcounts[static_cast<size_t>(i)] =
            static_cast<int>(rows * N);
        displs[static_cast<size_t>(i)] = offset;
        offset += static_cast<int>(rows * N);
    }

    // -----------------------------------------------------------------------
    // Allocate & initialise matrices
    // -----------------------------------------------------------------------
    std::vector<double> A_full;
    std::vector<double> B(N * N);
    std::vector<double> A_local(localCount);
    std::vector<double> C_local(localCount);

    if (world_rank == 0) {
        A_full.resize(N * N);
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", world_size);

#pragma omp parallel
#pragma omp single
        printf("OpenMP threads per rank: %d\n", omp_get_num_threads());

        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
        initMatrix(A_full, N);
        initMatrix(B, N);
    }

    // -----------------------------------------------------------------------
    // MPI data distribution
    // -----------------------------------------------------------------------
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    MPI_Scatterv(world_rank == 0 ? A_full.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), static_cast<int>(localCount), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // GPU memory allocation & H2D transfers
    // -----------------------------------------------------------------------
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, localCount * sizeof(double)), world_rank);
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)), world_rank);
        CUDA_CHECK(cudaMalloc(&d_C, localCount * sizeof(double)), world_rank);

        CUDA_CHECK(
            cudaMemcpy(d_A, A_local.data(), localCount * sizeof(double),
                       cudaMemcpyHostToDevice),
            world_rank);
        CUDA_CHECK(
            cudaMemcpy(d_B, B.data(), N * N * sizeof(double),
                       cudaMemcpyHostToDevice),
            world_rank);
    }

    // -----------------------------------------------------------------------
    // Timed GPU computation
    // -----------------------------------------------------------------------
    if (world_rank == 0)
        printf("Computing matrix multiplication...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (localRows > 0) {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim(static_cast<unsigned int>((N + TILE_SIZE - 1) / TILE_SIZE),
                     static_cast<unsigned int>(
                         (localRows + TILE_SIZE - 1) / TILE_SIZE));
        matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, localRows, N, N);
        CUDA_CHECK(cudaGetLastError(), world_rank);
    }
    CUDA_CHECK(cudaDeviceSynchronize(), world_rank);

    auto end = std::chrono::high_resolution_clock::now();

    // Reduce the maximum duration across all ranks for accurate timing
    long long dur_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    long long max_dur_ms;
    MPI_Reduce(&dur_ms, &max_dur_ms, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // D2H transfer
    // -----------------------------------------------------------------------
    if (localRows > 0) {
        CUDA_CHECK(
            cudaMemcpy(C_local.data(), d_C, localCount * sizeof(double),
                       cudaMemcpyDeviceToHost),
            world_rank);
    }

    // -----------------------------------------------------------------------
    // Gather results to rank 0
    // -----------------------------------------------------------------------
    std::vector<double> C;
    if (world_rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(localCount), MPI_DOUBLE,
                world_rank == 0 ? C.data() : nullptr, sendcounts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Output, validation, and cleanup (rank 0 only)
    // -----------------------------------------------------------------------
    if (world_rank == 0) {
        printf("Computation time: %lld ms\n", max_dur_ms);

        double gflops =
            (2.0 * static_cast<double>(N) * static_cast<double>(N) *
             static_cast<double>(N)) /
            (static_cast<double>(max_dur_ms) / 1000.0) / 1.0e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(A_full, B, C, N)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    if (localRows > 0) {
        CUDA_CHECK(cudaFree(d_A), world_rank);
        CUDA_CHECK(cudaFree(d_B), world_rank);
        CUDA_CHECK(cudaFree(d_C), world_rank);
    }

    MPI_Finalize();
    return 0;
}
