#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                    const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

constexpr int TILE_SIZE = 32;

// CUDA tiled matrix multiplication kernel: C[M][N] = A[M][K] * B[K][N]
template <int TS>
__global__ void matmulKernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C, int M, int N, int K) {
    __shared__ double As[TS][TS];
    __shared__ double Bs[TS][TS];

    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;

    double sum = 0.0;
    int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A
        if (row < M && t * TS + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] =
                A[row * K + t * TS + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        // Load tile of B
        if (col < N && t * TS + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] =
                B[(t * TS + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TS; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Initialize local rows of a matrix with deterministic pseudo-random values
void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t rowStart, const size_t rowEnd) {
#pragma omp parallel for collapse(2)
    for (size_t i = rowStart; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowStart) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Initialize a full NxN matrix
void initMatrixFull(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Perform local matrix multiplication on GPU
void gpuMatmul(const double* A_local, const double* B, double* C_local,
               const size_t local_rows, const size_t N) {
    if (local_rows == 0) return;

    double *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, local_rows * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, local_rows * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A_local, local_rows * N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * N * sizeof(double),
                          cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (local_rows + TILE_SIZE - 1) / TILE_SIZE);

    matmulKernel<TILE_SIZE><<<gridDim, blockDim>>>(
        d_A, d_B, d_C, static_cast<int>(local_rows), static_cast<int>(N),
        static_cast<int>(N));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C_local, d_C, local_rows * N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Simple validation: compute reference elements and compare
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
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got "
                       "%.10f (error: %.10e)\n",
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Set CUDA device in round-robin fashion across available GPUs
    int numGpus = 0;
    cudaGetDeviceCount(&numGpus);
    if (numGpus == 0) {
        fprintf(stderr, "No CUDA-capable device found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % numGpus));

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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

    // Compute row distribution across MPI ranks
    const size_t base = N / num_ranks;
    const size_t rem = N % num_ranks;
    const size_t local_start =
        static_cast<size_t>(rank) * base + (static_cast<size_t>(rank) < rem ? static_cast<size_t>(rank) : rem);
    const size_t local_count =
        base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // Allocate host memory
    std::vector<double> A_local(local_count * N);
    std::vector<double> B_full(N * N);
    std::vector<double> C_local(local_count * N);

    // Rank 0 builds full A for validation
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(N * N);
    }

    // Initialize matrices in parallel using OpenMP.
    // Each rank independently initializes its local rows of A and all of B.
    // Since getPseudoRndValue is deterministic, all ranks produce identical B.
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(A_local, N, local_start, local_start + local_count);
    if (rank == 0) {
        initMatrixFull(A_full, N);
    }
    initMatrixFull(B_full, N);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", num_ranks);
        printf("CUDA devices available: %d\n", numGpus);
        printf("Computing matrix multiplication...\n");
    }

    // Synchronize all ranks before timing the parallel computation
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Each rank computes its portion of C on GPU
    gpuMatmul(A_local.data(), B_full.data(), C_local.data(), local_count, N);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Prepare MPI_Gatherv parameters for collecting C blocks
    std::vector<int> recv_counts(num_ranks);
    std::vector<int> displs(num_ranks);
    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < num_ranks; ++r) {
            size_t rc = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            recv_counts[r] = static_cast<int>(rc * N);
            displs[r] = offset;
            offset += recv_counts[r];
        }
    }

    std::vector<double> C_full;
    if (rank == 0) {
        C_full.resize(N * N);
    }

    // Gather all local C blocks to rank 0
    MPI_Gatherv(C_local.data(), static_cast<int>(local_count * N), MPI_DOUBLE,
                C_full.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    // Rank 0 reports timing and handles results
    if (rank == 0) {
        double duration = end_time - start_time;
        long duration_ms = static_cast<long>(duration * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        // GFLOPS = (2 * N^3 operations) / time / 1e9
        double gflops = (2.0 * N * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A_full, B_full, C_full, N);

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
