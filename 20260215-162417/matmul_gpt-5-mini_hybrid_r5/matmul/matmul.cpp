#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>

#if USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t rowOffset) {
    const size_t rows = mat.size() / N;
    #pragma omp parallel for schedule(static)
    for (size_t ii = 0; ii < rows; ++ii) {
        for (size_t j = 0; j < N; ++j) {
            mat[ii * N + j] = getPseudoRndValue(N, ii + rowOffset, j);
        }
    }
}

#if USE_CUDA
// CUDA kernel: simple tiled multiplication
constexpr int TILE_DIM = 16;

__global__ void matMulKernel(const double* A, const double* B, double* C, size_t N, size_t Arows) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    double value = 0.0;
    for (int t = 0; t < (int)((N + TILE_DIM - 1) / TILE_DIM); ++t) {
        __shared__ double As[TILE_DIM][TILE_DIM];
        __shared__ double Bs[TILE_DIM][TILE_DIM];

        int aRow = row;
        int aCol = t * TILE_DIM + threadIdx.x;
        int bRow = t * TILE_DIM + threadIdx.y;
        int bCol = col;

        if (aRow < (int)Arows && aCol < (int)N)
            As[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (bRow < (int)N && bCol < (int)N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < (int)Arows && col < (int)N) {
        C[row * N + col] = value;
    }
}

void gpuMatrixMultiply(const std::vector<double>& A_local, const std::vector<double>& B, std::vector<double>& C_local, const size_t N) {
    const size_t Arows = C_local.size() / N;
    const size_t sizeA = Arows * N * sizeof(double);
    const size_t sizeB = N * N * sizeof(double);
    const size_t sizeC = Arows * N * sizeof(double);

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, A_local.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (Arows + TILE_DIM - 1) / TILE_DIM);

    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N, Arows);
    cudaDeviceSynchronize();

    cudaMemcpy(C_local.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#else
// CPU fallback using OpenMP parallel blocking
void gpuMatrixMultiply(const std::vector<double>& A_local, const std::vector<double>& B, std::vector<double>& C_local, const size_t N) {
    const size_t Arows = C_local.size() / N;
    const size_t BLOCK = 32;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t ii = 0; ii < Arows; ii += BLOCK) {
        for (size_t jj = 0; jj < N; jj += BLOCK) {
            for (size_t kk = 0; kk < N; kk += BLOCK) {
                size_t i_max = std::min(ii + BLOCK, Arows);
                size_t j_max = std::min(jj + BLOCK, N);
                size_t k_max = std::min(kk + BLOCK, N);
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t j = jj; j < j_max; ++j) {
                        double sum = 0.0;
                        for (size_t k = kk; k < k_max; ++k) {
                            sum += A_local[i * N + k] * B[k * N + j];
                        }
                        // accumulate into C_local
                        #pragma omp atomic
                        C_local[i * N + j] += sum;
                    }
                }
            }
        }
    }
}
#endif

// Simple validation: compute expected for a few points
bool validateResultDistributed(const std::vector<double>& A_full, const std::vector<double>& B_full,
                               const std::vector<double>& C_full, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) expected += A_full[i * N + k] * B_full[k * N + j];

            const double actual = C_full[i * N + j];
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse args only on rank 0 then broadcast
    if (world_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            }
        }
    }
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int v = validate ? 1 : 0;
    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (v != 0);
    int pr = printResults ? 1 : 0;
    MPI_Bcast(&pr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printResults = (pr != 0);

    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    // Determine rows per rank
    const size_t baseRows = N / world_size;
    const size_t remainder = N % world_size;
    const size_t local_rows = baseRows + (static_cast<size_t>(world_rank) < remainder ? 1 : 0);
    const size_t rowOffset = baseRows * world_rank + std::min(static_cast<size_t>(world_rank), remainder);

    // Allocate local matrices
    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N, 0.0);

    // Initialize local A and B on rank 0 then broadcast B
    initMatrixBlock(A_local, N, rowOffset);

    if (world_rank == 0) {
        std::vector<double> B_full(N * N);
        initMatrixBlock(B_full, N, 0);
        // copy into B for root
        std::copy(B_full.begin(), B_full.end(), B.begin());
    }
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Warm up and compute
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Compute on device or CPU fallback (per rank)
    gpuMatrixMultiply(A_local, B, C_local, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Get max time across ranks
    long long max_time = 0;
    MPI_Reduce(&duration_ms, &max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time (max across ranks): %lld ms\n", max_time);
        double gflops = (2.0 * N * N * N) / (static_cast<double>(max_time) / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather results to rank 0 for validation or printing
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        size_t rows_r = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recvcounts[r] = static_cast<int>(rows_r * N);
        displs[r] = static_cast<int>((baseRows * r + std::min(static_cast<size_t>(r), remainder)) * N);
    }

    std::vector<double> C_full;
    if (world_rank == 0) C_full.assign(N * N, 0.0);

    MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                world_rank == 0 ? C_full.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (printResults && world_rank == 0) {
        print_results(C_full, "MatrixC");
    }

    if (validate && world_rank == 0) {
        // For validation, need full A too
        std::vector<double> A_full(N * N);
        // Re-create full A on root
        initMatrixBlock(A_full, N, 0);
        bool valid = validateResultDistributed(A_full, B, C_full, N);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
