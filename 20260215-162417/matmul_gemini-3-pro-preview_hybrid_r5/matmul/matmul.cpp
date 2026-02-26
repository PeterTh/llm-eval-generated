#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define TILE_SIZE 32

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// CUDA Kernel for Matrix Multiplication (Tiled)
__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C, int N, int local_rows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;

    // Shared memory for tiles
    __shared__ double sA[TILE_SIZE][TILE_SIZE];
    __shared__ double sB[TILE_SIZE][TILE_SIZE];

    for (int k = 0; k < N; k += TILE_SIZE) {
        // Load tile from A
        if (row < local_rows && (k + threadIdx.x) < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + k + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        // Load tile from B
        if ((k + threadIdx.y) < N && col < N)
            sB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Multiply tiles
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < local_rows && col < N) {
        C[row * N + col] = sum;
    }
}

void initMatrixLocal(double* mat, const size_t N, const size_t start_row, const size_t num_rows) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }
}

void initMatrixFull(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    // We can't reduce on a bool easily in all OpenMP versions, using int for safety
    int valid_flag = 1;

    #pragma omp parallel for reduction(&:valid_flag)
    for (size_t idx = 0; idx < 25; ++idx) {
        size_t pi = idx / 5;
        size_t pj = idx % 5;
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
            valid_flag = 0;
        }
    }
    
    return valid_flag;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }
    
    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI Ranks: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine local workload
    size_t rows_per_rank = N / world_size;
    size_t remainder = N % world_size;
    size_t local_rows = rows_per_rank + (world_rank < (int)remainder ? 1 : 0);
    size_t start_row = 0;
    
    // Calculate start row
    if (world_rank < (int)remainder) {
        start_row = world_rank * (rows_per_rank + 1);
    } else {
        start_row = remainder * (rows_per_rank + 1) + (world_rank - remainder) * rows_per_rank;
    }

    // Allocate host memory
    // local_A holds the rows of A assigned to this rank
    std::vector<double> local_A(local_rows * N);
    std::vector<double> B(N * N); // Full B needed on all ranks
    std::vector<double> local_C(local_rows * N);
    
    // Initialize matrices
    // OpenMP is used inside these functions
    initMatrixLocal(local_A.data(), N, start_row, local_rows);
    initMatrixFull(B, N); // Every rank initializes full B locally to avoid communication

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, local_rows * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, local_rows * N * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, local_A.data(), local_rows * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

    // Configure kernel launch
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (local_rows + TILE_SIZE - 1) / TILE_SIZE);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, local_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(local_C.data(), d_C, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Gather results if validation or printing is requested
    std::vector<double> global_C;
    std::vector<double> global_A; 
    
    if (validate || printResults) {
        if (world_rank == 0) {
            global_C.resize(N * N);
            global_A.resize(N * N); // Needed for validation
        }

        // Gather C
        std::vector<int> recvcounts(world_size);
        std::vector<int> displs(world_size);

        int local_count = local_rows * N;
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < world_size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }

        MPI_Gatherv(local_C.data(), local_count, MPI_DOUBLE,
                    global_C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Regenerate A on Rank 0 is cleaner and saves bandwidth
        if (world_rank == 0) {
            initMatrixFull(global_A, N);
        }
    }

    if (world_rank == 0) {
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        if (printResults) {
            print_results(global_C, "MatrixC");
        }
        
        if (validate) {
            printf("Validating result...\n");
            // Need global_A for validation
            initMatrixFull(global_A, N);
            bool valid = validateResult(global_A, B, global_C, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
