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

// Tiled matrix multiplication kernel
// Assumes TILE_SIZE is defined.
#define TILE_SIZE 32

__global__ void matmul_kernel(const double* A, const double* B, double* C, 
                              int N, int rows_per_rank) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    
    // Iterate over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Shared memory for tiles
        __shared__ double As[TILE_SIZE][TILE_SIZE];
        __shared__ double Bs[TILE_SIZE][TILE_SIZE];

        // Load tile from A into shared memory
        // A is (rows_per_rank x N)
        // Check bounds for loading A
        if (row < rows_per_rank && (t * TILE_SIZE + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load tile from B into shared memory
        // B is (N x N)
        // Check bounds for loading B
        if ((t * TILE_SIZE + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Multiply tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < rows_per_rank && col < N) {
        C[row * N + col] = sum;
    }
}


// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Host function to coordinate GPU execution
void matrixMultiplyGPU(const std::vector<double>& local_A, const std::vector<double>& B, 
                       std::vector<double>& local_C, const size_t N, const size_t rows_per_rank) {
    double *d_A, *d_B, *d_C;

    size_t size_A = rows_per_rank * N * sizeof(double);
    size_t size_B = N * N * sizeof(double);
    size_t size_C = rows_per_rank * N * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, local_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (rows_per_rank + dimBlock.y - 1) / dimBlock.y);

    matmul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, rows_per_rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(local_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Use OpenMP for parallel validation
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    // Array to store results of checks, since we can't easily break from parallel loop or reduce a bool with early exit
    int failures = 0;
    
    #pragma omp parallel for reduction(+:failures)
    for (int k = 0; k < 25; ++k) {
        size_t pi = k / 5;
        size_t pj = k % 5;
        
        const size_t i = checkPoints[pi] % N;
        const size_t j = checkPoints[pj] % N;
        
        double expected = 0.0;
        for (size_t l = 0; l < N; ++l) {
            expected += A[i * N + l] * B[l * N + j];
        }
        
        const double actual = C[i * N + j];
        const double relError = std::abs((actual - expected) / (expected + 1e-10));
        
        if (relError > 1e-6) {
            #pragma omp critical
            {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
            }
            failures++;
        }
    }
    
    return failures == 0;
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Assign GPU based on rank (simple round-robin for multi-GPU nodes)
    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess || num_devices == 0) {
        // Fallback or error if no GPU? The task assumes accelerator cluster.
        // We will proceed but CUDA calls will fail if no device.
        // Let's assume there is at least 1 GPU if we are compiling for it.
        num_devices = 1; 
    }
    cudaSetDevice(rank % num_devices);

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
        }
    }
    
    // Broadcast N and other flags to ensure all ranks agree (though they parse same args)
    // It's good practice.
    // However, argv is usually consistent.
    
    if (N % size != 0) {
        if (rank == 0) printf("Error: N (%zu) must be divisible by MPI size (%d).\n", N, size);
        MPI_Finalize();
        return 1;
    }

    size_t rows_per_rank = N / size;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI Ranks: %d\n", size);
        printf("Rows per rank: %zu\n", rows_per_rank);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices
    // Rank 0 needs full A, B, C for initialization and validation
    std::vector<double> A, B, C;
    
    // All ranks allocate B (full size)
    B.resize(N * N);
    
    // Rank 0 allocates A and C
    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
        
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    
    // Local A chunk and C chunk
    std::vector<double> local_A(rows_per_rank * N);
    std::vector<double> local_C(rows_per_rank * N);

    // Distribute data
    // Broadcast B to all ranks
    // Using MPI_Bcast for large data. If N is large, count might overflow int.
    // 2^31 doubles is ~16GB.
    // Split into chunks if needed, but assuming N fits in int count for now.
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter A to local_A
    MPI_Scatter(A.data(), rows_per_rank * N, MPI_DOUBLE, 
                local_A.data(), rows_per_rank * N, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication on GPU
    matrixMultiplyGPU(local_A, B, local_C, N, rows_per_rank);

    // Synchronize before ending timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);
    
    // Gather results
    MPI_Gather(local_C.data(), rows_per_rank * N, MPI_DOUBLE,
               C.data(), rows_per_rank * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                // Don't abort here, just report
            }
        }
    }

    MPI_Finalize();
    return 0;
}
