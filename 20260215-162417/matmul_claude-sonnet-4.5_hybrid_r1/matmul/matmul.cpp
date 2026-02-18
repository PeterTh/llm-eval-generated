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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
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

// CUDA kernel for matrix multiplication
__global__ void matmulKernel(const double* A, const double* B, double* C, 
                             const size_t N, const size_t rowOffset) {
    const int TILE_SIZE = 32;
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory
        if (row < N && tile * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        
        if (tile * TILE_SIZE + ty < N && col < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, const size_t local_rows,
                    const size_t row_offset) {
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    size_t local_size = local_rows * N * sizeof(double);
    size_t full_size = N * N * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, local_size));
    CUDA_CHECK(cudaMalloc(&d_B, full_size));
    CUDA_CHECK(cudaMalloc(&d_C, local_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), local_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), full_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (local_rows + 31) / 32);
    
    matmulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, row_offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, local_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N, int rank) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;
    
    #pragma omp parallel for collapse(2) reduction(&& : valid)
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            if (!valid) continue;
            
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
                    if (rank == 0) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               i, j, expected, actual, relError);
                    }
                }
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set GPU device based on rank (for multi-GPU systems)
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(rank % device_count));
    
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", device_count);
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribute rows among MPI processes
    size_t rows_per_proc = N / size;
    size_t extra_rows = N % size;
    size_t local_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
    size_t row_offset = rank * rows_per_proc + std::min((size_t)rank, extra_rows);
    
    // Allocate matrices
    std::vector<double> A(N * N);  // Full matrix A (needed for validation)
    std::vector<double> B(N * N);  // Full matrix B (needed by all)
    std::vector<double> local_A(local_rows * N);  // Local rows of A
    std::vector<double> local_C(local_rows * N);  // Local rows of C
    std::vector<double> C(N * N);  // Full result (only on rank 0)
    
    // Initialize matrices (all processes initialize full matrices)
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Extract local rows of A
    #pragma omp parallel for
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            local_A[i * N + j] = A[(row_offset + i) * N + j];
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(local_A, B, local_C, N, local_rows, row_offset);
    
    // Gather results to rank 0
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    for (int r = 0; r < size; ++r) {
        size_t r_rows = rows_per_proc + (r < extra_rows ? 1 : 0);
        recvcounts[r] = r_rows * N;
        displs[r] = (r * rows_per_proc + std::min((size_t)r, extra_rows)) * N;
    }
    
    MPI_Gatherv(local_C.data(), local_rows * N, MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
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
            bool valid = validateResult(A, B, C, N, rank);
            
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
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
