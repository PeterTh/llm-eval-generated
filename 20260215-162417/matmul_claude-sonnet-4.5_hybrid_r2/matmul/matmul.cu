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
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const double* A, const double* B, double* C, 
                              size_t N, size_t row_start, size_t num_rows) {
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_row < num_rows && col < N) {
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) {
            sum += A[global_row * N + k] * B[k * N + col];
        }
        C[global_row * N + col] = sum;
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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Determine rows for this MPI process
    size_t rows_per_proc = N / size;
    size_t remainder = N % size;
    size_t row_start = rank * rows_per_proc + std::min((size_t)rank, remainder);
    size_t num_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, num_rows * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, num_rows * N * sizeof(double)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data() + row_start * N, num_rows * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel with 16x16 thread blocks
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (num_rows + blockDim.y - 1) / blockDim.y);
    
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, row_start, num_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C.data() + row_start * N, d_C, num_rows * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Gather results from all processes
    if (size > 1) {
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        for (int i = 0; i < size; ++i) {
            size_t start = i * rows_per_proc + std::min((size_t)i, remainder);
            size_t rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * N;
            displs[i] = start * N;
        }
        
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       C.data(), recvcounts.data(), displs.data(), 
                       MPI_DOUBLE, MPI_COMM_WORLD);
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set GPU device based on local rank
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus > 0) {
        CUDA_CHECK(cudaSetDevice(rank % num_gpus));
    }
    
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (rank 0 only for output)
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
        printf("Matrix Multiplication Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
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
            bool valid = validateResult(A, B, C, N);
            
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
    } else {
        // Non-root processes still need to participate in validation if enabled
        if (validate) {
            validateResult(A, B, C, N);
        }
    }
    
    MPI_Finalize();
    return 0;
}
