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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const double* A, const double* B, double* C, size_t N, size_t row_offset) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y + row_offset;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) {
            sum += A[(row) * N + k] * B[k * N + col];
        }
        C[(row) * N + col] = sum;
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, size_t row_start, size_t row_count) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        size_t block_size = row_count * N * sizeof(double);
        cudaMalloc(&d_A, block_size);
        cudaMalloc(&d_B, N * N * sizeof(double));
        cudaMalloc(&d_C, block_size);
        cudaMemcpy(d_A, A.data(), block_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x, (row_count + threads.y - 1) / threads.y);
        matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N, row_start);
        cudaMemcpy(C.data() + row_start * N, d_C, block_size, cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < row_count; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[(row_start + i) * N + j] = sum;
            }
        }
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
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<double> A, B, C;
    if (world_rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);
    } else {
        B.resize(N * N);
    }
    // Each process gets a block of rows
    size_t rows_per_proc = N / world_size;
    size_t extra = N % world_size;
    size_t row_start = world_rank * rows_per_proc + (world_rank < extra ? world_rank : extra);
    size_t row_count = rows_per_proc + (world_rank < extra ? 1 : 0);
    std::vector<double> local_A(row_count * N);
    std::vector<double> local_C(row_count * N);

    // Initialize matrices
    if (world_rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    // Scatter A, broadcast B
    MPI_Scatterv(A.data(),
        std::vector<int>(world_size, rows_per_proc * N).data(),
        std::vector<int>(world_size, 0).data(),
        MPI_DOUBLE, local_A.data(), row_count * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (world_rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(local_A, B, local_C, N, 0, row_count);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather C
    MPI_Gatherv(local_C.data(), row_count * N, MPI_DOUBLE, C.data(),
        std::vector<int>(world_size, rows_per_proc * N).data(),
        std::vector<int>(world_size, 0).data(),
        MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    MPI_Finalize();
    return 0;
}
