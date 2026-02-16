#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

void matrixMultiplyCUDA(const double* d_A, const double* d_B, double* d_C, size_t rows, size_t N, cublasHandle_t handle) {
    const double alpha = 1.0, beta = 0.0;
    // C = alpha * A * B + beta * C
    // d_A: rows x N, d_B: N x N, d_C: rows x N
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, rows, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, size_t row_start, size_t row_end) {
    // row_start inclusive, row_end exclusive
    #pragma omp parallel for collapse(2)
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[(i - row_start) * N + k] * B[k * N + j];
            }
            C[(i - row_start) * N + j] = sum;
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
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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

    // Block decomposition
    size_t rows_per_proc = N / world_size;
    size_t row_start = world_rank * rows_per_proc;
    size_t row_end = (world_rank == world_size - 1) ? N : row_start + rows_per_proc;
    size_t local_rows = row_end - row_start;

    // Allocate local matrices
    std::vector<double> local_A(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> local_C(local_rows * N);

    // Root initializes full A and B
    std::vector<double> A;
    if (world_rank == 0) {
        A.resize(N * N);
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }

    // Scatter A, broadcast B
    MPI_Scatter(A.data(), local_rows * N, MPI_DOUBLE, local_A.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // CUDA setup
    cudaSetDevice(0);
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, local_rows * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, local_rows * N * sizeof(double));
    cudaMemcpy(d_A, local_A.data(), local_rows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);

    if (world_rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // GPU matmul
    matrixMultiplyCUDA(d_A, d_B, d_C, local_rows, N, handle);
    cudaMemcpy(local_C.data(), d_C, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    // Gather C
    std::vector<double> C;
    if (world_rank == 0) C.resize(N * N);
    MPI_Gather(local_C.data(), local_rows * N, MPI_DOUBLE, C.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(C, "MatrixC");
        }
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
    }
    MPI_Finalize();
    return 0;
}
