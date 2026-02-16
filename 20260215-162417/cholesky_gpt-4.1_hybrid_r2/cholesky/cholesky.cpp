#include <algorithm>
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

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// CUDA kernel for sum reduction (off-diagonal and diagonal)
__global__ void sumReductionKernel(const double* A, double* result, size_t n, size_t i, size_t j, bool diagonal) {
    extern __shared__ double sdata[];
    size_t tid = threadIdx.x;
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    if (k < j) {
        if (diagonal)
            sum = A[j * n + k] * A[j * n + k];
        else
            sum = A[i * n + k] * A[j * n + k];
    }
    sdata[tid] = sum;
    __syncthreads();
    // Parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) result[blockIdx.x] = sdata[0];
}

// Host function for sum reduction using CUDA
static double cudaSumReduction(const std::vector<double>& A, size_t n, size_t i, size_t j, bool diagonal) {
    double* d_A;
    double* d_result;
    double result = 0.0;
    size_t numThreads = 256;
    size_t numBlocks = (j + numThreads - 1) / numThreads;
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_result, numBlocks * sizeof(double));
    sumReductionKernel<<<numBlocks, numThreads, numThreads * sizeof(double)>>>(d_A, d_result, n, i, j, diagonal);
    std::vector<double> h_result(numBlocks);
    cudaMemcpy(h_result.data(), d_result, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t b = 0; b < numBlocks; ++b) result += h_result[b];
    cudaFree(d_A);
    cudaFree(d_result);
    return result;
}

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t rows_per_proc = n / size;
    size_t row_start = rank * rows_per_proc;
    size_t row_end = (rank == size - 1) ? n : row_start + rows_per_proc;
    for (size_t i = 0; i < n; ++i) {
        // Only one rank computes the diagonal and broadcasts
        if (i >= row_start && i < row_end) {
            #pragma omp parallel for schedule(static)
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                if (i == j) {
                    // Diagonal element (use CUDA)
                    sum = cudaSumReduction(A, n, i, j, true);
                    const double val = A[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                        return false;
                    }
                    A[j * n + j] = sqrt(val);
                } else {
                    // Off-diagonal element (use CUDA)
                    sum = cudaSumReduction(A, n, i, j, false);
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                }
            }
        }
        // Broadcast updated row to all processes
        MPI_Bcast(&A[i * n], n, MPI_DOUBLE, (i / rows_per_proc), MPI_COMM_WORLD);
        // Zero out upper triangular part (parallelized)
        if (i >= row_start && i < row_end) {
            #pragma omp parallel for schedule(static)
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
            }
        }
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        if (validate) {
            A_orig = A; // Save original for validation
        }
    }
    // Broadcast matrix to all processes
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate && rank != 0) {
        A_orig = A;
    }
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(A, n);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(A, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
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
