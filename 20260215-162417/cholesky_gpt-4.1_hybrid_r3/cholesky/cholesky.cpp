#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_CUDA
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif
#endif

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
#ifdef USE_MPI
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    // Blocked Cholesky for hybrid parallelism
    size_t block_size = 64; // Tune for hardware
    for (size_t k = 0; k < n; k += block_size) {
        size_t kb = std::min(block_size, n - k);
        // 1. Factor diagonal block (single process, offload to GPU if available)
        bool error_flag = false;
#ifdef USE_MPI
        if ((k / block_size) % mpi_size == mpi_rank) {
#endif
#ifdef USE_CUDA
        // CUDA kernel for diagonal block (not shown, placeholder)
        // cholesky_cuda_kernel(&A[k * n + k], n, kb);
#endif
        for (size_t i = 0; i < kb; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                for (size_t l = 0; l < j; ++l) sum += A[(k+i)*n + (k+l)] * A[(k+j)*n + (k+l)];
                if (i == j) {
                    double val = A[(k+j)*n + (k+j)] - sum;
                    if (val <= 0.0) { error_flag = true; break; }
                    A[(k+j)*n + (k+j)] = sqrt(val);
                } else {
                    A[(k+i)*n + (k+j)] = (A[(k+i)*n + (k+j)] - sum) / A[(k+j)*n + (k+j)];
                }
            }
        }
#ifdef USE_MPI
        }
        // Broadcast diagonal block to all processes
        MPI_Bcast(&A[k * n + k], kb * n, MPI_DOUBLE, (k / block_size) % mpi_size, MPI_COMM_WORLD);
#endif
        // 2. Update trailing submatrix (parallel across processes and threads, offload to GPU if available)
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = k + kb; i < n; ++i) {
            for (size_t j = k; j < k + kb; ++j) {
                double sum = 0.0;
                for (size_t l = k; l < j; ++l) sum += A[i*n + l] * A[j*n + l];
                A[i*n + j] = (A[i*n + j] - sum) / A[j*n + j];
            }
        }
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = k + kb; i < n; ++i) {
            for (size_t j = k + kb; j <= i; ++j) {
                double sum = 0.0;
                for (size_t l = k; l < k + kb; ++l) sum += A[i*n + l] * A[j*n + l];
                A[i*n + j] -= sum;
            }
        }
    }
    // Zero out upper triangle
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) { for (size_t j = i + 1; j < n; ++j) A[i*n + j] = 0.0; }
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
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 1;
        }
    }
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
#endif
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
#ifdef USE_MPI
    }
#endif
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    // Generate positive definite matrix
    if (
#ifdef USE_MPI
        mpi_rank == 0
#else
        true
#endif
    ) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }
#ifdef USE_MPI
    MPI_Bcast(A.data(), n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    if (validate) {
        A_orig = A; // Save original for validation
    }
    // Perform Cholesky decomposition
    if (
#ifdef USE_MPI
        mpi_rank == 0
#else
        true
#endif
    ) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(A, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int global_success = success;
#ifdef USE_MPI
    MPI_Allreduce(&success, &global_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
#endif
    if (!global_success) {
        if (
#ifdef USE_MPI
            mpi_rank == 0
#else
            true
#endif
        ) printf("Cholesky decomposition failed\n");
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    if (
#ifdef USE_MPI
        mpi_rank == 0
#else
        true
#endif
    ) {
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
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 0;
            } else {
                printf("Validation: FAILED\n");
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 1;
            }
        }
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
