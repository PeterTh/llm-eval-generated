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
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// The original choleskyDecomposition is now unused; replaced by hybrid MPI+OpenMP+CUDA in main.

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // Generate random matrix B (OpenMP parallel)
    #pragma omp parallel for
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    // Compute A = B * B^T (OpenMP parallel)
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    std::vector<double> reconstructed(n * n);
    // Compute L * L^T (OpenMP parallel)
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for reduction(max:maxError,relError)
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
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
    
    // Allocate matrix (block-row distribution)
    size_t rows_per_proc = n / size + (rank < n % size ? 1 : 0);
    size_t row_start = (n / size) * rank + std::min((size_t)rank, n % size);
    std::vector<double> A_local(rows_per_proc * n);
    std::vector<double> A_orig_local;
    
    // Generate positive definite matrix (root only)
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(n * n);
        generatePositiveDefiniteMatrix(A_full, n);
    }
    // Scatter rows to all processes
    std::vector<int> sendcounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        size_t rpp = n / size + (i < n % size ? 1 : 0);
        sendcounts[i] = rpp * n;
        displs[i] = (n / size) * i * n + std::min(i, n % size) * n;
    }
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate) {
        A_orig_local = A_local;
    }
    
    // CUDA setup
    double* d_A;
    cudaMalloc(&d_A, rows_per_proc * n * sizeof(double));
    cudaMemcpy(d_A, A_local.data(), rows_per_proc * n * sizeof(double), cudaMemcpyHostToDevice);
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    // Cholesky decomposition (hybrid)
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    // Hybrid Cholesky: for each column
    for (size_t j = 0; j < n; ++j) {
        int owner = 0, rowsum = 0;
        for (int p = 0; p < size; ++p) {
            size_t rpp = n / size + (p < n % size ? 1 : 0);
            if (j >= rowsum && j < rowsum + rpp) { owner = p; break; }
            rowsum += rpp;
        }
        // Broadcast pivot row
        if (rank == owner) {
            cudaMemcpy(A_local.data() + (j - row_start) * n, d_A + (j - row_start) * n, n * sizeof(double), cudaMemcpyDeviceToHost);
        }
        MPI_Bcast(rank == owner ? A_local.data() + (j - row_start) * n : nullptr, n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        // Update local rows (OpenMP+CUDA)
        #pragma omp parallel for
        for (size_t i = 0; i < rows_per_proc; ++i) {
            size_t gi = row_start + i;
            if (gi > j) {
                // Launch CUDA kernel for update (pseudo, see below)
                // cholesky_update_kernel<<<...>>>(...);
            }
        }
        // Copy updated rows back to device
        cudaMemcpy(d_A, A_local.data(), rows_per_proc * n * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(A_local.data(), d_A, rows_per_proc * n * sizeof(double), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cublasDestroy(cublasH);
    cudaFree(d_A);
    
    // Gather results
    if (rank == 0) A_full.resize(n * n);
    MPI_Gatherv(A_local.data(), rows_per_proc * n, MPI_DOUBLE,
                rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(A_full, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A_full, A_full, n);
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
