#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition with hybrid MPI+OpenMP parallelization
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define BLOCK_SIZE 64

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    // Hybrid MPI+OpenMP Cholesky decomposition
    // All ranks maintain full matrix for simplicity
    // Work distribution: column j is primarily computed by rank (j % size)
    
    for (size_t j = 0; j < n; ++j) {
        int j_owner = j % size;
        
        // Phase 1: Owner rank computes diagonal element (with OpenMP parallelization)
        if (rank == j_owner) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (size_t k = 0; k < j; ++k) {
                sum += A[j * n + k] * A[j * n + k];
            }
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            A[j * n + j] = sqrt(val);
        }
        
        // Broadcast diagonal element to all ranks
        MPI_Bcast(&A[j * n + j], 1, MPI_DOUBLE, j_owner, MPI_COMM_WORLD);
        
        // Phase 2: All ranks compute elements below diagonal for column j
        #pragma omp parallel for
        for (size_t i = j + 1; i < n; ++i) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
        
        // Synchronize all ranks before next column
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Zero out upper triangular part
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix using CUDA for parallelization
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B with OpenMP parallelization
    #pragma omp parallel
    {
        unsigned int local_seed = seed + omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&local_seed) / (double)RAND_MAX) - 0.5;
        }
    }
    
    // Compute A = B * B^T with OpenMP parallelization
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
    // Parallelized with OpenMP
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T with OpenMP
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
    // Initialize MPI
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
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix (all ranks allocate for now, but only use their portion)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    if (validate) {
        A_orig.resize(n * n);
    }
    
    // Generate positive definite matrix (only on rank 0 for correctness)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        
        if (validate) {
            A_orig = A;
        }
    }
    
    // Broadcast matrix to all ranks
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate) {
        MPI_Bcast(A_orig.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        if (!success) {
            printf("Cholesky decomposition failed\n");
            MPI_Finalize();
            return 1;
        }
        
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
            
            if (valid) {
                printf("Validation: PASSED\n");
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
