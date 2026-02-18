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

// Hybrid MPI+OpenMP Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// All ranks maintain the full matrix. Each iteration, one rank computes column i,
// then broadcasts to all others. Within-rank parallelism uses OpenMP.

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int mpi_rank, int mpi_size) {
    // A is stored in row-major order
    // Process column by column
    
    for (size_t i = 0; i < n; ++i) {
        // Determine which rank owns column i (round-robin assignment)
        int owner = i % mpi_size;
        
        // Step 1: Compute diagonal element A[i,i] on owner rank
        if (mpi_rank == owner) {
            double sum = 0.0;
            for (size_t k = 0; k < i; ++k) {
                sum += A[i * n + k] * A[i * n + k];
            }
            double val = A[i * n + i] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
                A[i * n + i] = -1.0; // Signal error
            } else {
                A[i * n + i] = sqrt(val);
            }
        }
        
        // Broadcast the diagonal element
        MPI_Bcast(A.data() + i * n + i, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        // Check for error
        if (A[i * n + i] < 0.0) {
            return false;
        }
        
        // Step 2: Compute off-diagonal elements in column i
        // Only the owner rank computes, then broadcasts
        if (mpi_rank == owner) {
            #pragma omp parallel for schedule(static)
            for (size_t j = i + 1; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < i; ++k) {
                    sum += A[j * n + k] * A[i * n + k];
                }
                A[j * n + i] = (A[j * n + i] - sum) / A[i * n + i];
            }
        }
        
        // Broadcast entire column i (from i+1 to n) to all ranks
        for (size_t j = i + 1; j < n; ++j) {
            MPI_Bcast(A.data() + j * n + i, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        }
    }
    
    // Zero out upper triangular part (done in parallel)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix (distributed)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int mpi_rank) {
    // All ranks generate the same matrix with the same seed
    // Method: Create A = B * B^T where B is random
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;  // Same seed for all ranks
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    #pragma omp parallel for collapse(2) schedule(static)
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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
    
    // Synchronize matrix across all ranks (ensure consistency)
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n, int mpi_rank) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T (only rank 0 does this to save resources)
    if (mpi_rank == 0) {
        #pragma omp parallel for collapse(2) schedule(static)
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
    }
    
    // Broadcast result to all ranks
    bool result = (mpi_rank == 0);
    MPI_Bcast(&result, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    return result;
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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
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
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix (all ranks allocate full matrix)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (mpi_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, mpi_rank);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Synchronize before computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (mpi_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, mpi_rank, mpi_size);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Synchronize after computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (!success) {
        if (mpi_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation (only rank 0)
    if (printResults && mpi_rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation (only rank 0 prints results)
    if (validate) {
        if (mpi_rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, mpi_rank);
        
        if (mpi_rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
