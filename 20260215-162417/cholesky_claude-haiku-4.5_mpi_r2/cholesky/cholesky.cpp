#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition using MPI
// All processes maintain full matrix; computations are distributed per row
bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    // Perform Cholesky decomposition - process each row in order
    for (size_t i = 0; i < n; ++i) {
        // Determine which process owns row i
        int owner = 0;
        int row_offset = 0;
        for (int p = 0; p < size; ++p) {
            int rows_p = (n / size) + (p < n % size ? 1 : 0);
            if ((int)i < row_offset + rows_p) {
                owner = p;
                break;
            }
            row_offset += rows_p;
        }
        
        std::vector<double> row_i(n, 0.0);
        
        // Only the owner process computes row i (columns 0 to i)
        if (rank == owner) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                
                if (i == j) {
                    // Diagonal element
                    for (size_t k = 0; k < j; ++k) {
                        sum += A[j * n + k] * A[j * n + k];
                    }
                    const double val = A[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        return false;
                    }
                    A[j * n + j] = sqrt(val);
                } else {
                    // Off-diagonal element
                    for (size_t k = 0; k < j; ++k) {
                        sum += A[i * n + k] * A[j * n + k];
                    }
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                }
                row_i[j] = A[i * n + j];
            }
            
            // Zero out upper triangular part of row i
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
                row_i[j] = 0.0;
            }
        }
        
        // Broadcast row i to all processes
        MPI_Bcast(row_i.data(), n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        // Update A on non-owner processes with received row
        if (rank != owner) {
            for (size_t j = 0; j < n; ++j) {
                A[i * n + j] = row_i[j];
            }
        }
        
        // Synchronize before next iteration
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    return true;
}

// Generate a symmetric positive definite matrix (broadcast to all processes)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int rank, int size) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    if (rank == 0) {
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
    
    // Broadcast full matrix to all processes
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n, int rank, int size) {
    // All processes have the full matrix, so validation is local
    
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
    
    // Only rank 0 prints validation results
    if (rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
        
        // Check if error is within tolerance
        if (relError > 1e-6) {
            printf("Validation failed: relative error too large\n");
            return false;
        }
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
        } else if (rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
        }
    }
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI Parallel)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Number of processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // All processes maintain full matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix (full matrix on all processes)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, rank, size);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Synchronize before starting timer
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, size);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation (on rank 0)
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, rank, size);
        
        if (rank == 0) {
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
