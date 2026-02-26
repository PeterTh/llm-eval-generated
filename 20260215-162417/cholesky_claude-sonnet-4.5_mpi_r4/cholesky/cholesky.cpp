#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallelized Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// Uses row-wise distribution with broadcast of pivot columns

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    // A is stored in row-major order
    // Each process owns rows: rank, rank+size, rank+2*size, ...
    
    std::vector<double> pivot_col(n);  // Buffer for broadcasting pivot column
    
    for (size_t j = 0; j < n; ++j) {
        int owner = j % size;  // Process that owns row j
        
        // Step 1: Compute diagonal element (owner process)
        if (rank == owner) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[j * n + k] * A[j * n + k];
            }
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return false;
            }
            A[j * n + j] = sqrt(val);
            
            // Copy column j values (0 to j) to broadcast buffer
            for (size_t k = 0; k <= j; ++k) {
                pivot_col[k] = A[j * n + k];
            }
        }
        
        // Step 2: Broadcast column j (0 to j) to all processes
        MPI_Bcast(pivot_col.data(), j + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        // Update local copy if not owner
        if (rank != owner) {
            for (size_t k = 0; k <= j; ++k) {
                A[j * n + k] = pivot_col[k];
            }
        }
        
        // Step 3: Update rows i > j owned by this process
        for (size_t i = j + 1; i < n; ++i) {
            if (i % size == (size_t)rank) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * pivot_col[k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / pivot_col[j];
            }
        }
    }
    
    // Zero out upper triangular part for owned rows
    for (size_t i = 0; i < n; ++i) {
        if (i % size == (size_t)rank) {
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
            }
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix (all processes generate same matrix)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B (same on all processes)
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
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix (all processes need full matrix for row-cyclic distribution)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix (same on all processes)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Barrier to sync before timing
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
    
    // Gather all rows to rank 0 for validation and output
    // Each process sends its owned rows
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(n * n);
    }
    
    for (size_t i = 0; i < n; ++i) {
        int owner = i % size;
        if (rank == owner) {
            if (rank == 0) {
                // Copy to output buffer
                for (size_t j = 0; j < n; ++j) {
                    A_full[i * n + j] = A[i * n + j];
                }
            } else {
                // Send row to rank 0
                MPI_Send(&A[i * n], n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
            }
        } else if (rank == 0) {
            // Receive row from owner
            MPI_Recv(&A_full[i * n], n, MPI_DOUBLE, owner, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(A_full, "CholeskyL");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A_full, A_orig, n);
            
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
