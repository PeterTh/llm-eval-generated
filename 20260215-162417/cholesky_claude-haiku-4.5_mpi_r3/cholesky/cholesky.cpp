#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallelized Cholesky decomposition using 1D row distribution
// Each rank computes its rows, broadcasts them. All ranks use received data for computation.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int num_procs) {
    // Determine row range for this rank
    size_t rows_per_rank = n / num_procs;
    size_t row_start = rank * rows_per_rank;
    size_t row_end = (rank == num_procs - 1) ? n : row_start + rows_per_rank;
    
    // Global buffer to store all computed L rows (shared across all ranks via broadcasts)
    std::vector<double> L_global(n * n, 0.0);
    
    // Process each row sequentially
    for (size_t i = 0; i < n; ++i) {
        int owner_rank = i / rows_per_rank;
        if (owner_rank >= num_procs) owner_rank = num_procs - 1;
        
        // Owner of row i computes it
        if (rank == owner_rank && i >= row_start && i < row_end) {
            // Compute row i using previously computed rows
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                
                if (i == j) {
                    // Diagonal: A[i,i] = sqrt(A[i,i] - sum of squares)
                    for (size_t k = 0; k < j; ++k) {
                        sum += L_global[i * n + k] * L_global[i * n + k];
                    }
                    const double val = A[(i - row_start) * n + j] - sum;
                    if (val <= 0.0) {
                        fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", i);
                        return false;
                    }
                    A[(i - row_start) * n + j] = sqrt(val);
                } else {
                    // Off-diagonal: A[i,j] = (A[i,j] - sum) / A[j,j]
                    for (size_t k = 0; k < j; ++k) {
                        sum += L_global[i * n + k] * L_global[j * n + k];
                    }
                    A[(i - row_start) * n + j] = (A[(i - row_start) * n + j] - sum) / L_global[j * n + j];
                }
                // Store computed value in global buffer
                L_global[i * n + j] = A[(i - row_start) * n + j];
            }
        }
        
        // Broadcast row i from owner to all ranks
        MPI_Bcast(&L_global[i * n], i + 1, MPI_DOUBLE, owner_rank, MPI_COMM_WORLD);
    }
    
    // Copy computed rows back to local A for gather
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            A[(i - row_start) * n + j] = L_global[i * n + j];
        }
    }
    
    // Zero out upper triangular part
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[(i - row_start) * n + j] = 0.0;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only on rank 0)
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast matrix size to all ranks
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Number of MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Determine row range for this rank
    size_t rows_per_rank = n / num_procs;
    size_t row_start = rank * rows_per_rank;
    size_t row_end = (rank == num_procs - 1) ? n : row_start + rows_per_rank;
    size_t local_rows = row_end - row_start;
    
    // Allocate local row block (stores full rows, all columns)
    std::vector<double> A_local(local_rows * n);
    std::vector<double> A_orig;
    std::vector<double> A_global;
    
    if (rank == 0) {
        // Generate positive definite matrix on rank 0
        A_global.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_global, n);
        
        if (validate) {
            A_orig = A_global;
        }
    }
    
    // Scatter matrix rows to all ranks
    MPI_Scatter(rank == 0 ? A_global.data() : nullptr, local_rows * n, MPI_DOUBLE,
                A_local.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform MPI-parallel Cholesky decomposition
    bool success = choleskyDecomposition(A_local, n, rank, num_procs);
    
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
    
    // Gather result back to rank 0
    if (rank == 0) {
        A_global.clear();
        A_global.resize(n * n);
    }
    MPI_Gather(A_local.data(), local_rows * n, MPI_DOUBLE,
               rank == 0 ? A_global.data() : nullptr, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Print results for external validation (on rank 0 only)
    if (rank == 0 && printResults) {
        print_results(A_global, "CholeskyL");
    }
    
    // Validation (on rank 0 only)
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A_global, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    
    return (!validate || success) ? 0 : 1;
}
