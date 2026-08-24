#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-based Cholesky decomposition with row distribution
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    // A is stored in row-major order
    // Each process owns a contiguous block of rows
    
    std::vector<double> pivot_row(n);
    
    for (size_t j = 0; j < n; ++j) {
        // Determine which process owns row j
        int owner = (j * size) / n;
        
        if (rank == owner) {
            // This process computes the diagonal element
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[j * n + k] * A[j * n + k];
            }
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                if (rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
                return false;
            }
            A[j * n + j] = sqrt(val);
            
            // Update the rest of row j (only lower triangular part needed)
            for (size_t k = 0; k < j; ++k) {
                pivot_row[k] = A[j * n + k];
            }
            pivot_row[j] = A[j * n + j];
        }
        
        // Broadcast row j to all processes
        MPI_Bcast(pivot_row.data(), j + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        // Each process updates its local rows i where i > j
        size_t row_start = (rank * n) / size;
        size_t row_end = ((rank + 1) * n) / size;
        
        for (size_t i = std::max(row_start, j + 1); i < row_end; ++i) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * n + k] * pivot_row[k];
            }
            A[i * n + j] = (A[i * n + j] - sum) / pivot_row[j];
        }
    }
    
    // Zero out upper triangular part on each process
    size_t row_start = (rank * n) / size;
    size_t row_end = ((rank + 1) * n) / size;
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix (only on rank 0, then broadcast)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int rank) {
    if (rank == 0) {
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
    
    // Broadcast matrix to all processes
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool validateCholesky(std::vector<double>& L, const std::vector<double>& A_orig, const size_t n, int rank, int size) {
    // Gather the full L matrix on rank 0 for validation
    if (rank == 0) {
        // Rank 0 receives data from all other processes
        for (int p = 1; p < size; ++p) {
            size_t row_start = (p * n) / size;
            size_t row_end = ((p + 1) * n) / size;
            size_t num_rows = row_end - row_start;
            if (num_rows > 0) {
                MPI_Recv(L.data() + row_start * n, num_rows * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Other ranks send their portions
        size_t row_start = (rank * n) / size;
        size_t row_end = ((rank + 1) * n) / size;
        size_t num_rows = row_end - row_start;
        if (num_rows > 0) {
            MPI_Send(L.data() + row_start * n, num_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    bool valid = true;
    if (rank == 0) {
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
            valid = false;
        }
    }
    
    // Broadcast validation result to all processes
    int valid_int = valid ? 1 : 0;
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    return valid_int == 1;
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
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, rank);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Synchronize before timing
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
    
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        // Gather full matrix on rank 0 for printing
        if (rank == 0) {
            for (int p = 1; p < size; ++p) {
                size_t row_start = (p * n) / size;
                size_t row_end = ((p + 1) * n) / size;
                size_t num_rows = row_end - row_start;
                if (num_rows > 0) {
                    MPI_Recv(A.data() + row_start * n, num_rows * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            print_results(A, "CholeskyL");
        } else {
            size_t row_start = (rank * n) / size;
            size_t row_end = ((rank + 1) * n) / size;
            size_t num_rows = row_end - row_start;
            if (num_rows > 0) {
                MPI_Send(A.data() + row_start * n, num_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
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
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
