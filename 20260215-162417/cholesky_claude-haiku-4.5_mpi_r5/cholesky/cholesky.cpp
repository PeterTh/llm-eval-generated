#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI Cholesky decomposition using row-based distribution
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int num_procs) {
    // All processes have the full matrix in memory
    // Parallelize by distributing the computation of rows across processes
    
    for (size_t j = 0; j < n; ++j) {
        // Step 1: Compute diagonal element at (j, j) - only on rank 0
        if (rank == 0) {
            double sum = 0.0;
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
        
        // Broadcast row j from rank 0 to all processes
        MPI_Bcast(&A[j * n], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Step 2: Update rows below j - parallelize across processes
        size_t num_rows_to_update = n - j - 1;
        if (num_rows_to_update == 0) continue;
        
        size_t rows_per_rank = (num_rows_to_update + num_procs - 1) / num_procs;
        size_t start_idx = rank * rows_per_rank;
        size_t end_idx = std::min((rank + 1) * rows_per_rank, num_rows_to_update);
        
        // Compute local rows
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            size_t i = j + 1 + idx;
            
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
        
        // Synchronize: each rank sends its computed rows to all others via Allgather
        std::vector<double> send_buf(n * rows_per_rank, 0.0);
        std::vector<double> recv_buf(n * rows_per_rank * num_procs, 0.0);
        
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            size_t i = j + 1 + idx;
            for (size_t col = 0; col < n; ++col) {
                send_buf[(idx - start_idx) * n + col] = A[i * n + col];
            }
        }
        
        MPI_Allgather(send_buf.data(), n * rows_per_rank, MPI_DOUBLE,
                      recv_buf.data(), n * rows_per_rank, MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Copy received rows back into the matrix
        for (int src_rank = 0; src_rank < num_procs; ++src_rank) {
            size_t src_start_idx = src_rank * rows_per_rank;
            size_t src_end_idx = std::min((src_rank + 1) * rows_per_rank, num_rows_to_update);
            
            for (size_t idx = src_start_idx; idx < src_end_idx; ++idx) {
                size_t i = j + 1 + idx;
                for (size_t col = 0; col < n; ++col) {
                    A[i * n + col] = recv_buf[src_rank * n * rows_per_rank + (idx - src_start_idx) * n + col];
                }
            }
        }
    }
    
    // Zero out upper triangular part
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
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
        printf("Cholesky Decomposition Benchmark (MPI Parallel)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Number of MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // All processes allocate the full matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    if (rank == 0) {
        // Generate positive definite matrix
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        
        if (validate) {
            A_orig = A;
        }
    }
    
    // Broadcast matrix to all processes
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, num_procs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Check for errors across all processes
    int local_success = success ? 1 : 0;
    int global_success = 1;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    if (global_success == 0) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
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
    
    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation (rank 0 only)
    if (validate && rank == 0) {
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
    
    MPI_Finalize();
    return 0;
}
