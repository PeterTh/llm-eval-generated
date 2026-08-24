#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <mpi.h>

#include "../common/results_output.hpp"

// CUDA kernel declarations (if CUDA available)
#ifdef CUDA_AVAILABLE
extern "C" {
    void cuda_cholesky_kernel(double* d_A, size_t n, size_t start_row, size_t end_row);
    int cuda_init();
    int cuda_finalize();
}
#endif

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition
// Uses row-wise block distribution with MPI for distributed memory parallelism,
// OpenMP for shared memory parallelism within nodes, and potential CUDA for GPU acceleration
bool choleskyDecomposition(std::vector<double>& A, const size_t n, 
                           int mpi_rank, int mpi_size) {
    // A is stored in row-major order
    // Algorithm: Column-by-column Cholesky with row block distribution
    // Each MPI rank maintains complete rows but computes only its own rows for efficiency
    
    // Compute row range for this MPI rank
    size_t rows_per_rank = n / mpi_size;
    size_t start_row = mpi_rank * rows_per_rank;
    size_t end_row = (mpi_rank == mpi_size - 1) ? n : (mpi_rank + 1) * rows_per_rank;
    
    for (size_t j = 0; j < n; ++j) {
        // Step 1: Compute L[j,0..j-1] and L[j,j]
        // Only rank containing row j performs this
        if (j >= start_row && j < end_row) {
            double sum = 0.0;
            
            // Compute diagonal element with OpenMP
            #pragma omp parallel for reduction(+:sum)
            for (size_t k = 0; k < j; ++k) {
                sum += A[j * n + k] * A[j * n + k];
            }
            
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                printf("[Rank %d] Error: Matrix is not positive definite at diagonal element %zu\n", 
                       mpi_rank, j);
                return false;
            }
            A[j * n + j] = sqrt(val);
        }
        
        // Determine which rank owns row j
        size_t owner_rank = j / rows_per_rank;
        if (owner_rank >= (size_t)mpi_size) owner_rank = mpi_size - 1;
        
        // Broadcast entire row j from owner to all ranks
        MPI_Bcast(&A[j * n], n, MPI_DOUBLE, owner_rank, MPI_COMM_WORLD);
        
        // Step 2: Update rows i > j on all ranks using the new column j values
        // All ranks update their rows in parallel
        #pragma omp parallel for
        for (size_t i = j + 1; i < end_row; ++i) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
        
        // Ensure all ranks are synchronized before next column
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Zero out upper triangular part for rows this rank owns
    #pragma omp parallel for collapse(2)
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    // Gather all rows to all ranks (AllGather) so each rank has complete result
    std::vector<double> A_complete(n * n);
    std::vector<int> sendcounts(mpi_size);
    std::vector<int> displs(mpi_size);
    
    size_t rows_this_rank = end_row - start_row;
    for (int r = 0; r < mpi_size; ++r) {
        size_t r_rows = (r == mpi_size - 1) ? (n - r * rows_per_rank) : rows_per_rank;
        sendcounts[r] = r_rows * n;
        displs[r] = r * rows_per_rank * n;
    }
    
    MPI_Allgatherv(&A[start_row * n], rows_this_rank * n, MPI_DOUBLE,
                   A_complete.data(), sendcounts.data(), displs.data(), 
                   MPI_DOUBLE, MPI_COMM_WORLD);
    
    A = A_complete;
    
    return true;
}

// Generate a symmetric positive definite matrix (with MPI distribution)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, 
                                     int mpi_rank, int) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    // Only the root rank generates and then broadcasts
    
    if (mpi_rank == 0) {
        std::vector<double> B(n * n);
        unsigned int seed = 42;
        
        // Generate random matrix B
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        
        // Compute A = B * B^T with OpenMP acceleration
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
    
    // Broadcast matrix A from rank 0 to all ranks
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, 
                      const size_t n, int mpi_rank) {
    // Validate by computing L * L^T and comparing with original matrix
    // Only rank 0 performs validation
    
    if (mpi_rank != 0) {
        return true;
    }
    
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
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Only rank 0 prints progress
    bool print_progress = (mpi_rank == 0);
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0 && print_progress) {
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (print_progress && strcmp(argv[i], "-h") != 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (print_progress) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix (root generates and broadcasts)
    if (print_progress) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, mpi_rank, mpi_size);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition with synchronization across all ranks
    if (print_progress) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, mpi_rank, mpi_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Check success across all ranks
    int local_success = success ? 1 : 0;
    int global_success = 0;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    if (global_success == 0) {
        if (print_progress) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (print_progress) {
        printf("Computation time: %ld ms\n", global_duration_ms);
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation (only on rank 0)
    if (validate) {
        if (print_progress) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, mpi_rank);
        
        // Broadcast validation result
        int valid_int = valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (print_progress) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
