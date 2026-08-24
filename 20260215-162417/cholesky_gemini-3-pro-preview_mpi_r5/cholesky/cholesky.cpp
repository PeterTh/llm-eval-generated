#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (row-cyclic distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    // A is stored in row-major order
    // In distributed mode, each process holds its local rows.
    // However, to keep it simple with existing function signature and minimize changes,
    // we will assume A is the FULL matrix on rank 0, and we will distribute it.
    // BUT the requirement is "distributed memory cluster parallelism", so we should
    // not store the full matrix on all nodes if possible, or at least operate on distributed data.
    
    // Given the function signature takes `std::vector<double>& A`, we'll assume:
    // - Rank 0 has the valid full matrix initially.
    // - We distribute rows cyclically.
    // - We gather the result back to Rank 0 at the end.
    
    // Allocate local rows
    size_t local_rows_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i % size == (size_t)rank) {
            local_rows_count++;
        }
    }
    
    std::vector<double> local_A(local_rows_count * n);
    
    // Scatter matrix rows (inefficient but simple way to adapt existing code structure)
    // In a real app, data would be initialized distributed.
    // Here we just copy from A to local_A based on ownership.
    
    if (rank == 0) {
        // Send rows to other processes
        for (size_t i = 0; i < n; ++i) {
            int target = i % size;
            if (target == 0) {
                // Copy to local
                size_t local_idx = i / size;
                std::memcpy(&local_A[local_idx * n], &A[i * n], n * sizeof(double));
            } else {
                MPI_Send(&A[i * n], n, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // Receive rows
        for (size_t i = 0; i < local_rows_count; ++i) {
            MPI_Recv(&local_A[i * n], n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Buffer to receive the current pivot row
    std::vector<double> pivot_row(n);

    for (size_t j = 0; j < n; ++j) {
        int root = j % size;
        
        if (rank == root) {
            // I own row j. It is my (j / size)-th local row.
            size_t local_idx = j / size;
            double* current_row = &local_A[local_idx * n];
            
            // 1. Compute diagonal element L[j][j]
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += current_row[k] * current_row[k];
            }
            double val = current_row[j] - sum;
            if (val <= 0.0) {
                printf("Rank %d: Error: Matrix is not positive definite at diagonal element %zu\n", rank, j);
                // Propagate failure
                current_row[j] = -1.0;
                // Prepare pivot row for broadcast with failure marker
                std::memcpy(pivot_row.data(), current_row, (j + 1) * sizeof(double));
            } else {
                current_row[j] = sqrt(val);
                // Prepare pivot row for broadcast
                std::memcpy(pivot_row.data(), current_row, (j + 1) * sizeof(double));
            }
        }
        
        // Broadcast the pivot row part 0..j
        MPI_Bcast(pivot_row.data(), j + 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        if (pivot_row[j] < 0) return false; // Failure propagated

        // 2. Update local rows i > j
        // Iterate over my local rows.
        // My first local row index depends on rank.
        // If rank <= j % size, my first local row > j is at index (j / size) + 1
        // If rank > j % size, my first local row > j is at index (j / size)
        
        // Find smallest i > j such that i % size == rank
        size_t next_i;
        if ((size_t)rank > j % size) {
            next_i = (j / size) * size + rank;
        } else {
            next_i = (j / size + 1) * size + rank;
        }
        
        for (size_t i = next_i; i < n; i += size) {
            size_t local_idx = i / size;
            double* my_row = &local_A[local_idx * n];
            
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += my_row[k] * pivot_row[k];
            }
            my_row[j] = (my_row[j] - sum) / pivot_row[j];
        }
    }
    
    // Zero out upper triangular part (in local rows)
    for (size_t local_idx = 0; local_idx < local_rows_count; ++local_idx) {
        size_t global_i = local_idx * size + rank;
        for (size_t j = global_i + 1; j < n; ++j) {
            local_A[local_idx * n + j] = 0.0;
        }
    }
    
    // Gather results back to Rank 0
    if (rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            int target = i % size;
            if (target == 0) {
                // Copy from local
                size_t local_idx = i / size;
                std::memcpy(&A[i * n], &local_A[local_idx * n], n * sizeof(double));
            } else {
                MPI_Recv(&A[i * n], n, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t i = 0; i < local_rows_count; ++i) {
            MPI_Send(&local_A[i * n], n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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
        printf("Cholesky Decomposition Benchmark (MPI Parallel)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    // Only Rank 0 needs the full matrix for generation and validation
    std::vector<double> A;
    std::vector<double> A_orig;
    
    if (rank == 0) {
        A.resize(n * n);
        
        // Generate positive definite matrix
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        
        if (validate) {
            A_orig = A; // Save original for validation
        }
    }
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, size);
    
    // Synchronize after timing (choleskyDecomposition already has implicit synchronization but barrier is safer for timing)
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
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
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
