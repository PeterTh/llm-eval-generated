#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallelized Cholesky decomposition for distributed memory systems
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// Uses row distribution: each rank owns contiguous row blocks

bool choleskyDecomposition(std::vector<double>& A, const size_t n, 
                           size_t row_start, size_t row_end,
                           int mpi_rank, int) {
    // A is stored in row-major order, contains only rows [row_start, row_end)
    // Each rank processes its assigned row range
    // The diagonal row is always computed and broadcast by its owner rank
    
    std::vector<double> diag_row(n);
    bool local_success = true;
    
    for (size_t j = 0; j < n; ++j) {
        // Check for previous errors
        int global_error = local_success ? 0 : 1;
        int prev_error = 0;
        MPI_Allreduce(&global_error, &prev_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (prev_error) {
            return false;
        }
        
        // Compute diagonal element L[j,j]
        if (j >= row_start && j < row_end) {
            // This rank owns row j
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[(j - row_start) * n + k] * A[(j - row_start) * n + k];
            }
            const double val = A[(j - row_start) * n + j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", j);
                local_success = false;
                std::fill(diag_row.begin(), diag_row.end(), 0.0);
            } else {
                A[(j - row_start) * n + j] = sqrt(val);
                
                // Store diagonal row in temp buffer for broadcast
                for (size_t k = 0; k < n; ++k) {
                    diag_row[k] = (k <= j) ? A[(j - row_start) * n + k] : 0.0;
                }
            }
        } else {
            // Initialize diag_row to zeros for non-owner ranks
            std::fill(diag_row.begin(), diag_row.end(), 0.0);
        }
        
        // Broadcast diagonal row j to all ranks
        int owner_rank = (j >= row_start && j < row_end) ? mpi_rank : -1;
        int global_owner;
        MPI_Allreduce(&owner_rank, &global_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Bcast(diag_row.data(), n, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
        
        // Compute off-diagonal elements in rows i > j (that belong to this rank)
        for (size_t i = j + 1; i < n; ++i) {
            if (i >= row_start && i < row_end) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += A[(i - row_start) * n + k] * diag_row[k];
                }
                A[(i - row_start) * n + j] = (A[(i - row_start) * n + j] - sum) / diag_row[j];
            }
        }
    }
    
    // Zero out upper triangular part on this rank
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[(i - row_start) * n + j] = 0.0;
        }
    }
    
    return local_success;
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

bool validateCholesky(const std::vector<double>& L, size_t row_start, size_t row_end,
                      const std::vector<double>& A_orig, const size_t n, 
                      int mpi_rank, int mpi_size) {
    // Each rank validates its local rows by reconstructing and comparing
    // Gather L*L^T computation results across all ranks
    
    std::vector<double> reconstructed(n * n);
    
    // Each rank computes L * L^T for its rows against all columns of L
    // First gather L on all ranks
    std::vector<double> L_global(n * n);
    std::vector<int> sendcounts(mpi_size), displs(mpi_size);
    
    for (int i = 0; i < mpi_size; ++i) {
        size_t start = (n * i) / mpi_size;
        size_t end = (n * (i + 1)) / mpi_size;
        sendcounts[i] = (end - start) * n;
        displs[i] = start * n;
    }
    
    MPI_Allgatherv((void*)L.data(), (row_end - row_start) * n, MPI_DOUBLE,
                   L_global.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    // Compute L * L^T (only on rank 0 for efficiency)
    if (mpi_rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += L_global[i * n + k] * L_global[j * n + k];
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
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse for simplicity)
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
        } else if (mpi_rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI-parallelized)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Number of MPI processes: %d\n", mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribute rows among ranks
    size_t row_start = (n * mpi_rank) / mpi_size;
    size_t row_end = (n * (mpi_rank + 1)) / mpi_size;
    size_t local_rows = row_end - row_start;
    
    // Allocate local matrix (each rank stores its row block)
    std::vector<double> A(local_rows * n);
    std::vector<double> A_orig;
    
    // Prepare sendcounts and displs for Scatterv
    std::vector<int> sendcounts(mpi_size), displs(mpi_size);
    for (int i = 0; i < mpi_size; ++i) {
        size_t start_i = (n * i) / mpi_size;
        size_t end_i = (n * (i + 1)) / mpi_size;
        sendcounts[i] = (end_i - start_i) * n;
        displs[i] = start_i * n;
    }
    
    if (mpi_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    
    // Generate matrix on rank 0, then distribute
    std::vector<double> A_full;
    if (mpi_rank == 0) {
        A_full.resize(n * n);
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full;
        }
    }
    
    // Scatter matrix rows to all ranks using Scatterv for variable sizes
    MPI_Scatterv(mpi_rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A.data(), local_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    // Perform Cholesky decomposition
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, row_start, row_end, mpi_rank, mpi_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Reduce success status across all ranks
    int local_success = success ? 1 : 0;
    int global_success;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!global_success) {
        if (mpi_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation (from rank 0 after gathering)
    if (printResults && mpi_rank == 0) {
        std::vector<double> A_result(n * n);
        std::vector<int> sendcounts(mpi_size), displs(mpi_size);
        for (int i = 0; i < mpi_size; ++i) {
            size_t start_i = (n * i) / mpi_size;
            size_t end_i = (n * (i + 1)) / mpi_size;
            sendcounts[i] = (end_i - start_i) * n;
            displs[i] = start_i * n;
        }
        MPI_Gatherv(A.data(), local_rows * n, MPI_DOUBLE,
                    A_result.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        print_results(A_result, "CholeskyL");
    } else if (printResults) {
        std::vector<int> sendcounts(mpi_size), displs(mpi_size);
        for (int i = 0; i < mpi_size; ++i) {
            size_t start_i = (n * i) / mpi_size;
            size_t end_i = (n * (i + 1)) / mpi_size;
            sendcounts[i] = (end_i - start_i) * n;
            displs[i] = start_i * n;
        }
        MPI_Gatherv(A.data(), local_rows * n, MPI_DOUBLE,
                    nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    
    // Validation
    if (validate) {
        if (mpi_rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, row_start, row_end, A_orig, n, mpi_rank, mpi_size);
        
        if (mpi_rank == 0) {
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
