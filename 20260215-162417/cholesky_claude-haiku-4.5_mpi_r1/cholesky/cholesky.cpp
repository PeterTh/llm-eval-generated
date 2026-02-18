#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition using row-block distribution
// Each MPI process owns a contiguous block of rows
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n, const size_t local_rows, 
                           const size_t first_row, int rank, int size) {
    // A is stored in row-major order, contains only local rows
    // local_rows: number of rows owned by this process
    // first_row: global index of first row owned by this process
    // n: global matrix dimension
    //
    // Sequential row-by-row parallel Cholesky with row gathering:
    // For each row i in [0, n):
    //   1. All processes gather row i (from previous iterations)
    //   2. Process owning row i computes new L[i,:i+1]
    //   3. Broadcast row i to all processes
    
    std::vector<std::vector<double>> L_all(n, std::vector<double>(n, 0.0));  // Store all computed rows
    
    for (size_t i = 0; i < n; ++i) {
        // Determine which process owns row i
        int owner_rank = (i * size) / n;
        
        std::vector<double> L_row(n, 0.0);
        
        if (first_row <= i && i < first_row + local_rows) {
            // This process owns row i
            size_t local_idx = i - first_row;
            
            // Compute L[i, j] for j in [0, i]
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                
                if (i == j) {
                    // Diagonal element: L[i,i] = sqrt(A[i,i] - sum(L[i,k]^2 for k < i))
                    for (size_t k = 0; k < i; ++k) {
                        sum += L_row[k] * L_row[k];
                    }
                    const double val = A[local_idx * n + i] - sum;
                    if (val <= 0.0) {
                        printf("Rank %d: Error: Matrix is not positive definite at diagonal element %zu\n", rank, i);
                        return false;
                    }
                    L_row[i] = sqrt(val);
                } else {
                    // Off-diagonal element: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k < j)) / L[j,j]
                    for (size_t k = 0; k < j; ++k) {
                        sum += L_row[k] * L_all[j][k];
                    }
                    L_row[j] = (A[local_idx * n + j] - sum) / L_all[j][j];
                }
            }
        }
        
        // Broadcast row i to all processes
        MPI_Bcast(L_row.data(), n, MPI_DOUBLE, owner_rank, MPI_COMM_WORLD);
        
        // Store the computed row
        L_all[i] = L_row;
    }
    
    // Copy the computed Cholesky factor back to the local matrix
    for (size_t i = first_row; i < first_row + local_rows; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[(i - first_row) * n + j] = L_all[i][j];
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix (root process only)
// Then scatter rows to all processes
void generatePositiveDefiniteMatrix(std::vector<double>& A_local, const size_t n, 
                                     const size_t local_rows, const size_t first_row,
                                     int rank, int size) {
    // Only rank 0 generates the full matrix
    std::vector<double> A_full;
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        // Generate full matrix on root
        A_full.resize(n * n);
        
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
                A_full[i * n + j] = sum;
            }
        }
        
        // Add diagonal dominance to ensure positive definiteness
        for (size_t i = 0; i < n; ++i) {
            A_full[i * n + i] += n;
        }
        
        // Calculate send counts and displacements for each process
        for (int p = 0; p < size; ++p) {
            size_t p_first_row = (p * n) / size;
            size_t p_local_rows = ((p + 1) * n) / size - p_first_row;
            sendcounts[p] = p_local_rows * n;
            displs[p] = p_first_row * n;
        }
    }
    
    // Broadcast sendcounts and displs to all processes (needed for scatterv)
    MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Scatter matrix rows to all processes
    MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool validateCholesky(const std::vector<double>& L_local, const std::vector<double>& A_orig_local,
                      const size_t n, const size_t local_rows, const size_t first_row,
                      int rank, int size) {
    // Gather all L values to rank 0 for validation
    std::vector<double> L_full;
    std::vector<double> A_orig_full;
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    for (int p = 0; p < size; ++p) {
        size_t p_first_row = (p * n) / size;
        size_t p_local_rows = ((p + 1) * n) / size - p_first_row;
        recvcounts[p] = p_local_rows * n;
        displs[p] = p_first_row * n;
    }
    
    if (rank == 0) {
        L_full.resize(n * n);
        A_orig_full.resize(n * n);
    }
    
    // Gather L matrix
    MPI_Gatherv((void*)L_local.data(), local_rows * n, MPI_DOUBLE,
                L_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Gather A_orig matrix
    MPI_Gatherv((void*)A_orig_local.data(), local_rows * n, MPI_DOUBLE,
                A_orig_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    bool valid = true;
    if (rank == 0) {
        // Validate by computing L * L^T and comparing with original matrix
        std::vector<double> reconstructed(n * n);
        
        // Compute L * L^T
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += L_full[i * n + k] * L_full[j * n + k];
                }
                reconstructed[i * n + j] = sum;
            }
        }
        
        // Compare with original
        double maxError = 0.0;
        double relError = 0.0;
        
        for (size_t i = 0; i < n * n; ++i) {
            const double error = fabs(reconstructed[i] - A_orig_full[i]);
            maxError = std::max(maxError, error);
            
            const double rel = error / (fabs(A_orig_full[i]) + 1e-10);
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
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    return valid;
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
    
    // Parse command line arguments (all processes parse, but only rank 0 prints)
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
    
    // Calculate row distribution
    size_t local_rows = (n + size - 1) / size;  // ceiling division
    size_t first_row = rank * local_rows;
    if (first_row >= n) first_row = n;
    if (first_row + local_rows > n) local_rows = n - first_row;
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI-Parallel)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Number of MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate local matrix
    std::vector<double> A_local(local_rows * n);
    std::vector<double> A_orig_local;
    
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    
    // Generate and distribute matrix
    generatePositiveDefiniteMatrix(A_local, n, local_rows, first_row, rank, size);
    
    if (validate) {
        A_orig_local = A_local; // Save original for validation
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    // Perform Cholesky decomposition
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A_local, n, local_rows, first_row, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Check for errors across all processes
    int local_error = success ? 0 : 1;
    int global_error = 0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    if (global_error != 0) {
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
    
    // Gather and print results for external validation
    if (printResults) {
        std::vector<double> L_full;
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        if (rank == 0) {
            L_full.resize(n * n);
            for (int p = 0; p < size; ++p) {
                size_t p_first_row = (p * n) / size;
                size_t p_local_rows = ((p + 1) * n) / size - p_first_row;
                recvcounts[p] = p_local_rows * n;
                displs[p] = p_first_row * n;
            }
        }
        
        MPI_Gatherv(A_local.data(), local_rows * n, MPI_DOUBLE,
                    L_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(L_full, "CholeskyL");
        }
    }
    
    // Validation
    bool valid = true;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        valid = validateCholesky(A_local, A_orig_local, n, local_rows, first_row, rank, size);
        
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    
    return valid ? 0 : 1;
}
