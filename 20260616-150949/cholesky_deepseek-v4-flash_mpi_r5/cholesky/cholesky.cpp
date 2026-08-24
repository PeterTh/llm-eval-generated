#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition (right-looking, cyclic column distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Right-looking variant: after computing column j of L, updates the trailing
// submatrix A[j+1:n][j+1:n] -= L[j+1:n][j] * L[j+1:n][j]^T.
// This allows all processes to independently update their owned columns
// after each broadcast, providing good parallel scalability.
//
// Columns are distributed in a cyclic manner: process p owns column j iff
// j % world_size == p. Each process updates only its owned columns.

bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                            const int world_rank, const int world_size) {
    // Temporary buffer for packing/transmitting columns contiguously
    std::vector<double> col_buf(n);
    int fail_flag = 0;

    // Right-looking Cholesky: iterate over each column
    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % world_size);

        // ----- Step 1: Owner computes column j of L -----
        if (world_rank == owner) {
            if (A[j * n + j] <= 0.0) {
                if (world_rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                }
                fail_flag = 1;
            }

            if (!fail_flag) {
                // Diagonal: L[j][j] = sqrt(A[j][j])
                A[j * n + j] = sqrt(A[j * n + j]);
                const double diag = A[j * n + j];

                // Off-diagonal: L[i][j] = A[i][j] / L[j][j] for i > j
                for (size_t i = j + 1; i < n; ++i) {
                    A[i * n + j] /= diag;
                }

                // Zero out upper triangular part of column j
                for (size_t i = 0; i < j; ++i) {
                    A[i * n + j] = 0.0;
                }
            }
        }

        // Check for failure across all processes
        int any_fail;
        MPI_Allreduce(&fail_flag, &any_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (any_fail) {
            return false;
        }

        // ----- Step 2: Broadcast column j from owner to all -----
        // Pack column j into contiguous buffer for efficient broadcast
        if (world_rank == owner) {
            for (size_t i = 0; i < n; ++i) {
                col_buf[i] = A[i * n + j];
            }
        }
        MPI_Bcast(col_buf.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        // Unpack buffer into column j (all processes, including owner, for consistency)
        for (size_t i = 0; i < n; ++i) {
            A[i * n + j] = col_buf[i];
        }

        // ----- Step 3: Update trailing submatrix -----
        // Each process independently updates its own columns k > j:
        //   A[i][k] -= L[i][j] * L[k][j]   for all i >= k > j
        for (size_t k = j + 1; k < n; ++k) {
            if (static_cast<int>(k % world_size) == world_rank) {
                for (size_t i = k; i < n; ++i) {
                    A[i * n + k] -= A[i * n + j] * A[k * n + j];
                }
            }
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

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all processes parse, but only rank 0 prints errors)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (world_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix on all processes
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix on rank 0 only
    if (world_rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }
    
    // Broadcast the full matrix from rank 0 to all processes
    MPI_Bcast(A.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (validate && world_rank == 0) {
        A_orig = A; // Save original for validation
    }
    
    // Perform parallel Cholesky decomposition
    if (world_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    const double t_start = MPI_Wtime();
    bool success = choleskyDecomposition(A, n, world_rank, world_size);
    const double t_end = MPI_Wtime();
    
    const long local_duration_ms = static_cast<long>((t_end - t_start) * 1000.0);
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (world_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = ops / (static_cast<double>(duration_ms) / 1000.0) / 1.0e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults && world_rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    int valid_flag = 1;
    if (validate) {
        if (world_rank == 0) {
            printf("Validating result...\n");
            valid_flag = validateCholesky(A, A_orig, n) ? 1 : 0;
        }
        MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            if (valid_flag) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return valid_flag ? 0 : 1;
}
