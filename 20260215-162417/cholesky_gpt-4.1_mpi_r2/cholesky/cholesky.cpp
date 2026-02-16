#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition using MPI (row-wise block distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Block row distribution
    size_t rows_per_proc = n / size;
    size_t extra = n % size;
    size_t row_start = rank * rows_per_proc + (rank < extra ? rank : extra);
    size_t row_end = row_start + rows_per_proc + (rank < extra ? 1 : 0);

    for (size_t k = 0; k < n; ++k) {
        // Diagonal element computed by owner
        int owner = (k < (rows_per_proc + 1) * extra) ? k / (rows_per_proc + 1) : (k - extra) / rows_per_proc;
        double diag = 0.0;
        if (rank == owner) {
            double sum = 0.0;
            for (size_t s = 0; s < k; ++s)
                sum += A[k * n + s] * A[k * n + s];
            double val = A[k * n + k] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
                return false;
            }
            diag = sqrt(val);
            A[k * n + k] = diag;
        }
        // Broadcast diagonal
        MPI_Bcast(&A[k * n + k], 1, MPI_DOUBLE, owner, comm);
        diag = A[k * n + k];

        // Update column below diagonal
        for (size_t i = std::max(row_start, k + 1); i < row_end; ++i) {
            double sum = 0.0;
            for (size_t s = 0; s < k; ++s)
                sum += A[i * n + s] * A[k * n + s];
            A[i * n + k] = (A[i * n + k] - sum) / diag;
        }
        // Gather updated column
        MPI_Bcast(A.data() + (k + 1) * n + k, n - (k + 1), MPI_DOUBLE, owner, comm);
    }
    // Zero out upper triangle
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = i + 1; j < n; ++j)
            A[i * n + j] = 0.0;
    }
    // Gather all rows to all processes
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, A.data(), (row_end - row_start) * n, MPI_DOUBLE, comm);
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    // Generate positive definite matrix (only on root)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }
    // Broadcast matrix to all processes
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate && rank == 0) {
        A_orig = A; // Save original for validation
    }
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(A, n, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int global_success = success ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!global_success) {
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
                MPI_Finalize();
                return 0;
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
