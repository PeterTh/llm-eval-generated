#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t rowsPerRank = (n + size - 1) / size;
    const size_t startRow = (size_t)rank * rowsPerRank;
    const size_t endRow = std::min(startRow + rowsPerRank, n);

    bool ok = true;
    std::vector<double> col(n, 0.0);

    for (size_t k = 0; k < n; ++k) {
        // Diagonal element (computed by all ranks identically)
        double sum = 0.0;
        for (size_t t = 0; t < k; ++t) {
            sum += A[k * n + t] * A[k * n + t];
        }
        const double val = A[k * n + k] - sum;
        if (val <= 0.0) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            ok = false;
            break;
        }
        A[k * n + k] = sqrt(val);

        // Off-diagonal rows owned by this rank
        for (size_t i = startRow; i < endRow; ++i) {
            if (i <= k) continue;
            double s = 0.0;
            for (size_t t = 0; t < k; ++t) {
                s += A[i * n + t] * A[k * n + t];
            }
            A[i * n + k] = (A[i * n + k] - s) / A[k * n + k];
        }

        // Gather column k values across ranks
        for (size_t i = 0; i < n; ++i) col[i] = 0.0;
        for (size_t i = startRow; i < endRow; ++i) {
            if (i > k) col[i] = A[i * n + k];
        }
        // include diagonal? already on all ranks; ensure col[k] set
        if (k >= startRow && k < endRow) col[k] = A[k * n + k];
        MPI_Allreduce(MPI_IN_PLACE, col.data(), (int)n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t i = 0; i < n; ++i) {
            if (i > k) A[i * n + k] = col[i];
        }
        A[k * n + k] = col[k];
    }

    // Zero upper triangular for owned rows
    for (size_t i = startRow; i < endRow; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    // Zero rows not owned by this rank
    for (size_t i = 0; i < n; ++i) {
        if (i >= startRow && i < endRow) continue;
        for (size_t j = 0; j < n; ++j) A[i * n + j] = 0.0;
    }

    // Sync full matrix (all rows) across ranks
    MPI_Allreduce(MPI_IN_PLACE, A.data(), (int)(n * n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int okInt = ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &okInt, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return okInt == 1;
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
    std::atexit([]() { MPI_Finalize(); });
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long max_duration_ms = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &max_duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    if (rank == 0) printf("Computation time: %ld ms\n", max_duration_ms);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
