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

size_t rowStart(size_t n, int rank, int ranks);

bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           const size_t firstRow, const size_t localRows,
                           int rank, int ranks) {
    std::vector<double> panel(n);
    bool success = true;
    for (size_t k = 0; k < n; ++k) {
        int owner = 0;
        while (owner + 1 < ranks && k >= rowStart(n, owner + 1, ranks)) ++owner;
        if (rank == owner) {
            const size_t row = k - firstRow;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += A[row * n + j] * A[row * n + j];
            const double diagonal = A[row * n + k] - sum;
            if (diagonal <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
                success = false;
            } else {
                A[row * n + k] = std::sqrt(diagonal);
                for (size_t j = 0; j <= k; ++j) panel[j] = A[row * n + j];
            }
        }
        MPI_Bcast(&success, 1, MPI_C_BOOL, owner, MPI_COMM_WORLD);
        if (!success) return false;
        MPI_Bcast(panel.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        for (size_t local = 0; local < localRows; ++local) {
            const size_t i = firstRow + local;
            if (i <= k) continue;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += A[local * n + j] * panel[j];
            A[local * n + k] = (A[local * n + k] - sum) / panel[k];
        }
    }
    for (size_t local = 0; local < localRows; ++local)
        for (size_t j = firstRow + local + 1; j < n; ++j) A[local * n + j] = 0.0;
    return true;
}

size_t rowStart(size_t n, int rank, int ranks) {
    return n * static_cast<size_t>(rank) / static_cast<size_t>(ranks);
}

size_t rowCount(size_t n, int rank, int ranks) {
    return rowStart(n, rank + 1, ranks) - rowStart(n, rank, ranks);
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (n == 0) { if (rank == 0) printf("Matrix size must be positive\n"); MPI_Finalize(); return 1; }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    const size_t firstRow = rowStart(n, rank, ranks);
    const size_t localRows = rowCount(n, rank, ranks);
    std::vector<double> A(localRows * n);
    std::vector<double> fullA;
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        fullA.resize(n * n);
        generatePositiveDefiniteMatrix(fullA, n);
        if (validate) A_orig = fullA;
    }
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = static_cast<int>(rowCount(n, r, ranks) * n);
        displacements[r] = static_cast<int>(rowStart(n, r, ranks) * n);
    }
    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 A.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double>().swap(fullA);
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, firstRow, localRows, rank, ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    std::vector<double> result;
    if (rank == 0) result.resize(n * n);
    MPI_Gatherv(A.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? result.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);
    if (rank == 0) printf("Computation time: %ld ms\n", durationMs);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0;
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(result, "CholeskyL");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(result, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
