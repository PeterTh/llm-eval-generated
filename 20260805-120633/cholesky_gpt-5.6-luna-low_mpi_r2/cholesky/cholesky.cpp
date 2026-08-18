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

bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           size_t firstRow, size_t localRows, MPI_Comm comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    std::vector<double> pivot(n);
    bool success = true;
    for (size_t k = 0; k < n; ++k) {
        int pivotOwner = 0;
        // The caller's distribution is balanced; derive owner via all ranks' local ranges.
        int candidate = (firstRow <= k && k < firstRow + localRows) ? rank : -1;
        MPI_Allreduce(&candidate, &pivotOwner, 1, MPI_INT, MPI_MAX, comm);
        if (rank == pivotOwner) {
            double sum = 0.0;
            const double* row = &A[(k - firstRow) * n];
            for (size_t q = 0; q < k; ++q) sum += row[q] * row[q];
            const double val = row[k] - sum;
            if (val <= 0.0) success = false;
            else A[(k - firstRow) * n + k] = std::sqrt(val);
            if (success) {
                pivot[k] = A[(k - firstRow) * n + k];
                for (size_t q = 0; q < k; ++q) pivot[q] = A[(k - firstRow) * n + q];
                for (size_t q = k + 1; q < n; ++q) pivot[q] = 0.0;
                for (size_t q = k + 1; q < n; ++q) A[(k - firstRow) * n + q] = 0.0;
            }
        }
        MPI_Bcast(&success, 1, MPI_C_BOOL, pivotOwner, comm);
        if (!success) { if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", k); return false; }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_DOUBLE, pivotOwner, comm);
        const size_t begin = std::max(k + 1, firstRow);
        const size_t end = firstRow + localRows;
        for (size_t i = begin; i < end; ++i) {
            double sum = 0.0;
            double* row = &A[(i - firstRow) * n];
            for (size_t q = 0; q < k; ++q) sum += row[q] * pivot[q];
            row[k] = (row[k] - sum) / pivot[k];
            for (size_t q = i + 1; q < n; ++q) row[q] = 0.0;
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
    int rank, ranks;
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
            printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t localRows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    std::vector<double> globalA;
    if (rank == 0) globalA.resize(n * n);
    std::vector<double> A(localRows * n);
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(globalA, n);
    }
    std::vector<int> counts(ranks), displs(ranks);
    for (int p = 0; p < ranks; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < extra ? 1 : 0);
        counts[p] = static_cast<int>(rows * n);
        displs[p] = static_cast<int>((static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), extra)) * n);
    }
    MPI_Scatterv(rank == 0 ? globalA.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 A.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, firstRow, localRows, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localMs = duration.count(), elapsedMs = 0;
    MPI_Reduce(&localMs, &elapsedMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize(); return 1;
    }
    
    if (rank == 0) printf("Computation time: %lld ms\n", elapsedMs);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", ops / (elapsedMs / 1000.0) / 1e9);
    
    // Print results for external validation
    if (printResults) {
        if (rank == 0) globalA.resize(n * n);
        MPI_Gatherv(A.data(), counts[rank], MPI_DOUBLE, rank == 0 ? globalA.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(globalA, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        std::vector<double> gathered;
        if (rank == 0) gathered.resize(n * n);
        MPI_Gatherv(A.data(), counts[rank], MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        bool valid = true;
        if (rank == 0) valid = validateCholesky(gathered, globalA, n);
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        if (rank == 0 && valid) {
            printf("Validation: PASSED\n");
        } else if (rank == 0) {
            printf("Validation: FAILED\n");
        }
        MPI_Finalize(); return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
