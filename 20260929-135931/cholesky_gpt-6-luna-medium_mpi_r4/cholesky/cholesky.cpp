#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank,
                           int ranks, const std::vector<int>& counts) {
    std::vector<double> pivot(n);
    bool success = true;
    for (size_t k = 0; k < n; ++k) {
        const int owner = std::min(ranks - 1, static_cast<int>((k * static_cast<size_t>(ranks)) / n));
        const size_t ownerFirst = (static_cast<size_t>(owner) * n + ranks - 1) / ranks;
        if (rank == owner) {
            const size_t local = k - ownerFirst;
            double& diagonal = A[local * n + k];
            if (!(diagonal > 0.0)) success = false;
            else diagonal = std::sqrt(diagonal);
            if (success) {
                for (size_t j = k + 1; j < n; ++j)
                    A[local * n + j] /= diagonal;
                std::copy_n(A.data() + local * n, n, pivot.data());
            }
        }
        MPI_Bcast(&success, 1, MPI_CXX_BOOL, owner, MPI_COMM_WORLD);
        if (!success) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            return false;
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        // Each rank updates its owned rows of the trailing matrix.
        const size_t first = (static_cast<size_t>(rank) * n + ranks - 1) / ranks;
        for (int local = 0; local < counts[rank] / static_cast<int>(n); ++local) {
            const size_t i = first + local;
            if (i <= k) continue;
            double* row = A.data() + local * n;
            const double lik = row[k] / pivot[k];
            row[k] = lik;
            for (size_t j = k + 1; j < n; ++j) row[j] -= lik * pivot[j];
        }
    }
    const size_t first = (static_cast<size_t>(rank) * n + ranks - 1) / ranks;
    for (int local = 0; local < counts[rank] / static_cast<int>(n); ++local) {
        const size_t i = first + static_cast<size_t>(local);
        std::fill(A.begin() + static_cast<size_t>(local) * n + i + 1,
                  A.begin() + static_cast<size_t>(local + 1) * n, 0.0);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    
    if (n == 0 || n > static_cast<size_t>(INT_MAX) || n * n > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) printf("Invalid matrix size\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t first = (static_cast<size_t>(r) * n + ranks - 1) / ranks;
        const size_t last = (static_cast<size_t>(r + 1) * n + ranks - 1) / ranks;
        counts[r] = static_cast<int>((last - first) * n);
        displs[r] = static_cast<int>(first * n);
    }
    std::vector<double> A(static_cast<size_t>(counts[rank]));
    std::vector<double> full;
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        full.resize(n * n);
        generatePositiveDefiniteMatrix(full, n);
    }
    
    if (validate) {
        if (rank == 0) A_orig = full;
    }
    MPI_Scatterv(rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 A.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, ranks, counts);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) { if (rank == 0) printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    
    double elapsed = std::chrono::duration<double>(end - start).count(), maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> result(rank == 0 ? n * n : 0);
        MPI_Gatherv(A.data(), counts[rank], MPI_DOUBLE, rank == 0 ? result.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(result, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        std::vector<double> result(rank == 0 ? n * n : 0);
        MPI_Gatherv(A.data(), counts[rank], MPI_DOUBLE, rank == 0 ? result.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        bool valid = rank == 0 ? validateCholesky(result, A_orig, n) : true;
        
        int validInt = valid ? 1 : 0;
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (validInt) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize(); return 0;
}
