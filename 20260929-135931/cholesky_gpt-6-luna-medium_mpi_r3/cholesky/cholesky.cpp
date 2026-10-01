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

static void rowPartition(size_t n, int ranks, std::vector<int>& counts,
                         std::vector<int>& displs) {
    counts.resize(ranks);
    displs.resize(ranks);
    size_t offset = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = n / ranks + (static_cast<size_t>(r) < n % ranks);
        counts[r] = static_cast<int>(rows * n);
        displs[r] = static_cast<int>(offset * n);
        offset += rows;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           size_t firstRow, int rank, int ranks) {
    std::vector<double> pivot(n);
    for (size_t j = 0; j < n; ++j) {
        // Match contiguous, balanced row partitioning.
        int pivotOwner = 0;
        size_t boundary = 0;
        for (; pivotOwner < ranks; ++pivotOwner) {
            const size_t rows = n / ranks + (static_cast<size_t>(pivotOwner) < n % ranks);
            if (j < boundary + rows) break;
            boundary += rows;
        }
        const size_t localJ = j - boundary;
        int ok = 1;
        if (rank == pivotOwner) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += A[localJ * n + k] * A[localJ * n + k];
            const double val = A[localJ * n + j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                ok = 0;
            } else {
                A[localJ * n + j] = sqrt(val);
                for (size_t k = 0; k <= j; ++k) pivot[k] = A[localJ * n + k];
            }
        }
        MPI_Bcast(&ok, 1, MPI_INT, pivotOwner, MPI_COMM_WORLD);
        if (!ok) return false;
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, pivotOwner, MPI_COMM_WORLD);
        if (rank != pivotOwner) {
            // The pivot row's factor entries are also needed for this local row's future sums.
        }
        const size_t localRows = A.size() / n;
        for (size_t li = 0; li < localRows; ++li) {
            const size_t i = firstRow + li;
            if (i <= j) continue;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += A[li * n + k] * pivot[k];
            A[li * n + j] = (A[li * n + j] - sum) / pivot[j];
        }
    }
    for (size_t li = 0; n != 0 && li < A.size() / n; ++li) {
        const size_t i = firstRow + li;
        for (size_t col = i + 1; col < n; ++col) A[li * n + col] = 0.0;
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
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    std::vector<int> counts, displs;
    rowPartition(n, ranks, counts, displs);
    const size_t firstRow = n ? static_cast<size_t>(displs[rank]) / n : 0;
    const size_t localRows = n ? static_cast<size_t>(counts[rank]) / n : 0;
    std::vector<double> A(localRows * n);
    std::vector<double> fullA;
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        fullA.resize(n * n);
        generatePositiveDefiniteMatrix(fullA, n);
    }
    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 A.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (validate) {
        if (rank == 0) A_orig = fullA;
    }
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    bool localSuccess = choleskyDecomposition(A, n, firstRow, rank, ranks);
    int localOk = localSuccess ? 1 : 0, globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count(), maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = static_cast<long>(maxElapsed * 1000.0);
    
    if (!globalOk) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) printf("Computation time: %ld ms\n", duration);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = duration > 0 ? ops / (duration / 1000.0) / 1e9 : 0.0;
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults || validate) {
        if (rank == 0) fullA.resize(n * n);
        MPI_Gatherv(A.data(), counts[rank], MPI_DOUBLE,
                    rank == 0 ? fullA.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    if (printResults && rank == 0) {
        print_results(fullA, "CholeskyL");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(fullA, A_orig, n);
        
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
