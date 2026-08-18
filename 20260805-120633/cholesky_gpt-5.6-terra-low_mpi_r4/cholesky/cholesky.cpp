#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are distributed contiguously.  At step j the owner of row j broadcasts
// its completed prefix; every rank can then independently factor its local rows.
bool choleskyDecomposition(std::vector<double>& localA, size_t n, size_t firstRow,
                           const std::vector<size_t>& rowStarts, int rank,
                           MPI_Comm comm) {
    std::vector<double> pivot(n);
    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(
            std::upper_bound(rowStarts.begin(), rowStarts.end(), j) - rowStarts.begin() - 1);
        int positive = 1;
        if (rank == owner) {
            double* row = localA.data() + (j - firstRow) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * row[k];
            const double value = row[j] - sum;
            if (value <= 0.0) {
                positive = 0;
            } else {
                row[j] = std::sqrt(value);
                std::copy_n(row, j + 1, pivot.data());
            }
        }
        MPI_Bcast(&positive, 1, MPI_INT, owner, comm);
        if (!positive) return false;
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, comm);

        const size_t localRows = localA.size() / n;
        const size_t begin = std::max(j + 1, firstRow);
        const size_t end = firstRow + localRows;
        for (size_t i = begin; i < end; ++i) {
            double* row = localA.data() + (i - firstRow) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
            row[j] = (row[j] - sum) / pivot[j];
        }
    }
    // Preserve the original routine's lower-triangular result.
    for (size_t local = 0; local < localA.size() / n; ++local) {
        double* row = localA.data() + local * n;
        const size_t globalRow = firstRow + local;
        std::fill(row + globalRow + 1, row + n, 0.0);
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    constexpr size_t maxMpiMatrixDimension = 46340; // floor(sqrt(INT_MAX))
    if (n == 0 || n > maxMpiMatrixDimension) {
        if (rank == 0) printf("Matrix size must be between 1 and %zu\n", maxMpiMatrixDimension);
        MPI_Finalize();
        return 1;
    }
    int exitCode = 0;
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A;
    std::vector<double> A_orig;
    
    std::vector<int> counts(ranks), displacements(ranks);
    std::vector<size_t> rowStarts(ranks + 1);
    for (int process = 0; process < ranks; ++process) {
        const size_t start = n * static_cast<size_t>(process) / ranks;
        const size_t finish = n * static_cast<size_t>(process + 1) / ranks;
        rowStarts[process] = start;
        counts[process] = static_cast<int>((finish - start) * n);
        displacements[process] = static_cast<int>(start * n);
    }
    rowStarts[ranks] = n;
    const size_t firstRow = rowStarts[rank];
    const size_t localRows = rowStarts[rank + 1] - firstRow;
    std::vector<double> localA(localRows * n);

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        A.resize(n * n);
        generatePositiveDefiniteMatrix(A, n);
    }
    
    if (validate && rank == 0) {
        A_orig = A; // Save original for validation
    }
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 localA.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    bool success = choleskyDecomposition(localA, n, firstRow, rowStarts, rank, MPI_COMM_WORLD);
    
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Gatherv(localA.data(), counts[rank], MPI_DOUBLE, rank == 0 ? A.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
    printf("Computation time: %.0f ms\n", duration * 1000.0);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / duration / 1e9;
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
            exitCode = 1;
        }
    }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
