#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

size_t localRowCount(const size_t n, const int rank, const int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    return base + (static_cast<size_t>(rank) < n % static_cast<size_t>(ranks));
}

size_t firstLocalRow(const size_t n, const int rank, const int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    return static_cast<size_t>(rank) * base +
           std::min(static_cast<size_t>(rank), extra);
}

int ownerOfRow(const size_t row, const size_t n, const int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t longRows = (base + 1) * extra;
    return row < longRows ? static_cast<int>(row / (base + 1))
                          : static_cast<int>(extra + (row - longRows) / base);
}

// Each rank owns a contiguous group of complete rows.  This layout lets the
// trailing-row updates execute independently after the pivot is broadcast.
bool choleskyDecompositionMPI(std::vector<double>& localA, const size_t n,
                              const size_t firstRow, const int rank,
                              const int ranks) {
    if (n == 0) return true;
    const size_t rows = localA.size() / n;
    std::vector<double> pivot(n);
    for (size_t j = 0; j < n; ++j) {
        const int owner = ownerOfRow(j, n, ranks);
        double diagonal = 0.0;
        int positive = 1;

        if (rank == owner) {
            const size_t localJ = j - firstRow;
            double sum = 0.0;
            const double* row = localA.data() + localJ * n;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * row[k];
            }
            const double value = row[j] - sum;
            if (value <= 0.0) {
                positive = 0;
            } else {
                diagonal = sqrt(value);
                localA[localJ * n + j] = diagonal;
                std::copy_n(row, j, pivot.data());
                pivot[j] = diagonal;
            }
        }

        MPI_Bcast(&positive, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!positive) {
            return false;
        }
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        const size_t begin = std::max(j + 1, firstRow);
        const size_t end = firstRow + rows;
        for (size_t globalI = begin; globalI < end; ++globalI) {
            double* row = localA.data() + (globalI - firstRow) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * pivot[k];
            }
            row[j] = (row[j] - sum) / pivot[j];
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
    int rank = 0;
    int ranks = 1;
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

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> globalA;
    std::vector<double> A_orig;
    if (rank == 0) {
        globalA.resize(n * n);
        generatePositiveDefiniteMatrix(globalA, n);
        if (validate) A_orig = globalA;
    }

    std::vector<int> counts(ranks), displacements(ranks);
    for (int process = 0; process < ranks; ++process) {
        counts[process] = static_cast<int>(localRowCount(n, process, ranks) * n);
        displacements[process] = static_cast<int>(firstLocalRow(n, process, ranks) * n);
    }
    const size_t firstRow = firstLocalRow(n, rank, ranks);
    std::vector<double> localA(localRowCount(n, rank, ranks) * n);
    MPI_Scatterv(rank == 0 ? globalA.data() : nullptr, counts.data(),
                 displacements.data(), MPI_DOUBLE, localA.data(), counts[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecompositionMPI(localA, n, firstRow, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int result = success ? 0 : 1;
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    for (size_t localI = 0; localI < localRowCount(n, rank, ranks); ++localI) {
        const size_t globalI = firstRow + localI;
        std::fill(localA.begin() + localI * n + globalI + 1,
                  localA.begin() + localI * n + n, 0.0);
    }
    if (validate || printResults) {
        if (rank == 0) globalA.resize(n * n);
        MPI_Gatherv(localA.data(), counts[rank], MPI_DOUBLE,
                    rank == 0 ? globalA.data() : nullptr, counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / maxElapsed / 1e9);
        if (printResults) print_results(globalA, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            result = validateCholesky(globalA, A_orig, n) ? 0 : 1;
            printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
