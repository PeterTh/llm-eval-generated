#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Row-block distributed, left-looking Cholesky factorization.  Every rank owns
// complete rows, so all of its row updates are independent after a pivot row
// has been broadcast.  The diagonal dot product is itself split among ranks.
bool choleskyDecomposition(std::vector<double>& localA, size_t n,
                           size_t firstRow, size_t localRows,
                           const std::vector<int>& rowStarts,
                           int rank, int worldSize) {
    std::vector<double> pivot(n);

    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(
            std::upper_bound(rowStarts.begin(), rowStarts.end(), static_cast<int>(k)) -
            rowStarts.begin()) - 1;
        const size_t ownerLocalRow = k - static_cast<size_t>(rowStarts[owner]);

        // Make the already computed prefix of the pivot row available to all
        // ranks.  This also supplies the data for a distributed diagonal sum.
        if (rank == owner && k != 0) {
            std::memcpy(pivot.data(), &localA[ownerLocalRow * n],
                        k * sizeof(double));
        }
        if (k != 0) {
            MPI_Bcast(pivot.data(), static_cast<int>(k), MPI_DOUBLE, owner,
                      MPI_COMM_WORLD);
        }

        double localSum = 0.0;
        const size_t begin = k * static_cast<size_t>(rank) / worldSize;
        const size_t end = k * static_cast<size_t>(rank + 1) / worldSize;
        for (size_t t = begin; t < end; ++t) {
            localSum += pivot[t] * pivot[t];
        }
        double sum = 0.0;
        MPI_Allreduce(&localSum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        int positive = 1;
        if (rank == owner) {
            const double value = localA[ownerLocalRow * n + k] - sum;
            if (value <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
                positive = 0;
            } else {
                pivot[k] = sqrt(value);
                localA[ownerLocalRow * n + k] = pivot[k];
            }
        }
        MPI_Bcast(&positive, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!positive) {
            return false;
        }
        MPI_Bcast(&pivot[k], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Each process factors its local portion of column k independently.
        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            const size_t globalRow = firstRow + localRow;
            if (globalRow <= k) {
                continue;
            }
            double dot = 0.0;
            const double* row = &localA[localRow * n];
            for (size_t t = 0; t < k; ++t) {
                dot += row[t] * pivot[t];
            }
            localA[localRow * n + k] = (row[k] - dot) / pivot[k];
        }
    }

    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        std::fill(&localA[localRow * n + globalRow + 1],
                  &localA[(localRow + 1) * n], 0.0);
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    const size_t maxMpiElements = static_cast<size_t>(std::sqrt(std::numeric_limits<int>::max()));
    if (n == 0 || n > maxMpiElements) {
        if (rank == 0) printf("Matrix size must be between 1 and %zu\n", maxMpiElements);
        MPI_Finalize();
        return 1;
    }

    std::vector<int> rowCounts(worldSize), rowStarts(worldSize), counts(worldSize), displacements(worldSize);
    const size_t baseRows = n / static_cast<size_t>(worldSize);
    const size_t remainder = n % static_cast<size_t>(worldSize);
    size_t offset = 0;
    for (int process = 0; process < worldSize; ++process) {
        const size_t rows = baseRows + (static_cast<size_t>(process) < remainder ? 1 : 0);
        rowStarts[process] = static_cast<int>(offset);
        rowCounts[process] = static_cast<int>(rows);
        counts[process] = static_cast<int>(rows * n);
        displacements[process] = static_cast<int>(offset * n);
        offset += rows;
    }
    const size_t firstRow = static_cast<size_t>(rowStarts[rank]);
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    std::vector<double> localA(localRows * n);
    std::vector<double> A;
    std::vector<double> A_orig;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
        A.resize(n * n);
        generatePositiveDefiniteMatrix(A, n);
        if (validate) A_orig = A;
    }
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 localA.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, n, firstRow, localRows,
                                               rowStarts, rank, worldSize);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (validate || printResults) {
        if (rank == 0) A.resize(n * n);
        MPI_Gatherv(localA.data(), counts[rank], MPI_DOUBLE, rank == 0 ? A.data() : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(A, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            if (validateCholesky(A, A_orig, n)) {
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
