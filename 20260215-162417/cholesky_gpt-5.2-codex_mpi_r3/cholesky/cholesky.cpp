#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition with MPI (column-oriented, row-block distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
static int ownerOfRow(const size_t row, const size_t baseRows, const size_t remainder) {
    const size_t cutoff = (baseRows + 1) * remainder;
    if (row < cutoff) {
        return static_cast<int>(row / (baseRows + 1));
    }
    return static_cast<int>(remainder + (row - cutoff) / baseRows);
}

bool choleskyDecompositionMPI(std::vector<double>& localA,
                              const size_t n,
                              const size_t rowStart,
                              const size_t rowCount,
                              const size_t baseRows,
                              const size_t remainder,
                              MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<double> rowBuffer(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        const int owner = ownerOfRow(i, baseRows, remainder);
        if (rank == owner) {
            const size_t localIndex = i - rowStart;
            double* rowPtr = localA.data() + localIndex * n;
            double sum = 0.0;
            for (size_t k = 0; k < i; ++k) {
                const double val = rowPtr[k];
                sum += val * val;
            }
            const double diag = rowPtr[i] - sum;
            if (diag <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
                MPI_Abort(comm, 1);
            }
            rowPtr[i] = sqrt(diag);
            for (size_t k = 0; k <= i; ++k) {
                rowBuffer[k] = rowPtr[k];
            }
        }

        MPI_Bcast(rowBuffer.data(), static_cast<int>(i + 1), MPI_DOUBLE, owner, comm);
        const double diag = rowBuffer[i];

        for (size_t localRow = 0; localRow < rowCount; ++localRow) {
            const size_t globalRow = rowStart + localRow;
            if (globalRow <= i) {
                continue;
            }
            double* rowPtr = localA.data() + localRow * n;
            double sum = 0.0;
            for (size_t k = 0; k < i; ++k) {
                sum += rowPtr[k] * rowBuffer[k];
            }
            rowPtr[i] = (rowPtr[i] - sum) / diag;
        }
    }

    for (size_t localRow = 0; localRow < rowCount; ++localRow) {
        const size_t globalRow = rowStart + localRow;
        double* rowPtr = localA.data() + localRow * n;
        for (size_t j = globalRow + 1; j < n; ++j) {
            rowPtr[j] = 0.0;
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseOk = true;
    const char* unknownArg = nullptr;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseOk = false;
            if (!unknownArg) {
                unknownArg = argv[i];
            }
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (!parseOk) {
        if (rank == 0) {
            if (unknownArg) {
                printf("Unknown option: %s\n", unknownArg);
            } else {
                printf("Unknown option in arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    const size_t baseRows = n / static_cast<size_t>(size);
    const size_t remainder = n % static_cast<size_t>(size);
    const size_t rowCount = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t rowStart =
        static_cast<size_t>(rank) * baseRows + std::min(static_cast<size_t>(rank), remainder);

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    for (int r = 0; r < size; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        if (rows == 0) {
            counts[r] = 0;
            displs[r] = 0;
            continue;
        }
        const size_t start =
            static_cast<size_t>(r) * baseRows + std::min(static_cast<size_t>(r), remainder);
        counts[r] = static_cast<int>(rows * n);
        displs[r] = static_cast<int>(start * n);
    }

    const int localCount = static_cast<int>(rowCount * n);
    std::vector<double> localA(static_cast<size_t>(localCount));
    
    std::vector<double> A_full;
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        A_full.resize(n * n);
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full; // Save original for validation
        }
    }

    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_DOUBLE,
                 localA.data(),
                 localCount,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    bool success =
        choleskyDecompositionMPI(localA, n, rowStart, rowCount, baseRows, remainder, MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const double durationMs = maxTime * 1000.0;
        printf("Computation time: %.0f ms\n", durationMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / maxTime / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    const bool needGather = printResults || validate;
    if (needGather) {
        if (rank == 0) {
            A_full.assign(n * n, 0.0);
        }
        MPI_Gatherv(localA.data(),
                    localCount,
                    MPI_DOUBLE,
                    rank == 0 ? A_full.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
    }

    if (rank == 0 && printResults) {
        print_results(A_full, "CholeskyL");
    }

    int validFlag = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A_full, A_orig, n);
            validFlag = valid ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!validFlag) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
