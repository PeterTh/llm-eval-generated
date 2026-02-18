#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static size_t rowsForRank(const size_t n, const int size, const int rank) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t remainder = n % static_cast<size_t>(size);
    return base + (rank < static_cast<int>(remainder) ? 1 : 0);
}

static size_t rowStartForRank(const size_t n, const int size, const int rank) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t remainder = n % static_cast<size_t>(size);
    return static_cast<size_t>(rank) * base + std::min<size_t>(rank, remainder);
}

static int ownerForRow(const size_t n, const int size, const size_t row) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t remainder = n % static_cast<size_t>(size);
    const size_t cutoff = (base + 1) * remainder;
    if (row < cutoff) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(remainder + (row - cutoff) / base);
}

bool mpiCholeskyDecomposition(std::vector<double>& localA, const size_t n, const size_t localRows,
                              const size_t rowStart, MPI_Comm comm) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<double> rowBuffer;
    rowBuffer.reserve(n);
    bool localOk = true;

    for (size_t k = 0; k < n; ++k) {
        const int owner = ownerForRow(n, size, k);
        if (rank == owner) {
            const size_t localIndex = k - rowStart;
            double* row = localA.data() + localIndex * n;
            double sum = 0.0;
            for (size_t r = 0; r < k; ++r) {
                sum += row[r] * row[r];
            }
            const double val = row[k] - sum;
            if (val <= 0.0) {
                localOk = false;
                row[k] = 0.0;
            } else {
                row[k] = sqrt(val);
            }
            rowBuffer.resize(k + 1);
            std::memcpy(rowBuffer.data(), row, (k + 1) * sizeof(double));
            for (size_t j = k + 1; j < n; ++j) {
                row[j] = 0.0;
            }
        } else {
            rowBuffer.resize(k + 1);
        }

        MPI_Bcast(rowBuffer.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner, comm);
        const double diag = rowBuffer[k];
        if (diag == 0.0) {
            localOk = false;
            continue;
        }

        size_t updateStart = 0;
        if (k < rowStart) {
            updateStart = 0;
        } else if (k >= rowStart + localRows) {
            updateStart = localRows;
        } else {
            updateStart = k - rowStart + 1;
        }

        const double* rowk = rowBuffer.data();
        for (size_t localRow = updateStart; localRow < localRows; ++localRow) {
            double* row = localA.data() + localRow * n;
            double sum = 0.0;
            for (size_t r = 0; r < k; ++r) {
                sum += row[r] * rowk[r];
            }
            row[k] = (row[k] - sum) / diag;
        }
    }

    int globalOk = 0;
    const int localOkInt = localOk ? 1 : 0;
    MPI_Allreduce(&localOkInt, &globalOk, 1, MPI_INT, MPI_MIN, comm);
    return globalOk == 1;
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
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseOk = true;
    
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
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            parseOk = false;
            break;
        }
    }

    if (showHelp || !parseOk) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseOk ? 0 : 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    

    const size_t localRows = rowsForRank(n, size, rank);
    const size_t rowStart = rowStartForRank(n, size, rank);

    std::vector<double> fullA;
    std::vector<double> A_orig;
    if (rank == 0) {
        fullA.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(fullA, n);
        if (validate) {
            A_orig = fullA;
        }
    }

    std::vector<int> sendCounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t rows = rowsForRank(n, size, r);
        sendCounts[r] = static_cast<int>(rows * n);
        displs[r] = static_cast<int>(rowStartForRank(n, size, r) * n);
    }

    std::vector<double> localA(localRows * n);
    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, sendCounts.data(), displs.data(),
                 MPI_DOUBLE, localA.data(), static_cast<int>(localRows * n),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<double>().swap(fullA);
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const bool success = mpiCholeskyDecomposition(localA, n, localRows, rowStart, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    double maxTime = 0.0;
    const double localTime = end - start;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long durationMs = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = maxTime > 0.0 ? ops / maxTime / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullL;
    if (printResults || validate) {
        if (rank == 0) {
            fullL.resize(n * n);
        }
        MPI_Gatherv(localA.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
                    rank == 0 ? fullL.data() : nullptr, sendCounts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0 && printResults) {
        print_results(fullL, "CholeskyL");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(fullL, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        }
        printf("Validation: FAILED\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
