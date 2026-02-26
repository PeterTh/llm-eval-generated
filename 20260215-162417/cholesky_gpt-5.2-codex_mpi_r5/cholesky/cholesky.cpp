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

// Parallel Cholesky decomposition (row-blocked, MPI distributed)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
bool choleskyDecompositionMPI(std::vector<double>& A_local, const size_t n,
                              const size_t row_start, const size_t local_rows,
                              const size_t base_rows, const size_t remainder_rows,
                              const int rank) {
    std::vector<double> row_k(n);

    for (size_t k = 0; k < n; ++k) {
        const size_t cutoff = (base_rows + 1) * remainder_rows;
        const int owner = (k < cutoff)
            ? static_cast<int>(k / (base_rows + 1))
            : static_cast<int>(remainder_rows + (k - cutoff) / base_rows);

        int local_error_index = std::numeric_limits<int>::max();

        if (rank == owner) {
            const size_t local_k = k - row_start;
            double* row = &A_local[local_k * n];
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) {
                const double v = row[j];
                sum += v * v;
            }
            const double val = row[k] - sum;
            if (val <= 0.0) {
                local_error_index = static_cast<int>(k);
            } else {
                row[k] = sqrt(val);
                std::memcpy(row_k.data(), row, (k + 1) * sizeof(double));
            }
        }

        int global_error_index = std::numeric_limits<int>::max();
        MPI_Allreduce(&local_error_index, &global_error_index, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (global_error_index != std::numeric_limits<int>::max()) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n", global_error_index);
            }
            return false;
        }

        MPI_Bcast(row_k.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        size_t start_i = 0;
        if (k >= row_start) {
            const size_t offset = k - row_start + 1;
            start_i = (offset >= local_rows) ? local_rows : offset;
        }

        for (size_t ii = start_i; ii < local_rows; ++ii) {
            double* row = &A_local[ii * n];
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) {
                sum += row[j] * row_k[j];
            }
            row[k] = (row[k] - sum) / row_k[k];
        }
    }

    for (size_t ii = 0; ii < local_rows; ++ii) {
        const size_t global_i = row_start + ii;
        double* row = &A_local[ii * n];
        for (size_t j = global_i + 1; j < n; ++j) {
            row[j] = 0.0;
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
    bool argError = false;
    int exitCode = 0;
    
    // Parse command line arguments
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                argError = true;
                break;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
            exitCode = 0;
        } else if (argError) {
            printUsage(argv[0]);
            exitCode = 1;
        }
    }

    int shouldExit = (showHelp || argError) ? 1 : 0;
    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    unsigned long long n_bcast = static_cast<unsigned long long>(n);
    MPI_Bcast(&n_bcast, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n_bcast);

    int validateFlag = validate ? 1 : 0;
    int printFlag = printResults ? 1 : 0;
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validateFlag != 0);
    printResults = (printFlag != 0);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t base_rows = n / static_cast<size_t>(size);
    const size_t remainder_rows = n % static_cast<size_t>(size);

    std::vector<int> row_counts(size);
    std::vector<int> row_displs(size);
    std::vector<int> elem_counts(size);
    std::vector<int> elem_displs(size);
    size_t row_offset = 0;
    for (int r = 0; r < size; ++r) {
        const size_t rows = base_rows + (static_cast<size_t>(r) < remainder_rows ? 1 : 0);
        row_counts[r] = static_cast<int>(rows);
        row_displs[r] = static_cast<int>(row_offset);
        elem_counts[r] = static_cast<int>(rows * n);
        elem_displs[r] = static_cast<int>(row_offset * n);
        row_offset += rows;
    }

    const size_t local_rows = static_cast<size_t>(row_counts[rank]);
    const size_t row_start = static_cast<size_t>(row_displs[rank]);

    std::vector<double> A_local(local_rows * n);
    std::vector<double> A;
    std::vector<double> A_orig;

    if (rank == 0) {
        A.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        if (validate) {
            A_orig = A;
        }
    }

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, elem_counts.data(), elem_displs.data(), MPI_DOUBLE,
                 A_local.data(), elem_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    bool success = choleskyDecompositionMPI(A_local, n, row_start, local_rows,
                                            base_rows, remainder_rows, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", max_elapsed * 1000.0);
        double ops = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n) / 3.0;
        double gflops = ops / max_elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    const bool needGather = printResults || validate;
    if (needGather) {
        if (rank == 0) {
            A.resize(n * n);
        }
        MPI_Gatherv(A_local.data(), elem_counts[rank], MPI_DOUBLE,
                    rank == 0 ? A.data() : nullptr, elem_counts.data(), elem_displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
