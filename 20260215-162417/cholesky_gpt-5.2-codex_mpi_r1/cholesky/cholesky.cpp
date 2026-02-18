#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static inline size_t rowStart(const int rank, const int size, const size_t n) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    return base * static_cast<size_t>(rank) + (rank < static_cast<int>(rem) ? static_cast<size_t>(rank) : rem);
}

static inline size_t rowCount(const int rank, const int size, const size_t n) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    return base + (rank < static_cast<int>(rem) ? 1 : 0);
}

static inline int rowOwner(const size_t row, const int size, const size_t n) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    if (base == 0) {
        return static_cast<int>(row);
    }
    const size_t threshold = (base + 1) * rem;
    if (row < threshold) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - threshold) / base);
}

// MPI Cholesky decomposition with row-wise distribution
bool choleskyDecompositionMPI(std::vector<double>& A_local, const size_t n, const size_t row_start,
                              const size_t local_rows, const int rank, const int size, MPI_Comm comm) {
    std::vector<double> row_j(n);

    for (size_t j = 0; j < n; ++j) {
        const int owner = rowOwner(j, size, n);
        int local_error = -1;

        if (rank == owner) {
            const size_t local_j = j - row_start;
            double* row_ptr = A_local.data() + local_j * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row_ptr[k] * row_ptr[k];
            }
            const double val = row_ptr[j] - sum;
            if (val <= 0.0) {
                local_error = static_cast<int>(j);
            } else {
                row_ptr[j] = std::sqrt(val);
                std::memcpy(row_j.data(), row_ptr, (j + 1) * sizeof(double));
            }
        }

        int global_error = -1;
        MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, comm);
        if (global_error >= 0) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n", global_error);
            }
            return false;
        }

        MPI_Bcast(row_j.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, comm);
        const double diag = row_j[j];

        size_t local_i_start = 0;
        if (row_start <= j) {
            const size_t skip = j + 1 - row_start;
            local_i_start = skip < local_rows ? skip : local_rows;
        }

        for (size_t local_i = local_i_start; local_i < local_rows; ++local_i) {
            double* row_i = A_local.data() + local_i * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row_i[k] * row_j[k];
            }
            row_i[j] = (row_i[j] - sum) / diag;
        }
    }

    for (size_t local_i = 0; local_i < local_rows; ++local_i) {
        const size_t global_i = row_start + local_i;
        double* row_i = A_local.data() + local_i * n;
        for (size_t col = global_i + 1; col < n; ++col) {
            row_i[col] = 0.0;
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
    bool should_exit = false;
    int exit_code = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                should_exit = true;
                exit_code = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                should_exit = true;
                exit_code = 1;
            }
        }
    }

    uint64_t n64 = static_cast<uint64_t>(n);
    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;
    int should_exit_i = should_exit ? 1 : 0;
    int exit_code_i = exit_code;

    MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&should_exit_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    n = static_cast<size_t>(n64);
    validate = validate_i != 0;
    printResults = print_i != 0;
    should_exit = should_exit_i != 0;
    exit_code = exit_code_i;

    if (should_exit) {
        MPI_Finalize();
        return exit_code;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    const size_t local_rows = rowCount(rank, size, n);
    const size_t row_start = rowStart(rank, size, n);
    const int local_count = static_cast<int>(local_rows * n);

    std::vector<double> A_local(local_rows * n);
    std::vector<double> A_full;
    std::vector<double> A_orig;
    std::vector<int> sendcounts;
    std::vector<int> displs;

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        A_full.resize(n * n);
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full;
        }
        sendcounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            const size_t r_rows = rowCount(r, size, n);
            sendcounts[r] = static_cast<int>(r_rows * n);
            displs[r] = static_cast<int>(rowStart(r, size, n) * n);
        }
    }

    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr,
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_DOUBLE,
                 local_rows ? A_local.data() : nullptr,
                 local_count,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0 && !(validate || printResults)) {
        std::vector<double>().swap(A_full);
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecompositionMPI(A_local, n, row_start, local_rows, rank, size, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    const double local_duration = end - start;
    double max_duration = 0.0;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int success_i = success ? 1 : 0;
    int global_success = 0;
    MPI_Allreduce(&success_i, &global_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!global_success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long ms = static_cast<long>(max_duration * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        if (max_duration > 0.0) {
            const double gflops = ops / max_duration / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        } else {
            printf("Performance: inf GFLOPS\n");
        }
    }

    if (validate || printResults) {
        if (rank == 0 && A_full.size() != n * n) {
            A_full.resize(n * n);
        }
        MPI_Gatherv(local_rows ? A_local.data() : nullptr,
                    local_count,
                    MPI_DOUBLE,
                    rank == 0 ? A_full.data() : nullptr,
                    rank == 0 ? sendcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(A_full, "CholeskyL");
            }
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateCholesky(A_full, A_orig, n);
                if (valid) {
                    printf("Validation: PASSED\n");
                    MPI_Finalize();
                    return 0;
                }
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
