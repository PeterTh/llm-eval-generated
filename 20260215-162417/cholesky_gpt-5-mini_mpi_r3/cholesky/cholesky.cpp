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

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            
            if (i == j) {
                // Diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else {
                // Off-diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }
        
        // Zero out upper triangular part
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
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
            return 0;
        } else {
            if (i==1) {
                // proceed
            }
            if (i==1) {
                // noop
            }
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine row distribution (1D row-block)
    std::vector<int> rows(size);
    int base = (int)(n / size);
    int rem = (int)(n % size);
    for (int r = 0, offset = 0; r < size; ++r) {
        rows[r] = base + (r < rem ? 1 : 0);
        offset += rows[r];
    }

    std::vector<int> counts(size), displs(size), row_starts(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        row_starts[r] = offset;
        counts[r] = rows[r] * (int)n; // number of elements
        displs[r] = offset * (int)n;
        offset += rows[r];
    }

    // Prepare local storage
    int local_rows = rows[rank];
    std::vector<double> local_A((size_t)local_rows * n);

    std::vector<double> A; // full matrix only on rank 0
    std::vector<double> A_orig;
    if (rank == 0) {
        A.resize((size_t)n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        if (validate) {
            A_orig = A; // save original
        }
    }

    // Scatter rows to all ranks
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distributed Cholesky
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int global_ok = 1;
    for (int j = 0; j < (int)n; ++j) {
        // Determine owner of row j
        int owner = 0;
        for (int r = 0; r < size; ++r) {
            if (j >= row_starts[r] && j < row_starts[r] + rows[r]) { owner = r; break; }
        }

        std::vector<double> panel(j + 1);

        if (rank == owner) {
            int local_j = j - row_starts[rank];
            // Compute diagonal element L[j,j]
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                double v = local_A[(size_t)local_j * n + k];
                sum += v * v;
            }
            double val = local_A[(size_t)local_j * n + j] - sum;
            int local_ok = 1;
            if (val <= 0.0) {
                local_ok = 0;
            } else {
                local_A[(size_t)local_j * n + j] = sqrt(val);
            }
            // Fill panel (L[j,0..j])
            for (int k = 0; k <= j; ++k) panel[k] = local_A[(size_t)local_j * n + k];
            // Check positivity across ranks
            MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            if (!global_ok) break;
        } else {
            // Non-owner participates in reduction
            int dummy = 1;
            MPI_Allreduce(&dummy, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            if (!global_ok) break;
        }

        // Broadcast the panel from owner
        MPI_Bcast(panel.data(), j + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Update local rows i > j
        for (int i_local = 0; i_local < local_rows; ++i_local) {
            int i_global = row_starts[rank] + i_local;
            if (i_global > j) {
                double sum = 0.0;
                for (int k = 0; k < j; ++k) {
                    sum += local_A[(size_t)i_local * n + k] * panel[k];
                }
                local_A[(size_t)i_local * n + j] = (local_A[(size_t)i_local * n + j] - sum) / panel[j];
            }
            // zero upper triangular for cleanliness
            if (i_global <= j) {
                // nothing
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Gather results to rank 0
    if (rank == 0) {
        A.clear();
        A.resize((size_t)n * n);
    }
    MPI_Gatherv(local_A.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? A.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double duration_ms = (t1 - t0) * 1000.0;
        if (!global_ok) {
            printf("Cholesky decomposition failed: matrix not positive definite\n");
            MPI_Finalize();
            return 1;
        }
        printf("Computation time: %.0f ms\n", duration_ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }

        if (validate) {
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
    }

    MPI_Finalize();
    return 0;
}
