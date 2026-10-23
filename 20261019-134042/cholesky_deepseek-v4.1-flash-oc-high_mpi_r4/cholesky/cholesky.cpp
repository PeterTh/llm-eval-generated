#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (distributed, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

static size_t numLocalRows(const size_t n, const int rank, const int size) {
    if ((size_t)rank >= n) {
        return 0;
    }
    return (n - (size_t)rank + (size_t)size - 1) / (size_t)size;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, const int rank, const int size, const size_t localRows) {
    // A is stored in row-major order
    const size_t p = (size_t)size;
    std::vector<double> rowbuf(n);

    for (size_t j = 0; j < n; ++j) {
        const int owner = (int)(j % p);
        const size_t ownerRow = j / p;

        double* bcastPtr;
        if (rank == owner) {
            double* row = &A[ownerRow * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * row[k];
            }
            const double val = row[j] - sum;
            if (val <= 0.0) {
                row[j] = val;
            } else {
                row[j] = sqrt(val);
            }
            bcastPtr = row;
        } else {
            bcastPtr = rowbuf.data();
        }

        MPI_Bcast(bcastPtr, (int)(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        const double diag = bcastPtr[j];
        if (diag <= 0.0) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        size_t k0;
        if ((size_t)rank > j) {
            k0 = 0;
        } else {
            k0 = (j - (size_t)rank) / p + 1;
        }

        for (size_t k = k0; k < localRows; ++k) {
            double* row = &A[k * n];
            double sum = 0.0;
            for (size_t t = 0; t < j; ++t) {
                sum += row[t] * bcastPtr[t];
            }
            row[j] = (row[j] - sum) / diag;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, const int rank, const int size, const size_t localRows) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    A.assign(localRows * n, 0.0);
    if (localRows == 0) {
        return;
    }

    const size_t p = (size_t)size;
    std::vector<double> Blocal(localRows * n);
    std::vector<double> Bj(n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t j = 0; j < n; ++j) {
        const bool keep = ((j % p) == (size_t)rank);
        double* dst = keep ? &Blocal[(j / p) * n] : nullptr;
        for (size_t t = 0; t < n; ++t) {
            const double value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            if (keep) {
                dst[t] = value;
            }
        }
    }

    // Compute A = B * B^T
    seed = 42;
    for (size_t j = 0; j < n; ++j) {
        for (size_t t = 0; t < n; ++t) {
            Bj[t] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        for (size_t k = 0; k < localRows; ++k) {
            const double* Bi = &Blocal[k * n];
            double sum = 0.0;
            for (size_t t = 0; t < n; ++t) {
                sum += Bi[t] * Bj[t];
            }
            A[k * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t k = 0; k < localRows; ++k) {
        const size_t i = k * p + (size_t)rank;
        A[k * n + i] += n;
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

static void gatherRows(const std::vector<double>& local, const size_t n, const size_t localRows, const int rank, const int size, std::vector<double>& full) {
    if (n == 0) {
        if (rank == 0) {
            full.clear();
        }
        return;
    }

    const int mycount = (int)(localRows * n);
    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    std::vector<double> gathered;

    MPI_Gather(&mycount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = total;
            total += counts[r];
        }
        gathered.resize((size_t)total);
    }

    MPI_Gatherv(localRows > 0 ? local.data() : nullptr, mycount, MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        full.assign(n * n, 0.0);
        for (int r = 0; r < size; ++r) {
            const size_t cnt = (size_t)counts[r] / n;
            for (size_t k = 0; k < cnt; ++k) {
                const size_t i = k * (size_t)size + (size_t)r;
                std::copy_n(&gathered[(size_t)displs[r] + k * n], n, &full[i * n]);
            }
        }
    }
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

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t localRows = numLocalRows(n, rank, size);

    // Allocate local rows and generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> A;
    generatePositiveDefiniteMatrix(A, n, rank, size, localRows);
    std::vector<double> A_orig;
    if (validate) {
        A_orig = A;
    }

    // Zero out upper triangular part
    for (size_t k = 0; k < localRows; ++k) {
        const size_t i = k * (size_t)size + (size_t)rank;
        double* row = &A[k * n];
        for (size_t j = i + 1; j < n; ++j) {
            row[j] = 0.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, rank, size, localRows);

    auto end = std::chrono::high_resolution_clock::now();

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> fullA;
        gatherRows(A, n, localRows, rank, size, fullA);
        if (rank == 0) {
            print_results(fullA, "CholeskyL");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        std::vector<double> fullL;
        std::vector<double> fullOrig;
        gatherRows(A, n, localRows, rank, size, fullL);
        gatherRows(A_orig, n, localRows, rank, size, fullOrig);
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateCholesky(fullL, fullOrig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
