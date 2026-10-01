#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are assigned in blocks round-robin. Each panel is factored by its
// owner, then broadcast so every rank can compute its own rows independently.
static size_t localIndex(size_t row, size_t block, int ranks) {
    return (row / (block * static_cast<size_t>(ranks))) * block + row % block;
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, size_t block,
                           int rank, int ranks, const std::vector<size_t>& ownedRows) {
    for (size_t first = 0; first < n; first += block) {
        const size_t last = std::min(n, first + block);
        const int owner = static_cast<int>((first / block) % ranks);
        int failed = 0;
        size_t badDiagonal = 0;

        if (rank == owner) {
            for (size_t i = first; i < last && !failed; ++i) {
                double* row = A.data() + localIndex(i, block, ranks) * n;
                for (size_t j = first; j <= i; ++j) {
                    const double* pivot = A.data() + localIndex(j, block, ranks) * n;
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
                    if (i == j) {
                        const double value = row[j] - sum;
                        if (value <= 0.0) {
                            failed = 1;
                            badDiagonal = j;
                            break;
                        }
                        row[j] = std::sqrt(value);
                    } else {
                        row[j] = (row[j] - sum) / pivot[j];
                    }
                }
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) {
            if (rank == owner)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", badDiagonal);
            return false;
        }

        // Send only the computed prefix of each panel row. The prefix is
        // identical to the one used by the original left-looking algorithm.
        const size_t width = last;
        std::vector<double> panel((last - first) * width, 0.0);
        if (rank == owner) {
            for (size_t j = first; j < last; ++j) {
                const double* row = A.data() + localIndex(j, block, ranks) * n;
                std::copy_n(row, j + 1, panel.data() + (j - first) * width);
            }
        }
        MPI_Bcast(panel.data(), static_cast<int>(panel.size()), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        for (size_t local = 0; local < ownedRows.size(); ++local) {
            const size_t i = ownedRows[local];
            if (i < last) continue;
            double* row = A.data() + local * n;
            for (size_t j = first; j < last; ++j) {
                const double* pivot = panel.data() + (j - first) * width;
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
                row[j] = (row[j] - sum) / pivot[j];
            }
        }
    }
    for (size_t local = 0; local < ownedRows.size(); ++local) {
        const size_t i = ownedRows[local];
        double* row = A.data() + local * n;
        std::fill(row + i + 1, row + n, 0.0);
    }
    return true;
}

// Generate the same B on each rank, then form only the owned rows of A.
// Repeating the inexpensive random stream avoids a central matrix scatter.
void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n,
                                    const std::vector<size_t>& ownedRows) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    for (size_t local = 0; local < ownedRows.size(); ++local) {
        const size_t i = ownedRows[local];
        double* row = A.data() + local * n;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            row[j] = sum;
        }
        row[i] += n;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

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
    if (n == 0 || n > static_cast<size_t>(INT_MAX) / n) {
        if (rank == 0) fprintf(stderr, "Matrix size must be positive and fit MPI counts\n");
        MPI_Finalize();
        return 1;
    }
    const size_t block = std::max<size_t>(1, std::min<size_t>(32, n / (4 * static_cast<size_t>(ranks))));
    std::vector<size_t> ownedRows;
    for (size_t i = 0; i < n; ++i)
        if (static_cast<int>((i / block) % ranks) == rank) ownedRows.push_back(i);
    std::vector<double> A(ownedRows.size() * n);
    std::vector<double> A_orig;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, ownedRows);
    if (validate) A_orig = A;

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, n, block, rank, ranks, ownedRows);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / duration / 1e9);
    }

    if (printResults || validate) {
        std::vector<int> counts(ranks), displs(ranks);
        for (size_t i = 0; i < n; ++i)
            ++counts[(i / block) % ranks];
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            counts[r] *= static_cast<int>(n);
            displs[r] = total;
            total += counts[r];
        }
        std::vector<double> packed, full, originalPacked, original;
        if (rank == 0) packed.resize(n * n);
        MPI_Gatherv(A.data(), static_cast<int>(A.size()), MPI_DOUBLE,
                    rank == 0 ? packed.data() : nullptr, counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) {
            if (rank == 0) originalPacked.resize(n * n);
            MPI_Gatherv(A_orig.data(), static_cast<int>(A_orig.size()), MPI_DOUBLE,
                        rank == 0 ? originalPacked.data() : nullptr, counts.data(),
                        displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            full.resize(n * n);
            if (validate) original.resize(n * n);
            for (size_t i = 0; i < n; ++i) {
                const int owner = (i / block) % ranks;
                const size_t src = displs[owner] + localIndex(i, block, ranks) * n;
                std::copy_n(packed.data() + src, n, full.data() + i * n);
                if (validate)
                    std::copy_n(originalPacked.data() + src, n, original.data() + i * n);
            }
            if (printResults) print_results(full, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateCholesky(full, original, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                exitCode = valid ? 0 : 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return exitCode;
}
