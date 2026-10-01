#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

// Cyclic row ownership balances the shrinking trailing matrix across ranks.
static size_t localRows(size_t n, int rank, int ranks) {
    return rank < static_cast<int>(n) ? 1 + (n - 1 - rank) / ranks : 0;
}

static std::vector<int> rowCounts(size_t n, int ranks, size_t width) {
    std::vector<int> counts(ranks);
    for (int r = 0; r < ranks; ++r)
        counts[r] = static_cast<int>(localRows(n, r, ranks) * width);
    return counts;
}

static std::vector<int> displacements(const std::vector<int>& counts) {
    std::vector<int> offsets(counts.size());
    for (size_t r = 1; r < counts.size(); ++r)
        offsets[r] = offsets[r - 1] + counts[r - 1];
    return offsets;
}

static std::vector<double> reorderRows(const std::vector<double>& packed, size_t n,
                                       int ranks, const std::vector<int>& offsets) {
    std::vector<double> matrix(n * n);
    for (size_t i = 0; i < n; ++i) {
        const int owner = static_cast<int>(i % ranks);
        const size_t source = static_cast<size_t>(offsets[owner]) + (i / ranks) * n;
        std::copy_n(packed.data() + source, n, matrix.data() + i * n);
    }
    return matrix;
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int ranks) {
    const size_t rows = localRows(n, rank, ranks);
    std::vector<int> counts(ranks), offsets(ranks), starts(ranks);
    std::vector<double> send(rows), gathered(n), column(n);

    // Each rank owns complete rows of the active Schur complement. Gather its
    // pivot column once, then update only its own lower-triangular rows.
    for (size_t k = 0; k < n; ++k) {
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            starts[r] = k > static_cast<size_t>(r)
                ? static_cast<int>((k - r + ranks - 1) / ranks) : 0;
            counts[r] = static_cast<int>(localRows(n, r, ranks)) - starts[r];
            offsets[r] = total;
            total += counts[r];
        }
        const int first = starts[rank];
        for (int t = first; t < static_cast<int>(rows); ++t)
            send[t - first] = A[static_cast<size_t>(t) * n + k];
        MPI_Allgatherv(send.data(), counts[rank], MPI_DOUBLE,
                       gathered.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        for (size_t i = k; i < n; ++i) {
            const int owner = static_cast<int>(i % ranks);
            column[i] = gathered[static_cast<size_t>(offsets[owner]) + i / ranks - starts[owner]];
        }
        const double diagonal = column[k];
        if (diagonal <= 0.0) {
            if (rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            return false;
        }
        const double inverse = 1.0 / std::sqrt(diagonal);
        for (size_t j = k + 1; j < n; ++j)
            column[j] *= inverse;
        if (static_cast<int>(k % ranks) == rank)
            A[(k / ranks) * n + k] = std::sqrt(diagonal);

        for (size_t t = 0; t < rows; ++t) {
            const size_t i = static_cast<size_t>(rank) + t * ranks;
            if (i <= k) continue;
            double* const row = A.data() + t * n;
            const double factor = column[i];
            row[k] = factor;
            for (size_t j = k + 1; j <= i; ++j)
                row[j] -= factor * column[j];
        }
    }
    for (size_t t = 0; t < rows; ++t) {
        const size_t i = static_cast<size_t>(rank) + t * ranks;
        std::fill(A.begin() + t * n + i + 1, A.begin() + (t + 1) * n, 0.0);
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int rank, int ranks) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    if (rank == 0)
        for (size_t i = 0; i < n * n; ++i)
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    MPI_Bcast(B.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Compute A = B * B^T
    for (size_t t = 0; t < localRows(n, rank, ranks); ++t) {
        const size_t i = static_cast<size_t>(rank) + t * ranks;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[t * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t t = 0; t < localRows(n, rank, ranks); ++t) {
        const size_t i = static_cast<size_t>(rank) + t * ranks;
        A[t * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      size_t n, int rank, int ranks) {
    // Validate by computing L * L^T and comparing with original matrix
    
    const auto counts = rowCounts(n, ranks, n);
    const auto offsets = displacements(counts);
    std::vector<double> packed(n * n);
    MPI_Allgatherv(L.data(), static_cast<int>(L.size()), MPI_DOUBLE,
                   packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    const auto fullL = reorderRows(packed, n, ranks, offsets);
    
    // Compute L * L^T
    double maxError = 0.0;
    double relError = 0.0;
    for (size_t t = 0; t < localRows(n, rank, ranks); ++t) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[t * n + k] * fullL[j * n + k];
            }
            const double original = A_orig[t * n + j];
            const double error = fabs(sum - original);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(original) + 1e-10));
        }
    }
    
    // Compare with original
    double globalMaxError = 0.0, globalRelError = 0.0;
    MPI_Reduce(&maxError, &globalMaxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&relError, &globalRelError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Max absolute error: %.10e\n", globalMaxError);
        printf("Max relative error: %.10e\n", globalRelError);
    }
    
    // Check if error is within tolerance
    if (globalRelError > 1e-6) {
        if (rank == 0) printf("Validation failed: relative error too large\n");
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
    if (n == 0 || n > static_cast<size_t>(std::sqrt(INT_MAX))) {
        if (rank == 0) printf("Matrix size must be between 1 and %d\n", static_cast<int>(std::sqrt(INT_MAX)));
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(localRows(n, rank, ranks) * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, rank, ranks);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    bool success = choleskyDecomposition(A, n, rank, ranks);
    
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / duration / 1e9;
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        const auto counts = rowCounts(n, ranks, n);
        const auto offsets = displacements(counts);
        std::vector<double> packed;
        if (rank == 0) packed.resize(n * n);
        MPI_Gatherv(A.data(), static_cast<int>(A.size()), MPI_DOUBLE,
                    rank == 0 ? packed.data() : nullptr, counts.data(), offsets.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(reorderRows(packed, n, ranks, offsets), "CholeskyL");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n, rank, ranks);
        
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
