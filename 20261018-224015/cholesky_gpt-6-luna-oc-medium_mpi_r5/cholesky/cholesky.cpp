#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& local, const size_t n, int rank, int ranks) {
    std::vector<double> pivot(n);
    bool success = true;
    for (size_t i = 0; i < n; ++i) {
        const int owner = static_cast<int>(i % ranks);
        if (rank == owner) {
            const size_t localRow = i / ranks;
            double sum = 0.0;
            for (size_t k = 0; k < i; ++k)
                sum += local[localRow * n + k] * local[localRow * n + k];
            const double val = local[localRow * n + i] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
                success = false;
            } else {
                local[localRow * n + i] = sqrt(val);
            }
            std::copy_n(local.data() + localRow * n, n, pivot.data());
        }
        MPI_Bcast(&success, 1, MPI_C_BOOL, owner, MPI_COMM_WORLD);
        if (!success) return false;
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Each process updates its cyclically assigned rows using the shared pivot row.
        for (size_t localRow = 0; localRow < local.size() / n; ++localRow) {
            const size_t row = localRow * ranks + rank;
            if (row <= i) continue;
            double sum = 0.0;
            for (size_t k = 0; k < i; ++k)
                sum += local[localRow * n + k] * pivot[k];
            local[localRow * n + i] = (local[localRow * n + i] - sum) / pivot[i];
        }
        if (rank == owner)
            for (size_t col = i + 1; col < n; ++col) pivot[col] = 0.0;
    }
    for (size_t localRow = 0; localRow < local.size() / n; ++localRow) {
        const size_t row = localRow * ranks + rank;
        for (size_t col = row + 1; col < n; ++col) local[localRow * n + col] = 0.0;
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

    if (n == 0 || n > static_cast<size_t>(INT_MAX) || n * n > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) printf("Invalid matrix size\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    const size_t baseRows = n / static_cast<size_t>(ranks);
    const size_t extraRows = n % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    std::vector<int> counts(ranks), displacements(ranks);
    size_t packedOffset = 0;
    for (int p = 0; p < ranks; ++p) {
        const size_t rows = n / static_cast<size_t>(ranks) + (static_cast<size_t>(p) < n % static_cast<size_t>(ranks) ? 1 : 0);
        counts[p] = static_cast<int>(rows * n);
        displacements[p] = static_cast<int>(packedOffset);
        packedOffset += rows * n;
    }
    std::vector<double> A, A_orig, packed;
    if (rank == 0) {
        A.resize(n * n);
        generatePositiveDefiniteMatrix(A, n);
        if (validate) A_orig = A;
        packed.resize(n * n);
        for (int p = 0; p < ranks; ++p) {
            size_t destination = static_cast<size_t>(displacements[p]);
            for (size_t row = static_cast<size_t>(p); row < n; row += static_cast<size_t>(ranks)) {
                std::copy_n(A.data() + row * n, n, packed.data() + destination);
                destination += n;
            }
        }
    }
    std::vector<double> local(localRows * n);
    MPI_Scatterv(rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const bool success = choleskyDecomposition(local, n, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<double> gathered;
    if (rank == 0) gathered.resize(n * n);
    MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int p = 0; p < ranks; ++p) {
            size_t source = static_cast<size_t>(displacements[p]);
            for (size_t row = static_cast<size_t>(p); row < n; row += static_cast<size_t>(ranks)) {
                std::copy_n(packed.data() + source, n, gathered.data() + row * n);
                source += n;
            }
        }
        A.swap(gathered);
        const long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(A, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A, A_orig, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
