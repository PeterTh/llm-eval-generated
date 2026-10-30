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

// Distributed Cholesky decomposition using MPI (row-block distribution)
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = (int)((n + size - 1) / size);
    int localStart = rank * chunk;
    int localEnd = std::min((size_t)(localStart + chunk), n);
    int localRows = localEnd - localStart;
    if (localRows < 0) localRows = 0;

    std::vector<double> L((size_t)localRows * n, 0.0);
    // copy local rows from A
    for (int i = 0; i < localRows; ++i) {
        for (size_t j = 0; j < n; ++j) {
            L[(size_t)i * n + j] = A[(size_t)(localStart + i) * n + j];
        }
    }

    std::vector<double> pivotRow(n);

    for (size_t k = 0; k < n; ++k) {
        int owner = (int)(k / chunk);
        if (owner >= size) owner = size - 1;
        // ensure owner actually has row k
        while (owner > 0 && owner * chunk > (int)k) --owner;

        if (rank == owner) {
            int li = (int)k - localStart;
            double sum = 0.0;
            for (size_t t = 0; t < k; ++t) {
                double v = L[(size_t)li * n + t];
                sum += v * v;
            }
            double val = L[(size_t)li * n + k] - sum;
            if (val <= 0.0) {
                 if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
                int ok = 0;
                MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
                return false;
            }
            L[(size_t)li * n + k] = std::sqrt(val);
            for (size_t j = 0; j < n; ++j) pivotRow[j] = L[(size_t)li * n + j];
        }
        int ok = 1;
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) return false;
        MPI_Bcast(pivotRow.data(), (int)n, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // update local rows i > k
        for (int i = 0; i < localRows; ++i) {
            int gi = localStart + i;
            if (gi <= (int)k) continue;
            double sum = 0.0;
            for (size_t t = 0; t < k; ++t) {
                sum += L[(size_t)i * n + t] * pivotRow[t];
            }
            L[(size_t)i * n + k] = (L[(size_t)i * n + k] - sum) / pivotRow[k];
        }
    }

    // zero upper triangular
    for (int i = 0; i < localRows; ++i) {
        for (size_t j = (size_t)(localStart + i) + 1; j < n; ++j) {
            L[(size_t)i * n + j] = 0.0;
        }
    }

    // gather
    std::vector<int> counts(size), displs(size);
    // compute local row counts for all ranks
    for (int r = 0; r < size; ++r) {
        int rs = r * chunk;
        int re = std::min(rs + chunk, (int)n);
        counts[r] = (re - rs) * (int)n;
        if (counts[r] < 0) counts[r] = 0;
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + counts[r-1];

    MPI_Gatherv(L.data(), localRows * (int)n, MPI_DOUBLE,
                A.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // broadcast result to all ranks (for validation/print on rank0 only, but keep consistency)
    MPI_Bcast(A.data(), (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
             if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
     if (rank == 0) printf("Cholesky Decomposition Benchmark\n");
     if (rank == 0) printf("Matrix size: %zu x %zu\n", n, n);
     if (rank == 0) printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
     if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
     if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
         if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    
     if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
     if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    // Validation
    if (validate && rank == 0) {
         if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
             if (rank == 0) printf("Validation: PASSED\n");
            return 0;
        } else {
             if (rank == 0) printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    if (rank == 0) {}
    MPI_Finalize();
    return 0;
}
