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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Block row distribution
    std::vector<size_t> rowStart(size), rowEnd(size);
    size_t base = n / size, rem = n % size, cur = 0;
    for (int r = 0; r < size; ++r) {
        rowStart[r] = cur;
        size_t cnt = base + (r < (int)rem ? 1 : 0);
        rowEnd[r] = cur + cnt;
        cur += cnt;
    }
    size_t myStart = rowStart[rank], myEnd = rowEnd[rank];

    auto ownerOf = [&](size_t row) {
        for (int r = 0; r < size; ++r)
            if (row >= rowStart[r] && row < rowEnd[r]) return r;
        return size - 1;
    };

    for (size_t j = 0; j < n; ++j) {
        int owner = ownerOf(j);
        if (rank == owner) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += A[j * n + k] * A[j * n + k];
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                int ok = 0;
                MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
                return false;
            }
            A[j * n + j] = sqrt(val);
            int ok = 1;
            MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        } else {
            int ok;
            MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
            if (!ok) return false;
        }
        // Broadcast pivot row
        MPI_Bcast(&A[j * n], (int)n, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Update owned rows
        for (size_t i = myStart; i < myEnd; ++i) {
            if (i <= j) continue;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += A[i * n + k] * A[j * n + k];
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
    }

    // Zero upper triangular part for owned rows
    for (size_t i = myStart; i < myEnd; ++i)
        for (size_t j = i + 1; j < n; ++j)
            A[i * n + j] = 0.0;

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
    int rank, size;
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A;
    }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int ok = success ? 1 : 0;
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // broadcast success from any? use max
    int gok; MPI_Allreduce(&ok, &gok, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (!gok) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full matrix to rank 0
    std::vector<double> fullA;
    if (rank == 0) fullA.resize(n * n);
    // row counts
    std::vector<int> counts(size), displs(size);
    {
        size_t cur = 0; size_t base = n / size, rem = n % size;
        for (int r = 0; r < size; ++r) {
            size_t cnt = base + (r < (int)rem ? 1 : 0);
            counts[r] = (int)(cnt * n);
            displs[r] = (int)(cur * n);
            cur += cnt;
        }
    }
    std::vector<double> resultA;
    if (rank == 0) resultA.resize(n * n);
    std::vector<int> scnt(size), sdisp(size);
    {
        size_t cur = 0; size_t base2 = n / size, rem2 = n % size;
        for (int r = 0; r < size; ++r) {
            size_t cnt = base2 + (r < (int)rem2 ? 1 : 0);
            scnt[r] = (int)(cnt * n);
            sdisp[r] = (int)(cur * n);
            cur += cnt;
        }
    }
    size_t myRowCount = n / size + (rank < (int)(n % size) ? 1 : 0);
    size_t myStart = (size_t)rank * (n / size) + std::min((size_t)rank, n % size);
    MPI_Gatherv(&A[myStart * n], (int)(myRowCount * n), MPI_DOUBLE, rank == 0 ? resultA.data() : nullptr, scnt.data(), sdisp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0 && printResults) {
        print_results(resultA, "CholeskyL");
    }

    if (validate) {
        // need A_orig on rank0 too; generate already same, so rank0 has it
        std::vector<double> origA;
        if (rank == 0) {
            origA = A_orig; // rank0 generated same
        }
        // Actually A_orig only meaningful on rank0; but all ranks generated same A originally, so rank0's A_orig valid.
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateCholesky(resultA, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
