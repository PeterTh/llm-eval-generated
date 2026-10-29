#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>
#include "../common/results_output.hpp"

// Cyclic blocks balance both matrix generation and the shrinking trailing work.
static constexpr size_t block = 32;

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
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const long value = strtol(argv[++i], &end, 10);
            if (*end || value < 0 || value > INT_MAX / static_cast<long>(block)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", argv[i]);
                status = 1;
                break;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            status = 1;
            break;
        }
    }
    if (status) { MPI_Finalize(); return status; }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    // Replicate the deterministic input generator, while constructing only owned rows.
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& x : B) x = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    std::vector<size_t> rows;
    for (size_t i = 0; i < n; ++i)
        if (static_cast<int>((i / block) % ranks) == rank) rows.push_back(i);
    std::vector<double> A(rows.size() * n);
    for (size_t r = 0; r < rows.size(); ++r) {
        const size_t i = rows[r];
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i*n+k] * B[j*n+k];
            A[r*n+j] = sum + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
    B.clear(); B.shrink_to_fit();
    std::vector<double> original;
    if (validate) original = A;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    std::vector<double> panel(block * n);
    for (size_t first = 0; first < n; first += block) {
        const size_t count = std::min(block, n - first);
        const int owner = static_cast<int>((first / block) % ranks);
        int failed = 0;
        if (rank == owner) {
            const size_t local = (first / block / ranks) * block;
            for (size_t t = 0; t < count; ++t) {
                double* row = A.data() + (local + t) * n;
                const size_t i = first + t;
                for (size_t j = first; j <= i; ++j) {
                    const double* other = A.data() + (local + j - first) * n;
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k) sum += row[k] * other[k];
                    if (i == j) {
                        const double val = row[j] - sum;
                        if (val <= 0.0) { failed = 1; break; }
                        row[j] = sqrt(val);
                    } else row[j] = (row[j] - sum) / other[j];
                }
                if (failed) { printf("Error: Matrix is not positive definite at diagonal element %zu\n", i); break; }
                std::fill(row + i + 1, row + n, 0.0);
                std::copy(row, row + n, panel.data() + t * n);
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) { status = 1; break; }
        MPI_Bcast(panel.data(), static_cast<int>(count * n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        for (size_t r = 0; r < rows.size(); ++r) {
            if (rows[r] < first + count) continue;
            double* row = A.data() + r * n;
            for (size_t t = 0; t < count; ++t) {
                const size_t j = first + t;
                const double* other = panel.data() + t * n;
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) sum += row[k] * other[k];
                row[j] = (row[j] - sum) / other[j];
            }
        }
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (status) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = (double)n * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0);
    }
    if (validate || printResults) {
        // Only the reporting rank needs a full matrix.
        std::vector<double> full, fullOriginal;
        if (rank == 0) {
            full.resize(n*n);
            if (validate) fullOriginal.resize(n*n);
        }
        for (size_t first = 0; first < n; first += block) {
            const size_t count = std::min(block, n-first);
            const int owner = static_cast<int>((first / block) % ranks);
            const size_t local = (first / block / ranks) * block;
            const int elements = static_cast<int>(count * n);
            if (owner == 0 && rank == 0) {
                std::copy_n(A.data() + local*n, elements, full.data() + first*n);
                if (validate) std::copy_n(original.data() + local*n, elements, fullOriginal.data() + first*n);
            } else if (rank == owner) {
                MPI_Send(A.data() + local*n, elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                if (validate) MPI_Send(original.data() + local*n, elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            } else if (rank == 0) {
                MPI_Recv(full.data() + first*n, elements, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (validate) MPI_Recv(fullOriginal.data() + first*n, elements, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (rank == 0) {
            if (printResults) print_results(full, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                status = !validateCholesky(full, fullOriginal, n);
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
