#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static void computeRowDistribution(const size_t n, const int commSize, std::vector<int>& rowStarts,
                                   std::vector<int>& rowCounts) {
    rowStarts.resize(commSize);
    rowCounts.resize(commSize);
    const size_t base = n / static_cast<size_t>(commSize);
    const size_t rem = n % static_cast<size_t>(commSize);
    size_t start = 0;
    for (int r = 0; r < commSize; ++r) {
        const size_t cnt = base + (static_cast<size_t>(r) < rem ? 1u : 0u);
        rowStarts[r] = static_cast<int>(start);
        rowCounts[r] = static_cast<int>(cnt);
        start += cnt;
    }
}

static int ownerOfRow(const size_t row, const std::vector<int>& rowStarts, const std::vector<int>& rowCounts) {
    const int commSize = static_cast<int>(rowStarts.size());
    const int rrow = static_cast<int>(row);
    for (int r = 0; r < commSize; ++r) {
        const int s = rowStarts[r];
        const int e = s + rowCounts[r];
        if (rrow >= s && rrow < e) return r;
    }
    return commSize - 1;
}

static bool choleskyDecompositionMPI(std::vector<double>& A_local, const size_t n, const int rank,
                                    const std::vector<int>& rowStarts, const std::vector<int>& rowCounts,
                                    MPI_Comm comm) {
    const int myStart = rowStarts[rank];
    const int myRows = rowCounts[rank];

    std::vector<double> rowbuf(n);

    for (size_t j = 0; j < n; ++j) {
        const int owner = ownerOfRow(j, rowStarts, rowCounts);

        int ok = 1;
        if (rank == owner) {
            const int lj = static_cast<int>(j) - myStart;
            // Compute diagonal element L(j,j)
            double sum = 0.0;
            const double* Aj = &A_local[static_cast<size_t>(lj) * n];
            for (size_t k = 0; k < j; ++k) {
                const double v = Aj[k];
                sum += v * v;
            }
            const double val = Aj[j] - sum;
            if (val <= 0.0) {
                ok = 0;
            } else {
                A_local[static_cast<size_t>(lj) * n + j] = std::sqrt(val);
            }

            // Pack row j (lower part) for broadcast
            double* buf = rowbuf.data();
            for (size_t k = 0; k <= j; ++k) {
                buf[k] = A_local[static_cast<size_t>(lj) * n + k];
            }
        }

        MPI_Bcast(&ok, 1, MPI_INT, owner, comm);
        if (!ok) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        MPI_Bcast(rowbuf.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, comm);
        const double diag = rowbuf[j];

        // Update column j for my local rows i > j
        for (int li = 0; li < myRows; ++li) {
            const size_t i = static_cast<size_t>(myStart + li);
            if (i <= j) continue;

            double* Ai = &A_local[static_cast<size_t>(li) * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += Ai[k] * rowbuf[k];
            }
            Ai[j] = (Ai[j] - sum) / diag;
        }
    }

    // Zero out upper triangular part (local rows only)
    for (int li = 0; li < myRows; ++li) {
        const size_t i = static_cast<size_t>(myStart + li);
        double* Ai = &A_local[static_cast<size_t>(li) * n];
        for (size_t j = i + 1; j < n; ++j) {
            Ai[j] = 0.0;
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
    int commSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    size_t n = 512;
    int validate = 0;
    int printResults = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(std::atoi(argv[++i]));
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast options
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute distribution
    std::vector<int> rowStarts, rowCounts;
    computeRowDistribution(n, commSize, rowStarts, rowCounts);

    const int myRows = rowCounts[rank];

    std::vector<int> sendCounts(commSize), sendDispls(commSize);
    for (int r = 0; r < commSize; ++r) {
        sendCounts[r] = rowCounts[r] * static_cast<int>(n);
        sendDispls[r] = rowStarts[r] * static_cast<int>(n);
    }

    // Rank 0 generates the full matrix (to preserve exact original semantics)
    std::vector<double> A_full;
    std::vector<double> A_orig_full;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("MPI ranks: %d\n", commSize);
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

        A_full.resize(n * n);
        std::printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig_full = A_full;
        }
    }

    // Scatter rows
    std::vector<double> A_local(static_cast<size_t>(myRows) * n);
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                 A_local.data(), static_cast<int>(static_cast<size_t>(myRows) * n), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    // Compute Cholesky decomposition
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::high_resolution_clock::now();

    const bool success = choleskyDecompositionMPI(A_local, n, rank, rowStarts, rowCounts, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const long long localUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    long long maxUs = 0;
    MPI_Reduce(&localUs, &maxUs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    // Gather result
    if (rank == 0) {
        if (A_full.empty()) A_full.resize(n * n);
    }
    MPI_Gatherv(A_local.data(), static_cast<int>(static_cast<size_t>(myRows) * n), MPI_DOUBLE,
                rank == 0 ? A_full.data() : nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        const long long maxMs = (maxUs + 999) / 1000; // ceil to avoid 0 ms
        std::printf("Computation time: %lld ms\n", maxMs);
        const double ops = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n) / 3.0;
        const double secs = std::max(1e-6, static_cast<double>(maxUs) * 1e-6);
        const double gflops = ops / secs / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A_full, "CholeskyL");
        }

        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(A_full, A_orig_full, n);
            if (valid) {
                std::printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            }
            std::printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
