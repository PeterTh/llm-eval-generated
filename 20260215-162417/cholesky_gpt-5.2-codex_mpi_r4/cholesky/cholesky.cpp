#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI row-distributed Cholesky decomposition (unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
struct RowPartition {
    size_t start;
    size_t count;
};

RowPartition getRowPartition(const size_t n, const int size, const int rank) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t count = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    return {start, count};
}

int ownerOfRow(const size_t row, const size_t n, const int size) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t cutoff = (base + 1) * rem;
    if (row < cutoff) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - cutoff) / base);
}

bool choleskyDecompositionMPI(std::vector<double>& localA, const size_t n, const RowPartition& part, MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<double> rowBuffer(n, 0.0);
    int globalOk = 1;

    for (size_t k = 0; k < n; ++k) {
        const int owner = ownerOfRow(k, n, size);
        int localOk = 1;

        if (rank == owner) {
            const size_t localIdx = k - part.start;
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                const double val = localA[localIdx * n + p];
                sum += val * val;
                rowBuffer[p] = val;
            }
            const double diagVal = localA[localIdx * n + k] - sum;
            if (diagVal <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
                localOk = 0;
            } else {
                const double diag = sqrt(diagVal);
                localA[localIdx * n + k] = diag;
                rowBuffer[k] = diag;
            }
        }

        MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, comm);
        if (globalOk == 0) {
            return false;
        }

        MPI_Bcast(rowBuffer.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner, comm);

        if (part.count == 0) {
            continue;
        }

        for (size_t localIdx = 0; localIdx < part.count; ++localIdx) {
            const size_t i = part.start + localIdx;
            if (i <= k) {
                continue;
            }
            double sum = 0.0;
            const size_t rowOffset = localIdx * n;
            for (size_t p = 0; p < k; ++p) {
                sum += localA[rowOffset + p] * rowBuffer[p];
            }
            localA[rowOffset + k] = (localA[rowOffset + k] - sum) / rowBuffer[k];
        }
    }

    for (size_t localIdx = 0; localIdx < part.count; ++localIdx) {
        const size_t i = part.start + localIdx;
        const size_t rowOffset = localIdx * n;
        for (size_t j = i + 1; j < n; ++j) {
            localA[rowOffset + j] = 0.0;
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
    bool showHelp = false;
    int parseError = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseError = 1;
                break;
            }
        }
    }

    uint64_t n64 = static_cast<uint64_t>(n);
    int flags[4] = {parseError, showHelp, validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 4, MPI_INT, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n64);
    parseError = flags[0];
    showHelp = flags[1] != 0;
    validate = flags[2] != 0;
    printResults = flags[3] != 0;

    if (parseError || showHelp) {
        if (rank == 0 && showHelp && !parseError) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A;
    std::vector<double> A_orig;
    if (rank == 0) {
        A.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        if (validate) {
            A_orig = A;
        }
    }

    const RowPartition part = getRowPartition(n, size, rank);
    std::vector<double> localA(part.count * n);

    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        size_t offset = 0;
        for (int r = 0; r < size; ++r) {
            const RowPartition rp = getRowPartition(n, size, r);
            counts[r] = static_cast<int>(rp.count * n);
            displs[r] = static_cast<int>(offset * n);
            offset += rp.count;
        }
    }

    MPI_Scatterv(rank == 0 ? A.data() : nullptr,
                 rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_DOUBLE,
                 localA.data(),
                 static_cast<int>(part.count * n),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const bool success = choleskyDecompositionMPI(localA, n, part, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    double elapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> L;
    if (printResults || validate) {
        if (rank == 0) {
            L.resize(n * n);
        }
        MPI_Gatherv(localA.data(),
                    static_cast<int>(part.count * n),
                    MPI_DOUBLE,
                    rank == 0 ? L.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
    }

    if (rank == 0 && printResults) {
        print_results(L, "CholeskyL");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = validateCholesky(L, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        }
        printf("Validation: FAILED\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
