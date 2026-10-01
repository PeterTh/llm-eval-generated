#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are cyclically assigned to ranks. Each local row retains the original
// row-major layout, so the trailing update touches contiguous matrix entries.
static size_t localRowCount(size_t n, int rank, int ranks) {
    return rank >= static_cast<int>(n) ? 0 :
           1 + (n - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(ranks);
}

static void generateLocalMatrix(std::vector<double>& A, size_t n,
                                int rank, int ranks) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    const size_t rows = localRowCount(n, rank, ranks);
    for (size_t local = 0; local < rows; ++local) {
        const size_t i = static_cast<size_t>(rank) + local * ranks;
        const double* bi = B.data() + i * n;
        double* ai = A.data() + local * n;
        for (size_t j = 0; j < n; ++j) {
            const double* bj = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += bi[k] * bj[k];
            }
            ai[j] = sum + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
}

// Gather cyclic rows in rank order, then restore global row order on root.
static std::vector<double> gatherMatrix(const std::vector<double>& local,
                                        size_t n, int rank, int ranks) {
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> packed;
    if (rank == 0) {
        counts.resize(ranks);
        displacements.resize(ranks);
        size_t offset = 0;
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(localRowCount(n, r, ranks) * n);
            displacements[r] = static_cast<int>(offset);
            offset += counts[r];
        }
        packed.resize(n * n);
    }
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    if (rank != 0) return {};

    std::vector<double> global(n * n);
    for (int r = 0; r < ranks; ++r) {
        for (size_t localRow = 0; localRow < localRowCount(n, r, ranks); ++localRow) {
            const size_t row = static_cast<size_t>(r) + localRow * ranks;
            std::copy_n(packed.data() + displacements[r] + localRow * n,
                        n, global.data() + row * n);
        }
    }
    return global;
}

// Blocked upper-triangular Cholesky. Within a panel, pivot rows are broadcast
// and only panel rows are updated. Trailing rows are updated once per panel,
// reusing each matrix entry for all pivots in that panel.
static bool choleskyDecomposition(std::vector<double>& A, size_t n,
                                  int rank, int ranks) {
    constexpr size_t blockSize = 32;
    const size_t rows = localRowCount(n, rank, ranks);
    std::vector<double> pivot(n);
    std::vector<double> panel(n * blockSize);

    for (size_t begin = 0; begin < n; begin += blockSize) {
        const size_t end = std::min(n, begin + blockSize);
        const size_t width = end - begin;
        for (size_t k = begin; k < end; ++k) {
            const int owner = static_cast<int>(k % ranks);
            if (rank == owner) {
                double* row = A.data() + (k / ranks) * n;
                pivot[k] = row[k];
                if (pivot[k] > 0.0) {
                    pivot[k] = std::sqrt(pivot[k]);
                    row[k] = pivot[k];
                    const double inverse = 1.0 / pivot[k];
                    for (size_t j = k + 1; j < n; ++j) {
                        pivot[j] = row[j] * inverse;
                        row[j] = pivot[j];
                    }
                }
            }
            MPI_Bcast(pivot.data() + k, static_cast<int>(n - k), MPI_DOUBLE,
                      owner, MPI_COMM_WORLD);
            if (!(pivot[k] > 0.0) || !std::isfinite(pivot[k])) {
                if (rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
                }
                return false;
            }
            for (size_t j = k; j < n; ++j) {
                panel[j * blockSize + (k - begin)] = pivot[j];
            }
            for (size_t local = 0; local < rows; ++local) {
                const size_t i = static_cast<size_t>(rank) + local * ranks;
                if (i <= k || i >= end) continue;
                double* row = A.data() + local * n;
                const double scale = pivot[i];
                for (size_t j = i; j < n; ++j) {
                    row[j] -= scale * pivot[j];
                }
            }
        }
        for (size_t local = 0; local < rows; ++local) {
            const size_t i = static_cast<size_t>(rank) + local * ranks;
            if (i < end) continue;
            double* row = A.data() + local * n;
            const double* left = panel.data() + i * blockSize;
            for (size_t j = i; j < n; ++j) {
                const double* right = panel.data() + j * blockSize;
                double update = 0.0;
                for (size_t t = 0; t < width; ++t) {
                    update += left[t] * right[t];
                }
                row[j] -= update;
            }
        }
    }
    return true;
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
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed > INT_MAX ||
                parsed * parsed > INT_MAX) {
                if (rank == 0) printf("Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(parsed);
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

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> A(localRowCount(n, rank, ranks) * n);
    generateLocalMatrix(A, n, rank, ranks);
    std::vector<double> A_orig;
    if (validate) A_orig = gatherMatrix(A, n, rank, ranks);

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, n, rank, ranks);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0);
    }

    int valid = 1;
    if (printResults || validate) {
        // The factorization stores U. Convert its transpose to the original
        // lower-triangular output format before printing or validating.
        std::vector<double> upper = gatherMatrix(A, n, rank, ranks);
        if (rank == 0) {
            std::vector<double> lower(n * n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    lower[i * n + j] = upper[j * n + i];
                }
            }
            if (printResults) print_results(lower, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                valid = validateCholesky(lower, A_orig, n) ? 1 : 0;
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }
    if (validate) MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
