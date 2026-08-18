#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition.  The diagonal block has the
// usual data dependency, while the panel solve and all trailing tiles are
// independent and are shared between OpenMP threads.
namespace {

constexpr size_t kCholeskyBlockSize = 64;
constexpr size_t kUpdateTileSize = 64;

// The first trailing update provides the most independent tiles.  Keeping at
// least two tiles per worker avoids the substantial overhead of starting a
// large team for small matrices, while omp_get_max_threads() preserves an
// explicit lower OMP_NUM_THREADS setting.
int parallelThreadCount(const size_t n) {
    const size_t blockCount = (n + kCholeskyBlockSize - 1) / kCholeskyBlockSize;
    if (blockCount < 2) {
        return 1;
    }
    const size_t usefulThreads = std::max<size_t>(1, blockCount * (blockCount - 1) / 4);
    return static_cast<int>(std::min(usefulThreads,
                                     static_cast<size_t>(omp_get_max_threads())));
}

bool factorDiagonalBlock(std::vector<double>& A, const size_t n,
                         const size_t begin, const size_t end) {
    for (size_t i = begin; i < end; ++i) {
        double* const rowI = A.data() + i * n;

        for (size_t j = begin; j <= i; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t k = begin; k < j; ++k) {
                sum += rowI[k] * A[j * n + k];
            }

            if (i == j) {
                const double val = rowI[j] - sum;
                if (val <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                rowI[j] = sqrt(val);
            } else {
                rowI[j] = (rowI[j] - sum) / A[j * n + j];
            }
        }
    }

    return true;
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    bool isPositiveDefinite = true;
    const int threadCount = parallelThreadCount(n);

#pragma omp parallel default(none) num_threads(threadCount) shared(A, n, isPositiveDefinite)
    {
        for (size_t blockBegin = 0; blockBegin < n; blockBegin += kCholeskyBlockSize) {
            const size_t blockEnd = std::min(blockBegin + kCholeskyBlockSize, n);

            // This short, dependency-bound step prepares the next panel.
#pragma omp single
            {
                if (isPositiveDefinite) {
                    isPositiveDefinite = factorDiagonalBlock(A, n, blockBegin, blockEnd);
                }
            }

            if (isPositiveDefinite) {
                // Solve L_ik * L_kk^T = A_ik for every row below the diagonal block.
#pragma omp for schedule(static)
                for (size_t i = blockEnd; i < n; ++i) {
                    double* const rowI = A.data() + i * n;
                    for (size_t j = blockBegin; j < blockEnd; ++j) {
                        double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                        for (size_t k = blockBegin; k < j; ++k) {
                            sum += rowI[k] * A[j * n + k];
                        }
                        rowI[j] = (rowI[j] - sum) / A[j * n + j];
                    }
                }

                // Rank-k update of the lower triangular trailing matrix.  Tiling keeps
                // the working rows hot in cache, and each tile writes a disjoint region.
                const size_t trailingSize = n - blockEnd;
                const size_t tileCount = (trailingSize + kUpdateTileSize - 1) / kUpdateTileSize;
#pragma omp for schedule(dynamic, 1)
                for (size_t tile = 0; tile < tileCount * tileCount; ++tile) {
                    const size_t rowTile = tile / tileCount;
                    const size_t columnTile = tile % tileCount;
                    if (columnTile > rowTile) {
                        continue;
                    }

                    const size_t rowBegin = blockEnd + rowTile * kUpdateTileSize;
                    const size_t rowEnd = std::min(rowBegin + kUpdateTileSize, n);
                    const size_t columnBegin = blockEnd + columnTile * kUpdateTileSize;
                    const size_t columnEnd = std::min(columnBegin + kUpdateTileSize, n);

                    for (size_t i = rowBegin; i < rowEnd; ++i) {
                        double* const rowI = A.data() + i * n;
                        const size_t endColumn = std::min(columnEnd, i + 1);
                        for (size_t j = columnBegin; j < endColumn; ++j) {
                            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                            for (size_t k = blockBegin; k < blockEnd; ++k) {
                                sum += rowI[k] * A[j * n + k];
                            }
                            rowI[j] -= sum;
                        }
                    }
                }
            }
        }
    }

    if (!isPositiveDefinite) {
        return false;
    }

    // The algorithm only touches the lower triangle; preserve the original API's
    // zeroed upper triangle without adding work to the timed critical path per block.
#pragma omp parallel for num_threads(parallelThreadCount(n)) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
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
    
    // Compute A = B * B^T.  Each result element is independent.
#pragma omp parallel for num_threads(parallelThreadCount(n)) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
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
    
    // Compute L * L^T.  Rows of the reconstruction are independent.
#pragma omp parallel for num_threads(parallelThreadCount(n)) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
#pragma omp parallel for num_threads(parallelThreadCount(n)) reduction(max : maxError, relError) schedule(static)
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
