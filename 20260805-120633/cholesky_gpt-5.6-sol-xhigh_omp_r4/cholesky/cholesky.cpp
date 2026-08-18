#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// The tile size is large enough to make each OpenMP task compute-bound, while
// keeping the three tiles used by an update in the private caches of a core.
constexpr size_t kTileSize = 96;

static int threadsForMatrix(const size_t n) {
    // Very small matrices do not contain enough independent tile work to pay
    // for launching every hardware thread. One thread per 64 rows tracks the
    // available task parallelism well and still reaches the runtime/user cap
    // as the problem grows.
    const size_t usefulThreads = std::max<size_t>(1, n / 64 + (n % 64 != 0));
    return static_cast<int>(std::min(usefulThreads,
                                     static_cast<size_t>(omp_get_max_threads())));
}

// Factor a diagonal tile. Updates from all preceding block columns have already
// been applied by the task dependency graph.
static bool factorDiagonalTile(double* const A, const size_t n,
                               const size_t begin, const size_t end) {
    for (size_t i = begin; i < end; ++i) {
        double* const rowI = A + i * n;
        for (size_t j = begin; j <= i; ++j) {
            const double* const rowJ = A + j * n;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t p = begin; p < j; ++p) {
                sum += rowI[p] * rowJ[p];
            }

            if (i == j) {
                const double value = rowI[j] - sum;
                if (value <= 0.0) {
                    std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                rowI[j] = std::sqrt(value);
            } else {
                rowI[j] = (rowI[j] - sum) / rowJ[j];
            }
        }
    }
    return true;
}

// Solve A(rowBegin:rowEnd, kBegin:kEnd) against the factored diagonal tile.
static void solveBlockColumn(double* const A, const size_t n,
                             const size_t rowBegin, const size_t rowEnd,
                             const size_t kBegin, const size_t kEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const rowI = A + i * n;
        for (size_t j = kBegin; j < kEnd; ++j) {
            const double* const rowJ = A + j * n;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t p = kBegin; p < j; ++p) {
                sum += rowI[p] * rowJ[p];
            }
            rowI[j] = (rowI[j] - sum) / rowJ[j];
        }
    }
}

// A(i,j) -= L(i,k) * L(j,k)^T. Only the lower half of a diagonal
// destination tile is stored; off-diagonal destination tiles are rectangular.
static void updateTrailingTile(double* const A, const size_t n,
                               const size_t rowBegin, const size_t rowEnd,
                               const size_t colBegin, const size_t colEnd,
                               const size_t kBegin, const size_t kEnd) {
    const bool diagonalTile = rowBegin == colBegin;
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const rowI = A + i * n;
        const size_t lastColumn = diagonalTile ? std::min(colEnd, i + 1) : colEnd;
        size_t j = colBegin;

        // Four independent accumulators hide vector-add latency and reuse the
        // row from the block column across four output entries.
        for (; j + 3 < lastColumn; j += 4) {
            const double* const rowJ0 = A + j * n;
            const double* const rowJ1 = rowJ0 + n;
            const double* const rowJ2 = rowJ1 + n;
            const double* const rowJ3 = rowJ2 + n;
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;
#pragma omp simd reduction(+ : sum0, sum1, sum2, sum3)
            for (size_t p = kBegin; p < kEnd; ++p) {
                const double value = rowI[p];
                sum0 += value * rowJ0[p];
                sum1 += value * rowJ1[p];
                sum2 += value * rowJ2[p];
                sum3 += value * rowJ3[p];
            }
            rowI[j] -= sum0;
            rowI[j + 1] -= sum1;
            rowI[j + 2] -= sum2;
            rowI[j + 3] -= sum3;
        }

        for (; j < lastColumn; ++j) {
            const double* const rowJ = A + j * n;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t p = kBegin; p < kEnd; ++p) {
                sum += rowI[p] * rowJ[p];
            }
            rowI[j] -= sum;
        }
    }
}

// Cache-blocked, right-looking Cholesky decomposition.  OpenMP task
// dependencies form the tiled Cholesky DAG, allowing work from later block
// columns to start as soon as its inputs are ready instead of imposing a
// global barrier after every phase.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    double* const data = A.data();
    const size_t tileCount = (n + kTileSize - 1) / kTileSize;
    const int teamSize = threadsForMatrix(n);

    // One byte per tile is sufficient: dependencies use addresses as tokens,
    // not the token values.  It also avoids non-contiguous row-major tile
    // sections in OpenMP depend clauses.
    std::vector<unsigned char> dependencyTokens(tileCount * tileCount);
    unsigned char* const tokens = dependencyTokens.data();
    static_cast<void>(tokens); // Referenced by OpenMP depend clauses.
    std::atomic_bool success{true};

#pragma omp parallel num_threads(teamSize) default(none) shared(data, n, tileCount, tokens, success)
    {
#pragma omp single
        {
            for (size_t kTile = 0; kTile < tileCount; ++kTile) {
                const size_t kBegin = kTile * kTileSize;
                const size_t kEnd = std::min(n, kBegin + kTileSize);
                const size_t diagonalToken = kTile * tileCount + kTile;

#pragma omp task firstprivate(kBegin, kEnd, diagonalToken) depend(inout : tokens[diagonalToken]) shared(success) priority(2)
                {
                    if (success.load() &&
                        !factorDiagonalTile(data, n, kBegin, kEnd)) {
                        success.store(false);
                    }
                }

                for (size_t iTile = kTile + 1; iTile < tileCount; ++iTile) {
                    const size_t rowBegin = iTile * kTileSize;
                    const size_t rowEnd = std::min(n, rowBegin + kTileSize);
                    const size_t columnToken = iTile * tileCount + kTile;

#pragma omp task firstprivate(rowBegin, rowEnd, kBegin, kEnd, diagonalToken, columnToken) depend(in : tokens[diagonalToken]) depend(inout : tokens[columnToken]) shared(success) priority(1)
                    {
                        if (success.load()) {
                            solveBlockColumn(data, n, rowBegin, rowEnd, kBegin, kEnd);
                        }
                    }
                }

                for (size_t iTile = kTile + 1; iTile < tileCount; ++iTile) {
                    const size_t rowBegin = iTile * kTileSize;
                    const size_t rowEnd = std::min(n, rowBegin + kTileSize);
                    const size_t rowPanelToken = iTile * tileCount + kTile;

                    for (size_t jTile = kTile + 1; jTile <= iTile; ++jTile) {
                        const size_t colBegin = jTile * kTileSize;
                        const size_t colEnd = std::min(n, colBegin + kTileSize);
                        const size_t colPanelToken = jTile * tileCount + kTile;
                        const size_t outputToken = iTile * tileCount + jTile;

                        if (iTile == jTile) {
#pragma omp task firstprivate(rowBegin, rowEnd, colBegin, colEnd, kBegin, kEnd, rowPanelToken, outputToken) depend(in : tokens[rowPanelToken]) depend(inout : tokens[outputToken]) shared(success) priority(2)
                            {
                                if (success.load()) {
                                    updateTrailingTile(data, n, rowBegin, rowEnd,
                                                       colBegin, colEnd, kBegin, kEnd);
                                }
                            }
                        } else {
#pragma omp task firstprivate(rowBegin, rowEnd, colBegin, colEnd, kBegin, kEnd, rowPanelToken, colPanelToken, outputToken) depend(in : tokens[rowPanelToken], tokens[colPanelToken]) depend(inout : tokens[outputToken]) shared(success)
                            {
                                if (success.load()) {
                                    updateTrailingTile(data, n, rowBegin, rowEnd,
                                                       colBegin, colEnd, kBegin, kEnd);
                                }
                            }
                        }
                    }
                }
            }
#pragma omp taskwait
        }
    }

    if (!success.load()) {
        return false;
    }

    // The factorization only consumes the lower triangle. Clear the upper
    // triangle once, in parallel, after all factorization tasks complete.
#pragma omp parallel for num_threads(teamSize) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
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
    const int teamSize = threadsForMatrix(n);
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute the lower triangle of A = B * B^T. The dot products are
    // independent, and symmetry avoids doing every one twice.
#pragma omp parallel for num_threads(teamSize) schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness.
#pragma omp parallel for num_threads(teamSize) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    const int teamSize = threadsForMatrix(n);
    
    // Compute the lower triangle of L * L^T and mirror it.
#pragma omp parallel for num_threads(teamSize) schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k <= j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
            reconstructed[j * n + i] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
#pragma omp parallel for num_threads(teamSize) reduction(max : maxError, relError) schedule(static)
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
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = elapsedSeconds > 0.0 ? ops / elapsedSeconds / 1e9 : 0.0;
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
