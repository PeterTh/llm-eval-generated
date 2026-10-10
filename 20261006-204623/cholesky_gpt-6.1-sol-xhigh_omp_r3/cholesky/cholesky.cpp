#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr size_t tileSize = 64;

// Small matrices cannot supply enough work for a full machine-sized team.
int rowWorkers(size_t n) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                            std::max<size_t>(1, n / 16)));
}

// The upper triangle temporarily holds the transpose of completed panels.
// This lets the update kernel vectorize over contiguous columns without packing
// another copy of the matrix. Panel tasks also write their transposed tiles.
size_t factorDiagonal(double* matrix, size_t n, size_t first, size_t last) {
    for (size_t i = first; i < last; ++i) {
        double* row = matrix + i * n;
        for (size_t j = first; j <= i; ++j) {
            const double* other = matrix + j * n;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = first; k < j; ++k) {
                sum += row[k] * other[k];
            }
            const double value = row[j] - sum;
            if (i == j) {
                if (value <= 0.0) {
                    return j;
                }
                row[j] = std::sqrt(value);
            } else {
                row[j] = value / other[j];
                matrix[j * n + i] = row[j];
            }
        }
    }
    return n;
}

// Solve with contiguous row updates, then publish the transposed panel.
void solvePanel(double* matrix, size_t n, size_t rowFirst, size_t rowLast,
                size_t first, size_t last) {
    for (size_t i = rowFirst; i < rowLast; ++i) {
        double* row = matrix + i * n;
        for (size_t j = first; j < last; ++j) {
            const double* diagonalRow = matrix + j * n;
            const double value = row[j] / diagonalRow[j];
            row[j] = value;
            #pragma omp simd
            for (size_t k = j + 1; k < last; ++k) {
                row[k] -= value * diagonalRow[k];
            }
        }
    }
    for (size_t j = first; j < last; ++j) {
        #pragma omp simd
        for (size_t i = rowFirst; i < rowLast; ++i) {
            matrix[j * n + i] = matrix[i * n + j];
        }
    }
}

// A 4 x 8 register block reuses each panel load across several output entries.
// Only tiles on/below the diagonal are updated; unused entries above the
// diagonal inside a diagonal tile are harmless and simplify this hot kernel.
void updateTile(double* matrix, size_t n, size_t rowFirst, size_t rowLast,
                size_t colFirst, size_t colLast, size_t first, size_t last) {
    // Copy the transposed panel once per tile, avoiding matrix-stride cache
    // conflicts when the same panel is reused by all the register blocks.
    alignas(64) double packed[tileSize * tileSize];
    for (size_t k = first; k < last; ++k) {
        std::copy(matrix + k * n + colFirst, matrix + k * n + colLast,
                  packed + (k - first) * tileSize);
    }
    size_t i = rowFirst;
    for (; i + 4 <= rowLast; i += 4) {
        double* row0 = matrix + i * n;
        double* row1 = row0 + n;
        double* row2 = row1 + n;
        double* row3 = row2 + n;
        size_t j = colFirst;
        for (; j + 8 <= colLast; j += 8) {
            double sum0[8] = {}, sum1[8] = {}, sum2[8] = {}, sum3[8] = {};
            for (size_t k = first; k < last; ++k) {
                const double* transposed = packed + (k - first) * tileSize + j - colFirst;
                const double a0 = row0[k], a1 = row1[k];
                const double a2 = row2[k], a3 = row3[k];
                for (size_t c = 0; c < 8; ++c) {
                    const double b = transposed[c];
                    sum0[c] += a0 * b;
                    sum1[c] += a1 * b;
                    sum2[c] += a2 * b;
                    sum3[c] += a3 * b;
                }
            }
            #pragma omp simd
            for (size_t c = 0; c < 8; ++c) {
                row0[j + c] -= sum0[c];
                row1[j + c] -= sum1[c];
                row2[j + c] -= sum2[c];
                row3[j + c] -= sum3[c];
            }
        }
        for (; j < colLast; ++j) {
            const double* other = matrix + j * n;
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            #pragma omp simd reduction(+:sum0,sum1,sum2,sum3)
            for (size_t k = first; k < last; ++k) {
                sum0 += row0[k] * other[k];
                sum1 += row1[k] * other[k];
                sum2 += row2[k] * other[k];
                sum3 += row3[k] * other[k];
            }
            row0[j] -= sum0;
            row1[j] -= sum1;
            row2[j] -= sum2;
            row3[j] -= sum3;
        }
    }
    for (; i < rowLast; ++i) {
        double* row = matrix + i * n;
        for (size_t k = first; k < last; ++k) {
            const double a = row[k];
            const double* transposed = matrix + k * n;
            #pragma omp simd
            for (size_t j = colFirst; j < colLast; ++j) {
                row[j] -= a * transposed[j];
            }
        }
    }
}

} // namespace

// Tiled, right-looking Cholesky. Dependencies are keyed by each lower tile's
// first element; all tasks are siblings generated by the same single region.
// There is no global barrier between panels, so the next diagonal can start
// while independent trailing updates are still running.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* matrix = A.data();
    std::atomic<bool> failed{false};
    size_t failedDiagonal = n;

    const size_t tiles = n / tileSize + (n % tileSize != 0);
    const int workers = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), tiles > 1 ? 2 * tiles : 1));
    #pragma omp parallel num_threads(workers)
    {
        #pragma omp single
        {
            for (size_t first = 0; first < n; first += tileSize) {
                const size_t last = std::min(first + tileSize, n);
                #pragma omp task firstprivate(first,last) \
                    depend(inout:matrix[first*n+first]) shared(failed,failedDiagonal)
                {
                    if (!failed.load(std::memory_order_relaxed)) {
                        const size_t bad = factorDiagonal(matrix, n, first, last);
                        if (bad != n) {
                            failedDiagonal = bad;
                            failed.store(true, std::memory_order_relaxed);
                        }
                    }
                }
                for (size_t row = last; row < n; row += tileSize) {
                    const size_t rowLast = std::min(row + tileSize, n);
                    #pragma omp task firstprivate(row,rowLast,first,last) \
                        depend(in:matrix[first*n+first]) \
                        depend(inout:matrix[row*n+first]) shared(failed)
                    {
                        if (!failed.load(std::memory_order_relaxed)) {
                            solvePanel(matrix, n, row, rowLast, first, last);
                        }
                    }
                }
                // Issue nearer diagonal tiles first to shorten the critical path.
                for (size_t col = last; col < n; col += tileSize) {
                    const size_t colLast = std::min(col + tileSize, n);
                    for (size_t row = col; row < n; row += tileSize) {
                        const size_t rowLast = std::min(row + tileSize, n);
                        #pragma omp task firstprivate(row,rowLast,col,colLast,first,last) \
                            depend(in:matrix[row*n+first],matrix[col*n+first]) \
                            depend(inout:matrix[row*n+col]) shared(failed)
                        {
                            if (!failed.load(std::memory_order_relaxed)) {
                                updateTile(matrix, n, row, rowLast, col, colLast, first, last);
                            }
                        }
                    }
                }
            }
        }

        // The single region's implicit barrier also waits for its tasks.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(matrix + i * n + i + 1, matrix + (i + 1) * n, 0.0);
        }
    }
    if (failed.load(std::memory_order_relaxed)) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedDiagonal);
        return false;
    }

    return true;
}

// Generate a symmetric positive definite matrix.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Keep the original random stream: changing seeds per thread changes A.
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute each symmetric pair once. Dynamic rows balance the triangular
    // workload; SIMD reductions accelerate the contiguous inner products.
    #pragma omp parallel for num_threads(rowWorkers(n)) schedule(dynamic,1)
    for (size_t i = 0; i < n; ++i) {
        const double* row = B.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* other = B.data() + j * n;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += row[k] * other[k];
            }
            if (i == j) {
                sum += n;
            }
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;

    // L is lower triangular. Entries beyond min(i,j) contribute only zero;
    // compare both symmetric entries directly without a reconstruction buffer.
    #pragma omp parallel for num_threads(rowWorkers(n)) schedule(dynamic,1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        const double* row = L.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* other = L.data() + j * n;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k) {
                sum += row[k] * other[k];
            }
            const double original = A_orig[i * n + j];
            const double transposed = A_orig[j * n + i];
            const double error = std::fabs(sum - original);
            const double transposeError = std::fabs(sum - transposed);
            maxError = std::max(maxError, std::max(error, transposeError));
            relError = std::max(relError, error / (std::fabs(original) + 1e-10));
            relError = std::max(relError, transposeError / (std::fabs(transposed) + 1e-10));
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

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
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
