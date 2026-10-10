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

constexpr size_t maxTileSize = 128;

size_t blockWidth(const size_t n) {
    // Small tiles expose concurrency; larger ones amortize task scheduling.
    return n >= 2048 ? maxTileSize : size_t{64};
}

int workerCount(const size_t n) {
    const size_t width = blockWidth(n);
    const size_t tiles = n / width + (n % width != 0);
    // Avoid oversized teams for small matrices while respecting the user's limit.
    const size_t usefulWorkers = std::max(size_t{1}, 2 * tiles);
    return static_cast<int>(std::min(usefulWorkers,
                                    static_cast<size_t>(omp_get_max_threads())));
}

// Keep a 4-by-8 output block in registers while streaming the two panels.
// The right panel is transposed, so SIMD lanes access adjacent columns.
inline void updateMicroTile(double* c, const double* a, const double* bt,
                            const size_t stride, const size_t depth) {
    alignas(64) double c0[8], c1[8], c2[8], c3[8];
    #pragma omp simd
    for (size_t j = 0; j < 8; ++j) {
        c0[j] = c[j];
        c1[j] = c[stride + j];
        c2[j] = c[2 * stride + j];
        c3[j] = c[3 * stride + j];
    }
    for (size_t k = 0; k < depth; ++k) {
        const double a0 = a[k];
        const double a1 = a[stride + k];
        const double a2 = a[2 * stride + k];
        const double a3 = a[3 * stride + k];
        const double* b = bt + k * 8;
        #pragma omp simd
        for (size_t j = 0; j < 8; ++j) {
            c0[j] -= a0 * b[j];
            c1[j] -= a1 * b[j];
            c2[j] -= a2 * b[j];
            c3[j] -= a3 * b[j];
        }
    }
    #pragma omp simd
    for (size_t j = 0; j < 8; ++j) {
        c[j] = c0[j];
        c[stride + j] = c1[j];
        c[2 * stride + j] = c2[j];
        c[3 * stride + j] = c3[j];
    }
}

// C -= A * B^T, including partial tiles at the matrix boundary.
void updateTile(double* c, const double* a, const double* bt,
                const size_t stride, const size_t rows,
                const size_t columns, const size_t depth) {
    // Packing avoids cache-set conflicts and TLB misses from the matrix stride.
    alignas(64) double packed[maxTileSize * maxTileSize];
    for (size_t j = 0; j < columns; j += 8) {
        const size_t count = std::min(size_t{8}, columns - j);
        for (size_t k = 0; k < depth; ++k) {
            #pragma omp simd
            for (size_t column = 0; column < count; ++column) {
                packed[j * maxTileSize + k * 8 + column] =
                    bt[k * stride + j + column];
            }
        }
    }
    bt = packed;
    const size_t fullRows = rows / 4 * 4;
    const size_t fullColumns = columns / 8 * 8;
    for (size_t i = 0; i < fullRows; i += 4) {
        for (size_t j = 0; j < fullColumns; j += 8) {
            updateMicroTile(c + i * stride + j, a + i * stride,
                            bt + j * maxTileSize, stride, depth);
        }
    }
    for (size_t i = 0; i < rows; ++i) {
        const size_t start = i < fullRows ? fullColumns : 0;
        for (size_t j = start; j < columns; j += 8) {
            const size_t count = std::min(size_t{8}, columns - j);
            for (size_t k = 0; k < depth; ++k) {
                const double value = a[i * stride + k];
                const double* b = bt + j * maxTileSize + k * 8;
                #pragma omp simd
                for (size_t column = 0; column < count; ++column) {
                    c[i * stride + j + column] -= value * b[column];
                }
            }
        }
    }
}

bool factorDiagonal(double* a, const size_t stride, const size_t width,
                    const size_t offset) {
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < j; ++k) {
                sum += a[i * stride + k] * a[j * stride + k];
            }
            const double value = a[i * stride + j] - sum;
            if (i == j) {
                if (value <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                           offset + j);
                    return false;
                }
                a[i * stride + j] = sqrt(value);
            } else {
                a[i * stride + j] = value / a[j * stride + j];
            }
        }
    }
    // Store L^T in the tile's upper triangle for contiguous panel solves.
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = i + 1; j < width; ++j) {
            a[i * stride + j] = a[j * stride + i];
        }
    }
    return true;
}

void solvePanel(double* panel, const double* diagonal, double* transpose,
                const size_t stride, const size_t rows, const size_t width) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < width; ++j) {
            const double value = panel[i * stride + j] / diagonal[j * stride + j];
            panel[i * stride + j] = value;
            #pragma omp simd
            for (size_t k = j + 1; k < width; ++k) {
                panel[i * stride + k] -= value * diagonal[j * stride + k];
            }
        }
    }
    // The unused upper triangle provides scratch space for transposed panels.
    // Readers depend on completion of this entire solve-and-transpose task.
    for (size_t j = 0; j < width; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            transpose[j * stride + i] = panel[i * stride + j];
        }
    }
}

} // namespace

// Tiled right-looking Cholesky with a dependency graph instead of panel barriers.
// The top-left address of each lower tile is its OpenMP dependency token.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* data = A.data();
    const size_t tileSize = blockWidth(n);
    std::atomic<bool> success{true};

    #pragma omp parallel num_threads(workerCount(n)) shared(success)
    {
        #pragma omp single
        {
            for (size_t k = 0; k < n; k += tileSize) {
                const size_t width = std::min(tileSize, n - k);
                double* diagonal = data + k * n + k;
                #pragma omp task firstprivate(diagonal, n, width, k) \
                    shared(success) depend(inout:diagonal[0])
                {
                    if (success.load(std::memory_order_relaxed) &&
                        !factorDiagonal(diagonal, n, width, k)) {
                        success.store(false, std::memory_order_relaxed);
                    }
                }

                for (size_t i = k + width; i < n; i += tileSize) {
                    const size_t rows = std::min(tileSize, n - i);
                    double* panel = data + i * n + k;
                    double* transpose = data + k * n + i;
                    #pragma omp task firstprivate(panel, diagonal, transpose, n, rows, width) \
                        shared(success) depend(in:diagonal[0]) depend(inout:panel[0])
                    {
                        if (success.load(std::memory_order_relaxed)) {
                            solvePanel(panel, diagonal, transpose, n, rows, width);
                        }
                    }
                }

                // Submit the next diagonal and column first to expose lookahead.
                for (size_t j = k + width; j < n; j += tileSize) {
                    const size_t columns = std::min(tileSize, n - j);
                    [[maybe_unused]] double* right = data + j * n + k;
                    const double* transpose = data + k * n + j;
                    for (size_t i = j; i < n; i += tileSize) {
                        const size_t rows = std::min(tileSize, n - i);
                        double* left = data + i * n + k;
                        double* tile = data + i * n + j;
                        #pragma omp task firstprivate(tile, left, right, transpose, n, rows, columns, width) \
                            shared(success) depend(in:left[0], right[0]) depend(inout:tile[0])
                        {
                            if (success.load(std::memory_order_relaxed)) {
                                updateTile(tile, left, transpose, n, rows, columns, width);
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
        }
    }
    return success.load(std::memory_order_relaxed);
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
    
    // Compute each symmetric pair once. Keep the seeded rand_r stream serial.
    #pragma omp parallel for num_threads(workerCount(n)) schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
            if (i != j) {
                A[j * n + i] = sum;
            }
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;

    // Reconstruct symmetric pairs and compare directly, without a second matrix.
    #pragma omp parallel for num_threads(workerCount(n)) schedule(dynamic, 1) reduction(max:maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
            if (i != j) {
                const double upperError = fabs(sum - A_orig[j * n + i]);
                maxError = std::max(maxError, upperError);
                relError = std::max(relError,
                    upperError / (fabs(A_orig[j * n + i]) + 1e-10));
            }
        }
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
