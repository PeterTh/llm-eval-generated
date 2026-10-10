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
constexpr size_t blockSize = 64;
constexpr size_t maxBlockSize = 128;

// Avoid oversized teams for small matrices; OMP_NUM_THREADS remains an upper bound.
int matrixThreads(const size_t n) {
    const size_t blocks = n / blockSize + (n % blockSize != 0);
    return static_cast<int>(std::min(static_cast<size_t>(omp_get_max_threads()),
                                    std::max(size_t{1}, blocks)));
}

// C -= X * Y. Y is the transposed panel stored in the unused upper triangle.
// A 4-by-8 microtile keeps accumulators in registers and vectorizes across columns.
void updateTile(double* C, const double* X, const double* Y, const size_t stride,
                const size_t rows, const size_t cols, const size_t depth) {
    // Pack the shared operand into contiguous cache lines; large matrix strides
    // otherwise cause cache-set conflicts in every microtile.
    alignas(64) double packedY[maxBlockSize * (maxBlockSize + 8)];
    for (size_t k = 0; k < depth; ++k) {
        #pragma omp simd
        for (size_t j = 0; j < cols; ++j) {
            packedY[k * (cols + 8) + j] = Y[k * stride + j];
        }
    }
    Y = packedY;
    const size_t yStride = cols + 8;
    size_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            double c0[8], c1[8], c2[8], c3[8];
            #pragma omp simd
            for (size_t v = 0; v < 8; ++v) {
                c0[v] = C[i * stride + j + v];
                c1[v] = C[(i + 1) * stride + j + v];
                c2[v] = C[(i + 2) * stride + j + v];
                c3[v] = C[(i + 3) * stride + j + v];
            }
            for (size_t k = 0; k < depth; ++k) {
                const double x0 = X[i * stride + k];
                const double x1 = X[(i + 1) * stride + k];
                const double x2 = X[(i + 2) * stride + k];
                const double x3 = X[(i + 3) * stride + k];
                const double* y = Y + k * yStride + j;
                #pragma omp simd
                for (size_t v = 0; v < 8; ++v) {
                    c0[v] -= x0 * y[v];
                    c1[v] -= x1 * y[v];
                    c2[v] -= x2 * y[v];
                    c3[v] -= x3 * y[v];
                }
            }
            #pragma omp simd
            for (size_t v = 0; v < 8; ++v) {
                C[i * stride + j + v] = c0[v];
                C[(i + 1) * stride + j + v] = c1[v];
                C[(i + 2) * stride + j + v] = c2[v];
                C[(i + 3) * stride + j + v] = c3[v];
            }
        }
        for (size_t r = i; r < i + 4; ++r) {
            for (size_t k = 0; k < depth; ++k) {
                const double x = X[r * stride + k];
                #pragma omp simd
                for (size_t c = j; c < cols; ++c) {
                    C[r * stride + c] -= x * Y[k * yStride + c];
                }
            }
        }
    }
    for (; i < rows; ++i) {
        for (size_t k = 0; k < depth; ++k) {
            const double x = X[i * stride + k];
            #pragma omp simd
            for (size_t j = 0; j < cols; ++j) {
                C[i * stride + j] -= x * Y[k * yStride + j];
            }
        }
    }
}

// Factor a diagonal tile after all of its preceding updates have completed.
bool factorTile(double* D, const size_t stride, const size_t width,
                const size_t offset, std::atomic<size_t>& failed) {
    for (size_t j = 0; j < width; ++j) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t k = 0; k < j; ++k) {
            sum += D[j * stride + k] * D[j * stride + k];
        }
        const double value = D[j * stride + j] - sum;
        if (value <= 0.0) {
            failed.store(offset + j, std::memory_order_relaxed);
            return false;
        }
        const double diagonal = std::sqrt(value);
        D[j * stride + j] = diagonal;
        for (size_t i = j + 1; i < width; ++i) {
            double dot = 0.0;
            #pragma omp simd reduction(+:dot)
            for (size_t k = 0; k < j; ++k) {
                dot += D[i * stride + k] * D[j * stride + k];
            }
            const double value = (D[i * stride + j] - dot) / diagonal;
            D[i * stride + j] = value;
            D[j * stride + i] = value;
        }
    }
    return true;
}

// Solve P * D^T = the updated panel, then publish its transpose for tile updates.
void solveTile(double* P, const double* D, double* transposed,
               const size_t stride, const size_t rows, const size_t width) {
    double inverse[maxBlockSize];
    for (size_t j = 0; j < width; ++j) {
        inverse[j] = 1.0 / D[j * stride + j];
    }
    // Forward substitution on four rows at a time reuses the diagonal tile
    // and vectorizes the contiguous residual updates instead of short dot products.
    size_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        double* p0 = P + i * stride;
        double* p1 = p0 + stride;
        double* p2 = p1 + stride;
        double* p3 = p2 + stride;
        for (size_t j = 0; j < width; ++j) {
            const double x0 = p0[j] * inverse[j];
            const double x1 = p1[j] * inverse[j];
            const double x2 = p2[j] * inverse[j];
            const double x3 = p3[j] * inverse[j];
            p0[j] = x0;
            p1[j] = x1;
            p2[j] = x2;
            p3[j] = x3;
            #pragma omp simd
            for (size_t c = j + 1; c < width; ++c) {
                const double d = D[j * stride + c];
                p0[c] -= x0 * d;
                p1[c] -= x1 * d;
                p2[c] -= x2 * d;
                p3[c] -= x3 * d;
            }
        }
    }
    for (; i < rows; ++i) {
        double* row = P + i * stride;
        for (size_t j = 0; j < width; ++j) {
            const double x = row[j] * inverse[j];
            row[j] = x;
            #pragma omp simd
            for (size_t c = j + 1; c < width; ++c) {
                row[c] -= x * D[j * stride + c];
            }
        }
    }
    for (size_t j = 0; j < width; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            transposed[j * stride + i] = P[i * stride + j];
        }
    }
}
} // namespace

// In-place, tiled Cholesky: A = L * L^T. Dependencies allow later panels to
// start without a global barrier between panels, while serializing tile writers.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const size_t tileSize = n < 2048 ? blockSize : maxBlockSize;
    const size_t blocks = n / tileSize + (n % tileSize != 0);
    std::vector<unsigned char> dependencies(blocks * blocks);
    [[maybe_unused]] unsigned char* tokens = dependencies.data();
    double* data = A.data();
    std::atomic<size_t> failed{n};

    #pragma omp parallel num_threads(matrixThreads(n))
    {
        #pragma omp single
        {
            for (size_t k = 0; k < blocks; ++k) {
                const size_t kb = k * tileSize;
                const size_t width = std::min(tileSize, n - kb);
                const size_t kk = k * blocks + k;
                #pragma omp task firstprivate(kb, width, kk) depend(inout:tokens[kk]) shared(failed)
                {
                    if (failed.load(std::memory_order_relaxed) == n) {
                        factorTile(data + kb * n + kb, n, width, kb, failed);
                    }
                }
                for (size_t i = k + 1; i < blocks; ++i) {
                    const size_t ib = i * tileSize;
                    const size_t rows = std::min(tileSize, n - ib);
                    const size_t ik = i * blocks + k;
                    #pragma omp task firstprivate(ib, kb, rows, width, kk, ik) depend(in:tokens[kk]) depend(inout:tokens[ik]) shared(failed)
                    {
                        if (failed.load(std::memory_order_relaxed) == n) {
                            solveTile(data + ib * n + kb, data + kb * n + kb,
                                      data + kb * n + ib, n, rows, width);
                        }
                    }
                }
                for (size_t j = k + 1; j < blocks; ++j) {
                    const size_t jb = j * tileSize;
                    const size_t cols = std::min(tileSize, n - jb);
                    const size_t jk = j * blocks + k;
                    for (size_t i = j; i < blocks; ++i) {
                        const size_t ib = i * tileSize;
                        const size_t rows = std::min(tileSize, n - ib);
                        const size_t ik = i * blocks + k;
                        const size_t ij = i * blocks + j;
                        #pragma omp task firstprivate(ib, jb, kb, rows, cols, width, ik, jk, ij) depend(in:tokens[ik], tokens[jk]) depend(inout:tokens[ij]) shared(failed)
                        {
                            if (failed.load(std::memory_order_relaxed) == n) {
                                updateTile(data + ib * n + jb, data + ib * n + kb,
                                           data + kb * n + jb, n, rows, cols, width);
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
        // All tasks finish before erasing the transpose workspace.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
        }
    }
    const size_t badDiagonal = failed.load(std::memory_order_relaxed);
    if (badDiagonal != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", badDiagonal);
        return false;
    }
    return true;
}

// Generate the same seeded random B, then A = B * B^T + n * I.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // Keep the RNG sequence independent of the OpenMP thread count.
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    #pragma omp parallel for num_threads(matrixThreads(n)) schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            if (i == j) {
                A[i * n + i] = sum + n;
            } else {
                A[i * n + j] = sum;
                A[j * n + i] = sum;
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    // Only the common nonzero part of two lower-triangular rows contributes.
    // Compare both halves without allocating another n-by-n matrix.
    #pragma omp parallel for num_threads(matrixThreads(n)) schedule(dynamic, 1) reduction(max:maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            const double mirrorError = fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, std::max(error, mirrorError));
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
            relError = std::max(relError, mirrorError / (fabs(A_orig[j * n + i]) + 1e-10));
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
    // Use the full timer resolution now that small factorizations take under 1 ms.
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
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
