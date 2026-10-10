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

size_t chooseTileSize(size_t n) {
    // Small tiles expose work on small matrices; larger tiles amortize task and
    // packing costs as the matrix grows.
    return n <= 1024 ? 64 : (n <= 2048 ? 96 : 128);
}

int chooseThreadCount(size_t n, size_t tileSize) {
    // Bound idle workers on small inputs while honoring the OpenMP thread limit.
    const size_t tiles = n / tileSize + (n % tileSize != 0);
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                           std::max<size_t>(1, 2 * tiles)));
}

// Pack one input tile transposed so SIMD loads across output columns are
// contiguous. Padding avoids cache-set conflicts for power-of-two matrix sizes.
// Four rows by eight columns retain their sums in registers through the product.
template <bool subtract>
void productTile(double* C, const double* X, const double* Y, size_t n,
                size_t rows, size_t cols, size_t depth, size_t cStride) {
    constexpr size_t pitch = maxTileSize + 8;
    alignas(64) double YT[maxTileSize * pitch];
    for (size_t j = 0; j < cols; ++j) {
        for (size_t k = 0; k < depth; ++k) YT[k * pitch + j] = Y[j * n + k];
    }
    size_t i = 0;
    for (; i + 3 < rows; i += 4) {
        size_t j = 0;
        for (; j + 7 < cols; j += 8) {
            double s0[8] = {}, s1[8] = {}, s2[8] = {}, s3[8] = {};
            for (size_t k = 0; k < depth; ++k) {
                const double x0 = X[i * n + k], x1 = X[(i + 1) * n + k];
                const double x2 = X[(i + 2) * n + k], x3 = X[(i + 3) * n + k];
                #pragma omp simd
                for (size_t c = 0; c < 8; ++c) {
                    const double y = YT[k * pitch + j + c];
                    s0[c] += x0 * y;
                    s1[c] += x1 * y;
                    s2[c] += x2 * y;
                    s3[c] += x3 * y;
                }
            }
            #pragma omp simd
            for (size_t c = 0; c < 8; ++c) {
                if constexpr (subtract) {
                    C[i * cStride + j + c] -= s0[c];
                    C[(i + 1) * cStride + j + c] -= s1[c];
                    C[(i + 2) * cStride + j + c] -= s2[c];
                    C[(i + 3) * cStride + j + c] -= s3[c];
                } else {
                    C[i * cStride + j + c] += s0[c];
                    C[(i + 1) * cStride + j + c] += s1[c];
                    C[(i + 2) * cStride + j + c] += s2[c];
                    C[(i + 3) * cStride + j + c] += s3[c];
                }
            }
        }
        for (; j < cols; ++j) {
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            #pragma omp simd reduction(+:s0,s1,s2,s3)
            for (size_t k = 0; k < depth; ++k) {
                const double y = YT[k * pitch + j];
                s0 += X[i * n + k] * y;
                s1 += X[(i + 1) * n + k] * y;
                s2 += X[(i + 2) * n + k] * y;
                s3 += X[(i + 3) * n + k] * y;
            }
            if constexpr (subtract) {
                C[i * cStride + j] -= s0;
                C[(i + 1) * cStride + j] -= s1;
                C[(i + 2) * cStride + j] -= s2;
                C[(i + 3) * cStride + j] -= s3;
            } else {
                C[i * cStride + j] += s0;
                C[(i + 1) * cStride + j] += s1;
                C[(i + 2) * cStride + j] += s2;
                C[(i + 3) * cStride + j] += s3;
            }
        }
    }
    for (; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < depth; ++k) sum += X[i * n + k] * YT[k * pitch + j];
            if constexpr (subtract) C[i * cStride + j] -= sum;
            else C[i * cStride + j] += sum;
        }
    }
}

size_t factorDiagonal(double* D, size_t n, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        double* row = D + i * n;
        for (size_t j = 0; j < i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < j; ++k) sum += row[k] * D[j * n + k];
            row[j] = (row[j] - sum) / D[j * n + j];
        }
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t k = 0; k < i; ++k) sum += row[k] * row[k];
        const double val = row[i] - sum;
        if (val <= 0.0) return i;
        row[i] = sqrt(val);
    }
    return width;
}

// D stores the diagonal tile transposed and its diagonal reciprocals. Updating
// four right-hand sides together vectorizes across contiguous panel columns,
// including the short inner products near the start of the triangular solve.
void solvePanel(double* P, const double* D, size_t n, size_t rows, size_t width,
                size_t stride) {
    size_t i = 0;
    for (; i + 3 < rows; i += 4) {
        double* p0 = P + i * n;
        double* p1 = p0 + n;
        double* p2 = p1 + n;
        double* p3 = p2 + n;
        for (size_t j = 0; j < width; ++j) {
            const double* d = D + j * stride;
            const double x0 = p0[j] * d[j], x1 = p1[j] * d[j];
            const double x2 = p2[j] * d[j], x3 = p3[j] * d[j];
            p0[j] = x0; p1[j] = x1; p2[j] = x2; p3[j] = x3;
            #pragma omp simd
            for (size_t k = j + 1; k < width; ++k) {
                p0[k] -= x0 * d[k];
                p1[k] -= x1 * d[k];
                p2[k] -= x2 * d[k];
                p3[k] -= x3 * d[k];
            }
        }
    }
    for (; i < rows; ++i) {
        double* row = P + i * n;
        for (size_t j = 0; j < width; ++j) {
            const double* d = D + j * stride;
            const double x = row[j] * d[j];
            row[j] = x;
            #pragma omp simd
            for (size_t k = j + 1; k < width; ++k) row[k] -= x * d[k];
        }
    }
}
} // namespace

// In-place tiled Cholesky, A = L * L^T. A tile's first element is its
// dependency token; tasks always access the entire corresponding tile.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* data = A.data();
    const size_t tileSize = chooseTileSize(n);
    const int threads = chooseThreadCount(n, tileSize);
    const size_t tiles = n / tileSize + (n % tileSize != 0);
    const size_t diagonalStride = tileSize + 8;
    // Shared packed diagonals occupy O(n * tileSize) space and are read-only
    // after their factor task. The matrix-tile dependency also protects these.
    std::vector<double> packedDiagonals(tiles * tileSize * diagonalStride);
    double* packed = packedDiagonals.data();
    std::atomic<size_t> failed(n);
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single nowait
        {
            for (size_t k = 0; k < n; k += tileSize) {
                const size_t width = std::min(tileSize, n - k);
                double* diagonal = data + k * n + k;
                double* packedDiagonal = packed + (k / tileSize) * tileSize * diagonalStride;
                #pragma omp task firstprivate(diagonal,packedDiagonal,k,width) shared(failed) depend(inout:diagonal[0])
                {
                    if (failed.load(std::memory_order_relaxed) == n) {
                        const size_t bad = factorDiagonal(diagonal, n, width);
                        if (bad != width) {
                            failed.store(k + bad, std::memory_order_relaxed);
                        } else {
                            for (size_t j = 0; j < width; ++j) {
                                packedDiagonal[j * diagonalStride + j] = 1.0 / diagonal[j * n + j];
                                for (size_t i = j + 1; i < width; ++i)
                                    packedDiagonal[j * diagonalStride + i] = diagonal[i * n + j];
                            }
                        }
                    }
                }
                for (size_t i = k + width; i < n; i += tileSize) {
                    const size_t rows = std::min(tileSize, n - i);
                    double* panel = data + i * n + k;
                    #pragma omp task firstprivate(panel,diagonal,packedDiagonal,rows,width) shared(failed) depend(in:diagonal[0]) depend(inout:panel[0])
                    {
                        if (failed.load(std::memory_order_relaxed) == n)
                            solvePanel(panel, packedDiagonal, n, rows, width, diagonalStride);
                    }
                }
                // Queue the next diagonal's updates first to shorten the critical
                // path. Dependencies permit later panels to start before unrelated
                // trailing updates finish; there is no barrier per panel.
                for (size_t j = k + width; j < n; j += tileSize) {
                    const size_t cols = std::min(tileSize, n - j);
                    double* right = data + j * n + k;
                    for (size_t i = j; i < n; i += tileSize) {
                        const size_t rows = std::min(tileSize, n - i);
                        double* left = data + i * n + k;
                        double* target = data + i * n + j;
                        #pragma omp task firstprivate(target,left,right,rows,cols,width) shared(failed) depend(in:left[0],right[0]) depend(inout:target[0])
                        {
                            if (failed.load(std::memory_order_relaxed) == n)
                                productTile<true>(target, left, right, n, rows, cols, width, n);
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
        // Wait for the entire task graph before zeroing the upper triangle.
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
        }
    }
    const size_t bad = failed.load(std::memory_order_relaxed);
    if (bad != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", bad);
        return false;
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
    
    // Keep rand_r serial so every thread count uses the original seeded stream.
    // Only the lower triangle of the symmetric product needs to be computed.
    const size_t tileSize = chooseTileSize(n);
    const size_t tiles = n / tileSize + (n % tileSize != 0);
    const int threads = chooseThreadCount(n, tileSize);
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for collapse(2) schedule(dynamic,1)
        for (size_t ti = 0; ti < tiles; ++ti) {
            for (size_t tj = 0; tj < tiles; ++tj) {
                if (tj > ti) continue;
                const size_t i = ti * tileSize, j = tj * tileSize;
                const size_t rows = std::min(tileSize, n - i);
                const size_t cols = std::min(tileSize, n - j);
                // Accumulate in a compact tile to avoid repeatedly writing
                // matrix rows with a large, potentially aliasing cache stride.
                constexpr size_t stride = maxTileSize + 8;
                alignas(64) double target[maxTileSize * stride] = {};
                for (size_t k = 0; k < n; k += tileSize) {
                    productTile<false>(target, B.data() + i * n + k,
                                       B.data() + j * n + k, n, rows, cols,
                                       std::min(tileSize, n - k), stride);
                }
                for (size_t r = 0; r < rows; ++r)
                    std::copy_n(target + r * stride, cols, A.data() + (i + r) * n + j);
            }
        }
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) A[i * n + j] = A[j * n + i];
            A[i * n + i] += n;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Fuse the symmetric reconstruction and comparison. The known zero upper
    // triangle lets each dot product stop at its last nonzero element.
    double maxError = 0.0;
    double relError = 0.0;

    const int threads = chooseThreadCount(n, chooseTileSize(n));
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k) sum += L[i * n + k] * L[j * n + k];
            const double lower = A_orig[i * n + j], upper = A_orig[j * n + i];
            const double error = fabs(sum - lower), upperError = fabs(sum - upper);
            maxError = std::max(maxError, std::max(error, upperError));
            relError = std::max(relError, std::max(error / (fabs(lower) + 1e-10),
                                                 upperError / (fabs(upper) + 1e-10)));
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
    auto start = std::chrono::steady_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::steady_clock::now();
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
