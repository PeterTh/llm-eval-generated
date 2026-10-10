#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    const int threads = static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                                        std::max<size_t>(1, (N + 31) / 32)));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

namespace {
constexpr size_t rowBlock = 32;
constexpr size_t columnBlock = 64;
constexpr size_t depthBlock = 128;
constexpr size_t vectorColumns = 8;

// Keep a small output tile in registers while reusing each packed B vector
// across multiple rows. SIMD runs across columns, so every dot product still
// accumulates its terms in the original increasing-k order.
template <size_t Rows>
inline void multiplyTile(const double* A, const double* packedB, double* C,
                         size_t N, size_t depth) {
    double sums[Rows][vectorColumns];
    for (size_t r = 0; r < Rows; ++r) {
        #pragma omp simd
        for (size_t j = 0; j < vectorColumns; ++j) {
            sums[r][j] = C[r * N + j];
        }
    }
    for (size_t k = 0; k < depth; ++k) {
        for (size_t r = 0; r < Rows; ++r) {
            const double a = A[r * N + k];
            #pragma omp simd
            for (size_t j = 0; j < vectorColumns; ++j) {
                sums[r][j] += a * packedB[k * columnBlock + j];
            }
        }
    }
    for (size_t r = 0; r < Rows; ++r) {
        #pragma omp simd
        for (size_t j = 0; j < vectorColumns; ++j) {
            C[r * N + j] = sums[r][j];
        }
    }
}
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    const size_t rowTiles = (N + rowBlock - 1) / rowBlock;
    const size_t columnTiles = (N + columnBlock - 1) / columnBlock;
    // Give each worker at least one row block's worth of work. This also
    // reuses the initialization team's size and avoids excessive startup
    // overhead for small matrices on machines with many cores.
    const int threads = static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                                         std::max<size_t>(1, rowTiles)));
    #pragma omp parallel num_threads(threads)
    {
        // Each worker owns its packing buffer and disjoint output tiles.
        alignas(64) double packedB[depthBlock * columnBlock];
        #pragma omp for collapse(2) schedule(static)
        for (size_t ti = 0; ti < rowTiles; ++ti) {
            for (size_t tj = 0; tj < columnTiles; ++tj) {
                const size_t firstRow = ti * rowBlock;
                const size_t firstColumn = tj * columnBlock;
                const size_t rows = std::min(rowBlock, N - firstRow);
                const size_t columns = std::min(columnBlock, N - firstColumn);
                for (size_t i = 0; i < rows; ++i) {
                    double* out = C.data() + (firstRow + i) * N + firstColumn;
                    #pragma omp simd
                    for (size_t j = 0; j < columns; ++j) {
                        out[j] = 0.0;
                    }
                }
                for (size_t kk = 0; kk < N; kk += depthBlock) {
                    const size_t depth = std::min(depthBlock, N - kk);
                    for (size_t k = 0; k < depth; ++k) {
                        const double* src = B.data() + (kk + k) * N + firstColumn;
                        #pragma omp simd
                        for (size_t j = 0; j < columns; ++j) {
                            packedB[k * columnBlock + j] = src[j];
                        }
                    }
                    for (size_t i = 0; i < rows; i += 4) {
                        const size_t activeRows = std::min<size_t>(4, rows - i);
                        const double* a = A.data() + (firstRow + i) * N + kk;
                        double* out = C.data() + (firstRow + i) * N + firstColumn;
                        size_t j = 0;
                        for (; j + vectorColumns <= columns; j += vectorColumns) {
                            if (activeRows == 4) {
                                multiplyTile<4>(a, packedB + j, out + j, N, depth);
                            } else {
                                for (size_t r = 0; r < activeRows; ++r) {
                                    multiplyTile<1>(a + r * N, packedB + j,
                                                    out + r * N + j, N, depth);
                                }
                            }
                        }
                        // Handle matrices whose dimensions are not tile multiples.
                        for (size_t r = 0; r < activeRows; ++r) {
                            for (size_t tail = j; tail < columns; ++tail) {
                                double sum = out[r * N + tail];
                                for (size_t k = 0; k < depth; ++k) {
                                    sum += a[r * N + k] * packedB[k * columnBlock + tail];
                                }
                                out[r * N + tail] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
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
