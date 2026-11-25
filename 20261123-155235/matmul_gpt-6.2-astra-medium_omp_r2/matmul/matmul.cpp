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
    const size_t chunks = 1 + mat.size() / 16384;
    const int threads = std::min<size_t>(omp_get_max_threads(), chunks);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    constexpr size_t tileRows = 32;
    constexpr size_t tileCols = 64;
    constexpr size_t tileDepth = 128;
    constexpr size_t microRows = 4;
    constexpr size_t microCols = 8;
    const size_t rowTiles = (N + tileRows - 1) / tileRows;
    const size_t colTiles = (N + tileCols - 1) / tileCols;
    const int threads = std::max<size_t>(1, std::min<size_t>(omp_get_max_threads(),
                                                             rowTiles * colTiles));

    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (size_t ti = 0; ti < rowTiles; ++ti) {
        for (size_t tj = 0; tj < colTiles; ++tj) {
            const size_t firstRow = ti * tileRows;
            const size_t firstCol = tj * tileCols;
            const size_t endRow = std::min(firstRow + tileRows, N);
            const size_t cols = std::min(tileCols, N - firstCol);
            alignas(64) double panel[tileCols / microCols][tileDepth][microCols];

            for (size_t k0 = 0; k0 < N; k0 += tileDepth) {
                const size_t depth = std::min(tileDepth, N - k0);
                // Pack each strip of B in the order the register block consumes it.
                for (size_t k = 0; k < depth; ++k) {
                    for (size_t j = 0; j < cols; j += microCols) {
                        std::copy_n(&B[(k0 + k) * N + firstCol + j],
                                    std::min(microCols, cols - j), panel[j / microCols][k]);
                    }
                }
                for (size_t i = firstRow; i < endRow; i += microRows) {
                    for (size_t j = 0; j < cols; j += microCols) {
                        const size_t rows = std::min(microRows, endRow - i);
                        const size_t width = std::min(microCols, cols - j);
                        alignas(64) double sums[microRows][microCols] = {};
                        if (k0 != 0) {
                            for (size_t r = 0; r < rows; ++r)
                                std::copy_n(&C[(i + r) * N + firstCol + j], width, sums[r]);
                        }
                        const auto& strip = panel[j / microCols];
                        const size_t pairedDepth = depth - depth % 2;
                        auto accumulate = [&](size_t height, size_t lanes) {
                            for (size_t k = 0; k < pairedDepth; ++k) {
                                for (size_t r = 0; r < height; ++r) {
                                    const double a = A[(i + r) * N + k0 + k];
                                    #pragma omp simd
                                    for (size_t c = 0; c < lanes; ++c)
                                        sums[r][c] += a * strip[k][c];
                                }
                            }
                        };
                        if (rows == microRows && width == microCols)
                            accumulate(microRows, microCols);
                        else
                            accumulate(rows, width);

                        // Preserve the original dot product's scalar tail rounding.
                        if (pairedDepth < depth) {
                            for (size_t r = 0; r < rows; ++r) {
                                const double a = A[(i + r) * N + k0 + pairedDepth];
                                #pragma omp simd
                                for (size_t c = 0; c < width; ++c) {
#ifdef FP_FAST_FMA
                                    sums[r][c] = std::fma(a, strip[pairedDepth][c], sums[r][c]);
#else
                                    sums[r][c] += a * strip[pairedDepth][c];
#endif
                                }
                            }
                        }
                        for (size_t r = 0; r < rows; ++r)
                            std::copy_n(sums[r], width, &C[(i + r) * N + firstCol + j]);
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
