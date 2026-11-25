#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

namespace {
constexpr size_t panelWidth = 8;
constexpr size_t blockDepth = 128;

// Accumulate a small output tile in registers, vectorizing independent columns.
template <size_t rows>
inline void multiplyPanel(const double* a, const double panel[][panelWidth],
                          double* c, size_t stride, size_t depth, size_t width,
                          bool firstBlock) {
    alignas(64) double sums[rows][panelWidth]{};
    if (!firstBlock) {
        for (size_t r = 0; r < rows; ++r)
            std::copy_n(c + r * stride, width, sums[r]);
    }
    // Match the original dot product: rounded products in the paired loop
    // and a fused multiply-add for the final unpaired element.
    const size_t pairedDepth = depth - depth % 2;
    for (size_t k = 0; k < pairedDepth; ++k) {
        for (size_t r = 0; r < rows; ++r) {
            const double value = a[r * stride + k];
            #pragma omp simd
            for (size_t j = 0; j < panelWidth; ++j)
                sums[r][j] += value * panel[k][j];
        }
    }
    if (pairedDepth < depth) {
        for (size_t r = 0; r < rows; ++r) {
            const double value = a[r * stride + pairedDepth];
            #pragma omp simd
            for (size_t j = 0; j < panelWidth; ++j)
                sums[r][j] = std::fma(value, panel[pairedDepth][j], sums[r][j]);
        }
    }
    for (size_t r = 0; r < rows; ++r)
        std::copy_n(sums[r], width, c + r * stride);
}
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    constexpr size_t blockRows = 48;
    constexpr size_t blockCols = 64;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t row = 0; row < N; row += blockRows) {
        for (size_t col = 0; col < N; col += blockCols) {
            const size_t rowEnd = std::min(row + blockRows, N);
            const size_t colEnd = std::min(col + blockCols, N);
            alignas(64) double panel[blockDepth][panelWidth];
            for (size_t k = 0; k < N; k += blockDepth) {
                const size_t depth = std::min(blockDepth, N - k);
                for (size_t j = col; j < colEnd; j += panelWidth) {
                    const size_t width = std::min(panelWidth, colEnd - j);
                    // Pack a contiguous B panel and reuse it across the row block.
                    if (width == panelWidth) {
                        for (size_t t = 0; t < depth; ++t)
                            std::copy_n(&B[(k + t) * N + j], panelWidth, panel[t]);
                    } else {
                        for (size_t t = 0; t < depth; ++t) {
                            std::copy_n(&B[(k + t) * N + j], width, panel[t]);
                            std::fill(panel[t] + width, panel[t] + panelWidth, 0.0);
                        }
                    }
                    size_t i = row;
                    for (; i + 6 <= rowEnd; i += 6)
                        multiplyPanel<6>(&A[i * N + k], panel, &C[i * N + j],
                                         N, depth, width, k == 0);
                    for (; i < rowEnd; ++i)
                        multiplyPanel<1>(&A[i * N + k], panel, &C[i * N + j],
                                         N, depth, width, k == 0);
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
    double gflops = (2.0 * N * N * N) / (std::chrono::duration<double>(end - start).count()) / 1e9;
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
