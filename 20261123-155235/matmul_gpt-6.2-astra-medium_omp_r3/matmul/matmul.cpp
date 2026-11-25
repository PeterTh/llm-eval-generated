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
constexpr size_t tileRows = 32;
constexpr size_t tileCols = 64;
constexpr size_t tileDepth = 128;
constexpr size_t lanes = 8;

// Keep several output rows in registers while sharing one packed row of B.
template<size_t Rows>
inline void multiplyPanel(const double* a, const double* packed, double* c,
                          size_t stride, size_t depth, size_t width, bool first) {
    double sums[Rows][lanes] = {};
    if (!first) {
        for (size_t r = 0; r < Rows; ++r)
            std::copy_n(c + r * stride, width, sums[r]);
    }
    const size_t pairedDepth = depth - depth % 2;
    for (size_t k = 0; k < pairedDepth; ++k) {
        for (size_t r = 0; r < Rows; ++r) {
            const double value = a[r * stride + k];
            #pragma omp simd
            for (size_t j = 0; j < lanes; ++j)
                sums[r][j] += value * packed[k * lanes + j];
        }
    }
    // Match the scalar multiply-add at the end of an odd-length dot product.
    for (size_t r = 0; r < Rows; ++r) {
        if (pairedDepth != depth) {
            #pragma omp simd
            for (size_t j = 0; j < lanes; ++j)
                sums[r][j] = std::fma(a[r * stride + pairedDepth],
                                      packed[pairedDepth * lanes + j], sums[r][j]);
        }
        std::copy_n(sums[r], width, c + r * stride);
    }
}
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    // Each worker owns complete output tiles, including their entire k reduction.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t row = 0; row < N; row += tileRows) {
        for (size_t col = 0; col < N; col += tileCols) {
            const size_t endRow = row + std::min(tileRows, N - row);
            const size_t endCol = col + std::min(tileCols, N - col);
            alignas(64) double packed[tileDepth * lanes];
            for (size_t k = 0; k < N; k += tileDepth) {
                const size_t depth = std::min(tileDepth, N - k);
                for (size_t j = col; j < endCol; j += lanes) {
                    const size_t width = std::min(lanes, endCol - j);
                    if (width == lanes) {
                        for (size_t p = 0; p < depth; ++p)
                            std::copy_n(B.data() + (k + p) * N + j, lanes, packed + p * lanes);
                    } else {
                        for (size_t p = 0; p < depth; ++p) {
                            double* dst = packed + p * lanes;
                            std::copy_n(B.data() + (k + p) * N + j, width, dst);
                            std::fill(dst + width, dst + lanes, 0.0);
                        }
                    }
                    size_t i = row;
                    for (; i + 4 <= endRow; i += 4)
                        multiplyPanel<4>(A.data() + i * N + k, packed, C.data() + i * N + j,
                                         N, depth, width, k == 0);
                    for (; i < endRow; ++i)
                        multiplyPanel<1>(A.data() + i * N + k, packed, C.data() + i * N + j,
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
    const std::chrono::duration<double> elapsed = end - start;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / elapsed.count() / 1e9;
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
