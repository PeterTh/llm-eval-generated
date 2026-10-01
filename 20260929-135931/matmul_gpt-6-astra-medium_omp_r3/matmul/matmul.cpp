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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    constexpr size_t MR = 32, NC = 64, KC = 128;
    // Each worker owns an output tile for the entire reduction. Packing B
    // keeps the working set contiguous and avoids power-of-two row strides.
    #pragma omp parallel
    {
        alignas(64) double tile[MR][NC];
        alignas(64) double panel[KC][NC];
        #pragma omp for collapse(2) schedule(static)
        for (size_t ii = 0; ii < N; ii += MR) {
            for (size_t jj = 0; jj < N; jj += NC) {
                const size_t rows = std::min(MR, N - ii);
                const size_t cols = std::min(NC, N - jj);
                const size_t paddedCols = (cols + 7) / 8 * 8;
                for (auto& row : tile) {
                    std::fill_n(row, NC, 0.0);
                }
                for (size_t kk = 0; kk < N; kk += KC) {
                    const size_t depth = std::min(KC, N - kk);
                    for (size_t k = 0; k < depth; ++k) {
                        std::copy_n(B.data() + (kk + k) * N + jj, cols, panel[k]);
                        std::fill(panel[k] + cols, panel[k] + paddedCols, 0.0);
                    }
                    for (size_t i = 0; i < rows; i += 4) {
                        for (size_t j = 0; j < cols; j += 8) {
                            // Four rows by eight columns fit in SIMD registers.
                            double c0[8], c1[8], c2[8], c3[8];
                            #pragma GCC unroll 8
                            for (size_t x = 0; x < 8; ++x) {
                                c0[x] = tile[i][j + x];
                                c1[x] = tile[i + 1][j + x];
                                c2[x] = tile[i + 2][j + x];
                                c3[x] = tile[i + 3][j + x];
                            }
                            for (size_t k = 0; k < depth; ++k) {
                                const double a0 = A[(ii + i) * N + kk + k];
                                const double a1 = i + 1 < rows ? A[(ii + i + 1) * N + kk + k] : 0.0;
                                const double a2 = i + 2 < rows ? A[(ii + i + 2) * N + kk + k] : 0.0;
                                const double a3 = i + 3 < rows ? A[(ii + i + 3) * N + kk + k] : 0.0;
                                // Unroll columns so the compiler keeps SIMD accumulators
                                // in registers. Never reorder k: the
                                // summation order is independent of thread count.
                                #pragma GCC unroll 8
                                for (size_t x = 0; x < 8; ++x) {
                                    const double b = panel[k][j + x];
                                    c0[x] += a0 * b;
                                    c1[x] += a1 * b;
                                    c2[x] += a2 * b;
                                    c3[x] += a3 * b;
                                }
                            }
                            #pragma GCC unroll 8
                            for (size_t x = 0; x < 8; ++x) {
                                tile[i][j + x] = c0[x];
                                tile[i + 1][j + x] = c1[x];
                                tile[i + 2][j + x] = c2[x];
                                tile[i + 3][j + x] = c3[x];
                            }
                        }
                    }
                }
                for (size_t i = 0; i < rows; ++i) {
                    std::copy_n(tile[i], cols, C.data() + (ii + i) * N + jj);
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
