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
    // Output tiles are independent; splitting both dimensions supplies enough
    // work even when the matrix has fewer row tiles than available cores.
    constexpr size_t rowBlock = 32;
    constexpr size_t colBlock = 64;
    constexpr size_t depthBlock = 128;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < N; ii += rowBlock) {
        for (size_t jj = 0; jj < N; jj += colBlock) {
            const size_t rows = std::min(rowBlock, N - ii);
            const size_t cols = std::min(colBlock, N - jj);
            alignas(64) double tile[rowBlock][colBlock] = {};
            alignas(64) double packedB[depthBlock][colBlock];

            for (size_t kk = 0; kk < N; kk += depthBlock) {
                const size_t kend = std::min(N, kk + depthBlock);
                // Packing removes large row strides and cache-set conflicts
                // from the repeatedly used B panel, without shared scratch.
                for (size_t k = kk; k < kend; ++k) {
                    #pragma omp simd
                    for (size_t j = 0; j < cols; ++j) {
                        packedB[k - kk][j] = B[k * N + jj + j];
                    }
                }
                size_t i = 0;
                // A 4x8 register kernel reuses each vector of B across four
                // rows. SIMD lanes represent columns, never partial sums.
                for (; i + 4 <= rows; i += 4) {
                    size_t j = 0;
                    for (; j + 8 <= cols; j += 8) {
                        double s0[8], s1[8], s2[8], s3[8];
                        #pragma omp simd
                        for (size_t p = 0; p < 8; ++p) {
                            s0[p] = tile[i][j + p];
                            s1[p] = tile[i + 1][j + p];
                            s2[p] = tile[i + 2][j + p];
                            s3[p] = tile[i + 3][j + p];
                        }
                        for (size_t k = kk; k < kend; ++k) {
                            const double a0 = A[(ii + i) * N + k];
                            const double a1 = A[(ii + i + 1) * N + k];
                            const double a2 = A[(ii + i + 2) * N + k];
                            const double a3 = A[(ii + i + 3) * N + k];
                            const double* b = packedB[k - kk] + j;
                            #pragma omp simd
                            for (size_t p = 0; p < 8; ++p) {
                                s0[p] += a0 * b[p];
                                s1[p] += a1 * b[p];
                                s2[p] += a2 * b[p];
                                s3[p] += a3 * b[p];
                            }
                        }
                        #pragma omp simd
                        for (size_t p = 0; p < 8; ++p) {
                            tile[i][j + p] = s0[p];
                            tile[i + 1][j + p] = s1[p];
                            tile[i + 2][j + p] = s2[p];
                            tile[i + 3][j + p] = s3[p];
                        }
                    }
                    // Handle columns outside a complete register tile.
                    for (size_t r = i; r < i + 4; ++r) {
                        for (size_t k = kk; k < kend; ++k) {
                            const double a = A[(ii + r) * N + k];
                            #pragma omp simd
                            for (size_t p = j; p < cols; ++p) {
                                tile[r][p] += a * packedB[k - kk][p];
                            }
                        }
                    }
                }
                // Handle rows outside a complete register tile.
                for (; i < rows; ++i) {
                    for (size_t k = kk; k < kend; ++k) {
                        const double a = A[(ii + i) * N + k];
                        #pragma omp simd
                        for (size_t j = 0; j < cols; ++j) {
                            tile[i][j] += a * packedB[k - kk][j];
                        }
                    }
                }
            }
            for (size_t i = 0; i < rows; ++i) {
                #pragma omp simd
                for (size_t j = 0; j < cols; ++j) {
                    C[(ii + i) * N + jj + j] = tile[i][j];
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
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * N * N * N) / seconds / 1e9;
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
