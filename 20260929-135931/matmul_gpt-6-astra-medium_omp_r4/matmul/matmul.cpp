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
    constexpr size_t rows = 32, cols = 64, depth = 128;
    // Each worker owns an output tile, so no synchronization or reductions
    // are needed. Blocking both output dimensions also exposes enough work
    // for many cores when the matrix has relatively few rows.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < N; ii += rows) {
        for (size_t jj = 0; jj < N; jj += cols) {
            const size_t ni = std::min(rows, N - ii);
            const size_t nj = std::min(cols, N - jj);
            alignas(64) double tile[rows][cols] = {};
            for (size_t kk = 0; kk < N; kk += depth) {
                const size_t kend = std::min(N, kk + depth);
                size_t i = 0;
                // Keep four rows of eight output values in registers and
                // reuse every loaded B vector across all four rows.
                for (; i + 4 <= ni; i += 4) {
                    size_t j = 0;
                    for (; j + 8 <= nj; j += 8) {
                        double c0[8], c1[8], c2[8], c3[8];
                        #pragma omp simd
                        for (size_t x = 0; x < 8; ++x) {
                            c0[x] = tile[i][j + x];
                            c1[x] = tile[i + 1][j + x];
                            c2[x] = tile[i + 2][j + x];
                            c3[x] = tile[i + 3][j + x];
                        }
                        for (size_t k = kk; k < kend; ++k) {
                            const double a0 = A[(ii + i) * N + k];
                            const double a1 = A[(ii + i + 1) * N + k];
                            const double a2 = A[(ii + i + 2) * N + k];
                            const double a3 = A[(ii + i + 3) * N + k];
                            const double* b = B.data() + k * N + jj + j;
                            #pragma omp simd
                            for (size_t x = 0; x < 8; ++x) {
                                c0[x] += a0 * b[x];
                                c1[x] += a1 * b[x];
                                c2[x] += a2 * b[x];
                                c3[x] += a3 * b[x];
                            }
                        }
                        #pragma omp simd
                        for (size_t x = 0; x < 8; ++x) {
                            tile[i][j + x] = c0[x];
                            tile[i + 1][j + x] = c1[x];
                            tile[i + 2][j + x] = c2[x];
                            tile[i + 3][j + x] = c3[x];
                        }
                    }
                    for (size_t r = i; r < i + 4; ++r) {
                        for (size_t k = kk; k < kend; ++k) {
                            const double a = A[(ii + r) * N + k];
                            #pragma omp simd
                            for (size_t x = j; x < nj; ++x)
                                tile[r][x] += a * B[k * N + jj + x];
                        }
                    }
                }
                for (; i < ni; ++i) {
                    for (size_t k = kk; k < kend; ++k) {
                        const double a = A[(ii + i) * N + k];
                        #pragma omp simd
                        for (size_t j = 0; j < nj; ++j)
                            tile[i][j] += a * B[k * N + jj + j];
                    }
                }
            }
            for (size_t i = 0; i < ni; ++i) {
                #pragma omp simd
                for (size_t j = 0; j < nj; ++j)
                    C[(ii + i) * N + jj + j] = tile[i][j];
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
