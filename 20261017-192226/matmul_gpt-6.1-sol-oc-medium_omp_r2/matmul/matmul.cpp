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
    constexpr size_t rowBlock = 32;
    constexpr size_t colBlock = 64;
    constexpr size_t depthBlock = 128;

    // Each worker owns a complete output tile for all k, so no reductions,
    // atomics, or synchronization between depth blocks are needed.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < N; ii += rowBlock) {
        for (size_t jj = 0; jj < N; jj += colBlock) {
            const size_t iEnd = std::min(ii + rowBlock, N);
            const size_t jEnd = std::min(jj + colBlock, N);
            alignas(64) double packedB[depthBlock * colBlock];
            for (size_t i = ii; i < iEnd; ++i) {
                std::fill(C.data() + i * N + jj, C.data() + i * N + jEnd, 0.0);
            }

            for (size_t kk = 0; kk < N; kk += depthBlock) {
                const size_t kEnd = std::min(kk + depthBlock, N);
                // Pack narrow B panels contiguously to avoid strided cache
                // misses (especially for power-of-two matrix dimensions).
                for (size_t j = jj; j + 8 <= jEnd; j += 8) {
                    for (size_t k = kk; k < kEnd; ++k) {
                        double* panel = packedB + (j - jj) * depthBlock + (k - kk) * 8;
                        #pragma omp simd
                        for (size_t x = 0; x < 8; ++x) {
                            panel[x] = B[k * N + j + x];
                        }
                    }
                }
                size_t i = ii;
                for (; i + 4 <= iEnd; i += 4) {
                    size_t j = jj;
                    for (; j + 8 <= jEnd; j += 8) {
                        // A small register tile reuses each B vector across
                        // four rows and avoids loading/storing C at every k.
                        double c0[8], c1[8], c2[8], c3[8];
                        #pragma omp simd
                        for (size_t x = 0; x < 8; ++x) {
                            c0[x] = C[i * N + j + x];
                            c1[x] = C[(i + 1) * N + j + x];
                            c2[x] = C[(i + 2) * N + j + x];
                            c3[x] = C[(i + 3) * N + j + x];
                        }
                        for (size_t k = kk; k < kEnd; ++k) {
                            const double a0 = A[i * N + k];
                            const double a1 = A[(i + 1) * N + k];
                            const double a2 = A[(i + 2) * N + k];
                            const double a3 = A[(i + 3) * N + k];
                            const double* b = packedB + (j - jj) * depthBlock + (k - kk) * 8;
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
                            C[i * N + j + x] = c0[x];
                            C[(i + 1) * N + j + x] = c1[x];
                            C[(i + 2) * N + j + x] = c2[x];
                            C[(i + 3) * N + j + x] = c3[x];
                        }
                    }
                    // Handle partial column tiles without padded matrices.
                    for (size_t r = i; r < i + 4; ++r) {
                        for (size_t k = kk; k < kEnd; ++k) {
                            const double a = A[r * N + k];
                            #pragma omp simd
                            for (size_t c = j; c < jEnd; ++c) {
                                C[r * N + c] += a * B[k * N + c];
                            }
                        }
                    }
                }
                // Handle the final one to three rows in the same k order.
                for (; i < iEnd; ++i) {
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double a = A[i * N + k];
                        #pragma omp simd
                        for (size_t j = jj; j < jEnd; ++j) {
                            C[i * N + j] += a * B[k * N + j];
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
