#include <chrono>
#include <algorithm>
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
    constexpr size_t rowBlock = 4;
    constexpr size_t colBlock = 32;
    // Each tile belongs to one thread for the entire multiplication. The
    // innermost loop traverses contiguous columns, and k remains ordered for
    // every output element.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < N; ii += rowBlock) {
        for (size_t jj = 0; jj < N; jj += colBlock) {
            const size_t iEnd = std::min(ii + rowBlock, N);
            const size_t jEnd = std::min(jj + colBlock, N);
            if (iEnd - ii == rowBlock && jEnd - jj == colBlock) {
                // Separate accumulators let the compiler vectorize across
                // columns while reusing each loaded B value for four rows.
                double s0[colBlock] = {}, s1[colBlock] = {};
                double s2[colBlock] = {}, s3[colBlock] = {};
                for (size_t k = 0; k < N; ++k) {
                    const double a0 = A[ii * N + k];
                    const double a1 = A[(ii + 1) * N + k];
                    const double a2 = A[(ii + 2) * N + k];
                    const double a3 = A[(ii + 3) * N + k];
                    #pragma omp simd
                    for (size_t j = 0; j < colBlock; ++j) {
                        const double b = B[k * N + jj + j];
                        s0[j] += a0 * b;
                        s1[j] += a1 * b;
                        s2[j] += a2 * b;
                        s3[j] += a3 * b;
                    }
                }
                for (size_t j = 0; j < colBlock; ++j) {
                    C[ii * N + jj + j] = s0[j];
                    C[(ii + 1) * N + jj + j] = s1[j];
                    C[(ii + 2) * N + jj + j] = s2[j];
                    C[(ii + 3) * N + jj + j] = s3[j];
                }
            } else {
                for (size_t i = ii; i < iEnd; ++i) {
                    for (size_t j = jj; j < jEnd; ++j) {
                        C[i * N + j] = 0.0;
                    }
                }
                for (size_t k = 0; k < N; ++k) {
                    for (size_t i = ii; i < iEnd; ++i) {
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
