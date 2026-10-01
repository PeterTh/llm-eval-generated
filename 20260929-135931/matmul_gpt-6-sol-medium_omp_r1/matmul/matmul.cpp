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

int threadCount(const size_t N) {
    // Keep enough tiles per worker to amortize parallel-region overhead.
    const size_t rowTiles = N / 16 + (N % 16 != 0);
    const size_t columnTiles = N / 64 + (N % 64 != 0);
    const size_t usefulThreads = std::max<size_t>(1, std::min(N / 8, rowTiles * columnTiles));
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), usefulThreads));
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static) num_threads(threadCount(N))
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    constexpr size_t rowBlock = 16;
    constexpr size_t columnBlock = 64;
    constexpr size_t depthBlock = 128;
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();

    // Each thread owns complete, disjoint tiles of C. Keeping a small set of
    // rows and columns live in cache also makes accesses to B contiguous.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threadCount(N))
    for (size_t ii = 0; ii < N; ii += rowBlock) {
        for (size_t jj = 0; jj < N; jj += columnBlock) {
            const size_t iEnd = (N - ii < rowBlock) ? N : ii + rowBlock;
            const size_t jEnd = (N - jj < columnBlock) ? N : jj + columnBlock;
            for (size_t kk = 0; kk < N; kk += depthBlock) {
                const size_t kEnd = (N - kk < depthBlock) ? N : kk + depthBlock;
                size_t i = ii;
                for (; i + 3 < iEnd; i += 4) {
                    double* c0 = c + i * N;
                    double* c1 = c0 + N;
                    double* c2 = c1 + N;
                    double* c3 = c2 + N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double a0 = a[i * N + k];
                        const double a1 = a[(i + 1) * N + k];
                        const double a2 = a[(i + 2) * N + k];
                        const double a3 = a[(i + 3) * N + k];
                        const double* bRow = b + k * N;
                        #pragma omp simd
                        for (size_t j = jj; j < jEnd; ++j) {
                            const double value = bRow[j];
                            c0[j] += a0 * value;
                            c1[j] += a1 * value;
                            c2[j] += a2 * value;
                            c3[j] += a3 * value;
                        }
                    }
                }
                for (; i < iEnd; ++i) {
                    double* cRow = c + i * N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double aValue = a[i * N + k];
                        const double* bRow = b + k * N;
                        #pragma omp simd
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += aValue * bRow[j];
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
