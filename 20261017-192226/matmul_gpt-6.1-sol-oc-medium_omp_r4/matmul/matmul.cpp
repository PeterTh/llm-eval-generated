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

// Keep a small output microtile in registers while traversing a K block.
// SIMD lanes represent independent columns, so no floating-point reduction
// or reassociation of the original increasing-k accumulation is needed.
template <size_t Rows, size_t Cols, bool Partial = false>
inline void multiplyMicrotile(const double* A, const double* B, double* C,
                              const size_t N, const size_t i, const size_t j,
                              const size_t kb, const size_t kend,
                              const size_t bStride,
                              const size_t remaining = Cols) {
    const size_t columns = Partial ? remaining : Cols;
    alignas(64) double sum0[Cols], sum1[Cols], sum2[Cols], sum3[Cols];
    #pragma omp simd
    for (size_t c = 0; c < columns; ++c) {
        sum0[c] = kb == 0 ? 0.0 : C[i * N + j + c];
        if constexpr (Rows == 4) {
            sum1[c] = kb == 0 ? 0.0 : C[(i + 1) * N + j + c];
            sum2[c] = kb == 0 ? 0.0 : C[(i + 2) * N + j + c];
            sum3[c] = kb == 0 ? 0.0 : C[(i + 3) * N + j + c];
        }
    }
    for (size_t k = kb; k < kend; ++k) {
        const double a0 = A[i * N + k];
        if constexpr (Rows == 4) {
            const double a1 = A[(i + 1) * N + k];
            const double a2 = A[(i + 2) * N + k];
            const double a3 = A[(i + 3) * N + k];
            #pragma omp simd
            for (size_t c = 0; c < columns; ++c) {
                const double b = B[(k - kb) * bStride + c];
                sum0[c] += a0 * b;
                sum1[c] += a1 * b;
                sum2[c] += a2 * b;
                sum3[c] += a3 * b;
            }
        } else {
            #pragma omp simd
            for (size_t c = 0; c < columns; ++c) {
                sum0[c] += a0 * B[(k - kb) * bStride + c];
            }
        }
    }
    #pragma omp simd
    for (size_t c = 0; c < columns; ++c) {
        C[i * N + j + c] = sum0[c];
        if constexpr (Rows == 4) {
            C[(i + 1) * N + j + c] = sum1[c];
            C[(i + 2) * N + j + c] = sum2[c];
            C[(i + 3) * N + j + c] = sum3[c];
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    constexpr size_t blockRows = 32;
    constexpr size_t blockCols = 64;
    constexpr size_t blockK = 128;

    // Each worker owns a complete output tile for all K blocks. Collapsing
    // both tile dimensions also exposes parallelism for smaller matrices.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ib = 0; ib < N; ib += blockRows) {
        for (size_t jb = 0; jb < N; jb += blockCols) {
            const size_t iend = std::min(ib + blockRows, N);
            const size_t jend = std::min(jb + blockCols, N);
            // A compact, private B panel avoids power-of-two stride cache
            // conflicts and is reused by every row microtile in this tile.
            alignas(64) double packedB[blockK * blockCols];
            for (size_t kb = 0; kb < N; kb += blockK) {
                const size_t kend = std::min(kb + blockK, N);
                for (size_t k = kb; k < kend; ++k) {
                    #pragma omp simd
                    for (size_t j = jb; j < jend; ++j) {
                        packedB[(k - kb) * blockCols + j - jb] = B[k * N + j];
                    }
                }
                size_t i = ib;
                for (; i + 4 <= iend; i += 4) {
                    size_t j = jb;
                    for (; j + 8 <= jend; j += 8) {
                        multiplyMicrotile<4, 8>(A.data(), packedB + j - jb, C.data(),
                                                N, i, j, kb, kend, blockCols);
                    }
                    if (j < jend) {
                        multiplyMicrotile<4, 8, true>(A.data(), packedB + j - jb, C.data(),
                                                N, i, j, kb, kend, blockCols, jend - j);
                    }
                }
                for (; i < iend; ++i) {
                    size_t j = jb;
                    for (; j + 8 <= jend; j += 8) {
                        multiplyMicrotile<1, 8>(A.data(), packedB + j - jb, C.data(),
                                                N, i, j, kb, kend, blockCols);
                    }
                    if (j < jend) {
                        multiplyMicrotile<1, 8, true>(A.data(), packedB + j - jb, C.data(),
                                                N, i, j, kb, kend, blockCols, jend - j);
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
