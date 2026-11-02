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

// A fixed-width microkernel lets the compiler retain the partial sums in
// registers. memcpy provides safe unaligned vector loads and stores.
using Vec = double __attribute__((vector_size(32)));
static inline Vec loadVec(const double* p) {
    Vec v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
static inline void storeVec(double* p, Vec v) {
    std::memcpy(p, &v, sizeof(v));
}
static inline void multiply4x8(const double* a, const double* b, double* c,
                              size_t N, size_t i, size_t j, size_t bStride,
                              size_t kk, size_t kEnd) {
    const Vec zero = {};
    Vec s00 = zero, s01 = zero, s10 = zero, s11 = zero;
    Vec s20 = zero, s21 = zero, s30 = zero, s31 = zero;
    if (kk != 0) {
        s00 = loadVec(c + i * N + j);       s01 = loadVec(c + i * N + j + 4);
        s10 = loadVec(c + (i+1) * N + j);   s11 = loadVec(c + (i+1) * N + j + 4);
        s20 = loadVec(c + (i+2) * N + j);   s21 = loadVec(c + (i+2) * N + j + 4);
        s30 = loadVec(c + (i+3) * N + j);   s31 = loadVec(c + (i+3) * N + j + 4);
    }
    for (size_t k = kk; k < kEnd; ++k) {
        const double a0 = a[i * N + k], a1 = a[(i+1) * N + k];
        const double a2 = a[(i+2) * N + k], a3 = a[(i+3) * N + k];
        const Vec v0 = {a0, a0, a0, a0}, v1 = {a1, a1, a1, a1};
        const Vec v2 = {a2, a2, a2, a2}, v3 = {a3, a3, a3, a3};
        const Vec b0 = loadVec(b + (k-kk) * bStride), b1 = loadVec(b + (k-kk) * bStride + 4);
        s00 += v0*b0; s01 += v0*b1;
        s10 += v1*b0; s11 += v1*b1;
        s20 += v2*b0; s21 += v2*b1;
        s30 += v3*b0; s31 += v3*b1;
    }
    storeVec(c + i * N + j, s00);       storeVec(c + i * N + j + 4, s01);
    storeVec(c + (i+1) * N + j, s10);   storeVec(c + (i+1) * N + j + 4, s11);
    storeVec(c + (i+2) * N + j, s20);   storeVec(c + (i+2) * N + j + 4, s21);
    storeVec(c + (i+3) * N + j, s30);   storeVec(c + (i+3) * N + j + 4, s31);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    // Each thread owns complete output tiles, so no synchronization is needed
    // between k blocks and each element is accumulated in increasing k order.
    constexpr size_t rowBlock = 32;
    constexpr size_t colBlock = 64;
    constexpr size_t depthBlock = 256;
    constexpr size_t microCols = 8;
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < N; ii += rowBlock) {
        for (size_t jj = 0; jj < N; jj += colBlock) {
            const size_t iEnd = std::min(ii + rowBlock, N);
            const size_t jEnd = std::min(jj + colBlock, N);
            alignas(64) double packedB[depthBlock * colBlock];
            for (size_t kk = 0; kk < N; kk += depthBlock) {
                const size_t kEnd = std::min(kk + depthBlock, N);
                for (size_t k = kk; k < kEnd; ++k) {
                    #pragma omp simd
                    for (size_t x = 0; x < jEnd - jj; ++x)
                        packedB[(k - kk) * colBlock + x] = b[k * N + jj + x];
                }
                size_t i = ii;
                for (; i + 4 <= iEnd; i += 4) {
                    for (size_t j = jj; j < jEnd; j += microCols) {
                        if (jEnd - j >= microCols) {
                            multiply4x8(a, packedB + j - jj, c, N, i, j, colBlock, kk, kEnd);
                            continue;
                        }
                        const size_t width = jEnd - j;
                        double s0[microCols] = {}, s1[microCols] = {};
                        double s2[microCols] = {}, s3[microCols] = {};
                        if (kk != 0) {
                            #pragma omp simd
                            for (size_t x = 0; x < width; ++x) {
                                s0[x] = c[i * N + j + x];
                                s1[x] = c[(i + 1) * N + j + x];
                                s2[x] = c[(i + 2) * N + j + x];
                                s3[x] = c[(i + 3) * N + j + x];
                            }
                        }
                        for (size_t k = kk; k < kEnd; ++k) {
                            const double a0 = a[i * N + k];
                            const double a1 = a[(i + 1) * N + k];
                            const double a2 = a[(i + 2) * N + k];
                            const double a3 = a[(i + 3) * N + k];
                            #pragma omp simd
                            for (size_t x = 0; x < width; ++x) {
                                const double bv = packedB[(k - kk) * colBlock + (j - jj) + x];
                                s0[x] += a0 * bv;
                                s1[x] += a1 * bv;
                                s2[x] += a2 * bv;
                                s3[x] += a3 * bv;
                            }
                        }
                        #pragma omp simd
                        for (size_t x = 0; x < width; ++x) {
                            c[i * N + j + x] = s0[x];
                            c[(i + 1) * N + j + x] = s1[x];
                            c[(i + 2) * N + j + x] = s2[x];
                            c[(i + 3) * N + j + x] = s3[x];
                        }
                    }
                }
                // Handle the final one to three rows of a partial tile.
                for (; i < iEnd; ++i) {
                    for (size_t j = jj; j < jEnd; j += microCols) {
                        const size_t width = std::min(microCols, jEnd - j);
                        double sums[microCols] = {};
                        if (kk != 0) {
                            #pragma omp simd
                            for (size_t x = 0; x < width; ++x)
                                sums[x] = c[i * N + j + x];
                        }
                        for (size_t k = kk; k < kEnd; ++k) {
                            const double av = a[i * N + k];
                            #pragma omp simd
                            for (size_t x = 0; x < width; ++x)
                                sums[x] += av * packedB[(k - kk) * colBlock + (j - jj) + x];
                        }
                        #pragma omp simd
                        for (size_t x = 0; x < width; ++x)
                            c[i * N + j + x] = sums[x];
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
