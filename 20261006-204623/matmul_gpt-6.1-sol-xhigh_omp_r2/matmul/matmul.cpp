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

void initMatrix(std::vector<double>& mat, const size_t N) {
    // Keep initialization teams proportional to the amount of work.
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max<size_t>(1, N * N / 32768)));
    #pragma omp parallel for schedule(static) num_threads(threads)
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
    constexpr size_t depthBlock = 256;
    constexpr size_t lanes = 8;
    const size_t rowBlocks = N / rowBlock + (N % rowBlock != 0);
    const size_t colBlocks = N / colBlock + (N % colBlock != 0);
    // Limit startup/scheduling overhead for small matrices while allowing
    // larger problems to use the full OpenMP team requested by the runtime.
    const size_t usefulThreads = std::max<size_t>(
        1, std::min(rowBlocks * colBlocks, N / 16));
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), usefulThreads));
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();

    // Each tile has one owner for the entire reduction: no synchronization
    // or partial-sum reductions are needed, and k remains in ascending order.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (size_t bi = 0; bi < rowBlocks; ++bi) {
        for (size_t bj = 0; bj < colBlocks; ++bj) {
            const size_t iBegin = bi * rowBlock;
            const size_t iEnd = std::min(iBegin + rowBlock, N);
            const size_t jBegin = bj * colBlock;
            const size_t jEnd = std::min(jBegin + colBlock, N);

            for (size_t i = iBegin; i < iEnd; ++i) {
                #pragma omp simd
                for (size_t j = jBegin; j < jEnd; ++j) {
                    c[i * N + j] = 0.0;
                }
            }

            for (size_t kb = 0; kb < N; kb += depthBlock) {
                const size_t kEnd = std::min(kb + depthBlock, N);
                // Pack each eight-column panel in k order. The active panel
                // fits in L1, avoiding strided B reads and TLB/cache conflicts.
                // Scratch belongs to this tile's thread and is reused by rows.
                alignas(64) double packedB[colBlock * depthBlock];
                for (size_t j = jBegin; j + lanes <= jEnd; j += lanes) {
                    for (size_t k = kb; k < kEnd; ++k) {
                        double* dest = packedB + (j - jBegin) * depthBlock
                                              + (k - kb) * lanes;
                        #pragma omp simd
                        for (size_t lane = 0; lane < lanes; ++lane) {
                            dest[lane] = b[k * N + j + lane];
                        }
                    }
                }

                size_t i = iBegin;
                for (; i + 4 <= iEnd; i += 4) {
                    size_t j = jBegin;
                    for (; j + lanes <= jEnd; j += lanes) {
                        // Four rows share B loads; eight columns keep the
                        // accumulators small enough to stay in SIMD registers.
                        alignas(64) double s0[lanes], s1[lanes];
                        alignas(64) double s2[lanes], s3[lanes];
                        #pragma omp simd
                        for (size_t lane = 0; lane < lanes; ++lane) {
                            s0[lane] = c[i * N + j + lane];
                            s1[lane] = c[(i + 1) * N + j + lane];
                            s2[lane] = c[(i + 2) * N + j + lane];
                            s3[lane] = c[(i + 3) * N + j + lane];
                        }
                        for (size_t k = kb; k < kEnd; ++k) {
                            const double a0 = a[i * N + k];
                            const double a1 = a[(i + 1) * N + k];
                            const double a2 = a[(i + 2) * N + k];
                            const double a3 = a[(i + 3) * N + k];
                            const double* bRow = packedB + (j - jBegin) * depthBlock
                                                        + (k - kb) * lanes;
                            #pragma omp simd
                            for (size_t lane = 0; lane < lanes; ++lane) {
                                const double value = bRow[lane];
                                s0[lane] += a0 * value;
                                s1[lane] += a1 * value;
                                s2[lane] += a2 * value;
                                s3[lane] += a3 * value;
                            }
                        }
                        #pragma omp simd
                        for (size_t lane = 0; lane < lanes; ++lane) {
                            c[i * N + j + lane] = s0[lane];
                            c[(i + 1) * N + j + lane] = s1[lane];
                            c[(i + 2) * N + j + lane] = s2[lane];
                            c[(i + 3) * N + j + lane] = s3[lane];
                        }
                    }

                    // Handle a partial SIMD block at the right edge.
                    for (size_t k = kb; k < kEnd; ++k) {
                        #pragma omp simd
                        for (size_t jj = j; jj < jEnd; ++jj) {
                            const double value = b[k * N + jj];
                            c[i * N + jj] += a[i * N + k] * value;
                            c[(i + 1) * N + jj] += a[(i + 1) * N + k] * value;
                            c[(i + 2) * N + jj] += a[(i + 2) * N + k] * value;
                            c[(i + 3) * N + jj] += a[(i + 3) * N + k] * value;
                        }
                    }
                }

                // Handle remaining rows with contiguous, vectorized B reads.
                for (; i < iEnd; ++i) {
                    for (size_t k = kb; k < kEnd; ++k) {
                        const double value = a[i * N + k];
                        #pragma omp simd
                        for (size_t j = jBegin; j < jEnd; ++j) {
                            c[i * N + j] += value * b[k * N + j];
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
    auto start = std::chrono::steady_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
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
