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
    double* __restrict data = mat.data();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        double* __restrict row = data + i * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Cache-blocked, register-tiled, shared-memory parallel matrix multiplication.
//
// The output is partitioned into square blocks that are distributed over the
// available OpenMP threads.  Inside a block the work is done by a small
// register micro-kernel (MR rows x NR columns) so that every value of B is
// reused in registers for MR different rows instead of being re-read from
// memory.  The k dimension is additionally split into KC chunks so the panel
// of B used by a block stays cache resident and is reused by all row
// micro-tiles of that block.
//
// The block size is derived from the number of threads, which gives enough
// independent blocks for good load balance and parallel scalability while
// keeping the blocks large enough to amortise memory traffic.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double* __restrict c = C.data();

    const size_t maxThreads = static_cast<size_t>(omp_get_max_threads());
    const size_t side = static_cast<size_t>(
        std::ceil(std::sqrt(static_cast<double>(4 * maxThreads))));
    size_t block = std::min<size_t>(64, std::max<size_t>(1, N / std::max<size_t>(1, side)));
    if (block >= 32) {
        block = (block / 32) * 32;
    }
    block = std::max<size_t>(1, std::min(block, N));
    const size_t numBlocks = (N + block - 1) / block;

    constexpr size_t MR = 8;    // rows per register micro-tile
    constexpr size_t NR = 4;    // columns per register micro-tile
    constexpr size_t KC = 320;  // k-chunk size (cache blocking of B)

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t bi = 0; bi < numBlocks; ++bi) {
        for (size_t bj = 0; bj < numBlocks; ++bj) {
            const size_t i0 = bi * block;
            const size_t j0 = bj * block;
            const size_t iEnd = std::min(i0 + block, N);
            const size_t jEnd = std::min(j0 + block, N);
            const size_t iFull = i0 + ((iEnd - i0) / MR) * MR;
            const size_t jFull = j0 + ((jEnd - j0) / NR) * NR;

            for (size_t kk = 0; kk < N; kk += KC) {
                const size_t kEnd = std::min(kk + KC, N);

                for (size_t jj = j0; jj < jFull; jj += NR) {
                    for (size_t ii = i0; ii < iFull; ii += MR) {
                        double acc[MR][NR] = {};

                        for (size_t k = kk; k < kEnd; ++k) {
                            const double* __restrict bk = b + k * N + jj;
                            double bv[NR];
                            #pragma omp simd
                            for (size_t q = 0; q < NR; ++q) {
                                bv[q] = bk[q];
                            }
                            #pragma GCC unroll 8
                            for (size_t r = 0; r < MR; ++r) {
                                const double av = a[(ii + r) * N + k];
                                #pragma omp simd
                                for (size_t q = 0; q < NR; ++q) {
                                    acc[r][q] += av * bv[q];
                                }
                            }
                        }

                        for (size_t r = 0; r < MR; ++r) {
                            double* __restrict cr = c + (ii + r) * N + jj;
                            if (kk == 0) {
                                #pragma omp simd
                                for (size_t q = 0; q < NR; ++q) {
                                    cr[q] = acc[r][q];
                                }
                            } else {
                                #pragma omp simd
                                for (size_t q = 0; q < NR; ++q) {
                                    cr[q] += acc[r][q];
                                }
                            }
                        }
                    }
                }
            }

            // Remainder columns (and the last partial column micro-tile).
            for (size_t i = i0; i < iFull; ++i) {
                for (size_t j = jFull; j < jEnd; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < N; ++k) {
                        sum += a[i * N + k] * b[k * N + j];
                    }
                    c[i * N + j] = sum;
                }
            }

            // Remainder rows (including the bottom-right corner).
            for (size_t i = iFull; i < iEnd; ++i) {
                for (size_t j = j0; j < jEnd; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < N; ++k) {
                        sum += a[i * N + k] * b[k * N + j];
                    }
                    c[i * N + j] = sum;
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
