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
        #pragma omp simd
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Keep a small output block in registers, reusing each B vector across rows.
// Vectorize independent columns, never the reduction over k.
template <size_t Rows, size_t Cols>
inline void multiplyBlock(const double* A, const double* B, double* C,
                          const size_t N, const size_t i, const size_t j,
                          const size_t kb, const size_t ke,
                          const size_t bStride) {
    double sums[Rows][Cols];
    for (size_t r = 0; r < Rows; ++r) {
        #pragma omp simd
        for (size_t c = 0; c < Cols; ++c) {
            sums[r][c] = kb == 0 ? 0.0 : C[(i + r) * N + j + c];
        }
    }
    for (size_t k = kb; k < ke; ++k) {
        for (size_t r = 0; r < Rows; ++r) {
            const double a = A[(i + r) * N + k];
            #pragma omp simd
            for (size_t c = 0; c < Cols; ++c) {
                sums[r][c] += a * B[(k - kb) * bStride + c];
            }
        }
    }
    for (size_t r = 0; r < Rows; ++r) {
        #pragma omp simd
        for (size_t c = 0; c < Cols; ++c) {
            C[(i + r) * N + j + c] = sums[r][c];
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    constexpr size_t rowTile = 32;
    constexpr size_t colTile = 128;
    constexpr size_t depthTile = 128;
    const size_t rowTiles = (N + rowTile - 1) / rowTile;
    const size_t colTiles = (N + colTile - 1) / colTile;
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();

    // One thread owns each output tile for the entire reduction: no atomics,
    // shared partial sums, or barriers between depth blocks are needed.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ti = 0; ti < rowTiles; ++ti) {
        for (size_t tj = 0; tj < colTiles; ++tj) {
            const size_t ib = ti * rowTile;
            const size_t jb = tj * colTile;
            const size_t ie = std::min(ib + rowTile, N);
            const size_t je = std::min(jb + colTile, N);
            // A private packed panel avoids matrix-sized strides and cache
            // conflicts in B, and is reused by all microblocks in this tile.
            alignas(64) double panel[depthTile * colTile];
            for (size_t kb = 0; kb < N; kb += depthTile) {
                const size_t ke = std::min(kb + depthTile, N);
                for (size_t k = kb; k < ke; ++k) {
                    #pragma omp simd
                    for (size_t j = jb; j < je; ++j) {
                        panel[(k - kb) * colTile + j - jb] = b[k * N + j];
                    }
                }
                size_t i = ib;
                for (; i + 4 <= ie; i += 4) {
                    size_t j = jb;
                    for (; j + 8 <= je; j += 8) {
                        multiplyBlock<4, 8>(a, panel + j - jb, c,
                                            N, i, j, kb, ke, colTile);
                    }
                    for (; j < je; ++j) {
                        multiplyBlock<4, 1>(a, panel + j - jb, c,
                                            N, i, j, kb, ke, colTile);
                    }
                }
                for (; i < ie; ++i) {
                    size_t j = jb;
                    for (; j + 8 <= je; j += 8) {
                        multiplyBlock<1, 8>(a, panel + j - jb, c,
                                            N, i, j, kb, ke, colTile);
                    }
                    for (; j < je; ++j) {
                        multiplyBlock<1, 1>(a, panel + j - jb, c,
                                            N, i, j, kb, ke, colTile);
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
