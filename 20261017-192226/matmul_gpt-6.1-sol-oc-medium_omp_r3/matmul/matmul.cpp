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

namespace {
constexpr size_t rowBlock = 32;
constexpr size_t columnBlock = 64;
constexpr size_t depthBlock = 128;
constexpr size_t vectorColumns = 8;

// Independent columns are vectorized; the k reduction remains in its original
// order. Four rows reuse each loaded vector of the packed right-hand matrix.
template <size_t Rows>
inline void multiplyMicroTile(const double* a, const double* b, double* c,
                              size_t N, size_t depth, size_t columns) {
    alignas(64) double sums[Rows][vectorColumns] = {};
    for (size_t r = 0; r < Rows; ++r) {
        for (size_t j = 0; j < columns; ++j) {
            sums[r][j] = c[r * N + j];
        }
    }
    for (size_t k = 0; k < depth; ++k) {
        for (size_t r = 0; r < Rows; ++r) {
            const double value = a[r * N + k];
            #pragma omp simd
            for (size_t j = 0; j < vectorColumns; ++j) {
                sums[r][j] += value * b[k * columnBlock + j];
            }
        }
    }
    for (size_t r = 0; r < Rows; ++r) {
        for (size_t j = 0; j < columns; ++j) {
            c[r * N + j] = sums[r][j];
        }
    }
}
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    #pragma omp parallel
    {
        // Private packing storage keeps the active B panel contiguous and avoids
        // stride/TLB penalties. Each output tile has exactly one owning thread.
        alignas(64) double packedB[depthBlock * columnBlock];
        #pragma omp for collapse(2) schedule(static)
        for (size_t ii = 0; ii < N; ii += rowBlock) {
            for (size_t jj = 0; jj < N; jj += columnBlock) {
                const size_t rows = std::min(rowBlock, N - ii);
                const size_t columns = std::min(columnBlock, N - jj);
                for (size_t i = 0; i < rows; ++i) {
                    std::fill_n(C.data() + (ii + i) * N + jj, columns, 0.0);
                }
                for (size_t kk = 0; kk < N; kk += depthBlock) {
                    const size_t depth = std::min(depthBlock, N - kk);
                    for (size_t k = 0; k < depth; ++k) {
                        double* packedRow = packedB + k * columnBlock;
                        std::copy_n(B.data() + (kk + k) * N + jj, columns, packedRow);
                        std::fill(packedRow + columns, packedRow + columnBlock, 0.0);
                    }
                    for (size_t j = 0; j < columns; j += vectorColumns) {
                        const size_t width = std::min(vectorColumns, columns - j);
                        size_t i = 0;
                        for (; i + 4 <= rows; i += 4) {
                            multiplyMicroTile<4>(A.data() + (ii + i) * N + kk,
                                                 packedB + j,
                                                 C.data() + (ii + i) * N + jj + j,
                                                 N, depth, width);
                        }
                        for (; i < rows; ++i) {
                            multiplyMicroTile<1>(A.data() + (ii + i) * N + kk,
                                                 packedB + j,
                                                 C.data() + (ii + i) * N + jj + j,
                                                 N, depth, width);
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
