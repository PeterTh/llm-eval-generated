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
    const size_t threads = std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, (N + 7) / 8));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

namespace {
constexpr size_t tileRows = 32;
constexpr size_t tileCols = 64;
constexpr size_t panelDepth = 128;
constexpr size_t registerCols = 8;

template<size_t Rows>
void multiplyPanel(const double* a, size_t stride, const double panel[][tileCols],
                   double* c, size_t depth, size_t width, bool firstPanel) {
    for (size_t col = 0; col < width; col += registerCols) {
        const size_t lanes = std::min(registerCols, width - col);
        double sums[Rows][registerCols] = {};
        if (!firstPanel) {
            for (size_t r = 0; r < Rows; ++r)
                std::copy_n(c + r * stride + col, lanes, sums[r]);
        }

        // Match the reference dot product's paired products and scalar tail.
        const size_t pairedDepth = depth - depth % 2;
        for (size_t k = 0; k < pairedDepth; ++k) {
            for (size_t r = 0; r < Rows; ++r) {
                const double value = a[r * stride + k];
                #pragma omp simd
                for (size_t j = 0; j < registerCols; ++j)
                    sums[r][j] += value * panel[k][col + j];
            }
        }

        if (pairedDepth != depth) {
            for (size_t r = 0; r < Rows; ++r) {
                const double value = a[r * stride + pairedDepth];
                #pragma omp simd
                for (size_t j = 0; j < registerCols; ++j) {
#ifdef FP_FAST_FMA
                    sums[r][j] = std::fma(value, panel[pairedDepth][col + j], sums[r][j]);
#else
                    sums[r][j] += value * panel[pairedDepth][col + j];
#endif
                }
            }
        }

        for (size_t r = 0; r < Rows; ++r)
            std::copy_n(sums[r], lanes, c + r * stride + col);
    }
}
} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    // Each tile has one owner for the full dot product, including edge tiles.
    const size_t tiles = ((N + tileRows - 1) / tileRows) * ((N + tileCols - 1) / tileCols);
    const size_t threads = std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, tiles));
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (size_t row = 0; row < N; row += tileRows) {
        for (size_t col = 0; col < N; col += tileCols) {
            alignas(64) double panel[panelDepth][tileCols];
            const size_t endRow = std::min(row + tileRows, N);
            const size_t width = std::min(tileCols, N - col);
            for (size_t base = 0; base < N; base += panelDepth) {
                const size_t depth = std::min(panelDepth, N - base);
                for (size_t k = 0; k < depth; ++k) {
                    std::copy_n(&B[(base + k) * N + col], width, panel[k]);
                    // Pad the final SIMD group for a partial column tile.
                    std::fill_n(panel[k] + width, (registerCols - width % registerCols)
                                                  % registerCols, 0.0);
                }

                size_t r = row;
                for (; r + 4 <= endRow; r += 4)
                    multiplyPanel<4>(&A[r * N + base], N, panel, &C[r * N + col],
                                     depth, width, base == 0);
                for (; r < endRow; ++r)
                    multiplyPanel<1>(&A[r * N + base], N, panel, &C[r * N + col],
                                     depth, width, base == 0);
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
    double gflops = (2.0 * N * N * N) / std::chrono::duration<double>(end - start).count() / 1e9;
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
