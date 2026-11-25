#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {
constexpr size_t kernelRows = 6;
constexpr size_t panelCols = 8;
constexpr size_t tileRows = 8 * kernelRows;
constexpr size_t tileCols = 12 * panelCols;
constexpr size_t innerBlock = 256;

struct alignas(64) PackedRow {
    double values[panelCols];
};

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrices(std::vector<double>& A, std::vector<double>& B, const size_t N,
                  const int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const double value = getPseudoRndValue(N, i, j);
            A[i * N + j] = value;
            B[i * N + j] = value;
        }
    }
}

// GCC on hardware with FMA fuses the odd scalar tail of the original dot product.
#if defined(__GNUC__) && !defined(__clang__) && defined(FP_FAST_FMA)
constexpr bool fusedTail = true;
#else
constexpr bool fusedTail = false;
#endif

template <size_t Rows, bool FullPanel>
void multiplyPanel(const double* A, const PackedRow* packedB, double* C, const size_t N,
                   const size_t firstK, const size_t lastK, const size_t panelWidth) {
    const size_t width = FullPanel ? panelCols : panelWidth;
    double sums[Rows][panelCols] = {};
    if (firstK != 0) {
        for (size_t i = 0; i < Rows; ++i) {
            std::copy_n(C + i * N, width, sums[i]);
        }
    }

    const bool hasTail = fusedTail && lastK == N && N % 2 != 0;
    const size_t endK = lastK - hasTail;
    for (size_t k = firstK; k < endK; ++k) {
        const double* b = packedB[k].values;
        for (size_t i = 0; i < Rows; ++i) {
            const double a = A[i * N + k];
            #pragma omp simd
            for (size_t j = 0; j < panelCols; ++j) {
                sums[i][j] += a * b[j];
            }
        }
    }

    if (hasTail) {
        const double* b = packedB[endK].values;
        for (size_t i = 0; i < Rows; ++i) {
            const double a = A[i * N + endK];
            #pragma omp simd
            for (size_t j = 0; j < panelCols; ++j) {
                sums[i][j] = std::fma(a, b[j], sums[i][j]);
            }
        }
    }

    for (size_t i = 0; i < Rows; ++i) {
        std::copy_n(sums[i], width, C + i * N);
    }
}

template <bool FullPanel>
void multiplyTilePanel(const double* A, const PackedRow* packedB, double* C, const size_t N,
                       const size_t rows, const size_t firstK, const size_t lastK,
                       const size_t width) {
    size_t i = 0;
    for (; i + kernelRows <= rows; i += kernelRows) {
        multiplyPanel<kernelRows, FullPanel>(A + i * N, packedB, C + i * N, N,
                                            firstK, lastK, width);
    }
    for (; i < rows; ++i) {
        multiplyPanel<1, FullPanel>(A + i * N, packedB, C + i * N, N, firstK, lastK, width);
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const int threads) {
    const size_t panels = (N + panelCols - 1) / panelCols;
    const std::unique_ptr<PackedRow[]> packedB(new PackedRow[panels * N]);

    #pragma omp parallel num_threads(threads)
    {
        // Pack in blocks so a thread reads neighboring panels of B together.
        #pragma omp for collapse(2) schedule(static)
        for (size_t col = 0; col < N; col += tileCols) {
            for (size_t k = 0; k < N; k += innerBlock) {
                const size_t colEnd = std::min(col + tileCols, N);
                const size_t kEnd = std::min(k + innerBlock, N);
                for (size_t j = col; j < colEnd; j += panelCols) {
                    const size_t width = std::min(panelCols, N - j);
                    PackedRow* panel = packedB.get() + (j / panelCols) * N;
                    if (width == panelCols) {
                        for (size_t p = k; p < kEnd; ++p) {
                            std::copy_n(B.data() + p * N + j, panelCols, panel[p].values);
                        }
                    } else {
                        for (size_t p = k; p < kEnd; ++p) {
                            double* dest = panel[p].values;
                            std::copy_n(B.data() + p * N + j, width, dest);
                            std::fill(dest + width, dest + panelCols, 0.0);
                        }
                    }
                }
            }
        }

        // Each tile owns its output throughout the ordered accumulation over k.
        #pragma omp for collapse(2) schedule(static)
        for (size_t row = 0; row < N; row += tileRows) {
            for (size_t col = 0; col < N; col += tileCols) {
                const size_t rows = std::min(tileRows, N - row);
                const size_t colEnd = std::min(col + tileCols, N);
                for (size_t k = 0; k < N; k += innerBlock) {
                    const size_t kEnd = std::min(k + innerBlock, N);
                    for (size_t j = col; j < colEnd; j += panelCols) {
                        const size_t width = std::min(panelCols, N - j);
                        const PackedRow* panel = packedB.get() + (j / panelCols) * N;
                        const double* a = A.data() + row * N;
                        double* c = C.data() + row * N + j;
                        if (width == panelCols) {
                            multiplyTilePanel<true>(a, panel, c, N, rows, k, kEnd, width);
                        } else {
                            multiplyTilePanel<false>(a, panel, c, N, rows, k, kEnd, width);
                        }
                    }
                }
            }
        }
    }
}
}  // namespace

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return true;
    }
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

    // Keep enough multiply-adds per worker to amortize starting an OpenMP team.
    constexpr double workPerThread = 8.0 * 1024 * 1024;
    const int threads = static_cast<int>(std::min<double>(omp_get_max_threads(),
        std::max(1.0, std::ceil(static_cast<double>(N) * N * N / workPerThread))));

    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrices(A, B, N, threads);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N, threads);
    
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;
    
    printf("Computation time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / duration.count() / 1e9;
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
