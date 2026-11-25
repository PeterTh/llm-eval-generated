#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrices(std::vector<double>& A, std::vector<double>& B, const size_t N) {
    const int threads = static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                                        1 + N * N / 32768));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const double value = getPseudoRndValue(N, i, j);
            A[i * N + j] = value;
            B[i * N + j] = value;
        }
    }
}

namespace {

constexpr size_t panelWidth = 16;
constexpr size_t rowBlock = 48;
constexpr size_t columnBlock = 64;
constexpr size_t depthBlock = 128;
constexpr size_t kernelRows = 3;
constexpr double updatesPerThread = 4194304.0;

template <size_t Rows, bool FullPanel>
void multiplyPanel(const double* a, const double* packed, double* c, size_t n,
                   size_t depth, size_t columns, bool firstBlock, bool lastOdd) {
    alignas(64) double sums[Rows][panelWidth] = {};
    if constexpr (FullPanel) {
        columns = panelWidth;
    }
    if (!firstBlock) {
        for (size_t r = 0; r < Rows; ++r) {
            std::copy_n(c + r * n, columns, sums[r]);
        }
    }

    for (size_t k = 0; k < depth; ++k) {
        const double* b = packed + k * panelWidth;
        for (size_t r = 0; r < Rows; ++r) {
            const double value = a[r * n + k];
            #pragma omp simd
            for (size_t j = 0; j < panelWidth; ++j) {
                sums[r][j] += value * b[j];
            }
        }
    }

    // Match GCC's scalar remainder for the original dot product on FMA targets.
    if (lastOdd) {
        const double* b = packed + depth * panelWidth;
        for (size_t r = 0; r < Rows; ++r) {
            const double value = a[r * n + depth];
            #pragma omp simd
            for (size_t j = 0; j < panelWidth; ++j) {
                #if defined(__FMA__)
                sums[r][j] = std::fma(value, b[j], sums[r][j]);
                #else
                sums[r][j] += value * b[j];
                #endif
            }
        }
    }

    for (size_t r = 0; r < Rows; ++r) {
        std::copy_n(sums[r], columns, c + r * n);
    }
}

}  // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    const size_t panels = (N + panelWidth - 1) / panelWidth;
    const auto packedB = std::make_unique_for_overwrite<double[]>(panels * N * panelWidth);
    const size_t tiles = ((N + rowBlock - 1) / rowBlock) *
                         ((N + columnBlock - 1) / columnBlock);
    // Leave enough arithmetic per worker to amortize team startup and packing.
    const double work = static_cast<double>(N) * N * N;
    const int threads = static_cast<int>(std::min({static_cast<double>(omp_get_max_threads()),
                                                  static_cast<double>(tiles),
                                                  1.0 + work / updatesPerThread}));

    #pragma omp parallel num_threads(threads)
    {
        // Store each column panel contiguously along k for reuse by all row tiles.
        #pragma omp for collapse(2) schedule(static)
        for (size_t panel = 0; panel < panels; ++panel) {
            for (size_t kk = 0; kk < N; kk += depthBlock) {
                const size_t j = panel * panelWidth;
                const size_t columns = std::min(panelWidth, N - j);
                const size_t depthEnd = std::min(kk + depthBlock, N);
                double* dest = packedB.get() + (j * N + kk * panelWidth);
                if (columns == panelWidth) {
                    for (size_t k = kk; k < depthEnd; ++k, dest += panelWidth) {
                        std::copy_n(B.data() + k * N + j, panelWidth, dest);
                    }
                } else {
                    for (size_t k = kk; k < depthEnd; ++k, dest += panelWidth) {
                        std::copy_n(B.data() + k * N + j, columns, dest);
                        std::fill(dest + columns, dest + panelWidth, 0.0);
                    }
                }
            }
        }

        // A tile has a single writer, and each dot product accumulates in k order.
        #pragma omp for collapse(2) schedule(static)
        for (size_t ii = 0; ii < N; ii += rowBlock) {
            for (size_t jj = 0; jj < N; jj += columnBlock) {
                const size_t rowEnd = std::min(ii + rowBlock, N);
                const size_t columnEnd = std::min(jj + columnBlock, N);
                for (size_t kk = 0; kk < N; kk += depthBlock) {
                    const size_t depthEnd = std::min(kk + depthBlock, N);
                    const bool lastOdd = depthEnd == N && N % 2 != 0;
                    const size_t depth = depthEnd - kk - lastOdd;
                    for (size_t j = jj; j < columnEnd; j += panelWidth) {
                        const auto* packed = packedB.get() + (j * N + kk * panelWidth);
                        const size_t columns = std::min(panelWidth, N - j);
                        size_t i = ii;
                        for (; i + kernelRows <= rowEnd; i += kernelRows) {
                            const auto* a = A.data() + i * N + kk;
                            auto* c = C.data() + i * N + j;
                            if (columns == panelWidth) {
                                multiplyPanel<kernelRows, true>(a, packed, c, N, depth, columns, kk == 0, lastOdd);
                            } else {
                                multiplyPanel<kernelRows, false>(a, packed, c, N, depth, columns, kk == 0, lastOdd);
                            }
                        }
                        for (; i < rowEnd; ++i) {
                            const auto* a = A.data() + i * N + kk;
                            auto* c = C.data() + i * N + j;
                            multiplyPanel<1, false>(a, packed, c, N, depth, columns, kk == 0, lastOdd);
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
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrices(A, B, N);
    
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
