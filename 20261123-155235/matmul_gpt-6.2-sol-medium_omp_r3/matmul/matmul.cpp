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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

namespace {

constexpr size_t tileRows = 30;
constexpr size_t tileColumns = 64;
constexpr size_t blockDepth = 512;
constexpr size_t packingRows = 64;
constexpr size_t kernelRows = 6;
constexpr size_t kernelColumns = 8;

int workerCount(const size_t N) {
    const size_t rowTiles = (N + tileRows - 1) / tileRows;
    const size_t columnTiles = (N + tileColumns - 1) / tileColumns;
    // Amortize team overhead for small products and avoid idle workers.
    constexpr double workPerThread = 4.0 * 1024 * 1024;
    const double usefulWorkers = std::min<double>(rowTiles * columnTiles,
        std::max(1.0, std::ceil(static_cast<double>(N) * N * N / workPerThread)));
    return static_cast<int>(std::clamp(usefulWorkers, 1.0,
                                      static_cast<double>(omp_get_max_threads())));
}

void initMatrices(double* A, double* B, const size_t N, const int workers) {
    // Both inputs use the same deterministic generator.
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const double value = getPseudoRndValue(N, i, j);
            A[i * N + j] = value;
            B[i * N + j] = value;
        }
    }
}

template <size_t Rows, size_t Columns>
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void multiplyBlock(const double* a, const double* packed, double* c, const size_t N,
                   const size_t firstK, const size_t lastK, const size_t width) {
    const size_t columns = Columns == kernelColumns ? kernelColumns : width;
    double sums[Rows][kernelColumns];
    for (size_t row = 0; row < Rows; ++row) {
        #pragma omp simd
        for (size_t j = 0; j < columns; ++j) {
            sums[row][j] = firstK == 0 ? 0.0 : c[row * N + j];
        }
    }

#if defined(__GNUC__) && !defined(__clang__) && defined(FP_FAST_FMA)
    // GCC's original dot product rounds the vectorized pairs, then
    // contracts the final scalar multiply-add for odd N.
    const size_t pairedEnd = N & ~size_t{1};
    const size_t end = std::min(lastK, pairedEnd);
#else
    const size_t end = lastK;
#endif
    for (size_t k = firstK; k < end; ++k) {
        const double* b = packed + k * kernelColumns;
        for (size_t row = 0; row < Rows; ++row) {
            const double value = a[row * N + k];
            #pragma omp simd
            for (size_t j = 0; j < columns; ++j) {
                sums[row][j] += value * b[j];
            }
        }
    }

#if defined(__GNUC__) && !defined(__clang__) && defined(FP_FAST_FMA)
    if (end != lastK) {
        const double* b = packed + end * kernelColumns;
        for (size_t row = 0; row < Rows; ++row) {
            const double value = a[row * N + end];
            #pragma omp simd
            for (size_t j = 0; j < columns; ++j) {
                sums[row][j] = std::fma(value, b[j], sums[row][j]);
            }
        }
    }
#endif

    for (size_t row = 0; row < Rows; ++row) {
        #pragma omp simd
        for (size_t j = 0; j < columns; ++j) {
            c[row * N + j] = sums[row][j];
        }
    }
}

template <size_t Columns>
void multiplyPanel(const double* a, const double* packed, double* c, const size_t N,
                   const size_t rows, const size_t firstK, const size_t lastK,
                   const size_t columns) {
    size_t row = 0;
    for (; row + kernelRows <= rows; row += kernelRows) {
        multiplyBlock<kernelRows, Columns>(a + row * N, packed, c + row * N,
                                          N, firstK, lastK, columns);
    }
    for (; row < rows; ++row) {
        multiplyBlock<1, Columns>(a + row * N, packed, c + row * N,
                                  N, firstK, lastK, columns);
    }
}

void matrixMultiply(const double* A, const double* B, double* C, const size_t N,
                    const int workers) {
    const size_t panels = (N + kernelColumns - 1) / kernelColumns;
    const std::unique_ptr<double[]> packedB(new double[panels * N * kernelColumns]);

    #pragma omp parallel num_threads(workers)
    {
        // Each group of kernelColumns columns has its own contiguous N-row panel.
        #pragma omp for collapse(2) schedule(static)
        for (size_t j = 0; j < N; j += tileColumns) {
            for (size_t kk = 0; kk < N; kk += packingRows) {
                const size_t lastJ = std::min(j + tileColumns, N);
                const size_t lastFullJ = lastJ - lastJ % kernelColumns;
                const size_t lastK = std::min(kk + packingRows, N);
                for (size_t k = kk; k < lastK; ++k) {
                    for (size_t col = j; col < lastFullJ; col += kernelColumns) {
                        std::copy_n(B + k * N + col, kernelColumns,
                                    packedB.get() + col * N + k * kernelColumns);
                    }
                    if (lastFullJ < lastJ) {
                        std::copy_n(B + k * N + lastFullJ, lastJ - lastFullJ,
                                    packedB.get() + lastFullJ * N + k * kernelColumns);
                    }
                }
            }
        }

        // Each worker owns its C tiles for the entire ascending k traversal.
        #pragma omp for collapse(2) schedule(static)
        for (size_t i = 0; i < N; i += tileRows) {
            for (size_t j = 0; j < N; j += tileColumns) {
                const size_t rows = std::min(tileRows, N - i);
                const size_t lastJ = std::min(j + tileColumns, N);
                const double* a = A + i * N;
                for (size_t kk = 0; kk < N; kk += blockDepth) {
                    const size_t lastK = std::min(kk + blockDepth, N);
                    for (size_t col = j; col < lastJ; col += kernelColumns) {
                        const double* packed = packedB.get() + col * N;
                        double* c = C + i * N + col;
                        const size_t width = std::min(kernelColumns, N - col);
                        if (width == kernelColumns) {
                            multiplyPanel<kernelColumns>(a, packed, c, N, rows, kk, lastK, width);
                        } else {
                            multiplyPanel<0>(a, packed, c, N, rows, kk, lastK, width);
                        }
                    }
                }
            }
        }
    }
}

} // namespace

bool validateResult(const double* A, const double* B, const std::vector<double>& C,
                   const size_t N) {
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
    const std::unique_ptr<double[]> A(new double[N * N]);
    const std::unique_ptr<double[]> B(new double[N * N]);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    const int workers = workerCount(N);
    initMatrices(A.get(), B.get(), N, workers);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A.get(), B.get(), C.data(), N, workers);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * N * N * N) / seconds / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A.get(), B.get(), C, N);
        
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
