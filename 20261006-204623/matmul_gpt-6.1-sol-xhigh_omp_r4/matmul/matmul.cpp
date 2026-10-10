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

namespace {
constexpr size_t rowBlock = 128;
constexpr size_t columnBlock = 64;
constexpr size_t innerBlock = 128;
constexpr size_t microRows = 2;
constexpr size_t vectorColumns = 16;

int matrixThreadCount(const size_t N) {
    const size_t rowTiles = N / rowBlock + (N % rowBlock != 0);
    const size_t columnTiles = N / columnBlock + (N % columnBlock != 0);
    // Reuse the same team for initialization and multiplication, and avoid
    // creating idle workers when there are fewer output tiles than threads.
    return static_cast<int>(std::min(static_cast<size_t>(omp_get_max_threads()),
        std::max(size_t{1}, rowTiles * columnTiles)));
}

// Keep two rows of independent accumulators in SIMD registers. Only columns
// are vectorized: each element's products are added in increasing k order.
template <size_t Rows>
inline void multiplyMicroTile(const double* a, const double* packedB,
                              double* c, const size_t N, const size_t depth) {
    alignas(64) double sums[Rows][vectorColumns];
    for (size_t r = 0; r < Rows; ++r) {
        #pragma omp simd
        for (size_t j = 0; j < vectorColumns; ++j) {
            sums[r][j] = c[r * N + j];
        }
    }

    for (size_t k = 0; k < depth; ++k) {
        for (size_t r = 0; r < Rows; ++r) {
            const double value = a[r * N + k];
            #pragma omp simd
            for (size_t j = 0; j < vectorColumns; ++j) {
                sums[r][j] += value * packedB[k * vectorColumns + j];
            }
        }
    }

    for (size_t r = 0; r < Rows; ++r) {
        #pragma omp simd
        for (size_t j = 0; j < vectorColumns; ++j) {
            c[r * N + j] = sums[r][j];
        }
    }
}
} // namespace

void initMatrix(std::vector<double>& mat, const size_t N) {
    const int threads = matrixThreadCount(N);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    const size_t rowTiles = N / rowBlock + (N % rowBlock != 0);
    const size_t columnTiles = N / columnBlock + (N % columnBlock != 0);
    const size_t tileCount = rowTiles * columnTiles;
    const int threads = matrixThreadCount(N);
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();

    #pragma omp parallel num_threads(threads)
    {
        // Private, bounded scratch space avoids allocation and synchronization
        // inside the multiplication. Packing removes B's long row stride from
        // the innermost loop and lets multiple rows reuse each packed B panel.
        alignas(64) double packedB[innerBlock * columnBlock];

        // A worker owns an entire output tile, including all k blocks. There
        // are no shared accumulators, atomics, or floating-point reductions.
        #pragma omp for schedule(static)
        for (size_t tile = 0; tile < tileCount; ++tile) {
            const size_t iBegin = (tile / columnTiles) * rowBlock;
            const size_t jBegin = (tile % columnTiles) * columnBlock;
            const size_t rows = std::min(rowBlock, N - iBegin);
            const size_t columns = std::min(columnBlock, N - jBegin);
            const size_t fullColumns = columns - columns % vectorColumns;

            for (size_t i = 0; i < rows; ++i) {
                #pragma omp simd
                for (size_t j = 0; j < columns; ++j) {
                    c[(iBegin + i) * N + jBegin + j] = 0.0;
                }
            }

            for (size_t kBegin = 0; kBegin < N; kBegin += innerBlock) {
                const size_t depth = std::min(innerBlock, N - kBegin);
                for (size_t j = 0; j < fullColumns; j += vectorColumns) {
                    double* panel = packedB + j * innerBlock;
                    for (size_t k = 0; k < depth; ++k) {
                        #pragma omp simd
                        for (size_t lane = 0; lane < vectorColumns; ++lane) {
                            panel[k * vectorColumns + lane] =
                                b[(kBegin + k) * N + jBegin + j + lane];
                        }
                    }
                }

                for (size_t j = 0; j < fullColumns; j += vectorColumns) {
                    const double* panel = packedB + j * innerBlock;
                    size_t i = 0;
                    for (; i + microRows <= rows; i += microRows) {
                        multiplyMicroTile<microRows>(a + (iBegin + i) * N + kBegin,
                                                    panel, c + (iBegin + i) * N + jBegin + j,
                                                    N, depth);
                    }
                    for (; i < rows; ++i) {
                        multiplyMicroTile<1>(a + (iBegin + i) * N + kBegin,
                                             panel, c + (iBegin + i) * N + jBegin + j,
                                             N, depth);
                    }
                }

                // Handle an incomplete SIMD tile without touching padding or
                // adding extra products, including for matrices smaller than 16.
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = fullColumns; j < columns; ++j) {
                        double sum = c[(iBegin + i) * N + jBegin + j];
                        for (size_t k = 0; k < depth; ++k) {
                            sum += a[(iBegin + i) * N + kBegin + k] *
                                   b[(kBegin + k) * N + jBegin + j];
                        }
                        c[(iBegin + i) * N + jBegin + j] = sum;
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
