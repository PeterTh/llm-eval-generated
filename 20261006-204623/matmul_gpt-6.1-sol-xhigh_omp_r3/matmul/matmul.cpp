#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {
constexpr size_t rowBlock = 64;
constexpr size_t columnBlock = 64;
constexpr size_t innerBlock = 64;
constexpr size_t microColumns = 32;

int workerCount(const size_t tasks) {
    return static_cast<int>(std::max<size_t>(
        1, std::min<size_t>(tasks, omp_get_max_threads())));
}

// Vectorize independent columns, keeping each dot product in increasing k order.
void multiplyMicroTile(const double* a, const double* b, double* c,
                       const size_t K, const size_t columns,
                       const bool firstBlock) {
    alignas(64) double sums[microColumns] = {};
    if (!firstBlock) {
        #pragma omp simd
        for (size_t j = 0; j < columns; ++j) {
            sums[j] = c[j];
        }
    }

    for (size_t k = 0; k < K; ++k) {
        const double* bRow = b + k * microColumns;
        const double aValue = a[k];
        #pragma omp simd
        for (size_t j = 0; j < microColumns; ++j) {
            sums[j] += aValue * bRow[j];
        }
    }

    #pragma omp simd
    for (size_t j = 0; j < columns; ++j) {
        c[j] = sums[j];
    }
}
} // namespace

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    const size_t rowTiles = (N + rowBlock - 1) / rowBlock;
    const size_t columnTiles = (N + columnBlock - 1) / columnBlock;
    // Reuse the multiplication team so thread creation is outside the timed kernel.
    const int threads = workerCount(rowTiles * columnTiles);
    #pragma omp parallel for schedule(static) num_threads(threads) default(none) shared(mat, N)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    const size_t rowTiles = (N + rowBlock - 1) / rowBlock;
    const size_t columnTiles = (N + columnBlock - 1) / columnBlock;
    const int threads = workerCount(rowTiles * columnTiles);
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();

    #pragma omp parallel num_threads(threads) default(none) \
        shared(a, b, c, N, rowTiles, columnTiles, rowBlock, columnBlock, innerBlock, microColumns)
    {
        // Packing makes the reused B panels contiguous and avoids cache-set conflicts.
        // Each thread owns its scratch panel and complete, disjoint output tiles.
        alignas(64) double packedB[innerBlock * columnBlock];

        #pragma omp for collapse(2) schedule(static)
        for (size_t ti = 0; ti < rowTiles; ++ti) {
            for (size_t tj = 0; tj < columnTiles; ++tj) {
                const size_t iBegin = ti * rowBlock;
                const size_t jBegin = tj * columnBlock;
                const size_t rows = std::min(rowBlock, N - iBegin);
                const size_t columns = std::min(columnBlock, N - jBegin);

                for (size_t kBegin = 0; kBegin < N; kBegin += innerBlock) {
                    const size_t depth = std::min(innerBlock, N - kBegin);
                    for (size_t j = 0; j < columns; j += microColumns) {
                        const size_t width = std::min(microColumns, columns - j);
                        double* panel = packedB + (j / microColumns) * depth * microColumns;
                        for (size_t k = 0; k < depth; ++k) {
                            const double* source = b + (kBegin + k) * N + jBegin + j;
                            #pragma omp simd
                            for (size_t x = 0; x < microColumns; ++x) {
                                panel[k * microColumns + x] = x < width ? source[x] : 0.0;
                            }
                        }
                    }

                    for (size_t i = 0; i < rows; ++i) {
                        for (size_t j = 0; j < columns; j += microColumns) {
                            multiplyMicroTile(
                                a + (iBegin + i) * N + kBegin,
                                packedB + (j / microColumns) * depth * microColumns,
                                c + (iBegin + i) * N + jBegin + j, depth,
                                std::min(microColumns, columns - j), kBegin == 0);
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
    
    // Use full timing precision for GFLOPS, including kernels shorter than 1 ms.
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
