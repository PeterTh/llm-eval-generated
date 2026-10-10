#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
constexpr size_t innerBlock = 128;
constexpr size_t microColumns = 8;

// Keep independent output elements in registers while visiting k in order.
// SIMD runs across columns, so it does not reorder any dot-product reduction.
template <size_t Rows>
inline void multiplyMicroTile(const double* a, const double* b, double* c,
                              const size_t N, const size_t count,
                              const size_t columns, const bool firstPanel) {
    alignas(64) double sums[Rows][microColumns] = {};
    if (!firstPanel) {
        for (size_t r = 0; r < Rows; ++r) {
            #pragma omp simd
            for (size_t j = 0; j < columns; ++j) {
                sums[r][j] = c[r * N + j];
            }
        }
    }

    for (size_t k = 0; k < count; ++k) {
        const double* bRow = b + k * microColumns;
        for (size_t r = 0; r < Rows; ++r) {
            const double value = a[r * N + k];
            #pragma omp simd
            for (size_t j = 0; j < microColumns; ++j) {
                sums[r][j] += value * bRow[j];
            }
        }
    }

    for (size_t r = 0; r < Rows; ++r) {
        #pragma omp simd
        for (size_t j = 0; j < columns; ++j) {
            c[r * N + j] = sums[r][j];
        }
    }
}
} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    const size_t rowTiles = (N + rowBlock - 1) / rowBlock;
    const size_t columnTiles = (N + columnBlock - 1) / columnBlock;
    const size_t packedPanels = (N + microColumns - 1) / microColumns;

    // Pack B into narrow panels to avoid strided column accesses and keep each
    // inner block in cache. Padding lets the SIMD kernel handle edge columns.
    // Workers initialize their own packed panels, avoiding a serial zero-fill
    // and placing the scratch pages on the NUMA nodes that first use them.
    std::unique_ptr<double[]> packedB(new double[packedPanels * N * microColumns]);
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();
    double* packed = packedB.get();

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (size_t jp = 0; jp < packedPanels; ++jp) {
            for (size_t k = 0; k < N; ++k) {
                const size_t jBegin = jp * microColumns;
                const size_t columns = std::min(microColumns, N - jBegin);
                double* destination = packed + (jp * N + k) * microColumns;
                #pragma omp simd
                for (size_t j = 0; j < microColumns; ++j) {
                    destination[j] = j < columns ? b[k * N + jBegin + j] : 0.0;
                }
            }
        }

        // Each tile has a single owner for all k panels: no atomics, reductions,
        // or synchronization are needed while accumulating the output.
        #pragma omp for collapse(2) schedule(static)
        for (size_t ib = 0; ib < rowTiles; ++ib) {
            for (size_t jb = 0; jb < columnTiles; ++jb) {
                const size_t iBegin = ib * rowBlock;
                const size_t iEnd = std::min(iBegin + rowBlock, N);
                const size_t jBegin = jb * columnBlock;
                const size_t jEnd = std::min(jBegin + columnBlock, N);
                for (size_t kb = 0; kb < N; kb += innerBlock) {
                    const size_t count = std::min(innerBlock, N - kb);
                    for (size_t j = jBegin; j < jEnd; j += microColumns) {
                        const double* panel = packed +
                            ((j / microColumns) * N + kb) * microColumns;
                        const size_t columns = std::min(microColumns, jEnd - j);
                        size_t i = iBegin;
                        for (; i + 4 <= iEnd; i += 4) {
                            multiplyMicroTile<4>(a + i * N + kb, panel,
                                c + i * N + j, N, count, columns, kb == 0);
                        }
                        for (; i < iEnd; ++i) {
                            multiplyMicroTile<1>(a + i * N + kb, panel,
                                c + i * N + j, N, count, columns, kb == 0);
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
