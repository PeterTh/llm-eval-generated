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

constexpr size_t kernelRows = 4;
constexpr size_t kernelColumns = 8;
constexpr size_t tileRows = 64;
constexpr size_t tileColumns = 64;
constexpr size_t tileDepth = 128;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrices(double* A, double* B, const size_t N) {
    #pragma omp for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const double value = getPseudoRndValue(N, i, j);
            A[i * N + j] = value;
            B[i * N + j] = value;
        }
    }
}

int threadCount(const size_t N) {
    const size_t rowTiles = (N + tileRows - 1) / tileRows;
    const size_t columnTiles = (N + tileColumns - 1) / tileColumns;
    const auto maxThreads = static_cast<size_t>(omp_get_max_threads());
    // Amortize team overhead on small matrices.
    constexpr double productsPerThread = 8.0 * 1024 * 1024;
    const double usefulThreads = std::ceil(static_cast<double>(N) * N * N / productsPerThread);
    const auto workLimit = static_cast<size_t>(
        std::min(usefulThreads, static_cast<double>(maxThreads)));
    return static_cast<int>(std::max<size_t>(1, std::min(rowTiles * columnTiles, workLimit)));
}

template <size_t Rows>
void multiplyPanel(const double* a, const double (*b)[kernelColumns],
                   double* c, const size_t N, const size_t depth) {
    alignas(64) double sums[Rows][kernelColumns];
    for (size_t i = 0; i < Rows; ++i) {
        std::copy_n(c + i * tileColumns, kernelColumns, sums[i]);
    }

    // SIMD lanes are independent output columns. Each sum visits k in order.
    for (size_t k = 0; k < depth; ++k) {
        for (size_t i = 0; i < Rows; ++i) {
            const double value = a[i * N + k];
            #pragma omp simd
            for (size_t j = 0; j < kernelColumns; ++j) {
                sums[i][j] += value * b[k][j];
            }
        }
    }

    for (size_t i = 0; i < Rows; ++i) {
        std::copy_n(sums[i], kernelColumns, c + i * tileColumns);
    }
}

void matrixMultiply(const double* A, const double* B, double* C, const size_t N) {
    #pragma omp for collapse(2) schedule(static)
    for (size_t row = 0; row < N; row += tileRows) {
        for (size_t column = 0; column < N; column += tileColumns) {
            const size_t rows = std::min(tileRows, N - row);
            const size_t columns = std::min(tileColumns, N - column);
            const size_t fullPanels = columns / kernelColumns;
            const size_t extraColumns = columns % kernelColumns;
            const size_t panels = fullPanels + (extraColumns != 0);
            alignas(64) double tile[tileRows * tileColumns]{};
            alignas(64) double packedB[tileColumns / kernelColumns][tileDepth][kernelColumns];

            // Each thread owns its output tile and reuses each packed B panel across its rows.
            for (size_t firstK = 0; firstK < N; firstK += tileDepth) {
                const size_t depth = std::min(tileDepth, N - firstK);
                for (size_t k = 0; k < depth; ++k) {
                    const double* source = B + (firstK + k) * N + column;
                    for (size_t panel = 0; panel < fullPanels; ++panel) {
                        std::copy_n(source + panel * kernelColumns, kernelColumns, packedB[panel][k]);
                    }
                    if (extraColumns != 0) {
                        double* tail = packedB[fullPanels][k];
                        std::copy_n(source + fullPanels * kernelColumns, extraColumns, tail);
                        std::fill(tail + extraColumns, tail + kernelColumns, 0.0);
                    }
                }

                for (size_t panel = 0; panel < panels; ++panel) {
                    const size_t j = panel * kernelColumns;
                    size_t i = 0;
                    for (; i + kernelRows <= rows; i += kernelRows) {
                        multiplyPanel<kernelRows>(A + (row + i) * N + firstK, packedB[panel],
                                                  tile + i * tileColumns + j, N, depth);
                    }
                    for (; i < rows; ++i) {
                        multiplyPanel<1>(A + (row + i) * N + firstK, packedB[panel],
                                         tile + i * tileColumns + j, N, depth);
                    }
                }
            }

            for (size_t i = 0; i < rows; ++i) {
                std::copy_n(tile + i * tileColumns, columns, C + (row + i) * N + column);
            }
        }
    }
}

// Compare selected elements with a scalar dot product.
bool validateResult(const double* A, const double* B,
                   const std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return true;
    }
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

} // namespace

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
    auto A = std::make_unique_for_overwrite<double[]>(N * N);
    auto B = std::make_unique_for_overwrite<double[]>(N * N);
    std::vector<double> C(N * N);
    const int threads = threadCount(N);
    
    using Clock = std::chrono::steady_clock;
    Clock::time_point start;
    Clock::time_point end;

    printf("Initializing matrices...\n");
    // Keep the team active between first-touch initialization and the timed kernel.
    #pragma omp parallel num_threads(threads)
    {
        initMatrices(A.get(), B.get(), N);

        #pragma omp single
        {
            printf("Computing matrix multiplication...\n");
            start = Clock::now();
        }

        matrixMultiply(A.get(), B.get(), C.data(), N);

        #pragma omp master
        end = Clock::now();
    }

    const auto elapsed = end - start;
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    const double gflops = (2.0 * N * N * N) / std::chrono::duration<double>(elapsed).count() / 1e9;
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
