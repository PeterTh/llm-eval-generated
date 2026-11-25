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

constexpr size_t panelWidth = 8;
constexpr size_t kernelRows = 6;
constexpr size_t tileRows = 48;
constexpr size_t tileCols = 64;
constexpr size_t depthBlock = 512;

struct PanelRow {
    double values[panelWidth];
};

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrices(double* A, PanelRow* B, const size_t N, const int threads) {
    // B stores one contiguous N-by-panelWidth panel for each group of columns.
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; j += panelWidth) {
            double* packed = B[j / panelWidth * N + i].values;
            const size_t width = std::min(panelWidth, N - j);
            for (size_t lane = 0; lane < width; ++lane) {
                const double value = getPseudoRndValue(N, i, j + lane);
                A[i * N + j + lane] = value;
                packed[lane] = value;
            }
            std::fill(packed + width, packed + panelWidth, 0.0);
        }
    }
}

// Preserve the original GCC x86 result hash. Its vector loop rounds the
// products separately and uses hardware FMA only for an odd last term.
#if defined(__GNUC__) && !defined(__clang__) && defined(__SSE2__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif

#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__)
constexpr bool fuseOddTail = true;
#else
constexpr bool fuseOddTail = false;
#endif

template <size_t Rows>
void multiplyPanel(const double* A, const PanelRow* B, double* C, const size_t N,
                   const size_t width, const size_t begin, const size_t end) {
    double sums[Rows][panelWidth] = {};
    if (begin != 0) {
        for (size_t row = 0; row < Rows; ++row) {
            std::copy_n(C + row * N, width, sums[row]);
        }
    }

    const bool oddTail = fuseOddTail && end == N && N % 2 != 0;
    const size_t bulkEnd = end - oddTail;
    for (size_t k = begin; k < bulkEnd; ++k) {
        const double* right = B[k].values;
        for (size_t row = 0; row < Rows; ++row) {
            const double left = A[row * N + k];
            #pragma omp simd
            for (size_t lane = 0; lane < panelWidth; ++lane) {
                sums[row][lane] += left * right[lane];
            }
        }
    }

    if (oddTail) {
        const double* right = B[bulkEnd].values;
        for (size_t row = 0; row < Rows; ++row) {
            const double left = A[row * N + bulkEnd];
            #pragma omp simd
            for (size_t lane = 0; lane < panelWidth; ++lane) {
                sums[row][lane] = std::fma(left, right[lane], sums[row][lane]);
            }
        }
    }
    for (size_t row = 0; row < Rows; ++row) {
        std::copy_n(sums[row], width, C + row * N);
    }
}

#if defined(__GNUC__) && !defined(__clang__) && defined(__SSE2__)
#pragma GCC pop_options
#endif

void matrixMultiply(const double* A, const PanelRow* B, double* C, const size_t N,
                    const int threads) {
    // Each work item owns its output tile for the entire k loop.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (size_t ii = 0; ii < N; ii += tileRows) {
        for (size_t jj = 0; jj < N; jj += tileCols) {
            const size_t iEnd = std::min(ii + tileRows, N);
            const size_t jEnd = std::min(jj + tileCols, N);
            for (size_t kk = 0; kk < N; kk += depthBlock) {
                const size_t kEnd = std::min(kk + depthBlock, N);
                for (size_t j = jj; j < jEnd; j += panelWidth) {
                    const size_t width = std::min(panelWidth, jEnd - j);
                    const PanelRow* panel = B + j / panelWidth * N;
                    size_t i = ii;
                    for (; i + kernelRows <= iEnd; i += kernelRows) {
                        multiplyPanel<kernelRows>(A + i * N, panel, C + i * N + j,
                                                  N, width, kk, kEnd);
                    }
                    for (; i < iEnd; ++i) {
                        multiplyPanel<1>(A + i * N, panel, C + i * N + j,
                                         N, width, kk, kEnd);
                    }
                }
            }
        }
    }
}

// Recompute the sampled entries as independent dot products.
bool validateResult(const double* A, const PanelRow* B, const std::vector<double>& C,
                    const size_t N) {
    if (N == 0) {
        return true;
    }
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            const PanelRow* panel = B + j / panelWidth * N;
            const size_t lane = j % panelWidth;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * panel[k].values[lane];
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

} // namespace

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
    const size_t panels = (N + panelWidth - 1) / panelWidth;
    auto A = std::make_unique_for_overwrite<double[]>(N * N);
    auto B = std::make_unique_for_overwrite<PanelRow[]>(N * panels);
    std::vector<double> C(N * N);

    // Give each worker enough arithmetic to amortize team overhead on small jobs.
    constexpr double operationsPerThread = 4.0e6;
    const double operations = 2.0 * N * N * N;
    const int threads = static_cast<int>(std::clamp(
        operations / operationsPerThread, 1.0, static_cast<double>(omp_get_max_threads())));
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrices(A.get(), B.get(), N, threads);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A.get(), B.get(), C.data(), N, threads);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double gflops = operations / seconds / 1e9;
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
