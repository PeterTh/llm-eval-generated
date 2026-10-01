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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    constexpr size_t rows = 32, cols = 64, depth = 128;
    const size_t rowTiles = (N + rows - 1) / rows;
    const size_t colTiles = (N + cols - 1) / cols;

    #pragma omp parallel
    {
        // Private, cache-resident storage; padding also handles edge tiles.
        alignas(64) double packedB[depth][cols];
        alignas(64) double tile[rows][cols];

        // A tile has one owner for the entire reduction, so no synchronization
        // or atomic operations are needed when accumulating its elements.
        #pragma omp for collapse(2) schedule(static)
        for (size_t ti = 0; ti < rowTiles; ++ti) {
            for (size_t tj = 0; tj < colTiles; ++tj) {
                const size_t firstRow = ti * rows, firstCol = tj * cols;
                const size_t nr = std::min(rows, N - firstRow);
                const size_t nc = std::min(cols, N - firstCol);
                for (auto& row : tile)
                    std::fill_n(row, cols, 0.0);

                for (size_t kk = 0; kk < N; kk += depth) {
                    const size_t nk = std::min(depth, N - kk);
                    for (size_t k = 0; k < nk; ++k) {
                        #pragma omp simd
                        for (size_t j = 0; j < nc; ++j)
                            packedB[k][j] = B[(kk + k) * N + firstCol + j];
                        for (size_t j = nc; j < cols; ++j)
                            packedB[k][j] = 0.0;
                    }

                    for (size_t i = 0; i < nr; i += 4) {
                        const double* a0row = A.data() + (firstRow + i) * N + kk;
                        // Padded rows are never stored; reuse a valid input row.
                        const double* a1row = i + 1 < nr ? a0row + N : a0row;
                        const double* a2row = i + 2 < nr ? a0row + 2 * N : a0row;
                        const double* a3row = i + 3 < nr ? a0row + 3 * N : a0row;
                        for (size_t j = 0; j < nc; j += 8) {
                            // Four rows share each vector of B. The fixed-size
                            // accumulators let the compiler keep sums in registers.
                            double s0[8], s1[8], s2[8], s3[8];
                            #pragma omp simd
                            for (size_t v = 0; v < 8; ++v) {
                                s0[v] = tile[i][j + v];
                                s1[v] = tile[i + 1][j + v];
                                s2[v] = tile[i + 2][j + v];
                                s3[v] = tile[i + 3][j + v];
                            }
                            for (size_t k = 0; k < nk; ++k) {
                                const double a0 = a0row[k];
                                const double a1 = a1row[k];
                                const double a2 = a2row[k];
                                const double a3 = a3row[k];
                                #pragma omp simd
                                for (size_t v = 0; v < 8; ++v) {
                                    const double b = packedB[k][j + v];
                                    s0[v] += a0 * b;
                                    s1[v] += a1 * b;
                                    s2[v] += a2 * b;
                                    s3[v] += a3 * b;
                                }
                            }
                            #pragma omp simd
                            for (size_t v = 0; v < 8; ++v) {
                                tile[i][j + v] = s0[v];
                                tile[i + 1][j + v] = s1[v];
                                tile[i + 2][j + v] = s2[v];
                                tile[i + 3][j + v] = s3[v];
                            }
                        }
                    }
                }
                for (size_t i = 0; i < nr; ++i) {
                    #pragma omp simd
                    for (size_t j = 0; j < nc; ++j)
                        C[(firstRow + i) * N + firstCol + j] = tile[i][j];
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
