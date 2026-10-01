#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Avoid oversized teams for small matrices while respecting the OpenMP limit.
static int matrixThreads(size_t n) {
    const size_t blocks = n / 64 + (n % 64 != 0);
    return static_cast<int>(std::min(static_cast<size_t>(omp_get_max_threads()),
                                     std::max(size_t{1}, blocks)));
}

// Blocked right-looking Cholesky. Each phase owns disjoint output rows or
// tiles; the OpenMP barriers publish them before the next dependent phase.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t block = 64;
    bool success = true;
    double* const data = A.data();

    #pragma omp parallel num_threads(matrixThreads(n)) shared(success)
    {
        for (size_t first = 0; first < n; first += block) {
            const size_t end = std::min(first + block, n);

            // Earlier panels have already updated this diagonal block.
            #pragma omp single
            {
                for (size_t i = first; i < end && success; ++i) {
                    for (size_t j = first; j <= i; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = first; k < j; ++k)
                            sum += data[i * n + k] * data[j * n + k];
                        const double value = data[i * n + j] - sum;
                        if (i == j) {
                            if (value <= 0.0) {
                                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                                success = false;
                                break;
                            }
                            data[i * n + j] = sqrt(value);
                        } else {
                            data[i * n + j] = value / data[j * n + j];
                        }
                    }
                }
            }
            // All threads observe the same status after the single barrier.
            if (!success) break;

            // Solve the rectangular panel, independently for each row.
            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = first; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = first; k < j; ++k)
                        sum += data[i * n + k] * data[j * n + k];
                    data[i * n + j] = (data[i * n + j] - sum) / data[j * n + j];
                }
            }

            // Tile the lower triangular rank-k update for cache reuse. Dynamic
            // scheduling balances the triangular domain and the shrinking tail.
            const size_t tiles = (n - end + block - 1) / block;
            #pragma omp for schedule(dynamic, 1)
            for (size_t tile = 0; tile < tiles * tiles; ++tile) {
                const size_t ti = tile / tiles;
                const size_t tj = tile % tiles;
                if (tj > ti) continue;
                const size_t ib = end + ti * block;
                const size_t jb = end + tj * block;
                const size_t ie = std::min(ib + block, n);
                const size_t je = std::min(jb + block, n);
                for (size_t i = ib; i < ie; ++i) {
                    const size_t stop = std::min(je, i + 1);
                    size_t j = jb;
                    // Four dot products share the same row of the panel.
                    for (; j + 3 < stop; j += 4) {
                        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                        #pragma omp simd reduction(+:s0,s1,s2,s3)
                        for (size_t k = first; k < end; ++k) {
                            const double x = data[i * n + k];
                            s0 += x * data[j * n + k];
                            s1 += x * data[(j + 1) * n + k];
                            s2 += x * data[(j + 2) * n + k];
                            s3 += x * data[(j + 3) * n + k];
                        }
                        data[i * n + j] -= s0;
                        data[i * n + j + 1] -= s1;
                        data[i * n + j + 2] -= s2;
                        data[i * n + j + 3] -= s3;
                    }
                    for (; j < stop; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = first; k < end; ++k)
                            sum += data[i * n + k] * data[j * n + k];
                        data[i * n + j] -= sum;
                    }
                }
            }
        }
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            if (success)
                std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
        }
    }
    return success;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Symmetry halves the work. Each pair has exactly one writer, and
    // keeping random generation serial preserves the original seed sequence.
    #pragma omp parallel for num_threads(matrixThreads(n)) schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            if (i == j) sum += n;
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Only the shared nonzero prefix contributes to L * L^T. Compare
    // directly to avoid allocating another dense matrix for validation.
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for num_threads(matrixThreads(n)) schedule(dynamic, 8) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t limit = std::min(i, j) + 1;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < limit; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
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
