#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky, using the lower triangle in row-major order.
// One persistent OpenMP team avoids launching threads for every panel.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    double* const data = A.data();
    bool success = true;

    #pragma omp parallel shared(success)
    {
        for (size_t panel = 0; panel < n; panel += blockSize) {
            const size_t end = std::min(panel + blockSize, n);

            // The small diagonal block depends on preceding columns.
            #pragma omp single
            {
                for (size_t i = panel; i < end && success; ++i) {
                    double* const row = data + i * n;
                    for (size_t j = panel; j <= i; ++j) {
                        const double* const other = data + j * n;
                        double sum = 0.0;
                        for (size_t k = panel; k < j; ++k) {
                            sum += row[k] * other[k];
                        }
                        const double value = row[j] - sum;
                        if (i == j) {
                            if (value <= 0.0) {
                                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                                success = false;
                                break;
                            }
                            row[j] = sqrt(value);
                        } else {
                            row[j] = value / other[j];
                        }
                    }
                }
            } // Barrier publishes both the factor and the failure flag.
            if (!success) break;

            // Solve the panel below the diagonal block; rows are independent.
            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                double* const row = data + i * n;
                for (size_t j = panel; j < end; ++j) {
                    const double* const other = data + j * n;
                    double sum = 0.0;
                    for (size_t k = panel; k < j; ++k) {
                        sum += row[k] * other[k];
                    }
                    row[j] = (row[j] - sum) / other[j];
                }
            }

            // Each tile has a single owner. Keep its updates in cache and
            // vectorize across contiguous columns, without a sum reduction.
            const size_t tiles = (n - end + blockSize - 1) / blockSize;
            #pragma omp for collapse(2) schedule(dynamic, 1)
            for (size_t ti = 0; ti < tiles; ++ti) {
                for (size_t tj = 0; tj < tiles; ++tj) {
                    if (tj > ti) continue;
                    const size_t ib = end + ti * blockSize;
                    const size_t jb = end + tj * blockSize;
                    const size_t ie = std::min(ib + blockSize, n);
                    const size_t je = std::min(jb + blockSize, n);
                    // Transpose the panel tile into a small thread-local buffer
                    // so both the update and its input have unit-stride access.
                    alignas(64) double transposed[blockSize * blockSize];
                    const size_t width = je - jb;
                    for (size_t k = panel; k < end; ++k) {
                        for (size_t j = jb; j < je; ++j) {
                            transposed[(k - panel) * blockSize + j - jb] = data[j * n + k];
                        }
                    }
                    size_t i = ib;
                    // A 4-by-8 microtile keeps accumulators in SIMD registers
                    // throughout the panel update, reducing loads and stores.
                    for (; i + 4 <= ie; i += 4) {
                        size_t j = 0;
                        for (; j + 8 <= width; j += 8) {
                            double c0[8], c1[8], c2[8], c3[8];
                            #pragma omp simd
                            for (size_t lane = 0; lane < 8; ++lane) {
                                c0[lane] = data[i * n + jb + j + lane];
                                c1[lane] = data[(i + 1) * n + jb + j + lane];
                                c2[lane] = data[(i + 2) * n + jb + j + lane];
                                c3[lane] = data[(i + 3) * n + jb + j + lane];
                            }
                            for (size_t k = panel; k < end; ++k) {
                                const double a0 = data[i * n + k];
                                const double a1 = data[(i + 1) * n + k];
                                const double a2 = data[(i + 2) * n + k];
                                const double a3 = data[(i + 3) * n + k];
                                const double* const input = transposed + (k - panel) * blockSize + j;
                                #pragma omp simd
                                for (size_t lane = 0; lane < 8; ++lane) {
                                    const double b = input[lane];
                                    c0[lane] -= a0 * b;
                                    c1[lane] -= a1 * b;
                                    c2[lane] -= a2 * b;
                                    c3[lane] -= a3 * b;
                                }
                            }
                            #pragma omp simd
                            for (size_t lane = 0; lane < 8; ++lane) {
                                data[i * n + jb + j + lane] = c0[lane];
                                data[(i + 1) * n + jb + j + lane] = c1[lane];
                                data[(i + 2) * n + jb + j + lane] = c2[lane];
                                data[(i + 3) * n + jb + j + lane] = c3[lane];
                            }
                        }
                        for (size_t r = i; r < i + 4; ++r) {
                            for (size_t k = panel; k < end; ++k) {
                                const double factor = data[r * n + k];
                                const double* const input = transposed + (k - panel) * blockSize;
                                #pragma omp simd
                                for (size_t tail = j; tail < width; ++tail) {
                                    data[r * n + jb + tail] -= factor * input[tail];
                                }
                            }
                        }
                    }
                    for (; i < ie; ++i) {
                        double* const target = data + i * n + jb;
                        for (size_t k = panel; k < end; ++k) {
                            const double factor = data[i * n + k];
                            const double* const input = transposed + (k - panel) * blockSize;
                            #pragma omp simd
                            for (size_t j = 0; j < width; ++j) {
                                target[j] -= factor * input[j];
                            }
                        }
                    }
                }
            } // All updates must finish before factoring the next panel.
        }

        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
            }
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
    
    // Symmetry halves the work. Each pair is written by exactly one thread;
    // the dot products retain the original accumulation order.
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            if (i == j) sum += n;
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Reconstruct only the nonzero portion of each row, and compare on the
    // fly to avoid allocating another full matrix.
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
            const double mirroredError = fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, mirroredError);
            relError = std::max(relError, mirroredError / (fabs(A_orig[j * n + i]) + 1e-10));
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
