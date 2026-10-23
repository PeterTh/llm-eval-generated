#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Number of physical cores available to this process. Falls back to the
// logical CPU count when the topology cannot be determined.
unsigned physicalCoreCount() {
    const unsigned logical = std::thread::hardware_concurrency();
    if (logical == 0) {
        return 0;
    }

    unsigned width = 0;
    if (FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r")) {
        char line[256] = {};
        if (fgets(line, sizeof(line), f)) {
            unsigned siblings = 0;
            char* token = strtok(line, ",\n");
            while (token) {
                char* dash = strchr(token, '-');
                const unsigned first = static_cast<unsigned>(strtoul(token, nullptr, 10));
                unsigned last = first;
                if (dash) {
                    last = static_cast<unsigned>(strtoul(dash + 1, nullptr, 10));
                }
                if (last >= first) {
                    siblings += last - first + 1;
                }
                token = strtok(nullptr, ",\n");
            }
            width = siblings;
        }
        fclose(f);
    }

    if (width == 0 || width > logical) {
        return logical;
    }
    const unsigned cores = logical / width;
    if (cores < 2 && logical >= 4) {
        return logical;
    }
    return std::max(1u, cores);
}

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
    // Cache/register blocking: each tile holds MB rows of C and a NB-wide
    // column slice. A wider slice reuses each B row across more columns,
    // while narrower tiles keep enough independent work to scale across
    // many cores for small matrices.
    size_t MB = 8;
    size_t NB = 32;
    if (N >= 1024) {
        NB = 128;
    } else if (N < 256) {
        MB = 4;
        NB = 16;
    }

    // The k loop stays in ascending order, so every element is accumulated
    // in exactly the same order as the original implementation.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < N; ii += MB) {
        for (size_t jj = 0; jj < N; jj += NB) {
            const size_t iEnd = std::min(ii + MB, N);
            const size_t jEnd = std::min(jj + NB, N);

            for (size_t i = ii; i < iEnd; ++i) {
                const double* __restrict aRow = A.data() + i * N;
                double* __restrict cRow = C.data() + i * N;

                #pragma omp simd
                for (size_t j = jj; j < jEnd; ++j) {
                    cRow[j] = 0.0;
                }

                for (size_t k = 0; k < N; ++k) {
                    const double aik = aRow[k];
                    const double* __restrict bRow = B.data() + k * N;
                    if ((N & 1u) && k + 1 == N) {
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] = __builtin_fma(aik, bRow[j], cRow[j]);
                        }
                    } else {
                        #pragma omp simd
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += aik * bRow[j];
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
    // Avoid oversubscribing SMT siblings unless the user asked for a
    // specific thread count through the standard OpenMP environment.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const unsigned cores = physicalCoreCount();
        if (cores > 0) {
            omp_set_num_threads(static_cast<int>(cores));
        }
    }

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
