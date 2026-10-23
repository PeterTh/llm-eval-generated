#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#if defined(__linux__)
#include <sched.h>
#endif

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The trailing submatrix update is the dominant O(n^3) cost and is distributed
// over independent tiles. A single parallel region spans the whole
// factorization so that thread-team setup happens only once.

namespace {

inline int defaultThreadCount() {
    const int procs = omp_get_num_procs();
    int smt = 1;
#if defined(__linux__)
    FILE* f = std::fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (f != nullptr) {
        int count = 0;
        for (;;) {
            int first = 0;
            if (std::fscanf(f, "%d", &first) != 1) break;
            int last = first;
            int c = std::fgetc(f);
            if (c == '-') {
                if (std::fscanf(f, "%d", &last) != 1) break;
                c = std::fgetc(f);
            }
            count += last - first + 1;
            if (c == '\n' || c == EOF) break;
        }
        std::fclose(f);
        if (count > 0) smt = count;
    }
#endif
    const int physical = procs / smt;
    return physical > 0 ? physical : procs;
}


#if defined(__linux__)
inline void bindThreadIfNeeded(const int maxUnboundThreads) {
    if (omp_get_proc_bind() != omp_proc_bind_false) return;
    const int nthreads = omp_get_num_threads();
    if (nthreads <= maxUnboundThreads) return;

    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    const int ncpu = CPU_COUNT(&allowed);
    if (ncpu <= 0) return;

    const int target = static_cast<int>(static_cast<long long>(omp_get_thread_num()) * ncpu / nthreads);
    int seen = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            if (++seen == target) {
                cpu_set_t one;
                CPU_ZERO(&one);
                CPU_SET(cpu, &one);
                sched_setaffinity(0, sizeof(one), &one);
                break;
            }
        }
    }
}
#endif

inline size_t chooseBlockSize(const size_t n) {
    return std::min<size_t>(64, std::max<size_t>(n, 1));
}

inline size_t rowTileFromIndex(const size_t t) {
    size_t it = static_cast<size_t>((std::sqrt(8.0 * static_cast<double>(t) + 1.0) - 1.0) * 0.5);
    while (it > 0 && it * (it + 1) / 2 > t) --it;
    while ((it + 1) * (it + 2) / 2 <= t) ++it;
    return it;
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;

    double* const a = A.data();
    const size_t nb = chooseBlockSize(n);
    const int maxUnboundThreads = defaultThreadCount();
    bool ok = true;

    #pragma omp parallel shared(a, n, nb, ok)
    {
#if defined(__linux__)
        bindThreadIfNeeded(maxUnboundThreads);
#endif
        for (size_t k0 = 0; k0 < n; k0 += nb) {
            const size_t kb = std::min(nb, n - k0);

            // Factor the diagonal block (small O(nb^3) work).
            #pragma omp single
            {
                for (size_t jj = 0; jj < kb; ++jj) {
                    const size_t j = k0 + jj;
                    double* rowj = a + j * n;

                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t p = 0; p < jj; ++p) {
                        const double v = rowj[k0 + p];
                        sum += v * v;
                    }
                    const double val = rowj[j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        ok = false;
                        break;
                    }
                    rowj[j] = sqrt(val);

                    for (size_t ii = jj + 1; ii < kb; ++ii) {
                        double* rowi = a + (k0 + ii) * n;
                        double s = 0.0;
                        #pragma omp simd reduction(+:s)
                        for (size_t p = 0; p < jj; ++p) {
                            s += rowi[k0 + p] * rowj[k0 + p];
                        }
                        rowi[j] = (rowi[j] - s) / rowj[j];
                    }
                }
            }
            if (!ok) break;

            // Solve the panel below the diagonal block.
            #pragma omp for schedule(static)
            for (long ii = static_cast<long>(k0 + kb); ii < static_cast<long>(n); ++ii) {
                double* rowi = a + static_cast<size_t>(ii) * n;
                for (size_t jj = 0; jj < kb; ++jj) {
                    const size_t j = k0 + jj;
                    const double* rowj = a + j * n;
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t p = 0; p < jj; ++p) {
                        sum += rowi[k0 + p] * rowj[k0 + p];
                    }
                    rowi[j] = (rowi[j] - sum) / rowj[j];
                }
            }

            // Update the trailing lower-triangular submatrix. Each tile is an
            // independent task; flattening the triangle gives uniform chunks.
            const size_t trailing = n - (k0 + kb);
            if (trailing > 0) {
                const size_t rt = (trailing + nb - 1) / nb;
                const size_t totalTiles = rt * (rt + 1) / 2;
                #pragma omp for schedule(dynamic, 1)
                for (long tt = 0; tt < static_cast<long>(totalTiles); ++tt) {
                    const size_t t = static_cast<size_t>(tt);
                    const size_t it = rowTileFromIndex(t);
                    const size_t jt = t - it * (it + 1) / 2;

                    const size_t i0 = k0 + kb + it * nb;
                    const size_t j0 = k0 + kb + jt * nb;
                    const size_t ib = std::min(nb, n - i0);
                    const size_t jb = std::min(nb, n - j0);
                    const bool diagonalTile = (it == jt);

                    for (size_t ii = 0; ii < ib; ++ii) {
                        double* rowi = a + (i0 + ii) * n;
                        const double* pi = rowi + k0;
                        const size_t jlim = diagonalTile ? (ii + 1) : jb;
                        for (size_t jj = 0; jj < jlim; ++jj) {
                            const double* rowj = a + (j0 + jj) * n;
                            const double* pj = rowj + k0;
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (size_t p = 0; p < kb; ++p) {
                                sum += pi[p] * pj[p];
                            }
                            rowi[j0 + jj] -= sum;
                        }
                    }
                }
            }
        }
    }

    if (!ok) return false;

    // Zero out the upper triangular part.
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        double* row = a + static_cast<size_t>(i) * n;
        std::fill(row + static_cast<size_t>(i) + 1, row + n, 0.0);
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B (kept sequential to preserve the sequence)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    const double* const b = B.data();
    double* const a = A.data();
    #pragma omp parallel for collapse(2) schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        for (long j = 0; j < static_cast<long>(n); ++j) {
            const double* bi = b + static_cast<size_t>(i) * n;
            const double* bj = b + static_cast<size_t>(j) * n;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += bi[k] * bj[k];
            }
            a[static_cast<size_t>(i) * n + static_cast<size_t>(j)] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        a[static_cast<size_t>(i) * n + static_cast<size_t>(i)] += static_cast<double>(n);
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    const double* const l = L.data();
    const double* const a = A_orig.data();
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel for collapse(2) schedule(static) \
        reduction(max:maxError, relError)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        for (long j = 0; j < static_cast<long>(n); ++j) {
            const double* li = l + static_cast<size_t>(i) * n;
            const double* lj = l + static_cast<size_t>(j) * n;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += li[k] * lj[k];
            }
            const double orig = a[static_cast<size_t>(i) * n + static_cast<size_t>(j)];
            const double error = fabs(sum - orig);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(orig) + 1e-10);
            relError = std::max(relError, rel);
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

    const char* numThreads = std::getenv("OMP_NUM_THREADS");
    if (numThreads == nullptr || *numThreads == '\0') {
        omp_set_num_threads(defaultThreadCount());
    }

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
