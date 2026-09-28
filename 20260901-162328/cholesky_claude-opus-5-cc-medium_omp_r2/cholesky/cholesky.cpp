#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#ifdef __linux__
#include <sched.h>
#endif

#include "../common/results_output.hpp"

// OpenMP-parallel Cholesky decomposition (blocked, left-looking algorithm).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The blocking is chosen such that every output element is still accumulated by a
// single sequential dot product running over k in ascending order, exactly as in the
// original scalar code. Only the order in which *independent* elements are visited
// changes, so results are bit-identical to the sequential version.

namespace {

#ifdef __linux__
// Hardware threads to place the OpenMP threads on, one per physical core first.
std::vector<int> g_cpuOrder;
thread_local int g_pinnedCpu = -1;

// Pin the calling OpenMP thread. The kernels below are blocked for the private
// caches, so letting the scheduler migrate threads (and with it their working set)
// costs a large fraction of the achievable performance. This is called at the start
// of every parallel region because the OpenMP runtime may spawn additional worker
// threads later on, which would otherwise inherit the master's affinity mask.
inline void pinSelf() {
    const size_t tid = (size_t)omp_get_thread_num();
    if (tid >= g_cpuOrder.size() || g_pinnedCpu == g_cpuOrder[tid]) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(g_cpuOrder[tid], &set);
    if (sched_setaffinity(0, sizeof(set), &set) == 0) {
        g_pinnedCpu = g_cpuOrder[tid];
    }
}

void bindThreads(const size_t n) {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr) {
        return; // an explicit placement requested by the user always wins
    }

    cpu_set_t available;
    CPU_ZERO(&available);
    if (sched_getaffinity(0, sizeof(available), &available) != 0) {
        return;
    }

    // Order the usable CPUs so that one hardware thread per physical core comes
    // first: the kernels are limited by the per-core caches and FPU, so a second
    // thread on the same core mostly takes cache away from the first one.
    std::vector<int> cores, siblings;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &available)) {
            continue;
        }
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        int leader = c;
        if (FILE* f = fopen(path, "r")) {
            int first = -1;
            if (fscanf(f, "%d", &first) == 1 && first >= 0) {
                leader = first;
            }
            fclose(f);
        }
        (leader == c ? cores : siblings).push_back(c);
    }
    if (cores.empty()) {
        return;
    }

    // Without an explicit request, use one thread per physical core, but not more
    // than the problem can keep busy: for small matrices the barrier and wake-up
    // cost of a full-machine team outweighs the extra compute. The team size is
    // fixed here, once, because changing it later makes the OpenMP runtime tear
    // down and re-create its worker threads.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const size_t useful = std::max<size_t>(1, n / 32);
        const size_t threads = std::min(cores.size(), useful);
        if (threads < (size_t)omp_get_max_threads()) {
            omp_set_num_threads((int)threads);
        }
    }

    g_cpuOrder = cores;
    g_cpuOrder.insert(g_cpuOrder.end(), siblings.begin(), siblings.end());

    #pragma omp parallel
    {
        pinSelf();
    }
}
#else
inline void pinSelf() {}
void bindThreads(size_t) {}
#endif

// Blocking parameters for the C = X * Y^T kernel below.
constexpr size_t GEMM_KC = 128; // k block (packed panel stays in L2)
constexpr size_t GEMM_NC = 128; // column block
constexpr size_t GEMM_MR = 4;   // micro-kernel rows
constexpr size_t GEMM_NR = 8;   // micro-kernel columns

constexpr size_t CHOL_NB = 64; // Cholesky panel width

#if defined(__AVX2__)
// Separate multiply and add. The reference implementation is compiled into a packed
// multiply followed by sequential scalar adds, i.e. every product is rounded on its
// own; contracting into an FMA here would change the last bits of the result.
inline __m256d mul_add(const __m256d a, const __m256d b, __m256d c) {
    __m256d p = _mm256_mul_pd(a, b);
    asm("" : "+x"(p)); // keep the product rounded separately (no FMA contraction)
    return _mm256_add_pd(c, p);
}
#endif

// MRT x GEMM_NR micro-kernel:  Ctile[r][j] += sum_kk X[r][kk] * P[kk][j].
// Every accumulator is a private, strictly sequential chain over kk, which is exactly
// the operation sequence the scalar reference code performs.
template <size_t MRT>
inline void micro_kernel(const double* __restrict X, const size_t ldx, const double* __restrict P,
                         double* __restrict C, const size_t ldc, const size_t kc) {
#if defined(__AVX2__)
    __m256d c0[MRT], c1[MRT];
    for (size_t r = 0; r < MRT; ++r) {
        c0[r] = _mm256_loadu_pd(C + r * ldc + 0);
        c1[r] = _mm256_loadu_pd(C + r * ldc + 4);
    }

    for (size_t kk = 0; kk < kc; ++kk) {
        __m256d b0 = _mm256_loadu_pd(P + kk * GEMM_NR + 0);
        __m256d b1 = _mm256_loadu_pd(P + kk * GEMM_NR + 4);
        // Keep both column vectors in registers instead of re-loading them for
        // every row of the tile.
        asm("" : "+x"(b0), "+x"(b1));

        for (size_t r = 0; r < MRT; ++r) {
            const __m256d a = _mm256_broadcast_sd(X + r * ldx + kk);
            c0[r] = mul_add(a, b0, c0[r]);
            c1[r] = mul_add(a, b1, c1[r]);
        }
    }

    for (size_t r = 0; r < MRT; ++r) {
        _mm256_storeu_pd(C + r * ldc + 0, c0[r]);
        _mm256_storeu_pd(C + r * ldc + 4, c1[r]);
    }
#else
    double acc[MRT][GEMM_NR];
    for (size_t r = 0; r < MRT; ++r) {
        for (size_t jj = 0; jj < GEMM_NR; ++jj) {
            acc[r][jj] = C[r * ldc + jj];
        }
    }
    for (size_t kk = 0; kk < kc; ++kk) {
        for (size_t r = 0; r < MRT; ++r) {
            const double a = X[r * ldx + kk];
            for (size_t jj = 0; jj < GEMM_NR; ++jj) {
                acc[r][jj] += a * P[kk * GEMM_NR + jj];
            }
        }
    }
    for (size_t r = 0; r < MRT; ++r) {
        for (size_t jj = 0; jj < GEMM_NR; ++jj) {
            C[r * ldc + jj] = acc[r][jj];
        }
    }
#endif
}

// Dispatch to the micro-kernel variant matching the number of remaining rows.
inline void micro_kernel_dispatch(const size_t mr, const double* __restrict X, const size_t ldx,
                                  const double* __restrict P, double* __restrict C, const size_t ldc,
                                  const size_t kc) {
    switch (mr) {
    case 1: micro_kernel<1>(X, ldx, P, C, ldc, kc); break;
    case 2: micro_kernel<2>(X, ldx, P, C, ldc, kc); break;
    case 3: micro_kernel<3>(X, ldx, P, C, ldc, kc); break;
    case 4: micro_kernel<4>(X, ldx, P, C, ldc, kc); break;
    case 5: micro_kernel<5>(X, ldx, P, C, ldc, kc); break;
    default: micro_kernel<6>(X, ldx, P, C, ldc, kc); break;
    }
}

// C[i][j] = sum_{k=0}^{K-1} X[i][k] * Y[j][k], accumulated in ascending k order.
// C is overwritten (its previous contents are ignored). The rows of C are distributed
// over the threads, so the routine needs no synchronization at all.
void gemm_nt(const double* __restrict X, const size_t ldx, const double* __restrict Y, const size_t ldy,
             double* __restrict C, const size_t ldc, const size_t M, const size_t N, const size_t K) {
    // The reference loop is auto-vectorized four elements at a time, and the
    // remaining K % 4 products are contracted into fused multiply-adds. The blocked
    // loop below reproduces the first part, the epilogue below the second one.
    const size_t Kmain = K - K % 4;

    #pragma omp parallel
    {
        pinSelf();
        const size_t nthreads = (size_t)omp_get_num_threads();
        const size_t tid = (size_t)omp_get_thread_num();

        // Static, evenly balanced row distribution.
        const size_t base = M / nthreads;
        const size_t rem = M % nthreads;
        const size_t i0 = tid * base + std::min(tid, rem);
        const size_t i1 = i0 + base + (tid < rem ? 1 : 0);

        for (size_t i = i0; i < i1; ++i) {
            std::fill_n(C + i * ldc, N, 0.0);
        }

        if (Kmain > 0 && i1 > i0) {
            // Private copy of the packed panel: it is re-read once per row block, so
            // it has to live in this core's own L2.
            std::vector<double> pack(GEMM_NC * GEMM_KC);

            for (size_t jc = 0; jc < N; jc += GEMM_NC) {
                const size_t nc = std::min(GEMM_NC, N - jc);

                for (size_t kb = 0; kb < Kmain; kb += GEMM_KC) {
                    const size_t kc = std::min(GEMM_KC, Kmain - kb);

                    // Pack Y into k-major strips of GEMM_NR columns so that the
                    // micro-kernel can vectorize over columns while k stays sequential.
                    for (size_t js = 0; js < nc; js += GEMM_NR) {
                        const size_t nr = std::min(GEMM_NR, nc - js);
                        double* __restrict p = pack.data() + js * kc;
                        for (size_t kk = 0; kk < kc; ++kk) {
                            for (size_t jj = 0; jj < nr; ++jj) {
                                p[kk * nr + jj] = Y[(jc + js + jj) * ldy + kb + kk];
                            }
                        }
                    }

                    for (size_t i = i0; i < i1; i += GEMM_MR) {
                        const size_t mr = std::min(GEMM_MR, i1 - i);

                        for (size_t js = 0; js < nc; js += GEMM_NR) {
                            const size_t nr = std::min(GEMM_NR, nc - js);
                            const double* __restrict p = pack.data() + js * kc;

                            if (nr == GEMM_NR) {
                                micro_kernel_dispatch(mr, X + i * ldx + kb, ldx, p, C + i * ldc + jc + js, ldc,
                                                      kc);
                            } else {
                                // Edge tiles.
                                for (size_t r = 0; r < mr; ++r) {
                                    const double* __restrict x = X + (i + r) * ldx + kb;
                                    for (size_t jj = 0; jj < nr; ++jj) {
                                        double sum = C[(i + r) * ldc + jc + js + jj];
                                        for (size_t kk = 0; kk < kc; ++kk) {
                                            sum += x[kk] * p[kk * nr + jj];
                                        }
                                        C[(i + r) * ldc + jc + js + jj] = sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (Kmain != K) {
            for (size_t i = i0; i < i1; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    double sum = C[i * ldc + j];
                    for (size_t k = Kmain; k < K; ++k) {
                        sum = std::fma(X[i * ldx + k], Y[j * ldy + k], sum);
                    }
                    C[i * ldc + j] = sum;
                }
            }
        }
    }
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    double* __restrict a = A.data();

    // Partial sums sum_{k < j0} A[i][k] * A[j][k] for the current panel.
    std::vector<double> S(n * CHOL_NB);

    for (size_t j0 = 0; j0 < n; j0 += CHOL_NB) {
        const size_t nb = std::min(CHOL_NB, n - j0);
        const size_t m = n - j0;

        // Left-looking update of the panel: everything that the columns of this
        // panel owe to the already factored columns [0, j0).
        gemm_nt(a + j0 * n, n, a + j0 * n, n, S.data(), nb, m, nb, j0);

        // Factor the nb x nb diagonal block (tiny, kept sequential).
        for (size_t jj = 0; jj < nb; ++jj) {
            const size_t j = j0 + jj;

            double sum = S[jj * nb + jj];
            for (size_t k = j0; k < j; ++k) {
                sum += a[j * n + k] * a[j * n + k];
            }
            const double val = a[j * n + j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            const double diag = sqrt(val);
            a[j * n + j] = diag;

            for (size_t ii = jj + 1; ii < nb; ++ii) {
                const size_t i = j0 + ii;
                double s = S[ii * nb + jj];
                for (size_t k = j0; k < j; ++k) {
                    s += a[i * n + k] * a[j * n + k];
                }
                a[i * n + j] = (a[i * n + j] - s) / diag;
            }
        }

        // Remaining rows of the panel: fully independent across rows.
        #pragma omp parallel
        {
        pinSelf();
        #pragma omp for schedule(static)
        for (size_t i = j0 + nb; i < n; ++i) {
            double* __restrict ai = a + i * n;
            for (size_t jj = 0; jj < nb; ++jj) {
                const size_t j = j0 + jj;
                const double* __restrict aj = a + j * n;
                double sum = S[(i - j0) * nb + jj];
                for (size_t k = j0; k < j; ++k) {
                    sum += ai[k] * aj[k];
                }
                ai[j] = (ai[j] - sum) / aj[j];
            }
        }
        }
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        std::fill_n(a + i * n + i + 1, n - i - 1, 0.0);
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
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    gemm_nt(B.data(), n, B.data(), n, A.data(), n, n, n, n);

    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    gemm_nt(L.data(), n, L.data(), n, reconstructed.data(), n, n, n, n);

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
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
    
    bindThreads(n);

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
