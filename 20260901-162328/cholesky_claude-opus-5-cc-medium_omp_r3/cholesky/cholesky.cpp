#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <memory>

#include <omp.h>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Cholesky decomposition (OpenMP parallel, two-level blocked left-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Every element is still accumulated as a single sum running over k = 0 .. j-1 in
// increasing order: the partial sums over the already factorized columns are kept in a
// buffer and are simply continued by the following blocks. Each element is therefore
// computed with the same operations in the same order as by the sequential reference
// implementation, and the result does not depend on the number of threads.

namespace {

constexpr size_t MR = 6;    // rows of the GEMM micro kernel, granularity of the row ownership
constexpr size_t NR = 8;    // columns of the GEMM micro kernel (two AVX vectors)
constexpr size_t NB = 32;   // inner panel width, multiple of NR
constexpr size_t NBO = 128; // outer panel width, multiple of NB
constexpr size_t KC = 256;  // depth blocking of the panel update
constexpr size_t PB = 64;   // k blocking of the panel packing

// C[0..MRV)[0..NR) += sum_k Arows[r*lda + k] * Bp[k*ldb + c], k in increasing order.
template <int MRV>
inline void gemmMicroKernel(const double* __restrict Arows, const size_t lda, const double* __restrict Bp,
                            const size_t ldb, const size_t kc, double* __restrict C, const size_t ldc) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d c0[MRV], c1[MRV];
    for (int r = 0; r < MRV; ++r) {
        c0[r] = _mm256_loadu_pd(C + r * ldc);
        c1[r] = _mm256_loadu_pd(C + r * ldc + 4);
    }
    for (size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(Bp + k * ldb);
        const __m256d b1 = _mm256_loadu_pd(Bp + k * ldb + 4);
        for (int r = 0; r < MRV; ++r) {
            const __m256d a = _mm256_set1_pd(Arows[r * lda + k]);
            c0[r] = _mm256_fmadd_pd(a, b0, c0[r]);
            c1[r] = _mm256_fmadd_pd(a, b1, c1[r]);
        }
    }
    for (int r = 0; r < MRV; ++r) {
        _mm256_storeu_pd(C + r * ldc, c0[r]);
        _mm256_storeu_pd(C + r * ldc + 4, c1[r]);
    }
#else
    double acc[MRV][NR];
    for (int r = 0; r < MRV; ++r) {
        for (size_t c = 0; c < NR; ++c) {
            acc[r][c] = C[r * ldc + c];
        }
    }
    for (size_t k = 0; k < kc; ++k) {
        for (int r = 0; r < MRV; ++r) {
            const double a = Arows[r * lda + k];
            for (size_t c = 0; c < NR; ++c) {
                acc[r][c] += a * Bp[k * ldb + c];
            }
        }
    }
    for (int r = 0; r < MRV; ++r) {
        for (size_t c = 0; c < NR; ++c) {
            C[r * ldc + c] = acc[r][c];
        }
    }
#endif
}

inline void gemmMicroDispatch(const size_t mr, const double* Arows, const size_t lda, const double* Bp,
                              const size_t ldb, const size_t kc, double* C, const size_t ldc) {
    switch (mr) {
        case 6: gemmMicroKernel<6>(Arows, lda, Bp, ldb, kc, C, ldc); break;
        case 5: gemmMicroKernel<5>(Arows, lda, Bp, ldb, kc, C, ldc); break;
        case 4: gemmMicroKernel<4>(Arows, lda, Bp, ldb, kc, C, ldc); break;
        case 3: gemmMicroKernel<3>(Arows, lda, Bp, ldb, kc, C, ldc); break;
        case 2: gemmMicroKernel<2>(Arows, lda, Bp, ldb, kc, C, ldc); break;
        default: gemmMicroKernel<1>(Arows, lda, Bp, ldb, kc, C, ldc); break;
    }
}

// Row blocks of MR rows, defined on absolute row indices, are distributed cyclically over
// the threads. The mapping is identical in every phase and for every panel, so a thread
// always works on - and first touches - the same rows of the matrix (cache/NUMA locality).
inline size_t firstOwnedBlock(const size_t row0, const size_t tid, const size_t nthreads) {
    const size_t b = row0 / MR;
    return b + (tid + nthreads - b % nthreads) % nthreads;
}

// Pack the rows [P0, P0+pb) of A, columns [kBeg, kEnd), transposed into Bp. The panel is
// stored as a sequence of NR wide micro panels, each of them contiguous over k, which is
// exactly the order in which the micro kernel walks over it. Ktot is the k extent of the
// whole panel (starting at kOrigin), the columns beyond pb are zero padding.
inline void packMicroPanel(const double* __restrict A, const size_t n, const size_t P0, const size_t pb,
                           const size_t kOrigin, const size_t Ktot, const size_t kBeg, const size_t kEnd,
                           double* __restrict Bp, const size_t jb) {
    const size_t j0 = jb * NR;
    double* dst = &Bp[(jb * Ktot + (kBeg - kOrigin)) * NR];
    if (j0 + NR <= pb) {
        const double* src[NR];
        for (size_t c = 0; c < NR; ++c) {
            src[c] = &A[(P0 + j0 + c) * n];
        }
        for (size_t k = kBeg; k < kEnd; ++k, dst += NR) {
            for (size_t c = 0; c < NR; ++c) {
                dst[c] = src[c][k];
            }
        }
    } else {
        for (size_t k = kBeg; k < kEnd; ++k, dst += NR) {
            for (size_t c = 0; c < NR; ++c) {
                dst[c] = j0 + c < pb ? A[(P0 + j0 + c) * n + k] : 0.0;
            }
        }
    }
}

// S[i][colOff + c] (+)= sum_{k in [kBeg,kEnd)} A[i][k] * Bpanel[k][c], for all rows
// i >= row0 owned by this thread. The k blocking is the outermost loop and the row blocks
// are innermost, so the current micro panel of Bp stays in L1 and the rows of A that a
// thread owns stay in L2 while the panel is swept.
void panelUpdate(const double* __restrict A, const size_t n, const double* __restrict Bp,
                 double* __restrict S, const size_t ldS, const size_t colOff, const size_t width,
                 const size_t row0, const size_t kBeg, const size_t kEnd, const size_t tid,
                 const size_t nthreads, const bool zeroFirst) {
    if (zeroFirst) {
        for (size_t b = firstOwnedBlock(row0, tid, nthreads); b * MR < n; b += nthreads) {
            const size_t i0 = std::max(row0, b * MR);
            const size_t iEnd = std::min(n, (b + 1) * MR);
            for (size_t i = i0; i < iEnd; ++i) {
                double* const C = &S[i * ldS + colOff];
                for (size_t c = 0; c < width; ++c) {
                    C[c] = 0.0;
                }
            }
        }
    }
    const size_t Ktot = kEnd - kBeg;
    for (size_t k0 = kBeg; k0 < kEnd; k0 += KC) {
        const size_t kc = std::min(KC, kEnd - k0);
        for (size_t j0 = 0; j0 < width; j0 += NR) {
            const double* const Bk = &Bp[((j0 / NR) * Ktot + (k0 - kBeg)) * NR];
            for (size_t b = firstOwnedBlock(row0, tid, nthreads); b * MR < n; b += nthreads) {
                const size_t i0 = std::max(row0, b * MR);
                const size_t iEnd = std::min(n, (b + 1) * MR);
                if (i0 >= iEnd) {
                    continue;
                }
                gemmMicroDispatch(iEnd - i0, &A[i0 * n + k0], n, Bk, NR, kc, &S[i0 * ldS + colOff + j0], ldS);
            }
        }
    }
}

// Finish UR rows of the current panel: continue their partial sums over the columns of the
// panel itself and scale by the diagonal. The rows are independent of each other, the
// columns have to be processed in order.
template <int UR>
inline void panelTrailingRows(double* __restrict A, const size_t n, const double* __restrict S,
                              const size_t ldS, const size_t colOff, const size_t Jc, const size_t Jb,
                              const size_t i0) {
    for (size_t jj = 0; jj < Jb; ++jj) {
        const size_t j = Jc + jj;
        double s[UR];
        for (int r = 0; r < UR; ++r) {
            s[r] = S[(i0 + r) * ldS + colOff + jj];
        }
        for (size_t k = Jc; k < j; ++k) {
            const double bk = A[j * n + k];
            for (int r = 0; r < UR; ++r) {
                s[r] += A[(i0 + r) * n + k] * bk;
            }
        }
        const double d = A[j * n + j];
        for (int r = 0; r < UR; ++r) {
            A[(i0 + r) * n + j] = (A[(i0 + r) * n + j] - s[r]) / d;
        }
    }
}

inline void panelTrailingDispatch(const size_t m, double* A, const size_t n, const double* S, const size_t ldS,
                                  const size_t colOff, const size_t Jc, const size_t Jb, const size_t i0) {
    switch (m) {
        case 6: panelTrailingRows<6>(A, n, S, ldS, colOff, Jc, Jb, i0); break;
        case 5: panelTrailingRows<5>(A, n, S, ldS, colOff, Jc, Jb, i0); break;
        case 4: panelTrailingRows<4>(A, n, S, ldS, colOff, Jc, Jb, i0); break;
        case 3: panelTrailingRows<3>(A, n, S, ldS, colOff, Jc, Jb, i0); break;
        case 2: panelTrailingRows<2>(A, n, S, ldS, colOff, Jc, Jb, i0); break;
        default: panelTrailingRows<1>(A, n, S, ldS, colOff, Jc, Jb, i0); break;
    }
}

// The cpus this process may run on, in ascending order. Queried once while the threads
// still have their original affinity mask.
const std::vector<int>& allowedCpus() {
    static const std::vector<int> cpus = [] {
        std::vector<int> list;
#ifdef __linux__
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &mask)) {
                    list.push_back(c);
                }
            }
        }
#endif
        return list;
    }();
    return cpus;
}

// Keep every thread on its own cpu: the row distribution below relies on the threads
// staying where they first touched their part of the matrix. Skipped if the user asked
// for a binding of their own.
void bindThread([[maybe_unused]] const size_t tid) {
#ifdef __linux__
    static const bool userBinding = getenv("OMP_PROC_BIND") != nullptr || getenv("GOMP_CPU_AFFINITY") != nullptr;
    const std::vector<int>& cpus = allowedCpus();
    if (userBinding || cpus.empty()) {
        return;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpus[tid % cpus.size()], &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
#endif
}

// Number of hardware threads per physical core (1 if it cannot be determined).
size_t smtDegree() {
#ifdef __linux__
    FILE* const f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (f == nullptr) {
        return 1;
    }
    char line[256] = {};
    const char* const ok = fgets(line, sizeof(line), f);
    fclose(f);
    if (ok == nullptr) {
        return 1;
    }
    // the list is a comma separated set of cpu numbers and "lo-hi" ranges
    size_t count = 0;
    for (const char* p = line; *p != '\0';) {
        char* next = nullptr;
        const long lo = strtol(p, &next, 10);
        if (next == p) {
            break;
        }
        long hi = lo;
        if (*next == '-') {
            p = next + 1;
            hi = strtol(p, &next, 10);
        }
        count += static_cast<size_t>(hi - lo + 1);
        p = *next == ',' ? next + 1 : next + strlen(next);
    }
    return std::max<size_t>(1, count);
#else
    return 1;
#endif
}

// Threads used for the factorization. The kernels are FMA throughput bound and do not
// profit from SMT, while every additional thread makes the synchronization between the
// panels more expensive - so by default only one thread per physical core is used. An
// explicit OMP_NUM_THREADS is honoured, but for small matrices the thread count is still
// limited to keep a couple of row blocks per thread.
int threadCount(const size_t n) {
    // computed once: all parallel regions have to use the same number of threads, so that
    // they are served by the same (bound) pool of threads
    static const int threads = [n] {
        size_t t = static_cast<size_t>(omp_get_max_threads());
        if (getenv("OMP_NUM_THREADS") == nullptr) {
            t = std::max<size_t>(1, t / smtDegree());
        }
        const size_t byWork = std::max<size_t>(1, n / (2 * MR));
        return static_cast<int>(std::max<size_t>(1, std::min(t, byWork)));
    }();
    return threads;
}

} // namespace

// Touch the rows of a matrix from the threads that will own them during the
// factorization, so that the pages end up on the right NUMA node.
void firstTouchMatrix(std::vector<double>& A, const size_t n) {
    if (A.size() < n * n) {
        return;
    }
    double* const Ap = A.data();
#ifdef __linux__
    // The vector was zero filled by a single thread, which placed all of its pages on one
    // node. Dropping them makes the following writes the first touch again (the memory is
    // private and anonymous, so it reads back as zero either way).
    const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t base = reinterpret_cast<uintptr_t>(Ap);
    const uintptr_t beg = (base + pageSize - 1) & ~static_cast<uintptr_t>(pageSize - 1);
    const uintptr_t end = (base + n * n * sizeof(double)) & ~static_cast<uintptr_t>(pageSize - 1);
    if (end > beg) {
        madvise(reinterpret_cast<void*>(beg), end - beg, MADV_DONTNEED);
    }
#endif
    #pragma omp parallel num_threads(threadCount(n))
    {
        const size_t nthreads = static_cast<size_t>(omp_get_num_threads());
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        bindThread(tid);
        for (size_t b = tid; b * MR < n; b += nthreads) {
            const size_t iEnd = std::min(n, (b + 1) * MR);
            for (size_t i = b * MR; i < iEnd; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    Ap[i * n + j] = 0.0;
                }
            }
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    double* const Ap = A.data();
    // Packed transposed copies of the outer and inner panel rows, and the partial dot
    // products of all remaining rows against the panel rows (indexed by absolute row, so
    // that the ownership of the rows never changes). Allocated without initialization,
    // the first touch happens from the owning thread.
    std::unique_ptr<double[]> Bp(new double[n * NBO]);
    std::unique_ptr<double[]> Bi(new double[NBO * NB]);
    std::unique_ptr<double[]> S(new double[n * NBO]);
    double* const Bpp = Bp.get();
    double* const Bip = Bi.get();
    double* const Sp = S.get();
    bool failed = false;

    #pragma omp parallel num_threads(threadCount(n))
    {
        const size_t nthreads = static_cast<size_t>(omp_get_num_threads());
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        bindThread(tid);

        for (size_t JO = 0; JO < n; JO += NBO) {
            const size_t JbO = std::min(NBO, n - JO);
            const size_t K = JO; // columns factorized before this outer panel

            // Partial sums of all remaining rows against the outer panel rows.
            if (K > 0) {
                const size_t kblocks = (K + PB - 1) / PB;
                #pragma omp for collapse(2) schedule(static)
                for (size_t kb = 0; kb < kblocks; ++kb) {
                    for (size_t jb = 0; jb < NBO / NR; ++jb) {
                        const size_t kBeg = kb * PB;
                        packMicroPanel(Ap, n, JO, JbO, 0, K, kBeg, std::min(kBeg + PB, K), Bpp, jb);
                    }
                }
            }
            panelUpdate(Ap, n, Bpp, Sp, NBO, 0, NBO, JO, 0, K, tid, nthreads, true);
            #pragma omp barrier

            for (size_t Jc = JO; Jc < JO + JbO; Jc += NB) {
                const size_t Jb = std::min(NB, JO + JbO - Jc);
                const size_t colOff = Jc - JO;

                // Bring the partial sums up to date with the columns of the outer panel
                // that were factorized since the outer update.
                if (Jc > JO) {
                    #pragma omp for schedule(static)
                    for (size_t jb = 0; jb < NB / NR; ++jb) {
                        packMicroPanel(Ap, n, Jc, Jb, JO, Jc - JO, JO, Jc, Bip, jb);
                    }
                    panelUpdate(Ap, n, Bip, Sp, NBO, colOff, NB, Jc, JO, Jc, tid, nthreads, false);
                    #pragma omp barrier
                }

                // Factorize the diagonal block of the panel (small, strictly sequential).
                #pragma omp single
                {
                    for (size_t jj = 0; jj < Jb; ++jj) {
                        const size_t j = Jc + jj;
                        double sum = Sp[j * NBO + colOff + jj];
                        for (size_t k = Jc; k < j; ++k) {
                            sum += Ap[j * n + k] * Ap[j * n + k];
                        }
                        const double val = Ap[j * n + j] - sum;
                        if (val <= 0.0) {
                            // Matrix is not positive definite
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                            failed = true;
                            break;
                        }
                        Ap[j * n + j] = sqrt(val);
                        const double d = Ap[j * n + j];
                        for (size_t i = j + 1; i < Jc + Jb; ++i) {
                            double s = Sp[i * NBO + colOff + jj];
                            for (size_t k = Jc; k < j; ++k) {
                                s += Ap[i * n + k] * Ap[j * n + k];
                            }
                            Ap[i * n + j] = (Ap[i * n + j] - s) / d;
                        }
                    }
                }
                // the implicit barrier of the single construct publishes the block and 'failed'
                if (failed) {
                    break;
                }

                // Rows below the panel: independent of each other.
                const size_t tBeg = Jc + Jb;
                for (size_t b = firstOwnedBlock(tBeg, tid, nthreads); b * MR < n; b += nthreads) {
                    const size_t i0 = std::max(tBeg, b * MR);
                    const size_t iEnd = std::min(n, (b + 1) * MR);
                    if (i0 >= iEnd) {
                        continue;
                    }
                    panelTrailingDispatch(iEnd - i0, Ap, n, Sp, NBO, colOff, Jc, Jb, i0);
                }
                #pragma omp barrier
            }

            if (failed) {
                break;
            }
        }

        if (!failed) {
            // Zero out upper triangular part
            for (size_t b = tid; b * MR < n; b += nthreads) {
                const size_t iEnd = std::min(n, (b + 1) * MR);
                for (size_t i = b * MR; i < iEnd; ++i) {
                    for (size_t j = i + 1; j < n; ++j) {
                        Ap[i * n + j] = 0.0;
                    }
                }
            }
        }
    }

    return !failed;
}

namespace {

// C = X * X^T for a row-major n x n matrix X. Each element keeps the sequential
// k = 0..n-1 accumulation order of the reference code.
void gramProduct(const double* __restrict X, double* __restrict C, const size_t n) {
    constexpr size_t TB = 4;
    const size_t nb = (n + TB - 1) / TB;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(threadCount(n))
    for (size_t bi = 0; bi < nb; ++bi) {
        for (size_t bj = 0; bj < nb; ++bj) {
            const size_t i0 = bi * TB;
            const size_t j0 = bj * TB;
            const size_t mi = std::min(TB, n - i0);
            const size_t mj = std::min(TB, n - j0);
            if (mi == TB && mj == TB) {
                double acc[TB][TB] = {};
                for (size_t k = 0; k < n; ++k) {
                    double a[TB], b[TB];
                    for (size_t r = 0; r < TB; ++r) {
                        a[r] = X[(i0 + r) * n + k];
                        b[r] = X[(j0 + r) * n + k];
                    }
                    for (size_t r = 0; r < TB; ++r) {
                        for (size_t c = 0; c < TB; ++c) {
                            acc[r][c] += a[r] * b[c];
                        }
                    }
                }
                for (size_t r = 0; r < TB; ++r) {
                    for (size_t c = 0; c < TB; ++c) {
                        C[(i0 + r) * n + j0 + c] = acc[r][c];
                    }
                }
            } else {
                for (size_t i = i0; i < i0 + mi; ++i) {
                    for (size_t j = j0; j < j0 + mj; ++j) {
                        double sum = 0.0;
                        for (size_t k = 0; k < n; ++k) {
                            sum += X[i * n + k] * X[j * n + k];
                        }
                        C[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

} // namespace

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
    gramProduct(B.data(), A.data(), n);

    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for schedule(static) num_threads(threadCount(n))
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    gramProduct(L.data(), reconstructed.data(), n);

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for schedule(static) reduction(max : maxError, relError) num_threads(threadCount(n))
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Spread the matrix over the NUMA nodes according to the row distribution used
    // by the factorization
    firstTouchMatrix(A, n);

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
