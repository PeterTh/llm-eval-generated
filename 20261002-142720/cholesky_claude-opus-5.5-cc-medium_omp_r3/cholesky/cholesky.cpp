#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#include "../common/results_output.hpp"

// Parallel blocked Cholesky decomposition (right-looking with lookahead, OpenMP)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

namespace {

typedef double v4d __attribute__((vector_size(32)));

constexpr size_t MR = 6;          // micro-kernel rows
constexpr size_t NR = 8;          // micro-kernel cols (2 x 4 doubles)
constexpr size_t POTRF_BASE = 16; // unblocked base case of recursive potrf

// Pack rows [s, s+R) of an m x kc block (row-major, stride lda) into P[k][R]
template <size_t R>
inline void packSliver(const double* A, const size_t lda, const size_t m, const size_t s,
                       const size_t kc, double* P) {
    const size_t rows = std::min(R, m - s);
    for (size_t k = 0; k < kc; ++k) {
        for (size_t r = 0; r < rows; ++r) P[k * R + r] = A[(s + r) * lda + k];
        for (size_t r = rows; r < R; ++r) P[k * R + r] = 0.0;
    }
}

// C (mr x nr) -= A (mr rows of length kc, stride lda) * Bp (packed NR x kc)^T
template <size_t M>
inline void microKernel(const double* A, const size_t lda, const double* __restrict Bp, const size_t kc,
                        double* C, const size_t ldc, const size_t nr) {
    v4d c[M][2];
    const double* a[M];
    for (size_t r = 0; r < M; ++r) {
        c[r][0] = c[r][1] = (v4d){0, 0, 0, 0};
        a[r] = A + r * lda;
    }
    for (size_t k = 0; k < kc; ++k) {
        v4d b0, b1;
        memcpy(&b0, Bp + k * NR, sizeof(v4d));
        memcpy(&b1, Bp + k * NR + 4, sizeof(v4d));
#pragma GCC unroll 6
        for (size_t r = 0; r < M; ++r) {
            const double av = a[r][k];
            const v4d avv = (v4d){av, av, av, av};
            c[r][0] += avv * b0;
            c[r][1] += avv * b1;
        }
    }
    if (nr == NR) {
        for (size_t r = 0; r < M; ++r) {
            double* Cr = C + r * ldc;
            for (size_t j = 0; j < 4; ++j) Cr[j] -= c[r][0][j];
            for (size_t j = 0; j < 4; ++j) Cr[4 + j] -= c[r][1][j];
        }
    } else {
        for (size_t r = 0; r < M; ++r) {
            double* Cr = C + r * ldc;
            for (size_t j = 0; j < nr; ++j) Cr[j] -= (j < 4) ? c[r][0][j] : c[r][1][j - 4];
        }
    }
}

// Remaining rows (< MR) of a row block
inline void kernelTail(const double* A, const size_t lda, const double* Bp, const size_t kc,
                       double* C, const size_t ldc, size_t rows, const size_t nr) {
    if (rows >= 4) {
        microKernel<4>(A, lda, Bp, kc, C, ldc, nr);
        A += 4 * lda; C += 4 * ldc; rows -= 4;
    }
    if (rows >= 2) {
        microKernel<2>(A, lda, Bp, kc, C, ldc, nr);
        A += 2 * lda; C += 2 * ldc; rows -= 2;
    }
    if (rows >= 1) microKernel<1>(A, lda, Bp, kc, C, ldc, nr);
}

// Lower triangle of C (rows [i0,ie), cols [j0,je) of the trailing matrix) -= P * P^T,
// where P is the panel (stride lda) and bp its NR-packed copy; micro-blocks strictly
// above the diagonal are skipped.
inline void updateTile(const double* P, const size_t lda, const double* bp, const size_t kc,
                       double* C, const size_t ldc, const size_t i0, const size_t ie,
                       const size_t j0, const size_t je) {
    for (size_t j = j0; j < je; j += NR) {
        const size_t nr = std::min(NR, je - j);
        size_t i = std::max(i0, (j / MR) * MR);
        for (; i + MR <= ie; i += MR) {
            microKernel<MR>(P + i * lda, lda, bp + j * kc, kc, C + i * ldc + j, ldc, nr);
        }
        kernelTail(P + i * lda, lda, bp + j * kc, kc, C + i * ldc + j, ldc, ie - i, nr);
    }
}

// Panel rows X (mr <= NR rows, stride lda) := X * L^{-T}, where L is the factored
// bk x bk diagonal block and lp its NR-packed copy (rows of L in slivers of NR).
inline void trsmSliver(const double* L, const double* lp, double* X, const size_t lda,
                       const size_t mr, const size_t bk) {
    for (size_t j = 0; j < bk; j += NR) {
        const size_t nr = std::min(NR, bk - j);
        // X[:, j:j+nr] -= X[:, 0:j] * L[j:j+nr, 0:j]^T
        if (j > 0) {
            size_t i = 0;
            if (mr >= MR) {
                microKernel<MR>(X, lda, lp + j * bk, j, X + j, lda, nr);
                i = MR;
            }
            kernelTail(X + i * lda, lda, lp + j * bk, j, X + i * lda + j, lda, mr - i, nr);
        }
        // Small triangular solve on the nr x nr diagonal part (rows are independent)
        if (nr == NR) {
            const double* Ld = L + j * lda + j;
            double inv[NR];
            for (size_t c = 0; c < NR; ++c) inv[c] = 1.0 / Ld[c * lda + c];
            for (size_t r = 0; r < mr; ++r) {
                double* Xr = X + r * lda + j;
                double x[NR];
#pragma GCC unroll 8
                for (size_t c = 0; c < NR; ++c) x[c] = Xr[c];
#pragma GCC unroll 8
                for (size_t c = 0; c < NR; ++c) {
                    const double* Lc = Ld + c * lda;
                    double sum = 0.0;
#pragma GCC unroll 8
                    for (size_t k = 0; k < c; ++k) sum += x[k] * Lc[k];
                    x[c] = (x[c] - sum) * inv[c];
                }
#pragma GCC unroll 8
                for (size_t c = 0; c < NR; ++c) Xr[c] = x[c];
            }
            continue;
        }
        for (size_t jj = j; jj < j + nr; ++jj) {
            const double* Lj = L + jj * lda;
            const double d = Lj[jj];
            for (size_t r = 0; r < mr; ++r) {
                double* Xr = X + r * lda;
                double sum = 0.0;
                for (size_t k = j; k < jj; ++k) sum += Xr[k] * Lj[k];
                Xr[jj] = (Xr[jj] - sum) / d;
            }
        }
    }
}

// Sequential recursive Cholesky of an nb x nb diagonal block (lower part).
// Returns local index of the failing diagonal element, or -1.
long potrfBlock(double* A, const size_t lda, const size_t nb) {
    if (nb <= POTRF_BASE) {
        for (size_t i = 0; i < nb; ++i) {
            double* Ai = A + i * lda;
            for (size_t j = 0; j < i; ++j) {
                const double* Aj = A + j * lda;
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) sum += Ai[k] * Aj[k];
                Ai[j] = (Ai[j] - sum) / Aj[j];
            }
            double sum = 0.0;
            for (size_t k = 0; k < i; ++k) sum += Ai[k] * Ai[k];
            const double val = Ai[i] - sum;
            if (val <= 0.0) return (long)i;
            Ai[i] = sqrt(val);
        }
        return -1;
    }
    const size_t n1 = ((nb / 2 + NR - 1) / NR) * NR;
    const size_t n2 = nb - n1;
    const long r = potrfBlock(A, lda, n1);
    if (r >= 0) return r;
    double* A21 = A + n1 * lda;
    double* A22 = A21 + n1;
    std::vector<double> buf(((std::max(n1, n2) + NR - 1) / NR) * NR * n1);
    double* bp = buf.data();
    for (size_t s = 0; s < n1; s += NR) packSliver<NR>(A, lda, n1, s, n1, bp + s * n1);
    for (size_t s = 0; s < n2; s += NR) trsmSliver(A, bp, A21 + s * lda, lda, std::min(NR, n2 - s), n1);
    for (size_t s = 0; s < n2; s += NR) packSliver<NR>(A21, lda, n2, s, n1, bp + s * n1);
    updateTile(A21, lda, bp, n1, A22, lda, 0, n2, 0, n2);
    const long r2 = potrfBlock(A22, lda, n2);
    return r2 >= 0 ? (long)n1 + r2 : -1;
}

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// Spinning barrier for the threads of a parallel region (the per-step barriers are
// short and frequent; spinning avoids the latency of sleeping/waking threads)
struct SpinBarrier {
    alignas(64) std::atomic<size_t> count{0};
    alignas(64) std::atomic<size_t> generation{0};
    size_t nthreads = 0;

    void wait() {
        const size_t gen = generation.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == nthreads - 1) {
            count.store(0, std::memory_order_relaxed);
            generation.store(gen + 1, std::memory_order_release);
        } else {
            size_t spins = 0;
            while (generation.load(std::memory_order_acquire) == gen) {
                cpuRelax();
                if (++spins > (1u << 20)) sched_yield();
            }
        }
    }
};

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;

    // Panel width NB and trailing-update tile size TB (multiples of NR, TB divides NB)
    const size_t NB = n <= 1024 ? 48 : n <= 6144 ? 96 : n <= 12288 ? 192 : 288;
    const size_t TB = n <= 12288 ? 48 : 96;

    double* a = A.data();
    const size_t nsteps = (n + NB - 1) / NB;

    // Double-buffered packed copy of the current panel (rows below the diagonal block)
    std::unique_ptr<double[]> Bpack(new double[2 * (n + NR) * NB]);
    double* bpBuf[2] = {Bpack.get(), Bpack.get() + (n + NR) * NB};
    // Packed copy of the factored diagonal block (used by the panel solve)
    std::unique_ptr<double[]> Lpack(new double[(NB + NR) * NB]);
    double* lp = Lpack.get();
    auto packDiag = [=](const double* L, const size_t bk) {
        for (size_t s = 0; s < bk; s += NR) packSliver<NR>(L, n, bk, s, bk, lp + s * bk);
    };

    // Synchronization state for the step pipeline
    std::atomic<long> failIndex{-1};
    std::atomic<size_t> failBlock{SIZE_MAX};  // index of the diagonal block that failed
    std::atomic<size_t> diagStep{0};  // diagonal block of this step index is factored
    std::unique_ptr<std::atomic<size_t>[]> rowDone(new std::atomic<size_t>[n / TB + 2]);
    std::unique_ptr<std::atomic<size_t>[]> nextItem(new std::atomic<size_t>[nsteps + 1]);
    for (size_t i = 0; i < n / TB + 2; ++i) rowDone[i].store(0, std::memory_order_relaxed);
    std::unique_ptr<std::atomic<size_t>[]> diagTilesDone(new std::atomic<size_t>[nsteps + 1]);
    for (size_t i = 0; i <= nsteps; ++i) {
        nextItem[i].store(0, std::memory_order_relaxed);
        diagTilesDone[i].store(0, std::memory_order_relaxed);
    }

    SpinBarrier barrier;

    // Panel solve of rows [s, s+NR) below the diagonal block at k0, then pack them
    auto solvePanelSliver = [=](const size_t k0, const size_t bk, const size_t s, double* bp) {
        const size_t r0 = k0 + bk, m = n - r0;
        double* panel = a + r0 * n + k0;
        trsmSliver(a + k0 * n + k0, lp, panel + s * n, n, std::min(NR, m - s), bk);
        packSliver<NR>(panel, n, m, s, bk, bp + s * bk);
    };

#pragma omp parallel
    {
        // 2D thread grid pr x pc for block-cyclic tile ownership
        const size_t nthreads = omp_get_num_threads();
        const size_t tid = omp_get_thread_num();
        size_t pr = (size_t)sqrt((double)nthreads);
        while (nthreads % pr) --pr;
        const size_t pc = nthreads / pr;
        const size_t myRow = tid / pc, myCol = tid % pc;
        auto owner = [=](size_t gI, size_t gJ) { return (gI % pr) * pc + gJ % pc; };
#pragma omp single
        barrier.nthreads = nthreads;

        // Step 0: factor the first diagonal block and solve the first panel
#pragma omp single
        {
            const long r = potrfBlock(a, n, std::min(NB, n));
            if (r >= 0) {
                failIndex.store(r);
                failBlock.store(0);
            } else {
                packDiag(a, std::min(NB, n));
            }
        }
        if (failBlock.load() != 0 && n > NB) {
#pragma omp for schedule(dynamic, 1)
            for (size_t s = 0; s < n - NB; s += NR) solvePanelSliver(0, NB, s, bpBuf[0]);
        }

        for (size_t K = 0; K < nsteps; ++K) {
            // Invariant: diagonal block K factored and panel K solved/packed, unless
            // the factorization failed in a block <= K (block K+1 may fail during this step)
            if (failBlock.load(std::memory_order_relaxed) <= K) break;
            const size_t k0 = K * NB;
            const size_t bk = std::min(NB, n - k0);
            const size_t r0 = k0 + bk;   // first row/col of trailing matrix
            const size_t m = n - r0;     // trailing size
            if (m == 0) break;
            const double* panel = a + r0 * n + k0;
            const double* bp = bpBuf[K & 1];
            double* C = a + r0 * n + r0;

            // Owner-computes trailing update: tile (I, J) is always processed by the
            // same thread (2D block-cyclic over global tile indices) so it stays cached.
            //   - tiles of the next diagonal block are updated first, then it is factored (lookahead)
            //   - tiles of the next panel (J < nd) are processed before the rest
            //   - then panel-solve slivers of step K+1 are taken dynamically
            const size_t nt = (m + TB - 1) / TB;
            const size_t bk2 = std::min(NB, m);
            const size_t nd = (bk2 + TB - 1) / TB;
            const size_t m2 = m - bk2;
            const size_t nsl = (m2 + NR - 1) / NR;  // next-panel slivers
            const size_t g0 = r0 / TB;              // global tile index of trailing origin

            // Next diagonal block tiles first; the thread completing the last one factors it
            {
                bool did = false;
                for (size_t J = 0; J < nd; ++J) {
                    for (size_t I = J; I < nd; ++I) {
                        if (owner(g0 + I, g0 + J) != tid) continue;
                        const size_t i0 = I * TB, j0 = J * TB;
                        updateTile(panel, n, bp, bk, C, n, i0, std::min(bk2, i0 + TB), j0, std::min(bk2, j0 + TB));
                        did = true;
                    }
                }
                if (did) {
                    size_t mine = 0;
                    for (size_t J = 0; J < nd; ++J)
                        for (size_t I = J; I < nd; ++I) mine += owner(g0 + I, g0 + J) == tid;
                    if (diagTilesDone[K].fetch_add(mine, std::memory_order_acq_rel) + mine == nd * (nd + 1) / 2) {
                        const long r = potrfBlock(C, n, bk2);
                        if (r >= 0) {
                            failIndex.store((long)r0 + r);
                            failBlock.store(K + 1);
                        } else {
                            packDiag(C, bk2);
                        }
                        diagStep.store(K + 1, std::memory_order_release);
                    }
                }
            }
            // Own tiles: first local column J with (g0 + J) % pc == myCol, first row likewise
            const size_t J0 = (myCol + pc - g0 % pc) % pc;
            for (size_t J = J0; J < nt; J += pc) {
                const size_t Imin = std::max(J, J < nd ? nd : J);
                const size_t I0 = Imin + (myRow + pr - (g0 + Imin) % pr) % pr;
                for (size_t I = I0; I < nt; I += pr) {
                    const size_t i0 = I * TB, j0 = J * TB;
                    updateTile(panel, n, bp, bk, C, n, i0, std::min(m, i0 + TB), j0, std::min(m, j0 + TB));
                    if (J < nd) rowDone[g0 + I].fetch_add(1, std::memory_order_release);
                }
            }

            size_t t;
            while ((t = nextItem[K].fetch_add(1, std::memory_order_relaxed)) < nsl) {
                // Panel solve of step K+1 for rows [s, s+NR) below the next diagonal block
                const size_t s = t * NR;
                const size_t I = (bk2 + s) / TB;
                while (diagStep.load(std::memory_order_acquire) < K + 1) cpuRelax();
                while (rowDone[g0 + I].load(std::memory_order_acquire) < (K + 1) * nd) cpuRelax();
                if (failIndex.load(std::memory_order_relaxed) < 0) {
                    solvePanelSliver(r0, bk2, s, bpBuf[(K + 1) & 1]);
                }
            }
            barrier.wait();
        }
    }

    if (failIndex.load() >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %ld\n", failIndex.load());
        return false;
    }

    // Zero out upper triangular part
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        std::fill(a + i * n + i + 1, a + (i + 1) * n, 0.0);
    }

    return true;
}

// Configure the OpenMP threads for the benchmark:
//  - Pin threads to CPUs, unless the user configured thread affinity explicitly.
//    Consecutive thread ids are packed onto the hardware threads of a core (cores in
//    ascending order); with fewer threads than cores, threads are spread over cores.
//    Pinned threads keep their matrix tiles cache-resident across factorization steps.
//  - Set an explicit node-local memory policy (same placement as the default policy)
//    so that automatic NUMA balancing does not unmap/migrate pages during the run.
void configureOpenMPThreads() {
    std::vector<std::pair<int, int>> cpus;  // (core key, cpu); key = first hw thread of core
    std::vector<size_t> coreStart;          // index in cpus of the first cpu of each core
    cpu_set_t mask;
    const bool bind = !getenv("OMP_PROC_BIND") && !getenv("OMP_PLACES") && !getenv("GOMP_CPU_AFFINITY") &&
                      sched_getaffinity(0, sizeof(mask), &mask) == 0;
    if (bind) {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &mask)) continue;
            int first = c;
            char path[128];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            if (FILE* f = fopen(path, "r")) {
                if (fscanf(f, "%d", &first) != 1) first = c;
                fclose(f);
            }
            cpus.emplace_back(first, c);
        }
        std::sort(cpus.begin(), cpus.end());
        for (size_t i = 0; i < cpus.size(); ++i) {
            if (i == 0 || cpus[i].first != cpus[i - 1].first) coreStart.push_back(i);
        }
    }
#pragma omp parallel
    {
#ifdef SYS_set_mempolicy
        constexpr int MPOL_LOCAL_POLICY = 4;  // MPOL_LOCAL
        syscall(SYS_set_mempolicy, MPOL_LOCAL_POLICY, nullptr, 0UL);
#endif
        if (!cpus.empty()) {
            const size_t t = omp_get_thread_num(), nt = omp_get_num_threads();
            const int cpu = nt <= coreStart.size() ? cpus[coreStart[t * coreStart.size() / nt]].second
                                                   : cpus[t % cpus.size()].second;
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
    }
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
#pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
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
#pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
#pragma omp parallel for schedule(static) reduction(max: maxError, relError)
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
    
    configureOpenMPThreads();

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
