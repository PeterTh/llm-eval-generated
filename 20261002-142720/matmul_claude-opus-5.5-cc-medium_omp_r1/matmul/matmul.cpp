#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <new>

#include <omp.h>
#include <sched.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    double* const m = mat.data();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked, packed, OpenMP-parallel GEMM.
//
// Every C element is still computed as sum = 0; for k = 0..N-1: sum += a*b
// in strictly increasing k order (k blocks are processed in order and partial
// sums are carried through C), so results match the naive loop exactly.
// Products and additions are rounded separately (built with
// -ffp-contract=off), except that for odd N the final term is fused, which
// reproduces the reference build's rounding (its vectorised k loop handles
// pairs with mul+add and the odd remainder with a single FMA).
// ---------------------------------------------------------------------------
namespace {

constexpr size_t MR = 6;      // micro-tile rows
constexpr size_t NR = 8;      // micro-tile cols (2 AVX2 vectors)
constexpr size_t KC = 256;    // k block: B micro-panel KC*NR*8 = 16 KiB (L1)
constexpr size_t MCP = 16;    // A sub-block of MCP*MR rows x KC (192 KiB, L2)
constexpr size_t TN_MAX = 512; // max tile width: B block KC x TN_MAX <= 1 MiB (L3 reuse)

// Growable, 64-byte aligned scratch buffer. Packing buffers are kept for the
// lifetime of the program (like BLAS workspaces): releasing hundreds of MiB
// of freshly touched memory costs as much as the multiplication itself.
struct Workspace {
    double* p = nullptr;
    size_t bytes = 0;
    Workspace() = default;
    ~Workspace() { std::free(p); }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    double* get(const size_t n) {
        const size_t need = n * sizeof(double);
        if (need > bytes) {
            std::free(p);
            bytes = (need + 63) / 64 * 64;
            p = static_cast<double*>(std::aligned_alloc(64, bytes));
            if (!p) {
                bytes = 0;
                throw std::bad_alloc();
            }
        }
        return p;
    }
};

#if defined(__AVX2__) && defined(__FMA__)
template <bool FUSED>
static inline __attribute__((always_inline)) void mmStep(__m256d (&c)[MR][2], const double* __restrict Ap,
                                                         const double* __restrict Bp) {
    const __m256d b0 = _mm256_load_pd(Bp);
    const __m256d b1 = _mm256_load_pd(Bp + 4);
#pragma GCC unroll 8
    for (size_t r = 0; r < MR; ++r) {
        const __m256d ar = _mm256_broadcast_sd(Ap + r);
        if constexpr (FUSED) {
            c[r][0] = _mm256_fmadd_pd(ar, b0, c[r][0]);
            c[r][1] = _mm256_fmadd_pd(ar, b1, c[r][1]);
        } else {
            c[r][0] = _mm256_add_pd(c[r][0], _mm256_mul_pd(ar, b0));
            c[r][1] = _mm256_add_pd(c[r][1], _mm256_mul_pd(ar, b1));
        }
    }
}
#endif

// Full MR x NR tile: C (ldc) = (first ? 0 : C) + sum_k Ap[k][:] * Bp[k][:]
inline void microKernel(const double* __restrict Ap, const double* __restrict Bp,
                        double* __restrict C, const size_t ldc, const size_t kc,
                        const bool first, const bool fuseLast) {
    const size_t kMain = fuseLast ? kc - 1 : kc;
#if defined(__AVX2__) && defined(__FMA__)
    __m256d c[MR][2];
#pragma GCC unroll 8
    for (size_t r = 0; r < MR; ++r) {
        if (first) {
            c[r][0] = _mm256_setzero_pd();
            c[r][1] = _mm256_setzero_pd();
        } else {
            c[r][0] = _mm256_loadu_pd(C + r * ldc);
            c[r][1] = _mm256_loadu_pd(C + r * ldc + 4);
        }
    }
#pragma GCC unroll 4
    for (size_t k = 0; k < kMain; ++k) mmStep<false>(c, Ap + k * MR, Bp + k * NR);
    if (fuseLast) mmStep<true>(c, Ap + (kc - 1) * MR, Bp + (kc - 1) * NR);
#pragma GCC unroll 8
    for (size_t r = 0; r < MR; ++r) {
        _mm256_storeu_pd(C + r * ldc, c[r][0]);
        _mm256_storeu_pd(C + r * ldc + 4, c[r][1]);
    }
#else
    double c[MR][NR];
    for (size_t r = 0; r < MR; ++r)
        for (size_t q = 0; q < NR; ++q) c[r][q] = first ? 0.0 : C[r * ldc + q];
    for (size_t k = 0; k < kMain; ++k)
        for (size_t r = 0; r < MR; ++r)
            for (size_t q = 0; q < NR; ++q)
                c[r][q] += Ap[k * MR + r] * Bp[k * NR + q];
    if (fuseLast)
        for (size_t r = 0; r < MR; ++r)
            for (size_t q = 0; q < NR; ++q)
                c[r][q] = std::fma(Ap[(kc - 1) * MR + r], Bp[(kc - 1) * NR + q], c[r][q]);
    for (size_t r = 0; r < MR; ++r)
        for (size_t q = 0; q < NR; ++q) C[r * ldc + q] = c[r][q];
#endif
}

// Partial tile (mr <= MR, nr <= NR) at the matrix border, via a scratch tile.
inline void microKernelEdge(const double* Ap, const double* Bp, double* C,
                            const size_t ldc, const size_t kc, const bool first,
                            const bool fuseLast, const size_t mr, const size_t nr) {
    alignas(64) double t[MR * NR] = {};
    if (!first)
        for (size_t r = 0; r < mr; ++r)
            for (size_t q = 0; q < nr; ++q) t[r * NR + q] = C[r * ldc + q];
    microKernel(Ap, Bp, t, NR, kc, first, fuseLast);
    for (size_t r = 0; r < mr; ++r)
        for (size_t q = 0; q < nr; ++q) C[r * ldc + q] = t[r * NR + q];
}

// Choose an mT x nT grid of C tiles (in micro-panel units) for nThreads
// threads, balancing load (tiles per thread) against memory traffic
// (perimeter per area) with a simple cost model.
void chooseGrid(const size_t mPanels, const size_t nPanels, const size_t nThreads,
                size_t& bestM, size_t& bestN) {
    constexpr double TRAFFIC = 4.0;   // relative cost of loading one row/col per k
    const size_t nMin = std::min(nPanels, std::max<size_t>(1, (nPanels * NR + TN_MAX - 1) / TN_MAX));
    double best = -1.0;
    bestM = 1;
    bestN = nMin;
    for (size_t mT = 1; mT <= mPanels && mT <= 8 * nThreads; ++mT) {
        for (size_t nT = nMin; nT <= nPanels; ++nT) {
            const size_t tiles = mT * nT;
            const double tm = double((mPanels + mT - 1) / mT * MR);
            const double tn = double((nPanels + nT - 1) / nT * NR);
            const double rounds = double((tiles + nThreads - 1) / nThreads);
            const double cost = rounds * (tm * tn + TRAFFIC * (tm + tn));
            if (best < 0.0 || cost < best) {
                best = cost;
                bestM = mT;
                bestN = nT;
            }
            if (tiles > 8 * nThreads) break;
        }
    }
}

} // namespace

// Pin each OpenMP thread to its own CPU (in the process' affinity mask
// order), unless the user already requested a binding policy. Avoids thread
// migration, which badly hurts wake-up latency and cache reuse on many-core
// systems.
void bindOmpThreads() {
    if (omp_get_proc_bind() != omp_proc_bind_false) return;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return;
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    if (cpus.empty()) return;
    #pragma omp parallel
    {
        cpu_set_t own;
        CPU_ZERO(&own);
        CPU_SET(cpus[static_cast<size_t>(omp_get_thread_num()) % cpus.size()], &own);
        sched_setaffinity(0, sizeof(own), &own);
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;
    const double* const a = A.data();
    const double* const b = B.data();
    double* const c = C.data();

    const size_t mPanels = (N + MR - 1) / MR;
    const size_t nPanels = (N + NR - 1) / NR;
    const size_t kBlocks = (N + KC - 1) / KC;
    // Small problems do not amortise waking (and synchronising across
    // sockets) the whole machine: use at most one thread per ~4 MFLOP.
    const double flops = 2.0 * double(N) * double(N) * double(N);
    const int nThreads = static_cast<int>(
        std::clamp(flops / 4e6, 1.0, double(omp_get_max_threads())));
    size_t mTiles, nTiles;
    chooseGrid(mPanels, nPanels, static_cast<size_t>(nThreads), mTiles, nTiles);

    // Packed layouts (zero padded to full micro-panels):
    //   Ap[ip][k][0..MR)  rows ip*MR.., all k contiguous per panel
    //   Bp[jp][k][0..NR)  cols jp*NR.., all k contiguous per panel
    static Workspace wsA, wsB;
    double* const Ap = wsA.get(mPanels * N * MR);
    double* const Bp = wsB.get(nPanels * N * NR);

    #pragma omp parallel num_threads(nThreads)
    {
        #pragma omp for schedule(static) nowait
        for (size_t ip = 0; ip < mPanels; ++ip) {
            double* dst = Ap + ip * N * MR;
            const size_t i0 = ip * MR;
            const size_t mr = std::min(MR, N - i0);
            for (size_t k = 0; k < N; ++k) {
                for (size_t r = 0; r < mr; ++r) dst[k * MR + r] = a[(i0 + r) * N + k];
                for (size_t r = mr; r < MR; ++r) dst[k * MR + r] = 0.0;
            }
        }
        #pragma omp for schedule(static)
        for (size_t jp = 0; jp < nPanels; ++jp) {
            double* dst = Bp + jp * N * NR;
            const size_t j0 = jp * NR;
            const size_t nr = std::min(NR, N - j0);
            for (size_t k = 0; k < N; ++k) {
                const double* src = b + k * N + j0;
                for (size_t q = 0; q < nr; ++q) dst[k * NR + q] = src[q];
                for (size_t q = nr; q < NR; ++q) dst[k * NR + q] = 0.0;
            }
        }

        // Each task computes one tile of C over the full k range; k blocks
        // run in order, partial sums are carried in C.
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (size_t it = 0; it < mTiles; ++it) {
            for (size_t jt = 0; jt < nTiles; ++jt) {
                const size_t ip0 = it * mPanels / mTiles, ipEnd = (it + 1) * mPanels / mTiles;
                const size_t jp0 = jt * nPanels / nTiles, jpEnd = (jt + 1) * nPanels / nTiles;
                for (size_t kb = 0; kb < kBlocks; ++kb) {
                    const size_t k0 = kb * KC;
                    const size_t kc = std::min(KC, N - k0);
                    const bool first = (kb == 0);
                    const bool fuseLast = (N & 1) && (kb == kBlocks - 1);
                    for (size_t is = ip0; is < ipEnd; is += MCP) {
                        const size_t isEnd = std::min(ipEnd, is + MCP);
                        for (size_t jp = jp0; jp < jpEnd; ++jp) {
                            const size_t j = jp * NR;
                            const size_t nr = std::min(NR, N - j);
                            const double* bp = Bp + jp * N * NR + k0 * NR;
                            for (size_t ip = is; ip < isEnd; ++ip) {
                                const size_t i = ip * MR;
                                const size_t mr = std::min(MR, N - i);
                                const double* ap = Ap + ip * N * MR + k0 * MR;
                                if (mr == MR && nr == NR)
                                    microKernel(ap, bp, c + i * N + j, N, kc, first, fuseLast);
                                else
                                    microKernelEdge(ap, bp, c + i * N + j, N, kc, first, fuseLast,
                                                    mr, nr);
                            }
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
    
    bindOmpThreads();

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
