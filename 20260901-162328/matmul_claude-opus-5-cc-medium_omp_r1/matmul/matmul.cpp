#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "../common/results_output.hpp"

// Pins each OpenMP thread to its own CPU. Cache blocking and NUMA-local
// first touch are both worthless if the OS is free to migrate threads, and
// libgomp performs no binding at all unless it is asked to. Any binding
// configured through the OpenMP environment takes precedence, and only CPUs
// already allowed for this process are used (so taskset/cgroup limits and
// OMP_NUM_THREADS keep working).
void setupThreadAffinity() {
#if defined(__linux__)
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            cpus.push_back(cpu);
        }
    }
    if (cpus.empty()) {
        return;
    }

#pragma omp parallel
    {
        cpu_set_t mine;
        CPU_ZERO(&mine);
        CPU_SET(cpus[static_cast<size_t>(omp_get_thread_num()) % cpus.size()], &mine);
        pthread_setaffinity_np(pthread_self(), sizeof(mine), &mine);
    }
#endif
}

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    // Parallel row-wise initialization; this also spreads the pages over the
    // NUMA nodes by row (first touch), matching the row-major tiling that
    // matrixMultiply() uses to distribute work.
    double* __restrict m = mat.data();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked, packed, register-tiled, OpenMP-parallel matrix multiplication
// (GotoBLAS-style: cache blocking + packing + AVX2/FMA micro-kernel).
//
// Every element of C accumulates its products in strictly increasing k order
// with a single accumulator (partial results are carried in the C tile across
// k-blocks), i.e. the summation order per output element is exactly the one of
// the original serial loop nest -- only the loop nesting and the data layout of
// the inputs change, and zero padding of edge micro-panels adds exact zeros.
// Results are therefore reproducible and, like the original -O3 -march=native
// build, use fused multiply-add; they agree with the serial code to the last
// ulp (bit-identical whenever the products are exact, e.g. for N = 2^k).
// ---------------------------------------------------------------------------

// Micro-kernel tile: MR rows x NR columns of C held in vector registers.
constexpr size_t MR = 6;
constexpr size_t NR = 8;

// Cache blocking (per thread): A block is MC x KC, B block is KC x NC.
constexpr size_t KC = 256;
constexpr size_t MC = MR * 6;
constexpr size_t NC = NR * 32;

// C[0..MR-1][0..NR-1] += sum_{k<kc} ap[k][*] * bp[k][*], k ascending.
// ap holds MR values per k, bp holds NR values per k, both contiguous.
static inline void microKernel(const double* __restrict ap, const double* __restrict bp,
                               const size_t kc, const size_t ldb, double* __restrict c,
                               const size_t ldc, const bool accumulate) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (accumulate) {
        c00 = _mm256_loadu_pd(c + 0 * ldc);
        c01 = _mm256_loadu_pd(c + 0 * ldc + 4);
        c10 = _mm256_loadu_pd(c + 1 * ldc);
        c11 = _mm256_loadu_pd(c + 1 * ldc + 4);
        c20 = _mm256_loadu_pd(c + 2 * ldc);
        c21 = _mm256_loadu_pd(c + 2 * ldc + 4);
        c30 = _mm256_loadu_pd(c + 3 * ldc);
        c31 = _mm256_loadu_pd(c + 3 * ldc + 4);
        c40 = _mm256_loadu_pd(c + 4 * ldc);
        c41 = _mm256_loadu_pd(c + 4 * ldc + 4);
        c50 = _mm256_loadu_pd(c + 5 * ldc);
        c51 = _mm256_loadu_pd(c + 5 * ldc + 4);
    } else {
        c00 = c01 = c10 = c11 = c20 = c21 = _mm256_setzero_pd();
        c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_pd();
    }

    for (size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(bp + k * ldb);
        const __m256d b1 = _mm256_loadu_pd(bp + k * ldb + 4);
        const double* __restrict av = ap + k * MR;

        __m256d a = _mm256_broadcast_sd(av + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(av + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(av + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(av + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(av + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40);
        c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(av + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50);
        c51 = _mm256_fmadd_pd(a, b1, c51);
    }

    _mm256_storeu_pd(c + 0 * ldc, c00);
    _mm256_storeu_pd(c + 0 * ldc + 4, c01);
    _mm256_storeu_pd(c + 1 * ldc, c10);
    _mm256_storeu_pd(c + 1 * ldc + 4, c11);
    _mm256_storeu_pd(c + 2 * ldc, c20);
    _mm256_storeu_pd(c + 2 * ldc + 4, c21);
    _mm256_storeu_pd(c + 3 * ldc, c30);
    _mm256_storeu_pd(c + 3 * ldc + 4, c31);
    _mm256_storeu_pd(c + 4 * ldc, c40);
    _mm256_storeu_pd(c + 4 * ldc + 4, c41);
    _mm256_storeu_pd(c + 5 * ldc, c50);
    _mm256_storeu_pd(c + 5 * ldc + 4, c51);
#else
    // Portable path: the compiler vectorizes the NR dimension.
    double acc[MR][NR];
    for (size_t i = 0; i < MR; ++i) {
        for (size_t j = 0; j < NR; ++j) {
            acc[i][j] = accumulate ? c[i * ldc + j] : 0.0;
        }
    }
    for (size_t k = 0; k < kc; ++k) {
        for (size_t i = 0; i < MR; ++i) {
            const double a = ap[k * MR + i];
            for (size_t j = 0; j < NR; ++j) {
                acc[i][j] += a * bp[k * ldb + j];
            }
        }
    }
    for (size_t i = 0; i < MR; ++i) {
        for (size_t j = 0; j < NR; ++j) {
            c[i * ldc + j] = acc[i][j];
        }
    }
#endif
}

// Packs B[pc..pc+kc)[jc..jc+nc) into micro-panels of NR columns
// (layout: panel, then k, then NR contiguous values). Edges are zero-padded.
static inline void packB(const double* __restrict B, const size_t N, const size_t pc,
                         const size_t kc, const size_t jc, const size_t nc,
                         double* __restrict Bp) {
    for (size_t jp = 0; jp * NR < nc; ++jp) {
        double* __restrict dst = Bp + jp * kc * NR;
        const size_t j = jc + jp * NR;
        const size_t nr = std::min(NR, jc + nc - j);
        if (nr == NR) {
            for (size_t k = 0; k < kc; ++k) {
                const double* __restrict src = B + (pc + k) * N + j;
                for (size_t t = 0; t < NR; ++t) {
                    dst[k * NR + t] = src[t];
                }
            }
        } else {
            for (size_t k = 0; k < kc; ++k) {
                const double* __restrict src = B + (pc + k) * N + j;
                for (size_t t = 0; t < NR; ++t) {
                    dst[k * NR + t] = (t < nr) ? src[t] : 0.0;
                }
            }
        }
    }
}

// Packs A[ic..ic+mc)[pc..pc+kc) into micro-panels of MR rows
// (layout: panel, then k, then MR contiguous values). Edges are zero-padded.
static inline void packA(const double* __restrict A, const size_t N, const size_t ic,
                         const size_t mc, const size_t pc, const size_t kc,
                         double* __restrict Ap) {
    for (size_t ip = 0; ip * MR < mc; ++ip) {
        double* __restrict dst = Ap + ip * kc * MR;
        const size_t i = ic + ip * MR;
        const size_t mr = std::min(MR, ic + mc - i);
        for (size_t r = 0; r < MR; ++r) {
            const double* __restrict src = A + (i + r) * N + pc;
            if (r < mr) {
                for (size_t k = 0; k < kc; ++k) {
                    dst[k * MR + r] = src[k];
                }
            } else {
                for (size_t k = 0; k < kc; ++k) {
                    dst[k * MR + r] = 0.0;
                }
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double* __restrict c = C.data();

    const size_t rowBlocks = (N + MR - 1) / MR;
    const size_t colBlocks = (N + NR - 1) / NR;
    const size_t nThreads = static_cast<size_t>(omp_get_max_threads());

    // 2D thread grid over (row blocks x column blocks). A pure 1D row split
    // starves threads as soon as N/MR < nThreads, and since the memory traffic
    // of a thread grows with the perimeter of its C tile, squarish tiles are
    // the cheapest; keeping every thread busy comes first.
    size_t pr = nThreads, pc = 1, bestActive = 0, bestCost = ~size_t(0);
    for (size_t cand = 1; cand <= nThreads; ++cand) {
        if (nThreads % cand != 0) {
            continue;
        }
        const size_t rows = nThreads / cand;
        const size_t active = std::min(rows, rowBlocks) * std::min(cand, colBlocks);
        const size_t cost =
            ((rowBlocks + rows - 1) / rows) * MR + ((colBlocks + cand - 1) / cand) * NR;
        if (active > bestActive || (active == bestActive && cost < bestCost)) {
            bestActive = active;
            bestCost = cost;
            pr = rows;
            pc = cand;
        }
    }

    // Blocking sizes and packing buffers are identical for all threads, so the
    // whole workspace is allocated once here: per-thread allocation inside the
    // parallel region costs milliseconds of malloc/mmap serialization on
    // many-core machines, which dwarfs the work itself for small matrices.
    const size_t maxRows = ((rowBlocks + pr - 1) / pr) * MR;
    const size_t maxCols = ((colBlocks + pc - 1) / pc) * NR;
    // The k-block size trades cache blocking quality against the size of the
    // (freshly faulted) packing workspace, which is pure overhead for small
    // matrices; scale it with the problem instead of always using the maximum.
    const size_t kcMax = std::min(std::min(KC, N), std::max<size_t>(64, N / 32));
    const size_t mcMax = std::min(MC, maxRows);
    const size_t ncMax = std::min(NC, maxCols);

    // Packing B pays off only if the packed block is reused by enough rows of
    // A; otherwise B is streamed straight from the matrix, where its NR values
    // per k are contiguous as well.
    const bool packBlockB = maxRows >= 4 * MR;

    const size_t aWords = mcMax * kcMax;
    const size_t bWords = (packBlockB ? ncMax : NR) * kcMax;
    // Slices are multiples of a cache line and start on one, so that packing
    // by one thread never invalidates a line owned by another.
    constexpr size_t lineWords = 8;
    const size_t slice = ((aWords + bWords + lineWords - 1) / lineWords) * lineWords;
    // Uninitialized on purpose: the pages are first-touched by their own thread.
    std::unique_ptr<double[]> workspace(new double[slice * nThreads + lineWords]);
    const uintptr_t wsAddr = reinterpret_cast<uintptr_t>(workspace.get());
    double* const wsBase = reinterpret_cast<double*>((wsAddr + 63) & ~uintptr_t(63));

#pragma omp parallel num_threads(nThreads)
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const size_t rowRank = tid / pc;
        const size_t colRank = tid % pc;

        // Contiguous, evenly sized ranges of whole micro-tiles per thread.
        const size_t iStart = (rowBlocks * rowRank / pr) * MR;
        const size_t iEnd = std::min((rowBlocks * (rowRank + 1) / pr) * MR, N);
        const size_t jStart = (colBlocks * colRank / pc) * NR;
        const size_t jEnd = std::min((colBlocks * (colRank + 1) / pc) * NR, N);

        if (iStart < iEnd && jStart < jEnd) {
            double* __restrict Ap = wsBase + tid * slice;
            double* __restrict Bp = Ap + aWords;
            double tile[MR * NR] = {};

            for (size_t jb = jStart; jb < jEnd; jb += ncMax) {
                const size_t nc = std::min(ncMax, jEnd - jb);
                for (size_t kb = 0; kb < N; kb += kcMax) {
                    const size_t kc = std::min(kcMax, N - kb);
                    const bool accumulate = kb != 0;
                    if (packBlockB) {
                        packB(b, N, kb, kc, jb, nc, Bp);
                    }

                    for (size_t ib = iStart; ib < iEnd; ib += mcMax) {
                        const size_t mc = std::min(mcMax, iEnd - ib);
                        packA(a, N, ib, mc, kb, kc, Ap);

                        for (size_t jp = 0; jp * NR < nc; ++jp) {
                            const size_t j = jb + jp * NR;
                            const size_t nr = std::min(NR, jEnd - j);

                            const double* __restrict bp;
                            size_t ldb;
                            if (packBlockB) {
                                bp = Bp + jp * kc * NR;
                                ldb = NR;
                            } else if (nr == NR) {
                                bp = b + kb * N + j;
                                ldb = N;
                            } else {
                                // Zero-pad the trailing partial column panel so
                                // that the kernel never reads past the matrix.
                                packB(b, N, kb, kc, j, nr, Bp);
                                bp = Bp;
                                ldb = NR;
                            }

                            for (size_t ip = 0; ip * MR < mc; ++ip) {
                                const size_t i = ib + ip * MR;
                                const size_t mr = std::min(MR, iEnd - i);
                                double* __restrict cp = c + i * N + j;

                                if (mr == MR && nr == NR) {
                                    microKernel(Ap + ip * kc * MR, bp, kc, ldb, cp, N, accumulate);
                                } else {
                                    // Edge tile: stage the partial results so the
                                    // kernel can work on a padded MR x NR tile.
                                    if (accumulate) {
                                        for (size_t r = 0; r < mr; ++r) {
                                            for (size_t t = 0; t < nr; ++t) {
                                                tile[r * NR + t] = cp[r * N + t];
                                            }
                                        }
                                    }
                                    microKernel(Ap + ip * kc * MR, bp, kc, ldb, tile, NR,
                                                accumulate);
                                    for (size_t r = 0; r < mr; ++r) {
                                        for (size_t t = 0; t < nr; ++t) {
                                            cp[r * N + t] = tile[r * NR + t];
                                        }
                                    }
                                }
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

    setupThreadAffinity();
    
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
