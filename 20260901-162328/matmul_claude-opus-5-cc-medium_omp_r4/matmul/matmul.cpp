#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Allocates an N-element vector whose pages are first-touched in parallel, so
// that they are distributed over the NUMA nodes according to the (static) work
// distribution used by the compute kernels. Touching the reserved storage
// before resize() is what pins the pages; resize() only re-writes the zeros.
void allocateDistributed(std::vector<double>& vec, const size_t n) {
    vec.reserve(n);
    double* const data = vec.data();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        data[i] = 0.0;
    }
    vec.resize(n);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// --- Blocked, packed GEMM (BLIS-style) parallelized over 2D tiles of C ---------
//
// Register block of the micro-kernel: MR rows x NR columns of C are accumulated
// in registers (NR is a multiple of the 256-bit SIMD width of 4 doubles).
constexpr size_t MR = 6;
constexpr size_t NR = 8;
// Depth of one packed panel (K-blocking).
constexpr size_t KC = 128;
constexpr size_t NV = NR / 4;  // SIMD vectors (of 4 doubles) per register block row
// The two packed panels of a thread should fit in the L2 cache of a core.
constexpr size_t L2Budget = 512 * 1024;

// Computes MR x NR outer products over kc steps of the packed panels and stores
// the result into the mr_eff x nr_eff sub-block of C (overwrite for the first
// K-block, accumulate afterwards).
static inline void microKernel(const size_t kc, const double* __restrict Ap, const double* __restrict Bp,
                               double* __restrict Cp, const size_t ldc, const bool first,
                               const size_t mr_eff, const size_t nr_eff) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d acc[MR][NV];
    for (size_t r = 0; r < MR; ++r) {
        for (size_t c = 0; c < NV; ++c) {
            acc[r][c] = _mm256_setzero_pd();
        }
    }

// Unrolling this loop would spill the accumulators out of the 16 AVX registers.
#pragma GCC unroll 1
    for (size_t k = 0; k < kc; ++k) {
        const double* __restrict b = Bp + k * NR;
        const double* __restrict a = Ap + k * MR;
        __m256d bv[NV];
        for (size_t c = 0; c < NV; ++c) {
            bv[c] = _mm256_loadu_pd(b + 4 * c);
        }
        for (size_t r = 0; r < MR; ++r) {
            const __m256d av = _mm256_broadcast_sd(a + r);
            for (size_t c = 0; c < NV; ++c) {
                acc[r][c] = _mm256_fmadd_pd(av, bv[c], acc[r][c]);
            }
        }
    }

    if (mr_eff == MR && nr_eff == NR) {
        for (size_t r = 0; r < MR; ++r) {
            double* __restrict dst = Cp + r * ldc;
            for (size_t c = 0; c < NV; ++c) {
                const __m256d v = first ? acc[r][c] : _mm256_add_pd(_mm256_loadu_pd(dst + 4 * c), acc[r][c]);
                _mm256_storeu_pd(dst + 4 * c, v);
            }
        }
    } else {
        alignas(32) double tmp[MR][NR];
        for (size_t r = 0; r < MR; ++r) {
            for (size_t c = 0; c < NV; ++c) {
                _mm256_store_pd(&tmp[r][4 * c], acc[r][c]);
            }
        }
        for (size_t r = 0; r < mr_eff; ++r) {
            for (size_t c = 0; c < nr_eff; ++c) {
                if (first) {
                    Cp[r * ldc + c] = tmp[r][c];
                } else {
                    Cp[r * ldc + c] += tmp[r][c];
                }
            }
        }
    }
#else
    double acc[MR][NR];
    for (size_t r = 0; r < MR; ++r) {
        for (size_t c = 0; c < NR; ++c) {
            acc[r][c] = 0.0;
        }
    }

    for (size_t k = 0; k < kc; ++k) {
        const double* __restrict b = Bp + k * NR;
        const double* __restrict a = Ap + k * MR;
        for (size_t r = 0; r < MR; ++r) {
            const double av = a[r];
            for (size_t c = 0; c < NR; ++c) {
                acc[r][c] += av * b[c];
            }
        }
    }

    for (size_t r = 0; r < mr_eff; ++r) {
        for (size_t c = 0; c < nr_eff; ++c) {
            if (first) {
                Cp[r * ldc + c] = acc[r][c];
            } else {
                Cp[r * ldc + c] += acc[r][c];
            }
        }
    }
#endif
}

// Packs A[i0:i1, pc:pc+kc] into MR-row panels: Ap[(panel * kc + k) * MR + r].
static void packA(const double* __restrict A, const size_t N, const size_t i0, const size_t i1,
                  const size_t pc, const size_t kc, double* __restrict Ap) {
    const size_t m = i1 - i0;
    size_t panel = 0;
    for (size_t i = 0; i < m; i += MR, ++panel) {
        const size_t mr_eff = std::min(MR, m - i);
        double* __restrict dst = Ap + panel * kc * MR;
        for (size_t k = 0; k < kc; ++k) {
            for (size_t r = 0; r < mr_eff; ++r) {
                dst[k * MR + r] = A[(i0 + i + r) * N + pc + k];
            }
            for (size_t r = mr_eff; r < MR; ++r) {
                dst[k * MR + r] = 0.0;
            }
        }
    }
}

// Packs B[pc:pc+kc, j0:j1] into NR-column panels: Bp[(panel * kc + k) * NR + c].
static void packB(const double* __restrict B, const size_t N, const size_t pc, const size_t kc,
                  const size_t j0, const size_t j1, double* __restrict Bp) {
    const size_t n = j1 - j0;
    size_t panel = 0;
    for (size_t j = 0; j < n; j += NR, ++panel) {
        const size_t nr_eff = std::min(NR, n - j);
        double* __restrict dst = Bp + panel * kc * NR;
        for (size_t k = 0; k < kc; ++k) {
            const double* __restrict src = B + (pc + k) * N + j0 + j;
            for (size_t c = 0; c < nr_eff; ++c) {
                dst[k * NR + c] = src[c];
            }
            for (size_t c = nr_eff; c < NR; ++c) {
                dst[k * NR + c] = 0.0;
            }
        }
    }
}

// Rounds x up to the next multiple of m.
static inline size_t roundUp(const size_t x, const size_t m) noexcept { return ((x + m - 1) / m) * m; }

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    const size_t threads = static_cast<size_t>(omp_get_max_threads());

    // Pick the C tile (MC x NC) that one thread works on: blocks must be large
    // enough to amortize packing and to keep the micro-kernel efficient, small
    // enough that the tile grid balances over all threads and that the packed
    // panels stay in L2 (shared by the two SMT threads of a core).
    constexpr size_t mcChoices[] = {384, 256, 192, 128, 96, 64, 48};
    constexpr size_t ncChoices[] = {256, 128, 96, 64, 48, 32};
    size_t MC = roundUp(std::min<size_t>(mcChoices[0], N), MR);
    size_t NC = roundUp(std::min<size_t>(ncChoices[0], N), NR);
    double bestScore = -1.0;
    for (const size_t mcRaw : mcChoices) {
        const size_t mc = roundUp(std::min(mcRaw, N), MR);
        const size_t tilesI = (N + mc - 1) / mc;
        for (const size_t ncRaw : ncChoices) {
            const size_t nc = roundUp(std::min(ncRaw, N), NR);
            const size_t tilesJ = (N + nc - 1) / nc;
            if ((mc + nc) * KC * sizeof(double) > L2Budget) {
                continue;  // packed panels would not fit in L2
            }
            const size_t tiles = tilesI * tilesJ;
            // Fraction of the peak that the dynamically scheduled tile grid can
            // reach: with few tiles per thread the makespan is dominated by the
            // slowest thread and by the partially filled last "round" of tiles.
            const double balance = 1.0 / (1.0 + static_cast<double>(threads) / static_cast<double>(tiles));
            // Relative efficiency of a tile: the smaller it is, the more often
            // A and B are re-packed and C is re-read per flop.
            const double efficiency =
                (static_cast<double>(mc) / (mc + 64)) * (static_cast<double>(nc) / (nc + 64));
            const double score = balance * efficiency;
            if (score > bestScore) {
                bestScore = score;
                MC = mc;
                NC = nc;
            }
        }
    }

    const size_t tilesI = (N + MC - 1) / MC;
    const size_t tilesJ = (N + NC - 1) / NC;

    const double* __restrict Ad = A.data();
    const double* __restrict Bd = B.data();
    double* __restrict Cd = C.data();

#pragma omp parallel
    {
        // Per-thread packing buffers, first-touched by their owner thread.
        std::vector<double> ApackBuf(MC * KC);
        std::vector<double> BpackBuf(NC * KC);
        double* const Apack = ApackBuf.data();
        double* const Bpack = BpackBuf.data();

#pragma omp for collapse(2) schedule(dynamic)
        for (size_t it = 0; it < tilesI; ++it) {
            for (size_t jt = 0; jt < tilesJ; ++jt) {
                const size_t i0 = it * MC;
                const size_t i1 = std::min(i0 + MC, N);
                const size_t j0 = jt * NC;
                const size_t j1 = std::min(j0 + NC, N);

                for (size_t pc = 0; pc < N; pc += KC) {
                    const size_t kc = std::min(KC, N - pc);
                    packB(Bd, N, pc, kc, j0, j1, Bpack);
                    packA(Ad, N, i0, i1, pc, kc, Apack);

                    const bool first = (pc == 0);
                    size_t jp = 0;
                    for (size_t j = j0; j < j1; j += NR, ++jp) {
                        const size_t nr_eff = std::min(NR, j1 - j);
                        const double* Bp = Bpack + jp * kc * NR;
                        size_t ip = 0;
                        for (size_t i = i0; i < i1; i += MR, ++ip) {
                            const size_t mr_eff = std::min(MR, i1 - i);
                            microKernel(kc, Apack + ip * kc * MR, Bp, Cd + i * N + j, N, first,
                                        mr_eff, nr_eff);
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
    
    // Allocate matrices (pages distributed over NUMA nodes by first touch)
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;
    allocateDistributed(A, N * N);
    allocateDistributed(B, N * N);
    allocateDistributed(C, N * N);
    
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
