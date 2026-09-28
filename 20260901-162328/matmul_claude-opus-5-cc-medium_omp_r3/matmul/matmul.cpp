#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../common/results_output.hpp"

// Width (in doubles) of one SIMD register of the target machine.
#if defined(__AVX512F__)
#define MM_VECW 8
#elif defined(__AVX__)
#define MM_VECW 4
#else
#define MM_VECW 2
#endif

namespace {

// Native SIMD vector of doubles, plus an unaligned view used for load/store.
using vecd = double __attribute__((vector_size(MM_VECW * sizeof(double))));
using vecdu = double __attribute__((vector_size(MM_VECW * sizeof(double)), aligned(sizeof(double))));

// Cache blocking parameters. The register tile is MR x NR; per k panel a thread
// keeps an MC x KC block of A and a KC x NC block of B in its private caches
// while the MC x NC block of C it owns stays resident across all k panels.
constexpr size_t VW = MM_VECW;
constexpr size_t MR = 6;       // rows of C per register tile
constexpr size_t NR = 2 * VW;  // columns of C per register tile
constexpr size_t MC = 256;     // rows of C per cache block
constexpr size_t NC = 256;     // columns of C per cache block
constexpr size_t KC = 48;      // depth of one k panel

// Pin the OpenMP threads to distinct CPUs unless the runtime already binds them
// (thread migration between the NUMA nodes costs more than half the throughput
// on a multi socket machine). Only CPUs the process may run on are used.
void bindThreads() {
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        return;
    }

    cpu_set_t allowed;
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
        cpu_set_t self;
        CPU_ZERO(&self);
        CPU_SET(cpus[static_cast<size_t>(omp_get_thread_num()) % cpus.size()], &self);
        sched_setaffinity(0, sizeof(self), &self);
    }
}

// Drop the pages of a freshly allocated (and therefore zero filled by a single
// thread) buffer, so that the parallel initialization below decides their NUMA
// placement by first touch. Only whole pages inside the buffer are released,
// and their contents stay zero, so this is invisible to the program.
void releasePages(std::vector<double>& mat) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t addr = reinterpret_cast<uintptr_t>(mat.data());
    const uintptr_t beg = (addr + page - 1) & ~(static_cast<uintptr_t>(page) - 1);
    const uintptr_t end = (addr + mat.size() * sizeof(double)) & ~(static_cast<uintptr_t>(page) - 1);
    if (end > beg) {
        madvise(reinterpret_cast<void*>(beg), end - beg, MADV_DONTNEED);
    }
}

} // namespace

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    releasePages(mat);

    // Row blocked distribution: matches how the multiplication below splits the
    // matrices, so every thread initializes the data it will work on.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

namespace {

// C[0..MR, 0..NR] += Apack[0..MR, 0..kc] * Bpack[0..kc, 0..NR] with the whole C
// tile held in vector registers. Apack stores MR contiguous values per k, Bpack
// NR contiguous values per k. The k iterations run in increasing order and
// accumulate into the running value of C, exactly like the scalar reference
// implementation does, so the results are bit identical to it.
inline void microKernel(const double* __restrict__ aPack, const double* __restrict__ bPack,
                        double* __restrict__ c, const size_t ldc, const size_t kc) {
    vecd acc[MR][2];
    #pragma GCC unroll 8
    for (size_t i = 0; i < MR; ++i) {
        acc[i][0] = *reinterpret_cast<const vecdu*>(c + i * ldc);
        acc[i][1] = *reinterpret_cast<const vecdu*>(c + i * ldc + VW);
    }

    for (size_t k = 0; k < kc; ++k) {
        const vecd b0 = *reinterpret_cast<const vecdu*>(bPack + k * NR);
        const vecd b1 = *reinterpret_cast<const vecdu*>(bPack + k * NR + VW);
        #pragma GCC unroll 8
        for (size_t i = 0; i < MR; ++i) {
            const double aik = aPack[k * MR + i];
            acc[i][0] += b0 * aik;
            acc[i][1] += b1 * aik;
        }
    }

    #pragma GCC unroll 8
    for (size_t i = 0; i < MR; ++i) {
        *reinterpret_cast<vecdu*>(c + i * ldc) = acc[i][0];
        *reinterpret_cast<vecdu*>(c + i * ldc + VW) = acc[i][1];
    }
}

// Copy a kc x nBlk block of B into NR wide, k major panels, zero padded.
inline void packB(const double* __restrict__ Bp, double* __restrict__ dst, const size_t N,
                  const size_t kBeg, const size_t kc, const size_t jBeg, const size_t nBlk) {
    for (size_t q = 0; q * NR < nBlk; ++q) {
        const size_t nr = std::min(NR, nBlk - q * NR);
        double* __restrict__ panel = dst + q * NR * kc;
        for (size_t k = 0; k < kc; ++k) {
            const double* __restrict__ src = Bp + (kBeg + k) * N + jBeg + q * NR;
            for (size_t j = 0; j < nr; ++j) {
                panel[k * NR + j] = src[j];
            }
            for (size_t j = nr; j < NR; ++j) {
                panel[k * NR + j] = 0.0;
            }
        }
    }
}

// Copy an mBlk x kc block of A into MR tall, k major panels, zero padded.
inline void packA(const double* __restrict__ Ap, double* __restrict__ dst, const size_t N,
                  const size_t kBeg, const size_t kc, const size_t iBeg, const size_t mBlk) {
    for (size_t p = 0; p * MR < mBlk; ++p) {
        const size_t mr = std::min(MR, mBlk - p * MR);
        double* __restrict__ panel = dst + p * MR * kc;
        for (size_t i = 0; i < mr; ++i) {
            const double* __restrict__ src = Ap + (iBeg + p * MR + i) * N + kBeg;
            for (size_t k = 0; k < kc; ++k) {
                panel[k * MR + i] = src[k];
            }
        }
        for (size_t i = mr; i < MR; ++i) {
            for (size_t k = 0; k < kc; ++k) {
                panel[k * MR + i] = 0.0;
            }
        }
    }
}

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const double* __restrict__ Ap = A.data();
    const double* __restrict__ Bp = B.data();
    double* __restrict__ Cp = C.data();

    // Shrink the cache blocks if the default ones would not expose enough
    // independent C blocks to keep every thread evenly busy.
    const size_t threads = static_cast<size_t>(omp_get_max_threads());
    size_t mc = MC;
    size_t nc = NC;
    while ((N + mc - 1) / mc * ((N + nc - 1) / nc) < 2 * threads && (mc > MR || nc > NR)) {
        if (mc * NR >= nc * MR && mc > MR) {
            mc = std::max(MR, (mc / 2 + MR - 1) / MR * MR);
        } else if (nc > NR) {
            nc = std::max(NR, (nc / 2 + NR - 1) / NR * NR);
        } else {
            break;
        }
    }

    const size_t iBlocks = (N + mc - 1) / mc;
    const size_t jBlocks = (N + nc - 1) / nc;
    const size_t kcMax = std::min(KC, N);
    const size_t aPackSize = (mc + MR - 1) / MR * MR * kcMax;
    const size_t bPackSize = (nc + NR - 1) / NR * NR * kcMax;

    // A single workspace for all threads; every thread first touches (and from
    // then on only uses) its own slice. Allocating per thread inside the
    // parallel region instead would cost more than the multiplication itself
    // for small matrices, because every thread has to set up its own heap.
    const size_t perThread = (aPackSize + bPackSize + NR - 1) / NR * NR;
    std::unique_ptr<double[]> workspace(new double[threads * perThread + NR]);
    double* const workspaceBase = workspace.get();

    #pragma omp parallel
    {
        // C is overwritten (not accumulated) by the reference implementation,
        // so every element has to start from zero before the k panels are added.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            double* __restrict__ crow = Cp + i * N;
            for (size_t j = 0; j < N; ++j) {
                crow[j] = 0.0;
            }
        }

        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        std::unique_ptr<double[]> spare;
        double* slice = workspaceBase + tid * perThread;
        if (tid >= threads) { // only reachable if the runtime hands out extra threads
            spare.reset(new double[perThread]);
            slice = spare.get();
        }
        double* const aPack = slice;
        double* const bPack = slice + aPackSize;
        double cTile[MR * NR];

        // Every (ib, jb) block of C is owned by exactly one thread, and the k
        // panels of a block are traversed in increasing order.
        #pragma omp for collapse(2) schedule(static)
        for (size_t ib = 0; ib < iBlocks; ++ib) {
            for (size_t jb = 0; jb < jBlocks; ++jb) {
                const size_t iBeg = ib * mc;
                const size_t mBlk = std::min(mc, N - iBeg);
                const size_t jBeg = jb * nc;
                const size_t nBlk = std::min(nc, N - jBeg);

                for (size_t kBeg = 0; kBeg < N; kBeg += KC) {
                    const size_t kc = std::min(KC, N - kBeg);

                    packA(Ap, aPack, N, kBeg, kc, iBeg, mBlk);
                    packB(Bp, bPack, N, kBeg, kc, jBeg, nBlk);

                    for (size_t p = 0; p * MR < mBlk; ++p) {
                        const size_t mr = std::min(MR, mBlk - p * MR);
                        const double* a = aPack + p * MR * kc;

                        for (size_t q = 0; q * NR < nBlk; ++q) {
                            const size_t nr = std::min(NR, nBlk - q * NR);
                            const double* b = bPack + q * NR * kc;
                            double* c = Cp + (iBeg + p * MR) * N + jBeg + q * NR;

                            if (mr == MR && nr == NR) {
                                microKernel(a, b, c, N, kc);
                            } else {
                                // Partial tile: run the kernel on a padded copy
                                // of C so that it may write whole vectors.
                                for (size_t i = 0; i < MR; ++i) {
                                    for (size_t j = 0; j < NR; ++j) {
                                        cTile[i * NR + j] =
                                            (i < mr && j < nr) ? c[i * N + j] : 0.0;
                                    }
                                }
                                microKernel(a, b, cTile, NR, kc);
                                for (size_t i = 0; i < mr; ++i) {
                                    for (size_t j = 0; j < nr; ++j) {
                                        c[i * N + j] = cTile[i * NR + j];
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
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    bindThreads();
    initMatrix(A, N);
    initMatrix(B, N);

    // Distribute the (still zero) result matrix over the NUMA nodes the same
    // way, so that the multiplication does not pay for the page faults.
    releasePages(C);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * N + j] = 0.0;
        }
    }
    
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
