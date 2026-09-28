#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#include <unistd.h>

#include "../common/results_output.hpp"

// Micro-kernel dimensions: MR rows of C times NR columns of C are accumulated
// in registers (MR * NR / 4 = 12 AVX2 vector accumulators).
constexpr size_t MR = 6;
constexpr size_t NR = 8;
// Work units (rectangles of the C tile grid) handed to each thread; more than
// one allows the dynamic schedule to even out imbalances.
constexpr size_t UNITS_PER_THREAD = 2;
// Amount of packed A a thread keeps resident in its private L2 cache.
constexpr size_t L2_A_BYTES = 64 * 1024;

// The hardware threads sharing a physical core with the given CPU, as listed
// by the kernel (empty if the topology is not exposed).
std::vector<int> threadSiblings(const int cpu) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE* f = fopen(path, "r");
    if (f == nullptr) {
        return {};
    }
    char buf[512] = {};
    const size_t read = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[read] = '\0';

    // The list is a comma separated sequence of single CPUs and ranges.
    std::vector<int> siblings;
    for (const char* p = buf; *p != '\0';) {
        char* next = nullptr;
        const long lo = strtol(p, &next, 10);
        if (next == p) {
            break;
        }
        long hi = lo;
        if (*next == '-') {
            hi = strtol(next + 1, &next, 10);
        }
        for (long c = lo; c <= hi; ++c) {
            siblings.push_back(static_cast<int>(c));
        }
        p = (*next == ',') ? next + 1 : next;
    }
    return siblings;
}

// Number of hardware threads that share one physical core (1 if unknown).
size_t hardwareThreadsPerCore() {
    static const size_t perCore = std::max<size_t>(1, threadSiblings(0).size());
    return perCore;
}

// CPUs usable by this process, grouped by SMT level: level 0 holds one CPU per
// physical core, level 1 their second hardware thread, and so on.
std::vector<std::vector<int>> cpusBySmtLevel() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return {};
    }

    std::vector<std::vector<int>> smtLevels;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        const std::vector<int> siblings = threadSiblings(cpu);
        size_t level = 0;
        for (const int sibling : siblings) {
            if (sibling == cpu) {
                break;
            }
            if (CPU_ISSET(sibling, &allowed)) {
                ++level;
            }
        }
        if (smtLevels.size() <= level) {
            smtLevels.resize(level + 1);
        }
        smtLevels[level].push_back(cpu);
    }

    return smtLevels;
}

// Pin the threads of the current team, spread over the physical cores. libgomp
// parses its environment before main() runs, so the placement cannot be set
// with setenv(); it is left to the runtime if the user configured it.
void pinThreads() {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr) {
        return;
    }
    const std::vector<std::vector<int>> levels = cpusBySmtLevel();
    if (levels.empty() || levels[0].empty()) {
        return;
    }
    const size_t nCores = levels[0].size();

    #pragma omp parallel
    {
        const size_t nThreads = static_cast<size_t>(omp_get_num_threads());
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        int cpu;
        if (nThreads <= nCores) {
            // One thread per physical core, spread over the whole machine.
            cpu = levels[0][(tid * nCores) / nThreads];
        } else {
            // All cores are busy; further threads go to the SMT siblings.
            const auto& level = levels[(tid / nCores) % levels.size()];
            cpu = level[(tid % nCores) % level.size()];
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}

// k blocking factor: bounds the packed slices a thread works on at once. When
// SMT siblings share the private caches, both slices have to be half as large.
size_t chooseKC(const size_t nThreads) {
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    const size_t logical = online > 0 ? static_cast<size_t>(online) : nThreads;
    const size_t cores = std::max<size_t>(1, logical / hardwareThreadsPerCore());
    return nThreads > cores ? 128 : 256;
}

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Drop the pages of a freshly allocated buffer so that the parallel loop
// writing it first can place them NUMA-locally (std::vector zero-initializes
// on the master thread, which would otherwise bind everything to one node).
void resetFirstTouch(void* ptr, const size_t bytes) {
    const uintptr_t pageSize = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t begin = (base + pageSize - 1) & ~(pageSize - 1);
    const uintptr_t end = (base + bytes) & ~(pageSize - 1);
    if (end > begin) {
        madvise(reinterpret_cast<void*>(begin), end - begin, MADV_DONTNEED);
    }
}

// Distribute the pages of a matrix over the NUMA nodes and pre-fault them, so
// that no page fault storm happens inside the timed multiplication.
void firstTouchMatrix(std::vector<double>& mat, const size_t N) {
    resetFirstTouch(mat.data(), mat.size() * sizeof(double));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = 0.0;
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    resetFirstTouch(mat.data(), mat.size() * sizeof(double));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Accumulate the full K-length dot products of an MR x NR tile of C in
// registers. The reduction over k runs in ascending order with a single
// accumulator per element, exactly as in the scalar reference implementation.
#if defined(__AVX2__) && defined(__FMA__)
static void microKernel(const double* __restrict ap, const double* __restrict bp, const size_t K,
                        double* __restrict c, const size_t ldc, const size_t mr, const size_t nr,
                        const bool first) {
    const bool edge = (mr != MR) || (nr != NR);
    // Edge tiles are staged through a padded buffer so that the inner loop
    // below can always work on a full MR x NR tile.
    double stage[MR][NR];
    double* cc = c;
    size_t ldcc = ldc;
    if (edge) {
        cc = &stage[0][0];
        ldcc = NR;
        if (!first) {
            for (size_t r = 0; r < mr; ++r) {
                for (size_t j = 0; j < nr; ++j) {
                    stage[r][j] = c[r * ldc + j];
                }
            }
        }
    }

    // 12 explicit accumulator registers (MR rows x NR/4 vectors).
#define MM_ACC_ZERO(r) __m256d c##r##0 = _mm256_setzero_pd(), c##r##1 = _mm256_setzero_pd()
#define MM_ACC_LOAD(r) __m256d c##r##0 = _mm256_loadu_pd(cc + (r) * ldcc), \
                                c##r##1 = _mm256_loadu_pd(cc + (r) * ldcc + 4)
#define MM_ACC_STORE(r) _mm256_storeu_pd(cc + (r) * ldcc, c##r##0), \
                        _mm256_storeu_pd(cc + (r) * ldcc + 4, c##r##1)
#define MM_FMA(r) { const __m256d av = _mm256_broadcast_sd(a + (r)); \
                    c##r##0 = _mm256_fmadd_pd(av, b0, c##r##0); \
                    c##r##1 = _mm256_fmadd_pd(av, b1, c##r##1); }

    if (first) {
        MM_ACC_ZERO(0); MM_ACC_ZERO(1); MM_ACC_ZERO(2);
        MM_ACC_ZERO(3); MM_ACC_ZERO(4); MM_ACC_ZERO(5);
        for (size_t k = 0; k < K; ++k) {
            const __m256d b0 = _mm256_load_pd(bp + k * NR);
            const __m256d b1 = _mm256_load_pd(bp + k * NR + 4);
            const double* __restrict a = ap + k * MR;
            MM_FMA(0); MM_FMA(1); MM_FMA(2); MM_FMA(3); MM_FMA(4); MM_FMA(5);
        }
        MM_ACC_STORE(0); MM_ACC_STORE(1); MM_ACC_STORE(2);
        MM_ACC_STORE(3); MM_ACC_STORE(4); MM_ACC_STORE(5);
    } else {
        MM_ACC_LOAD(0); MM_ACC_LOAD(1); MM_ACC_LOAD(2);
        MM_ACC_LOAD(3); MM_ACC_LOAD(4); MM_ACC_LOAD(5);
        for (size_t k = 0; k < K; ++k) {
            const __m256d b0 = _mm256_load_pd(bp + k * NR);
            const __m256d b1 = _mm256_load_pd(bp + k * NR + 4);
            const double* __restrict a = ap + k * MR;
            MM_FMA(0); MM_FMA(1); MM_FMA(2); MM_FMA(3); MM_FMA(4); MM_FMA(5);
        }
        MM_ACC_STORE(0); MM_ACC_STORE(1); MM_ACC_STORE(2);
        MM_ACC_STORE(3); MM_ACC_STORE(4); MM_ACC_STORE(5);
    }

#undef MM_ACC_ZERO
#undef MM_ACC_LOAD
#undef MM_ACC_STORE
#undef MM_FMA

    if (edge) {
        for (size_t r = 0; r < mr; ++r) {
            for (size_t j = 0; j < nr; ++j) {
                c[r * ldc + j] = stage[r][j];
            }
        }
    }
}
#else
static void microKernel(const double* __restrict ap, const double* __restrict bp, const size_t K,
                        double* __restrict c, const size_t ldc, const size_t mr, const size_t nr,
                        const bool first) {
    double acc[MR][NR] = {};
    if (!first) {
        for (size_t r = 0; r < mr; ++r) {
            for (size_t j = 0; j < nr; ++j) {
                acc[r][j] = c[r * ldc + j];
            }
        }
    }

    for (size_t k = 0; k < K; ++k) {
        const double* __restrict a = ap + k * MR;
        const double* __restrict b = bp + k * NR;
        for (size_t r = 0; r < MR; ++r) {
            for (size_t j = 0; j < NR; ++j) {
                acc[r][j] += a[r] * b[j];
            }
        }
    }

    for (size_t r = 0; r < mr; ++r) {
        for (size_t j = 0; j < nr; ++j) {
            c[r * ldc + j] = acc[r][j];
        }
    }
}
#endif

// Number of doubles of scratch space the packed operands need (including slack
// for 64 byte alignment).
size_t packedScratchSize(const size_t N) {
    const size_t nTiles = (N + MR - 1) / MR;
    const size_t nPanels = (N + NR - 1) / NR;
    return nTiles * N * MR + nPanels * N * NR + 2 * 8;
}

// The packed operands are carved out of the caller provided scratch space and
// aligned to 64 bytes, so that the micro-kernel's vector loads never straddle
// cache lines.
double* align64(double* p) {
    const uintptr_t v = reinterpret_cast<uintptr_t>(p);
    return reinterpret_cast<double*>((v + 63) & ~uintptr_t(63));
}

// Touch the scratch space with the same thread distribution that the packing
// loops use: its pages then are faulted in and NUMA local before the timed
// multiplication starts. Interleaving the tiles and panels over the threads
// (chunk size one) spreads the shared packed operands over all NUMA nodes.
void firstTouchScratch(std::vector<double>& scratch, const size_t N) {
    if (N == 0) {
        return;
    }
    resetFirstTouch(scratch.data(), scratch.size() * sizeof(double));

    const size_t nTiles = (N + MR - 1) / MR;
    const size_t nPanels = (N + NR - 1) / NR;
    double* const Ap = align64(scratch.data());
    double* const Bp = align64(Ap + nTiles * N * MR);

    #pragma omp parallel
    {
        #pragma omp for schedule(static, 1) nowait
        for (size_t t = 0; t < nTiles; ++t) {
            std::fill_n(Ap + t * N * MR, N * MR, 0.0);
        }
        #pragma omp for schedule(static, 1) nowait
        for (size_t p = 0; p < nPanels; ++p) {
            std::fill_n(Bp + p * N * NR, N * NR, 0.0);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, std::vector<double>& scratch) {
    if (N == 0) {
        return;
    }

    const size_t nTiles = (N + MR - 1) / MR;   // row tiles of A / C
    const size_t nPanels = (N + NR - 1) / NR;  // column panels of B / C
    const size_t KC = chooseKC(static_cast<size_t>(omp_get_max_threads()));
    // Number of row tiles grouped together so that their packed A block stays
    // in the private L2 cache while all column panels sweep over it.
    const size_t TB = std::max<size_t>(1, L2_A_BYTES / (MR * KC * sizeof(double)));

    // A is repacked row-tile-wise and B column-panel-wise below, so that the
    // micro-kernel streams both operands contiguously; edges are zero padded,
    // which leaves the values of the valid elements untouched.
    double* const Ap = align64(scratch.data());
    double* const Bp = align64(Ap + nTiles * N * MR);

    #pragma omp parallel
    {
        #pragma omp for schedule(static, 1) nowait
        for (size_t t = 0; t < nTiles; ++t) {
            double* __restrict dst = Ap + t * N * MR;
            const size_t mr = std::min(MR, N - t * MR);
            for (size_t k = 0; k < N; ++k) {
                for (size_t r = 0; r < MR; ++r) {
                    dst[k * MR + r] = (r < mr) ? A[(t * MR + r) * N + k] : 0.0;
                }
            }
        }

        #pragma omp for schedule(static, 1)
        for (size_t p = 0; p < nPanels; ++p) {
            double* __restrict dst = Bp + p * N * NR;
            const size_t nr = std::min(NR, N - p * NR);
            for (size_t k = 0; k < N; ++k) {
                for (size_t j = 0; j < NR; ++j) {
                    dst[k * NR + j] = (j < nr) ? B[k * N + p * NR + j] : 0.0;
                }
            }
        }

        // Two-dimensional decomposition of the C tile grid into rectangles of
        // row tiles times column panels. A rectangle is the unit of work: its C
        // elements are accumulated by a single thread over all k blocks in
        // ascending order, which needs no synchronization and reproduces the
        // reference summation order exactly. Rectangles maximize the cache
        // reuse of the packed operands (every A slice is reused by all panels of
        // the rectangle and vice versa), and handing out several of them per
        // thread dynamically balances the load.
        const size_t nThreads = static_cast<size_t>(omp_get_num_threads());
        const size_t nUnits = nThreads * UNITS_PER_THREAD;
        size_t gT = 1;
        for (size_t d = 1; d * d <= nUnits; ++d) {
            if (nUnits % d == 0) {
                gT = d;
            }
        }
        const size_t gP = nUnits / gT;

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (size_t gi = 0; gi < gT; ++gi) {
            for (size_t gj = 0; gj < gP; ++gj) {
                const size_t tBegin = (nTiles * gi) / gT;
                const size_t tEnd = (nTiles * (gi + 1)) / gT;
                const size_t pBegin = (nPanels * gj) / gP;
                const size_t pEnd = (nPanels * (gj + 1)) / gP;

                for (size_t kc = 0; kc < N; kc += KC) {
                    const size_t kb = std::min(KC, N - kc);
                    // Row tiles are additionally grouped so that the A block of
                    // a group stays in L2 while all column panels sweep over it.
                    for (size_t tb = tBegin; tb < tEnd; tb += TB) {
                        const size_t tbEnd = std::min(tb + TB, tEnd);
                        for (size_t p = pBegin; p < pEnd; ++p) {
                            const size_t nr = std::min(NR, N - p * NR);
                            for (size_t t = tb; t < tbEnd; ++t) {
                                microKernel(Ap + t * N * MR + kc * MR,
                                            Bp + p * N * NR + kc * NR, kb,
                                            &C[(t * MR) * N + p * NR], N,
                                            std::min(MR, N - t * MR), nr, kc == 0);
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
    
    // Thread placement is fixed up front: the NUMA first touch of the matrices
    // must match the threads that later compute on them.
    pinThreads();

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    firstTouchMatrix(C, N);

    // Scratch space for the packed operands. Like the matrices themselves it is
    // allocated outside of the timed region; its pages are faulted in NUMA
    // locally by the packing loops.
    std::vector<double> scratch(packedScratchSize(N));
    firstTouchScratch(scratch, N);

    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N, scratch);
    
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
