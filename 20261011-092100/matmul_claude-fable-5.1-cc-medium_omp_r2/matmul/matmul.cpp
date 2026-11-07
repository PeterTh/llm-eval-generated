#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    double* const m = mat.data();
    // Parallel initialization: deterministic (each element depends only on its
    // indices) and provides NUMA-friendly first-touch placement of the pages.
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// OpenMP matrix multiplication
//
// Semantics: every C[i][j] is computed as  ((0.0 + A[i][0]*B[0][j]) + A[i][1]*B[1][j]) + ...
// i.e. the k-loop is always traversed in increasing order, from 0.0, exactly
// as in the reference triple loop. Blocking over k merely splits that ordered
// sum into consecutive segments, so the sequence of floating-point operations
// per element is identical to the sequential version.
//
// Structure (Goto/BLIS style, with OpenMP over 2D tiles of C):
//   * B is packed once (in parallel) into column panels of NR contiguous
//     doubles per k. This gives unit-stride, aliasing-free access to B in the
//     micro-kernel (the natural row-major stride is a multiple of 4 KiB for
//     power-of-two N, which thrashes L1 sets).
//   * C is partitioned into 2D tiles (bi x bj) that are distributed statically
//     over the threads in column-major tile order, so that threads that are
//     neighbours (same L3) work on the same B panel at the same time. The tile
//     size is chosen from N and the thread count with a small cost model that
//     balances load balance against memory traffic (which is ~ 1/bi + 1/bj).
//   * Each tile is processed in k-blocks of BK. The A block (bi x BK) is packed
//     into a thread-private buffer in MR-row panels so that both operands are
//     read with unit stride from L1/L2 inside the micro-kernel.
//   * The micro-kernel computes an MR x NR tile of C with the accumulators
//     held in SIMD registers (MR*NR/4 256-bit vectors) using GCC/Clang
//     vector extensions (no external dependencies).
// ---------------------------------------------------------------------------
namespace {

constexpr size_t VL = 4;            // doubles per 256-bit vector
constexpr size_t MR = 6;            // rows of C per micro-kernel
constexpr size_t NR = 8;            // columns of C per micro-kernel (2 vectors)
constexpr size_t NV = NR / VL;      // vectors per row
constexpr size_t BK = 256;          // k extent of a cache block
constexpr size_t MAX_BI = 192;      // max rows per tile (multiple of MR)
constexpr size_t MAX_BJ = 256;      // max cols per tile (multiple of NR)

typedef double v4d __attribute__((vector_size(VL * sizeof(double))));

inline v4d loadv(const double* p) noexcept {
    v4d v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
inline void storev(double* p, const v4d v) noexcept { __builtin_memcpy(p, &v, sizeof(v)); }

// C[0..R)[0..ncols) += Ap[0..kc)[0..R) * Bp[0..kc)[0..NR)
//   ap  : packed A panel, kc consecutive groups of MR doubles (rows 0..R-1 used)
//   bp  : packed B panel, kc consecutive groups of NR doubles
//   c   : C tile, stride ldc
//   ncols: valid columns (<= NR); columns beyond ncols are not stored
template <size_t R>
inline void microKernel(const double* __restrict ap, const double* __restrict bp,
                        double* __restrict c, const size_t ldc,
                        const size_t kc, const size_t ncols) noexcept {
    v4d acc[R][NV];
    if (ncols == NR) {
        for (size_t r = 0; r < R; ++r)
            for (size_t v = 0; v < NV; ++v) acc[r][v] = loadv(c + r * ldc + v * VL);
    } else {
        for (size_t r = 0; r < R; ++r) {
            double tmp[NR] = {};
            for (size_t j = 0; j < ncols; ++j) tmp[j] = c[r * ldc + j];
            for (size_t v = 0; v < NV; ++v) acc[r][v] = loadv(tmp + v * VL);
        }
    }

    for (size_t k = 0; k < kc; ++k) {
        v4d b[NV];
        for (size_t v = 0; v < NV; ++v) b[v] = loadv(bp + k * NR + v * VL);
        for (size_t r = 0; r < R; ++r) {
            const double s = ap[k * MR + r];
            const v4d av = {s, s, s, s};
            for (size_t v = 0; v < NV; ++v) acc[r][v] += av * b[v];
        }
    }

    if (ncols == NR) {
        for (size_t r = 0; r < R; ++r)
            for (size_t v = 0; v < NV; ++v) storev(c + r * ldc + v * VL, acc[r][v]);
    } else {
        for (size_t r = 0; r < R; ++r) {
            double tmp[NR];
            for (size_t v = 0; v < NV; ++v) storev(tmp + v * VL, acc[r][v]);
            for (size_t j = 0; j < ncols; ++j) c[r * ldc + j] = tmp[j];
        }
    }
}

inline void runKernel(const size_t rows, const double* ap, const double* bp,
                      double* c, const size_t ldc, const size_t kc, const size_t ncols) noexcept {
    switch (rows) {
        case 6: microKernel<6>(ap, bp, c, ldc, kc, ncols); break;
        case 5: microKernel<5>(ap, bp, c, ldc, kc, ncols); break;
        case 4: microKernel<4>(ap, bp, c, ldc, kc, ncols); break;
        case 3: microKernel<3>(ap, bp, c, ldc, kc, ncols); break;
        case 2: microKernel<2>(ap, bp, c, ldc, kc, ncols); break;
        case 1: microKernel<1>(ap, bp, c, ldc, kc, ncols); break;
        default: break;
    }
}

// Pack A[0..mc)[0..kc) (row stride lda) into MR-row panels: [mc/MR][kc][MR].
// Rows beyond mc in the last panel are zero filled (never read into C).
inline void packA(const double* __restrict a, const size_t lda, double* __restrict ap,
                  const size_t mc, const size_t kc) noexcept {
    for (size_t i0 = 0; i0 < mc; i0 += MR) {
        const size_t rows = std::min(MR, mc - i0);
        double* dst = ap + i0 * kc;
        if (rows == MR) {
            for (size_t k = 0; k < kc; ++k, dst += MR)
                for (size_t r = 0; r < MR; ++r) dst[r] = a[(i0 + r) * lda + k];
        } else {
            for (size_t k = 0; k < kc; ++k, dst += MR) {
                size_t r = 0;
                for (; r < rows; ++r) dst[r] = a[(i0 + r) * lda + k];
                for (; r < MR; ++r) dst[r] = 0.0;
            }
        }
    }
}

// Choose the tile size (bi x bj) for a given N and thread count.
// Cost model per thread: rounds * (compute + memory traffic), where
// rounds = ceil(tiles / nthreads), compute ~ bi*bj and traffic ~ (bi + bj)
// scaled by an empirical flops-per-byte balance factor.
inline void chooseTileSize(const size_t N, const size_t nthreads, size_t& bi, size_t& bj) noexcept {
    constexpr double kTraffic = 48.0;
    double bestCost = -1.0;
    bi = MAX_BI;
    bj = MAX_BJ;
    for (size_t cbi = MAX_BI; cbi >= MR; cbi = (cbi > MR) ? std::max(MR, (cbi / 2 + MR - 1) / MR * MR) : 0) {
        for (size_t cbj = MAX_BJ; cbj >= NR; cbj /= 2) {
            const size_t tiles = ((N + cbi - 1) / cbi) * ((N + cbj - 1) / cbj);
            const size_t rounds = (tiles + nthreads - 1) / nthreads;
            const double cost = static_cast<double>(rounds) *
                (static_cast<double>(cbi) * static_cast<double>(cbj) +
                 kTraffic * static_cast<double>(cbi + cbj));
            if (bestCost < 0.0 || cost < bestCost) {
                bestCost = cost;
                bi = cbi;
                bj = cbj;
            }
            if (cbj == NR) break;
        }
        if (cbi == MR) break;
    }
}

}  // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;

    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double* __restrict c = C.data();

    const size_t numPanels = (N + NR - 1) / NR;
    // Packed B: [panel][k][NR]; the last panel is zero padded.
    // Deliberately not value-initialised: the pages are first touched by the
    // parallel packing loop (fast, and NUMA-local to the packing thread).
    std::unique_ptr<double[]> packedB(new double[numPanels * N * NR]);
    double* __restrict bp = packedB.get();

    // Tile size is a pure performance heuristic, so using the expected team
    // size here (instead of omp_get_num_threads() inside the region) is safe.
    size_t bi = MAX_BI, bj = MAX_BJ;
    chooseTileSize(N, static_cast<size_t>(std::max(1, omp_get_max_threads())), bi, bj);

#pragma omp parallel
    {
        // Pack B into column panels. Work is split over (panel, k-block) pairs.
        const size_t numKBlocks = (N + BK - 1) / BK;
#pragma omp for collapse(2) schedule(static)
        for (size_t p = 0; p < numPanels; ++p) {
            for (size_t kb = 0; kb < numKBlocks; ++kb) {
                const size_t j0 = p * NR;
                const size_t nc = std::min(NR, N - j0);
                const size_t k0 = kb * BK;
                const size_t k1 = std::min(N, k0 + BK);
                double* dst = bp + (p * N + k0) * NR;
                for (size_t k = k0; k < k1; ++k, dst += NR) {
                    const double* src = b + k * N + j0;
                    size_t j = 0;
                    for (; j < nc; ++j) dst[j] = src[j];
                    for (; j < NR; ++j) dst[j] = 0.0;
                }
            }
        }
        // (implicit barrier after the packing loop)

        // Thread-private packed A block (MAX_BI x BK doubles). Thread-local
        // static storage: no per-thread heap allocation inside the parallel
        // region (mmap/munmap serialise badly across >100 threads).
        static thread_local double packedA[MAX_BI * BK] __attribute__((aligned(64)));
        double* __restrict ap = packedA;

        const size_t numRowTiles = (N + bi - 1) / bi;
        const size_t numColTiles = (N + bj - 1) / bj;
        const size_t numTiles = numRowTiles * numColTiles;

        // Column-major tile order + static schedule: consecutive threads work
        // on the same B panel (shared through L3) and on different A blocks.
#pragma omp for schedule(static)
        for (size_t t = 0; t < numTiles; ++t) {
            const size_t ti = t % numRowTiles;
            const size_t tj = t / numRowTiles;
            const size_t i0 = ti * bi;
            const size_t mc = std::min(bi, N - i0);
            const size_t j0 = tj * bj;
            const size_t nc = std::min(bj, N - j0);
            const size_t p0 = j0 / NR;
            const size_t np = (nc + NR - 1) / NR;

            // Start from 0.0 exactly like the reference "double sum = 0.0".
            for (size_t i = i0; i < i0 + mc; ++i)
                for (size_t j = j0; j < j0 + nc; ++j) c[i * N + j] = 0.0;

            for (size_t k0 = 0; k0 < N; k0 += BK) {
                const size_t kc = std::min(BK, N - k0);
                packA(a + i0 * N + k0, N, ap, mc, kc);
                for (size_t p = 0; p < np; ++p) {
                    const size_t jj = j0 + p * NR;
                    const size_t ncols = std::min(NR, N - jj);
                    const double* panel = bp + ((p0 + p) * N + k0) * NR;
                    for (size_t i = 0; i < mc; i += MR) {
                        const size_t rows = std::min(MR, mc - i);
                        runKernel(rows, ap + i * kc, panel, c + (i0 + i) * N + jj, N, kc, ncols);
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

// ---------------------------------------------------------------------------
// Default thread placement.
//
// The compute kernel is bound by per-core SIMD throughput and L2 capacity, so
// the best configuration is one thread per physical core, pinned. If the user
// has not configured the OpenMP runtime (OMP_NUM_THREADS / OMP_PROC_BIND /
// OMP_PLACES), we select that configuration ourselves from the CPUs the
// process is allowed to run on (respecting taskset/cgroups). Any explicit user
// setting takes precedence and disables this logic.
// ---------------------------------------------------------------------------
#ifdef __linux__
// Reads e.g. "/sys/devices/system/cpu/cpu5/topology/core_id"; returns -1 on failure.
static long readCpuTopologyValue(const int cpu, const char* leaf) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, leaf);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

// One allowed CPU per physical core (falls back to all allowed CPUs if the
// topology cannot be read). Empty on failure.
static std::vector<int> selectOneCpuPerCore() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return {};

    std::vector<int> cpus;
    std::vector<std::pair<long, long>> seenCores;  // (package id, core id)
    bool topologyOk = true;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        const long pkg = readCpuTopologyValue(cpu, "physical_package_id");
        const long core = readCpuTopologyValue(cpu, "core_id");
        if (pkg < 0 || core < 0) { topologyOk = false; cpus.push_back(cpu); continue; }
        const std::pair<long, long> key(pkg, core);
        if (std::find(seenCores.begin(), seenCores.end(), key) != seenCores.end()) continue;
        seenCores.push_back(key);
        cpus.push_back(cpu);
    }
    if (!topologyOk) {
        // Mixed/unknown topology: use every allowed CPU instead of guessing.
        cpus.clear();
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    }
    return cpus;
}
#endif

static void configureThreads() {
#ifdef __linux__
    if (getenv("OMP_NUM_THREADS") || getenv("OMP_PROC_BIND") || getenv("OMP_PLACES")) return;
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    const std::vector<int> cpus = selectOneCpuPerCore();
    if (cpus.empty()) return;

    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(cpus.size()));

    // Bind each thread of the (persistent) OpenMP worker pool to its core.
#pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpus[tid % cpus.size()], &one);
        pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
    }
#endif
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
    
    configureThreads();

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
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
