#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
// nvcc's front-end cannot parse GCC's AMX intrinsic headers (unused here), so
// suppress them before pulling in <immintrin.h> for the AVX2/FMA host kernel.
#define _AMXTILEINTRIN_H_INCLUDED
#define _AMXINT8INTRIN_H_INCLUDED
#define _AMXBF16INTRIN_H_INCLUDED
#define _AMXFP16INTRIN_H_INCLUDED
#define _AMXCOMPLEXINTRIN_H_INCLUDED
#endif
#include <immintrin.h>

#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA matrix multiplication.
//
// Decomposition:
//   * MPI      : C is split into contiguous row blocks, one per rank.  A and B
//                are generated redundantly on every rank from the closed-form
//                pseudo-random formula, so no input communication is required.
//                Only the resulting C row blocks are gathered on rank 0.
//   * CUDA     : each rank drives one GPU which computes the upper part of the
//                rank's row block with a register-tiled DGEMM kernel.  A and B
//                are generated directly in device memory, the result is copied
//                back asynchronously while the CPU keeps working.
//   * OpenMP   : all cores of the rank compute the remaining rows with a
//                cache-blocked, packed, AVX2/FMA micro-kernel; one thread per
//                physical core, pinned, and the node's cores split between the
//                ranks that share them.
//
// The GPU/CPU work split is auto-tuned before the timed region: short trial runs
// of both engines (running concurrently, as in the real pass) yield their row
// rates, from which the split that lets them finish simultaneously is computed.
//
// Environment variables: MATMUL_GPU_FRAC overrides the auto-tuned split,
// MATMUL_DEBUG prints per-rank timings to stderr.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,        \
                    __LINE__);                                                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                          \
        }                                                                                          \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                                       const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// ---------------------------------------------------------------------------
// CPU side: blocked + packed DGEMM
// ---------------------------------------------------------------------------

// Micro-kernel dimensions (4 rows x 12 columns == 12 AVX2 accumulators)
constexpr size_t MR = 4;
constexpr size_t NR = 12;
// Cache blocking parameters
constexpr size_t KC = 512;  // k panel (packed B fits into L2)
constexpr size_t NC = 1008; // j panel, multiple of NR
constexpr size_t MC_MAX = 128;

static inline size_t roundUp(const size_t v, const size_t m) { return ((v + m - 1) / m) * m; }

// Pack row k of the B block [k0..k0+kc) x [j0..j0+nc) into the NR-wide panels.
// B is read strictly sequentially, which keeps the hardware prefetchers and the
// TLB happy; the scattered writes go into the (L2 resident) packing buffer.
static inline void packBRow(const double* B, const size_t N, const size_t k0, const size_t kc,
                            const size_t j0, const size_t nc, const size_t k, double* Bp) {
    const double* src = &B[(k0 + k) * N + j0];
    const size_t nfull = (std::min(nc, N - j0) / NR) * NR;
    size_t jp = 0;
    for (; jp < nfull; jp += NR) {
        double* dst = Bp + jp * kc + k * NR;
        for (size_t j = 0; j < NR; ++j) dst[j] = src[jp + j];
    }
    for (; jp < nc; jp += NR) { // tail panel, zero padded
        double* dst = Bp + jp * kc + k * NR;
        const size_t jvalid = (j0 + jp < N) ? std::min(NR, N - j0 - jp) : 0;
        size_t j = 0;
        for (; j < jvalid; ++j) dst[j] = src[jp + j];
        for (; j < NR; ++j) dst[j] = 0.0;
    }
}

// Pack an MC x KC block of A into MR-row panels
static inline void packABlock(const double* A, const size_t N, const size_t i0, const size_t mc,
                              const size_t k0, const size_t kc, double* dst) {
    for (size_t ip = 0; ip < mc; ip += MR) {
        double* p = dst + ip * kc;
        const size_t ivalid = std::min(MR, mc - ip);
        for (size_t k = 0; k < kc; ++k) {
            size_t i = 0;
            for (; i < ivalid; ++i) p[k * MR + i] = A[(i0 + ip + i) * N + k0 + k];
            for (; i < MR; ++i) p[k * MR + i] = 0.0;
        }
    }
}

// 4x12 AVX2/FMA micro-kernel, C is accessed with row stride ldc
static inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                               double* __restrict__ C, const size_t ldc, const size_t kc,
                               const bool first) {
    __m256d c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11;
    if (first) {
        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = _mm256_setzero_pd();
    } else {
        c0 = _mm256_loadu_pd(C + 0);
        c1 = _mm256_loadu_pd(C + 4);
        c2 = _mm256_loadu_pd(C + 8);
        c3 = _mm256_loadu_pd(C + ldc + 0);
        c4 = _mm256_loadu_pd(C + ldc + 4);
        c5 = _mm256_loadu_pd(C + ldc + 8);
        c6 = _mm256_loadu_pd(C + 2 * ldc + 0);
        c7 = _mm256_loadu_pd(C + 2 * ldc + 4);
        c8 = _mm256_loadu_pd(C + 2 * ldc + 8);
        c9 = _mm256_loadu_pd(C + 3 * ldc + 0);
        c10 = _mm256_loadu_pd(C + 3 * ldc + 4);
        c11 = _mm256_loadu_pd(C + 3 * ldc + 8);
    }

    for (size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(Bp + k * NR + 0);
        const __m256d b1 = _mm256_loadu_pd(Bp + k * NR + 4);
        const __m256d b2 = _mm256_loadu_pd(Bp + k * NR + 8);

        __m256d a = _mm256_broadcast_sd(Ap + k * MR + 0);
        c0 = _mm256_fmadd_pd(a, b0, c0);
        c1 = _mm256_fmadd_pd(a, b1, c1);
        c2 = _mm256_fmadd_pd(a, b2, c2);
        a = _mm256_broadcast_sd(Ap + k * MR + 1);
        c3 = _mm256_fmadd_pd(a, b0, c3);
        c4 = _mm256_fmadd_pd(a, b1, c4);
        c5 = _mm256_fmadd_pd(a, b2, c5);
        a = _mm256_broadcast_sd(Ap + k * MR + 2);
        c6 = _mm256_fmadd_pd(a, b0, c6);
        c7 = _mm256_fmadd_pd(a, b1, c7);
        c8 = _mm256_fmadd_pd(a, b2, c8);
        a = _mm256_broadcast_sd(Ap + k * MR + 3);
        c9 = _mm256_fmadd_pd(a, b0, c9);
        c10 = _mm256_fmadd_pd(a, b1, c10);
        c11 = _mm256_fmadd_pd(a, b2, c11);
    }

    _mm256_storeu_pd(C + 0, c0);
    _mm256_storeu_pd(C + 4, c1);
    _mm256_storeu_pd(C + 8, c2);
    _mm256_storeu_pd(C + ldc + 0, c3);
    _mm256_storeu_pd(C + ldc + 4, c4);
    _mm256_storeu_pd(C + ldc + 8, c5);
    _mm256_storeu_pd(C + 2 * ldc + 0, c6);
    _mm256_storeu_pd(C + 2 * ldc + 4, c7);
    _mm256_storeu_pd(C + 2 * ldc + 8, c8);
    _mm256_storeu_pd(C + 3 * ldc + 0, c9);
    _mm256_storeu_pd(C + 3 * ldc + 4, c10);
    _mm256_storeu_pd(C + 3 * ldc + 8, c11);
}

// Scratch buffers of the OpenMP team (allocated once, reused for every wave)
struct CpuWorkspace {
    std::vector<double*> Ap; // one packed A block per thread
    double* Bp = nullptr;    // packed B panel, shared by the whole team

    ~CpuWorkspace() {
        for (double* p : Ap) free(p);
        free(Bp);
    }
};

// Computes C[i0..i1) = A[i0..i1) * B.  Called by one thread, spawns the team.
// C points at the local row block (row i0 of the global matrix is row 0 of Cloc).
// Loop order is the classic jc -> kc -> ic blocking: the packed B panel is built
// once per (jc, kc) tile by the whole team and then reused by every row block,
// while each thread keeps its own packed A block in L2.
static void cpuGemmRows(const double* A, const double* B, double* Cloc, const size_t N,
                        const size_t i0, const size_t i1, const size_t rowOffset,
                        CpuWorkspace& ws) {
    if (i1 <= i0) return;
    const size_t rows = i1 - i0;
    // Row block size: at least one block per thread, but small enough to keep the
    // packed A block resident in L2.
    const size_t nthreads = static_cast<size_t>(omp_get_max_threads());
    const size_t mc =
        std::min(MC_MAX, roundUp(std::max<size_t>((rows + nthreads - 1) / nthreads, MR), MR));

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* const Ap = ws.Ap[tid];
        double* const Bp = ws.Bp;
        const size_t nblocks = (rows + mc - 1) / mc;
        double edge[MR * NR];

        for (size_t jc = 0; jc < N; jc += NC) {
            const size_t nc = std::min(NC, N - jc);
            const size_t npanels = (nc + NR - 1) / NR;
            for (size_t k0 = 0; k0 < N; k0 += KC) {
                const size_t kc = std::min(KC, N - k0);

                // Cooperatively pack the B panel once for the whole team
#pragma omp for schedule(static)
                for (size_t k = 0; k < kc; ++k) {
                    packBRow(B, N, k0, kc, jc, nc, k, Bp);
                }

#pragma omp for schedule(static)
                for (size_t b = 0; b < nblocks; ++b) {
                    const size_t ib = b * mc;
                    const size_t mcur = std::min(mc, rows - ib);
                    packABlock(A, N, i0 + ib, mcur, k0, kc, Ap);

                    for (size_t p = 0; p < npanels; ++p) {
                        const size_t jr = p * NR;
                        const size_t jvalid = std::min(NR, nc - jr);
                        for (size_t ir = 0; ir < mcur; ir += MR) {
                            const size_t ivalid = std::min(MR, mcur - ir);
                            double* Cblk = &Cloc[(i0 - rowOffset + ib + ir) * N + jc + jr];
                            if (ivalid == MR && jvalid == NR) {
                                microKernel(Ap + ir * kc, Bp + p * NR * kc, Cblk, N, kc, k0 == 0);
                            } else {
                                if (k0 != 0) {
                                    for (size_t x = 0; x < ivalid; ++x)
                                        for (size_t y = 0; y < jvalid; ++y)
                                            edge[x * NR + y] = Cblk[x * N + y];
                                }
                                microKernel(Ap + ir * kc, Bp + p * NR * kc, edge, NR, kc, k0 == 0);
                                for (size_t x = 0; x < ivalid; ++x)
                                    for (size_t y = 0; y < jvalid; ++y)
                                        Cblk[x * N + y] = edge[x * NR + y];
                            }
                        }
                    }
                }
                // implicit barrier: Bp may only be repacked once everybody is done
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------

constexpr int GPU_TILE = 64;  // block tile (rows and columns of C)
constexpr int GPU_KTILE = 16; // depth of one shared-memory step

// C[64x64] tile per block, 16x16 threads, 4x4 accumulators per thread.
__global__ __launch_bounds__(256) void dgemmKernel(const double* __restrict__ A,
                                                   const double* __restrict__ B,
                                                   double* __restrict__ C, const int K,
                                                   const int lda, const int ldb, const int ldc) {
    __shared__ double As[GPU_KTILE][GPU_TILE + 1];
    __shared__ double Bs[GPU_KTILE][GPU_TILE];

    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const size_t r0 = static_cast<size_t>(blockIdx.y) * GPU_TILE;
    const size_t c0 = static_cast<size_t>(blockIdx.x) * GPU_TILE;

    double acc[4][4] = {};

    for (int t = 0; t < K; t += GPU_KTILE) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int rr = ty * 4 + i;
            As[tx][rr] = A[(r0 + rr) * lda + t + tx];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            Bs[ty][tx + 16 * i] = B[(t + ty) * static_cast<size_t>(ldb) + c0 + tx + 16 * i];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < GPU_KTILE; ++k) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[k][ty * 4 + i];
                b[i] = Bs[k][tx * 4 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            C[(r0 + ty * 4 + i) * static_cast<size_t>(ldc) + c0 + tx * 4 + j] = acc[i][j];
}

// Fill the padded device copies of A (rank's row block) and B directly on the GPU
__global__ void initDeviceA(double* A, const size_t N, const size_t rowOffset, const size_t rows,
                            const int ld) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (j >= static_cast<size_t>(ld)) return;
    A[i * ld + j] = (i < rows && j < N) ? getPseudoRndValue(N, rowOffset + i, j) : 0.0;
}

__global__ void initDeviceB(double* B, const size_t N, const int ld) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (j >= static_cast<size_t>(ld)) return;
    B[i * ld + j] = (i < N && j < N) ? getPseudoRndValue(N, i, j) : 0.0;
}

struct GpuContext {
    bool available = false;
    int device = -1;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    int ldk = 0; // padded k / lda
    int ldn = 0; // padded n / ldb / ldc
    size_t mpad = 0;
    cudaStream_t stream[2] = {nullptr, nullptr};
    cudaEvent_t evStart = nullptr;
    cudaEvent_t evEnd[2] = {nullptr, nullptr};
    void* pinBase = nullptr; // page aligned start of the pinned host block
};

// Enqueue the GEMM for local rows [g0, g1) plus the asynchronous copy back.
// A completion event is recorded on every stream that received work, so the
// device time can be measured without being distorted by the host's progress.
// Returns a bit mask of the streams that received work.
static int gpuLaunch(GpuContext& gpu, double* Cloc, const size_t N, const size_t g0,
                     const size_t g1, const bool copyAsync) {
    const size_t rows = g1 - g0;
    if (rows == 0) return 0;

    int used = 0;

    constexpr size_t CHUNKS = 4;
    const size_t chunk = roundUp((rows + CHUNKS - 1) / CHUNKS, GPU_TILE);
    int s = 0;
    for (size_t off = 0; off < rows; off += chunk, s ^= 1) {
        used |= 1 << s;
        const size_t crows = std::min(chunk, rows - off);
        const size_t base = g0 + off;
        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned>(gpu.ldn / GPU_TILE),
                  static_cast<unsigned>(roundUp(crows, GPU_TILE) / GPU_TILE));
        dgemmKernel<<<grid, block, 0, gpu.stream[s]>>>(gpu.dA + base * gpu.ldk, gpu.dB,
                                                       gpu.dC + base * gpu.ldn, gpu.ldk, gpu.ldk,
                                                       gpu.ldn, gpu.ldn);
        if (copyAsync) {
            CUDA_CHECK(cudaMemcpy2DAsync(Cloc + base * N, N * sizeof(double),
                                         gpu.dC + base * gpu.ldn, gpu.ldn * sizeof(double),
                                         N * sizeof(double), crows, cudaMemcpyDeviceToHost,
                                         gpu.stream[s]));
        }
    }
    for (int st = 0; st < 2; ++st)
        if (used & (1 << st)) CUDA_CHECK(cudaEventRecord(gpu.evEnd[st], gpu.stream[st]));
    return used;
}

// Device time of the work enqueued by gpuLaunch (streams given by `used`)
static double gpuElapsed(GpuContext& gpu, const int used) {
    double t = 0.0;
    for (int st = 0; st < 2; ++st) {
        if (!(used & (1 << st))) continue;
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, gpu.evStart, gpu.evEnd[st]));
        t = std::max(t, static_cast<double>(ms) / 1e3);
    }
    return t;
}

// ---------------------------------------------------------------------------
// Original (reference) helpers
// ---------------------------------------------------------------------------

void initMatrix(double* mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const double* A, const double* B, const double* C, const size_t N) {
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

// ---------------------------------------------------------------------------
// Runtime configuration helpers
// ---------------------------------------------------------------------------

// Sibling hardware threads of a logical CPU (SMT), read from sysfs
static std::vector<int> cpuSiblings(const int cpu) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    std::vector<int> sib;
    FILE* f = fopen(path, "r");
    if (f) {
        char buf[512] = {0};
        if (fgets(buf, sizeof(buf), f)) {
            const char* p = buf;
            while (*p) {
                if (*p >= '0' && *p <= '9') {
                    long a = strtol(p, const_cast<char**>(&p), 10);
                    long b = a;
                    if (*p == '-') b = strtol(p + 1, const_cast<char**>(&p), 10);
                    for (long c = a; c <= b; ++c) sib.push_back(static_cast<int>(c));
                } else {
                    ++p;
                }
            }
        }
        fclose(f);
    }
    if (sib.empty()) sib.push_back(cpu);
    return sib;
}

// Physical cores this rank should use.  The CPUs the process may run on are
// split evenly between the ranks sharing them, so that ranks never oversubscribe
// the same cores.  Launchers frequently bind a rank to a single core, which is
// useless for a hybrid code: if the restriction is a mere affinity mask (and not
// a cgroup, which cannot be widened) the rank reclaims its share of the node.
static std::vector<std::vector<int>> myCores(MPI_Comm shmComm, const int localRank,
                                             const int localSize) {
    const long nodeLogical = std::max(1L, sysconf(_SC_NPROCESSORS_ONLN));

    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
        for (long c = 0; c < nodeLogical && c < CPU_SETSIZE; ++c) CPU_SET(c, &mask);

    bool wholeNode = CPU_COUNT(&mask) >= nodeLogical;
    if (!wholeNode) {
        cpu_set_t full;
        CPU_ZERO(&full);
        for (long c = 0; c < nodeLogical && c < CPU_SETSIZE; ++c) CPU_SET(c, &full);
        if (sched_setaffinity(0, sizeof(full), &full) == 0) {
            mask = full;
            wholeNode = true;
        }
    }

    int groupSize = localSize, groupIdx = localRank;
    if (!wholeNode) {
        // group the ranks that share an identical mask (the launcher has already
        // split the node between them)
        const int maskBytes = static_cast<int>(sizeof(cpu_set_t));
        std::vector<char> all(static_cast<size_t>(maskBytes) * localSize);
        MPI_Allgather(&mask, maskBytes, MPI_BYTE, all.data(), maskBytes, MPI_BYTE, shmComm);
        groupSize = 0;
        groupIdx = 0;
        for (int r = 0; r < localSize; ++r) {
            if (memcmp(all.data() + static_cast<size_t>(r) * maskBytes, &mask, maskBytes) == 0) {
                if (r < localRank) ++groupIdx;
                ++groupSize;
            }
        }
        if (groupSize < 1) groupSize = 1;
    }

    // one entry per physical core (represented by its sibling list)
    std::vector<std::vector<int>> cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) continue;
        std::vector<int> sib = cpuSiblings(cpu);
        if (sib[0] != cpu) continue; // not the representative of this core
        cores.push_back(sib);
    }
    if (cores.empty()) return cores;

    const size_t per = cores.size() / static_cast<size_t>(groupSize);
    if (per == 0) return {}; // more ranks than cores: leave placement to the OS
    const size_t begin = per * static_cast<size_t>(groupIdx);
    return std::vector<std::vector<int>>(cores.begin() + begin, cores.begin() + begin + per);
}

// Pin every OpenMP thread of the team to its own physical core
static void pinThreads(const std::vector<std::vector<int>>& cores) {
    if (cores.empty()) return;
#pragma omp parallel
    {
        const size_t t = static_cast<size_t>(omp_get_thread_num()) % cores.size();
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int cpu : cores[t]) CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_size(shmComm, &localSize);

    // One OpenMP thread per physical core owned by this rank, pinned
    const std::vector<std::vector<int>> cores = myCores(shmComm, localRank, localSize);
    if (!cores.empty() && !getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(static_cast<int>(cores.size()));
    pinThreads(cores);
    {   // warn if the launcher confined this rank to fewer cores than its fair share
        long nodeLogical = std::max(1L, sysconf(_SC_NPROCESSORS_ONLN));
        const size_t fair = static_cast<size_t>(nodeLogical) / std::max(1, localSize);
        if (rank == 0 && !cores.empty() && cores.size() * 4 < fair)
            fprintf(stderr,
                    "Warning: only %zu cores available per rank; run with '--bind-to none' or "
                    "'--bind-to socket' to use the whole node\n",
                    cores.size());
    }

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Comm_free(&shmComm);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Comm_free(&shmComm);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", nranks, omp_get_max_threads());
    }

    // ---- Row block distribution ------------------------------------------
    // Ranks are ordered by (node, rank within node) so that the rows owned by
    // one node form a contiguous range: the node's ranks can then write their
    // results straight into one shared-memory buffer and only the node leaders
    // take part in the final gather.
    int leaderRank = rank;
    MPI_Bcast(&leaderRank, 1, MPI_INT, 0, shmComm);
    std::vector<int> allLeaders(nranks), allLocalRanks(nranks);
    MPI_Allgather(&leaderRank, 1, MPI_INT, allLeaders.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&localRank, 1, MPI_INT, allLocalRanks.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> nodeLeaders; // distinct leaders, in ascending rank order
    for (int r = 0; r < nranks; ++r)
        if (allLeaders[r] == r) nodeLeaders.push_back(r);
    const int nnodes = static_cast<int>(nodeLeaders.size());
    auto nodeIndexOf = [&](const int r) {
        for (int n = 0; n < nnodes; ++n)
            if (nodeLeaders[n] == allLeaders[r]) return n;
        return 0;
    };

    std::vector<int> order; // ranks sorted by (node, local rank)
    order.reserve(nranks);
    for (int n = 0; n < nnodes; ++n)
        for (int lr = 0; lr < nranks; ++lr)
            for (int r = 0; r < nranks; ++r)
                if (nodeIndexOf(r) == n && allLocalRanks[r] == lr) order.push_back(r);

    std::vector<int> rowCounts(nranks), rowOffsets(nranks);
    {
        const size_t base = N / nranks, rem = N % nranks;
        size_t off = 0;
        for (int p = 0; p < nranks; ++p) {
            const int r = order[p];
            const size_t cnt = base + (static_cast<size_t>(p) < rem ? 1 : 0);
            rowCounts[r] = static_cast<int>(cnt);
            rowOffsets[r] = static_cast<int>(off);
            off += cnt;
        }
    }
    const size_t myRows = static_cast<size_t>(rowCounts[rank]);
    const size_t myOffset = static_cast<size_t>(rowOffsets[rank]);

    // rows owned by each node (contiguous by construction)
    std::vector<int> nodeRowCounts(nnodes, 0), nodeRowOffsets(nnodes, 0);
    for (int n = 0; n < nnodes; ++n) nodeRowOffsets[n] = static_cast<int>(N);
    for (int r = 0; r < nranks; ++r) {
        const int n = nodeIndexOf(r);
        nodeRowCounts[n] += rowCounts[r];
        nodeRowOffsets[n] = std::min(nodeRowOffsets[n], rowOffsets[r]);
    }
    const int myNode = nodeIndexOf(rank);
    const size_t nodeOffset = static_cast<size_t>(nodeRowOffsets[myNode]);

    // ---- Allocation (raw buffers, NUMA friendly first touch) --------------
    constexpr size_t HUGE_PAGE = 2u << 20;
    auto allocate = [](size_t elems) {
        const size_t bytes = roundUp(elems * sizeof(double) + HUGE_PAGE, HUGE_PAGE);
        void* p = aligned_alloc(HUGE_PAGE, bytes);
        if (!p) {
            fprintf(stderr, "allocation of %zu bytes failed\n", bytes);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        madvise(p, bytes, MADV_HUGEPAGE); // large pages cut TLB misses during packing
        return static_cast<double*>(p);
    };

    double* A = allocate(N * N);
    double* B = allocate(N * N);
    // C lives in one shared-memory segment per node: every rank of the node
    // computes directly into it, so collecting the results costs nothing within
    // a node and only the node leaders exchange data between nodes.
    MPI_Win cwin = MPI_WIN_NULL;
    double* Cnode = nullptr;
    {
        const size_t leaderElems =
            (rank == 0) ? N * N : static_cast<size_t>(nodeRowCounts[myNode]) * N;
        const MPI_Aint bytes =
            (localRank == 0) ? static_cast<MPI_Aint>(leaderElems * sizeof(double)) : 0;
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Win_allocate_shared(bytes, sizeof(double), info, shmComm, &Cnode, &cwin);
        MPI_Info_free(&info);
        if (localRank != 0) {
            MPI_Aint qsize = 0;
            int qdisp = 0;
            MPI_Win_shared_query(cwin, 0, &qsize, &qdisp, &Cnode);
        }
    }
    // this rank's slice of the node buffer
    double* C = Cnode + (myOffset - nodeOffset) * N;

    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < myRows; ++i) {
        double* row = &C[i * N];
        for (size_t j = 0; j < N; ++j) row[j] = 0.0;
    }

    // ---- GPU setup --------------------------------------------------------
    GpuContext gpu;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) deviceCount = 0;
    bool pinned = false;
    if (deviceCount > 0 && myRows > 0) {
        gpu.device = localRank % deviceCount;
        CUDA_CHECK(cudaSetDevice(gpu.device));
        gpu.ldk = static_cast<int>(roundUp(N, GPU_TILE));
        gpu.ldn = gpu.ldk;
        gpu.mpad = roundUp(myRows, GPU_TILE);
        CUDA_CHECK(cudaMalloc(&gpu.dA, gpu.mpad * gpu.ldk * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&gpu.dB, static_cast<size_t>(gpu.ldk) * gpu.ldn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&gpu.dC, gpu.mpad * gpu.ldn * sizeof(double)));
        CUDA_CHECK(cudaStreamCreate(&gpu.stream[0]));
        CUDA_CHECK(cudaStreamCreate(&gpu.stream[1]));
        CUDA_CHECK(cudaEventCreate(&gpu.evStart));
        CUDA_CHECK(cudaEventCreate(&gpu.evEnd[0]));
        CUDA_CHECK(cudaEventCreate(&gpu.evEnd[1]));

        // Generate the operands directly in device memory (no host transfer)
        {
            const int tpb = 256;
            dim3 gridA((gpu.ldk + tpb - 1) / tpb, static_cast<unsigned>(gpu.mpad));
            initDeviceA<<<gridA, tpb>>>(gpu.dA, N, myOffset, myRows, gpu.ldk);
            dim3 gridB((gpu.ldn + tpb - 1) / tpb, static_cast<unsigned>(gpu.ldk));
            initDeviceB<<<gridB, tpb>>>(gpu.dB, N, gpu.ldn);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Pin the local C block (page aligned) so that the result copy can
        // overlap with the host computation
        const size_t page = 4096;
        void* pinBase = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(C) & ~(page - 1));
        const size_t pinBytes =
            roundUp(static_cast<size_t>(static_cast<char*>(static_cast<void*>(C)) -
                                        static_cast<char*>(pinBase)) +
                        myRows * N * sizeof(double),
                    page);
        pinned = cudaHostRegister(pinBase, pinBytes, cudaHostRegisterDefault) == cudaSuccess;
        gpu.pinBase = pinned ? pinBase : nullptr;
        if (!pinned) cudaGetLastError();
        if (getenv("MATMUL_DEBUG")) fprintf(stderr, "[rank %d] C pinned: %d\n", rank, (int)pinned);
        gpu.available = true;
    } else if (rank == 0 && deviceCount == 0) {
        printf("Warning: no CUDA device visible, running on CPU only\n");
    }

    // ---- CPU workspace ----------------------------------------------------
    const int nthreads = omp_get_max_threads();
    CpuWorkspace ws;
    ws.Ap.assign(nthreads, nullptr);
    {
        bool ok = true;
        for (int t = 0; t < nthreads; ++t) {
            ws.Ap[t] = static_cast<double*>(aligned_alloc(64, MC_MAX * KC * sizeof(double)));
            ok = ok && ws.Ap[t] != nullptr;
        }
        ws.Bp = static_cast<double*>(aligned_alloc(64, roundUp(NC, NR) * KC * sizeof(double)));
        if (!ok || !ws.Bp) {
            fprintf(stderr, "allocation of the packing buffers failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Communicator of the node leaders (only they take part in the final gather)
    MPI_Comm leaderComm;
    MPI_Comm_split(MPI_COMM_WORLD, localRank == 0 ? 0 : MPI_UNDEFINED, rank, &leaderComm);

    // Row datatype for the gather of C
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    // ---- Calibration ------------------------------------------------------
    // Measure the achievable device and host row rates (and the fixed cost of a
    // host pass, which packs all of B once) so that the work split below is
    // balanced.  This also brings the GPU clocks out of their idle state.  The
    // results of these trial runs are recomputed by the timed run below.
    double gpuRate = 0.0, cpuRate = 0.0, cpuFixed = 0.0;
    if (myRows > 0) {
        // Both engines run concurrently, exactly as in the timed pass, so that
        // their mutual interference is part of the measurement.  Two host trials
        // of different size separate the fixed cost of a host pass (which packs
        // all of B once) from the per-row rate.
        const size_t t1 = std::min(myRows, std::max<size_t>(MR, roundUp(myRows / 4, MR)));
        const size_t trials[2] = {t1, std::min(myRows, 3 * t1)};
        double times[2] = {0.0, 0.0};

        for (int i = 0; i < 2; ++i) {
            const size_t gTrial =
                gpu.available ? std::min(myRows, roundUp(trials[i], GPU_TILE)) : 0;
            int used = 0;
            if (gTrial > 0) {
                CUDA_CHECK(cudaEventRecord(gpu.evStart, gpu.stream[0]));
                used = gpuLaunch(gpu, C, N, 0, gTrial, false);
            }
            const auto t0 = std::chrono::high_resolution_clock::now();
            cpuGemmRows(A, B, C, N, myOffset, myOffset + trials[i], myOffset, ws);
            times[i] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            if (gTrial > 0) {
                CUDA_CHECK(cudaDeviceSynchronize());
                const double gt = gpuElapsed(gpu, used);
                if (gt > 0.0) gpuRate = std::max(gpuRate, static_cast<double>(gTrial) / gt);
            }
        }

        if (getenv("MATMUL_DEBUG"))
            fprintf(stderr, "[rank %d] trials: %zu rows %.4f s, %zu rows %.4f s\n", rank,
                    trials[0], times[0], trials[1], times[1]);
        if (trials[1] > trials[0] && times[1] > times[0]) {
            cpuRate = static_cast<double>(trials[1] - trials[0]) / (times[1] - times[0]);
            cpuFixed = std::max(0.0, times[0] - static_cast<double>(trials[0]) / cpuRate);
        } else if (times[0] > 0.0) {
            cpuRate = static_cast<double>(trials[0]) / times[0];
        }
    }

    // Rows for the GPU: t_gpu == t_cpu  =>  g/rGpu == fixed + (R-g)/rCpu
    size_t gpuRows = 0;
    if (gpu.available && gpuRate > 0.0) {
        gpuRows = myRows;
        if (cpuRate > 0.0) {
            const double g = gpuRate * (static_cast<double>(myRows) + cpuFixed * cpuRate) /
                             (gpuRate + cpuRate);
            gpuRows = static_cast<size_t>(std::max(0.0, std::min<double>(myRows, g)));
        }
        gpuRows = std::min(roundUp(gpuRows, GPU_TILE), myRows);
    }
    if (const char* f = getenv("MATMUL_GPU_FRAC"))
        gpuRows = gpu.available ? std::min(roundUp(static_cast<size_t>(myRows * atof(f)), GPU_TILE), myRows) : 0;
    if (getenv("MATMUL_DEBUG"))
        fprintf(stderr, "[rank %d] rates: gpu %.1f rows/s, cpu %.1f rows/s, fixed %.3f s -> gpu rows %zu of %zu\n",
                rank, gpuRate, cpuRate, cpuFixed, gpuRows, myRows);

    // ---- Computation ------------------------------------------------------
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    {
        int usedStreams = 0;
        if (gpuRows > 0) {
            CUDA_CHECK(cudaEventRecord(gpu.evStart, gpu.stream[0]));
            usedStreams = gpuLaunch(gpu, C, N, 0, gpuRows, pinned);
        }

        const auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuGemmRows(A, B, C, N, myOffset + gpuRows, myOffset + myRows, myOffset, ws);
        const double cpuTime =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cpuStart).count();

        double gpuTime = 0.0;
        if (gpuRows > 0) {
            if (!pinned) {
                // fall back to a blocking copy when C could not be pinned
                CUDA_CHECK(cudaMemcpy2D(C, N * sizeof(double), gpu.dC, gpu.ldn * sizeof(double),
                                        N * sizeof(double), gpuRows, cudaMemcpyDeviceToHost));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            gpuTime = gpuElapsed(gpu, usedStreams);
        }

        if (getenv("MATMUL_DEBUG"))
            fprintf(stderr, "[rank %d] gpu rows %zu in %.3f s, cpu rows %zu in %.3f s\n", rank,
                    gpuRows, gpuTime, myRows - gpuRows, cpuTime);
    }

    // Collect the row blocks: intra-node results are already in place, so only
    // the node leaders exchange their contiguous blocks.
    const auto gatherStart = std::chrono::high_resolution_clock::now();
    MPI_Barrier(shmComm);
    if (nnodes > 1 && localRank == 0) {
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : Cnode, nodeRowCounts[myNode], rowType, Cnode,
                    nodeRowCounts.data(), nodeRowOffsets.data(), rowType, 0, leaderComm);
    }

    if (getenv("MATMUL_DEBUG"))
        fprintf(stderr, "[rank %d] gather %.3f s\n", rank,
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - gatherStart).count());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            std::vector<double> Cvec(Cnode, Cnode + N * N);
            print_results(Cvec, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, Cnode, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- Cleanup ----------------------------------------------------------
    MPI_Type_free(&rowType);
    if (leaderComm != MPI_COMM_NULL) MPI_Comm_free(&leaderComm);
    if (gpu.available) {
        if (gpu.pinBase) cudaHostUnregister(gpu.pinBase);
        cudaFree(gpu.dA);
        cudaFree(gpu.dB);
        cudaFree(gpu.dC);
        cudaStreamDestroy(gpu.stream[0]);
        cudaStreamDestroy(gpu.stream[1]);
        cudaEventDestroy(gpu.evStart);
        cudaEventDestroy(gpu.evEnd[0]);
        cudaEventDestroy(gpu.evEnd[1]);
    }
    free(A);
    free(B);
    MPI_Win_free(&cwin);
    MPI_Comm_free(&shmComm);
    MPI_Finalize();
    return exitCode;
}
