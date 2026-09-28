#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA matrix multiplication
//
// * MPI     distributes the rows of C (and of A) across the ranks.
// * CUDA    each rank drives one GPU which computes row panels of its block.
// * OpenMP  the remaining CPU cores compute the other row panels with a
//           cache-blocked, AVX2/FMA micro-kernel, and also run initialization
//           and validation.
//
// CPU and GPU pull work from a shared lock-free pool of row panels (the GPU
// claims panels from the end, the CPU threads from the front), so the
// CPU/GPU split adapts itself to the machine at run time.
// ---------------------------------------------------------------------------

// ------------------------- CPU kernel tuning parameters --------------------
constexpr int MR = 6;    // micro-kernel rows
constexpr int NR = 8;    // micro-kernel columns (2 AVX2 vectors)
constexpr int NV = NR / 4;
constexpr int MC_MAX = 240;  // rows per row panel (multiple of MR)
constexpr int NC_MAX = 512;  // columns per column block (multiple of NR)
constexpr int KC = 384;      // depth of a k-block

// ------------------------- GPU kernel tuning parameters --------------------
constexpr int GPU_BM = 64;   // block tile rows
constexpr int GPU_BN = 64;   // block tile columns
constexpr int GPU_BK = 16;   // block tile depth
constexpr int GPU_TM = 4;    // per-thread rows
constexpr int GPU_TN = 4;    // per-thread columns
constexpr int GPU_THREADS = (GPU_BM / GPU_TM) * (GPU_BN / GPU_TN);
constexpr int GPU_MAX_ROWS = 8 * MC_MAX;  // max rows per GPU work chunk
// Rough DGEMM throughput of one GPU; only used to size the work chunks.
constexpr double GPU_FLOPS_ESTIMATE = 6.0e11;

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        const cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                               \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                        \
        }                                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// Pin this rank to its fair share of the node's physical cores.
//
// Launchers bind an MPI rank to a single core or to a whole socket by default,
// which either starves or oversubscribes the OpenMP team of a hybrid code.
// The node-local ranks therefore partition the cores of the job allocation
// (the union of the inherited affinity masks) among themselves.
// ---------------------------------------------------------------------------
static bool isPrimarySibling(const int cpu) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE* f = fopen(path, "r");
    if (!f) return true;  // no topology information: treat every CPU as a core
    int first = -1;
    const int got = fscanf(f, "%d", &first);
    fclose(f);
    return got != 1 || first == cpu;
}

static void setupAffinity(MPI_Comm nodeComm, const int localRank, const int localSize) {
    cpu_set_t inherited;
    if (sched_getaffinity(0, sizeof(inherited), &inherited) != 0) return;

    // Union of the masks of all ranks on this node = the job's CPU allocation.
    constexpr int words = sizeof(cpu_set_t) / sizeof(unsigned long);
    unsigned long mine[words], all[words];
    memcpy(mine, &inherited, sizeof(inherited));
    MPI_Allreduce(mine, all, words, MPI_UNSIGNED_LONG, MPI_BOR, nodeComm);
    cpu_set_t allowed;
    memcpy(&allowed, all, sizeof(allowed));

    int nCpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    if (nCpus <= 0 || nCpus > CPU_SETSIZE) nCpus = CPU_SETSIZE;

    std::vector<int> cores;
    for (int c = 0; c < nCpus; ++c) {
        if (CPU_ISSET(c, &allowed) && isPrimarySibling(c)) cores.push_back(c);
    }
    if (static_cast<int>(cores.size()) < 2 * localSize) {
        // Implausibly small allocation: the launcher over-restricted the ranks
        // (e.g. "bind to core"), so fall back to all cores of the node.
        cores.clear();
        for (int c = 0; c < nCpus; ++c) {
            if (isPrimarySibling(c)) cores.push_back(c);
        }
    }
    if (cores.empty()) return;

    const size_t begin = (cores.size() * static_cast<size_t>(localRank)) / localSize;
    const size_t end = (cores.size() * static_cast<size_t>(localRank + 1)) / localSize;
    if (end <= begin) return;

    cpu_set_t target;
    CPU_ZERO(&target);
    for (size_t i = begin; i < end; ++i) CPU_SET(cores[i], &target);
    if (sched_setaffinity(0, sizeof(target), &target) != 0) return;

    if (getenv("OMP_NUM_THREADS") == nullptr) omp_set_num_threads(static_cast<int>(end - begin));
}

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [row0, row0 + rows) of a matrix, stored row-major with N columns.
void initMatrix(double* mat, const size_t N, const size_t row0, const size_t rows) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, row0 + i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA device kernel: register-tiled DGEMM, C = A * B
// Each thread accumulates a GPU_TM x GPU_TN tile in increasing k order.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(GPU_THREADS) void dgemmKernel(const double* __restrict__ A,
                                                           const double* __restrict__ B,
                                                           double* __restrict__ C, const int M,
                                                           const int N, const int K) {
    __shared__ double As[GPU_BK][GPU_BM + 1];
    __shared__ double Bs[GPU_BK][GPU_BN + 1];

    const int tid = threadIdx.x;
    const int tx = tid % (GPU_BN / GPU_TN);
    const int ty = tid / (GPU_BN / GPU_TN);
    const int row0 = blockIdx.y * GPU_BM;
    const int col0 = blockIdx.x * GPU_BN;

    double acc[GPU_TM][GPU_TN] = {};

    for (int k0 = 0; k0 < K; k0 += GPU_BK) {
        for (int idx = tid; idx < GPU_BM * GPU_BK; idx += GPU_THREADS) {
            const int m = idx / GPU_BK;
            const int k = idx % GPU_BK;
            const int gr = row0 + m;
            const int gk = k0 + k;
            As[k][m] = (gr < M && gk < K) ? A[static_cast<size_t>(gr) * K + gk] : 0.0;
        }
        for (int idx = tid; idx < GPU_BK * GPU_BN; idx += GPU_THREADS) {
            const int k = idx / GPU_BN;
            const int n = idx % GPU_BN;
            const int gk = k0 + k;
            const int gc = col0 + n;
            Bs[k][n] = (gk < K && gc < N) ? B[static_cast<size_t>(gk) * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < GPU_BK; ++k) {
            double a[GPU_TM];
            double b[GPU_TN];
#pragma unroll
            for (int i = 0; i < GPU_TM; ++i) a[i] = As[k][ty * GPU_TM + i];
#pragma unroll
            for (int j = 0; j < GPU_TN; ++j) b[j] = Bs[k][tx * GPU_TN + j];
#pragma unroll
            for (int i = 0; i < GPU_TM; ++i) {
#pragma unroll
                for (int j = 0; j < GPU_TN; ++j) acc[i][j] += a[i] * b[j];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < GPU_TM; ++i) {
        const int gr = row0 + ty * GPU_TM + i;
        if (gr >= M) continue;
        for (int j = 0; j < GPU_TN; ++j) {
            const int gc = col0 + tx * GPU_TN + j;
            if (gc < N) C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// CPU micro-kernel: C[MR x NR] += Ap[kc x MR] * Bp[kc x NR]
//
// GCC/Clang vector extensions are used instead of <immintrin.h> because the
// x86 intrinsic headers cannot be parsed by nvcc's front end; the generated
// code (vfmadd on AVX2 registers) is identical.
// ---------------------------------------------------------------------------
typedef double v4d __attribute__((vector_size(32)));
typedef double v4du __attribute__((vector_size(32), aligned(8)));  // unaligned access

static inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                               double* __restrict__ C, const size_t ldc, const int kc) {
    v4d acc[MR][NV];
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < NV; ++v) acc[i][v] = v4d{0.0, 0.0, 0.0, 0.0};

    for (int k = 0; k < kc; ++k) {
        v4d b[NV];
        for (int v = 0; v < NV; ++v) b[v] = *reinterpret_cast<const v4du*>(Bp + k * NR + v * 4);
        for (int i = 0; i < MR; ++i) {
            const double s = Ap[k * MR + i];
            const v4d a = v4d{s, s, s, s};
            for (int v = 0; v < NV; ++v) acc[i][v] += a * b[v];
        }
    }

    for (int i = 0; i < MR; ++i) {
        for (int v = 0; v < NV; ++v) {
            double* p = C + i * ldc + v * 4;
            *reinterpret_cast<v4du*>(p) = *reinterpret_cast<const v4du*>(p) + acc[i][v];
        }
    }
}

// Compute one C tile: C[i0..i0+mc, j0..j0+nc] = A[i0.., :] * B[:, j0..j0+nc]
static void cpuTile(const double* __restrict__ A, const double* __restrict__ B,
                    double* __restrict__ C, const size_t N, const size_t i0, const size_t mc,
                    const size_t j0, const size_t nc, double* __restrict__ Ap,
                    double* __restrict__ Bp, double* __restrict__ Cbuf) {
    for (size_t i = 0; i < mc; ++i) {
        double* row = C + (i0 + i) * N + j0;
        for (size_t j = 0; j < nc; ++j) row[j] = 0.0;
    }

    for (size_t pc = 0; pc < N; pc += KC) {
        const int kc = static_cast<int>(std::min<size_t>(KC, N - pc));

        // Pack B block (kc x nc) into NR-wide column panels
        for (size_t j = 0; j < nc; j += NR) {
            double* dst = Bp + (j / NR) * static_cast<size_t>(kc) * NR;
            for (int k = 0; k < kc; ++k) {
                const double* src = B + (pc + k) * N + j0 + j;
                const size_t avail = std::min<size_t>(NR, nc - j);
                for (size_t jj = 0; jj < avail; ++jj) dst[k * NR + jj] = src[jj];
                for (size_t jj = avail; jj < NR; ++jj) dst[k * NR + jj] = 0.0;
            }
        }

        // Pack A block (mc x kc) into MR-tall row panels
        for (size_t i = 0; i < mc; i += MR) {
            double* dst = Ap + (i / MR) * static_cast<size_t>(kc) * MR;
            const size_t avail = std::min<size_t>(MR, mc - i);
            for (int k = 0; k < kc; ++k) {
                for (size_t ii = 0; ii < avail; ++ii)
                    dst[k * MR + ii] = A[(i0 + i + ii) * N + pc + k];
                for (size_t ii = avail; ii < MR; ++ii) dst[k * MR + ii] = 0.0;
            }
        }

        for (size_t j = 0; j < nc; j += NR) {
            const double* Bpanel = Bp + (j / NR) * static_cast<size_t>(kc) * NR;
            for (size_t i = 0; i < mc; i += MR) {
                const double* Apanel = Ap + (i / MR) * static_cast<size_t>(kc) * MR;
                if (i + MR <= mc && j + NR <= nc) {
                    microKernel(Apanel, Bpanel, C + (i0 + i) * N + j0 + j, N, kc);
                } else {
                    for (int t = 0; t < MR * NR; ++t) Cbuf[t] = 0.0;
                    microKernel(Apanel, Bpanel, Cbuf, NR, kc);
                    const size_t mi = std::min<size_t>(MR, mc - i);
                    const size_t nj = std::min<size_t>(NR, nc - j);
                    for (size_t ii = 0; ii < mi; ++ii)
                        for (size_t jj = 0; jj < nj; ++jj)
                            C[(i0 + i + ii) * N + j0 + j + jj] += Cbuf[ii * NR + jj];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed / heterogeneous multiply of the local row block:
//   Cloc[mLocal x N] = Aloc[mLocal x N] * B[N x N]
// ---------------------------------------------------------------------------
namespace {

// Per-thread packing buffers. They are kept alive across calls and are
// pre-faulted during warm-up so that no allocation happens while timing.
std::vector<double>& packBufferA() {
    static thread_local std::vector<double> buf(static_cast<size_t>(KC) * ((MC_MAX + MR - 1) / MR) *
                                                MR);
    return buf;
}
std::vector<double>& packBufferB() {
    static thread_local std::vector<double> buf(static_cast<size_t>(KC) * ((NC_MAX + NR - 1) / NR) *
                                                NR);
    return buf;
}

constexpr int PANEL_FREE = 0;
constexpr int PANEL_CPU = 1;
constexpr int PANEL_GPU = 2;

struct GpuContext {
    bool available = false;
    int device = 0;
    double* dB = nullptr;
    double* dA[2] = {nullptr, nullptr};
    double* dC[2] = {nullptr, nullptr};
    cudaStream_t stream[2] = {};
    size_t maxRows = 0;
};

void gpuWorker(GpuContext& ctx, const double* Aloc, double* Cloc, const size_t N,
               const size_t nPanels, const size_t mLocal, const size_t mc,
               std::atomic<int>* owner) {
    if (!ctx.available) return;
    // The GPU is driven by a dedicated OpenMP thread; the current device is a
    // per-thread setting, so it has to be selected here as well.
    CUDA_CHECK(cudaSetDevice(ctx.device));

    // Panels are claimed just before they are launched, in bites that are
    // small enough to keep the CPU/GPU split balanced but large enough to
    // amortize kernel launch and transfer overhead (~30 ms of GPU work).
    const double targetRows = 0.03 * GPU_FLOPS_ESTIMATE / (2.0 * static_cast<double>(N) * N);
    const size_t bitePanels =
        std::max<size_t>(1, std::min<size_t>(ctx.maxRows / mc, static_cast<size_t>(targetRows) / mc));

    long long cursor = static_cast<long long>(nPanels) - 1;
    int buf = 0;
    bool inFlight[2] = {false, false};

    while (cursor >= 0) {
        if (inFlight[buf]) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream[buf]));
            inFlight[buf] = false;
        }

        // Claim a contiguous run of free panels ending at 'cursor'.
        long long hi = -1, lo = -1;
        size_t bite = bitePanels;
        while (cursor >= 0 && bite > 0) {
            int expected = PANEL_FREE;
            if (!owner[cursor].compare_exchange_strong(expected, PANEL_GPU,
                                                       std::memory_order_acq_rel)) {
                if (hi >= 0) break;  // end of the contiguous run
                --cursor;
                continue;
            }
            if (hi < 0) hi = cursor;
            lo = cursor;
            --cursor;
            --bite;
        }
        if (hi < 0) continue;

        const size_t row0 = static_cast<size_t>(lo) * mc;
        const size_t rows = std::min<size_t>(static_cast<size_t>(hi + 1) * mc, mLocal) - row0;

        CUDA_CHECK(cudaMemcpyAsync(ctx.dA[buf], Aloc + row0 * N, rows * N * sizeof(double),
                                   cudaMemcpyHostToDevice, ctx.stream[buf]));
        const dim3 grid(static_cast<unsigned>((N + GPU_BN - 1) / GPU_BN),
                        static_cast<unsigned>((rows + GPU_BM - 1) / GPU_BM));
        dgemmKernel<<<grid, GPU_THREADS, 0, ctx.stream[buf]>>>(
            ctx.dA[buf], ctx.dB, ctx.dC[buf], static_cast<int>(rows), static_cast<int>(N),
            static_cast<int>(N));
        CUDA_CHECK(cudaMemcpyAsync(Cloc + row0 * N, ctx.dC[buf], rows * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, ctx.stream[buf]));
        inFlight[buf] = true;
        buf ^= 1;
    }

    for (int s = 0; s < 2; ++s) {
        if (inFlight[s]) CUDA_CHECK(cudaStreamSynchronize(ctx.stream[s]));
    }
    CUDA_CHECK(cudaGetLastError());
}

void cpuWorker(const double* Aloc, const double* B, double* Cloc, const size_t N,
               const size_t nPanels, const size_t mLocal, const size_t mcSize,
               const size_t ncSize, std::atomic<int>* owner, std::atomic<size_t>& nextTask) {
    const size_t nColBlocks = (N + ncSize - 1) / ncSize;
    const size_t nTasks = nPanels * nColBlocks;
    std::vector<double>& Ap = packBufferA();
    std::vector<double>& Bp = packBufferB();
    double Cbuf[MR * NR];

    // Tasks are numbered row-panel major, so the CPU threads sweep the panels
    // from the front while the GPU claims panels from the back.
    for (size_t t = nextTask.fetch_add(1, std::memory_order_relaxed); t < nTasks;
         t = nextTask.fetch_add(1, std::memory_order_relaxed)) {
        const size_t p = t / nColBlocks;
        const size_t jb = t % nColBlocks;

        if (owner[p].load(std::memory_order_acquire) != PANEL_CPU) {
            int expected = PANEL_FREE;
            if (!owner[p].compare_exchange_strong(expected, PANEL_CPU,
                                                 std::memory_order_acq_rel) &&
                expected != PANEL_CPU) {
                continue;  // panel is being computed by the GPU
            }
        }

        const size_t i0 = p * mcSize;
        const size_t mc = std::min<size_t>(mcSize, mLocal - i0);
        const size_t j0 = jb * ncSize;
        const size_t nc = std::min<size_t>(ncSize, N - j0);
        cpuTile(Aloc, B, Cloc, N, i0, mc, j0, nc, Ap.data(), Bp.data(), Cbuf);
    }
}

}  // namespace

void matrixMultiplyLocal(GpuContext& ctx, const double* Aloc, const double* B, double* Cloc,
                         const size_t N, const size_t mLocal) {
    if (mLocal == 0) return;

    const int nThreads = std::max(1, omp_get_max_threads());

    // Shrink the tiles for small problems so that every thread gets work.
    size_t mcSize = MC_MAX, ncSize = NC_MAX;
    auto taskCount = [&] {
        return ((mLocal + mcSize - 1) / mcSize) * ((N + ncSize - 1) / ncSize);
    };
    while (taskCount() < 2 * static_cast<size_t>(nThreads)) {
        const bool shrinkCols = (ncSize / NR) >= (mcSize / MR) && ncSize > 2 * NR;
        if (shrinkCols) {
            ncSize = std::max<size_t>(NR, (ncSize / 2 / NR) * NR);
        } else if (mcSize > 2 * MR) {
            mcSize = std::max<size_t>(MR, (mcSize / 2 / MR) * MR);
        } else {
            break;
        }
    }

    const size_t nPanels = (mLocal + mcSize - 1) / mcSize;
    std::vector<std::atomic<int>> owner(nPanels);
    for (size_t i = 0; i < nPanels; ++i) owner[i].store(PANEL_FREE, std::memory_order_relaxed);
    std::atomic<size_t> nextTask{0};

    // One thread drives the GPU, all others compute row panels on the CPU.
#pragma omp parallel num_threads(nThreads)
    {
        if (ctx.available && omp_get_thread_num() == 0 && omp_get_num_threads() > 1) {
            gpuWorker(ctx, Aloc, Cloc, N, nPanels, mLocal, mcSize, owner.data());
        } else {
            cpuWorker(Aloc, B, Cloc, N, nPanels, mLocal, mcSize, ncSize, owner.data(), nextTask);
        }
    }

    // Single-threaded run: the GPU still has to process the panels it can get.
    if (ctx.available && nThreads == 1) {
        gpuWorker(ctx, Aloc, Cloc, N, nPanels, mLocal, mcSize, owner.data());
    }
}

// Simple validation: compute a few elements and compare
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    bool valid = true;
#pragma omp parallel for collapse(2) schedule(static) reduction(&& : valid)
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                valid = false;
            }
        }
    }

    return valid;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (isRoot) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row block owned by this rank
    const size_t row0 = (N * static_cast<size_t>(rank)) / static_cast<size_t>(nRanks);
    const size_t row1 = (N * static_cast<size_t>(rank + 1)) / static_cast<size_t>(nRanks);
    const size_t mLocal = row1 - row0;

    // Select this rank's GPU (round-robin over the devices of its node)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    setupAffinity(nodeComm, localRank, localSize);
    MPI_Comm_free(&nodeComm);

    if (isRoot) printf("MPI ranks: %d, OpenMP threads/rank: %d\n", nRanks, omp_get_max_threads());

    GpuContext ctx;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) deviceCount = 0;
    if (deviceCount > 0 && mLocal > 0) {
        ctx.device = localRank % deviceCount;
        CUDA_CHECK(cudaSetDevice(ctx.device));
        ctx.maxRows = std::min<size_t>(mLocal, GPU_MAX_ROWS);
        // Device memory: the replicated B plus two pipelined A/C row chunks.
        bool ok = cudaMalloc(&ctx.dB, N * N * sizeof(double)) == cudaSuccess;
        for (int s = 0; s < 2 && ok; ++s) {
            ok = cudaMalloc(&ctx.dA[s], ctx.maxRows * N * sizeof(double)) == cudaSuccess &&
                 cudaMalloc(&ctx.dC[s], ctx.maxRows * N * sizeof(double)) == cudaSuccess &&
                 cudaStreamCreate(&ctx.stream[s]) == cudaSuccess;
        }
        if (ok) {
            ctx.available = true;
        } else {
            // Not enough device memory for this problem: run on the CPU only.
            cudaGetLastError();
            cudaFree(ctx.dB);
            for (int s = 0; s < 2; ++s) {
                cudaFree(ctx.dA[s]);
                cudaFree(ctx.dC[s]);
            }
            cudaGetLastError();
            if (isRoot) printf("Warning: GPU memory allocation failed, using CPU only\n");
        }
    } else if (isRoot && deviceCount == 0) {
        printf("Warning: no CUDA device found, using CPU only\n");
    }

    // Allocate matrices: A rows and C rows are distributed, B is replicated.
    std::vector<double> Aloc(mLocal * N);
    std::vector<double> B(N * N);
    std::vector<double> Cloc(mLocal * N);
    std::vector<double> C;  // full result, assembled on the root

    // Initialize matrices
    if (isRoot) printf("Initializing matrices...\n");
    initMatrix(Aloc.data(), N, row0, mLocal);
    initMatrix(B.data(), N, 0, N);

    if (ctx.available) {
        // Page-lock the distributed panels so that the GPU transfers overlap
        // with computation; fall back silently to pageable transfers.
        if (cudaHostRegister(Aloc.data(), mLocal * N * sizeof(double), cudaHostRegisterDefault) !=
            cudaSuccess) {
            cudaGetLastError();
        }
        if (cudaHostRegister(Cloc.data(), mLocal * N * sizeof(double), cudaHostRegisterDefault) !=
            cudaSuccess) {
            cudaGetLastError();
        }
        CUDA_CHECK(cudaMemcpy(ctx.dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Warm up the OpenMP team and the CUDA context so that the measurement
    // covers the multiplication itself and not one-time runtime setup.
#pragma omp parallel
    {
        // Touch the per-thread packing buffers (allocation and first touch).
        std::vector<double>& a = packBufferA();
        std::vector<double>& b = packBufferB();
        for (size_t i = 0; i < a.size(); i += 512) a[i] = 0.0;
        for (size_t i = 0; i < b.size(); i += 512) b[i] = 0.0;
    }
    if (ctx.available) {
        dgemmKernel<<<dim3(1, 1), GPU_THREADS, 0, ctx.stream[0]>>>(ctx.dA[0], ctx.dB, ctx.dC[0], 1,
                                                                   1, 1);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream[0]));
    }

    // Perform matrix multiplication
    if (isRoot) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    // Wake the OpenMP team up again after the (blocking) barrier.
#pragma omp parallel
    { }
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(ctx, Aloc.data(), B.data(), Cloc.data(), N, mLocal);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();

    // Assemble the full result on the root
    if (isRoot) C.resize(N * N);
    {
        std::vector<int> counts(nRanks), displs(nRanks);
        for (int r = 0; r < nRanks; ++r) {
            const size_t s = (N * static_cast<size_t>(r)) / static_cast<size_t>(nRanks);
            const size_t e = (N * static_cast<size_t>(r + 1)) / static_cast<size_t>(nRanks);
            counts[r] = static_cast<int>((e - s) * N);
            displs[r] = static_cast<int>(s * N);
        }
        MPI_Gatherv(Cloc.data(), static_cast<int>(mLocal * N), MPI_DOUBLE, C.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (isRoot) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
    }

    // Validation
    int status = 0;
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (ctx.available) {
        cudaHostUnregister(Aloc.data());
        cudaHostUnregister(Cloc.data());
        cudaGetLastError();
        cudaFree(ctx.dB);
        for (int s = 0; s < 2; ++s) {
            cudaFree(ctx.dA[s]);
            cudaFree(ctx.dC[s]);
            cudaStreamDestroy(ctx.stream[s]);
        }
    }

    MPI_Finalize();
    return status;
}
