// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//   * MPI:    the rows of C (and A) are block-distributed over the ranks. B is a
//             deterministic function, so every rank generates it locally (no
//             broadcast). Finished row blocks are streamed to rank 0 while the
//             computation is still running.
//   * CUDA:   each rank drives one GPU (preferring a GPU on its NUMA node) with
//             a register-tiled, double-buffered FP64 kernel.
//   * OpenMP: the remaining cores of each rank compute rows concurrently with
//             the GPU using a cache-blocked, vectorized CPU kernel. Rows are
//             handed out dynamically according to the measured GPU and CPU
//             throughput, so both finish at about the same time.
//
// Every element C[i][j] is accumulated from 0.0 in the original k order with
// fused multiply-adds on both GPU and CPU, so the result does not depend on
// the number of ranks/threads or on how rows are split between GPU and CPU.
//
// Recommended launch: one rank per GPU, each bound to its share of cores, e.g.
//   mpirun -np 4 --map-by ppr:2:socket:pe=32 ./matmul -n 16384
// (by default each rank uses all CPUs it is bound to, shared among node-local
// ranks with overlapping CPU sets; OMP_NUM_THREADS overrides this).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <atomic>
#include <thread>

#include <unistd.h>
#include <sched.h>
#include <cctype>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [rowBegin, rowEnd) of an NxN matrix into mat (row-major, local rows)
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = mat + (i - rowBegin) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

__global__ void initMatrixKernel(double* __restrict__ mat, const size_t N,
                                 const size_t rowBegin, const size_t rows) {
    const size_t total = rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = idx / N;
        const size_t j = idx - i * N;
        mat[idx] = getPseudoRndValue(N, rowBegin + i, j);
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: C[M x N] = A[M x N] * B[N x N], (lda = ldb = ldc = N)
// Block tile BM x BN, K step BK, each thread computes TM x TN outputs.
// ---------------------------------------------------------------------------
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TX = 16;  // threads in x (columns)
constexpr int TY = 16;  // threads in y (rows)
constexpr int TM = BM / TY;
constexpr int TN = BN / TX;

__global__ void __launch_bounds__(TX * TY)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const int M, const int N) {
    __shared__ double As[2][BK][BM + 1];
    __shared__ double Bs[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    constexpr int NT = TX * TY;
    constexpr int A_LOADS = BM * BK / NT;
    constexpr int B_LOADS = BK * BN / NT;
    double aReg[A_LOADS];
    double bReg[B_LOADS];

    // Out-of-range elements are loaded as zero; fma(0, 0, s) == s exactly
    // for the non-negative partial sums of finite values, so padding does
    // not change the result.
    auto loadGlobal = [&](int k0) {
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int e = tid + l * NT;
            const int r = e / BK, c = e % BK;
            const int gr = rowBase + r, gc = k0 + c;
            aReg[l] = (gr < M && gc < N) ? A[(size_t)gr * N + gc] : 0.0;
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int e = tid + l * NT;
            const int r = e / BN, c = e % BN;
            const int gr = k0 + r, gc = colBase + c;
            bReg[l] = (gr < N && gc < N) ? B[(size_t)gr * N + gc] : 0.0;
        }
    };
    auto storeShared = [&](int buf) {
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int e = tid + l * NT;
            As[buf][e % BK][e / BK] = aReg[l];
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int e = tid + l * NT;
            Bs[buf][e / BN][e % BN] = bReg[l];
        }
    };

    const int numTiles = (N + BK - 1) / BK;
    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < numTiles) loadGlobal((t + 1) * BK);

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[cur][k][ty + i * TY];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[cur][k][tx + j * TX];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }

        if (t + 1 < numTiles) storeShared(cur ^ 1);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + ty + i * TY;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + tx + j * TX;
            if (gc < N) C[(size_t)gr * N + gc] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// CPU kernel: rows [r0, r1) of C (local indexing) using the same per-element
// k-ordered FMA accumulation as the GPU. B is pre-packed into column panels of
// CPU_NR columns (Bp[panel][k][CPU_NR], zero padded) so the micro-kernel
// streams it contiguously. Vectorization is across columns only, so every
// C element is still a sequential k-ordered sum.
// ---------------------------------------------------------------------------
constexpr size_t CPU_MR = 6;    // rows per micro-tile
constexpr size_t CPU_NR = 8;    // columns per micro-tile (2 x AVX2 vectors)
constexpr size_t CPU_KC = 256;  // k-blocking: A micro-panel slice stays in L1
constexpr size_t CPU_MC = 192;  // max rows per CPU team block

// Pack column panels [p0, p1) of B into Bp (Bp[panel][k][CPU_NR], zero padded)
static void packBPanels(const double* __restrict__ B, double* __restrict__ Bp,
                        const size_t N, const size_t p0, const size_t p1) {
    for (size_t k = 0; k < N; ++k) {
        const double* src = B + k * N;
        for (size_t p = p0; p < p1; ++p) {
            const size_t j0 = p * CPU_NR;
            const size_t jn = std::min(CPU_NR, N - j0);
            double* dst = Bp + (p * N + k) * CPU_NR;
            if (jn == CPU_NR) {
                std::memcpy(dst, src + j0, CPU_NR * sizeof(double));
            } else {
                for (size_t j = 0; j < CPU_NR; ++j) dst[j] = (j < jn) ? src[j0 + j] : 0.0;
            }
        }
    }
}

// 4-wide double vector (GCC vector extension; maps to AVX2 registers)
typedef double v4d __attribute__((vector_size(32)));

template <size_t MR>
static inline void cpuMicroKernel(const double* __restrict__ ap, const double* __restrict__ bp,
                                  double* __restrict__ C, const size_t N, const size_t k0,
                                  const size_t k1, const size_t jn, const bool first) {
    v4d acc[MR][2];
    for (size_t i = 0; i < MR; ++i) {
        if (first) {
            acc[i][0] = v4d{0.0, 0.0, 0.0, 0.0};
            acc[i][1] = v4d{0.0, 0.0, 0.0, 0.0};
        } else if (jn == CPU_NR) {
            std::memcpy(&acc[i][0], C + i * N, sizeof(v4d));
            std::memcpy(&acc[i][1], C + i * N + 4, sizeof(v4d));
        } else {
            double tmp[CPU_NR] = {};
            for (size_t j = 0; j < jn; ++j) tmp[j] = C[i * N + j];
            std::memcpy(&acc[i][0], tmp, sizeof(v4d));
            std::memcpy(&acc[i][1], tmp + 4, sizeof(v4d));
        }
    }
    for (size_t k = k0; k < k1; ++k) {
        const v4d b0 = *reinterpret_cast<const v4d*>(bp + k * CPU_NR);
        const v4d b1 = *reinterpret_cast<const v4d*>(bp + k * CPU_NR + 4);
        for (size_t i = 0; i < MR; ++i) {
            const double av = ap[k * MR + i];
            const v4d a = {av, av, av, av};
            acc[i][0] = a * b0 + acc[i][0];  // contracted to FMA (-ffp-contract=fast)
            acc[i][1] = a * b1 + acc[i][1];
        }
    }
    for (size_t i = 0; i < MR; ++i) {
        if (jn == CPU_NR) {
            std::memcpy(C + i * N, &acc[i][0], sizeof(v4d));
            std::memcpy(C + i * N + 4, &acc[i][1], sizeof(v4d));
        } else {
            double tmp[CPU_NR];
            std::memcpy(tmp, &acc[i][0], sizeof(v4d));
            std::memcpy(tmp + 4, &acc[i][1], sizeof(v4d));
            for (size_t j = 0; j < jn; ++j) C[i * N + j] = tmp[j];
        }
    }
}

// Pack local rows [r0, r1) of A into row micro-panels: the panel of rows
// [i, i+m) (m = min(MR, r1-i)) is stored k-major at Ap + (i - r0) * N.
static void packARows(const double* __restrict__ A, double* __restrict__ Ap, const size_t N,
                      const size_t r0, const size_t i, const size_t m) {
    double* dst = Ap + (i - r0) * N;
    for (size_t r = 0; r < m; ++r) {
        const double* src = A + (i + r) * N;
        for (size_t k = 0; k < N; ++k) dst[k * m + r] = src[k];
    }
}

// C[r0:r1, panels p0:p1] for local rows (C indexed by local row) from the
// packed A block (packARows with the same r0) and packed B.
template <size_t MR>
static inline void cpuRowTile(const double* __restrict__ Ap, const double* __restrict__ Bp,
                              double* __restrict__ C, const size_t r0, const size_t i,
                              const size_t p0, const size_t p1, const size_t N,
                              const size_t k0, const size_t k1, const bool first) {
    for (size_t p = p0; p < p1; ++p) {
        const size_t j0 = p * CPU_NR;
        cpuMicroKernel<MR>(Ap + (i - r0) * N, Bp + p * N * CPU_NR, C + i * N + j0, N, k0, k1,
                           std::min(CPU_NR, N - j0), first);
    }
}

static void cpuMatmulBlock(const double* __restrict__ Ap, const double* __restrict__ Bp,
                           double* __restrict__ C, const size_t r0, const size_t r1,
                           const size_t pBegin, const size_t pEnd, const size_t N) {
    // Panel groups whose K-slice of B (~128 KiB) stays in L2 across row tiles
    constexpr size_t PG = std::max<size_t>(1, (128 * 1024) / (CPU_KC * CPU_NR * sizeof(double)));
    for (size_t k0 = 0; k0 < N; k0 += CPU_KC)
    for (size_t p0 = pBegin; p0 < pEnd; p0 += PG) {
        const size_t p1 = std::min(pEnd, p0 + PG);
        const size_t k1 = std::min(N, k0 + CPU_KC);
        const bool first = (k0 == 0);
        size_t i = r0;
        for (; i + CPU_MR <= r1; i += CPU_MR)
            cpuRowTile<CPU_MR>(Ap, Bp, C, r0, i, p0, p1, N, k0, k1, first);
        switch (r1 - i) {
            case 5: cpuRowTile<5>(Ap, Bp, C, r0, i, p0, p1, N, k0, k1, first); break;
            case 4: cpuRowTile<4>(Ap, Bp, C, r0, i, p0, p1, N, k0, k1, first); break;
            case 3: cpuRowTile<3>(Ap, Bp, C, r0, i, p0, p1, N, k0, k1, first); break;
            case 2: cpuRowTile<2>(Ap, Bp, C, r0, i, p0, p1, N, k0, k1, first); break;
            case 1: cpuRowTile<1>(Ap, Bp, C, r0, i, p0, p1, N, k0, k1, first); break;
            default: break;
        }
    }
}

// NUMA node of a CUDA device (from sysfs), -1 if unknown
static int deviceNumaNode(int device) {
    char busId[32] = {};
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), device) != cudaSuccess) return -1;
    for (char* c = busId; *c; ++c) *c = static_cast<char>(tolower(*c));
    char path[128];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", busId);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int node = -1;
    if (fscanf(f, "%d", &node) != 1) node = -1;
    fclose(f);
    return node;
}

// NUMA node this process is bound to, -1 if unbound or spanning several nodes
static int processNumaNode() {
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return -1;
    int node = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) continue;
        int cpuNode = -1;
        for (int n = 0; n < 64 && cpuNode < 0; ++n) {
            char path[128];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/node%d", cpu, n);
            if (access(path, F_OK) == 0) cpuNode = n;
        }
        if (cpuNode < 0 || (node >= 0 && cpuNode != node)) return -1;
        node = cpuNode;
    }
    return node;
}

// Assign GPUs to the node-local ranks: prefer devices on the rank's NUMA
// node, and spread ranks evenly over the devices.
static int selectDevice(MPI_Comm nodeComm, int numDevices) {
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    const int myNode = processNumaNode();
    std::vector<int> rankNode(localSize);
    MPI_Allgather(&myNode, 1, MPI_INT, rankNode.data(), 1, MPI_INT, nodeComm);
    std::vector<int> devNode(numDevices), used(numDevices, 0);
    for (int d = 0; d < numDevices; ++d) devNode[d] = deviceNumaNode(d);

    int mine = localRank % numDevices;
    for (int r = 0; r < localSize; ++r) {
        int best = -1;
        for (int pass = 0; pass < 2 && best < 0; ++pass) {
            for (int d = 0; d < numDevices; ++d) {
                if (pass == 0 && (rankNode[r] < 0 || devNode[d] != rankNode[r])) continue;
                if (best < 0 || used[d] < used[best]) best = d;
            }
        }
        ++used[best];
        if (r == localRank) mine = best;
    }
    return mine;
}

// Spin barrier for the CPU worker subset of a parallel region (the GPU driver
// and receiver threads must not take part, so omp barrier cannot be used)
class SpinBarrier {
  public:
    explicit SpinBarrier(int n) : n_(n) {}
    void wait() {
        const unsigned gen = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            count_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        } else {
            for (unsigned spins = 0; gen_.load(std::memory_order_acquire) == gen; ++spins) {
                if (spins < 4096) {
#if defined(__x86_64__) || defined(__i386__)
                    asm volatile("pause");  // spare the SMT sibling's resources
#endif
                } else {
                    std::this_thread::yield();
                }
            }
        }
    }

  private:
    const int n_;
    std::atomic<int> count_{0};
    std::atomic<unsigned> gen_{0};
};

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
    // Threads concurrently stream finished row blocks to rank 0
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block row distribution over ranks
    std::vector<int> rowCounts(nranks), rowDispls(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t b = N * r / nranks, e = N * (r + 1) / nranks;
        rowCounts[r] = static_cast<int>(e - b);
        rowDispls[r] = static_cast<int>(b);
    }
    const size_t rowBegin = rowDispls[rank];
    const size_t localRows = rowCounts[rank];

    // Select a node-local GPU
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localSize = 1;
    MPI_Comm_size(nodeComm, &localSize);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = selectDevice(nodeComm, numDevices);
    CUDA_CHECK(cudaSetDevice(device));
    // Block (instead of spinning) while waiting for the GPU: the CPU cores are
    // busy computing as well
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    int gpuSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&gpuSMs, cudaDevAttrMultiProcessorCount, device));

    // Unless the user chose a thread count, share the CPUs this process may
    // run on with the node-local ranks whose CPU sets overlap with it.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
            for (int c = 0; c < std::min<long>(CPU_SETSIZE, sysconf(_SC_NPROCESSORS_ONLN)); ++c) CPU_SET(c, &mask);
        }
        std::vector<cpu_set_t> masks(localSize);
        MPI_Allgather(&mask, sizeof(cpu_set_t), MPI_BYTE, masks.data(), sizeof(cpu_set_t), MPI_BYTE, nodeComm);
        int sharing = 0;
        for (auto& other : masks) {
            cpu_set_t common;
            CPU_AND(&common, &mask, &other);
            if (CPU_COUNT(&common) > 0) ++sharing;
        }
        const int perRank = std::max(1, CPU_COUNT(&mask) / std::max(1, sharing));
        omp_set_num_threads(std::max(2, perRank));
    }

    // Stream finished row blocks to rank 0 while computing (needs concurrent
    // MPI calls from several threads and row offsets that fit into a tag).
    int* tagUB = nullptr;
    int tagFlag = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tagUB, &tagFlag);
    const size_t maxLocalRows = *std::max_element(rowCounts.begin(), rowCounts.end());
    const bool streamResults = nranks > 1 && provided >= MPI_THREAD_MULTIPLE && tagFlag &&
                               maxLocalRows <= static_cast<size_t>(*tagUB);
    // Rank 0 receives with several threads in parallel, each serving the
    // ranks of one result communicator: rank s sends via resultComms[(s-1) % R]
    const int numResultComms = streamResults ? std::min(nranks - 1, 8) : 0;
    std::vector<MPI_Comm> resultComms(numResultComms);
    for (auto& comm : resultComms) MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    // OpenMP thread roles: 0 = GPU driver, 1..cpuThreads = CPU workers, then
    // the result receivers (rank 0) or the result sender (other ranks)
    omp_set_dynamic(0);  // thread roles below rely on getting the requested team size
    const int nthreads = std::min(omp_get_max_threads(), omp_get_thread_limit());
    const int numReceivers = (rank == 0) ? numResultComms : 0;
    const int numSenders = (streamResults && rank != 0) ? 1 : 0;
    const int cpuThreads = std::max(0, nthreads - 1 - numReceivers - numSenders);

    // Allocate matrices. Rank 0 holds the full matrices (for output/validation),
    // other ranks only their rows of A and C, plus the full B.
    std::vector<double> A, B, C;
    B.resize(N * N);
    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
    } else {
        A.resize(localRows * N);
    }
    const size_t localBytes = std::max<size_t>(localRows, 1) * N * sizeof(double);

    // Initialize matrices (host copies for the CPU part and validation)
    if (rank == 0) printf("Initializing matrices...\n");
    if (rank == 0) {
        initMatrixRows(A.data(), N, 0, N);
    } else {
        initMatrixRows(A.data(), N, rowBegin, rowBegin + localRows);
    }
    initMatrixRows(B.data(), N, 0, N);
    const double* hA = (rank == 0) ? A.data() + rowBegin * N : A.data();  // local rows of A

    // Local rows of C in page-locked memory; rank 0 writes directly into C
    double* hC = nullptr;
    bool hCRegistered = false;
    if (rank == 0) {
        hC = C.data() + rowBegin * N;
        if (localRows > 0) {
            hCRegistered = cudaHostRegister(hC, localRows * N * sizeof(double),
                                            cudaHostRegisterDefault) == cudaSuccess;
            if (!hCRegistered) (void)cudaGetLastError();  // fall back to pageable copies
        }
    } else {
        CUDA_CHECK(cudaMallocHost(&hC, localBytes));
    }

    // Column-panel packed copy of B for the CPU micro-kernel (filled in the timed region)
    const size_t numPanels = (N + CPU_NR - 1) / CPU_NR;
    double* Bp = nullptr;
    double* Ap = nullptr;  // packed row block of A
    if (cpuThreads > 0) {
        Ap = static_cast<double*>(std::aligned_alloc(64, CPU_MC * std::max<size_t>(N, 1) * sizeof(double)));
        Bp = static_cast<double*>(std::aligned_alloc(64, std::max<size_t>(numPanels, 1) * CPU_NR * N * sizeof(double) + 64));
        if (Bp == nullptr || Ap == nullptr) {
            fprintf(stderr, "Allocation of packed A/B failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Map the pages up front (parallel first touch) so that the packing in
        // the timed region does not pay for page faults
        const size_t bpElems = std::max<size_t>(numPanels, 1) * CPU_NR * N;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < bpElems; i += 512) Bp[i] = 0.0;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < CPU_MC * N; i += 512) Ap[i] = 0.0;
    }

    // Device matrices, generated directly on the GPU
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, localBytes));
    CUDA_CHECK(cudaMalloc(&dB, std::max<size_t>(N * N, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, localBytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (localRows > 0) {
        initMatrixKernel<<<1024, 256, 0, stream>>>(dA, N, rowBegin, localRows);
        CUDA_CHECK(cudaGetLastError());
    }
    if (N > 0) {
        initMatrixKernel<<<2048, 256, 0, stream>>>(dB, N, 0, N);
        CUDA_CHECK(cudaGetLastError());
    }
    // Warm up the kernel (module load) outside of the timed region
    matmulKernel<<<dim3(1, 1), dim3(TX, TY), 0, stream>>>(dA, dB, dC, 0, (int)N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // GPU chunk size limit: a few full waves of thread blocks
    int blocksPerSM = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, matmulKernel, TX * TY, 0));
    const size_t colBlocks = (N + BN - 1) / BN;
    const size_t waveBlocks = static_cast<size_t>(std::max(1, blocksPerSM)) * gpuSMs;
    const size_t gpuMaxRows = std::max<size_t>(1, (4 * waveBlocks + colBlocks - 1) / std::max<size_t>(colBlocks, 1)) * BM;

    // Row datatype for result transfers (keeps counts small for large N)
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(std::max<size_t>(N, 1)), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Dynamic GPU/CPU work sharing within the rank. Unclaimed local rows are
    // [front, back). OpenMP thread 0 drives the GPU and claims row chunks from
    // the front; the CPU worker threads pack B and then claim row blocks from
    // the back, each worker computing its own column panels of the block.
    // Chunk sizes follow the measured GPU/CPU throughput and shrink with the
    // remaining work so that both finish at about the same time. Finished
    // chunks are sent to rank 0 immediately, where receiver threads store
    // them into C, overlapping the result collection with computation.
    {
        using clock = std::chrono::steady_clock;
        auto seconds = [](clock::time_point a, clock::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        auto roundUp = [](size_t x, size_t m) { return (x + m - 1) / m * m; };

        // Shared scheduling state (atomics so that every thread always reads
        // the current value; updates happen under the lock)
        std::atomic<size_t> front{0}, back{localRows};
        std::atomic<double> gpuRowsDone{0}, gpuBusy{0}, cpuRowsDone{0}, cpuBusy{0};
        omp_lock_t lock, sendLock;
        omp_init_lock(&lock);
        omp_init_lock(&sendLock);

        // CPU worker state: current row block and B packing progress
        std::atomic<size_t> blkB{0}, blkE{0}, nextPack{0};
        SpinBarrier cpuBarrier(cpuThreads);

        // Fraction of the remaining work the GPU should take (measured rates)
        auto gpuShare = [&]() {
            if (gpuRowsDone.load() > 0 && cpuRowsDone.load() > 0) {
                const double g = gpuRowsDone.load() / gpuBusy.load();
                const double c = cpuRowsDone.load() / cpuBusy.load();
                return g / (g + c);
            }
            return 0.5;
        };
        // Hand a finished block of local rows [b, e) over to the sender thread
        std::vector<std::pair<size_t, size_t>> sendQueue;
        auto chunkDone = [&](size_t b, size_t e) {
            if (!streamResults || rank == 0 || e <= b) return;
            omp_set_lock(&sendLock);
            sendQueue.emplace_back(b, e);
            omp_unset_lock(&sendLock);
        };

        #pragma omp parallel num_threads(1 + cpuThreads + numReceivers + numSenders)
        {
            const int role = omp_get_thread_num();
            if (omp_get_num_threads() != 1 + cpuThreads + numReceivers + numSenders) {
                fprintf(stderr, "Could not start the required number of OpenMP threads\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (role == 0) {
                // GPU driver
                for (;;) {
                    omp_set_lock(&lock);
                    const size_t remaining = back.load() - front.load();
                    size_t rows = std::min(remaining, gpuMaxRows);
                    if (cpuThreads > 0) {
                        const size_t share = roundUp(static_cast<size_t>(remaining * gpuShare() / 2), BM);
                        rows = std::min(rows, std::max<size_t>(share, BM));
                    }
                    const size_t b = front.load();
                    front.store(b + rows);
                    omp_unset_lock(&lock);
                    if (rows == 0) break;

                    const auto ts = clock::now();
                    const dim3 grid(static_cast<unsigned>(colBlocks), static_cast<unsigned>((rows + BM - 1) / BM));
                    matmulKernel<<<grid, dim3(TX, TY), 0, stream>>>(dA + b * N, dB, dC + b * N,
                                                                     static_cast<int>(rows), static_cast<int>(N));
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpyAsync(hC + b * N, dC + b * N, rows * N * sizeof(double),
                                               cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    const double dt = seconds(ts, clock::now());
                    chunkDone(b, b + rows);

                    omp_set_lock(&lock);
                    gpuRowsDone.store(gpuRowsDone.load() + rows);
                    gpuBusy.store(gpuBusy.load() + dt);
                    omp_unset_lock(&lock);
                }
            } else if (role <= cpuThreads) {
                // CPU worker w of cpuThreads
                const size_t w = role - 1, nw = cpuThreads;

                // Pack B into column panels, a few panels per task
                constexpr size_t PP = 8;
                for (size_t pb; (pb = nextPack.fetch_add(PP)) < numPanels;)
                    packBPanels(B.data(), Bp, N, pb, std::min(numPanels, pb + PP));

                const size_t p0 = numPanels * w / nw, p1 = numPanels * (w + 1) / nw;
                clock::time_point blkStart;
                for (;;) {
                    if (w == 0) {
                        // Account the finished block and claim the next one
                        const auto now = clock::now();
                        const size_t prevB = blkB.load(), prevE = blkE.load();
                        chunkDone(prevB, prevE);
                        omp_set_lock(&lock);
                        if (prevE > prevB) {
                            cpuRowsDone.store(cpuRowsDone.load() + (prevE - prevB));
                            cpuBusy.store(cpuBusy.load() + seconds(blkStart, now));
                        }
                        const size_t remaining = back.load() - front.load();
                        const size_t share = roundUp(static_cast<size_t>(remaining * (1.0 - gpuShare()) / 2), CPU_MR);
                        const size_t rows = std::min(remaining, std::min(CPU_MC, std::max(share, CPU_MR)));
                        const size_t e = back.load();
                        back.store(e - rows);
                        omp_unset_lock(&lock);
                        blkB.store(e - rows);
                        blkE.store(e);
                        blkStart = now;
                    }
                    cpuBarrier.wait();  // block claimed (and B packed, first time)
                    const size_t b = blkB.load(), e = blkE.load();
                    if (b >= e) break;
                    const size_t tiles = (e - b + CPU_MR - 1) / CPU_MR;
                    for (size_t t = w; t < tiles; t += nw)
                        packARows(hA, Ap, N, b, b + t * CPU_MR, std::min(CPU_MR, e - b - t * CPU_MR));
                    cpuBarrier.wait();  // A block packed
                    if (p0 < p1) cpuMatmulBlock(Ap, Bp, hC, b, e, p0, p1, N);
                    cpuBarrier.wait();  // block finished
                }
            } else if (rank != 0) {
                // Result sender: blocking sends keep the transfers progressing
                const MPI_Comm comm = resultComms[(rank - 1) % numResultComms];
                size_t sent = 0;
                while (sent < localRows) {
                    std::pair<size_t, size_t> chunk{0, 0};
                    omp_set_lock(&sendLock);
                    if (!sendQueue.empty()) {
                        chunk = sendQueue.back();
                        sendQueue.pop_back();
                    }
                    omp_unset_lock(&sendLock);
                    if (chunk.second <= chunk.first) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                        continue;
                    }
                    MPI_Send(hC + chunk.first * N, static_cast<int>(chunk.second - chunk.first), rowType, 0,
                             static_cast<int>(chunk.first), comm);
                    sent += chunk.second - chunk.first;
                }
            } else {
                // Rank 0 result receiver: store remote row blocks into C
                const int r = role - 1 - cpuThreads;
                size_t expected = 0, received = 0;
                for (int src = 1 + r; src < nranks; src += numResultComms) expected += rowCounts[src];
                while (received < expected) {
                    MPI_Message msg;
                    MPI_Status status;
                    int found = 0;
                    MPI_Improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, resultComms[r], &found, &msg, &status);
                    if (!found) {
                        // Poll gently, leaving the core to the compute threads
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                        continue;
                    }
                    int count = 0;
                    MPI_Get_count(&status, rowType, &count);
                    const size_t row = static_cast<size_t>(rowDispls[status.MPI_SOURCE]) + status.MPI_TAG;
                    MPI_Mrecv(C.data() + row * N, count, rowType, &msg, MPI_STATUS_IGNORE);
                    received += count;
                }
            }
        }
        omp_destroy_lock(&lock);
        omp_destroy_lock(&sendLock);
    }

    // Without streaming, collect the result on rank 0 now
    if (!streamResults && nranks > 1) {
        if (rank == 0) {
            MPI_Gatherv(MPI_IN_PLACE, 0, rowType, C.data(), rowCounts.data(), rowDispls.data(),
                        rowType, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(hC, static_cast<int>(localRows), rowType, nullptr, rowCounts.data(),
                        rowDispls.data(), rowType, 0, MPI_COMM_WORLD);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count(), max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) duration = std::chrono::milliseconds(max_duration_ms);

    MPI_Type_free(&rowType);
    for (auto& comm : resultComms) MPI_Comm_free(&comm);
    MPI_Comm_free(&nodeComm);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    if (rank == 0) {
        if (hCRegistered) CUDA_CHECK(cudaHostUnregister(hC));
    } else {
        CUDA_CHECK(cudaFreeHost(hC));
    }
    std::free(Bp);
    std::free(Ap);

    int exitCode = 0;
    if (rank == 0) {
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
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
