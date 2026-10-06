// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//  * MPI:    the rows of C (and A) are block-distributed over the ranks; every rank
//            generates its own rows of A and the full B (deterministic, no communication),
//            and finished result rows are streamed to rank 0 while the computation goes on.
//  * Inside a rank the local rows are shared dynamically between
//      - the GPU (CUDA tiled DGEMM kernel), which claims row chunks from the front, and
//      - the CPU cores (OpenMP, AVX2/FMA register-blocked DGEMM), which claim row slabs
//        from the back,
//    so that both devices finish at about the same time regardless of their relative speed.
//    One OpenMP team holds the GPU driver thread, the communication thread and one pinned
//    compute thread per physical core.
//
// Every C element is accumulated over k in ascending order with fused multiply-adds on both
// devices, i.e. with the same arithmetic as the original sequential loop.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <deque>
#include <array>
#include <thread>

#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
// The CUDA front end does not know GCC's AMX tile builtins; they are not used here.
#define _AMXTILEINTRIN_H_INCLUDED
#include <immintrin.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [rowBegin, rowEnd) of an NxN matrix into mat (row pitch N)
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ----------------------------------------------------------------------------------------
// GPU part
// ----------------------------------------------------------------------------------------
constexpr int GBM = 64;   // rows of C per block
constexpr int GBN = 64;   // cols of C per block
constexpr int GBK = 16;   // k-tile
constexpr int GTX = 16;   // threads per block in x
constexpr int GTY = 16;   // threads per block in y
constexpr int GRM = GBM / GTY;  // 4 rows per thread
constexpr int GRN = GBN / GTX;  // 4 cols per thread

// C[M x Np] = A[M x Kp] * B[Kp x Np]; all dimensions padded to tile multiples with zeros.
__global__ void __launch_bounds__(GTX * GTY)
dgemmKernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
            const int Kp, const int Np) {
    __shared__ double As[2][GBK][GBM + 1];
    __shared__ double Bs[2][GBK][GBN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * GTX + tx;
    const size_t row0 = static_cast<size_t>(blockIdx.y) * GBM;
    const size_t col0 = static_cast<size_t>(blockIdx.x) * GBN;

    const double* Ab = A + row0 * Kp;
    const double* Bb = B + col0;

    double acc[GRM][GRN];
    #pragma unroll
    for (int i = 0; i < GRM; ++i)
        #pragma unroll
        for (int j = 0; j < GRN; ++j) acc[i][j] = 0.0;

    constexpr int LOADS = (GBM * GBK) / (GTX * GTY);  // 4
    double ra[LOADS], rb[LOADS];

    auto loadGlobal = [&](int k0) {
        #pragma unroll
        for (int l = 0; l < LOADS; ++l) {
            const int idx = tid + l * GTX * GTY;
            const int ar = idx / GBK, ac = idx % GBK;
            ra[l] = Ab[static_cast<size_t>(ar) * Kp + k0 + ac];
            const int br = idx / GBN, bc = idx % GBN;
            rb[l] = Bb[static_cast<size_t>(k0 + br) * Np + bc];
        }
    };
    auto storeShared = [&](int buf) {
        #pragma unroll
        for (int l = 0; l < LOADS; ++l) {
            const int idx = tid + l * GTX * GTY;
            As[buf][idx % GBK][idx / GBK] = ra[l];
            Bs[buf][idx / GBN][idx % GBN] = rb[l];
        }
    };

    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < Kp; k0 += GBK) {
        const bool hasNext = (k0 + GBK) < Kp;
        if (hasNext) loadGlobal(k0 + GBK);

        #pragma unroll
        for (int k = 0; k < GBK; ++k) {
            double a[GRM], b[GRN];
            #pragma unroll
            for (int i = 0; i < GRM; ++i) a[i] = As[buf][k][ty + i * GTY];
            #pragma unroll
            for (int j = 0; j < GRN; ++j) b[j] = Bs[buf][k][tx + j * GTX];
            #pragma unroll
            for (int i = 0; i < GRM; ++i)
                #pragma unroll
                for (int j = 0; j < GRN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }

        if (hasNext) {
            storeShared(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

    #pragma unroll
    for (int i = 0; i < GRM; ++i)
        #pragma unroll
        for (int j = 0; j < GRN; ++j)
            C[(row0 + ty + i * GTY) * Np + col0 + tx + j * GTX] = acc[i][j];
}

// ----------------------------------------------------------------------------------------
// CPU part (AVX2 + FMA, 6x8 register-blocked micro-kernel on packed panels)
// ----------------------------------------------------------------------------------------
constexpr int MR = 6;     // micro-tile rows
constexpr int NR = 8;     // micro-tile cols
constexpr int KC = 256;   // k-block (A micro-panel KC*MR doubles = 12 KB, L1-resident)
constexpr int SW = 6;     // panels per sub-strip (B k-slice KC*SW*NR doubles = 96 KB, in L2)
constexpr int SLAB = 240; // rows per CPU slab (C sub-strip SLAB*SW*NR doubles = 90 KB, in L2)

// c[MR x NR] (row stride ldc) (+)= sum_k a[k*MR + r] * b[k*NR + c]
static inline void microKernel(const int kc, const double* __restrict__ a, const double* __restrict__ b,
                               double* __restrict__ c, const size_t ldc, const bool accumulate) {
    __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (accumulate) {
        c00 = _mm256_loadu_pd(c + 0 * ldc); c01 = _mm256_loadu_pd(c + 0 * ldc + 4);
        c10 = _mm256_loadu_pd(c + 1 * ldc); c11 = _mm256_loadu_pd(c + 1 * ldc + 4);
        c20 = _mm256_loadu_pd(c + 2 * ldc); c21 = _mm256_loadu_pd(c + 2 * ldc + 4);
        c30 = _mm256_loadu_pd(c + 3 * ldc); c31 = _mm256_loadu_pd(c + 3 * ldc + 4);
        c40 = _mm256_loadu_pd(c + 4 * ldc); c41 = _mm256_loadu_pd(c + 4 * ldc + 4);
        c50 = _mm256_loadu_pd(c + 5 * ldc); c51 = _mm256_loadu_pd(c + 5 * ldc + 4);
    } else {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_pd();
    }
    for (int k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_load_pd(b);
        const __m256d b1 = _mm256_load_pd(b + 4);
        __m256d av;
        av = _mm256_broadcast_sd(a + 0); c00 = _mm256_fmadd_pd(av, b0, c00); c01 = _mm256_fmadd_pd(av, b1, c01);
        av = _mm256_broadcast_sd(a + 1); c10 = _mm256_fmadd_pd(av, b0, c10); c11 = _mm256_fmadd_pd(av, b1, c11);
        av = _mm256_broadcast_sd(a + 2); c20 = _mm256_fmadd_pd(av, b0, c20); c21 = _mm256_fmadd_pd(av, b1, c21);
        av = _mm256_broadcast_sd(a + 3); c30 = _mm256_fmadd_pd(av, b0, c30); c31 = _mm256_fmadd_pd(av, b1, c31);
        av = _mm256_broadcast_sd(a + 4); c40 = _mm256_fmadd_pd(av, b0, c40); c41 = _mm256_fmadd_pd(av, b1, c41);
        av = _mm256_broadcast_sd(a + 5); c50 = _mm256_fmadd_pd(av, b0, c50); c51 = _mm256_fmadd_pd(av, b1, c51);
        a += MR;
        b += NR;
    }
    _mm256_storeu_pd(c + 0 * ldc, c00); _mm256_storeu_pd(c + 0 * ldc + 4, c01);
    _mm256_storeu_pd(c + 1 * ldc, c10); _mm256_storeu_pd(c + 1 * ldc + 4, c11);
    _mm256_storeu_pd(c + 2 * ldc, c20); _mm256_storeu_pd(c + 2 * ldc + 4, c21);
    _mm256_storeu_pd(c + 3 * ldc, c30); _mm256_storeu_pd(c + 3 * ldc + 4, c31);
    _mm256_storeu_pd(c + 4 * ldc, c40); _mm256_storeu_pd(c + 4 * ldc + 4, c41);
    _mm256_storeu_pd(c + 5 * ldc, c50); _mm256_storeu_pd(c + 5 * ldc + 4, c51);
}

// Packed panel layout (A: W = MR rows, B: W = NR cols): k-block-major, so that the data of one
// k-block of all panels is contiguous: [k-block][panel][k within block][W].
static inline size_t packedOffset(const size_t k0, const size_t panel, const size_t nPanels,
                                  const size_t N, const size_t W) {
    const size_t kc = std::min<size_t>(KC, N - k0);
    return k0 * nPanels * W + panel * kc * W;
}

// Compute C rows [r0, r1) (local row indices) x cols [c0, c1) on the CPU (c0 multiple of NR).
// Each thread owns a column strip; it is processed in sub-strips of SW panels such that the
// B k-slice (KC x SW*NR) and the C sub-strip (slab rows x SW*NR) stay in the thread's private
// L2 cache, while the packed A slab is streamed from L3, where it is shared by all threads.
// Ap: packed A of the slab starting at local row r0 (nPanelsA MR-row panels).
// Bp: packed B (nPanelsB NR-col panels).
static void cpuStrip(const double* Ap, const double* Bp, double* C, const size_t N,
                     const size_t nPanelsA, const size_t nPanelsB,
                     const size_t r0, const size_t r1, const size_t c0, const size_t c1) {
    alignas(32) double tmp[MR * NR];
    for (size_t s0 = c0; s0 < c1; s0 += SW * NR) {
        const size_t s1 = std::min(c1, s0 + SW * NR);
        for (size_t k0 = 0; k0 < N; k0 += KC) {
            const int kc = static_cast<int>(std::min<size_t>(KC, N - k0));
            const bool acc = k0 > 0;
            for (size_t i = r0; i < r1; i += MR) {
                const double* ap = Ap + packedOffset(k0, (i - r0) / MR, nPanelsA, N, MR);
                const size_t mr = std::min<size_t>(MR, r1 - i);
                for (size_t j = s0; j < s1; j += NR) {
                    const double* bp = Bp + packedOffset(k0, j / NR, nPanelsB, N, NR);
                    const size_t nc = std::min<size_t>(NR, s1 - j);
                    double* c = C + i * N + j;
                    if (mr == MR && nc == NR) {
                        microKernel(kc, ap, bp, c, N, acc);
                    } else {
                        if (acc)
                            for (size_t r = 0; r < mr; ++r)
                                for (size_t q = 0; q < nc; ++q) tmp[r * NR + q] = c[r * N + q];
                        microKernel(kc, ap, bp, tmp, NR, acc);
                        for (size_t r = 0; r < mr; ++r)
                            for (size_t q = 0; q < nc; ++q) c[r * N + q] = tmp[r * NR + q];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------------------
// Validation (rank 0)
// ----------------------------------------------------------------------------------------
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

static int readTopologyValue(const int cpu, const char* name) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, name);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}

// CPU resources of this rank. The CPUs of the rank's affinity mask are shared with the other
// ranks on the node that have the same mask; one hardware thread per physical core is selected
// for the compute threads. If the launcher bound the rank much more tightly than its fair share
// of the node (e.g. the default bind-to-core of a plain "mpirun -np 1/2"), the mask is widened
// to the whole node first.
struct CpuResources {
    std::vector<int> cores;  // one CPU id per physical core used by the compute threads
    cpu_set_t rankSet;       // all hardware threads belonging to these cores
};

static CpuResources selectCpus(MPI_Comm nodeComm) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    sched_getaffinity(0, sizeof(mask), &mask);
    int nodeSize, nodeRank;
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_rank(nodeComm, &nodeRank);

    const int online = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    const int fair = std::max(1, online / nodeSize);
    int widen = (CPU_COUNT(&mask) * 2 <= fair && online <= CPU_SETSIZE) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &widen, 1, MPI_INT, MPI_MAX, nodeComm);
    if (widen) {
        CPU_ZERO(&mask);
        for (int c = 0; c < online; ++c) CPU_SET(c, &mask);
    }

    std::vector<cpu_set_t> all(nodeSize);
    MPI_Allgather(&mask, sizeof(cpu_set_t), MPI_BYTE, all.data(), sizeof(cpu_set_t), MPI_BYTE, nodeComm);
    int sharing = 0, myIndex = 0;
    for (int r = 0; r < nodeSize; ++r) {
        if (CPU_EQUAL(&all[r], &mask)) {
            if (r < nodeRank) ++myIndex;
            ++sharing;
        }
    }

    // Group the CPUs of the mask by physical core
    struct Core { int pkg, id; std::vector<int> cpus; };
    std::vector<Core> coreList;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &mask)) continue;
        const int pkg = readTopologyValue(c, "physical_package_id");
        const int id = readTopologyValue(c, "core_id");
        auto it = std::find_if(coreList.begin(), coreList.end(), [&](const Core& k) {
            return id >= 0 && k.pkg == pkg && k.id == id;
        });
        if (it == coreList.end()) coreList.push_back({pkg, id, {c}});
        else it->cpus.push_back(c);
    }

    CpuResources res;
    CPU_ZERO(&res.rankSet);
    const size_t n = coreList.size();
    size_t b = n * myIndex / sharing, e = n * (myIndex + 1) / sharing;
    if (b == e) { b = myIndex % n; e = b + 1; }  // more ranks than cores
    for (size_t i = b; i < e; ++i) {
        res.cores.push_back(coreList[i].cpus.front());
        for (int c : coreList[i].cpus) CPU_SET(c, &res.rankSet);
    }
    return res;
}

// Page-locked host memory for asynchronous transfers. Ordinary pages registered with CUDA are
// used instead of cudaMallocHost so that MPI's shared-memory single-copy path still works.
// Host buffer backed by transparent huge pages where available (fewer TLB misses)
static double* allocHuge(const size_t count) {
    constexpr size_t kHuge = 2u << 20;
    const size_t bytes = (std::max<size_t>(count, 1) * sizeof(double) + kHuge - 1) / kHuge * kHuge;
    double* p = static_cast<double*>(std::aligned_alloc(kHuge, bytes));
    if (!p) {
        fprintf(stderr, "Out of host memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

static double* allocPinned(const size_t count) {
    const size_t bytes = (std::max<size_t>(count, 1) * sizeof(double) + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    double* p = allocHuge(count);
    CUDA_CHECK(cudaHostRegister(p, bytes, cudaHostRegisterDefault));
    return p;
}

static void freePinned(double* p) {
    cudaHostUnregister(p);
    std::free(p);
}

// Barrier for the subset of OpenMP threads doing the CPU computation
class SpinBarrier {
  public:
    explicit SpinBarrier(const int n) : n_(n) {}
    void wait() {
        const int gen = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) == n_ - 1) {
            count_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        } else {
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == gen) {
                _mm_pause();
                if (++spins > 4096) std::this_thread::yield();
            }
        }
    }

  private:
    const int n_;
    alignas(64) std::atomic<int> count_{0};
    alignas(64) std::atomic<int> gen_{0};
};

static void pinCurrentThread(const int cpu) {
    cpu_set_t m;
    CPU_ZERO(&m);
    CPU_SET(cpu, &m);
    sched_setaffinity(0, sizeof(m), &m);
}

constexpr int kHeaderTag = 1;
constexpr int kDataTagBase = 2;

int main(int argc, char** argv) {
    // Open MPI's shared-memory single-copy (CMA) path is much slower than its pipelined copy for
    // the large result messages streamed to rank 0; prefer the latter unless the user chose.
    setenv("OMPI_MCA_btl_vader_single_copy_mechanism", "none", 0);
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank, nranks;
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

    // ---- Node-local setup: GPU selection and thread count ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank;
    MPI_Comm_rank(nodeComm, &localRank);
    int nDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDev));
    if (nDev == 0) {
        fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % nDev));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    CUDA_CHECK(cudaFree(nullptr));  // create the context outside of the timed region

    // OpenMP threads: one compute thread per physical core of this rank, plus one (mostly
    // sleeping) thread driving the GPU and one for the communication. OMP_NUM_THREADS, if set,
    // gives the number of compute threads + 1.
    const CpuResources cpuRes = selectCpus(nodeComm);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuRes.rankSet);
    const int nCores = static_cast<int>(cpuRes.cores.size());
    const int nCpuThreads = getenv("OMP_NUM_THREADS") ? std::max(1, omp_get_max_threads() - 1) : nCores;
    const bool pinThreads = nCpuThreads <= nCores;
    omp_set_dynamic(0);
    omp_set_num_threads(nCpuThreads + 2);  // the initialization loops create the thread pool

    // ---- Row decomposition over ranks ----
    const size_t rowBegin = N * rank / nranks;
    const size_t rowEnd = N * (rank + 1) / nranks;
    const size_t myRows = rowEnd - rowBegin;

    // Padded sizes for the GPU and CPU kernels
    const size_t Kp = (N + GBK - 1) / GBK * GBK;
    const size_t Np = (N + GBN - 1) / GBN * GBN;
    const size_t Nnr = (N + NR - 1) / NR * NR;
    const size_t maxSlabRows = SLAB;

    // Host buffers (pinned for asynchronous transfers)
    double *hA = nullptr, *hB = nullptr, *hC = nullptr;
    hA = allocPinned(myRows * N);
    hB = allocPinned(N * N);
    // Result: rank 0 computes its rows directly into the full result matrix (rows 0..myRows-1)
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
        hC = C.data();
        if (!C.empty()) CUDA_CHECK(cudaHostRegister(hC, N * N * sizeof(double), cudaHostRegisterDefault));
    } else {
        hC = allocPinned(myRows * N);
    }
    double* Bp = allocHuge(Nnr * N);
    double* Ap = allocHuge((maxSlabRows + MR + 8) * N);

    // GPU chunk size: enough 64x64 blocks to fill the device
    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, localRank % nDev));
    const size_t colBlocks = Np / GBN;
    size_t gpuChunk = GBM * std::max<size_t>(1, (4 * smCount + colBlocks - 1) / colBlocks);
    gpuChunk = std::min(gpuChunk, (myRows + GBM - 1) / GBM * GBM);
    gpuChunk = std::max<size_t>(gpuChunk, GBM);

    // Device buffers: full (padded) B, two A/C chunk buffers for double buffering
    double *dB = nullptr, *dA[2] = {nullptr, nullptr}, *dC[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&dB, Kp * Np * sizeof(double)));
    CUDA_CHECK(cudaMemset(dB, 0, Kp * Np * sizeof(double)));
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMalloc(&dA[s], gpuChunk * Kp * sizeof(double)));
        CUDA_CHECK(cudaMemset(dA[s], 0, gpuChunk * Kp * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC[s], gpuChunk * Np * sizeof(double)));
    }
    cudaStream_t streams[2];
    cudaEvent_t done[2], bReady;
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&done[s], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&bReady, cudaEventDisableTiming));

    // Initialize matrices (each rank: its rows of A and the full B)
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(hA, N, rowBegin, rowEnd);
    initMatrixRows(hB, N, 0, N);
    // Touch the packing buffers so that page faults are not part of the timing
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < Nnr * N; ++i) Bp[i] = 0.0;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (maxSlabRows + MR + 8) * N; ++i) Ap[i] = 0.0;

    // One row of the matrix as an MPI datatype (keeps message counts small)
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(std::max<size_t>(N, 1)), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);
    // Results are streamed to rank 0 during the computation if MPI supports concurrent calls
    // from several threads, otherwise they are gathered at the end.
    const bool streamResults = provided >= MPI_THREAD_MULTIPLE;

    // Warm-up outside the timed region: load the CUDA kernel module
    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, dgemmKernel));
        dgemmKernel<<<dim3(1, 1), dim3(GTX, GTY), 0, streams[0]>>>(dA[0], dB, dC[0], GBK, GBN);
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    }

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Shared work range of local rows: GPU takes from the front, CPU from the back
    std::mutex workMutex;
    // (atomics so that the compiler never privatizes them in the nested parallel regions)
    std::atomic<size_t> workLo{0}, workHi{myRows};
    // Throughput estimates (rows per second) used to balance the tail between GPU and CPU
    std::atomic<double> gpuRate{0.0}, cpuRate{0.0};
    std::atomic<size_t> gpuPending{0};  // rows claimed by the GPU but not finished yet
    std::atomic<size_t> cpuPending{0};  // rows of the slab the CPU is working on

    // Finished row blocks of ranks != 0 are streamed to rank 0 while the computation goes on:
    // a header {first global row, row count, data tag} followed by the rows themselves.
    std::mutex sendMutex;
    std::deque<std::array<long long, 3>> sendHeaders;  // deque: stable addresses for Isend
    std::vector<MPI_Request> sendRequests;
    std::atomic<int> sendCount{0};
    auto sendRows = [&](const size_t localLo, const size_t cnt) {
        if (!streamResults || rank == 0 || cnt == 0) return;
        std::lock_guard<std::mutex> lock(sendMutex);
        const int tag = kDataTagBase + sendCount++ % 30000;  // stays below MPI_TAG_UB
        sendHeaders.push_back({static_cast<long long>(rowBegin + localLo), static_cast<long long>(cnt), tag});
        MPI_Request req[2];
        MPI_Isend(sendHeaders.back().data(), 3, MPI_LONG_LONG, 0, kHeaderTag, MPI_COMM_WORLD, &req[0]);
        if (cnt * N <= static_cast<size_t>(INT_MAX))  // predefined type: fastest transport path
            MPI_Isend(hC + localLo * N, static_cast<int>(cnt * N), MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &req[1]);
        else
            MPI_Isend(hC + localLo * N, static_cast<int>(cnt), rowType, 0, tag, MPI_COMM_WORLD, &req[1]);
        sendRequests.push_back(req[0]);
        sendRequests.push_back(req[1]);
    };
    // Thread roles in the (single, flat) parallel region: thread 0 drives the GPU, thread 1 (if
    // present) is the communication thread, all others are CPU compute workers.
    const bool commThread = streamResults && nranks > 1;
    const int nExtra = commThread ? 2 : 1;
    std::atomic<int> producersLeft{2};  // GPU driver and CPU workers
    std::atomic<size_t> slabLo{0}, slabHi{0};  // current CPU slab
    SpinBarrier cpuBarrier(nCpuThreads);

    #pragma omp parallel num_threads(nCpuThreads + nExtra)
    {
        const int tnum = omp_get_thread_num();
        if (tnum == 0) {
            // ================= GPU driver =================
            CUDA_CHECK(cudaMemcpy2DAsync(dB, Np * sizeof(double), hB, N * sizeof(double),
                                         N * sizeof(double), N, cudaMemcpyHostToDevice, streams[0]));
            CUDA_CHECK(cudaEventRecord(bReady, streams[0]));
            CUDA_CHECK(cudaStreamWaitEvent(streams[1], bReady, 0));

            bool inFlight[2] = {false, false};
            size_t chunkLo[2] = {0, 0}, chunkCnt[2] = {0, 0};
            int s = 0;
            auto tPrev = std::chrono::high_resolution_clock::now();
            size_t rowsDone = 0;
            while (true) {
                size_t lo, cnt;
                {
                    std::lock_guard<std::mutex> lock(workMutex);
                    const size_t remaining = workHi - workLo;
                    cnt = std::min(gpuChunk, remaining);
                    // Until the throughputs are known, claim at most a quarter of the rows
                    if (rowsDone == 0) cnt = std::min(cnt, std::max<size_t>(GBM, (myRows / 4 + GBM - 1) / GBM * GBM));
                    const double gr = gpuRate.load(), cr = cpuRate.load();
                    if (gr > 0.0 && cr > 0.0) {
                        // Take only the GPU's share of what is left such that GPU and CPU finish
                        // together, counting the work both have already in progress
                        const double queued = static_cast<double>(gpuPending.load());
                        const double total = remaining + queued + cpuPending.load();
                        const double share = total * gr / (gr + cr) - queued;
                        if (share < 8.0) cnt = 0;
                        else cnt = std::min(cnt, static_cast<size_t>(std::ceil(share)));
                    }
                    gpuPending += cnt;
                    lo = workLo;
                    workLo += cnt;
                }
                if (cnt == 0) break;

                if (inFlight[s]) CUDA_CHECK(cudaEventSynchronize(done[s]));
                cudaStream_t st = streams[s];
                CUDA_CHECK(cudaMemcpy2DAsync(dA[s], Kp * sizeof(double), hA + lo * N, N * sizeof(double),
                                             N * sizeof(double), cnt, cudaMemcpyHostToDevice, st));
                const size_t rowsPad = (cnt + GBM - 1) / GBM * GBM;
                dim3 block(GTX, GTY);
                dim3 grid(static_cast<unsigned>(colBlocks), static_cast<unsigned>(rowsPad / GBM));
                dgemmKernel<<<grid, block, 0, st>>>(dA[s], dB, dC[s], static_cast<int>(Kp), static_cast<int>(Np));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy2DAsync(hC + lo * N, N * sizeof(double), dC[s], Np * sizeof(double),
                                             N * sizeof(double), cnt, cudaMemcpyDeviceToHost, st));
                CUDA_CHECK(cudaEventRecord(done[s], st));
                inFlight[s] = true;
                chunkLo[s] = lo;
                chunkCnt[s] = cnt;
                // Wait for the previous chunk (the GPU keeps working on the one just queued)
                const int p = s ^ 1;
                if (inFlight[p]) {
                    CUDA_CHECK(cudaEventSynchronize(done[p]));
                    inFlight[p] = false;
                    rowsDone += chunkCnt[p];
                    gpuPending -= chunkCnt[p];
                    sendRows(chunkLo[p], chunkCnt[p]);
                    // Throughput of the last chunk (the GPU was busy since the previous completion)
                    const auto now = std::chrono::high_resolution_clock::now();
                    const double el = std::chrono::duration<double>(now - tPrev).count();
                    if (el > 0.0) gpuRate.store(chunkCnt[p] / el);
                    tPrev = now;
                }
                s ^= 1;
            }
            for (int q = 0; q < 2; ++q) {
                const int p = s ^ q ^ 1;  // older chunk first
                if (inFlight[p]) {
                    CUDA_CHECK(cudaEventSynchronize(done[p]));
                    gpuPending -= chunkCnt[p];
                    sendRows(chunkLo[p], chunkCnt[p]);
                }
            }
            --producersLeft;
        } else if (tnum >= nExtra) {
            // ================= CPU workers =================
            const size_t wid = tnum - nExtra, nw = nCpuThreads;
            if (pinThreads) pinCurrentThread(cpuRes.cores[wid]);

            // Pack B into NR-column panels (zero-padded columns)
            const size_t nPanels = Nnr / NR;
            for (size_t jp = nPanels * wid / nw; jp < nPanels * (wid + 1) / nw; ++jp) {
                const size_t j0 = jp * NR;
                for (size_t k0 = 0; k0 < N; k0 += KC) {
                    double* dst = Bp + packedOffset(k0, jp, nPanels, N, NR);
                    for (size_t k = k0; k < std::min<size_t>(N, k0 + KC); ++k)
                        for (size_t q = 0; q < NR; ++q)
                            dst[(k - k0) * NR + q] = (j0 + q < N) ? hB[k * N + j0 + q] : 0.0;
                }
            }

            // Static column strip of this worker (in units of NR-wide panels)
            const size_t c0 = std::min(N, nPanels * wid / nw * NR);
            const size_t c1 = std::min(N, nPanels * (wid + 1) / nw * NR);

            auto tSlab = std::chrono::high_resolution_clock::now();
            while (true) {
                if (wid == 0) {
                    std::lock_guard<std::mutex> lock(workMutex);
                    const size_t remaining = workHi - workLo;
                    size_t cnt = std::min(maxSlabRows, remaining);
                    const double gr = gpuRate.load(), cr = cpuRate.load();
                    if (gr > 0.0 && cr > 0.0) {
                        // Near the end only take the CPU's fair share so that GPU and CPU
                        // finish together (the CPU has no work in progress at this point)
                        const double share = (remaining + gpuPending.load()) * cr / (gr + cr);
                        cnt = std::min(cnt, std::max<size_t>(MR, static_cast<size_t>(std::ceil(share))));
                    }
                    if (remaining - std::min(cnt, remaining) < 8) cnt = remaining;  // no tiny leftovers
                    cnt = std::min(cnt, remaining);
                    workHi -= cnt;
                    cpuPending = cnt;
                    slabLo = workHi.load();
                    slabHi = workHi.load() + cnt;
                    tSlab = std::chrono::high_resolution_clock::now();
                }
                cpuBarrier.wait();  // publishes the slab (and, the first time, the packed B)
                const size_t lo = slabLo, hi = slabHi;
                if (lo == hi) break;

                // Pack the slab of A into MR-row panels (zero-padded rows)
                const size_t panels = (hi - lo + MR - 1) / MR;
                for (size_t ip = panels * wid / nw; ip < panels * (wid + 1) / nw; ++ip) {
                    const size_t i0 = lo + ip * MR;
                    for (size_t k0 = 0; k0 < N; k0 += KC) {
                        double* dst = Ap + packedOffset(k0, ip, panels, N, MR);
                        for (size_t k = k0; k < std::min<size_t>(N, k0 + KC); ++k)
                            for (size_t r = 0; r < MR; ++r)
                                dst[(k - k0) * MR + r] = (i0 + r < hi) ? hA[(i0 + r) * N + k] : 0.0;
                    }
                }
                cpuBarrier.wait();

                if (c0 < c1) cpuStrip(Ap, Bp, hC, N, panels, nPanels, lo, hi, c0, c1);
                cpuBarrier.wait();

                if (wid == 0) {
                    cpuPending = 0;
                    sendRows(lo, hi - lo);
                    const double el = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - tSlab).count();
                    if (el > 0.0) cpuRate.store((hi - lo) / el);
                }
            }
            if (wid == 0) --producersLeft;
        } else {
            // ================= Communication thread =================
            // Data receives are posted as soon as their headers arrive so that the transfers
            // from all ranks proceed concurrently.
            long long rowsToReceive = rank == 0 ? static_cast<long long>(N - myRows) : 0;
            if (rank != 0) {
                // Drive the progress of the streamed sends until all have completed
                while (true) {
                    const bool last = producersLeft.load() == 0;
                    int allDone = 1;
                    {
                        std::lock_guard<std::mutex> lock(sendMutex);
                        if (!sendRequests.empty())
                            MPI_Testall(static_cast<int>(sendRequests.size()), sendRequests.data(), &allDone,
                                        MPI_STATUSES_IGNORE);
                    }
                    if (last && allDone) break;
                    std::this_thread::sleep_for(std::chrono::microseconds(allDone ? 50 : 5));
                }
            }
            std::deque<std::array<long long, 3>> headers;
            std::vector<MPI_Request> pending;
            std::vector<long long> pendingRows;
            std::vector<int> doneIdx;
            while (rowsToReceive > 0) {
                bool progress = false;
                int flag = 0;
                MPI_Status st;
                MPI_Iprobe(MPI_ANY_SOURCE, kHeaderTag, MPI_COMM_WORLD, &flag, &st);
                if (flag) {
                    headers.emplace_back();
                    long long* hdr = headers.back().data();
                    MPI_Recv(hdr, 3, MPI_LONG_LONG, st.MPI_SOURCE, kHeaderTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pending.emplace_back();
                    pendingRows.push_back(hdr[1]);
                    double* dst = C.data() + static_cast<size_t>(hdr[0]) * N;
                    const size_t elems = static_cast<size_t>(hdr[1]) * N;
                    if (elems <= static_cast<size_t>(INT_MAX))
                        MPI_Irecv(dst, static_cast<int>(elems), MPI_DOUBLE, st.MPI_SOURCE,
                                  static_cast<int>(hdr[2]), MPI_COMM_WORLD, &pending.back());
                    else
                        MPI_Irecv(dst, static_cast<int>(hdr[1]), rowType, st.MPI_SOURCE,
                                  static_cast<int>(hdr[2]), MPI_COMM_WORLD, &pending.back());
                    progress = true;
                }
                if (!pending.empty()) {
                    int nDone = 0;
                    doneIdx.resize(pending.size());
                    MPI_Testsome(static_cast<int>(pending.size()), pending.data(), &nDone, doneIdx.data(),
                                 MPI_STATUSES_IGNORE);
                    if (nDone > 0 && nDone != MPI_UNDEFINED) {
                        for (int d = 0; d < nDone; ++d) rowsToReceive -= pendingRows[doneIdx[d]];
                        size_t w = 0;
                        for (size_t q = 0; q < pending.size(); ++q) {
                            if (pending[q] != MPI_REQUEST_NULL) {
                                pending[w] = pending[q];
                                pendingRows[w] = pendingRows[q];
                                ++w;
                            }
                        }
                        pending.resize(w);
                        pendingRows.resize(w);
                    }
                    progress = true;  // keep driving the outstanding transfers
                }
                if (!progress) std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        }
    }

    if (streamResults) {
        if (rank != 0 && !sendRequests.empty()) MPI_Waitall(static_cast<int>(sendRequests.size()), sendRequests.data(), MPI_STATUSES_IGNORE);
    } else {
        // Gather the result rows on rank 0
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            const size_t b = N * r / nranks, e = N * (r + 1) / nranks;
            counts[r] = static_cast<int>(e - b);
            displs[r] = static_cast<int>(b);
        }
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : hC, static_cast<int>(myRows), rowType, rank == 0 ? C.data() : nullptr,
                    counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
    }
    MPI_Type_free(&rowType);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(duration_ms);

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
            std::vector<double> A(N * N), B(N * N);
            initMatrixRows(A.data(), N, 0, N);
            std::memcpy(B.data(), hB, N * N * sizeof(double));
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

    for (int s = 0; s < 2; ++s) {
        cudaFree(dA[s]);
        cudaFree(dC[s]);
        cudaStreamDestroy(streams[s]);
        cudaEventDestroy(done[s]);
    }
    cudaEventDestroy(bReady);
    cudaFree(dB);
    freePinned(hA);
    freePinned(hB);
    if (rank == 0) {
        if (!C.empty()) cudaHostUnregister(C.data());
    } else {
        freePinned(hC);
    }
    std::free(Ap);
    std::free(Bp);
    MPI_Comm_free(&nodeComm);

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
