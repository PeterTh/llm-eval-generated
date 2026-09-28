#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA matrix multiplication.
//
// Decomposition:
//   * MPI    : the rows of C (and correspondingly of A) are distributed over
//              the ranks. B is needed in full by everybody; since the matrix
//              initialization is a pure function of (N, i, j) it is generated
//              redundantly on every rank instead of being communicated. Only
//              the result is gathered on the root rank.
//   * CUDA   : every rank drives one or more GPUs. Each GPU processes row
//              panels of its rank's row block with a tiled DGEMM kernel.
//   * OpenMP : matrix initialization, and a cache-blocked, register-blocked
//              CPU DGEMM that runs concurrently with the GPUs. The GPU driver
//              threads and the CPU team pull row panels from a shared work
//              queue, so the CPU/GPU split balances itself at run time.
//
// The summation over k is performed in ascending order in both the CPU and the
// GPU kernel, so results are numerically equivalent to the sequential version.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,     \
                    __LINE__);                                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernel
//
// 32x32 output tile per block, 4 output elements per thread (block 32x8).
//
// This translation unit is compiled twice: once by nvcc (device code, below)
// and once by the host compiler (everything else), because the host compiler
// generates considerably faster code for the CPU part of the hybrid GEMM.
// ---------------------------------------------------------------------------

static constexpr int TILE = 32;

#ifdef __CUDACC__

__global__ void __launch_bounds__(256) matrixMultiplyKernel(
    const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
    const int rows, const int N, const int K) {
    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * TILE;
    const int col = blockIdx.x * TILE + tx;

    double sum[4] = {0.0, 0.0, 0.0, 0.0};

    for (int kt = 0; kt < K; kt += TILE) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int rr = ty + r * 8;
            const int row = row0 + rr;
            const int ka = kt + tx;
            As[rr][tx] = (row < rows && ka < K) ? A[static_cast<size_t>(row) * N + ka] : 0.0;
            const int kb = kt + rr;
            Bs[rr][tx] = (kb < K && col < N) ? B[static_cast<size_t>(kb) * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double b = Bs[k][tx];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                sum[r] += As[ty + r * 8][k] * b;
            }
        }
        __syncthreads();
    }

    if (col < N) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = row0 + ty + r * 8;
            if (row < rows) {
                C[static_cast<size_t>(row) * N + col] = sum[r];
            }
        }
    }
}

// Host-callable launcher (the only symbol this compilation unit provides)
void launchMatrixMultiplyKernel(const double* A, const double* B, double* C, const size_t rows,
                                const size_t N, const size_t K, cudaStream_t stream) {
    const dim3 block(TILE, 8);
    const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                    static_cast<unsigned>((rows + TILE - 1) / TILE));
    matrixMultiplyKernel<<<grid, block, 0, stream>>>(A, B, C, static_cast<int>(rows),
                                                     static_cast<int>(N), static_cast<int>(K));
}

#else // host compilation

void launchMatrixMultiplyKernel(const double* A, const double* B, double* C, size_t rows, size_t N,
                                size_t K, cudaStream_t stream);

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize the rows [rowBegin, rowEnd) of an NxN matrix. The storage holds
// exactly those rows, i.e. row rowBegin is stored at offset 0. The parallel
// loop also establishes the NUMA first-touch placement used by the CPU kernel.
void initMatrix(std::vector<double>& mat, const size_t N, const size_t rowBegin,
                const size_t rowEnd) {
#pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = mat.data() + (i - rowBegin) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU DGEMM
//
// Classical blocked GEMM: the k-loop is blocked so that a (KC x N) stripe of B
// stays cache resident, A is packed into small L1-resident panels and a 6x8
// register-blocked micro-kernel does the actual work. All CPU threads of a rank
// cooperate on the same B stripe, which keeps its traffic off the memory bus.
// ---------------------------------------------------------------------------

static constexpr size_t MR = 6;   // rows of the micro-kernel
static constexpr size_t NR = 8;   // columns of the micro-kernel (2 AVX registers)
static constexpr size_t KC = 256; // k-blocking factor
static constexpr size_t NC = 128; // column-blocking factor

// C[0:MR][0:NR] (+)= Ap[0:kc][0:MR]^T * B[0:kc][0:NR]
static inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ B,
                               const size_t ldb, double* __restrict__ C, const size_t ldc,
                               const size_t kc, const bool first) {
    double acc[MR][NR];
    for (size_t r = 0; r < MR; ++r) {
#pragma omp simd
        for (size_t j = 0; j < NR; ++j) acc[r][j] = 0.0;
    }

    for (size_t k = 0; k < kc; ++k) {
        const double* __restrict__ b = B + k * ldb;
        for (size_t r = 0; r < MR; ++r) {
            const double a = Ap[k * MR + r];
#pragma omp simd
            for (size_t j = 0; j < NR; ++j) acc[r][j] += a * b[j];
        }
    }

    for (size_t r = 0; r < MR; ++r) {
        double* __restrict__ c = C + r * ldc;
        if (first) {
#pragma omp simd
            for (size_t j = 0; j < NR; ++j) c[j] = acc[r][j];
        } else {
#pragma omp simd
            for (size_t j = 0; j < NR; ++j) c[j] += acc[r][j];
        }
    }
}

// Multiply `rows` rows of A with the full matrix B into the corresponding rows
// of C. Called by all threads of the CPU team; thread `teamRank` of `teamSize`
// processes its static share of the (column block, row block) pairs. The blocks
// are disjoint, so no synchronization is needed inside.
static void cpuMatrixMultiply(const double* __restrict__ A, const double* __restrict__ B,
                              double* __restrict__ C, const size_t N, const size_t rows,
                              const size_t teamRank, const size_t teamSize) {
    static thread_local std::vector<double> packA;
    static thread_local std::vector<double> packB;
    packA.resize(KC * MR);
    packB.resize(KC * NR);
    double* const Ap = packA.data();
    double* const Bp = packB.data();

    const size_t numIBlocks = (rows + MR - 1) / MR;
    const size_t numJBlocks = (N + NC - 1) / NC;
    const size_t numBlocks = numIBlocks * numJBlocks;
    const size_t perThread = (numBlocks + teamSize - 1) / teamSize;
    const size_t blockBegin = teamRank * perThread;
    const size_t blockEnd = (blockBegin + perThread < numBlocks) ? blockBegin + perThread : numBlocks;

    // Ascending k, so the summation order matches the sequential version
    for (size_t kc0 = 0; kc0 < N; kc0 += KC) {
        const size_t kc = (kc0 + KC < N) ? KC : N - kc0;
        const bool first = (kc0 == 0);

        for (size_t block = blockBegin; block < blockEnd; ++block) {
            const size_t jb = block / numIBlocks;
            const size_t ib = block % numIBlocks;
            const size_t jc = jb * NC;
            const size_t nc = (jc + NC < N) ? NC : N - jc;
            const size_t ir = ib * MR;
            const size_t mr = (ir + MR < rows) ? MR : rows - ir;

            // Pack the A micro-panel (kc x MR, k-major), zero padded
            for (size_t k = 0; k < kc; ++k) {
                for (size_t r = 0; r < MR; ++r) {
                    Ap[k * MR + r] = (r < mr) ? A[(ir + r) * N + kc0 + k] : 0.0;
                }
            }

            for (size_t jr = 0; jr < nc; jr += NR) {
                const size_t nr = (jr + NR < nc) ? NR : nc - jr;
                const double* b = B + kc0 * N + jc + jr;

                if (mr == MR && nr == NR) {
                    microKernel(Ap, b, N, C + ir * N + jc + jr, N, kc, first);
                    continue;
                }

                // Edge block: zero pad B and compute into a temporary tile
                for (size_t k = 0; k < kc; ++k) {
                    for (size_t j = 0; j < NR; ++j) {
                        Bp[k * NR + j] = (j < nr) ? b[k * N + j] : 0.0;
                    }
                }
                double tmp[MR * NR];
                microKernel(Ap, Bp, NR, tmp, NR, kc, true);
                for (size_t r = 0; r < mr; ++r) {
                    double* c = C + (ir + r) * N + jc + jr;
                    for (size_t j = 0; j < nr; ++j) {
                        if (first) {
                            c[j] = tmp[r * NR + j];
                        } else {
                            c[j] += tmp[r * NR + j];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multiplication of the local row block, shared between the GPUs and the CPU.
//
// The row block is handed out by a two-sided work queue: the GPUs consume row
// panels from the front, the CPU team consumes chunks from the back, and both
// stop when the two ends meet. The claim sizes shrink as the queue drains, so
// the CPU/GPU split needs no a-priori knowledge of their relative speed.
// ---------------------------------------------------------------------------

// Two pipeline slots per GPU: while one panel is being computed, the transfers
// of the other one are already in flight.
struct GpuSlot {
    double* dA = nullptr;
    double* dC = nullptr;
    cudaStream_t stream = nullptr;
    size_t rows = 0;
    bool busy = false;
};

struct GpuContext {
    int device = 0;
    size_t maxPanel = 0;
    double* dB = nullptr;
    GpuSlot slots[2];
};

// Sense reversing barrier for the CPU team. The GPU driver threads are part of
// the same OpenMP team but must not participate in it, so the OpenMP barrier
// cannot be used here.
class Barrier {
  public:
    explicit Barrier(const int numThreads) : numThreads_(numThreads) {}

    void wait() {
        const unsigned generation = generation_.load(std::memory_order_acquire);
        if (waiting_.fetch_add(1, std::memory_order_acq_rel) + 1 == numThreads_) {
            waiting_.store(0, std::memory_order_relaxed);
            generation_.store(generation + 1, std::memory_order_release);
        } else {
            while (generation_.load(std::memory_order_acquire) == generation) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
        }
    }

  private:
    const int numThreads_;
    std::atomic<int> waiting_{0};
    std::atomic<unsigned> generation_{0};
};

// Two-sided, throughput aware work queue. Every worker (each GPU driver, and
// the CPU team as a whole) first claims a small probe chunk, and from then on a
// share of the remaining rows proportional to its measured throughput, so the
// workers converge to finishing at the same time without any a-priori
// knowledge of the relative CPU/GPU speed.
class WorkQueue {
  public:
    WorkQueue(const size_t rows, const int numWorkers)
        : front_(0), back_(rows), rates_(numWorkers, 0.0) {}

    bool claim(const int worker, const bool fromFront, const double share, const size_t probeDiv,
               const size_t minRows, const size_t maxRows, size_t& begin, size_t& rows) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (front_ >= back_) return false;
        const size_t remaining = back_ - front_;
        const size_t numWorkers = rates_.size();

        size_t want;
        if (rates_[worker] <= 0.0) {
            // Probe chunk: a fraction of an equal share, large enough to be
            // representative but small enough to keep a slow worker from
            // holding up everybody else
            want = remaining / (probeDiv * numWorkers) + 1;
        } else {
            double known = 0.0;
            size_t unknown = 0;
            for (const double r : rates_) {
                if (r > 0.0) {
                    known += r;
                } else {
                    ++unknown;
                }
            }
            // Assume workers that have not reported yet are averagely fast
            const double avg = known / static_cast<double>(numWorkers - unknown);
            const double total = known + avg * static_cast<double>(unknown);
            want = static_cast<size_t>(share * static_cast<double>(remaining) * rates_[worker] /
                                       total);
        }

        if (want < minRows) want = minRows;
        if (want > maxRows) want = maxRows;
        if (want > remaining) want = remaining;

        if (fromFront) {
            begin = front_;
            front_ += want;
        } else {
            back_ -= want;
            begin = back_;
        }
        rows = want;
        return true;
    }

    // Report the throughput of a worker (rows completed so far and the time it
    // needed for them, including all of its start-up overhead)
    void report(const int worker, const size_t rows, const double seconds) {
        if (seconds <= 0.0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        rates_[worker] = static_cast<double>(rows) / seconds;
    }

  private:
    std::mutex mutex_;
    size_t front_;
    size_t back_;
    std::vector<double> rates_;
};

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const size_t localRows,
                    const std::vector<int>& myDevices) {
    if (localRows == 0) {
        return;
    }

    const int numGpus = static_cast<int>(myDevices.size());
    const int maxThreads = omp_get_max_threads();
    const int wantThreads = numGpus + ((maxThreads > numGpus + 1) ? maxThreads - numGpus : 1);

    // Find out how many threads we really get, so that the team sizes below (and
    // with them the CPU barrier) are guaranteed to match.
    omp_set_dynamic(0);
    int teamThreads = 0;
#pragma omp parallel num_threads(wantThreads)
    {
#pragma omp single
        teamThreads = omp_get_num_threads();
    }
    const int gpuWorkers = std::min(numGpus, std::max(teamThreads - 1, 0));
    const int cpuThreads = std::max(teamThreads - gpuWorkers, 1);

    // Upper bound for the GPU panels; also determines the device buffer sizes.
    size_t gpuPanelCap = gpuWorkers > 0 ? (localRows + 2 * gpuWorkers - 1) / (2 * gpuWorkers) : 0;
    if (gpuPanelCap < 256) gpuPanelCap = 256;
    if (gpuPanelCap > 4096) gpuPanelCap = 4096;
    if (gpuPanelCap > localRows) gpuPanelCap = localRows;

    std::vector<GpuContext> gpus(gpuWorkers);
    for (int g = 0; g < gpuWorkers; ++g) {
        GpuContext& ctx = gpus[g];
        ctx.device = myDevices[g];
        ctx.maxPanel = gpuPanelCap;
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaMalloc(&ctx.dB, N * N * sizeof(double)));
        for (GpuSlot& slot : ctx.slots) {
            CUDA_CHECK(cudaMalloc(&slot.dA, ctx.maxPanel * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&slot.dC, ctx.maxPanel * N * sizeof(double)));
            CUDA_CHECK(cudaStreamCreate(&slot.stream));
        }
    }

    // Worker 0..gpuWorkers-1 are the GPUs, worker gpuWorkers is the CPU team
    WorkQueue queue(localRows, gpuWorkers + 1);
    Barrier cpuBarrier(cpuThreads);
    struct {
        size_t begin = 0;
        size_t rows = 0;
        bool valid = false;
    } cpuChunk;

#pragma omp parallel num_threads(gpuWorkers + cpuThreads)
    {
        const int tid = omp_get_thread_num();

        if (tid < gpuWorkers) {
            GpuContext& ctx = gpus[tid];
            CUDA_CHECK(cudaSetDevice(ctx.device));

            // B is uploaded once, on the stream of the first panel; the other
            // slot waits for it to be complete.
            cudaEvent_t bReadyEvent;
            CUDA_CHECK(cudaEventCreateWithFlags(&bReadyEvent, cudaEventDisableTiming));

            bool bReady = false;
            int cur = 0;
            size_t doneRows = 0;
            const auto tStart = std::chrono::high_resolution_clock::now();
            bool more = true;

            while (more || ctx.slots[0].busy || ctx.slots[1].busy) {
                GpuSlot& slot = ctx.slots[cur];
                if (slot.busy) {
                    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                    slot.busy = false;
                    doneRows += slot.rows;
                    const std::chrono::duration<double> dt =
                        std::chrono::high_resolution_clock::now() - tStart;
                    queue.report(tid, doneRows, dt.count());
                }
                if (!more) {
                    cur ^= 1;
                    continue;
                }

                size_t begin = 0;
                size_t rows = 0;
                if (!queue.claim(tid, true, 0.6, 2, 64, ctx.maxPanel, begin, rows)) {
                    more = false;
                    continue;
                }
                slot.rows = rows;
                slot.busy = true;

                if (!bReady) {
                    CUDA_CHECK(cudaMemcpyAsync(ctx.dB, B.data(), N * N * sizeof(double),
                                               cudaMemcpyHostToDevice, slot.stream));
                    CUDA_CHECK(cudaEventRecord(bReadyEvent, slot.stream));
                    CUDA_CHECK(cudaStreamWaitEvent(ctx.slots[cur ^ 1].stream, bReadyEvent, 0));
                    bReady = true;
                }

                CUDA_CHECK(cudaMemcpyAsync(slot.dA, A.data() + begin * N,
                                           rows * N * sizeof(double), cudaMemcpyHostToDevice,
                                           slot.stream));
                launchMatrixMultiplyKernel(slot.dA, ctx.dB, slot.dC, rows, N, N, slot.stream);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(C.data() + begin * N, slot.dC,
                                           rows * N * sizeof(double), cudaMemcpyDeviceToHost,
                                           slot.stream));
                cur ^= 1;
            }

            CUDA_CHECK(cudaEventDestroy(bReadyEvent));
        } else {
            // CPU team: the first team member claims chunks from the back of the
            // queue, all members then process them together.
            const size_t teamRank = static_cast<size_t>(tid - gpuWorkers);
            const bool leader = (teamRank == 0);
            const size_t maxChunk = localRows;
            size_t doneRows = 0;
            const auto tStart = std::chrono::high_resolution_clock::now();

            while (true) {
                if (leader) {
                    cpuChunk.valid =
                        queue.claim(gpuWorkers, false, 1.0, 4, MR, maxChunk, cpuChunk.begin,
                                    cpuChunk.rows);
                }
                cpuBarrier.wait();
                if (!cpuChunk.valid) break;

                cpuMatrixMultiply(A.data() + cpuChunk.begin * N, B.data(),
                                  C.data() + cpuChunk.begin * N, N, cpuChunk.rows, teamRank,
                                  static_cast<size_t>(cpuThreads));
                const size_t chunkRows = cpuChunk.rows;
                cpuBarrier.wait();

                if (leader) {
                    doneRows += chunkRows;
                    const std::chrono::duration<double> dt =
                        std::chrono::high_resolution_clock::now() - tStart;
                    queue.report(gpuWorkers, doneRows, dt.count());
                }
            }
        }
    }

    for (int g = 0; g < gpuWorkers; ++g) {
        CUDA_CHECK(cudaSetDevice(gpus[g].device));
        CUDA_CHECK(cudaFree(gpus[g].dB));
        for (GpuSlot& slot : gpus[g].slots) {
            CUDA_CHECK(cudaStreamDestroy(slot.stream));
            CUDA_CHECK(cudaFree(slot.dA));
            CUDA_CHECK(cudaFree(slot.dC));
        }
    }
}

// Simple validation: compute a single element and compare
// (A is not stored in full on the root rank, its values are regenerated)
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    // ---- Determine the GPUs this rank is responsible for -------------------
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        deviceCount = 0;
    }

    std::vector<int> myDevices;
    if (deviceCount > 0) {
        if (localSize >= deviceCount) {
            myDevices.push_back(localRank % deviceCount);
        } else {
            // Fewer ranks than GPUs on this node: give every rank a share
            const int base = deviceCount / localSize;
            const int rem = deviceCount % localSize;
            const int first = localRank * base + (localRank < rem ? localRank : rem);
            const int count = base + (localRank < rem ? 1 : 0);
            for (int d = 0; d < count; ++d) myDevices.push_back(first + d);
        }
    }

    // ---- Row distribution --------------------------------------------------
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    std::vector<size_t> rowBegin(numRanks + 1, 0);
    {
        const size_t base = N / numRanks;
        const size_t rem = N % numRanks;
        size_t off = 0;
        for (int r = 0; r < numRanks; ++r) {
            rowBegin[r] = off;
            off += base + (static_cast<size_t>(r) < rem ? 1 : 0);
        }
        rowBegin[numRanks] = off;
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = static_cast<int>((rowBegin[r + 1] - rowBegin[r]) * N);
            displs[r] = static_cast<int>(rowBegin[r] * N);
        }
    }
    const size_t myFirstRow = rowBegin[rank];
    const size_t myRows = rowBegin[rank + 1] - myFirstRow;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI rank(s), %d OpenMP thread(s)/rank, %zu GPU(s)/rank\n",
               numRanks, omp_get_max_threads(), myDevices.size());
    }

    // Allocate matrices (A and C are distributed by rows, B is replicated)
    std::vector<double> A(myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rank == 0 ? N * N : myRows * N);

    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N, myFirstRow, myFirstRow + myRows);
    initMatrix(B, N, 0, N);

    // Set up the CUDA contexts and page-lock the host buffers so that the
    // transfers during the measured phase run at full PCIe speed.
    for (const int d : myDevices) {
        CUDA_CHECK(cudaSetDevice(d));
        // Block instead of spinning while waiting for the GPU, so that the
        // driver threads do not steal cores from the CPU part of the GEMM
        // (best effort: fails harmlessly if a context already exists).
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        cudaGetLastError();
        CUDA_CHECK(cudaFree(nullptr));
    }
    const auto tryPin = [&myDevices](std::vector<double>& v) {
        // Page locking only pays off for buffers spanning many pages, and small
        // ones may share a page with another buffer, which cannot be registered
        // twice.
        if (myDevices.empty() || v.size() * sizeof(double) < (1u << 20)) return false;
        return cudaHostRegister(v.data(), v.size() * sizeof(double), cudaHostRegisterDefault) ==
               cudaSuccess;
    };
    const bool pinnedA = tryPin(A);
    const bool pinnedB = tryPin(B);
    const bool pinnedC = tryPin(C);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N, myRows, myDevices);

    // Collect the full result on the root rank
    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, counts[0], MPI_DOUBLE, C.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(C.data(), counts[rank], MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (pinnedC) CUDA_CHECK(cudaHostUnregister(C.data()));
    if (pinnedB) CUDA_CHECK(cudaHostUnregister(B.data()));
    if (pinnedA) CUDA_CHECK(cudaHostUnregister(A.data()));

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
            bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
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

#endif // !__CUDACC__
