#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <utility>
#include <mutex>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <cuda_runtime.h>

#ifdef __CUDACC__
// The AMX intrinsic headers rely on GCC target pragmas that the nvcc front end
// cannot parse; none of them are needed here, so keep them out.
#define _AMXTILEINTRIN_H_INCLUDED
#define _AMXINT8INTRIN_H_INCLUDED
#define _AMXBF16INTRIN_H_INCLUDED
#endif
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// -----------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA matrix multiplication.
//
//  * MPI   distributes contiguous row blocks of C (and the matching rows of A)
//          over the ranks; every rank holds the full A and B (both are cheap to
//          regenerate locally, so no broadcast is needed).
//  * CUDA  computes row panels of the local block with a tiled shared-memory
//          DGEMM kernel; every GPU of the rank is driven by its own worker thread.
//  * OpenMP runs a cache-blocked, AVX2 register-blocked DGEMM with packed panels
//          on the remaining cores at the same time.
//
//  Both sides pull from one work queue over a 2D tile grid of the local block:
//  devices take whole row panels from the front, host threads take single tiles
//  from the back, and the host is throttled to its measured throughput share so
//  the split adapts to any matrix size and host/device performance ratio.
// -----------------------------------------------------------------------------

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// CUDA kernel: C[M x N] = A[M x K] * B[K x N] with K == N (square problem)
// -----------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

// Block tile: 64x64 output, 16-deep K step, 4x4 outputs per thread, 256 threads.
static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 16;
static constexpr int TM = 4;
static constexpr int TN = 4;

__global__ void dgemmKernel(const double* __restrict__ A, const double* __restrict__ B,
                            double* __restrict__ C, const int M, const int N) {
    __shared__ double As[BK][BM]; // transposed: As[k][m]
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    // Global -> shared load mapping (4 consecutive elements per thread)
    const int aM = tid / 4;
    const int aK = (tid % 4) * 4;
    const int bK = tid / 16;
    const int bN = (tid % 16) * 4;

    // Output mapping: thread (tx, ty) owns rows ty*TM.. and cols tx*TN..
    const int tx = tid % 16;
    const int ty = tid / 16;

    double acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    const int gmA = row0 + aM;
    const int gnB = col0 + bN;

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Load A tile (zero padded at the boundaries)
        {
            const double* p = A + static_cast<size_t>(gmA) * N + k0 + aK;
            double v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[i] = (gmA < M && (k0 + aK + i) < N) ? p[i] : 0.0;
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                As[aK + i][aM] = v[i];
            }
        }
        // Load B tile
        {
            const int gk = k0 + bK;
            const double* p = B + static_cast<size_t>(gk) * N + gnB;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                Bs[bK][bN + i] = (gk < N && (gnB + i) < N) ? p[i] : 0.0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM];
            double b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[kk][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + ty * TM + i;
        if (r >= M) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col0 + tx * TN + j;
            if (c < N) {
                C[static_cast<size_t>(r) * N + c] = acc[i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// CPU DGEMM: cache blocked, 6x8 register blocked micro kernel
// -----------------------------------------------------------------------------

// Blocking parameters (tuned for 512 KiB L2 / 32 MiB L3 per CCX)
static constexpr size_t KC = 512; // K block
static constexpr size_t NC = 512; // N block
static constexpr size_t MR = 6;   // micro kernel rows
static constexpr size_t NR = 8;   // micro kernel columns (2 AVX2 vectors)

// C[MR x NR] += Ap[kc x MR] * Bp[kc x NR], both packed contiguously
static inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                               double* __restrict__ C, const size_t ld, const size_t kc) {
    __m256d c0 = _mm256_loadu_pd(C + 0 * ld);
    __m256d c1 = _mm256_loadu_pd(C + 0 * ld + 4);
    __m256d c2 = _mm256_loadu_pd(C + 1 * ld);
    __m256d c3 = _mm256_loadu_pd(C + 1 * ld + 4);
    __m256d c4 = _mm256_loadu_pd(C + 2 * ld);
    __m256d c5 = _mm256_loadu_pd(C + 2 * ld + 4);
    __m256d c6 = _mm256_loadu_pd(C + 3 * ld);
    __m256d c7 = _mm256_loadu_pd(C + 3 * ld + 4);
    __m256d c8 = _mm256_loadu_pd(C + 4 * ld);
    __m256d c9 = _mm256_loadu_pd(C + 4 * ld + 4);
    __m256d ca = _mm256_loadu_pd(C + 5 * ld);
    __m256d cb = _mm256_loadu_pd(C + 5 * ld + 4);

    for (size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(Bp);
        const __m256d b1 = _mm256_loadu_pd(Bp + 4);
        Bp += NR;
        __m256d a;
        a = _mm256_broadcast_sd(Ap + 0);
        c0 = _mm256_fmadd_pd(a, b0, c0);
        c1 = _mm256_fmadd_pd(a, b1, c1);
        a = _mm256_broadcast_sd(Ap + 1);
        c2 = _mm256_fmadd_pd(a, b0, c2);
        c3 = _mm256_fmadd_pd(a, b1, c3);
        a = _mm256_broadcast_sd(Ap + 2);
        c4 = _mm256_fmadd_pd(a, b0, c4);
        c5 = _mm256_fmadd_pd(a, b1, c5);
        a = _mm256_broadcast_sd(Ap + 3);
        c6 = _mm256_fmadd_pd(a, b0, c6);
        c7 = _mm256_fmadd_pd(a, b1, c7);
        a = _mm256_broadcast_sd(Ap + 4);
        c8 = _mm256_fmadd_pd(a, b0, c8);
        c9 = _mm256_fmadd_pd(a, b1, c9);
        a = _mm256_broadcast_sd(Ap + 5);
        ca = _mm256_fmadd_pd(a, b0, ca);
        cb = _mm256_fmadd_pd(a, b1, cb);
        Ap += MR;
    }

    _mm256_storeu_pd(C + 0 * ld, c0);
    _mm256_storeu_pd(C + 0 * ld + 4, c1);
    _mm256_storeu_pd(C + 1 * ld, c2);
    _mm256_storeu_pd(C + 1 * ld + 4, c3);
    _mm256_storeu_pd(C + 2 * ld, c4);
    _mm256_storeu_pd(C + 2 * ld + 4, c5);
    _mm256_storeu_pd(C + 3 * ld, c6);
    _mm256_storeu_pd(C + 3 * ld + 4, c7);
    _mm256_storeu_pd(C + 4 * ld, c8);
    _mm256_storeu_pd(C + 4 * ld + 4, c9);
    _mm256_storeu_pd(C + 5 * ld, ca);
    _mm256_storeu_pd(C + 5 * ld + 4, cb);
}

// Scalar fall back for partial row / column blocks at the matrix borders
static inline void edgeKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                              double* __restrict__ C, const size_t ld, const size_t kc,
                              const size_t mr, const size_t nr) {
    for (size_t i = 0; i < mr; ++i) {
        for (size_t k = 0; k < kc; ++k) {
            const double a = Ap[k * MR + i];
            const double* __restrict__ b = Bp + k * NR;
            for (size_t j = 0; j < nr; ++j) {
                C[i * ld + j] += a * b[j];
            }
        }
    }
}

// Scratch space for the packed A / B blocks of one CPU worker
struct PackBuffers {
    std::vector<double> a;
    std::vector<double> b;
    PackBuffers(const size_t maxRows)
        : a(((maxRows + MR - 1) / MR) * MR * KC + NR), b((NC + NR) * KC + NR) {}
};

// C[rows x nc] = A[rows x N] * B[N x nc] for the column block [n0, n0 + nc);
// A and C point at the first row of the panel.
static void cpuTile(const double* __restrict__ A, const double* __restrict__ B,
                    double* __restrict__ C, const size_t N, const size_t rows, const size_t n0,
                    const size_t nc, PackBuffers& buf) {
    for (size_t i = 0; i < rows; ++i) {
        memset(C + i * N + n0, 0, nc * sizeof(double));
    }
    double* const Apack = buf.a.data();
    double* const Bpack = buf.b.data();
    const size_t panels = (nc + NR - 1) / NR;

    for (size_t k0 = 0; k0 < N; k0 += KC) {
        const size_t kc = std::min(KC, N - k0);

        // Pack the A block as MR-row panels: Ap[panel][k][i]
        for (size_t i0 = 0; i0 < rows; i0 += MR) {
            const size_t mr = std::min(MR, rows - i0);
            double* dst = Apack + i0 * kc;
            for (size_t k = 0; k < kc; ++k) {
                for (size_t i = 0; i < MR; ++i) {
                    dst[k * MR + i] = (i < mr) ? A[(i0 + i) * N + k0 + k] : 0.0;
                }
            }
        }

        // Pack the B block as NR-column panels: Bp[panel][k][j]
        for (size_t p = 0; p < panels; ++p) {
            double* dst = Bpack + p * kc * NR;
            const size_t n = n0 + p * NR;
            const size_t nr = std::min(NR, n0 + nc - n);
            for (size_t k = 0; k < kc; ++k) {
                const double* src = B + (k0 + k) * N + n;
                for (size_t j = 0; j < NR; ++j) {
                    dst[k * NR + j] = (j < nr) ? src[j] : 0.0;
                }
            }
        }

        for (size_t i0 = 0; i0 < rows; i0 += MR) {
            const size_t mr = std::min(MR, rows - i0);
            for (size_t p = 0; p < panels; ++p) {
                const size_t n = n0 + p * NR;
                const size_t nr = std::min(NR, n0 + nc - n);
                double* Cp = C + i0 * N + n;
                if (mr == MR && nr == NR) {
                    microKernel(Apack + i0 * kc, Bpack + p * kc * NR, Cp, N, kc);
                } else {
                    edgeKernel(Apack + i0 * kc, Bpack + p * kc * NR, Cp, N, kc, mr, nr);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Shared work queue over a 2D tile grid of the local C block
// -----------------------------------------------------------------------------
//
// The local rows are cut into blocks of MC rows, the columns into blocks of NC.
// Tiles are numbered row-block major. GPU workers claim whole row blocks from
// the front of that range (they always process all columns at once), CPU threads
// claim single tiles from the back. Both sides therefore run at their own
// natural granularity and the host/device split adjusts itself automatically.

static constexpr size_t MC_CANDIDATES[] = {768, 512, 384, 256, 128, 64}; // rows per tile
static constexpr size_t TILES_PER_THREAD = 4; // work items each host thread should see
static constexpr size_t GPU_MAX_ROWS = 512; // rows per GPU launch

// Large tiles keep the DRAM traffic for B low, small tiles balance better. Pick
// the largest tile height that still gives every thread something to steal.
static size_t chooseTileRows(const size_t rows, const size_t colBlocks, const int threads) {
    for (const size_t mc : MC_CANDIDATES) {
        const size_t tiles = ((rows + mc - 1) / mc) * colBlocks;
        if (tiles >= static_cast<size_t>(threads) * TILES_PER_THREAD) {
            return mc;
        }
    }
    return MC_CANDIDATES[sizeof(MC_CANDIDATES) / sizeof(size_t) - 1];
}

class WorkQueue {
  public:
    enum class Claim { Got, Reserved, Empty };

    WorkQueue(const size_t rowBlocks, const size_t colBlocks, const int gpuWorkers,
              const int cpuThreads)
        : cols(colBlocks), head(0), tail(rowBlocks * colBlocks), active(gpuWorkers),
          workers(std::max(1, gpuWorkers)), threads(std::max(1, cpuThreads)) {}

    // Claim up to maxRowBlocks complete row blocks from the front (GPU side)
    bool takeRowBlocks(const size_t maxRowBlocks, size_t& firstRowBlock, size_t& rowBlocks) {
        std::lock_guard<std::mutex> lock(mtx);
        const size_t avail = (tail > head) ? (tail - head) / cols : 0;
        if (avail == 0) {
            if (active > 0) {
                --active; // this device is done, release its reservation
            }
            return false;
        }
        size_t want = (avail + 3) / 4; // guided: a quarter of what is left
        want = std::min(std::max<size_t>(want, 1), maxRowBlocks);
        rowBlocks = std::min(want, avail);
        firstRowBlock = head / cols;
        head += rowBlocks * cols;
        return true;
    }

    void reportRowBlocks(const size_t rowBlocks, const double seconds) {
        std::lock_guard<std::mutex> lock(mtx);
        gpuTiles += rowBlocks * cols;
        gpuSeconds += seconds;
    }

    // Claim a single tile from the back (CPU side). Work is held back while the
    // host already owns more than its throughput share of what is left, so that
    // the devices never run dry while host threads are still busy.
    Claim takeTile(size_t& rowBlock, size_t& colBlock) {
        std::lock_guard<std::mutex> lock(mtx);
        if (head >= tail) {
            return Claim::Empty;
        }
        if (active > 0) {
            const size_t open = tail - head;
            if (gpuTiles > 0 && cpuTiles > 0) {
                const double gpuRate = workers * static_cast<double>(gpuTiles) / gpuSeconds;
                const double cpuRate = threads * static_cast<double>(cpuTiles) / cpuSeconds;
                const double share = cpuRate / (cpuRate + gpuRate);
                if (static_cast<double>(inFlight + 1) > share * (open + inFlight)) {
                    return Claim::Reserved;
                }
            } else if (open <= static_cast<size_t>(active) * cols) {
                return Claim::Reserved; // keep one row block per device in reserve
            }
        }
        --tail;
        rowBlock = tail / cols;
        colBlock = tail % cols;
        ++inFlight;
        return Claim::Got;
    }

    void reportTile(const double seconds) {
        std::lock_guard<std::mutex> lock(mtx);
        --inFlight;
        ++cpuTiles;
        cpuSeconds += seconds;
    }

  private:
    std::mutex mtx;
    const size_t cols;
    size_t head;
    size_t tail;
    int active;                // GPU workers that still want work
    const int workers;         // GPU workers of this rank
    const int threads;         // CPU worker threads of this rank
    size_t inFlight = 0;       // tiles claimed by CPU threads but not finished
    size_t cpuTiles = 0;
    size_t gpuTiles = 0;
    double cpuSeconds = 0.0;   // summed over CPU threads
    double gpuSeconds = 0.0;   // summed over GPU workers
};

// Compute the local row block: C[0..rows) = A[0..rows) * B
static void computeLocal(const double* A, const double* B, double* C, const size_t N,
                         const size_t rows, const int firstDevice, const int numDevices) {
    if (rows == 0) {
        return;
    }

    const int maxThreads = omp_get_max_threads();
    const int gpuWorkers = std::min(numDevices, maxThreads);
    const size_t colBlocks = (N + NC - 1) / NC;
    const size_t MC = chooseTileRows(rows, colBlocks, std::max(1, maxThreads - gpuWorkers));
    const size_t rowBlocks = (rows + MC - 1) / MC;
    const size_t gpuMaxBlocks = std::max<size_t>(1, GPU_MAX_ROWS / MC);
    WorkQueue queue(rowBlocks, colBlocks, gpuWorkers, maxThreads - gpuWorkers);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        if (tid < gpuWorkers) {
            // ---- GPU worker -------------------------------------------------
            CUDA_CHECK(cudaSetDevice(firstDevice + tid));
            const size_t chunk = std::min(gpuMaxBlocks * MC, rows);
            double *dA = nullptr, *dB = nullptr, *dC = nullptr;
            CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dA, chunk * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dC, chunk * N * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice));

            size_t firstBlock = 0, blocks = 0;
            while (queue.takeRowBlocks(gpuMaxBlocks, firstBlock, blocks)) {
                const double tb0 = omp_get_wtime();
                const size_t start = firstBlock * MC;
                const size_t count = std::min(blocks * MC, rows - start);
                CUDA_CHECK(cudaMemcpy(dA, A + start * N, count * N * sizeof(double),
                                      cudaMemcpyHostToDevice));
                const dim3 block(256);
                const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                                static_cast<unsigned>((count + BM - 1) / BM));
                dgemmKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(count),
                                             static_cast<int>(N));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(C + start * N, dC, count * N * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                queue.reportRowBlocks(blocks, omp_get_wtime() - tb0);
            }

            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
        }

        // ---- CPU workers (GPU threads join once the queue is drained) --------
        PackBuffers buf(MC);
        size_t rb = 0, cb = 0;
        for (;;) {
            const WorkQueue::Claim claim = queue.takeTile(rb, cb);
            if (claim == WorkQueue::Claim::Empty) {
                break;
            }
            if (claim == WorkQueue::Claim::Reserved) {
                // Only GPU work is left: give the device threads room to run
                const timespec ts{0, 200000};
                nanosleep(&ts, nullptr);
                continue;
            }
            const size_t start = rb * MC;
            const size_t count = std::min(MC, rows - start);
            const size_t n0 = cb * NC;
            const double tile0 = omp_get_wtime();
            cpuTile(A + start * N, B, C + start * N, N, count, n0, std::min(NC, N - n0), buf);
            queue.reportTile(omp_get_wtime() - tile0);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    // Rank placement inside the node
    int localRank = 0, localSize = 1;
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &localSize);
        MPI_Comm_free(&nodeComm);
    }

    // Give every rank on the node a disjoint set of physical cores and run one
    // thread per core. Launchers frequently pin a rank to a single core or leave
    // all local ranks on the same CPUs, which would cripple the OpenMP part;
    // SMT siblings are kept in the same set but do not add threads.
    {
        // Group the online CPUs into physical cores
        const long onlineCpus = sysconf(_SC_NPROCESSORS_ONLN);
        std::vector<std::pair<int, int>> coreIds; // (package, core) in discovery order
        std::vector<std::vector<int>> coreCpus;
        for (long cpu = 0; cpu < onlineCpus && cpu < CPU_SETSIZE; ++cpu) {
            int pkg = 0, core = static_cast<int>(cpu);
            char path[128];
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%ld/topology/physical_package_id", cpu);
            if (FILE* f = fopen(path, "r")) {
                if (fscanf(f, "%d", &pkg) != 1) {
                    pkg = 0;
                }
                fclose(f);
            }
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/topology/core_id", cpu);
            if (FILE* f = fopen(path, "r")) {
                if (fscanf(f, "%d", &core) != 1) {
                    core = static_cast<int>(cpu);
                }
                fclose(f);
            }
            const std::pair<int, int> id(pkg, core);
            const auto it = std::find(coreIds.begin(), coreIds.end(), id);
            if (it == coreIds.end()) {
                coreIds.push_back(id);
                coreCpus.push_back({static_cast<int>(cpu)});
            } else {
                coreCpus[static_cast<size_t>(it - coreIds.begin())].push_back(
                    static_cast<int>(cpu));
            }
        }

        int threads = 0;
        const size_t numCores = coreCpus.size();
        if (numCores > 0) {
            const size_t first = static_cast<size_t>(localRank) * numCores / localSize;
            const size_t last = static_cast<size_t>(localRank + 1) * numCores / localSize;
            cpu_set_t mine;
            CPU_ZERO(&mine);
            for (size_t c = first; c < last; ++c) {
                for (const int cpu : coreCpus[c]) {
                    CPU_SET(cpu, &mine);
                }
            }
            if (CPU_COUNT(&mine) > 0 && sched_setaffinity(0, sizeof(mine), &mine) == 0) {
                threads = static_cast<int>(last - first);
            }
        }
        if (threads <= 0) { // could not apply our own placement, keep what we got
            cpu_set_t mask;
            CPU_ZERO(&mask);
            threads = (sched_getaffinity(0, sizeof(mask), &mask) == 0) ? CPU_COUNT(&mask)
                                                                      : omp_get_max_threads();
        }
        if (getenv("OMP_NUM_THREADS") == nullptr && threads > 0) {
            omp_set_num_threads(threads);
        }
    }

    // Distribute the node's GPUs over the ranks running on that node; a rank that
    // owns several devices drives all of them with one worker thread each.
    int firstDevice = 0, myDevices = 1;
    {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "No CUDA device available on rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (localSize >= deviceCount) {
            firstDevice = localRank % deviceCount;
            myDevices = 1;
        } else {
            firstDevice = localRank * deviceCount / localSize;
            myDevices = (localRank + 1) * deviceCount / localSize - firstDevice;
        }
        for (int d = 0; d < myDevices; ++d) { // establish all contexts up front
            CUDA_CHECK(cudaSetDevice(firstDevice + d));
            CUDA_CHECK(cudaFree(nullptr));
        }
        CUDA_CHECK(cudaSetDevice(firstDevice));
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d, GPUs per rank: %d\n", nranks,
               omp_get_max_threads(), myDevices);
    }

    // Row block owned by this rank
    const size_t base = N / static_cast<size_t>(nranks);
    const size_t rem = N % static_cast<size_t>(nranks);
    const size_t myRows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t myFirst = base * static_cast<size_t>(rank) +
                           std::min(static_cast<size_t>(rank), rem);

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rank == 0 ? N * N : myRows * N);

    // Initialize matrices (deterministic, so every rank builds its own copy)
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Rank 0 computes directly into its slot of the full result matrix
    double* myC = C.data() + (rank == 0 ? myFirst * N : 0);
    computeLocal(A.data() + myFirst * N, B.data(), myC, N, myRows, firstDevice, myDevices);

    // Collect the full result on rank 0
    constexpr size_t MAX_MSG = 1ull << 27; // stay well below INT_MAX elements per message
    if (rank == 0) {
        for (int r = 1; r < nranks; ++r) {
            const size_t rows = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t first = base * static_cast<size_t>(r) +
                                 std::min(static_cast<size_t>(r), rem);
            size_t remaining = rows * N;
            double* dst = C.data() + first * N;
            while (remaining > 0) {
                const size_t n = std::min(remaining, MAX_MSG);
                MPI_Recv(dst, static_cast<int>(n), MPI_DOUBLE, r, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                dst += n;
                remaining -= n;
            }
        }
    } else {
        size_t remaining = myRows * N;
        const double* src = C.data();
        while (remaining > 0) {
            const size_t n = std::min(remaining, MAX_MSG);
            MPI_Send(src, static_cast<int>(n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            src += n;
            remaining -= n;
        }
    }

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
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);

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
