// Hybrid MPI + OpenMP + CUDA dense matrix multiplication benchmark.
//
// Decomposition:
//   * MPI    : the rows of A (and C) are distributed in contiguous blocks over
//              the ranks; every rank regenerates the full B matrix locally from
//              the deterministic initialisation formula, so no data has to be
//              communicated before the multiplication.
//   * CUDA   : each rank is bound to one GPU (local rank modulo device count)
//              that holds B and streams row chunks of A through a register
//              tiled FP64 kernel.
//   * OpenMP : the rank's CPU cores compute 2-D tiles of C with a packed,
//              register-blocked FMA kernel while one thread drives the GPU.
//              The GPU claims large row blocks from the top of the local row
//              range and the CPU threads claim tiles from the bottom, so the
//              split between GPU and CPU adapts to their relative speed.
//
// All code paths accumulate the dot products in the same k order using fused
// multiply-adds, so the result is bit-identical regardless of which device
// computed a given element.

#include <algorithm>
#include <chrono>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>

#if defined(__GNUC__) || defined(__clang__)
#define MATMUL_HAVE_VECTOR_EXT 1
#else
#define MATMUL_HAVE_VECTOR_EXT 0
#endif

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

#define MPI_CHECK(call)                                                                  \
    do {                                                                                 \
        const int err__ = (call);                                                        \
        if (err__ != MPI_SUCCESS) {                                                      \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__,     \
                    __LINE__);                                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Matrix initialisation
// ---------------------------------------------------------------------------

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                                       const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialise rows [rowBegin, rowBegin + numRows) of an N x N matrix into a
// buffer that holds exactly those rows.
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t numRows) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < numRows; ++r) {
        const size_t i = rowBegin + r;
        double* row = mat + r * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixRows(mat.data(), N, 0, N);
}

// ---------------------------------------------------------------------------
// CUDA kernel: C[rows x N] = A[rows x N] * B[N x N]
// 64x64 output tile per block, 16x16 threads, 4x4 outputs per thread.
// ---------------------------------------------------------------------------
namespace gpu {

constexpr int BM = 64;   // rows of C per block
constexpr int BN = 64;   // cols of C per block
constexpr int BK = 16;   // k depth per shared-memory stage
constexpr int TX = 16;   // threads in x
constexpr int TY = 16;   // threads in y
constexpr int TM = BM / TY;  // 4 rows per thread
constexpr int TN = BN / TX;  // 4 cols per thread

__global__ void __launch_bounds__(TX * TY)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
             const size_t N, const size_t rows) {
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;

    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    double acc[TM][TN];
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] = 0.0;

    const size_t numTiles = (N + BK - 1) / BK;
    for (size_t t = 0; t < numTiles; ++t) {
        const size_t kBase = t * BK;

        // Load A tile (BM x BK): 1024 elements, 4 per thread.
#pragma unroll
        for (int e = 0; e < (BM * BK) / (TX * TY); ++e) {
            const int idx = tid + e * TX * TY;
            const int r = idx / BK;
            const int k = idx % BK;
            const size_t gr = rowBase + r;
            const size_t gk = kBase + k;
            As[k][r] = (gr < rows && gk < N) ? A[gr * N + gk] : 0.0;
        }
        // Load B tile (BK x BN): 1024 elements, 4 per thread, coalesced on columns.
#pragma unroll
        for (int e = 0; e < (BK * BN) / (TX * TY); ++e) {
            const int idx = tid + e * TX * TY;
            const int k = idx / BN;
            const int c = idx % BN;
            const size_t gk = kBase + k;
            const size_t gc = colBase + c;
            Bs[k][c] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM];
            double b[TN];
#pragma unroll
            for (int m = 0; m < TM; ++m) a[m] = As[k][ty + m * TY];
#pragma unroll
            for (int n = 0; n < TN; ++n) b[n] = Bs[k][tx + n * TX];
#pragma unroll
            for (int m = 0; m < TM; ++m)
#pragma unroll
                for (int n = 0; n < TN; ++n) acc[m][n] = fma(a[m], b[n], acc[m][n]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        const size_t gr = rowBase + ty + m * TY;
        if (gr >= rows) continue;
#pragma unroll
        for (int n = 0; n < TN; ++n) {
            const size_t gc = colBase + tx + n * TX;
            if (gc < N) C[gr * N + gc] = acc[m][n];
        }
    }
}

}  // namespace gpu

// ---------------------------------------------------------------------------
// CPU kernel: C[i0:i1, j0:j1] = A[i0:i1, :] * B[:, j0:j1]
// GotoBLAS-style: A is packed into k-major MR-row panels and B into a
// contiguous column block (per-thread buffers; this also avoids cache-set
// aliasing for power-of-two N).  A 6x8 register-blocked micro-kernel
// accumulates with FMAs; the accumulation order over k is sequential, matching
// the GPU kernel bit for bit.
// ---------------------------------------------------------------------------
namespace cpu {

constexpr size_t MR = 6;           // micro-kernel rows
constexpr size_t NR = 8;           // micro-kernel cols
constexpr size_t TILE_ROWS = 240;  // rows per work tile (multiple of MR)
constexpr size_t TILE_COLS = 128;  // cols per work tile (multiple of NR)
constexpr size_t KBLK = 256;       // k depth per packed block

struct Packs {
    std::vector<double> Ap;  // ceil(TILE_ROWS/MR) panels of kc x MR (k-major)
    std::vector<double> Bp;  // kc x TILE_COLS, row stride TILE_COLS
    Packs() : Ap(((TILE_ROWS + MR - 1) / MR) * MR * KBLK), Bp(KBLK * TILE_COLS) {}
};

#if MATMUL_HAVE_VECTOR_EXT
typedef double v4d __attribute__((vector_size(32)));
typedef double v4d_u __attribute__((vector_size(32), aligned(8), may_alias));
static inline v4d loadu(const double* p) { return *reinterpret_cast<const v4d_u*>(p); }
static inline void storeu(double* p, v4d v) { *reinterpret_cast<v4d_u*>(p) = v; }
#endif

// MR x NR block of C (row stride N) += Ap[kc x MR panel] * Bp[kc x NR]
// (a*b + c on vector types is contracted to an FMA by the host compiler).
static inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                               double* __restrict__ C, const size_t N, const size_t kc) {
#if MATMUL_HAVE_VECTOR_EXT
    v4d acc[MR][NR / 4];
    for (size_t m = 0; m < MR; ++m)
        for (size_t n = 0; n < NR / 4; ++n) acc[m][n] = loadu(C + m * N + n * 4);
    for (size_t k = 0; k < kc; ++k) {
        const double* b = Bp + k * TILE_COLS;
        const double* a = Ap + k * MR;
        v4d bv[NR / 4];
        for (size_t n = 0; n < NR / 4; ++n) bv[n] = loadu(b + n * 4);
        for (size_t m = 0; m < MR; ++m) {
            const v4d av = {a[m], a[m], a[m], a[m]};
            for (size_t n = 0; n < NR / 4; ++n) acc[m][n] = av * bv[n] + acc[m][n];
        }
    }
    for (size_t m = 0; m < MR; ++m)
        for (size_t n = 0; n < NR / 4; ++n) storeu(C + m * N + n * 4, acc[m][n]);
#else
    double acc[MR][NR];
    for (size_t m = 0; m < MR; ++m)
        for (size_t n = 0; n < NR; ++n) acc[m][n] = C[m * N + n];
    for (size_t k = 0; k < kc; ++k) {
        const double* b = Bp + k * TILE_COLS;
        const double* a = Ap + k * MR;
        for (size_t m = 0; m < MR; ++m)
            for (size_t n = 0; n < NR; ++n) acc[m][n] = std::fma(a[m], b[n], acc[m][n]);
    }
    for (size_t m = 0; m < MR; ++m)
        for (size_t n = 0; n < NR; ++n) C[m * N + n] = acc[m][n];
#endif
}

// Generic kernel for `rows` (<= MR) rows of a panel and an arbitrary column count.
static void edgeKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                       double* __restrict__ C, const size_t N, const size_t kc, const size_t rows,
                       const size_t cols) {
    for (size_t k = 0; k < kc; ++k) {
        const double* b = Bp + k * TILE_COLS;
        for (size_t m = 0; m < rows; ++m) {
            const double a = Ap[k * MR + m];
            double* c = C + m * N;
            for (size_t j = 0; j < cols; ++j) c[j] = std::fma(a, b[j], c[j]);
        }
    }
}

static void multiplyTile(const double* __restrict__ A, const double* __restrict__ B,
                         double* __restrict__ C, const size_t N, const size_t i0, const size_t i1,
                         const size_t j0, const size_t j1, Packs& p) {
    const size_t rows = i1 - i0;
    const size_t cols = j1 - j0;
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j) C[i * N + j] = 0.0;

    const size_t numGroups = (rows + MR - 1) / MR;
    const size_t cFull = (cols / NR) * NR;
    double* Ap = p.Ap.data();
    double* Bp = p.Bp.data();

    for (size_t k0 = 0; k0 < N; k0 += KBLK) {
        const size_t kc = std::min(KBLK, N - k0);

        // Pack A rows into k-major MR-row panels (zero padded).
        for (size_t g = 0; g < numGroups; ++g) {
            double* panel = Ap + g * kc * MR;
            const size_t gRows = std::min(MR, rows - g * MR);
            for (size_t m = 0; m < gRows; ++m) {
                const double* srcA = A + (i0 + g * MR + m) * N + k0;
                for (size_t k = 0; k < kc; ++k) panel[k * MR + m] = srcA[k];
            }
            for (size_t m = gRows; m < MR; ++m)
                for (size_t k = 0; k < kc; ++k) panel[k * MR + m] = 0.0;
        }
        // Pack the B block.
        for (size_t k = 0; k < kc; ++k) {
            const double* srcB = B + (k0 + k) * N + j0;
            double* dst = Bp + k * TILE_COLS;
            for (size_t j = 0; j < cols; ++j) dst[j] = srcB[j];
        }

        for (size_t g = 0; g < numGroups; ++g) {
            const double* panel = Ap + g * kc * MR;
            double* crow = C + (i0 + g * MR) * N + j0;
            const size_t gRows = std::min(MR, rows - g * MR);
            if (gRows == MR) {
                for (size_t c = 0; c < cFull; c += NR) microKernel(panel, Bp + c, crow + c, N, kc);
                if (cFull < cols)
                    edgeKernel(panel, Bp + cFull, crow + cFull, N, kc, MR, cols - cFull);
            } else {
                edgeKernel(panel, Bp, crow, N, kc, gRows, cols);
            }
        }
    }
}

}  // namespace cpu

// ---------------------------------------------------------------------------
// Per-rank hybrid multiplication: C_local = A_local * B
// A_local / C_local hold `numRows` rows of N columns.
// ---------------------------------------------------------------------------
struct GpuContext {
    static constexpr int NUM_STREAMS = 2;
    double* dB = nullptr;
    double* dA[NUM_STREAMS] = {nullptr, nullptr};
    double* dC[NUM_STREAMS] = {nullptr, nullptr};
    cudaStream_t streams[NUM_STREAMS] = {nullptr, nullptr};
    size_t maxRows = 0;  // capacity of dA / dC in rows
};

// Work pool over the local rows.  Rows are grouped into chunks of
// cpu::TILE_ROWS; the CPU threads claim (chunk, column tile) items in ascending
// order, the GPU claims whole chunks from the top.  The two never overlap.
class WorkPool {
public:
    WorkPool(size_t numRows, size_t N, size_t maxGpuChunks)
        : numChunks_((numRows + cpu::TILE_ROWS - 1) / cpu::TILE_ROWS),
          numColTiles_((N + cpu::TILE_COLS - 1) / cpu::TILE_COLS),
          gpuLow_(numChunks_),
          maxGpuChunks_(std::max<size_t>(1, maxGpuChunks)) {}

    // Returns false when no CPU work is left.
    bool claimCpuTile(size_t& chunk, size_t& colTile) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t c = cpuNextTile_ / numColTiles_;
        if (c >= gpuLow_) return false;
        chunk = c;
        colTile = cpuNextTile_ % numColTiles_;
        ++cpuNextTile_;
        return true;
    }

    // Claims a block of chunks [first, first + count) for the GPU.
    bool claimGpuChunks(size_t& first, size_t& count) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t cpuBoundary = (cpuNextTile_ + numColTiles_ - 1) / numColTiles_;
        if (gpuLow_ <= cpuBoundary) return false;
        const size_t avail = gpuLow_ - cpuBoundary;
        // Take a shrinking fraction of what is left so the GPU's final block
        // (its load-imbalance tail) stays small.
        size_t take = (avail + 7) / 8;
        take = std::min(take, maxGpuChunks_);
        take = std::max<size_t>(take, 1);
        gpuLow_ -= take;
        first = gpuLow_;
        count = take;
        return true;
    }

private:
    std::mutex mutex_;
    const size_t numChunks_;
    const size_t numColTiles_;
    size_t cpuNextTile_ = 0;
    size_t gpuLow_;
    const size_t maxGpuChunks_;
};

static void hybridMultiply(const double* A, const double* B, double* C, const size_t N,
                           const size_t numRows, GpuContext& ctx) {
    if (numRows == 0) return;

    WorkPool pool(numRows, N, ctx.maxRows / cpu::TILE_ROWS);

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            // GPU driver: streams row blocks through the device, double buffered.
            int s = 0;
            bool busy[GpuContext::NUM_STREAMS] = {false, false};
            size_t first = 0, count = 0;
            while (pool.claimGpuChunks(first, count)) {
                const size_t r0 = first * cpu::TILE_ROWS;
                const size_t r1 = std::min(numRows, (first + count) * cpu::TILE_ROWS);
                const size_t rows = r1 - r0;

                cudaStream_t st = ctx.streams[s];
                if (busy[s]) CUDA_CHECK(cudaStreamSynchronize(st));
                CUDA_CHECK(cudaMemcpyAsync(ctx.dA[s], A + r0 * N, rows * N * sizeof(double),
                                           cudaMemcpyHostToDevice, st));
                const dim3 block(gpu::TX, gpu::TY);
                const dim3 grid(static_cast<unsigned>((N + gpu::BN - 1) / gpu::BN),
                                static_cast<unsigned>((rows + gpu::BM - 1) / gpu::BM));
                gpu::matmulKernel<<<grid, block, 0, st>>>(ctx.dA[s], ctx.dB, ctx.dC[s], N, rows);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(C + r0 * N, ctx.dC[s], rows * N * sizeof(double),
                                           cudaMemcpyDeviceToHost, st));
                busy[s] = true;
                s = (s + 1) % GpuContext::NUM_STREAMS;
            }
            for (int i = 0; i < GpuContext::NUM_STREAMS; ++i) {
                if (busy[i]) CUDA_CHECK(cudaStreamSynchronize(ctx.streams[i]));
            }
        }

        // CPU workers (the GPU driver joins once the GPU has no work left).
        cpu::Packs packs;
        size_t chunk = 0, colTile = 0;
        while (pool.claimCpuTile(chunk, colTile)) {
            const size_t i0 = chunk * cpu::TILE_ROWS;
            const size_t i1 = std::min(numRows, i0 + cpu::TILE_ROWS);
            const size_t j0 = colTile * cpu::TILE_COLS;
            const size_t j1 = std::min(N, j0 + cpu::TILE_COLS);
            cpu::multiplyTile(A, B, C, N, i0, i1, j0, j1, packs);
        }
    }
}

// ---------------------------------------------------------------------------
// Validation (rank 0 only, on the gathered full matrices)
// ---------------------------------------------------------------------------

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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

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
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", size, omp_get_max_threads());
    }

    // ---- GPU selection: local rank modulo number of visible devices ----
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    // ---- Row distribution over ranks ----
    const size_t rowsPerRank = N / size;
    const size_t remainder = N % size;
    const size_t rowBegin = rank * rowsPerRank + (static_cast<size_t>(rank) < remainder
                                                      ? static_cast<size_t>(rank) : remainder);
    const size_t numRows = rowsPerRank + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Allocate matrices (A and C as local row blocks, B replicated), pinned so
    // that the GPU transfers can run asynchronously.
    double* Aloc = nullptr;
    double* Cloc = nullptr;
    std::vector<double> B(N * N);
    const size_t localElems = numRows * N;
    if (localElems > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&Aloc), localElems * sizeof(double),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&Cloc), localElems * sizeof(double),
                                 cudaHostAllocDefault));
    }
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(Aloc, N, rowBegin, numRows);
    initMatrix(B, N);

    // ---- GPU resources: B resident, double-buffered A/C row chunks ----
    GpuContext ctx;
    {
        // Device buffers hold a multiple of cpu::TILE_ROWS rows (the GPU claims
        // work in that unit), capped by a memory budget and the local row count.
        constexpr size_t TARGET_CHUNK_BYTES = size_t(256) << 20;  // 256 MiB per A/C buffer
        size_t chunks = TARGET_CHUNK_BYTES / (cpu::TILE_ROWS * N * sizeof(double));
        chunks = std::clamp<size_t>(chunks, 1, 8);
        const size_t localChunks = (numRows + cpu::TILE_ROWS - 1) / cpu::TILE_ROWS;
        if (localChunks > 0) chunks = std::min(chunks, localChunks);
        ctx.maxRows = chunks * cpu::TILE_ROWS;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.dB), N * N * sizeof(double)));
        for (int s = 0; s < GpuContext::NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.dA[s]),
                                  ctx.maxRows * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.dC[s]),
                                  ctx.maxRows * N * sizeof(double)));
            CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.streams[s], cudaStreamNonBlocking));
        }
        CUDA_CHECK(cudaMemcpy(ctx.dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();
    
    hybridMultiply(Aloc, B.data(), Cloc, N, numRows, ctx);
    
    auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    auto duration = std::chrono::milliseconds(static_cast<long>(maxSeconds * 1000.0));
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Gather the full C on rank 0 (row-contiguous blocks).
        MPI_Datatype rowType;
        MPI_CHECK(MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType));
        MPI_CHECK(MPI_Type_commit(&rowType));
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t rr = static_cast<size_t>(r);
            counts[r] = static_cast<int>(rowsPerRank + (rr < remainder ? 1 : 0));
            displs[r] = static_cast<int>(rr * rowsPerRank + (rr < remainder ? rr : remainder));
        }
        std::vector<double> C;
        if (rank == 0) C.resize(N * N);
        MPI_CHECK(MPI_Gatherv(Cloc, static_cast<int>(numRows), rowType, C.data(), counts.data(),
                              displs.data(), rowType, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Type_free(&rowType));

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(C, "MatrixC");
            }
    
            // Validation
            if (validate) {
                printf("Validating result...\n");
                std::vector<double> A(N * N);
                initMatrix(A, N);
                bool valid = validateResult(A, B, C, N);
        
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    // ---- Cleanup ----
    for (int s = 0; s < GpuContext::NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamDestroy(ctx.streams[s]));
        CUDA_CHECK(cudaFree(ctx.dA[s]));
        CUDA_CHECK(cudaFree(ctx.dC[s]));
    }
    CUDA_CHECK(cudaFree(ctx.dB));
    if (Aloc) CUDA_CHECK(cudaFreeHost(Aloc));
    if (Cloc) CUDA_CHECK(cudaFreeHost(Cloc));

    MPI_Finalize();
    return exitCode;
}
