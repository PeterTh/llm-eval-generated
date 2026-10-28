// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//   MPI     Rows of A and C are block-distributed across ranks. Every rank
//           generates its own rows of A and the full B (the generator is a
//           pure function), so no input communication is required.
//   CUDA    Each rank drives one GPU (local rank modulo device count).
//   OpenMP  One team thread drives the GPU; the remaining threads compute on
//           the CPU with a register-blocked, packed-B kernel.
//           Both devices claim 64x64 output tiles from a shared atomic counter,
//           so the faster device automatically receives more tiles.
//
// Numerical semantics: every element C[i][j] is accumulated over k = 0 .. N-1 in ascending
// order, with each product rounded before it is added (acc = acc + A[i][k] * B[k][j]).
// The final term of an odd-length dot product is fused (acc = fma(...)), reproducing the
// rounding sequence of the original serial code as compiled. Both CPU and GPU follow this
// exact sequence, so results are bitwise identical to the original for any decomposition.
// Floating-point contraction is disabled for the CPU build (-ffp-contract=off) for this reason.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr size_t TILE = 64;        // output tile edge (rows and columns)
constexpr size_t MR = 4;           // CPU micro-kernel rows
constexpr size_t NR = 8;           // CPU micro-kernel columns (packed B panel width)
constexpr size_t GPU_MIN_BATCH = 16;   // smallest tile batch per GPU launch (tail balance)
constexpr size_t GPU_MAX_BATCH = 256;  // largest tile batch per GPU launch (device saturation)
constexpr size_t GPU_BK = 16;      // GPU k-step (shared-memory tile depth)
constexpr int GPU_THREADS = 256;   // 16x16 threads, each computing a 4x4 sub-tile

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fill nRows rows (starting at global row firstRow) of an N-column matrix.
void initRows(double* mat, const size_t firstRow, const size_t nRows, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

// Pack B into column panels of width NR: Bp[(p*N + k)*NR + c] = B[k][p*NR + c] (zero padded).
void packB(const double* B, double* Bp, const size_t N) {
    const size_t npan = (N + NR - 1) / NR;
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < npan; ++p) {
        for (size_t k = 0; k < N; ++k) {
            for (size_t c = 0; c < NR; ++c) {
                const size_t col = p * NR + c;
                Bp[(p * N + k) * NR + c] = (col < N) ? B[k * N + col] : 0.0;
            }
        }
    }
}

// Shared, read-only state for one rank plus the dynamic tile counter.
struct RankContext {
    size_t N;           // matrix dimension
    size_t rowsLocal;   // rows of A/C owned by this rank
    size_t rowsPad;     // rowsLocal rounded up to TILE (zero padded A rows)
    size_t CG;          // tile columns: ceil(N / TILE)
    size_t NPAN;        // packed B panels: ceil(N / NR)
    size_t numTiles;    // ceil(rowsLocal / TILE) * CG
    const double* A;    // rowsPad x N, row-major
    const double* B;    // N x N, row-major
    const double* Bp;   // packed B
    double* C;          // rowsLocal x N, row-major (pinned, mapped for GPU writes)
    std::atomic<size_t> nextTile{0};
};

// CPU micro-kernel: c[MR][NR] (row stride TILE) += A(MR rows, k-range) * packed B panel.
// Products are rounded and added in ascending k order (see the semantics note above).
// When fuseLast is set, the final k of the range is applied with a fused multiply-add.
// The accumulator block is kept in vector registers; each lane is independent.
inline void microKernel(const double* __restrict a, const size_t lda, const double* __restrict bp,
                        const size_t kLen, double* __restrict c, const bool fuseLast) noexcept {
    double acc[MR][NR];
    for (size_t r = 0; r < MR; ++r) {
        for (size_t j = 0; j < NR; ++j) {
            acc[r][j] = c[r * TILE + j];
        }
    }
    const size_t kPlain = fuseLast ? kLen - 1 : kLen;
    for (size_t k = 0; k < kPlain; ++k) {
        const double* __restrict b = bp + k * NR;
        for (size_t r = 0; r < MR; ++r) {
            const double av = a[r * lda + k];
            #pragma omp simd
            for (size_t j = 0; j < NR; ++j) {
                acc[r][j] = acc[r][j] + av * b[j];
            }
        }
    }
    if (fuseLast) {
        const size_t k = kLen - 1;
        const double* __restrict b = bp + k * NR;
        for (size_t r = 0; r < MR; ++r) {
            const double av = a[r * lda + k];
            #pragma omp simd
            for (size_t j = 0; j < NR; ++j) {
                acc[r][j] = std::fma(av, b[j], acc[r][j]);
            }
        }
    }
    for (size_t r = 0; r < MR; ++r) {
        for (size_t j = 0; j < NR; ++j) {
            c[r * TILE + j] = acc[r][j];
        }
    }
}

// Computes one TILE x TILE output tile. The k dimension is blocked (KC) to keep the
// A and B working sets in cache; partial sums persist in `buf` between blocks, so each
// element still accumulates k = 0..N-1 in order.
void cpuTile(const RankContext& ctx, const size_t t) noexcept {
    constexpr size_t KC = 256;
    const size_t N = ctx.N;
    const size_t rowBase = (t / ctx.CG) * TILE;
    const size_t pBegin = (t % ctx.CG) * (TILE / NR);
    const size_t pEnd = std::min(pBegin + TILE / NR, ctx.NPAN);

    double buf[TILE][TILE] = {};

    for (size_t k0 = 0; k0 < N; k0 += KC) {
        const size_t kLen = std::min(KC, N - k0);
        const bool fuseLast = (N % 2 == 1) && (k0 + kLen == N);
        for (size_t mt = 0; mt < TILE; mt += MR) {
            const size_t r0 = rowBase + mt;
            if (r0 >= ctx.rowsLocal) break;

            for (size_t p = pBegin; p < pEnd; ++p) {
                const size_t cb = (p - pBegin) * NR;
                microKernel(ctx.A + r0 * N + k0, N, ctx.Bp + (p * N + k0) * NR, kLen,
                            &buf[mt][cb], fuseLast);
            }
        }
    }

    const size_t rows = std::min(TILE, ctx.rowsLocal - rowBase);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < TILE; ++c) {
            const size_t col = pBegin * NR + c;
            if (col < N) {
                ctx.C[(rowBase + r) * N + col] = buf[r][c];
            }
        }
    }
}

// GPU kernel: one block per 64x64 output tile, 16x16 threads, each thread owns a 4x4
// sub-tile. The k loop advances in BK-deep shared-memory tiles in ascending order, so each
// accumulator sees the same rounding sequence as the CPU kernel (see the semantics note).
__global__ void gemmTilesKernel(const double* __restrict__ A, const double* __restrict__ B,
                                double* __restrict__ C, const size_t firstTile,
                                const size_t CG, const size_t rowsLocal, const size_t N) {
    __shared__ double As[GPU_BK][TILE];   // As[kk][row]
    __shared__ double Bs[GPU_BK][TILE];   // Bs[kk][col]

    const size_t t = firstTile + blockIdx.x;
    const size_t rowBase = (t / CG) * TILE;
    const size_t colBase = (t % CG) * TILE;
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;

    double acc[4][4] = {};

    for (size_t k0 = 0; k0 < N; k0 += GPU_BK) {
        const int kLen = (int)((N - k0 < GPU_BK) ? (N - k0) : GPU_BK);
        const bool fuseLast = (N % 2 == 1) && (k0 + (size_t)kLen == N);

        // Load A tile (64 x 16) and B tile (16 x 64): 1024 elements each, 4 per thread.
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int idx = tid + m * GPU_THREADS;
            const int r = idx / (int)GPU_BK;
            const int kk = idx % (int)GPU_BK;
            double av = 0.0;
            if (kk < kLen && rowBase + r < rowsLocal) {
                av = A[(rowBase + r) * N + k0 + kk];
            }
            As[kk][r] = av;

            const int kb = idx / (int)TILE;
            const int c = idx % (int)TILE;
            double bv = 0.0;
            if (kb < kLen && colBase + c < N) {
                bv = B[(k0 + kb) * N + colBase + c];
            }
            Bs[kb][c] = bv;
        }
        __syncthreads();

        for (int kk = 0; kk < kLen; ++kk) {
            double a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[kk][ty + 16 * i];   // interleaved for conflict-free shared reads
                b[i] = Bs[kk][tx + 16 * i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // Non-contracted multiply and add; the last term of odd N is fused.
                    acc[i][j] = (fuseLast && kk == kLen - 1) ? fma(a[i], b[j], acc[i][j])
                                                             : __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t row = rowBase + ty + 16 * i;
            const size_t col = colBase + tx + 16 * j;
            if (row < rowsLocal && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// Runs on OpenMP thread 0: drives this rank's GPU, claiming batches of tiles.
void gpuWorker(RankContext& ctx, const int device) {
    CHECK_CUDA(cudaSetDevice(device));
    if (ctx.numTiles == 0) return;

    const size_t N = ctx.N;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, ctx.rowsPad * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dB, N * N * sizeof(double)));
    CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dC), ctx.C, 0));

    CHECK_CUDA(cudaMemcpy(dA, ctx.A, ctx.rowsPad * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, ctx.B, N * N * sizeof(double), cudaMemcpyHostToDevice));

    for (;;) {
        // Guided scheduling: large batches while work remains, shrinking towards the end.
        size_t t0 = ctx.nextTile.load(std::memory_order_relaxed);
        size_t count = 0;
        do {
            if (t0 >= ctx.numTiles) break;
            const size_t remaining = ctx.numTiles - t0;
            count = std::min(remaining, std::clamp(remaining / 8, GPU_MIN_BATCH, GPU_MAX_BATCH));
        } while (!ctx.nextTile.compare_exchange_weak(t0, t0 + count));
        if (t0 >= ctx.numTiles) break;

        gemmTilesKernel<<<static_cast<unsigned int>(count), GPU_THREADS>>>(
            dA, dB, dC, t0, ctx.CG, ctx.rowsLocal, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
}

// Runs on the remaining OpenMP threads: claims tiles and computes them on the CPU.
void cpuWorker(RankContext& ctx) {
    for (;;) {
        const size_t t = ctx.nextTile.fetch_add(1);
        if (t >= ctx.numTiles) break;
        cpuTile(ctx, t);
    }
}

// Validation on rank 0. `actual` holds the 5x5 sampled C entries (row-major over pi, pj).
bool validateResult(const double* B, const double* actual, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }

            const double actualVal = actual[pi * 5 + pj];
            const double relError = std::abs((actualVal - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actualVal, relError);
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

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Node-local rank selects the GPU
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaFree(nullptr));  // create the CUDA context before timing

    // Split hardware threads among node-local ranks unless the user chose a count
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI ranks, %d OpenMP threads/rank, CUDA device %s per rank\n",
               nranks, omp_get_max_threads(), prop.name);
    }

    // Row partition: rank r owns rows [rowStart, rowStart + rowsLocal)
    const size_t baseRows = N / nranks;
    const size_t remRows = N % nranks;
    const size_t rowsLocal = baseRows + ((size_t)rank < remRows ? 1 : 0);
    const size_t rowStart = (size_t)rank * baseRows + std::min<size_t>(rank, remRows);

    RankContext ctx{};
    ctx.N = N;
    ctx.rowsLocal = rowsLocal;
    ctx.CG = (N + TILE - 1) / TILE;
    ctx.rowsPad = ((rowsLocal + TILE - 1) / TILE) * TILE;
    ctx.NPAN = (N + NR - 1) / NR;
    ctx.numTiles = (ctx.rowsPad / TILE) * ctx.CG;

    // Host buffers: pinned for fast PCIe transfers; C is mapped so the GPU writes it directly
    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&A),
                              std::max<size_t>(1, ctx.rowsPad * N) * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&B), std::max<size_t>(1, N * N) * sizeof(double)));
    CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&C),
                             std::max<size_t>(1, rowsLocal * N) * sizeof(double), cudaHostAllocMapped));
    std::vector<double> Bp(ctx.NPAN * N * NR);

    if (root) printf("Initializing matrices...\n");
    initRows(A, rowStart, rowsLocal, N);
    if (ctx.rowsPad > rowsLocal) {
        std::memset(A + rowsLocal * N, 0, (ctx.rowsPad - rowsLocal) * N * sizeof(double));
    }
    initRows(B, 0, N, N);
    packB(B, Bp.data(), N);

    ctx.A = A;
    ctx.B = B;
    ctx.Bp = Bp.data();
    ctx.C = C;

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            gpuWorker(ctx, device);
        } else {
            cpuWorker(ctx);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full C on rank 0 for result output
    if (printResults) {
        std::vector<double> fullC;
        std::vector<int> counts;
        std::vector<int> displs;
        if (root) {
            fullC.resize(N * N);
            counts.resize(nranks);
            displs.resize(nranks);
            for (int r = 0; r < nranks; ++r) {
                const size_t rows = baseRows + ((size_t)r < remRows ? 1 : 0);
                const size_t start_r = (size_t)r * baseRows + std::min<size_t>(r, remRows);
                counts[r] = (int)(rows * N);
                displs[r] = (int)(start_r * N);
            }
        }
        MPI_Gatherv(C, (int)(rowsLocal * N), MPI_DOUBLE, fullC.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) {
            print_results(fullC, "MatrixC");
        }
    }

    int exitCode = 0;

    // Validation: each rank contributes the sampled C entries it owns
    if (validate) {
        constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
        double localSamples[25] = {};
        double samples[25] = {};
        for (size_t pi = 0; pi < 5; ++pi) {
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = checkPoints[pi] % N;
                const size_t j = checkPoints[pj] % N;
                if (i >= rowStart && i < rowStart + rowsLocal) {
                    localSamples[pi * 5 + pj] = C[(i - rowStart) * N + j];
                }
            }
        }
        MPI_Reduce(localSamples, samples, 25, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (root) {
            printf("Validating result...\n");
            const bool valid = validateResult(B, samples, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));
    CHECK_CUDA(cudaFreeHost(C));
    MPI_Finalize();
    return exitCode;
}
