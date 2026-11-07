// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//   * MPI   : C = A * B is partitioned by contiguous row blocks of A / C across
//             ranks. Every rank owns its A row block and a full copy of B
//             (both generated locally from the deterministic generator, so no
//             broadcast is needed). Finished C chunks are streamed to rank 0
//             with non-blocking sends while the GPUs are still computing.
//   * CUDA  : each rank drives one GPU (local rank -> device, round-robin).
//             The local row block is processed in chunks on two streams so
//             host<->device transfers overlap with the tiled DGEMM kernel.
//   * OpenMP: multi-threaded matrix initialisation, result packing and
//             host-side validation on every rank.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err__));                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

#define MPI_CHECK(call)                                                           \
    do {                                                                          \
        int err__ = (call);                                                       \
        if (err__ != MPI_SUCCESS) {                                               \
            fprintf(stderr, "MPI error %s at %s:%d (code %d)\n", #call, __FILE__, \
                    __LINE__, err__);                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Deterministic matrix generator (identical to the original serial code)
// ---------------------------------------------------------------------------
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialise rows [rowBegin, rowBegin + numRows) of an N x N matrix into `mat`
// (which holds numRows * N elements). OpenMP parallel over rows.
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

// ---------------------------------------------------------------------------
// CUDA tiled DGEMM kernel: C[M x N] = A[M x K] * B[K x N] (row-major)
// Block tile BM x BN, K-step BK, each thread computes a TM x TN micro tile.
// Shared memory is double buffered and global loads are prefetched into
// registers so that loads of tile t+1 overlap the FMAs of tile t.
// ---------------------------------------------------------------------------
namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_X = BN / TN;  // 16
constexpr int THREADS_Y = BM / TM;  // 16
constexpr int NTHREADS = THREADS_X * THREADS_Y;  // 256
constexpr int A_LOADS_PER_THREAD = (BM * BK) / NTHREADS;  // 4
constexpr int B_LOADS_PER_THREAD = (BK * BN) / NTHREADS;  // 4

static_assert(A_LOADS_PER_THREAD == 4 && B_LOADS_PER_THREAD == 4, "tile config");

__global__ void __launch_bounds__(NTHREADS)
dgemmKernel(const double* __restrict__ A, const double* __restrict__ B,
            double* __restrict__ C, const size_t M, const size_t N, const size_t K) {
    __shared__ double As[2][BK][BM + 1];
    __shared__ double Bs[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const size_t blockRow = static_cast<size_t>(blockIdx.y) * BM;
    const size_t blockCol = static_cast<size_t>(blockIdx.x) * BN;

    // A tile (BM rows x BK cols): each thread loads 4 consecutive k-elements of one row.
    const int aRow = tid / (BK / 4);        // 0..63
    const int aCol = (tid % (BK / 4)) * 4;  // 0,4,8,12
    // B tile (BK rows x BN cols): each thread loads 4 consecutive n-elements of one row.
    const int bRow = tid / (BN / 4);        // 0..15
    const int bCol = (tid % (BN / 4)) * 4;  // 0..60

    const size_t gARow = blockRow + aRow;
    const size_t gBCol = blockCol + bCol;
    const bool aRowOk = gARow < M;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    double aReg[4], bReg[4];

    auto loadGlobal = [&](size_t kBase) {
        const size_t kA = kBase + aCol;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const size_t k = kA + t;
            aReg[t] = (aRowOk && k < K) ? A[gARow * K + k] : 0.0;
        }
        const size_t kB = kBase + bRow;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const size_t n = gBCol + t;
            bReg[t] = (kB < K && n < N) ? B[kB * N + n] : 0.0;
        }
    };

    auto storeShared = [&](int buf) {
#pragma unroll
        for (int t = 0; t < 4; ++t) As[buf][aCol + t][aRow] = aReg[t];
#pragma unroll
        for (int t = 0; t < 4; ++t) Bs[buf][bRow][bCol + t] = bReg[t];
    };

    const size_t numTiles = (K + BK - 1) / BK;

    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    for (size_t tile = 0; tile < numTiles; ++tile) {
        const int cur = static_cast<int>(tile & 1);
        const int nxt = cur ^ 1;

        if (tile + 1 < numTiles) loadGlobal((tile + 1) * BK);

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[cur][k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[cur][k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }

        if (tile + 1 < numTiles) {
            storeShared(nxt);
        }
        __syncthreads();
    }

    // Epilogue
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t row = blockRow + ty * TM + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t col = blockCol + tx * TN + j;
            if (col < N) C[row * N + col] = acc[i][j];
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Chunking of a rank's row block. Every rank (including the receiver) derives
// the same layout from (localRows, N), so per-chunk messages can be matched.
// ---------------------------------------------------------------------------
size_t chunkRowsFor(const size_t localRows, const size_t N) {
    if (localRows == 0 || N == 0) return 0;
    const size_t rowBytes = N * sizeof(double);
    // Keep per-chunk buffers around 128 MiB, aligned to BM rows.
    size_t chunkRows = std::max<size_t>(BM, (128ull << 20) / rowBytes);
    chunkRows = (chunkRows / BM) * BM;
    chunkRows = std::min(chunkRows, localRows);
    return chunkRows == 0 ? localRows : chunkRows;
}

// ---------------------------------------------------------------------------
// GPU driver for one rank: localC[localRows x N] = localA[localRows x N] * B
// Host buffers must be pinned. The row block is processed in chunks on two
// streams so that H2D (A chunk) / kernel / D2H (C chunk) overlap. As soon as a
// chunk has landed in host memory, `onChunkDone(rowOffset, rows)` is invoked
// (in chunk order) so the caller can ship it off while the GPU keeps working.
// ---------------------------------------------------------------------------
template <typename ChunkCallback>
void matrixMultiplyGPU(const double* localA, const double* B, double* localC,
                       const size_t localRows, const size_t N, ChunkCallback&& onChunkDone) {
    if (localRows == 0 || N == 0) return;

    constexpr int NUM_STREAMS = 2;
    const size_t rowBytes = N * sizeof(double);
    const size_t chunkRows = chunkRowsFor(localRows, N);
    const size_t numChunks = (localRows + chunkRows - 1) / chunkRows;

    double* dB = nullptr;
    double* dA[NUM_STREAMS] = {nullptr, nullptr};
    double* dC[NUM_STREAMS] = {nullptr, nullptr};
    cudaStream_t streams[NUM_STREAMS];

    CUDA_CHECK(cudaMalloc(&dB, N * rowBytes));
    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaMalloc(&dA[s], chunkRows * rowBytes));
        CUDA_CHECK(cudaMalloc(&dC[s], chunkRows * rowBytes));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
    }

    // B is needed by every chunk: upload once on stream 0 and make stream 1 wait.
    cudaEvent_t bReady;
    CUDA_CHECK(cudaEventCreateWithFlags(&bReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemcpyAsync(dB, B, N * rowBytes, cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaEventRecord(bReady, streams[0]));
    for (int s = 1; s < NUM_STREAMS; ++s) CUDA_CHECK(cudaStreamWaitEvent(streams[s], bReady, 0));

    std::vector<cudaEvent_t> chunkDone(numChunks);
    for (auto& e : chunkDone) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

    const dim3 block(THREADS_X, THREADS_Y);
    for (size_t c = 0; c < numChunks; ++c) {
        const int s = static_cast<int>(c % NUM_STREAMS);
        const size_t r0 = c * chunkRows;
        const size_t rows = std::min(chunkRows, localRows - r0);
        const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                        static_cast<unsigned>((rows + BM - 1) / BM));

        CUDA_CHECK(cudaMemcpyAsync(dA[s], localA + r0 * N, rows * rowBytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        dgemmKernel<<<grid, block, 0, streams[s]>>>(dA[s], dB, dC[s], rows, N, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(localC + r0 * N, dC[s], rows * rowBytes,
                                   cudaMemcpyDeviceToHost, streams[s]));
        CUDA_CHECK(cudaEventRecord(chunkDone[c], streams[s]));
    }

    // Hand chunks to the caller as they complete (in order) while later chunks run.
    for (size_t c = 0; c < numChunks; ++c) {
        CUDA_CHECK(cudaEventSynchronize(chunkDone[c]));
        const size_t r0 = c * chunkRows;
        onChunkDone(r0, std::min(chunkRows, localRows - r0));
    }

    for (int s = 0; s < NUM_STREAMS; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    for (auto& e : chunkDone) CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaEventDestroy(bReady));
    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(dA[s]));
        CUDA_CHECK(cudaFree(dC[s]));
    }
    CUDA_CHECK(cudaFree(dB));
}

// ---------------------------------------------------------------------------
// Validation: recompute a few elements on the host from the generator and
// compare with the gathered C (rank 0 only). Same points/tolerance as original.
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
#pragma omp critical
                {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                    valid = false;
                }
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool exitEarly = false;

    // Parse command line arguments (all ranks see identical argv)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            exitEarly = true;
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitEarly = true;
            exitCode = 1;
            break;
        }
    }
    if (exitEarly) {
        MPI_Finalize();
        return exitCode;
    }

    // ---- GPU selection: one device per rank, round-robin over node-local ranks
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(0));  // establish context now so it is not timed later

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallel config: %d MPI ranks, %d OpenMP threads/rank, %d GPU(s)/node, %d ranks/node\n",
               size, omp_get_max_threads(), deviceCount, localSize);
    }

    // ---- Row-block decomposition
    const size_t rowsPerRank = N / size;
    const size_t remainder = N % size;
    const size_t localRows = rowsPerRank + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t rowBegin = static_cast<size_t>(rank) * rowsPerRank +
                            std::min(static_cast<size_t>(rank), remainder);

    // ---- Allocate (pinned host memory for async transfers)
    // Rank 0 computes its own row block directly into the full result matrix C
    // (its block starts at row 0); other ranks use a pinned local buffer.
    double* localA = nullptr;
    double* B = nullptr;
    double* localC = nullptr;
    const size_t localElems = localRows * N;
    const size_t fullElems = N * N;
    std::vector<double> C;
    CUDA_CHECK(cudaHostAlloc(&localA, std::max<size_t>(localElems, 1) * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&B, std::max<size_t>(fullElems, 1) * sizeof(double), cudaHostAllocDefault));
    if (rank == 0) {
        C.resize(fullElems);
        if (localElems > 0) {
            CUDA_CHECK(cudaHostRegister(C.data(), localElems * sizeof(double), cudaHostRegisterDefault));
        }
        localC = C.data();
    } else {
        CUDA_CHECK(cudaHostAlloc(&localC, std::max<size_t>(localElems, 1) * sizeof(double), cudaHostAllocDefault));
    }

    // ---- Initialise matrices (OpenMP, each rank generates what it needs)
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(localA, N, rowBegin, localRows);
    initMatrixRows(B, N, 0, N);

    // Row datatype keeps MPI counts small even for huge N.
    MPI_Datatype rowType;
    MPI_CHECK(MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType));
    MPI_CHECK(MPI_Type_commit(&rowType));

    // ---- Compute
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    // Rank 0 pre-posts one receive per (rank, chunk) straight into C so that
    // remote chunks stream in while all GPUs are still busy.
    std::vector<MPI_Request> requests;
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            const size_t rr = rowsPerRank + (static_cast<size_t>(r) < remainder ? 1 : 0);
            const size_t rb = static_cast<size_t>(r) * rowsPerRank + std::min(static_cast<size_t>(r), remainder);
            const size_t cr = chunkRowsFor(rr, N);
            for (size_t r0 = 0; r0 < rr; r0 += cr) {
                const size_t rows = std::min(cr, rr - r0);
                requests.emplace_back();
                MPI_CHECK(MPI_Irecv(C.data() + (rb + r0) * N, static_cast<int>(rows), rowType,
                                    r, static_cast<int>(r0 / cr), MPI_COMM_WORLD, &requests.back()));
            }
        }
    }

    const size_t myChunkRows = chunkRowsFor(localRows, N);
    matrixMultiplyGPU(localA, B, localC, localRows, N, [&](size_t r0, size_t rows) {
        if (rank != 0) {
            requests.emplace_back();
            MPI_CHECK(MPI_Isend(localC + r0 * N, static_cast<int>(rows), rowType, 0,
                                static_cast<int>(r0 / myChunkRows), MPI_COMM_WORLD, &requests.back()));
        }
    });

    if (!requests.empty()) {
        MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MPI_CHECK(MPI_Type_free(&rowType));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

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
            bool valid = validateResult(C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    CUDA_CHECK(cudaFreeHost(localA));
    CUDA_CHECK(cudaFreeHost(B));
    if (rank == 0) {
        if (localElems > 0) CUDA_CHECK(cudaHostUnregister(C.data()));
    } else {
        CUDA_CHECK(cudaFreeHost(localC));
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exitCode;
}
