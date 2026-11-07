// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//   * MPI   : C = A * B is split into contiguous blocks of rows of A / C across
//             ranks (1-D row-block decomposition). Every rank generates its own
//             rows of A and a full copy of B directly from the deterministic
//             generator, so no input data has to be communicated. Each rank is
//             pinned to one GPU (local rank modulo #GPUs on the node).
//   * CUDA  : the local block-row product is computed on the GPU with a
//             register-blocked, shared-memory tiled DGEMM kernel. Host<->device
//             transfers of A/C are chunked over two streams so that copies
//             overlap with compute on the device.
//   * OpenMP: all host-side O(N^2) work (matrix generation, padding, validation
//             reference, assembling the result) runs multi-threaded.
//
// Semantics match the original serial program: identical inputs, identical
// output text, identical validation rule, rank 0 produces all output.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,       \
                    __LINE__, cudaGetErrorString(err__));                          \
            MPI_Abort(MPI_COMM_WORLD, 2);                                          \
        }                                                                          \
    } while (0)

#define MPI_CHECK(call)                                                            \
    do {                                                                           \
        int err__ = (call);                                                        \
        if (err__ != MPI_SUCCESS) {                                                \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call,         \
                    __FILE__, __LINE__);                                           \
            MPI_Abort(MPI_COMM_WORLD, 3);                                          \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Input generation (unchanged from the original program)
// ---------------------------------------------------------------------------
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fill rows [rowBegin, rowBegin+rows) of the NxN generator matrix into `mat`
// (row-major, leading dimension N). OpenMP parallel over rows.
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t rows) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
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
// CUDA DGEMM kernel
//   C[M x Np] = A[M x Kp] * B[Kp x Np]  (row-major, zero-padded operands)
//   Block tile 64x64, K-step 16, 256 threads, 4x4 outputs per thread.
// ---------------------------------------------------------------------------
namespace gemm {
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
static_assert(THREADS == 256, "tile configuration expects 256 threads");
static_assert((BM * BK) % THREADS == 0 && (BK * BN) % THREADS == 0, "tile load split");
}  // namespace gemm

__global__ void __launch_bounds__(gemm::THREADS)
dgemmKernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
            const int Kp, const int Np) {
    using namespace gemm;
    // +1 padding on the K dimension of As removes shared-memory bank conflicts
    // for the strided per-thread reads in the inner loop.
    __shared__ double As[BM][BK + 1];
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN);  // 0..15 -> column group
    const int ty = tid / (BN / TN);  // 0..15 -> row group

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    const double* Ablk = A + static_cast<size_t>(rowBase) * Kp;
    const double* Bblk = B + colBase;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < Kp; k0 += BK) {
        // Cooperative, coalesced tile loads: A tile 64x16, B tile 16x64.
#pragma unroll
        for (int l = 0; l < (BM * BK) / THREADS; ++l) {
            const int idx = tid + l * THREADS;
            const int r = idx / BK;
            const int kk = idx % BK;
            As[r][kk] = Ablk[static_cast<size_t>(r) * Kp + k0 + kk];
        }
#pragma unroll
        for (int l = 0; l < (BK * BN) / THREADS; ++l) {
            const int idx = tid + l * THREADS;
            const int kk = idx / BN;
            const int c = idx % BN;
            Bs[kk][c] = Bblk[static_cast<size_t>(k0 + kk) * Np + c];
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    double* Cblk = C + static_cast<size_t>(rowBase) * Np + colBase;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            Cblk[static_cast<size_t>(ty * TM + i) * Np + tx * TN + j] = acc[i][j];
}

// ---------------------------------------------------------------------------
// Local block-row product on the GPU:
//   Clocal[rows x N] = Alocal[rows x N] * B[N x N]
// Host buffers must be pinned (cudaMallocHost) for asynchronous transfers.
// Padding to tile multiples is done on the device via pitched copies into
// zero-initialised buffers, so the kernel needs no bounds checks.
// ---------------------------------------------------------------------------
static size_t roundUp(const size_t v, const size_t m) { return (v + m - 1) / m * m; }

void matrixMultiplyLocal(const double* Alocal, const double* B, double* Clocal,
                         const size_t N, const size_t rows) {
    using namespace gemm;
    if (rows == 0 || N == 0) return;

    const size_t Kp = roundUp(N, BK);  // padded inner dimension
    const size_t Np = roundUp(N, BN);  // padded column count
    const size_t rowBytes = N * sizeof(double);

    // Chunk the local rows so that copies of one chunk overlap with compute
    // of the other. Aim for >= 4 chunks when there is enough work.
    constexpr int NSTREAMS = 2;
    size_t chunkRows = roundUp(std::max<size_t>(1, (rows + 3) / 4), BM);
    chunkRows = std::max<size_t>(chunkRows, BM);
    chunkRows = std::min<size_t>(chunkRows, roundUp(rows, BM));
    // Cap chunk memory (A + C padded) at roughly 1 GiB per slot.
    const size_t perRowBytes = (Kp + Np) * sizeof(double);
    const size_t maxChunk = std::max<size_t>(BM, roundUp((1ull << 30) / perRowBytes, BM) );
    chunkRows = std::min(chunkRows, maxChunk);

    // Device buffers
    double* dB = nullptr;
    double* dA[NSTREAMS] = {nullptr, nullptr};
    double* dC[NSTREAMS] = {nullptr, nullptr};
    cudaStream_t streams[NSTREAMS];

    CUDA_CHECK(cudaMalloc(&dB, Kp * Np * sizeof(double)));
    for (int s = 0; s < NSTREAMS; ++s) {
        CUDA_CHECK(cudaMalloc(&dA[s], chunkRows * Kp * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC[s], chunkRows * Np * sizeof(double)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
    }

    // Zero padding regions (only needed where the padded shape exceeds N).
    if (Kp != N || Np != N) {
        CUDA_CHECK(cudaMemsetAsync(dB, 0, Kp * Np * sizeof(double), streams[0]));
        for (int s = 0; s < NSTREAMS; ++s) {
            CUDA_CHECK(cudaMemsetAsync(dA[s], 0, chunkRows * Kp * sizeof(double), streams[0]));
        }
    }

    // Upload B once (all chunks depend on it).
    CUDA_CHECK(cudaMemcpy2DAsync(dB, Np * sizeof(double), B, rowBytes, rowBytes, N,
                                 cudaMemcpyHostToDevice, streams[0]));
    cudaEvent_t bReady;
    CUDA_CHECK(cudaEventCreateWithFlags(&bReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(bReady, streams[0]));
    for (int s = 1; s < NSTREAMS; ++s) CUDA_CHECK(cudaStreamWaitEvent(streams[s], bReady, 0));

    const dim3 block(THREADS);
    int slot = 0;
    for (size_t r0 = 0; r0 < rows; r0 += chunkRows, slot = (slot + 1) % NSTREAMS) {
        const size_t cr = std::min(chunkRows, rows - r0);
        const size_t crp = roundUp(cr, BM);
        cudaStream_t st = streams[slot];

        CUDA_CHECK(cudaMemcpy2DAsync(dA[slot], Kp * sizeof(double), Alocal + r0 * N, rowBytes,
                                     rowBytes, cr, cudaMemcpyHostToDevice, st));
        // Rows [cr, crp) of the slot may hold stale data from a previous chunk;
        // they only affect padded output rows, which are never copied back.
        const dim3 grid(static_cast<unsigned>(Np / BN), static_cast<unsigned>(crp / BM));
        dgemmKernel<<<grid, block, 0, st>>>(dA[slot], dB, dC[slot], static_cast<int>(Kp),
                                            static_cast<int>(Np));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy2DAsync(Clocal + r0 * N, rowBytes, dC[slot], Np * sizeof(double),
                                     rowBytes, cr, cudaMemcpyDeviceToHost, st));
    }

    for (int s = 0; s < NSTREAMS; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    CUDA_CHECK(cudaEventDestroy(bReady));
    for (int s = 0; s < NSTREAMS; ++s) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(dA[s]));
        CUDA_CHECK(cudaFree(dC[s]));
    }
    CUDA_CHECK(cudaFree(dB));
}

// ---------------------------------------------------------------------------
// Simple validation: compute a few elements on the host and compare.
// Same check points, tolerance and messages as the original program. The 25
// reference dot products are computed in parallel; the comparison runs in the
// original loop order so the first reported failure is identical.
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    double expectedAll[25];

#pragma omp parallel for schedule(static)
    for (int p = 0; p < 25; ++p) {
        const size_t i = checkPoints[p / 5] % N;
        const size_t j = checkPoints[p % 5] % N;
        double expected = 0.0;
        for (size_t k = 0; k < N; ++k) {
            expected += A[i * N + k] * B[k * N + j];
        }
        expectedAll[p] = expected;
    }

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            const double expected = expectedAll[pi * 5 + pj];
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

// Select the GPU for this rank: ranks sharing a node are spread round-robin
// over the node's visible devices.
static void selectDevice() {
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                  &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0) {
        fprintf(stderr, "No CUDA device available: %s\n", cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context now, outside timing
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, nranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    const bool root = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical on all ranks).
    int earlyExit = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            earlyExit = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            earlyExit = 1;
            break;
        }
    }
    if (earlyExit >= 0) {
        MPI_CHECK(MPI_Finalize());
        return earlyExit;
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    selectDevice();

    // Row-block decomposition: rank r owns rows [rowBegin, rowBegin + rows).
    const size_t base = N / nranks;
    const size_t rem = N % nranks;
    const size_t rows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t rowBegin = static_cast<size_t>(rank) * base + std::min<size_t>(rank, rem);

    // Allocate matrices. Local A/C blocks live in pinned host memory for
    // asynchronous device transfers; B is replicated on every rank.
    double* Alocal = nullptr;
    double* Clocal = nullptr;
    CUDA_CHECK(cudaMallocHost(&Alocal, std::max<size_t>(1, rows * N) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&Clocal, std::max<size_t>(1, rows * N) * sizeof(double)));
    std::vector<double> B(N * N);

    // Initialize matrices (deterministic generator -> no communication).
    if (root) printf("Initializing matrices...\n");
    initMatrixRows(Alocal, N, rowBegin, rows);
    initMatrix(B, N);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(Alocal, B.data(), Clocal, N, rows);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));
    duration = std::chrono::milliseconds(max_ms);

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Assemble the full result on rank 0 only when it is needed for output.
    std::vector<double> C;
    if (printResults || validate) {
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            const size_t rr = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t rb = static_cast<size_t>(r) * base + std::min<size_t>(r, rem);
            counts[r] = static_cast<int>(rr * N);
            displs[r] = static_cast<int>(rb * N);
        }
        if (N * N > static_cast<size_t>(INT32_MAX)) {
            // Element counts exceed the 32-bit MPI count range; fall back to a
            // row-by-row collective-free transfer via a contiguous row type.
            MPI_Datatype rowType;
            MPI_CHECK(MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType));
            MPI_CHECK(MPI_Type_commit(&rowType));
            for (int r = 0; r < nranks; ++r) {
                counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
                displs[r] = static_cast<int>(static_cast<size_t>(r) * base + std::min<size_t>(r, rem));
            }
            if (root) C.resize(N * N);
            MPI_CHECK(MPI_Gatherv(Clocal, static_cast<int>(rows), rowType,
                                  root ? C.data() : nullptr, counts.data(), displs.data(),
                                  rowType, 0, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Type_free(&rowType));
        } else {
            if (root) C.resize(N * N);
            MPI_CHECK(MPI_Gatherv(Clocal, static_cast<int>(rows * N), MPI_DOUBLE,
                                  root ? C.data() : nullptr, counts.data(), displs.data(),
                                  MPI_DOUBLE, 0, MPI_COMM_WORLD));
        }
    }

    int exitCode = 0;

    // Print results for external validation
    if (printResults && root) {
        print_results(C, "MatrixC");
    }

    // Validation
    if (validate) {
        if (root) {
            printf("Validating result...\n");
            // Rank 0 only holds its own rows of A; regenerate the full matrix
            // (deterministic) for the reference computation.
            std::vector<double> A(N * N);
            initMatrix(A, N);
            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    CUDA_CHECK(cudaFreeHost(Alocal));
    CUDA_CHECK(cudaFreeHost(Clocal));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
