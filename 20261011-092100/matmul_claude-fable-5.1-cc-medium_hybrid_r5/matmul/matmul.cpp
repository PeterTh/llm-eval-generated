// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//   * MPI    : the rows of A (and therefore of C) are block-distributed across
//              ranks; every rank holds a full copy of B (initialised locally,
//              no communication needed). Row blocks of C are gathered to rank 0.
//   * CUDA   : each rank drives one GPU (local rank modulo device count) and
//              computes its C row block with a register-tiled DGEMM kernel.
//              Host<->device traffic is pipelined over two streams in chunks.
//   * OpenMP : all host-side O(N^2) work (matrix initialisation, validation,
//              host buffer handling) is multi-threaded.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialise rows [rowBegin, rowBegin + rows) of an NxN matrix into a
// rows x N buffer (OpenMP parallel).
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

// ---------------------------------------------------------------------------
// CUDA DGEMM kernel: C[M x N] = A[M x K] * B[K x N], row-major, leading dim.
// Block tile 64x64, K-step 16, 256 threads, each computing a 4x4 micro tile.
// ---------------------------------------------------------------------------
namespace {
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
}  // namespace

__global__ void __launch_bounds__(THREADS)
dgemmKernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
            const int M, const int N, const int K) {
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN + 1];

    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN);  // 0..15 -> column group
    const int ty = tid / (BN / TN);  // 0..15 -> row group

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    // Load mapping for A tile (BM x BK = 1024 elements, 4 per thread):
    //   consecutive threads read consecutive k for coalescing within a row.
    const int aRow = tid / BK;        // 0..15
    const int aCol = tid % BK;        // 0..15
    // Load mapping for B tile (BK x BN = 1024 elements, 4 per thread):
    const int bRow = tid / BN;        // 0..3
    const int bCol = tid % BN;        // 0..63

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int s = 0; s < BM / (THREADS / BK); ++s) {  // 4 iterations
            const int r = aRow + s * (THREADS / BK);
            const int gr = rowBase + r;
            const int gk = k0 + aCol;
            As[aCol][r] = (gr < M && gk < K) ? A[static_cast<size_t>(gr) * K + gk] : 0.0;
        }
#pragma unroll
        for (int s = 0; s < BK / (THREADS / BN); ++s) {  // 4 iterations
            const int r = bRow + s * (THREADS / BN);
            const int gk = k0 + r;
            const int gc = colBase + bCol;
            Bs[r][bCol] = (gk < K && gc < N) ? B[static_cast<size_t>(gk) * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + ty * TM + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + tx * TN + j;
            if (gc < N) C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
        }
    }
}

// Compute the local row block C_local = A_local * B on the GPU.
// A_local: localRows x N (host, pinned), B: N x N (host, pinned),
// C_local: localRows x N (host, pinned).
void matrixMultiplyLocal(const double* A_local, const double* B, double* C_local,
                         const size_t N, const size_t localRows) {
    if (localRows == 0 || N == 0) return;

    const size_t bytesB = N * N * sizeof(double);
    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, bytesB));

    // Chunk rows so that transfers of chunk i+1 overlap compute of chunk i.
    constexpr int NSTREAMS = 2;
    const size_t chunkRows = std::max<size_t>(BM, ((localRows + 2 * NSTREAMS - 1) / (2 * NSTREAMS) + BM - 1) / BM * BM);
    const size_t numChunks = (localRows + chunkRows - 1) / chunkRows;

    cudaStream_t streams[NSTREAMS];
    double* dA[NSTREAMS];
    double* dC[NSTREAMS];
    for (int s = 0; s < NSTREAMS; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&dA[s], chunkRows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC[s], chunkRows * N * sizeof(double)));
    }

    // B is needed by every chunk: upload on stream 0 and make others wait.
    cudaEvent_t bReady;
    CUDA_CHECK(cudaEventCreateWithFlags(&bReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemcpyAsync(dB, B, bytesB, cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaEventRecord(bReady, streams[0]));
    for (int s = 1; s < NSTREAMS; ++s) CUDA_CHECK(cudaStreamWaitEvent(streams[s], bReady, 0));

    for (size_t c = 0; c < numChunks; ++c) {
        const int s = static_cast<int>(c % NSTREAMS);
        const size_t r0 = c * chunkRows;
        const size_t rows = std::min(chunkRows, localRows - r0);
        const size_t bytes = rows * N * sizeof(double);

        CUDA_CHECK(cudaMemcpyAsync(dA[s], A_local + r0 * N, bytes, cudaMemcpyHostToDevice, streams[s]));
        const dim3 block(THREADS);
        const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN), static_cast<unsigned>((rows + BM - 1) / BM));
        dgemmKernel<<<grid, block, 0, streams[s]>>>(dA[s], dB, dC[s], static_cast<int>(rows),
                                                    static_cast<int>(N), static_cast<int>(N));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(C_local + r0 * N, dC[s], bytes, cudaMemcpyDeviceToHost, streams[s]));
    }

    for (int s = 0; s < NSTREAMS; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    CUDA_CHECK(cudaEventDestroy(bReady));
    for (int s = 0; s < NSTREAMS; ++s) {
        CUDA_CHECK(cudaFree(dA[s]));
        CUDA_CHECK(cudaFree(dC[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    CUDA_CHECK(cudaFree(dB));
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool ok = true;

#pragma omp parallel for collapse(2) schedule(static)
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
#pragma omp critical
                {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                    ok = false;
                }
            }
        }
    }

    return ok;
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

    // Parse command line arguments (identical on all ranks)
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

    // Select a GPU: one per local (node) rank, round-robin over devices.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(0));  // establish context before timing

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
               size, omp_get_max_threads(), deviceCount);
    }

    // Row-block decomposition of A / C over ranks.
    const size_t rowsBase = N / static_cast<size_t>(size);
    const size_t rowsRem = N % static_cast<size_t>(size);
    auto rowBeginOf = [&](int r) {
        const size_t ur = static_cast<size_t>(r);
        return ur * rowsBase + std::min(ur, rowsRem);
    };
    const size_t rowBegin = rowBeginOf(rank);
    const size_t localRows = rowBeginOf(rank + 1) - rowBegin;

    // Allocate matrices. Rank 0 keeps the full A and C (its row block is the
    // leading part of each); other ranks hold only their row blocks.
    std::vector<double> A(rank == 0 ? N * N : localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rank == 0 ? N * N : localRows * N);

    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    if (rank == 0) {
        initMatrixRows(A.data(), N, 0, N);
    } else {
        initMatrixRows(A.data(), N, rowBegin, localRows);
    }
    initMatrixRows(B.data(), N, 0, N);

    // Pin host buffers for fast asynchronous transfers.
    if (!A.empty()) CUDA_CHECK(cudaHostRegister(A.data(), A.size() * sizeof(double), cudaHostRegisterDefault));
    if (!B.empty()) CUDA_CHECK(cudaHostRegister(B.data(), B.size() * sizeof(double), cudaHostRegisterDefault));
    if (!C.empty()) CUDA_CHECK(cudaHostRegister(C.data(), C.size() * sizeof(double), cudaHostRegisterDefault));

    // Gather layout for C row blocks.
    std::vector<int> recvCounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        recvCounts[r] = static_cast<int>((rowBeginOf(r + 1) - rowBeginOf(r)) * N);
        displs[r] = static_cast<int>(rowBeginOf(r) * N);
    }

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A.data(), B.data(), C.data(), N, localRows);

    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, C.data(), recvCounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE, nullptr, nullptr,
                    nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDuration / 1000.0) / 1e9;
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

    if (!A.empty()) CUDA_CHECK(cudaHostUnregister(A.data()));
    if (!B.empty()) CUDA_CHECK(cudaHostUnregister(B.data()));
    if (!C.empty()) CUDA_CHECK(cudaHostUnregister(C.data()));

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
