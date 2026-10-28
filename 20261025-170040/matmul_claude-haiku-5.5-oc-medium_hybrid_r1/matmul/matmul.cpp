// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//   - MPI:    C is split into contiguous row blocks, one per rank. Every rank
//             generates A and B locally (deterministic), so no input traffic.
//   - OpenMP: within a rank, one thread drives the local GPU while the other
//             threads run CPU workers. All of them pull row chunks from a
//             shared atomic scheduler, so the faster device automatically
//             takes more of the work.
//   - CUDA:   shared-memory tiled GEMM with per-thread register blocking.
//
// Numerical semantics: every element C[i][j] is accumulated as
//     sum = 0; for k = 0..N-1: sum += A[i][k] * B[k][j]
// in strictly increasing k, exactly like the reference loop, on both the CPU
// and the GPU path. Results are therefore bit-identical to the serial code
// (given FMA contraction, which both gcc -march=native and nvcc -fmad=true
// apply to this expression).

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// Work-distribution granularity (rows of C).
constexpr size_t GPU_CHUNK_ROWS = 128;
constexpr size_t CPU_CHUNK_ROWS = 16;

// CPU micro-kernel blocking.
constexpr size_t CPU_ROW_BLOCK = 8;
constexpr size_t CPU_COL_TILE = 512;

// GPU tiled kernel: 64x64 output tile per block, 16x16 threads, 4x4 per thread.
constexpr int GEMM_BM = 64;
constexpr int GEMM_BN = 64;
constexpr int GEMM_BK = 16;
constexpr int GEMM_TM = 4;
constexpr int GEMM_TN = 4;
constexpr int GEMM_THREADS = 256;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, static_cast<size_t>(i), j);
        }
    }
}

// Dynamic row scheduler shared by the GPU driver and the CPU workers.
struct RowScheduler {
    std::atomic<size_t> next;
    size_t end;
};

bool grabRows(RowScheduler& s, size_t chunk, size_t& r0, size_t& r1) {
    const size_t begin = s.next.fetch_add(chunk, std::memory_order_relaxed);
    if (begin >= s.end) return false;
    r0 = begin;
    r1 = std::min(begin + chunk, s.end);
    return true;
}

// CPU path: computes rows [r0, r1) of C = A*B. Column tiles are kept in a
// small accumulator so each B row is reused across several A rows, while each
// element still accumulates over k in increasing order.
void cpuRows(const double* A, const double* B, double* C, size_t N, size_t r0, size_t r1) {
    std::vector<double> acc(CPU_ROW_BLOCK * CPU_COL_TILE);

    for (size_t rb = r0; rb < r1; rb += CPU_ROW_BLOCK) {
        const size_t nr = std::min(rb + CPU_ROW_BLOCK, r1) - rb;

        for (size_t jb = 0; jb < N; jb += CPU_COL_TILE) {
            const size_t nj = std::min(jb + CPU_COL_TILE, N) - jb;

            for (size_t r = 0; r < nr; ++r) {
                std::fill_n(acc.data() + r * CPU_COL_TILE, nj, 0.0);
            }

            for (size_t k = 0; k < N; ++k) {
                const double* bRow = B + k * N + jb;
                for (size_t r = 0; r < nr; ++r) {
                    const double a = A[(rb + r) * N + k];
                    double* c = acc.data() + r * CPU_COL_TILE;
                    for (size_t j = 0; j < nj; ++j) {
                        c[j] += a * bRow[j];
                    }
                }
            }

            for (size_t r = 0; r < nr; ++r) {
                std::copy_n(acc.data() + r * CPU_COL_TILE, nj, C + (rb + r) * N + jb);
            }
        }
    }
}

// GPU path: C_local[i][j] for global rows rowBase + i (i < rows).
// Each thread owns a TM x TN micro-tile; the k-loop walks tiles of BK in
// increasing order and accumulates into one register per output element.
__global__ void __launch_bounds__(GEMM_THREADS)
gemmTiled(const double* __restrict__ A, const double* __restrict__ B,
          double* __restrict__ Cchunk, size_t rowBase, size_t rows, size_t N) {
    __shared__ double As[GEMM_BM][GEMM_BK + 1];
    __shared__ double Bs[GEMM_BK][GEMM_BN + 1];

    const int tid = threadIdx.x;
    const int tx = tid % (GEMM_BN / GEMM_TN);
    const int ty = tid / (GEMM_BN / GEMM_TN);
    const size_t bi = static_cast<size_t>(blockIdx.y) * GEMM_BM;  // local row base
    const size_t bj = static_cast<size_t>(blockIdx.x) * GEMM_BN;  // column base

    double acc[GEMM_TM][GEMM_TN];
#pragma unroll
    for (int m = 0; m < GEMM_TM; ++m)
#pragma unroll
        for (int n = 0; n < GEMM_TN; ++n) acc[m][n] = 0.0;

    for (size_t k0 = 0; k0 < N; k0 += GEMM_BK) {
        const int kLen = (N - k0 < static_cast<size_t>(GEMM_BK)) ? static_cast<int>(N - k0) : GEMM_BK;

        for (int l = tid; l < GEMM_BM * GEMM_BK; l += GEMM_THREADS) {
            const int r = l / GEMM_BK;
            const int c = l % GEMM_BK;
            const size_t gr = bi + r;
            As[r][c] = (gr < rows && c < kLen) ? A[(rowBase + gr) * N + k0 + c] : 0.0;
        }
        for (int l = tid; l < GEMM_BK * GEMM_BN; l += GEMM_THREADS) {
            const int r = l / GEMM_BN;
            const int c = l % GEMM_BN;
            const size_t gc = bj + c;
            Bs[r][c] = (r < kLen && gc < N) ? B[(k0 + r) * N + gc] : 0.0;
        }
        __syncthreads();

        for (int kk = 0; kk < kLen; ++kk) {
            double a[GEMM_TM];
            double b[GEMM_TN];
#pragma unroll
            for (int m = 0; m < GEMM_TM; ++m) a[m] = As[ty * GEMM_TM + m][kk];
#pragma unroll
            for (int n = 0; n < GEMM_TN; ++n) b[n] = Bs[kk][tx * GEMM_TN + n];
#pragma unroll
            for (int m = 0; m < GEMM_TM; ++m)
#pragma unroll
                for (int n = 0; n < GEMM_TN; ++n) acc[m][n] += a[m] * b[n];
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < GEMM_TM; ++m) {
        const size_t gr = bi + ty * GEMM_TM + m;
        if (gr >= rows) continue;
#pragma unroll
        for (int n = 0; n < GEMM_TN; ++n) {
            const size_t gc = bj + tx * GEMM_TN + n;
            if (gc < N) Cchunk[gr * N + gc] = acc[m][n];
        }
    }
}

// Driver loop for the GPU: owns one OpenMP thread and one device.
void gpuDriver(const double* A, const double* B, double* C, size_t N, size_t gpuChunk,
               RowScheduler& sched, int devId) {
    CUDA_CHECK(cudaSetDevice(devId));

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    const size_t matBytes = N * N * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dA, matBytes));
    CUDA_CHECK(cudaMalloc(&dB, matBytes));
    CUDA_CHECK(cudaMalloc(&dC, gpuChunk * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, A, matBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, matBytes, cudaMemcpyHostToDevice));

    size_t r0 = 0;
    size_t r1 = 0;
    while (grabRows(sched, gpuChunk, r0, r1)) {
        const size_t rows = r1 - r0;
        const dim3 block(GEMM_THREADS);
        const dim3 grid(static_cast<unsigned>((N + GEMM_BN - 1) / GEMM_BN),
                        static_cast<unsigned>((rows + GEMM_BM - 1) / GEMM_BM));
        gemmTiled<<<grid, block>>>(dA, dB, dC, r0, rows, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C + r0 * N, dC, rows * N * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// Computes rows [r0, r1) of C on this rank using the GPU driver thread (team
// thread 0) and CPU workers (remaining team threads).
void matrixMultiplyLocal(const double* A, const double* B, double* C, size_t N,
                         size_t r0, size_t r1, int devId, int teamSize) {
    const size_t rows = r1 - r0;
    if (rows == 0 || N == 0) return;

    RowScheduler sched{};
    sched.next.store(r0, std::memory_order_relaxed);
    sched.end = r1;
    const size_t gpuChunk = std::min(GPU_CHUNK_ROWS, rows);

#pragma omp parallel num_threads(teamSize)
    {
        if (omp_get_thread_num() == 0) {
            gpuDriver(A, B, C, N, gpuChunk, sched, devId);
        } else {
            size_t c0 = 0;
            size_t c1 = 0;
            while (grabRows(sched, CPU_CHUNK_ROWS, c0, c1)) {
                cpuRows(A, B, C, N, c0, c1);
            }
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

// First row of rank p's block when N rows are split across P ranks.
size_t rowBegin(size_t p, size_t P, size_t N) { return N * p / P; }

int runBenchmark(int argc, char** argv, int rank, int size) {
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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Resource setup: one GPU per local rank (round-robin), OpenMP team sized
    // to split the node's hardware threads among the local ranks.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int devId = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(devId));
    CUDA_CHECK(cudaFree(nullptr));  // create the CUDA context outside the timed region

    const int hwThreads = std::max(1, omp_get_num_procs());
    const int teamSize = std::max(1, hwThreads / localSize);
    omp_set_num_threads(teamSize);

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (every rank holds full A and B; C is full for gathering)
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    const size_t r0 = rowBegin(static_cast<size_t>(rank), static_cast<size_t>(size), N);
    const size_t r1 = rowBegin(static_cast<size_t>(rank) + 1, static_cast<size_t>(size), N);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A.data(), B.data(), C.data(), N, r0, r1, devId, teamSize);

    auto end = std::chrono::high_resolution_clock::now();
    const double localSec = std::chrono::duration<double>(end - start).count();

    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full C on the root only when it is needed for output or validation
    const bool needFull = printResults || validate;
    if (needFull && size > 1) {
        if (N * N > static_cast<size_t>(INT_MAX)) {
            if (root) fprintf(stderr, "Matrix too large to gather with int counts\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int p = 0; p < size; ++p) {
            const size_t pb = rowBegin(p, size, N);
            const size_t pe = rowBegin(p + 1, size, N);
            counts[p] = static_cast<int>((pe - pb) * N);
            displs[p] = static_cast<int>(pb * N);
        }
        if (root) {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, C.data(), counts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(C.data() + r0 * N, counts[rank], MPI_DOUBLE, nullptr, nullptr, nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    if (!root) return 0;

    const long long ms = static_cast<long long>(maxSec * 1000.0);
    printf("Computation time: %lld ms\n", ms);

    // Calculate GFLOPS
    const double gflops = (2.0 * N * N * N) / maxSec / 1e9;
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
