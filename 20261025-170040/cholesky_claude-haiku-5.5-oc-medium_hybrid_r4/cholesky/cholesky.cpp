// Hybrid MPI + OpenMP + CUDA Cholesky decomposition benchmark.
//
// Layout and algorithm
//   - The matrix is padded to N = nblk * kBlock with identity rows/columns, so every
//     block is full. The leading n x n block of the padded factor is the factor of the
//     original matrix.
//   - Block columns are distributed cyclically over MPI ranks: block column jb is owned
//     by rank jb % size. Each rank keeps its block columns resident on its GPU.
//   - Step k: the owner of block column k factors the diagonal block and solves the
//     panel below it on the host (OpenMP), then the panel is broadcast (MPI). Every rank
//     applies the trailing update to its block columns on its GPU (CUDA). Block columns
//     k+1 and k+2 are updated first on a lookahead stream and a bulk stream respectively,
//     so the owner of the next panel can factor it while the bulk of the update runs.
//
// Semantics match the sequential reference: same generated input, same lower-triangular
// output with zeroed upper part, same failure reporting and same validation. Blocking
// changes the order of floating-point accumulation across block boundaries, so results
// agree with the scalar algorithm to rounding error rather than bit-for-bit.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr size_t kBlock = 128;   // block width (columns per block column)
constexpr size_t kTile = 64;     // rows per GPU thread block
constexpr size_t kKChunk = 256;  // k-chunk used by the matrix generator

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err_));                              \
            MPI_Abort(MPI_COMM_WORLD, 2);                                             \
        }                                                                             \
    } while (0)

// Page-locked host buffer so that host<->device transfers run at DMA speed.
struct PinnedDeleter {
    void operator()(double* p) const { cudaFreeHost(p); }
};
using PinnedBuffer = std::unique_ptr<double[], PinnedDeleter>;

PinnedBuffer allocPinned(size_t count) {
    double* p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(1, count) * sizeof(double)));
    return PinnedBuffer(p);
}

// Number of block columns owned by `rank` under the cyclic distribution.
size_t localBlockCount(size_t nblk, int rank, int size) {
    return nblk / size + (static_cast<size_t>(rank) < nblk % size ? 1 : 0);
}

// Generates block column jb of the padded N x N SPD matrix (stored as N x kBlock,
// row-major). Inside the n x n region the values are B*B^T + n*I, accumulated in the
// same k-ascending order as the reference generator. Outside it the matrix is identity.
void generateBlockColumn(double* out, const double* B, size_t n, size_t N, size_t jb) {
    const size_t j0 = jb * kBlock;
    const size_t validCols = j0 < n ? std::min(kBlock, n - j0) : 0;
    std::fill(out, out + N * kBlock, 0.0);

    std::vector<double> packed(kKChunk * kBlock);
    for (size_t kc0 = 0; kc0 < n; kc0 += kKChunk) {
        const size_t kc1 = std::min(n, kc0 + kKChunk);
        // Transpose the block's rows of B into [k][c] order for contiguous access.
        for (size_t k = kc0; k < kc1; ++k) {
            double* dst = packed.data() + (k - kc0) * kBlock;
            for (size_t c = 0; c < kBlock; ++c) {
                dst[c] = c < validCols ? B[(j0 + c) * n + k] : 0.0;
            }
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            double* acc = out + i * kBlock;
            const double* bi = B + i * n;
            for (size_t k = kc0; k < kc1; ++k) {
                const double a = bi[k];
                const double* bt = packed.data() + (k - kc0) * kBlock;
                for (size_t c = 0; c < kBlock; ++c) {
                    acc[c] += a * bt[c];
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t c = 0; c < kBlock; ++c) {
            const size_t j = j0 + c;
            double& v = out[i * kBlock + c];
            if (i < n && j < n) {
                if (i == j) v += static_cast<double>(n);
            } else {
                v = (i == j) ? 1.0 : 0.0;
            }
        }
    }
}

// Factors the diagonal block of the panel and solves the rows below it, in place.
// P is M x kBlock row-major; row 0 corresponds to global row kRow0.
// Returns 0 on success, otherwise (global index of the failing pivot + 1).
size_t factorPanel(double* P, size_t M, size_t kRow0) {
    constexpr size_t NB = kBlock;
    // Diagonal block: same arithmetic as the sequential algorithm.
    for (size_t i = 0; i < NB; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += P[i * NB + k] * P[j * NB + k];
            }
            if (i == j) {
                const double val = P[j * NB + j] - sum;
                if (val <= 0.0) {
                    return kRow0 + j + 1;
                }
                P[j * NB + j] = sqrt(val);
            } else {
                P[i * NB + j] = (P[i * NB + j] - sum) / P[j * NB + j];
            }
        }
    }
    // Rows below the diagonal block: independent forward substitutions.
    #pragma omp parallel for schedule(static)
    for (size_t r = NB; r < M; ++r) {
        double* row = P + r * NB;
        for (size_t j = 0; j < NB; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * P[j * NB + k];
            }
            row[j] = (row[j] - sum) / P[j * NB + j];
        }
    }
    return 0;
}

// Trailing update for a range of owned block columns. Grid: x = row tiles of a column,
// y = column in the range. Column y is global block column firstJ + y * jStep, stored at
// base + y * colElems. For its rows r in [j0, N) (j0 = its first row):
//   C[r][c] -= sum_t Q[r][t] * Pj[c][t]
// where Q are panel rows (global row r -> panel row r - kRow0) and Pj are the panel rows
// of the block column. Each thread accumulates a 16 x 8 register tile for its tile.
__global__ void __launch_bounds__(256)
trailingUpdateKernel(double* __restrict__ base, const double* __restrict__ panel,
                     size_t kRow0, size_t firstJ, size_t jStep, size_t N, size_t colElems) {
    constexpr int RB = kTile / 16;   // rows per thread
    constexpr int CB = kBlock / 16;  // columns per thread

    const size_t y = blockIdx.y;
    const size_t j0 = (firstJ + y * jStep) * kBlock;
    if (blockIdx.x >= (N - j0) / kTile) {
        return;  // this column has fewer tiles than the largest one in the grid
    }

    __shared__ double As[16][kTile];
    __shared__ double Bs[16][kBlock];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t r0 = j0 + static_cast<size_t>(blockIdx.x) * kTile;
    const double* Q = panel + (r0 - kRow0) * kBlock;
    const double* Pj = panel + (j0 - kRow0) * kBlock;
    double* colBlock = base + y * colElems;

    double acc[RB][CB];
    #pragma unroll
    for (int i = 0; i < RB; ++i) {
        #pragma unroll
        for (int jj = 0; jj < CB; ++jj) acc[i][jj] = 0.0;
    }

    for (int t0 = 0; t0 < static_cast<int>(kBlock); t0 += 16) {
        for (int e = tid; e < static_cast<int>(kTile * 16); e += 256) {
            const int r = e >> 4, t = e & 15;
            As[t][r] = Q[r * kBlock + t0 + t];
        }
        for (int e = tid; e < static_cast<int>(kBlock * 16); e += 256) {
            const int c = e >> 4, t = e & 15;
            Bs[t][c] = Pj[c * kBlock + t0 + t];
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            double a[RB];
            double b[CB];
            #pragma unroll
            for (int i = 0; i < RB; ++i) a[i] = As[t][ty + 16 * i];
            #pragma unroll
            for (int jj = 0; jj < CB; ++jj) b[jj] = Bs[t][tx + 16 * jj];
            #pragma unroll
            for (int i = 0; i < RB; ++i) {
                #pragma unroll
                for (int jj = 0; jj < CB; ++jj) acc[i][jj] += a[i] * b[jj];
            }
        }
        __syncthreads();
    }

    double* C = colBlock + r0 * kBlock;
    #pragma unroll
    for (int i = 0; i < RB; ++i) {
        #pragma unroll
        for (int jj = 0; jj < CB; ++jj) {
            C[(ty + 16 * i) * kBlock + tx + 16 * jj] -= acc[i][jj];
        }
    }
}

// Enqueues the trailing update of colCount consecutive owned block columns on stream s.
// The first column is global block column firstJ; subsequent ones are jStep apart.
void launchTrailingUpdate(cudaStream_t s, double* base, size_t colCount, size_t colElems,
                          const double* panel, size_t kRow0, size_t firstJ, size_t jStep,
                          size_t N) {
    if (colCount == 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned int>((N - firstJ * kBlock) / kTile),
                    static_cast<unsigned int>(colCount));
    const dim3 block(16, 16);
    trailingUpdateKernel<<<grid, block, 0, s>>>(base, panel, kRow0, firstJ, jStep, N,
                                                colElems);
}

// Copies block column jb (N x kBlock, row-major) into the n x n row-major matrix A.
// With lowerOnly the strictly upper part is set to zero.
void unpackBlockColumn(const double* blk, size_t jb, size_t n, bool lowerOnly,
                       std::vector<double>& A) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < kBlock; ++c) {
            const size_t j = jb * kBlock + c;
            if (j >= n) break;
            if (lowerOnly && j > i) continue;
            A[i * n + j] = blk[i * kBlock + c];
        }
    }
}

// Gathers the block-column distributed storage onto rank 0 as an n x n row-major matrix,
// one block column at a time. Returns the matrix on rank 0 and an empty vector elsewhere.
std::vector<double> gatherMatrix(const double* local, size_t n, size_t N,
                                 size_t nblk, int rank, int size, bool lowerOnly) {
    const size_t colElems = N * kBlock;
    std::vector<double> A;
    std::vector<double> recvBuf;
    if (rank == 0) {
        A.assign(n * n, 0.0);
        recvBuf.resize(colElems);
    }
    for (size_t jb = 0; jb < nblk; ++jb) {
        const int owner = static_cast<int>(jb % size);
        const size_t lj = jb / size;
        if (rank == 0) {
            if (owner == 0) {
                unpackBlockColumn(local + lj * colElems, jb, n, lowerOnly, A);
            } else {
                MPI_Recv(recvBuf.data(), static_cast<int>(colElems), MPI_DOUBLE, owner,
                         static_cast<int>(jb), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                unpackBlockColumn(recvBuf.data(), jb, n, lowerOnly, A);
            }
        } else if (rank == owner) {
            MPI_Send(local + lj * colElems, static_cast<int>(colElems), MPI_DOUBLE, 0,
                     static_cast<int>(jb), MPI_COMM_WORLD);
        }
    }
    return A;
}

// Distributed right-looking blocked Cholesky with lookahead. hLoc holds this rank's block
// columns on entry and the factor on success. Returns 0 on success, otherwise the global
// failing index + 1 (identical on every rank).
int factorizeDistributed(double* hLoc, size_t N, size_t nblk, int rank, int size) {
    const size_t colElems = N * kBlock;
    const size_t Lc = localBlockCount(nblk, rank, size);
    auto bytes = [](size_t elems) { return std::max<size_t>(1, elems) * sizeof(double); };

    // Device storage: owned block columns, plus a ring of panel buffers.
    constexpr size_t kDRing = 3;
    double* dLoc = nullptr;
    double* dPanel[kDRing] = {nullptr, nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&dLoc, bytes(Lc * colElems)));
    for (size_t q = 0; q < kDRing; ++q) {
        CUDA_CHECK(cudaMalloc(&dPanel[q], bytes(colElems)));
    }
    if (Lc > 0) {
        CUDA_CHECK(cudaMemcpy(dLoc, hLoc, Lc * colElems * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    // Streams: sUp uploads panels, sLA updates the lookahead column, sBulk the rest.
    cudaStream_t sUp, sLA, sBulk;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sUp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sLA, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sBulk, cudaStreamNonBlocking));
    cudaEvent_t evUp, evLADone, evPiece, evBulkDone[kDRing];
    CUDA_CHECK(cudaEventCreateWithFlags(&evUp, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evLADone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evPiece, cudaEventDisableTiming));
    for (size_t q = 0; q < kDRing; ++q) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evBulkDone[q], cudaEventDisableTiming));
    }

    // Pinned ring of host panel buffers: asynchronous copies read/write a buffer until the
    // matching events complete, so host buffer q is reused only after evUpDone[q] and
    // evWbDone[q].
    constexpr size_t kRing = 4;
    PinnedBuffer hPanelBuf[kRing];
    cudaEvent_t evUpDone[kRing], evWbDone[kRing];
    for (size_t q = 0; q < kRing; ++q) {
        hPanelBuf[q] = allocPinned(colElems);
        CUDA_CHECK(cudaEventCreateWithFlags(&evUpDone[q], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evWbDone[q], cudaEventDisableTiming));
    }
    int failure = 0;

    for (size_t k = 0; k < nblk; ++k) {
        const int owner = static_cast<int>(k % size);
        const size_t kRow0 = k * kBlock;
        const size_t M = N - kRow0;
        const size_t panelElems = M * kBlock;
        const size_t q = k % kRing;
        double* hp = hPanelBuf[q].get();

        if (k >= kRing) {
            CUDA_CHECK(cudaEventSynchronize(evUpDone[q]));
            CUDA_CHECK(cudaEventSynchronize(evWbDone[q]));
        }

        // Owner: fetch block column k (up to date through step k-1), factor the panel.
        int status = 0;
        if (rank == owner) {
            double* src = dLoc + (k / size) * colElems + kRow0 * kBlock;
            CUDA_CHECK(cudaMemcpyAsync(hp, src, panelElems * sizeof(double),
                                       cudaMemcpyDeviceToHost, sLA));
            CUDA_CHECK(cudaStreamSynchronize(sLA));
            status = static_cast<int>(factorPanel(hp, M, kRow0));
            if (status == 0) {
                // Keep the device copy of block column k equal to the factor.
                CUDA_CHECK(cudaMemcpyAsync(src, hp, panelElems * sizeof(double),
                                           cudaMemcpyHostToDevice, sLA));
                CUDA_CHECK(cudaEventRecord(evWbDone[q], sLA));
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (status != 0) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n",
                       status - 1);
            }
            failure = status;
            break;
        }
        MPI_Bcast(hp, static_cast<int>(panelElems), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Upload panel k. Its buffer was last read by the updates of step k-kDRing.
        const size_t dq = k % kDRing;
        double* dP = dPanel[dq];
        if (k > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(sUp, evLADone, 0));
        }
        if (k >= kDRing) {
            CUDA_CHECK(cudaStreamWaitEvent(sUp, evBulkDone[dq], 0));
        }
        CUDA_CHECK(cudaMemcpyAsync(dP, hp, panelElems * sizeof(double),
                                   cudaMemcpyHostToDevice, sUp));
        CUDA_CHECK(cudaEventRecord(evUp, sUp));
        CUDA_CHECK(cudaEventRecord(evUpDone[q], sUp));

        // Lookahead: update block column k+1 on sLA. It needs the update from step k-1 of
        // column k+1, which was issued first in that step's bulk work (evPiece).
        CUDA_CHECK(cudaStreamWaitEvent(sLA, evUp, 0));
        if (k > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(sLA, evPiece, 0));
        }
        if (k + 1 < nblk && static_cast<int>((k + 1) % size) == rank) {
            launchTrailingUpdate(sLA, dLoc + ((k + 1) / size) * colElems, 1, colElems, dP,
                                 kRow0, k + 1, size, N);
        }
        CUDA_CHECK(cudaEventRecord(evLADone, sLA));

        // Bulk: column k+2 first (the next step's lookahead depends on it), then all other
        // owned columns with j > k+2. Owned columns are cyclic, so they form a local range.
        CUDA_CHECK(cudaStreamWaitEvent(sBulk, evUp, 0));
        if (k + 2 < nblk && static_cast<int>((k + 2) % size) == rank) {
            launchTrailingUpdate(sBulk, dLoc + ((k + 2) / size) * colElems, 1, colElems, dP,
                                 kRow0, k + 2, size, N);
        }
        CUDA_CHECK(cudaEventRecord(evPiece, sBulk));
        size_t lj0 = 0;
        while (lj0 < Lc && lj0 * size + rank <= k + 2) {
            ++lj0;
        }
        launchTrailingUpdate(sBulk, dLoc + lj0 * colElems, Lc - lj0, colElems, dP, kRow0,
                             lj0 * size + rank, size, N);
        CUDA_CHECK(cudaEventRecord(evBulkDone[dq], sBulk));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (failure == 0 && Lc > 0) {
        CUDA_CHECK(cudaMemcpy(hLoc, dLoc, Lc * colElems * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventDestroy(evUp));
    CUDA_CHECK(cudaEventDestroy(evLADone));
    CUDA_CHECK(cudaEventDestroy(evPiece));
    for (size_t q = 0; q < kDRing; ++q) {
        CUDA_CHECK(cudaEventDestroy(evBulkDone[q]));
    }
    for (size_t q = 0; q < kRing; ++q) {
        CUDA_CHECK(cudaEventDestroy(evUpDone[q]));
        CUDA_CHECK(cudaEventDestroy(evWbDone[q]));
    }
    CUDA_CHECK(cudaStreamDestroy(sUp));
    CUDA_CHECK(cudaStreamDestroy(sLA));
    CUDA_CHECK(cudaStreamDestroy(sBulk));
    CUDA_CHECK(cudaFree(dLoc));
    for (size_t q = 0; q < kDRing; ++q) {
        CUDA_CHECK(cudaFree(dPanel[q]));
    }
    return failure;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    std::vector<double> reconstructed(n * n);

    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int size) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Per-node rank placement: one GPU per local rank (round-robin), and OpenMP threads
    // divided among the ranks sharing a node.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI ranks, %d OpenMP threads/rank, %d CUDA device(s)/node\n",
               size, omp_get_max_threads(), numDevices);
    }

    const size_t nblk = (n + kBlock - 1) / kBlock;
    const size_t N = nblk * kBlock;
    const size_t Lc = localBlockCount(nblk, rank, size);

    // Generate the positive definite matrix (local block columns only)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    PinnedBuffer hLoc = allocPinned(Lc * N * kBlock);
    {
        std::vector<double> B(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        for (size_t lj = 0; lj < Lc; ++lj) {
            generateBlockColumn(hLoc.get() + lj * N * kBlock, B.data(), n, N,
                                lj * size + rank);
        }
    }

    std::vector<double> A_orig;
    if (validate) {
        A_orig = gatherMatrix(hLoc.get(), n, N, nblk, rank, size, false);  // save original
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    // Timed region: decomposition plus transfer of the factor to host memory.
    const int failure = factorizeDistributed(hLoc.get(), N, nblk, rank, size);
    const double t1 = MPI_Wtime();

    if (failure != 0) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }

    const double localElapsed = t1 - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long durationMs = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Outside the timed region: assemble the full factor only when it is needed.
    std::vector<double> A;
    if (printResults || validate) {
        A = gatherMatrix(hLoc.get(), n, N, nblk, rank, size, true);
    }

    if (rank == 0 && printResults) {
        print_results(A, "CholeskyL");
    }

    int rc = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
