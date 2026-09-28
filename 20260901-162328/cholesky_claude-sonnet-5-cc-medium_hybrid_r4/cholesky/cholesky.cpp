#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA blocked ("right-looking") Cholesky decomposition.
//
// The n x n matrix is replicated in device memory on every MPI rank. Block
// rows are distributed across ranks in a cyclic fashion (row-block i is owned
// by rank i % worldSize). For each panel k:
//   1. The owner of the diagonal block factorizes it (small, sequential) and
//      broadcasts it to every rank.
//   2. Every rank that owns a row-block i > k solves the triangular panel
//      update A[i,k] on its GPU; results are broadcast so every rank's
//      replica stays consistent.
//   3. Every rank updates the trailing sub-blocks it owns (A[i,j] -=
//      A[i,k]*A[j,k]^T) entirely on its GPU; these blocks never need to
//      leave the owning rank's device memory until they become a future
//      diagonal or panel block, at which point they are already correct.
// Each of the three steps is dispatched as a single batched GPU kernel
// launch per rank (rather than one launch per block) to minimize launch and
// synchronization overhead. OpenMP parallelizes the host-side work: building
// the per-rank list of trailing-update block descriptors every panel step,
// and the O(n^2)/O(n^3) matrix generation, validation, and output-formatting
// loops.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

namespace {
constexpr int kTile = 16;
// Maximum block size (NB). The panel-solve kernel keeps each thread's row
// segment in a private local array of this size, so it bounds that array
// rather than any shared-memory budget (see panelSolveKernel). NB also
// controls how much of the O(n^3) work lands on the single-threaded host
// diagonal-block factorization (O(n*NB^2) total): larger blocks mean fewer,
// bigger GPU trailing-update/panel-solve launches, but more host-side
// sequential factorization work, so 64 is a practical balance rather than a
// hardware limit.
constexpr int kMaxBlock = 64;
}

// Triangular (forward substitution) panel solve:
//   for each row r in [rowStart, rowStart+isize):
//     for c in [kStart, kStart+ksize):
//       A[r][c] = (A[r][c] - sum_{k'=kStart}^{c-1} A[r][k']*A[c][k']) / A[c][c]
//
// Each thread's own row segment is staged into a private local buffer so the
// O(ksize^2) dependent work per row only issues one global load/store per
// element instead of re-reading it on every later column. The diagonal block
// L_kk is read straight from global memory: it is small enough (<= 512 KiB)
// to stay resident in the GPU's L2 cache, and every thread across the whole
// batched launch reads the exact same addresses, which the cache serves as a
// broadcast, so an explicit shared-memory copy would not be a meaningfully
// larger win yet caps the block size to whatever fits in shared memory.
// Row-blocks owned by a given rank at a fixed panel step k are equally
// spaced (i0, i0+stride, i0+2*stride, ...), so a single grid-y-batched launch
// can solve every owned panel block in one kernel dispatch instead of one
// launch per block: blockIdx.y selects which owned row-block, giving the GPU
// enough independent work in flight to reach good occupancy.
__global__ void panelSolveKernel(double* A, long long n, long long kStart, int ksize,
                                  long long i0, long long stride, long long NBval,
                                  long long nTotal) {
    const long long i = i0 + static_cast<long long>(blockIdx.y) * stride;
    const long long rowStart = i * NBval;
    const int isize = static_cast<int>(min(NBval, nTotal - rowStart));

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= isize) return;
    long long r = rowStart + gidx;

    double rowBuf[kMaxBlock];
    for (int kk = 0; kk < ksize; ++kk) rowBuf[kk] = A[r * n + (kStart + kk)];

    for (int cc = 0; cc < ksize; ++cc) {
        const double* Lrow = A + (kStart + cc) * n + kStart;
        double sum = 0.0;
        for (int kk = 0; kk < cc; ++kk) {
            sum += rowBuf[kk] * Lrow[kk];
        }
        rowBuf[cc] = (rowBuf[cc] - sum) / Lrow[cc];
    }

    for (int kk = 0; kk < ksize; ++kk) A[r * n + (kStart + kk)] = rowBuf[kk];
}

// Gathers every panel block owned by this rank (equally spaced i0, i0+stride,
// ...) into one contiguous buffer, so the whole set can be transferred with a
// single memcpy and a single MPI_Allgatherv instead of one exchange per
// block.
__global__ void packPanelKernel(const double* A, long long n, long long kStart, int ksize,
                                 long long i0, long long stride, long long NBval,
                                 long long nTotal, double* dest) {
    const long long i = i0 + static_cast<long long>(blockIdx.y) * stride;
    const long long rowStart = i * NBval;
    const int isize = static_cast<int>(min(NBval, nTotal - rowStart));

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= isize) return;
    long long r = rowStart + gidx;
    double* dst = dest + static_cast<long long>(blockIdx.y) * NBval * ksize +
                  static_cast<long long>(gidx) * ksize;
    for (int kk = 0; kk < ksize; ++kk) dst[kk] = A[r * n + (kStart + kk)];
}

// Inverse of packPanelKernel: scatters one rank's contribution (as laid out
// by packPanelKernel) back into the replicated matrix.
__global__ void unpackPanelKernel(double* A, long long n, long long kStart, int ksize,
                                   long long i0, long long stride, long long NBval,
                                   long long nTotal, const double* src) {
    const long long i = i0 + static_cast<long long>(blockIdx.y) * stride;
    const long long rowStart = i * NBval;
    const int isize = static_cast<int>(min(NBval, nTotal - rowStart));

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= isize) return;
    long long r = rowStart + gidx;
    const double* s = src + static_cast<long long>(blockIdx.y) * NBval * ksize +
                       static_cast<long long>(gidx) * ksize;
    for (int kk = 0; kk < ksize; ++kk) A[r * n + (kStart + kk)] = s[kk];
}

// Descriptor for one trailing-update block pair, used to batch every pair
// owned by a rank at a given panel step into a single kernel launch
// (blockIdx.z selects the pair).
struct BlockPair {
    long long iStart;
    long long jStart;
    int isize;
    int jsize;
};

// Trailing update (SYRK/GEMM-like): C[i,j] -= A[i,k] * A[j,k]^T
__global__ void trailingUpdateKernel(double* A, long long n, long long kStart, int ksize,
                                      const BlockPair* pairs) {
    const BlockPair p = pairs[blockIdx.z];
    __shared__ double As[kTile][kTile];
    __shared__ double Bs[kTile][kTile];

    long long row = p.iStart + blockIdx.y * kTile + threadIdx.y;
    long long col = p.jStart + blockIdx.x * kTile + threadIdx.x;

    double acc = 0.0;
    for (int t = 0; t < ksize; t += kTile) {
        int kA = t + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < p.iStart + p.isize && kA < ksize) ? A[row * n + (kStart + kA)] : 0.0;

        long long colB = p.jStart + blockIdx.x * kTile + threadIdx.x;
        int kB = t + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (colB < p.jStart + p.jsize && kB < ksize) ? A[colB * n + (kStart + kB)] : 0.0;

        __syncthreads();
#pragma unroll
        for (int e = 0; e < kTile; ++e) {
            acc += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < p.iStart + p.isize && col < p.jStart + p.jsize) {
        A[row * n + col] -= acc;
    }
}

// Selects the GPU this rank should use (round-robin over node-local ranks)
// and forces CUDA context creation immediately. Context creation is a fixed
// (not problem-size-dependent) per-process cost of several hundred
// milliseconds; doing it once up front, before the benchmark timer starts,
// keeps the reported time representative of the actual parallel algorithm.
int selectAndWarmUpDevice() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_free(&shmComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA-capable device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // force context creation now
    return device;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int device) {
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    CUDA_CHECK(cudaSetDevice(device));

    const long long N = static_cast<long long>(n);
    const size_t NB = std::min<size_t>(kMaxBlock, std::max<size_t>(n, 1));
    const size_t numBlocks = (n + NB - 1) / NB;

    auto blockStart = [&](size_t b) { return b * NB; };
    auto blockSize = [&](size_t b) { return static_cast<int>(std::min(NB, n - blockStart(b))); };

    // Row-blocks in [start, numBlocks) owned by rank r (i % worldSize == r)
    // are equally spaced starting at firstOwned(start, r); these helpers
    // compute that first index and the count without needing to iterate.
    auto firstOwned = [&](size_t start, int r) -> long long {
        long long off = (static_cast<long long>(r) - static_cast<long long>(start % worldSize) +
                          worldSize) %
                         worldSize;
        return static_cast<long long>(start) + off;
    };
    auto ownedCountFrom = [&](size_t start, int r) -> size_t {
        long long i0 = firstOwned(start, r);
        if (i0 >= static_cast<long long>(numBlocks)) return 0;
        return static_cast<size_t>((static_cast<long long>(numBlocks) - i0 + worldSize - 1) /
                                    worldSize);
    };

    // Upload the (rank 0 generated) matrix to every rank's device replica.
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    MPI_Bcast(A.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const int numStreams = std::min(8, std::max(1, omp_get_max_threads()));
    std::vector<cudaStream_t> streams(numStreams);
    for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

    // Pinned (page-locked) host staging buffers: cudaMemcpy2DAsync on regular
    // pageable memory silently falls back to a synchronous staged copy, which
    // serializes every panel broadcast; pinning keeps the D2H/H2D transfers
    // genuinely asynchronous and fast.
    double* diagBuf = nullptr;
    CUDA_CHECK(cudaMallocHost(&diagBuf, NB * NB * sizeof(double)));

    // Panel exchange buffers: every rank's set of owned panel blocks for a
    // given step k is packed into one contiguous region (device + pinned
    // host), so the whole column can move with a single memcpy and a single
    // MPI_Allgatherv instead of one exchange per block. Sized for the worst
    // case (all remaining row-blocks packed by one rank).
    const size_t maxPanelElems = numBlocks * NB * NB;
    double *dPanelSend = nullptr, *dPanelRecv = nullptr;
    double *hPanelSend = nullptr, *hPanelRecv = nullptr;
    CUDA_CHECK(cudaMalloc(&dPanelSend, maxPanelElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPanelRecv, maxPanelElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hPanelSend, maxPanelElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hPanelRecv, maxPanelElems * sizeof(double)));

    BlockPair* dPairs = nullptr;
    size_t dPairsCapacity = 0;
    std::vector<BlockPair> hostPairs;
    std::vector<size_t> ownedRows, rowOffsets;

    bool failed = false;
    long long failGlobalIndex = -1;

    for (size_t k = 0; k < numBlocks && !failed; ++k) {
        const long long kStart = static_cast<long long>(blockStart(k));
        const int ksize = blockSize(k);
        const int ownerK = static_cast<int>(k % worldSize);

        // --- Step 1: factorize the diagonal block (owner only), on host ---
        int errInfo[2] = {0, -1};  // {failed?, local failing index}
        if (rank == ownerK) {
            CUDA_CHECK(cudaMemcpy2D(diagBuf, ksize * sizeof(double),
                                     dA + kStart * N + kStart, N * sizeof(double),
                                     ksize * sizeof(double), ksize, cudaMemcpyDeviceToHost));

            // This factorizes a single small (<=256x256) diagonal block with a
            // sequentially-dependent algorithm, so it is run single-threaded
            // on the host; spawning OpenMP thread teams here would cost far
            // more than the tiny amount of work involved.
            for (int jj = 0; jj < ksize && !errInfo[0]; ++jj) {
                double sum = 0.0;
                for (int kk = 0; kk < jj; ++kk) {
                    double v = diagBuf[jj * ksize + kk];
                    sum += v * v;
                }
                const double val = diagBuf[jj * ksize + jj] - sum;
                if (val <= 0.0) {
                    errInfo[0] = 1;
                    errInfo[1] = jj;
                    break;
                }
                const double Ljj = std::sqrt(val);
                diagBuf[jj * ksize + jj] = Ljj;

                for (int ii = jj + 1; ii < ksize; ++ii) {
                    double s = 0.0;
                    for (int kk = 0; kk < jj; ++kk) {
                        s += diagBuf[ii * ksize + kk] * diagBuf[jj * ksize + kk];
                    }
                    diagBuf[ii * ksize + jj] = (diagBuf[ii * ksize + jj] - s) / Ljj;
                }
            }

            if (!errInfo[0]) {
                CUDA_CHECK(cudaMemcpy2D(dA + kStart * N + kStart, N * sizeof(double),
                                         diagBuf, ksize * sizeof(double),
                                         ksize * sizeof(double), ksize, cudaMemcpyHostToDevice));
            }
        }

        MPI_Bcast(errInfo, 2, MPI_INT, ownerK, MPI_COMM_WORLD);
        if (errInfo[0]) {
            failed = true;
            failGlobalIndex = kStart + errInfo[1];
            break;
        }

        MPI_Bcast(diagBuf, ksize * ksize, MPI_DOUBLE, ownerK, MPI_COMM_WORLD);
        if (rank != ownerK) {
            CUDA_CHECK(cudaMemcpy2D(dA + kStart * N + kStart, N * sizeof(double),
                                     diagBuf, ksize * sizeof(double),
                                     ksize * sizeof(double), ksize, cudaMemcpyHostToDevice));
        }

        // --- Step 2: panel solve for owned row-blocks, then exchange ---
        const size_t panelStart = k + 1;
        const size_t remaining = numBlocks - panelStart;
        const size_t myCount = ownedCountFrom(panelStart, rank);
        const long long myI0 = firstOwned(panelStart, rank);

        if (myCount > 0) {
            // Every owned row-block is solved by one batched kernel launch
            // (grid.y = number of owned blocks) so the GPU sees all of the
            // independent per-block work at once instead of trickling small
            // launches through streams.
            const int threads = static_cast<int>(NB);
            dim3 grid(1, static_cast<unsigned int>(myCount));
            panelSolveKernel<<<grid, threads>>>(dA, N, kStart, ksize, myI0,
                                                 static_cast<long long>(worldSize),
                                                 static_cast<long long>(NB), N);
        }

        if (worldSize > 1 && remaining > 0) {
            // Pack every panel block this rank owns into one contiguous
            // region and exchange the whole column with a single
            // MPI_Allgatherv, instead of one Bcast per block.
            if (myCount > 0) {
                dim3 grid(1, static_cast<unsigned int>(myCount));
                packPanelKernel<<<grid, static_cast<int>(NB)>>>(dA, N, kStart, ksize, myI0,
                                                                 static_cast<long long>(worldSize),
                                                                 static_cast<long long>(NB), N,
                                                                 dPanelSend);
            }

            std::vector<int> recvCounts(worldSize), displs(worldSize);
            int totalRecv = 0;
            for (int r = 0; r < worldSize; ++r) {
                const size_t cnt = ownedCountFrom(panelStart, r);
                recvCounts[r] = static_cast<int>(cnt * NB * ksize);
                displs[r] = totalRecv;
                totalRecv += recvCounts[r];
            }
            const int sendCount = recvCounts[rank];

            if (sendCount > 0) {
                CUDA_CHECK(cudaMemcpyAsync(hPanelSend, dPanelSend, sendCount * sizeof(double),
                                            cudaMemcpyDeviceToHost, streams[0]));
                CUDA_CHECK(cudaStreamSynchronize(streams[0]));
            }

            MPI_Allgatherv(sendCount > 0 ? hPanelSend : nullptr, sendCount, MPI_DOUBLE, hPanelRecv,
                            recvCounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            if (totalRecv > 0) {
                CUDA_CHECK(cudaMemcpyAsync(dPanelRecv, hPanelRecv, totalRecv * sizeof(double),
                                            cudaMemcpyHostToDevice, streams[0]));
                CUDA_CHECK(cudaStreamSynchronize(streams[0]));
            }

            for (int r = 0; r < worldSize; ++r) {
                if (r == rank || recvCounts[r] == 0) continue;
                const size_t cntR = recvCounts[r] / (NB * ksize);
                const long long i0R = firstOwned(panelStart, r);
                dim3 grid(1, static_cast<unsigned int>(cntR));
                unpackPanelKernel<<<grid, static_cast<int>(NB), 0, streams[r % streams.size()]>>>(
                    dA, N, kStart, ksize, i0R, static_cast<long long>(worldSize),
                    static_cast<long long>(NB), N, dPanelRecv + displs[r]);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --- Step 3: trailing update for owned (i, j) block pairs, GPU-only ---
        // Row-blocks owned by this rank are equally spaced (myI0, myI0+P, ...),
        // each contributing (i-k) pairs (j = k+1..i); precompute the output
        // offsets so the (i, j) descriptors for every owned row can be filled
        // independently, in parallel, instead of via a sequential push_back.
        ownedRows.clear();
        for (long long i = myI0; i < static_cast<long long>(numBlocks);
             i += static_cast<long long>(worldSize)) {
            ownedRows.push_back(static_cast<size_t>(i));
        }
        rowOffsets.assign(ownedRows.size() + 1, 0);
        for (size_t idx = 0; idx < ownedRows.size(); ++idx) {
            rowOffsets[idx + 1] = rowOffsets[idx] + (ownedRows[idx] - k);
        }
        hostPairs.resize(rowOffsets.back());

#pragma omp parallel for schedule(dynamic)
        for (size_t idx = 0; idx < ownedRows.size(); ++idx) {
            const size_t i = ownedRows[idx];
            size_t out = rowOffsets[idx];
            for (size_t j = k + 1; j <= i; ++j, ++out) {
                BlockPair p;
                p.iStart = static_cast<long long>(blockStart(i));
                p.jStart = static_cast<long long>(blockStart(j));
                p.isize = blockSize(i);
                p.jsize = blockSize(j);
                hostPairs[out] = p;
            }
        }

        if (!hostPairs.empty()) {
            if (hostPairs.size() > dPairsCapacity) {
                if (dPairs) CUDA_CHECK(cudaFree(dPairs));
                dPairsCapacity = hostPairs.size();
                CUDA_CHECK(cudaMalloc(&dPairs, dPairsCapacity * sizeof(BlockPair)));
            }
            CUDA_CHECK(cudaMemcpy(dPairs, hostPairs.data(), hostPairs.size() * sizeof(BlockPair),
                                   cudaMemcpyHostToDevice));

            // One batched kernel launch handles every (i, j) block pair this
            // rank owns for this panel step (blockIdx.z indexes the pair),
            // giving the GPU scheduler the whole trailing update at once.
            dim3 threads(kTile, kTile);
            dim3 blocks((NB + kTile - 1) / kTile, (NB + kTile - 1) / kTile,
                        static_cast<unsigned int>(hostPairs.size()));
            trailingUpdateKernel<<<blocks, threads>>>(dA, N, kStart, ksize, dPairs);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    if (dPairs) CUDA_CHECK(cudaFree(dPairs));

    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFreeHost(diagBuf));
    CUDA_CHECK(cudaFree(dPanelSend));
    CUDA_CHECK(cudaFree(dPanelRecv));
    CUDA_CHECK(cudaFreeHost(hPanelSend));
    CUDA_CHECK(cudaFreeHost(hPanelRecv));

    if (failed) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %lld\n",
                   failGlobalIndex);
        }
        return false;
    }

    // Zero out the upper triangular part (matches the reference sequential
    // implementation's output layout).
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; ++i) {
        for (long long j = i + 1; j < N; ++j) {
            A[i * N + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (kept sequential so the generated matrix is
    // bit-for-bit identical to the original single-threaded implementation).
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T (independent per output element, safe to parallelize)
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
#pragma omp parallel for schedule(static)
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

#pragma omp parallel for reduction(max : maxError, relError) schedule(static)
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badArg = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            badArg = true;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (badArg) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Pick a GPU and force CUDA context creation now, before the benchmark
    // timer starts, so the reported time reflects the algorithm rather than
    // one-time per-process context setup.
    const int device = selectAndWarmUpDevice();

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix (rank 0 only; broadcast happens
    // inside choleskyDecomposition)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }

    if (validate && rank == 0) {
        A_orig = A;  // Save original for validation
    }

    // Perform Cholesky decomposition (collective across all ranks)
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, device);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);

        if (valid) {
            printf("Validation: PASSED\n");
            exitCode = 0;
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
