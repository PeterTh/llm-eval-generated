#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Algorithm: right-looking blocked Cholesky with a one-dimensional block-cyclic
// distribution of block columns over the MPI ranks.  Every rank drives one GPU
// and keeps its own block columns resident in device memory; the heavy lifting
// (panel solve / trailing update) is done by cuBLAS in double precision, the
// small diagonal factorizations by a custom CUDA kernel.  OpenMP shares the
// node's cores between the ranks placed on it and parallelizes the host-side
// assembly of the gathered factor.
//
// The matrix is stored row-major, exactly as in the sequential reference code.
// A row-major lower triangular factor L is bit-for-bit the same memory as a
// column-major upper triangular factor U with U = L^T, and a row-major
// symmetric matrix is the same memory as its column-major counterpart.  All
// device-side work therefore uses column-major storage with CUBLAS_FILL_MODE_UPPER
// and produces A = U^T * U, which is the requested L * L^T with L = U^T.
//
// Communication per step is a broadcast of the nb x nb diagonal block plus an
// allgather of the nb x n block row of U ("the panel").  Both go through pinned
// host staging buffers on a dedicated CUDA stream, so they work with any MPI
// implementation, CUDA-aware or not.  One step of look-ahead is used: the panel
// of step k+1 is factorized and exchanged while the bulk of the trailing update
// of step k runs on the GPU, which hides the transfers behind compute.

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        const cublasStatus_t st_ = (call);                                                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                                            \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__);                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// Distributed context
// ---------------------------------------------------------------------------

struct Dist {
    int rank = 0;
    int nranks = 1;

    size_t n = 0;   // logical matrix size
    int nb = 0;     // block size
    int nblk = 0;   // number of block columns (matrix is padded to nblk * nb)
    size_t np = 0;  // padded matrix size = nblk * nb

    int nlocblk = 0;     // number of block columns owned by this rank
    int maxlocblk = 0;   // maximum over all ranks (uniform buffer sizes)
    size_t loccols = 0;  // nlocblk * nb

    cudaStream_t stream = nullptr;      // compute stream
    cudaStream_t commStream = nullptr;  // panel staging transfers
    cudaEvent_t evTrsm = nullptr;
    cudaEvent_t evD2H = nullptr;
    cudaEvent_t evH2D = nullptr;
    cublasHandle_t blas = nullptr;

    double* dA = nullptr;       // np x loccols, column-major, ld = np
    double* dAorig = nullptr;   // copy of dA for validation (optional)
    double* dD = nullptr;       // nb x nb diagonal block, ld = nb
    double* dW[2] = {nullptr};  // double-buffered nb x np panel (block row of U), ld = nb
    int cur = 0;                // index of the panel currently being consumed
    int* dFlag = nullptr;       // "not positive definite" marker

    double* hD = nullptr;  // pinned nb x nb staging buffer
    double* hSend = nullptr;
    double* hRecv = nullptr;

    std::vector<int> counts;
    std::vector<int> displs;

    // Ownership helpers (block-cyclic over block columns).
    int owner(int b) const { return b % nranks; }
    bool mine(int b) const { return owner(b) == rank; }
    size_t localColOf(int b) const { return (size_t)(b / nranks) * nb; }
    // First locally owned block with index > k, expressed as a local block index.
    int firstLocalBlockAfter(int k) const { return k / nranks + (rank > k % nranks ? 0 : 1); }
    // Global block index of the m-th locally owned block with index > k.
    int globalBlockAfter(int k, int m) const { return (firstLocalBlockAfter(k) + m) * nranks + rank; }
    int numLocalBlocksAfter(int k) const { return std::max(0, nlocblk - firstLocalBlockAfter(k)); }
};

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Unblocked right-looking Cholesky of one nbb x nbb diagonal block (column-major,
// upper triangle) executed by a single thread block.  The pivot row is cached in
// shared memory; the trailing update is spread over the whole thread block.
__global__ void potrfUpperKernel(double* __restrict__ A, const int lda, const int nbb, int* __restrict__ flag,
    const long long globalOffset) {
    extern __shared__ double srow[];
    __shared__ int failed;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nthreads = blockDim.x * blockDim.y;

    if (tid == 0) { failed = 0; }
    __syncthreads();

    for (int j = 0; j < nbb; ++j) {
        if (tid == 0) {
            const double val = A[j + (size_t)j * lda];
            if (val <= 0.0) {
                failed = 1;
                // Record the global index of the first offending diagonal element.
                if (atomicCAS(flag, 0, 1) == 0) { flag[1] = (int)(globalOffset + j); }
            } else {
                A[j + (size_t)j * lda] = sqrt(val);
            }
        }
        __syncthreads();
        if (failed) { return; }

        const double dinv = 1.0 / A[j + (size_t)j * lda];
        for (int c = j + 1 + tid; c < nbb; c += nthreads) {
            const double v = A[j + (size_t)c * lda] * dinv;
            A[j + (size_t)c * lda] = v;
            srow[c] = v;
        }
        __syncthreads();

        // Trailing update: A[r, c] -= srow[r] * srow[c] for j < r <= c < nbb.
        for (int c = j + 1 + (int)threadIdx.y; c < nbb; c += blockDim.y) {
            const double sc = srow[c];
            double* col = A + (size_t)c * lda;
            for (int r = j + 1 + (int)threadIdx.x; r <= c; r += blockDim.x) { col[r] -= srow[r] * sc; }
        }
        __syncthreads();
    }
}

// B[i * n + k] = rand_r()/RAND_MAX - 0.5 for the idx-th call of the reference
// generator.  rand_r advances its LCG three times per call, so the state of call
// idx is reachable by jumping 3*idx steps ahead - computed here per element with
// fast exponentiation, which makes the generation embarrassingly parallel while
// staying bit-identical to the sequential code.
__global__ void generateBKernel(double* __restrict__ B, const size_t count) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= count) { return; }

    // Jump ahead 3*idx steps of s -> 1103515245*s + 12345 starting from seed 42.
    unsigned int a = 1103515245u, c = 12345u;  // current composed transform
    unsigned int ra = 1u, rc = 0u;             // accumulated transform
    unsigned long long e = 3ull * idx;
    while (e) {
        if (e & 1ull) {
            rc = a * rc + c;
            ra = a * ra;
        }
        c = a * c + c;
        a = a * a;
        e >>= 1;
    }
    unsigned int s = ra * 42u + rc;

    // One rand_r() call.
    s = s * 1103515245u + 12345u;
    int result = (int)((s / 65536u) % 2048u);
    s = s * 1103515245u + 12345u;
    result <<= 10;
    result ^= (int)((s / 65536u) % 1024u);
    s = s * 1103515245u + 12345u;
    result <<= 10;
    result ^= (int)((s / 65536u) % 1024u);

    B[idx] = (double)result / 2147483647.0 - 0.5;
}

// Add the diagonal dominance term and turn the padding region into an identity
// block so that the padded matrix factorizes uniformly.
__global__ void finishMatrixKernel(double* __restrict__ A, const size_t np, const size_t n, const int nb,
    const int nranks, const int rank, const int nlocblk, const double diagAdd) {
    const size_t total = np * (size_t)nlocblk * nb;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
        idx += (size_t)blockDim.x * gridDim.x) {
        const size_t lcol = idx / np;
        const size_t row = idx - lcol * np;
        const size_t gcol = ((lcol / nb) * (size_t)nranks + rank) * nb + (lcol % nb);
        if (row >= n || gcol >= n) {
            A[idx] = (row == gcol) ? 1.0 : 0.0;
        } else if (row == gcol) {
            A[idx] += diagAdd;
        }
    }
}

// Zero the strictly lower (column-major) triangle, i.e. the strictly upper
// triangle of the row-major factor L, matching the reference implementation.
__global__ void zeroLowerKernel(double* __restrict__ A, const size_t np, const int nb, const int nranks,
    const int rank, const int nlocblk) {
    const size_t total = np * (size_t)nlocblk * nb;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
        idx += (size_t)blockDim.x * gridDim.x) {
        const size_t lcol = idx / np;
        const size_t row = idx - lcol * np;
        const size_t gcol = ((lcol / nb) * (size_t)nranks + rank) * nb + (lcol % nb);
        if (row > gcol) { A[idx] = 0.0; }
    }
}

// Reduce max absolute and max relative error over the locally owned columns.
__global__ void errorKernel(const double* __restrict__ R, const double* __restrict__ Aorig, const size_t np,
    const size_t n, const int nb, const int nranks, const int rank, const int nlocblk,
    double* __restrict__ out) {
    __shared__ double sMax[256];
    __shared__ double sRel[256];

    const size_t total = np * (size_t)nlocblk * nb;
    double maxErr = 0.0, relErr = 0.0;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
        idx += (size_t)blockDim.x * gridDim.x) {
        const size_t lcol = idx / np;
        const size_t row = idx - lcol * np;
        const size_t gcol = ((lcol / nb) * (size_t)nranks + rank) * nb + (lcol % nb);
        if (row >= n || gcol >= n) { continue; }
        const double a = Aorig[idx];
        const double e = fabs(R[idx] - a);
        maxErr = fmax(maxErr, e);
        relErr = fmax(relErr, e / (fabs(a) + 1e-10));
    }

    sMax[threadIdx.x] = maxErr;
    sRel[threadIdx.x] = relErr;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            sMax[threadIdx.x] = fmax(sMax[threadIdx.x], sMax[threadIdx.x + s]);
            sRel[threadIdx.x] = fmax(sRel[threadIdx.x], sRel[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[2 * blockIdx.x] = sMax[0];
        out[2 * blockIdx.x + 1] = sRel[0];
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

int pickBlockSize(size_t n) {
    // Large enough to keep the cuBLAS updates efficient, small enough to keep
    // the panel critical path and the padding of the last block cheap.
    int nb = (n <= 512) ? 128 : 256;
    while (nb > 64 && (size_t)nb > n) { nb /= 2; }
    return nb;
}

void setupDist(Dist& d, size_t n) {
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nranks);

    d.n = n;
    d.nb = pickBlockSize(n);
    d.nblk = (int)((n + d.nb - 1) / d.nb);
    d.np = (size_t)d.nblk * d.nb;

    d.nlocblk = d.nblk / d.nranks + (d.rank < d.nblk % d.nranks ? 1 : 0);
    d.maxlocblk = (d.nblk + d.nranks - 1) / d.nranks;
    d.loccols = (size_t)d.nlocblk * d.nb;

    // Bind one GPU per rank, distributing the ranks of a node over its devices.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, d.rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    // Share the node's cores between the ranks placed on it instead of letting
    // every rank spawn a full set of OpenMP threads.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_max_threads() / localSize));
    }

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % ndev));

    CUDA_CHECK(cudaStreamCreate(&d.stream));
    CUDA_CHECK(cudaStreamCreate(&d.commStream));
    CUDA_CHECK(cudaEventCreateWithFlags(&d.evTrsm, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&d.evD2H, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&d.evH2D, cudaEventDisableTiming));
    CUBLAS_CHECK(cublasCreate(&d.blas));
    CUBLAS_CHECK(cublasSetStream(d.blas, d.stream));

    const size_t locElems = d.np * std::max<size_t>(d.loccols, 1);
    CUDA_CHECK(cudaMalloc(&d.dA, locElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.dD, (size_t)d.nb * d.nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.dW[0], (size_t)d.nb * d.np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.dW[1], (size_t)d.nb * d.np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.dFlag, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d.dFlag, 0, 2 * sizeof(int)));

    CUDA_CHECK(cudaHostAlloc(&d.hD, (size_t)d.nb * d.nb * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&d.hSend, (size_t)d.nb * std::max<size_t>(d.loccols, d.nb) * sizeof(double),
        cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&d.hRecv, (size_t)d.nb * d.nranks * d.maxlocblk * d.nb * sizeof(double),
        cudaHostAllocDefault));

    d.counts.resize(d.nranks);
    d.displs.resize(d.nranks);
}

void teardownDist(Dist& d) {
    cudaFree(d.dA);
    cudaFree(d.dAorig);
    cudaFree(d.dD);
    cudaFree(d.dW[0]);
    cudaFree(d.dW[1]);
    cudaFree(d.dFlag);
    cudaFreeHost(d.hD);
    cudaFreeHost(d.hSend);
    cudaFreeHost(d.hRecv);
    cublasDestroy(d.blas);
    cudaEventDestroy(d.evTrsm);
    cudaEventDestroy(d.evD2H);
    cudaEventDestroy(d.evH2D);
    cudaStreamDestroy(d.commStream);
    cudaStreamDestroy(d.stream);
}

// ---------------------------------------------------------------------------
// Matrix generation:  A = B * B^T + n * I  with B the reference random matrix
// ---------------------------------------------------------------------------

void generatePositiveDefiniteMatrixDist(Dist& d) {
    const size_t n = d.n;

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));

    const size_t count = n * n;
    const int threads = 256;
    const size_t blocks = (count + threads - 1) / threads;
    generateBKernel<<<(unsigned)std::min<size_t>(blocks, 2147483647u), threads, 0, d.stream>>>(dB, count);
    CUDA_CHECK(cudaGetLastError());

    // B is stored row-major, i.e. it is the column-major matrix Q = B^T.
    // A[i][j] = sum_k B[i,k]*B[j,k] = (Q^T Q)[i,j].
    const double one = 1.0, zero = 0.0;
    for (int lb = 0; lb < d.nlocblk; ++lb) {
        const int b = lb * d.nranks + d.rank;
        const size_t gcol = (size_t)b * d.nb;
        if (gcol >= n) { continue; }
        const int ncols = (int)std::min<size_t>(d.nb, n - gcol);
        CUBLAS_CHECK(cublasDgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, ncols, (int)n, &one, dB, (int)n,
            dB + gcol * n, (int)n, &zero, d.dA + (size_t)lb * d.nb * d.np, (int)d.np));
    }

    finishMatrixKernel<<<1024, 256, 0, d.stream>>>(d.dA, d.np, n, d.nb, d.nranks, d.rank, d.nlocblk, (double)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    CUDA_CHECK(cudaFree(dB));
}

// ---------------------------------------------------------------------------
// Distributed blocked Cholesky
// ---------------------------------------------------------------------------

// Blocked Cholesky of a single nb x nb diagonal block: the small ib x ib
// sub-blocks go through the custom kernel, the rest through cuBLAS.
void potrfDiagBlock(Dist& d, double* blk, const int ld, const int nbb, const long long globalOffset) {
    const int ib = 64;
    const double one = 1.0, minusOne = -1.0;
    const dim3 threads(32, 8);

    for (int j = 0; j < nbb; j += ib) {
        const int jb = std::min(ib, nbb - j);
        double* djj = blk + j + (size_t)j * ld;
        potrfUpperKernel<<<1, threads, (size_t)jb * sizeof(double), d.stream>>>(djj, ld, jb, d.dFlag,
            globalOffset + j);
        CUDA_CHECK(cudaGetLastError());

        const int rest = nbb - j - jb;
        if (rest > 0) {
            double* right = blk + j + (size_t)(j + jb) * ld;
            CUBLAS_CHECK(cublasDtrsm(d.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT, jb, rest, &one, djj, ld, right, ld));
            CUBLAS_CHECK(cublasDsyrk(d.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, rest, jb, &minusOne, right, ld,
                &one, blk + (j + jb) + (size_t)(j + jb) * ld, ld));
        }
    }
}

// Factorize the diagonal block of block column k on its owner and broadcast it.
// Returns false if the matrix is not positive definite.
bool factorDiagonal(Dist& d, const int k) {
    const int owner = d.owner(k);
    long long info[1] = {-1};

    if (d.rank == owner) {
        const size_t lcol = d.localColOf(k);
        double* blk = d.dA + (size_t)k * d.nb + lcol * d.np;
        potrfDiagBlock(d, blk, (int)d.np, d.nb, (long long)k * d.nb);

        CUDA_CHECK(cudaMemcpy2DAsync(d.hD, (size_t)d.nb * sizeof(double), blk, d.np * sizeof(double),
            (size_t)d.nb * sizeof(double), (size_t)d.nb, cudaMemcpyDeviceToHost, d.stream));
        int flag[2];
        CUDA_CHECK(cudaMemcpyAsync(flag, d.dFlag, 2 * sizeof(int), cudaMemcpyDeviceToHost, d.stream));
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        info[0] = flag[0] ? (long long)flag[1] : -1;
    }

    MPI_Bcast(info, 1, MPI_LONG_LONG, owner, MPI_COMM_WORLD);
    if (info[0] >= 0) {
        if (d.rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)info[0]);
        }
        return false;
    }

    MPI_Bcast(d.hD, d.nb * d.nb, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpyAsync(d.dD, d.hD, (size_t)d.nb * d.nb * sizeof(double), cudaMemcpyHostToDevice, d.stream));
    return true;
}

// Triangular solve of the block row k of the locally owned block columns > k
// (only the rows [k*nb, (k+1)*nb) of those columns) and staging of the result
// into the pinned send buffer on the side stream.
void solvePanel(Dist& d, const int k) {
    const int firstLocal = d.firstLocalBlockAfter(k);
    const int nlocAfter = d.numLocalBlocksAfter(k);
    const size_t ncols = (size_t)nlocAfter * d.nb;

    double* B = d.dA + (size_t)k * d.nb + (size_t)firstLocal * d.nb * d.np;
    if (ncols > 0) {
        const double one = 1.0;
        CUBLAS_CHECK(cublasDtrsm(d.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
            CUBLAS_DIAG_NON_UNIT, d.nb, (int)ncols, &one, d.dD, d.nb, B, (int)d.np));
    }
    // Stage the panel on the side stream so that the trailing update can be
    // issued to the compute stream while the transfer is still in flight.  The
    // event also orders the reuse of the pinned buffers against the compute
    // stream for ranks that own nothing beyond block column k.
    CUDA_CHECK(cudaEventRecord(d.evTrsm, d.stream));
    CUDA_CHECK(cudaStreamWaitEvent(d.commStream, d.evTrsm, 0));
    if (ncols > 0) {
        CUDA_CHECK(cudaMemcpy2DAsync(d.hSend, (size_t)d.nb * sizeof(double), B, d.np * sizeof(double),
            (size_t)d.nb * sizeof(double), ncols, cudaMemcpyDeviceToHost, d.commStream));
    }
    CUDA_CHECK(cudaEventRecord(d.evD2H, d.commStream));
}

// Exchange the panel of step k between all ranks, reorder the per-rank
// contributions into global column order and upload the panel into the
// currently unused device panel buffer.
void exchangePanel(Dist& d, const int k) {
    for (int r = 0; r < d.nranks; ++r) {
        const int firstR = k / d.nranks + (r > k % d.nranks ? 0 : 1);
        const int nlocR = d.nblk / d.nranks + (r < d.nblk % d.nranks ? 1 : 0);
        d.counts[r] = std::max(0, nlocR - firstR) * d.nb * d.nb;
        d.displs[r] = (r == 0) ? 0 : d.displs[r - 1] + d.counts[r - 1];
    }

    // The GPU is busy with the trailing update while this transfer runs.
    CUDA_CHECK(cudaEventSynchronize(d.evD2H));
    MPI_Allgatherv(d.hSend, d.counts[d.rank], MPI_DOUBLE, d.hRecv, d.counts.data(), d.displs.data(), MPI_DOUBLE,
        MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventSynchronize(d.evH2D));  // previous upload must have drained hRecv

    // Scatter the per-rank contributions straight into their global column
    // positions in the device panel: the blocks of a rank are separated by a
    // constant stride, so one strided copy per rank is enough.
    const size_t blockBytes = (size_t)d.nb * d.nb * sizeof(double);
    const size_t colOffset = (size_t)(k + 1) * d.nb;
    double* dst = d.dW[1 - d.cur];
    for (int r = 0; r < d.nranks; ++r) {
        const int firstR = k / d.nranks + (r > k % d.nranks ? 0 : 1);
        const int nlocR = d.nblk / d.nranks + (r < d.nblk % d.nranks ? 1 : 0);
        const int nblocksR = nlocR - firstR;
        if (nblocksR <= 0) { continue; }
        const int b0 = firstR * d.nranks + r;
        CUDA_CHECK(cudaMemcpy2DAsync(dst + ((size_t)b0 * d.nb - colOffset) * d.nb,
            (size_t)d.nranks * blockBytes, d.hRecv + d.displs[r], blockBytes, blockBytes, (size_t)nblocksR,
            cudaMemcpyHostToDevice, d.commStream));
    }
    CUDA_CHECK(cudaEventRecord(d.evH2D, d.commStream));
    CUDA_CHECK(cudaStreamWaitEvent(d.stream, d.evH2D, 0));
    d.cur = 1 - d.cur;
}

// Trailing update of the locally owned block columns b > kFrom using panel k,
// restricted to the global rows [row0, min(row1, diagonal of b)).  The symmetric
// rank-nb update of the diagonal block is applied when row1 reaches beyond it.
void applyUpdate(Dist& d, const int k, const int kFrom, const size_t row0, const size_t row1) {
    const double minusOne = -1.0, one = 1.0;
    const size_t colOffset = (size_t)(k + 1) * d.nb;
    const int nlocAfter = d.numLocalBlocksAfter(kFrom);

    for (int m = 0; m < nlocAfter; ++m) {
        const int b = d.globalBlockAfter(kFrom, m);
        const size_t lcol = d.localColOf(b);
        const size_t diag = (size_t)b * d.nb;
        double* C = d.dA + lcol * d.np;
        const double* W = d.dW[d.cur];
        const double* Wb = W + (diag - colOffset) * d.nb;

        const size_t rEnd = std::min(row1, diag);
        if (rEnd > row0) {
            CUBLAS_CHECK(cublasDgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)(rEnd - row0), d.nb, d.nb, &minusOne,
                W + (row0 - colOffset) * d.nb, d.nb, Wb, d.nb, &one, C + row0, (int)d.np));
        }
        if (row1 > diag) {
            CUBLAS_CHECK(cublasDsyrk(d.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, d.nb, d.nb, &minusOne, Wb, d.nb,
                &one, C + diag, (int)d.np));
        }
    }
}

// Touch every kernel used by the factorization once so that CUDA's lazy module
// loading and cuBLAS' internal initialization do not land in the timed region.
void warmupGpu(Dist& d) {
    const int m = d.nb;
    const size_t elems = 3 * (size_t)m * m;
    double* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, elems * sizeof(double)));

    std::vector<double> h(elems, 0.0);
    for (int i = 0; i < m; ++i) { h[i + (size_t)i * m] = 1.0; }
    CUDA_CHECK(cudaMemcpyAsync(scratch, h.data(), elems * sizeof(double), cudaMemcpyHostToDevice, d.stream));

    const double one = 1.0, minusOne = -1.0, zero = 0.0;
    potrfDiagBlock(d, scratch, m, m, 0);
    CUBLAS_CHECK(cublasDtrsm(d.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
        m, m, &one, scratch, m, scratch + (size_t)m * m, m));
    CUBLAS_CHECK(cublasDsyrk(d.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, m, m, &minusOne,
        scratch + (size_t)m * m, m, &one, scratch + 2 * (size_t)m * m, m));
    CUBLAS_CHECK(cublasDgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, m, m, m, &minusOne, scratch, m,
        scratch + (size_t)m * m, m, &zero, scratch + 2 * (size_t)m * m, m));
    CUDA_CHECK(cudaMemcpy2DAsync(d.hSend, (size_t)m * sizeof(double), scratch, (size_t)m * sizeof(double),
        (size_t)m * sizeof(double), (size_t)m, cudaMemcpyDeviceToHost, d.commStream));
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    CUDA_CHECK(cudaStreamSynchronize(d.commStream));
    CUDA_CHECK(cudaFree(scratch));

    // Also get the collectives used per step out of their first-call path.
    MPI_Bcast(d.hD, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::fill(d.counts.begin(), d.counts.end(), 1);
    for (int r = 0; r < d.nranks; ++r) { d.displs[r] = r; }
    MPI_Allgatherv(d.hSend, 1, MPI_DOUBLE, d.hRecv, d.counts.data(), d.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}

bool choleskyDecompositionDist(Dist& d) {
    if (!factorDiagonal(d, 0)) { return false; }

    solvePanel(d, 0);
    exchangePanel(d, 0);

    for (int k = 0; k < d.nblk - 1; ++k) {
        const size_t rowA = (size_t)(k + 1) * d.nb;

        // Look-ahead: bring the next block row up to date so that its diagonal
        // block can be factorized and its panel sent while the bulk of this
        // step's update is still running on the GPU.
        if (d.mine(k + 1)) {
            const double minusOne = -1.0, one = 1.0;
            const size_t lcol = d.localColOf(k + 1);
            CUBLAS_CHECK(cublasDsyrk(d.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, d.nb, d.nb, &minusOne,
                d.dW[d.cur], d.nb, &one, d.dA + rowA + lcol * d.np, (int)d.np));
        }
        applyUpdate(d, k, k + 1, rowA, rowA + d.nb);

        if (!factorDiagonal(d, k + 1)) { return false; }
        solvePanel(d, k + 1);

        // Bulk of the trailing update of step k (everything below block row k+1).
        applyUpdate(d, k, k + 1, rowA + d.nb, d.np);

        exchangePanel(d, k + 1);
    }

    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    return true;
}

// ---------------------------------------------------------------------------
// Result assembly and validation
// ---------------------------------------------------------------------------

// Zero the (row-major) upper triangle and gather the factor into a full
// row-major n x n matrix on every rank.
void gatherFactor(Dist& d, std::vector<double>& full) {
    zeroLowerKernel<<<1024, 256, 0, d.stream>>>(d.dA, d.np, d.nb, d.nranks, d.rank, d.nlocblk);
    CUDA_CHECK(cudaGetLastError());

    const size_t n = d.n;
    const size_t stride = (size_t)d.maxlocblk * d.nb;  // uniform column count per rank
    std::vector<double> sendbuf(n * stride, 0.0);
    std::vector<double> recvbuf((size_t)d.nranks * n * stride);

    for (int lb = 0; lb < d.nlocblk; ++lb) {
        const int b = lb * d.nranks + d.rank;
        const size_t gcol = (size_t)b * d.nb;
        if (gcol >= n) { continue; }
        const size_t ncols = std::min<size_t>(d.nb, n - gcol);
        CUDA_CHECK(cudaMemcpy2DAsync(sendbuf.data() + (size_t)lb * d.nb * n, n * sizeof(double),
            d.dA + (size_t)lb * d.nb * d.np, d.np * sizeof(double), n * sizeof(double), ncols,
            cudaMemcpyDeviceToHost, d.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(d.stream));

    MPI_Allgather(sendbuf.data(), (int)(n * stride), MPI_DOUBLE, recvbuf.data(), (int)(n * stride), MPI_DOUBLE,
        MPI_COMM_WORLD);

    full.resize(n * n);
    const int nblk = d.nblk;
#pragma omp parallel for schedule(static)
    for (int b = 0; b < nblk; ++b) {
        const size_t gcol = (size_t)b * d.nb;
        if (gcol >= n) { continue; }
        const size_t ncols = std::min<size_t>(d.nb, n - gcol);
        const int r = b % d.nranks;
        const size_t lb = b / d.nranks;
        const double* src = recvbuf.data() + (size_t)r * n * stride + lb * d.nb * n;
        memcpy(full.data() + gcol * n, src, ncols * n * sizeof(double));
    }
}

bool validateCholeskyDist(Dist& d, const std::vector<double>& full) {
    const size_t n = d.n;

    double* dFull = nullptr;
    double* dR = nullptr;
    CUDA_CHECK(cudaMalloc(&dFull, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dR, d.np * std::max<size_t>(d.loccols, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(dFull, full.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, d.stream));
    CUDA_CHECK(cudaMemsetAsync(dR, 0, d.np * std::max<size_t>(d.loccols, 1) * sizeof(double), d.stream));

    // full is row-major L, i.e. column-major U; reconstruct L*L^T = U^T*U.
    const double one = 1.0, zero = 0.0;
    for (int lb = 0; lb < d.nlocblk; ++lb) {
        const int b = lb * d.nranks + d.rank;
        const size_t gcol = (size_t)b * d.nb;
        if (gcol >= n) { continue; }
        const int ncols = (int)std::min<size_t>(d.nb, n - gcol);
        CUBLAS_CHECK(cublasDgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, ncols, (int)n, &one, dFull, (int)n,
            dFull + gcol * n, (int)n, &zero, dR + (size_t)lb * d.nb * d.np, (int)d.np));
    }

    const int nblocks = 256;
    double* dErr = nullptr;
    CUDA_CHECK(cudaMalloc(&dErr, 2 * nblocks * sizeof(double)));
    errorKernel<<<nblocks, 256, 0, d.stream>>>(dR, d.dAorig, d.np, n, d.nb, d.nranks, d.rank, d.nlocblk, dErr);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> hErr(2 * nblocks);
    CUDA_CHECK(cudaMemcpyAsync(hErr.data(), dErr, 2 * nblocks * sizeof(double), cudaMemcpyDeviceToHost, d.stream));
    CUDA_CHECK(cudaStreamSynchronize(d.stream));

    double local[2] = {0.0, 0.0};
    for (int i = 0; i < nblocks; ++i) {
        local[0] = std::max(local[0], hErr[2 * i]);
        local[1] = std::max(local[1], hErr[2 * i + 1]);
    }
    double global[2];
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    cudaFree(dFull);
    cudaFree(dR);
    cudaFree(dErr);

    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) { printf("Validation failed: relative error too large\n"); }
    }
    return global[1] <= 1e-6;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) { printUsage(argv[0]); }
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (n == 0) {
        if (rank == 0) { printf("Computation time: 0 ms\n"); }
        MPI_Finalize();
        return 0;
    }

    Dist d;
    setupDist(d, n);

    // Generate positive definite matrix (distributed, one GPU per rank)
    if (rank == 0) { printf("Generating positive definite matrix...\n"); }
    generatePositiveDefiniteMatrixDist(d);

    if (validate) {
        const size_t bytes = d.np * std::max<size_t>(d.loccols, 1) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d.dAorig, bytes));
        CUDA_CHECK(cudaMemcpyAsync(d.dAorig, d.dA, bytes, cudaMemcpyDeviceToDevice, d.stream));
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
    }

    // Perform Cholesky decomposition
    if (rank == 0) { printf("Computing Cholesky decomposition...\n"); }
    warmupGpu(d);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const bool success = choleskyDecompositionDist(d);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) { printf("Cholesky decomposition failed\n"); }
        teardownDist(d);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> full;
    if (printResults || validate) { gatherFactor(d, full); }

    // Print results for external validation
    if (printResults && rank == 0) { print_results(full, "CholeskyL"); }

    int status = 0;
    if (validate) {
        if (rank == 0) { printf("Validating result...\n"); }
        const bool valid = validateCholeskyDist(d, full);
        if (rank == 0) { printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
        status = valid ? 0 : 1;
    }

    teardownDist(d);
    MPI_Finalize();
    return status;
}
