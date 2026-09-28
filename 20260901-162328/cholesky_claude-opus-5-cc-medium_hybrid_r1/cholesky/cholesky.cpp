#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Decomposes a positive definite matrix A into L * L^T where L is lower triangular.
//
// The blocked right-looking algorithm is used: the matrix is split into block
// columns that are distributed cyclically over the MPI ranks, one GPU per rank.
// Every rank keeps its block columns resident on its GPU in column major layout,
// factorizes the panels it owns with cuSOLVER/cuBLAS and applies the trailing
// updates as local DGEMMs.  A one step lookahead lets the owner of the next block
// column factorize it on a high priority stream while the bulk updates of the
// current step are still running, so the panel exchange overlaps with compute.
// Panels are exchanged hierarchically: through a node local shared memory window
// inside a node and along a pipelined ring between the node leaders.
// OpenMP parallelizes the host side work (result assembly and error reductions).

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t status_ = (call);                                                                            \
        if (status_ != cudaSuccess) {                                                                                  \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(status_), __FILE__, __LINE__);                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        const cublasStatus_t status_ = (call);                                                                         \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                                                        \
            printf("cuBLAS error %d at %s:%d\n", (int)status_, __FILE__, __LINE__);                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

#define CUSOLVER_CHECK(call)                                                                                           \
    do {                                                                                                               \
        const cusolverStatus_t status_ = (call);                                                                       \
        if (status_ != CUSOLVER_STATUS_SUCCESS) {                                                                      \
            printf("cuSOLVER error %d at %s:%d\n", (int)status_, __FILE__, __LINE__);                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// Block size of the blocked algorithm (number of columns per distributed block).
static constexpr size_t kBlockSize = 256;

// Number of panel buffers used to pipeline factorization, communication and updates.
static constexpr int kPanelSlots = 3;

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Add a constant to the diagonal entries A(c,c) of a block column.
// The block column is stored column major with leading dimension ld; the first
// column of the block is the global column colStart.
__global__ void addDiagonalKernel(double* block, size_t ld, size_t colStart, size_t cols, double value) {
    const size_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < cols) { block[o * ld + (colStart + o)] += value; }
}

// Zero the strictly upper triangular part (row < column) of a block column.
__global__ void zeroUpperKernel(double* block, size_t ld, size_t colStart, size_t cols, size_t rows) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t o = blockIdx.y;
    if (o >= cols) { return; }
    const size_t limit = colStart + o; // rows [0, limit) are above the diagonal
    if (row < limit && row < rows) { block[o * ld + row] = 0.0; }
}

// Per-thread-block partial maxima of the absolute and relative reconstruction error.
__global__ void errorReduceKernel(const double* rec, const double* orig, size_t count, double* absOut,
                                  double* relOut) {
    __shared__ double sAbs[256];
    __shared__ double sRel[256];

    double localAbs = 0.0;
    double localRel = 0.0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const double a = orig[i];
        const double err = fabs(rec[i] - a);
        localAbs = fmax(localAbs, err);
        localRel = fmax(localRel, err / (fabs(a) + 1e-10));
    }
    sAbs[threadIdx.x] = localAbs;
    sRel[threadIdx.x] = localRel;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sAbs[threadIdx.x] = fmax(sAbs[threadIdx.x], sAbs[threadIdx.x + s]);
            sRel[threadIdx.x] = fmax(sRel[threadIdx.x], sRel[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        absOut[blockIdx.x] = sAbs[0];
        relOut[blockIdx.x] = sRel[0];
    }
}

// ---------------------------------------------------------------------------
// Distribution helpers
// ---------------------------------------------------------------------------

struct Distribution {
    size_t n = 0;         // matrix dimension
    size_t nb = 0;        // block size
    size_t numBlocks = 0; // number of block columns
    int rank = 0;
    int size = 1;
    size_t localBlocks = 0; // number of block columns owned by this rank

    int owner(size_t blockIdx) const { return (int)(blockIdx % (size_t)size); }
    size_t localIndex(size_t blockIdx) const { return blockIdx / (size_t)size; }
    size_t blockStart(size_t blockIdx) const { return blockIdx * nb; }
    size_t blockCols(size_t blockIdx) const { return std::min(nb, n - blockIdx * nb); }
    // Offset (in elements) of a locally owned block column inside the local buffer.
    size_t localOffset(size_t blockIdx) const { return localIndex(blockIdx) * nb * n; }
};

// Topology of the run: the ranks of a node exchange panels through a shared
// memory window, only the node leaders exchange them over the network.
struct Cluster {
    int rank = 0;
    int size = 1;
    MPI_Comm node = MPI_COMM_NULL;    // ranks sharing memory with this rank
    MPI_Comm leaders = MPI_COMM_NULL; // one rank per node (MPI_COMM_NULL elsewhere)
    int nodeRank = 0;
    int nodeSize = 1;
    int nodeId = 0;
    int numNodes = 1;
    std::vector<int> nodeOfRank; // node id of every world rank
};

// State shared by the GPU phases of the benchmark.
struct GpuContext {
    cudaStream_t stream = nullptr;      // trailing updates (low priority)
    cudaStream_t panelStream = nullptr; // panel factorization / transfers (high priority)
    cublasHandle_t blas = nullptr;
    cublasHandle_t blasPanel = nullptr;
    cusolverDnHandle_t solver = nullptr;
    double* work = nullptr; // cuSOLVER workspace
    int workSize = 0;
    int* devInfo = nullptr;
};

// ---------------------------------------------------------------------------
// Matrix generation (distributed over the block columns)
// ---------------------------------------------------------------------------

// Generates the block columns of A = B * B^T + n*I owned by this rank directly
// on the device.  B is the same pseudo random matrix as in the sequential code.
void generatePositiveDefiniteMatrix(double* dA, const Distribution& dist, GpuContext& gpu) {
    const size_t n = dist.n;

    // B is row major; interpreting the same memory column major yields B^T, so
    // A = B * B^T becomes (B^T)^T * (B^T) in column major terms.
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));

    const double one = 1.0;
    const double zero = 0.0;
    for (size_t j = dist.rank; j < dist.numBlocks; j += (size_t)dist.size) {
        const size_t jb = dist.blockStart(j);
        const size_t cols = dist.blockCols(j);
        double* dst = dA + dist.localOffset(j);
        CUBLAS_CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)cols, (int)n, &one, dB, (int)n,
                                 dB + jb * n, (int)n, &zero, dst, (int)n));
        const int threads = 128;
        addDiagonalKernel<<<(unsigned int)((cols + threads - 1) / threads), threads, 0, gpu.stream>>>(
            dst, n, jb, cols, (double)n);
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    CUDA_CHECK(cudaFree(dB));
}

// ---------------------------------------------------------------------------
// Factorization
// ---------------------------------------------------------------------------

// Factorizes the (already fully updated) block column k on its owner and packs
// the resulting panel (rows k*nb..n-1) contiguously into the device panel
// buffer.  Purely asynchronous: no host synchronization happens here so that
// the caller can keep queueing work.
static void factorPanel(size_t k, const Distribution& dist, GpuContext& gpu, double* dA, double* dPanel) {
    const size_t n = dist.n;
    const size_t kb = dist.blockStart(k);
    const size_t cols = dist.blockCols(k);
    const size_t rows = n - kb;

    double* block = dA + dist.localOffset(k); // column major, ld = n
    double* diag = block + kb;                // diagonal block A(kb:kb+cols, kb:kb+cols)

    CUSOLVER_CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, (int)cols, diag, (int)n, gpu.work,
                                    gpu.workSize, gpu.devInfo));

    if (rows > cols) {
        const double one = 1.0;
        CUBLAS_CHECK(cublasDtrsm(gpu.blasPanel, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                 CUBLAS_DIAG_NON_UNIT, (int)(rows - cols), (int)cols, &one, diag, (int)n,
                                 block + kb + cols, (int)n));
    }

    CUDA_CHECK(cudaMemcpy2DAsync(dPanel, rows * sizeof(double), block + kb, n * sizeof(double), rows * sizeof(double),
                                 cols, cudaMemcpyDeviceToDevice, gpu.panelStream));
}

// Applies the update of panel k to the locally owned block column j (j > k):
//   A(jb:n, j) -= panel(jb:n, :) * panel(jb:jb+cols_j, :)^T
static void updateBlockColumn(size_t j, size_t k, const Distribution& dist, cublasHandle_t handle, double* dA,
                              const double* dPanel) {
    const size_t n = dist.n;
    const size_t kb = dist.blockStart(k);
    const size_t kcols = dist.blockCols(k);
    const size_t panelRows = n - kb;

    const size_t jb = dist.blockStart(j);
    const size_t jcols = dist.blockCols(j);
    const size_t rows = n - jb;

    const double minusOne = -1.0;
    const double one = 1.0;
    double* dst = dA + dist.localOffset(j) + jb; // rows jb..n-1 of the block column
    const double* left = dPanel + (jb - kb);     // panel rows jb..n-1
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, (int)rows, (int)jcols, (int)kcols, &minusOne, left,
                             (int)panelRows, left, (int)panelRows, &one, dst, (int)n));
}

// Pipelined ring broadcast between the node leaders.
//
// The panel is forwarded chunk by chunk along the ring starting at the leader of
// the owning node, which sustains close to the point to point bandwidth (the
// generic MPI_Bcast reaches only a fraction of it for panels of several
// megabytes).
static void ringBroadcast(double* buffer, size_t count, int root, MPI_Comm comm, int rank, int size) {
    if (size < 2) { return; }

    constexpr size_t chunk = 64 * 1024; // 512 KB per message
    const int pos = (rank - root + size) % size;
    const int from = (root + pos - 1) % size;
    const int to = (root + pos + 1) % size;
    const bool isRoot = (pos == 0);
    const bool isLast = (pos == size - 1);

    MPI_Request send = MPI_REQUEST_NULL;
    int tag = 0;
    for (size_t off = 0; off < count; off += chunk, ++tag) {
        const size_t len = std::min(chunk, count - off);
        if (!isRoot) { MPI_Recv(buffer + off, (int)len, MPI_DOUBLE, from, tag, comm, MPI_STATUS_IGNORE); }
        if (!isLast) {
            if (send != MPI_REQUEST_NULL) { MPI_Wait(&send, MPI_STATUS_IGNORE); }
            MPI_Isend(buffer + off, (int)len, MPI_DOUBLE, to, tag, comm, &send);
        }
    }
    if (send != MPI_REQUEST_NULL) { MPI_Wait(&send, MPI_STATUS_IGNORE); }
}

// Hands the freshly factorized panel of the owner to all other ranks.
//
// Inside a node the panel is exchanged through a shared memory window: the owner
// copies it there once and every other rank uploads it to its own GPU directly
// from that buffer, which avoids the copy in/copy out of a message based
// broadcast.  Between nodes only the leaders exchange the panel.  hPanel holds
// dataCount panel values plus one trailing element carrying the factorization
// status, so a failed panel is detected by every rank.
//
// The caller guarantees (via the preceding barrier) that no rank is still
// reading the shared slot.
static void publishPanel(double* hPanel, double* dPanel, size_t dataCount, int root, const Cluster& cl,
                         GpuContext& gpu) {
    const bool isOwner = (cl.rank == root);

    if (isOwner) {
        CUDA_CHECK(cudaMemcpyAsync(hPanel, dPanel, dataCount * sizeof(double), cudaMemcpyDeviceToHost,
                                   gpu.panelStream));
        int info = 0;
        CUDA_CHECK(cudaMemcpyAsync(&info, gpu.devInfo, sizeof(int), cudaMemcpyDeviceToHost, gpu.panelStream));
        CUDA_CHECK(cudaStreamSynchronize(gpu.panelStream));
        hPanel[dataCount] = (double)info;
    }
    MPI_Barrier(cl.node); // the panel is now visible to every rank of the owning node

    if (cl.numNodes > 1) {
        if (cl.leaders != MPI_COMM_NULL) {
            ringBroadcast(hPanel, dataCount + 1, cl.nodeOfRank[root], cl.leaders, cl.nodeId, cl.numNodes);
        }
        MPI_Barrier(cl.node); // the panel received by the leader is visible node wide
    }

    if (!isOwner) {
        CUDA_CHECK(cudaMemcpyAsync(dPanel, hPanel, dataCount * sizeof(double), cudaMemcpyHostToDevice,
                                   gpu.panelStream));
    }
}

// Blocked right-looking Cholesky over the distributed block columns.
//
// The panel path (lookahead update, factorization, transfers) runs on a high
// priority stream, the bulk trailing updates on a second stream, so that the
// critical path of step k+1 is not queued behind the trailing updates of step k.
// CUDA events express the cross stream dependencies; the panel buffers are
// triple buffered so that a panel can be received while the previous one is
// still being consumed.
bool choleskyDecomposition(double* dA, const Distribution& dist, const Cluster& cl, GpuContext& gpu,
                           double* const dPanel[kPanelSlots], double* const hPanel[kPanelSlots],
                           size_t& failColumn) {
    const size_t n = dist.n;
    const size_t numBlocks = dist.numBlocks;

    auto panelCount = [&](size_t k) { return (n - dist.blockStart(k)) * dist.blockCols(k) + 1; };
    auto owns = [&](size_t j) { return dist.owner(j) == dist.rank; };

    cudaEvent_t panelReady[kPanelSlots]; // panel slot filled on the panel stream
    cudaEvent_t updateDone[kPanelSlots]; // trailing updates of a step finished
    bool updateDoneValid[kPanelSlots] = {false, false, false};
    for (int i = 0; i < kPanelSlots; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&panelReady[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&updateDone[i], cudaEventDisableTiming));
    }
    cudaEvent_t nextColumnReady; // lookahead column fully updated on the update stream
    CUDA_CHECK(cudaEventCreateWithFlags(&nextColumnReady, cudaEventDisableTiming));
    bool nextColumnValid = false;

    bool failed = false;

    // Factorize and distribute the very first panel.
    if (owns(0)) { factorPanel(0, dist, gpu, dA, dPanel[0]); }
    publishPanel(hPanel[0], dPanel[0], panelCount(0) - 1, dist.owner(0), cl, gpu);
    CUDA_CHECK(cudaEventRecord(panelReady[0], gpu.panelStream));
    if (hPanel[0][panelCount(0) - 1] != 0.0) {
        failColumn = (size_t)hPanel[0][panelCount(0) - 1] - 1;
        failed = true;
    }

    for (size_t k = 0; !failed && k < numBlocks; ++k) {
        const size_t kn = k + 1;
        const int cur = (int)(k % kPanelSlots);
        const int nxt = (int)(kn % kPanelSlots);

        CUDA_CHECK(cudaStreamWaitEvent(gpu.stream, panelReady[cur], 0));

        // Lookahead: the owner of the next block column updates and factorizes it
        // first, so that its exchange overlaps with the remaining trailing updates.
        if (kn < numBlocks) {
            // The target panel slot was last read by the trailing updates of step
            // k-2; neither the owner nor the upload of a receiver may overwrite it
            // before those have finished.
            if (updateDoneValid[nxt]) { CUDA_CHECK(cudaStreamWaitEvent(gpu.panelStream, updateDone[nxt], 0)); }
            if (owns(kn)) {
                if (nextColumnValid) { CUDA_CHECK(cudaStreamWaitEvent(gpu.panelStream, nextColumnReady, 0)); }
                updateBlockColumn(kn, k, dist, gpu.blasPanel, dA, dPanel[cur]);
                factorPanel(kn, dist, gpu, dA, dPanel[nxt]);
            }
        }

        // Remaining trailing updates (block columns k+2 and beyond).  They are
        // queued before the rank enters the panel exchange, so the GPU stays busy
        // while the panel travels.
        nextColumnValid = false;
        for (size_t j = (size_t)dist.rank; j < numBlocks; j += (size_t)dist.size) {
            if (j < k + 2) { continue; }
            updateBlockColumn(j, k, dist, gpu.blas, dA, dPanel[cur]);
            if (j == k + 2) {
                // This rank owns the lookahead column of the next step.
                CUDA_CHECK(cudaEventRecord(nextColumnReady, gpu.stream));
                nextColumnValid = true;
            }
        }
        CUDA_CHECK(cudaEventRecord(updateDone[cur], gpu.stream));
        updateDoneValid[cur] = true;

        if (kn < numBlocks) {
            const size_t count = panelCount(kn);
            // Nobody may read the shared slot any more once the owner refills it:
            // the upload of its previous content has to be complete everywhere.
            CUDA_CHECK(cudaEventSynchronize(panelReady[nxt]));
            MPI_Barrier(cl.node);
            publishPanel(hPanel[nxt], dPanel[nxt], count - 1, dist.owner(kn), cl, gpu);
            CUDA_CHECK(cudaEventRecord(panelReady[nxt], gpu.panelStream));
            if (hPanel[nxt][count - 1] != 0.0) {
                failColumn = dist.blockStart(kn) + (size_t)hPanel[nxt][count - 1] - 1;
                failed = true;
            }
        }
    }

    for (int i = 0; i < kPanelSlots; ++i) {
        CUDA_CHECK(cudaEventDestroy(panelReady[i]));
        CUDA_CHECK(cudaEventDestroy(updateDone[i]));
    }
    CUDA_CHECK(cudaEventDestroy(nextColumnReady));

    if (failed) {
        CUDA_CHECK(cudaStreamSynchronize(gpu.panelStream));
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.panelStream));

    // Zero out the upper triangular part of the locally owned block columns.
    for (size_t j = (size_t)dist.rank; j < numBlocks; j += (size_t)dist.size) {
        const size_t jb = dist.blockStart(j);
        const size_t cols = dist.blockCols(j);
        const int threads = 256;
        const dim3 grid((unsigned int)((jb + cols + threads - 1) / threads), (unsigned int)cols);
        zeroUpperKernel<<<grid, threads, 0, gpu.stream>>>(dA + dist.localOffset(j), n, jb, cols, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    return true;
}

// Touches every library entry point used by the factorization so that the one
// time initialization of cuBLAS/cuSOLVER (module loading, workspace setup) does
// not end up in the measured region.
void warmUpGpu(const Distribution& dist, GpuContext& gpu, double* hPanel) {
    const int m = (int)std::min<size_t>(64, dist.nb);
    double* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, (size_t)m * m * sizeof(double)));

    const double one = 1.0;
    const double zero = 0.0;
    for (cudaStream_t stream : {gpu.stream, gpu.panelStream}) {
        cublasHandle_t handle = (stream == gpu.stream) ? gpu.blas : gpu.blasPanel;
        CUDA_CHECK(cudaMemsetAsync(scratch, 0, (size_t)m * m * sizeof(double), stream));
        addDiagonalKernel<<<1, 128, 0, stream>>>(scratch, m, 0, m, 1.0);
        CUSOLVER_CHECK(
            cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, m, scratch, m, gpu.work, gpu.workSize, gpu.devInfo));
        CUBLAS_CHECK(cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                 CUBLAS_DIAG_NON_UNIT, m, m, &one, scratch, m, scratch, m));
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, m, m, &one, scratch, m, scratch, m, &zero,
                                 scratch, m));
        zeroUpperKernel<<<dim3(1, (unsigned int)m), 128, 0, stream>>>(scratch, m, 0, m, m);
        CUDA_CHECK(cudaMemcpyAsync(hPanel, scratch, sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(scratch, hPanel, sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(scratch));
}

// ---------------------------------------------------------------------------
// Result assembly and validation
// ---------------------------------------------------------------------------

// Collects the distributed (column major) block columns into the full row major
// matrix on rank 0.
void gatherMatrix(const double* dA, const Distribution& dist, GpuContext& gpu, std::vector<double>& A) {
    const size_t n = dist.n;
    std::vector<double> buffer(n * dist.nb);
    if (dist.rank == 0) { A.resize(n * n); }

    for (size_t j = 0; j < dist.numBlocks; ++j) {
        const size_t jb = dist.blockStart(j);
        const size_t cols = dist.blockCols(j);
        const int owner = dist.owner(j);

        if (dist.rank == owner) {
            CUDA_CHECK(cudaMemcpyAsync(buffer.data(), dA + dist.localOffset(j), n * cols * sizeof(double),
                                       cudaMemcpyDeviceToHost, gpu.stream));
            CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
            if (owner != 0) { MPI_Send(buffer.data(), (int)(n * cols), MPI_DOUBLE, 0, (int)j, MPI_COMM_WORLD); }
        }
        if (dist.rank == 0) {
            if (owner != 0) {
                MPI_Recv(buffer.data(), (int)(n * cols), MPI_DOUBLE, owner, (int)j, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t o = 0; o < cols; ++o) {
                    A[i * n + jb + o] = buffer[o * n + i];
                }
            }
        }
    }
}

// Distributed validation: every rank reconstructs its own block columns of
// L * L^T and compares them against the originally generated matrix.
bool validateCholesky(const double* dA, const double* dAorig, const Distribution& dist, GpuContext& gpu) {
    const size_t n = dist.n;

    // Replicate the full factor on every device (column major).
    double* dL = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
    std::vector<double> buffer(n * dist.nb);
    for (size_t j = 0; j < dist.numBlocks; ++j) {
        const size_t jb = dist.blockStart(j);
        const size_t cols = dist.blockCols(j);
        const int owner = dist.owner(j);
        if (dist.rank == owner) {
            CUDA_CHECK(cudaMemcpyAsync(buffer.data(), dA + dist.localOffset(j), n * cols * sizeof(double),
                                       cudaMemcpyDeviceToHost, gpu.stream));
            CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
        }
        MPI_Bcast(buffer.data(), (int)(n * cols), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpyAsync(dL + jb * n, buffer.data(), n * cols * sizeof(double), cudaMemcpyHostToDevice,
                                   gpu.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    }

    const int reduceBlocks = 1024;
    double* dPartAbs = nullptr;
    double* dPartRel = nullptr;
    double* dRec = nullptr;
    CUDA_CHECK(cudaMalloc(&dPartAbs, reduceBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPartRel, reduceBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dRec, n * dist.nb * sizeof(double)));
    std::vector<double> partAbs(reduceBlocks);
    std::vector<double> partRel(reduceBlocks);

    double maxError = 0.0;
    double relError = 0.0;
    const double one = 1.0;
    const double zero = 0.0;

    for (size_t j = (size_t)dist.rank; j < dist.numBlocks; j += (size_t)dist.size) {
        const size_t jb = dist.blockStart(j);
        const size_t cols = dist.blockCols(j);
        // R(:, jb:jb+cols) = L * L(jb:jb+cols, :)^T
        CUBLAS_CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_T, (int)n, (int)cols, (int)n, &one, dL, (int)n,
                                 dL + jb, (int)n, &zero, dRec, (int)n));
        errorReduceKernel<<<reduceBlocks, 256, 0, gpu.stream>>>(dRec, dAorig + dist.localOffset(j), n * cols,
                                                                dPartAbs, dPartRel);
        CUDA_CHECK(cudaMemcpyAsync(partAbs.data(), dPartAbs, reduceBlocks * sizeof(double), cudaMemcpyDeviceToHost,
                                   gpu.stream));
        CUDA_CHECK(cudaMemcpyAsync(partRel.data(), dPartRel, reduceBlocks * sizeof(double), cudaMemcpyDeviceToHost,
                                   gpu.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));

        double blockAbs = 0.0;
        double blockRel = 0.0;
#pragma omp parallel for schedule(static) reduction(max : blockAbs, blockRel)
        for (int i = 0; i < reduceBlocks; ++i) {
            blockAbs = std::max(blockAbs, partAbs[i]);
            blockRel = std::max(blockRel, partRel[i]);
        }
        maxError = std::max(maxError, blockAbs);
        relError = std::max(relError, blockRel);
    }

    CUDA_CHECK(cudaFree(dPartAbs));
    CUDA_CHECK(cudaFree(dPartRel));
    CUDA_CHECK(cudaFree(dRec));
    CUDA_CHECK(cudaFree(dL));

    double reduced[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, reduced, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = reduced[0];
    relError = reduced[1];

    if (dist.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (dist.rank == 0) { printf("Validation failed: relative error too large\n"); }
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
    (void)provided;

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
        MPI_Finalize();
        return 0;
    }

    // Determine the node topology: ranks of a node share their panel buffers.
    Cluster cl;
    cl.rank = rank;
    cl.size = worldSize;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &cl.node);
    MPI_Comm_rank(cl.node, &cl.nodeRank);
    MPI_Comm_size(cl.node, &cl.nodeSize);
    MPI_Comm_split(MPI_COMM_WORLD, cl.nodeRank == 0 ? 0 : MPI_UNDEFINED, rank, &cl.leaders);
    if (cl.leaders != MPI_COMM_NULL) {
        MPI_Comm_rank(cl.leaders, &cl.nodeId);
        MPI_Comm_size(cl.leaders, &cl.numNodes);
    }
    MPI_Bcast(&cl.nodeId, 1, MPI_INT, 0, cl.node);
    MPI_Bcast(&cl.numNodes, 1, MPI_INT, 0, cl.node);
    cl.nodeOfRank.resize(worldSize);
    MPI_Allgather(&cl.nodeId, 1, MPI_INT, cl.nodeOfRank.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Bind each rank to one of the GPUs of its node.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) { printf("Error: no CUDA device available\n"); }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(cl.nodeRank % deviceCount));

    Distribution dist;
    dist.n = n;
    dist.nb = std::min(kBlockSize, n);
    dist.numBlocks = (n + dist.nb - 1) / dist.nb;
    dist.rank = rank;
    dist.size = worldSize;
    dist.localBlocks = (dist.numBlocks + (size_t)worldSize - 1 - (size_t)rank) / (size_t)worldSize;

    GpuContext gpu;
    int priorityLow = 0;
    int priorityHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&gpu.stream, cudaStreamNonBlocking, priorityLow));
    CUDA_CHECK(cudaStreamCreateWithPriority(&gpu.panelStream, cudaStreamNonBlocking, priorityHigh));
    CUBLAS_CHECK(cublasCreate(&gpu.blas));
    CUBLAS_CHECK(cublasSetStream(gpu.blas, gpu.stream));
    CUBLAS_CHECK(cublasCreate(&gpu.blasPanel));
    CUBLAS_CHECK(cublasSetStream(gpu.blasPanel, gpu.panelStream));
    CUSOLVER_CHECK(cusolverDnCreate(&gpu.solver));
    CUSOLVER_CHECK(cusolverDnSetStream(gpu.solver, gpu.panelStream));
    CUDA_CHECK(cudaMalloc(&gpu.devInfo, sizeof(int)));

    // Local device storage for the owned block columns (column major, ld = n).
    const size_t localElems = std::max<size_t>(dist.localBlocks * dist.nb * n, 1);
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, localElems * sizeof(double)));

    // cuSOLVER workspace for the diagonal block factorizations.
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER, (int)dist.nb, dA, (int)n,
                                               &gpu.workSize));
    CUDA_CHECK(cudaMalloc(&gpu.work, std::max(gpu.workSize, 1) * sizeof(double)));

    // Panel buffers (triple buffered for the lookahead pipeline).  The host side
    // buffers live in a window shared by all ranks of the node, so a panel is
    // written once per node and uploaded to every GPU straight from there.
    const size_t panelElems = dist.nb * n + 1;
    double* dPanel[kPanelSlots] = {nullptr, nullptr, nullptr};
    double* hPanel[kPanelSlots] = {nullptr, nullptr, nullptr};
    MPI_Win panelWin[kPanelSlots];
    for (int i = 0; i < kPanelSlots; ++i) {
        CUDA_CHECK(cudaMalloc(&dPanel[i], dist.nb * n * sizeof(double)));
        double* base = nullptr;
        const MPI_Aint bytes = (cl.nodeRank == 0) ? (MPI_Aint)(panelElems * sizeof(double)) : 0;
        MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, cl.node, &base, &panelWin[i]);
        MPI_Aint segment = 0;
        int unit = 0;
        MPI_Win_shared_query(panelWin[i], 0, &segment, &unit, &base);
        hPanel[i] = base;
        // Page locking the shared segment keeps the device transfers at full speed.
        cudaHostRegister(hPanel[i], panelElems * sizeof(double), cudaHostRegisterDefault);
        cudaGetLastError();
    }

    // Generate positive definite matrix
    if (rank == 0) { printf("Generating positive definite matrix...\n"); }
    generatePositiveDefiniteMatrix(dA, dist, gpu);

    double* dAorig = nullptr;
    if (validate) {
        CUDA_CHECK(cudaMalloc(&dAorig, localElems * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(dAorig, dA, localElems * sizeof(double), cudaMemcpyDeviceToDevice, gpu.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    }

    // Perform Cholesky decomposition
    if (rank == 0) { printf("Computing Cholesky decomposition...\n"); }
    warmUpGpu(dist, gpu, hPanel[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    size_t failColumn = 0;
    bool success = choleskyDecomposition(dA, dist, cl, gpu, dPanel, hPanel, failColumn);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", failColumn);
            printf("Cholesky decomposition failed\n");
        }
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

    // Print results for external validation
    if (printResults) {
        std::vector<double> A;
        gatherMatrix(dA, dist, gpu, A);
        if (rank == 0) { print_results(A, "CholeskyL"); }
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        if (rank == 0) { printf("Validating result...\n"); }
        bool valid = validateCholesky(dA, dAorig, dist, gpu);

        if (valid) {
            if (rank == 0) { printf("Validation: PASSED\n"); }
        } else {
            if (rank == 0) { printf("Validation: FAILED\n"); }
            exitCode = 1;
        }
        CUDA_CHECK(cudaFree(dAorig));
    }

    for (int i = 0; i < kPanelSlots; ++i) {
        CUDA_CHECK(cudaFree(dPanel[i]));
        cudaHostUnregister(hPanel[i]);
        cudaGetLastError();
        MPI_Win_free(&panelWin[i]);
    }
    MPI_Comm_free(&cl.node);
    if (cl.leaders != MPI_COMM_NULL) { MPI_Comm_free(&cl.leaders); }
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(gpu.work));
    CUDA_CHECK(cudaFree(gpu.devInfo));
    CUSOLVER_CHECK(cusolverDnDestroy(gpu.solver));
    CUBLAS_CHECK(cublasDestroy(gpu.blas));
    CUBLAS_CHECK(cublasDestroy(gpu.blasPanel));
    CUDA_CHECK(cudaStreamDestroy(gpu.stream));
    CUDA_CHECK(cudaStreamDestroy(gpu.panelStream));

    MPI_Finalize();
    return exitCode;
}
