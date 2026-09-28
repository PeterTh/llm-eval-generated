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
// Decomposes a positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization strategy (right-looking blocked algorithm):
//   * MPI    : the matrix is distributed over the ranks with a one dimensional
//              block-column-cyclic layout (block size NB).  Every step the owner of the
//              current panel factorizes it and broadcasts it; all ranks then update the
//              trailing block columns they own.  A look-ahead produces the panel of the next
//              step on a separate stream and broadcasts it (MPI_Ibcast) while the bulk of the
//              trailing update is still running on the GPU, hiding the communication.
//   * CUDA   : the local part of the matrix permanently resides in device memory.  All
//              O(n^3) work (trailing updates, triangular solves, matrix generation and
//              validation) is executed on the GPU with cuBLAS; custom kernels handle the
//              layout/bookkeeping operations.  Ranks are mapped round-robin onto the GPUs
//              of their compute node.
//   * OpenMP : the small diagonal-block factorization (which carries the sequential
//              dependency of the algorithm and needs the positive-definiteness test) is
//              done on the CPU, as well as the host side packing, transposition and the
//              error reduction of the validation.
//
// Internally the matrix is kept in column-major order.  A is symmetric, so its row-major
// and column-major representations are identical; only the resulting lower triangular
// factor L has to be transposed back to row-major at the very end.

#define CUDA_CHECK(expr)                                                                            \
    do {                                                                                            \
        const cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,         \
                    __LINE__);                                                                      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                           \
        }                                                                                           \
    } while (0)

#define CUBLAS_CHECK(expr)                                                                          \
    do {                                                                                            \
        const cublasStatus_t st_ = (expr);                                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                         \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                           \
        }                                                                                           \
    } while (0)

namespace {

// Block size of the distribution / the blocked algorithm (set in selectBlockSize()).
int NB = 256;

int g_rank = 0;
int g_size = 1;
cublasHandle_t g_blas = nullptr;

struct LocalPanel {
    int gp;   // global panel index
    int c0;   // first global column of the panel
    int w;    // number of columns
    int lcol; // first column inside the local (column-major) storage
};

// Description of the local part of the distributed matrix.
struct Distribution {
    int n = 0;
    int npanels = 0;
    int nlocalcols = 0;
    std::vector<LocalPanel> local;

    void build(int n_) {
        n = n_;
        npanels = (n + NB - 1) / NB;
        local.clear();
        nlocalcols = 0;
        for (int p = g_rank; p < npanels; p += g_size) {
            const int c0 = p * NB;
            const int w = std::min(NB, n - c0);
            local.push_back({p, c0, w, nlocalcols});
            nlocalcols += w;
        }
    }

    static int owner(int panel) { return panel % g_size; }
};

// ---------------------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------------------

// Adds a constant to the diagonal elements of the locally stored columns.
__global__ void addToDiagonalKernel(double* A, int ld, int c0, int lcol, int w, double value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w) {
        A[(size_t)(lcol + i) * ld + (c0 + i)] += value;
    }
}

// Zeroes the strictly upper triangular part of a column-major matrix (in place).
__global__ void zeroUpperKernel(double* A, int ld, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && row < col) {
        A[(size_t)col * ld + row] = 0.0;
    }
}

// Transposes a column-major n x n matrix into row-major layout (i.e. a plain transpose).
__global__ void transposeKernel(const double* __restrict__ src, double* __restrict__ dst, int n) {
    __shared__ double tile[32][33];
    const int x = blockIdx.x * 32 + threadIdx.x; // row in src
    const int y = blockIdx.y * 32 + threadIdx.y; // col in src
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = src[(size_t)y * n + x];
    }
    __syncthreads();
    const int tx = blockIdx.y * 32 + threadIdx.x;
    const int ty = blockIdx.x * 32 + threadIdx.y;
    if (tx < n && ty < n) {
        dst[(size_t)ty * n + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

// ---------------------------------------------------------------------------------------
// CPU (OpenMP) factorization of a small diagonal block, column-major, lower triangle.
// Returns the index of the failing column or -1 on success.
// ---------------------------------------------------------------------------------------
int factorizeDiagonalBlock(double* D, const int w) {
    for (int c = 0; c < w; ++c) {
        const double val = D[(size_t)c * w + c];
        if (val <= 0.0) {
            return c;
        }
        const double d = sqrt(val);
        D[(size_t)c * w + c] = d;
        const double inv = 1.0 / d;

        const int rest = w - c - 1;
        if (rest <= 0) {
            continue;
        }

        double* col = D + (size_t)c * w;
        for (int r = c + 1; r < w; ++r) {
            col[r] *= inv;
        }

        // Rank-1 update of the trailing submatrix (lower triangle only).
        const int nthreads = std::min(omp_get_max_threads(), std::max(1, rest / 8));
#pragma omp parallel for schedule(static) num_threads(nthreads) if (nthreads > 1)
        for (int j = c + 1; j < w; ++j) {
            double* dst = D + (size_t)j * w;
            const double f = col[j];
#pragma omp simd
            for (int r = j; r < w; ++r) {
                dst[r] -= col[r] * f;
            }
        }
    }
    return -1;
}

// Per-rank device/host workspace of the factorization.
struct Workspace {
    cudaStream_t comp = nullptr;  // bulk trailing updates
    cudaStream_t look = nullptr;  // look-ahead panel (critical path)
    cudaEvent_t evComp = nullptr; // end of the bulk updates of the last step
    cudaEvent_t evLook = nullptr; // end of the look-ahead panel work
    cudaEvent_t evPanel[2] = {nullptr, nullptr}; // panel buffer b is on the device
    double* hPanel[2] = {nullptr, nullptr};      // pinned: [0] = info, [1...] = panel data
    double* dPanel[2] = {nullptr, nullptr};
    double* hDiag = nullptr; // pinned diagonal block

    void create(int n, int nb) {
        CUDA_CHECK(cudaStreamCreate(&comp));
        CUDA_CHECK(cudaStreamCreate(&look));
        CUDA_CHECK(cudaEventCreateWithFlags(&evComp, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evLook, cudaEventDisableTiming));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&evPanel[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaMallocHost(&hPanel[b], ((size_t)n * nb + 1) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dPanel[b], (size_t)n * nb * sizeof(double)));
        }
        CUDA_CHECK(cudaMallocHost(&hDiag, (size_t)nb * nb * sizeof(double)));
        // Make all events valid targets for stream waits.
        CUDA_CHECK(cudaEventRecord(evComp, comp));
        CUDA_CHECK(cudaEventRecord(evLook, look));
        CUDA_CHECK(cudaEventRecord(evPanel[0], comp));
        CUDA_CHECK(cudaEventRecord(evPanel[1], comp));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void destroy() {
        CUDA_CHECK(cudaStreamDestroy(comp));
        CUDA_CHECK(cudaStreamDestroy(look));
        CUDA_CHECK(cudaEventDestroy(evComp));
        CUDA_CHECK(cudaEventDestroy(evLook));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaEventDestroy(evPanel[b]));
            CUDA_CHECK(cudaFreeHost(hPanel[b]));
            CUDA_CHECK(cudaFree(dPanel[b]));
        }
        CUDA_CHECK(cudaFreeHost(hDiag));
    }
};

// Factorizes panel p (owner only) and packs it into hbuf[1...]; hbuf[0] receives the index of
// the first non positive definite column or -1.  All device work runs on the look-ahead
// stream, which must already be synchronized with the producers of the panel data.
void factorizeAndPackPanel(double* dA, const Distribution& dist, Workspace& ws, int p,
                           double* hbuf) {
    const int n = dist.n;
    const int start = p * NB;
    const int w = std::min(NB, n - start);
    const int m = n - start;
    const LocalPanel& lp = dist.local[p / g_size];
    double* dPan = dA + (size_t)lp.lcol * n + start; // m x w, ld = n

    // 1) Factorize the w x w diagonal block on the CPU (OpenMP).
    CUDA_CHECK(cudaMemcpy2DAsync(ws.hDiag, w * sizeof(double), dPan, n * sizeof(double),
                                 w * sizeof(double), w, cudaMemcpyDeviceToHost, ws.look));
    CUDA_CHECK(cudaStreamSynchronize(ws.look));
    const int info = factorizeDiagonalBlock(ws.hDiag, w);
    hbuf[0] = (double)info;
    if (info >= 0) {
        return;
    }

    CUDA_CHECK(cudaMemcpy2DAsync(dPan, n * sizeof(double), ws.hDiag, w * sizeof(double),
                                 w * sizeof(double), w, cudaMemcpyHostToDevice, ws.look));

    // 2) Triangular solve for the rows below the diagonal block: B := B * D^-T.
    if (m > w) {
        const double one = 1.0;
        CUBLAS_CHECK(cublasSetStream(g_blas, ws.look));
        CUBLAS_CHECK(cublasDtrsm(g_blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                 CUBLAS_DIAG_NON_UNIT, m - w, w, &one, dPan, n, dPan + w, n));
    }

    // 3) Pack the panel for the broadcast.
    CUDA_CHECK(cudaMemcpy2DAsync(hbuf + 1, m * sizeof(double), dPan, n * sizeof(double),
                                 m * sizeof(double), w, cudaMemcpyDeviceToHost, ws.look));
    CUDA_CHECK(cudaStreamSynchronize(ws.look));
}

// ---------------------------------------------------------------------------------------
// Distributed blocked Cholesky factorization (right looking, with look-ahead).
//
// dA holds the local block columns (column-major, leading dimension n).  On success the
// local columns contain the corresponding columns of L (the strictly upper triangular part
// of the diagonal blocks keeps stale values and is cleared afterwards).
// ---------------------------------------------------------------------------------------
bool choleskyDecomposition(double* dA, const Distribution& dist, Workspace& ws) {
    const int n = dist.n;
    const double neg = -1.0;
    const double one = 1.0;

    auto panelElems = [&](int p) {
        const int start = p * NB;
        return (size_t)(n - start) * std::min(NB, n - start);
    };

    // Prologue: the first panel needs no update at all.
    int cur = 0;
    {
        const int owner = Distribution::owner(0);
        if (g_rank == owner) {
            factorizeAndPackPanel(dA, dist, ws, 0, ws.hPanel[cur]);
        }
        MPI_Bcast(ws.hPanel[cur], (int)panelElems(0) + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (ws.hPanel[cur][0] >= 0.0) {
            if (g_rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       (size_t)ws.hPanel[cur][0]);
            }
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(ws.dPanel[cur], ws.hPanel[cur] + 1,
                                   panelElems(0) * sizeof(double), cudaMemcpyHostToDevice,
                                   ws.comp));
        CUDA_CHECK(cudaEventRecord(ws.evPanel[cur], ws.comp));
    }

    for (int k = 0; k < dist.npanels; ++k) {
        const int kstart = k * NB;
        const int w = std::min(NB, n - kstart);
        const int m = n - kstart;
        const int nxt = k + 1;
        const int nxtBuf = 1 - cur;
        MPI_Request req = MPI_REQUEST_NULL;

        const int nxtOwner = (nxt < dist.npanels) ? Distribution::owner(nxt) : -1;
        const bool doLookahead = (nxt < dist.npanels) && (g_rank == nxtOwner);

        // The look-ahead panel is updated on its own stream; it depends on the bulk updates of
        // the previous step (these waits have to be expressed before the events are re-armed).
        if (doLookahead) {
            CUDA_CHECK(cudaStreamWaitEvent(ws.look, ws.evComp, 0));
            CUDA_CHECK(cudaStreamWaitEvent(ws.look, ws.evPanel[cur], 0));
        }

        // Bulk trailing update of the remaining locally owned panels.  It is enqueued first so
        // that the GPU is busy while the host factorizes the look-ahead panel and while the
        // panel broadcast is in flight.
        CUBLAS_CHECK(cublasSetStream(g_blas, ws.comp));
        for (const LocalPanel& lp : dist.local) {
            if (lp.gp <= k || lp.gp == nxt) {
                continue;
            }
            const int mm = n - lp.c0;
            const double* dLeft = ws.dPanel[cur] + (lp.c0 - kstart);
            double* dC = dA + (size_t)lp.lcol * n + lp.c0;
            CUBLAS_CHECK(cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, mm, lp.w, w, &neg, dLeft, m,
                                     dLeft, m, &one, dC, n));
        }
        CUDA_CHECK(cudaEventRecord(ws.evComp, ws.comp));

        // Look-ahead: produce and broadcast the panel needed by the next step, overlapping the
        // communication with the bulk update running on the GPU.
        if (nxt < dist.npanels) {
            // The host buffer is reused; its last (asynchronous) upload must have finished.
            CUDA_CHECK(cudaEventSynchronize(ws.evPanel[nxtBuf]));
            if (doLookahead) {
                const LocalPanel& lp = dist.local[nxt / g_size];
                const int mm = n - lp.c0;
                const double* dLeft = ws.dPanel[cur] + (lp.c0 - kstart);
                double* dC = dA + (size_t)lp.lcol * n + lp.c0;
                CUBLAS_CHECK(cublasSetStream(g_blas, ws.look));
                CUBLAS_CHECK(cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, mm, lp.w, w, &neg,
                                         dLeft, m, dLeft, m, &one, dC, n));

                factorizeAndPackPanel(dA, dist, ws, nxt, ws.hPanel[nxtBuf]);
            }
            CUDA_CHECK(cudaEventRecord(ws.evLook, ws.look));
            MPI_Ibcast(ws.hPanel[nxtBuf], (int)panelElems(nxt) + 1, MPI_DOUBLE, nxtOwner,
                       MPI_COMM_WORLD, &req);
        }

        if (nxt < dist.npanels) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            if (ws.hPanel[nxtBuf][0] >= 0.0) {
                CUDA_CHECK(cudaDeviceSynchronize());
                if (g_rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                           (size_t)(nxt * NB + (size_t)ws.hPanel[nxtBuf][0]));
                }
                return false;
            }
            // The buffer is reused, so wait until its previous contents have been consumed.
            CUDA_CHECK(cudaStreamWaitEvent(ws.comp, ws.evLook, 0));
            CUDA_CHECK(cudaMemcpyAsync(ws.dPanel[nxtBuf], ws.hPanel[nxtBuf] + 1,
                                       panelElems(nxt) * sizeof(double), cudaMemcpyHostToDevice,
                                       ws.comp));
            CUDA_CHECK(cudaEventRecord(ws.evPanel[nxtBuf], ws.comp));
            cur = nxtBuf;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

// ---------------------------------------------------------------------------------------
// Distributed generation of a symmetric positive definite matrix: A = B * B^T + n * I.
// Every rank builds the identical random matrix B and computes its own block columns of A
// on the GPU.
// ---------------------------------------------------------------------------------------
void generatePositiveDefiniteMatrix(double* dA, const Distribution& dist) {
    const int n = dist.n;

    std::vector<double> B((size_t)n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // B is row-major; interpreted as column-major the same buffer is X = B^T, hence
    // A = B * B^T = X^T * X.
    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
    B.clear();
    B.shrink_to_fit();

    const double one = 1.0;
    const double zero = 0.0;
    for (const LocalPanel& lp : dist.local) {
        CUBLAS_CHECK(cublasDgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, lp.w, n, &one, dB, n,
                                 dB + (size_t)lp.c0 * n, n, &zero, dA + (size_t)lp.lcol * n, n));
        const int threads = 128;
        addToDiagonalKernel<<<(lp.w + threads - 1) / threads, threads>>>(dA, n, lp.c0, lp.lcol,
                                                                        lp.w, (double)n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

// Collects the distributed column-major matrix on rank 0.
void gatherToRoot(const double* dA, const Distribution& dist, double* hFull, double* hBuf) {
    const int n = dist.n;

    if (g_rank == 0) {
        size_t li = 0;
        for (int p = 0; p < dist.npanels; ++p) {
            const int c0 = p * NB;
            const int w = std::min(NB, n - c0);
            double* dst = hFull + (size_t)c0 * n;
            if (Distribution::owner(p) == 0) {
                const LocalPanel& lp = dist.local[li++];
                CUDA_CHECK(cudaMemcpy(dst, dA + (size_t)lp.lcol * n, (size_t)n * w * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            } else {
                MPI_Recv(dst, n * w, MPI_DOUBLE, Distribution::owner(p), p, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (const LocalPanel& lp : dist.local) {
            CUDA_CHECK(cudaMemcpy(hBuf, dA + (size_t)lp.lcol * n, (size_t)n * lp.w * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            MPI_Send(hBuf, n * lp.w, MPI_DOUBLE, 0, lp.gp, MPI_COMM_WORLD);
        }
    }
}

// Validates by computing L * L^T (on the GPU) and comparing it with the original matrix.
// L is given in column-major layout, A_orig is symmetric (layout agnostic).
bool validateCholesky(const double* Lcm, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed((size_t)n * n);

    double* dL = nullptr;
    double* dR = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dR, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dL, Lcm, (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));

    const double one = 1.0;
    const double zero = 0.0;
    CUBLAS_CHECK(cublasSetStream(g_blas, nullptr));
    CUBLAS_CHECK(cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, (int)n, (int)n, (int)n, &one, dL,
                             (int)n, dL, (int)n, &zero, dR, (int)n));
    CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, (size_t)n * n * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dR));

    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
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

// Selects the GPU for this rank (round-robin over the devices of the compute node).
void selectDevice() {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
    int localRank = 0;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        if (g_rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    CUDA_CHECK(cudaFree(nullptr)); // establish the context
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

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
            if (g_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (n == 0) {
        if (g_rank == 0) {
            printf("Generating positive definite matrix...\n");
            printf("Computing Cholesky decomposition...\n");
            printf("Computation time: 0 ms\n");
            std::vector<double> empty;
            if (printResults) {
                print_results(empty, "CholeskyL");
            }
            if (validate) {
                printf("Validating result...\n");
                printf("Validation: PASSED\n");
            }
        }
        MPI_Finalize();
        return 0;
    }

    selectDevice();
    CUBLAS_CHECK(cublasCreate(&g_blas));

    // Choose the block size so that every rank owns a healthy number of panels: small blocks
    // shorten the critical path (panel factorization and broadcast), large blocks make the
    // trailing GEMMs more efficient.  About 28 panels per rank is a good compromise.
    NB = (int)((n + 28 * g_size - 1) / (28 * g_size));
    NB = std::min(256, std::max(64, (NB + 31) / 32 * 32));
    NB = std::min<int>(NB, (int)n);

    Distribution dist;
    dist.build((int)n);

    // Local block columns of the matrix (column-major, leading dimension n).
    double* dA = nullptr;
    if (dist.nlocalcols > 0) {
        CUDA_CHECK(cudaMalloc(&dA, (size_t)n * dist.nlocalcols * sizeof(double)));
    }

    // Streams, events and (pinned) communication buffers.
    const int nbEff = std::min<int>(NB, (int)n);
    Workspace ws;
    ws.create((int)n, nbEff);
    double* hStage = ws.hPanel[0] + 1; // host staging area for the gather

    if (g_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(dA, dist);

    std::vector<double> A_orig;
    if (validate) {
        // Save the original matrix (symmetric: row- and column-major coincide).
        if (g_rank == 0) {
            A_orig.resize((size_t)n * n);
        }
        gatherToRoot(dA, dist, g_rank == 0 ? A_orig.data() : nullptr, hStage);
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(dA, dist, ws);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Clear the upper triangular part of L (everything above the diagonal blocks plus the
    // stale values inside them) and collect the factor on rank 0.
    for (const LocalPanel& lp : dist.local) {
        if (lp.c0 > 0) {
            CUDA_CHECK(cudaMemset2D(dA + (size_t)lp.lcol * n, n * sizeof(double), 0,
                                    lp.c0 * sizeof(double), lp.w));
        }
        dim3 block(16, 16);
        dim3 grid((lp.w + 15) / 16, (lp.w + 15) / 16);
        zeroUpperKernel<<<grid, block>>>(dA + (size_t)lp.lcol * n + lp.c0, (int)n, lp.w);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> Lcm;
    if (g_rank == 0) {
        Lcm.resize((size_t)n * n);
    }
    gatherToRoot(dA, dist, g_rank == 0 ? Lcm.data() : nullptr, hStage);

    int status = 0;
    if (g_rank == 0) {
        // Transpose the column-major factor into the row-major result layout on the GPU.
        std::vector<double> A((size_t)n * n);
        {
            double* dSrc = nullptr;
            double* dDst = nullptr;
            CUDA_CHECK(cudaMalloc(&dSrc, (size_t)n * n * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dDst, (size_t)n * n * sizeof(double)));
            CUDA_CHECK(
                cudaMemcpy(dSrc, Lcm.data(), (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
            dim3 block(32, 32);
            dim3 grid(((int)n + 31) / 32, ((int)n + 31) / 32);
            transposeKernel<<<grid, block>>>(dSrc, dDst, (int)n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(A.data(), dDst, (size_t)n * n * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(dSrc));
            CUDA_CHECK(cudaFree(dDst));
        }

        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(Lcm.data(), A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    ws.destroy();
    if (dA != nullptr) {
        CUDA_CHECK(cudaFree(dA));
    }
    CUBLAS_CHECK(cublasDestroy(g_blas));

    MPI_Finalize();
    return status;
}
