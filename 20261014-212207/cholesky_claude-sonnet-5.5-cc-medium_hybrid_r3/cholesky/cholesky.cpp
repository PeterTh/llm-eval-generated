#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization:
//  * MPI:    1D block-cyclic distribution of block columns (one rank per GPU). A right-looking
//            blocked algorithm with one-step look-ahead: the owner of the next panel updates and
//            factors it first and broadcasts it while every rank continues the trailing update.
//  * CUDA:   matrix generation, diagonal block factorization, triangular solve of the panel,
//            trailing-matrix update (tiled DGEMM-like kernel) and validation all run on the GPU.
//  * OpenMP: host-side work (generation of B, assembling the final row-major matrix).

#define NB 64     // block (panel) width, also the GEMM tile size
#define KT 16     // k-chunk of the tiled kernels

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),         \
                    __FILE__, __LINE__);                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

// Computes tiles of X * X^T (row-major X, n x n, zero padded to N).
//  VAL == false: generation of the positive definite matrix A = B * B^T + n*I for the local
//                block columns (column-major local storage, leading dimension N, padded with I).
//  VAL == true : validation, compares L * L^T (X = L) with B * B^T + n*I (Y = B).
template <bool VAL>
__global__ void __launch_bounds__(256)
xxtKernel(const double* __restrict__ X, const double* __restrict__ Y, int n, int N,
          double* __restrict__ C, int rank, int P, unsigned long long* __restrict__ errOut) {
    __shared__ double sA[KT * (NB + 1)];
    __shared__ double sB[KT * (NB + 1)];
    __shared__ double tA[VAL ? KT * (NB + 1) : 1];
    __shared__ double tB[VAL ? KT * (NB + 1) : 1];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    int i0, j0, cloc = 0;
    if (VAL) {
        i0 = blockIdx.x * NB;
        j0 = blockIdx.y * NB;
    } else {
        cloc = blockIdx.y;
        j0 = (cloc * P + rank) * NB;
        i0 = blockIdx.x * NB;
        if (i0 < j0) return;  // strictly upper tile, never used
    }

    double acc[4][4] = {};
    double acc2[VAL ? 4 : 1][VAL ? 4 : 1] = {};

    for (int k0 = 0; k0 < n; k0 += KT) {
        for (int r = 0; r < 4; ++r) {
            const int idx = tid + 256 * r;
            const int kk = idx & (KT - 1), ii = idx / KT;
            const int k = k0 + kk;
            const int ri = i0 + ii, rj = j0 + ii;
            const bool kok = k < n;
            sA[kk * (NB + 1) + ii] = (kok && ri < n) ? X[(size_t)ri * n + k] : 0.0;
            sB[kk * (NB + 1) + ii] = (kok && rj < n) ? X[(size_t)rj * n + k] : 0.0;
            if (VAL) {
                tA[kk * (NB + 1) + ii] = (kok && ri < n) ? Y[(size_t)ri * n + k] : 0.0;
                tB[kk * (NB + 1) + ii] = (kok && rj < n) ? Y[(size_t)rj * n + k] : 0.0;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                a[q] = sA[kk * (NB + 1) + tx + 16 * q];
                b[q] = sB[kk * (NB + 1) + ty + 16 * q];
            }
#pragma unroll
            for (int p = 0; p < 4; ++p)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[p][q] += a[p] * b[q];
            if (VAL) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    a[q] = tA[kk * (NB + 1) + tx + 16 * q];
                    b[q] = tB[kk * (NB + 1) + ty + 16 * q];
                }
#pragma unroll
                for (int p = 0; p < 4; ++p)
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc2[VAL ? p : 0][VAL ? q : 0] += a[p] * b[q];
            }
        }
        __syncthreads();
    }

    double maxAbs = 0.0, maxRel = 0.0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int i = i0 + tx + 16 * p;
            const int j = j0 + ty + 16 * q;
            if (VAL) {
                if (i < n && j < n) {
                    double ref = acc2[VAL ? p : 0][VAL ? q : 0];
                    if (i == j) ref += (double)n;
                    const double err = fabs(acc[p][q] - ref);
                    maxAbs = fmax(maxAbs, err);
                    maxRel = fmax(maxRel, err / (fabs(ref) + 1e-10));
                }
            } else {
                double v = acc[p][q];
                if (i == j) v += (i < n) ? (double)n : 1.0;
                C[(size_t)N * (cloc * NB + ty + 16 * q) + i] = v;
            }
        }
    }
    if (VAL) {
        // non-negative doubles order like their bit patterns
        atomicMax(&errOut[0], (unsigned long long)__double_as_longlong(maxAbs));
        atomicMax(&errOut[1], (unsigned long long)__double_as_longlong(maxRel));
    }
}

// Unblocked Cholesky of one NB x NB diagonal block (single thread block, shared memory).
__global__ void __launch_bounds__(256)
diagKernel(double* __restrict__ A, int N, int gj0, double* __restrict__ stat) {
    __shared__ double s[NB * (NB + 1)];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < NB * NB; idx += 256) {
        const int i = idx % NB, j = idx / NB;
        s[i * (NB + 1) + j] = A[(size_t)N * j + i];
    }
    __syncthreads();
    for (int j = 0; j < NB; ++j) {
        if (tid == 0) {
            double v = s[j * (NB + 1) + j];
            if (v <= 0.0) {
                if (*stat < 0.0) *stat = (double)(gj0 + j);
                v = 1.0;
            }
            s[j * (NB + 1) + j] = sqrt(v);
        }
        __syncthreads();
        if (tid < NB && tid > j) s[tid * (NB + 1) + j] /= s[j * (NB + 1) + j];
        __syncthreads();
        for (int idx = tid; idx < NB * NB; idx += 256) {
            const int i = idx % NB, k = idx / NB;
            if (k > j && i >= k) s[i * (NB + 1) + k] -= s[i * (NB + 1) + j] * s[k * (NB + 1) + j];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < NB * NB; idx += 256) {
        const int i = idx % NB, j = idx / NB;
        A[(size_t)N * j + i] = (i >= j) ? s[i * (NB + 1) + j] : 0.0;
    }
}

// Panel solve: X * L11^T = A21, one thread per row of the panel.
__global__ void __launch_bounds__(128)
trsmKernel(const double* __restrict__ L11, double* __restrict__ P, int N, int m) {
    __shared__ double s[NB * (NB + 1)];
    for (int idx = threadIdx.x; idx < NB * NB; idx += blockDim.x) {
        const int i = idx % NB, j = idx / NB;
        s[i * (NB + 1) + j] = L11[(size_t)N * j + i];
    }
    __syncthreads();
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    double x[NB];
#pragma unroll
    for (int j = 0; j < NB; ++j) x[j] = P[(size_t)N * j + row];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        double t = x[j];
#pragma unroll
        for (int k = 0; k < j; ++k) t -= x[k] * s[j * (NB + 1) + k];
        x[j] = t / s[j * (NB + 1) + j];
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) P[(size_t)N * j + row] = x[j];
}

// Trailing update of local block columns c0 .. c0+gridDim.y-1:
//   A[i, j] -= sum_k Lp[i - r0, k] * Lp[j - r0, k]   for i >= j (tile granularity)
__global__ void __launch_bounds__(256)
updateKernel(const double* __restrict__ Lp, int ldp, int r0, double* __restrict__ A, int N,
             int c0, int rank, int P) {
    __shared__ double sA[KT * (NB + 1)];
    __shared__ double sB[KT * (NB + 1)];
    const int c = c0 + blockIdx.y;
    const int j0 = (c * P + rank) * NB;
    const int i0 = j0 + blockIdx.x * NB;
    if (i0 >= N) return;
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;

    double acc[4][4] = {};
    for (int k0 = 0; k0 < NB; k0 += KT) {
        for (int r = 0; r < 4; ++r) {
            const int idx = tid + 256 * r;
            const int ii = idx & (NB - 1), kk = idx / NB;
            sA[kk * (NB + 1) + ii] = Lp[(size_t)ldp * (k0 + kk) + (i0 - r0 + ii)];
            sB[kk * (NB + 1) + ii] = Lp[(size_t)ldp * (k0 + kk) + (j0 - r0 + ii)];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                a[q] = sA[kk * (NB + 1) + tx + 16 * q];
                b[q] = sB[kk * (NB + 1) + ty + 16 * q];
            }
#pragma unroll
            for (int p = 0; p < 4; ++p)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[p][q] += a[p] * b[q];
        }
        __syncthreads();
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int p = 0; p < 4; ++p)
            A[(size_t)N * (c * NB + ty + 16 * q) + i0 + tx + 16 * p] -= acc[p][q];
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

// Generate B (row-major, n x n) exactly as the sequential reference does, convert with OpenMP.
static void generateB(std::vector<double>& B, const size_t n) {
    std::vector<int> raw(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) raw[i] = rand_r(&seed);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)(n * n); ++i) {
        B[i] = (raw[i] / (double)RAND_MAX) - 0.5;
    }
}

struct Ctx {
    int rank = 0, P = 1;
    int n = 0, N = 0, nbk = 0, nlb = 0;
    double* dA = nullptr;
    double* dLp[2] = {nullptr, nullptr};
    double* dStat = nullptr;
    double* hbuf[2] = {nullptr, nullptr};
    cudaStream_t sMain, sCrit, sCopy;
    std::vector<cudaEvent_t> bulkDone, ready;
};

static int firstLocalBlock(const Ctx& cx, int g) {  // first local block index with global >= g
    if (g <= cx.rank) return 0;
    return (g - cx.rank + cx.P - 1) / cx.P;
}

// Owner side: factor diagonal block K, solve the panel below it and stage it to the host buffer.
static void factorPanel(Ctx& cx, int K, cudaStream_t s, double* hb) {
    const int c = K / cx.P;
    const int N = cx.N;
    double* diag = cx.dA + (size_t)N * c * NB + (size_t)K * NB;
    diagKernel<<<1, 256, 0, s>>>(diag, N, K * NB, cx.dStat);
    const int m = N - (K + 1) * NB;
    if (m > 0) {
        double* pan = diag + NB;
        trsmKernel<<<(m + 127) / 128, 128, 0, s>>>(diag, pan, N, m);
    }
    CUDA_CHECK(cudaMemcpyAsync(hb, cx.dStat, sizeof(double), cudaMemcpyDeviceToHost, s));
    if (m > 0) {
        CUDA_CHECK(cudaMemcpy2DAsync(hb + 1, (size_t)m * sizeof(double), diag + NB,
                                     (size_t)N * sizeof(double), (size_t)m * sizeof(double), NB,
                                     cudaMemcpyDeviceToHost, s));
    }
}

static void launchUpdate(const Ctx& cx, int K, int c0, int count, cudaStream_t s) {
    if (count <= 0) return;
    const int N = cx.N;
    const int r0 = (K + 1) * NB;
    const double* Lp;
    int ldp;
    if (K % cx.P == cx.rank) {
        Lp = cx.dA + (size_t)N * (K / cx.P) * NB + r0;
        ldp = N;
    } else {
        Lp = cx.dLp[K & 1];
        ldp = N - r0;
    }
    const int gFirst = c0 * cx.P + cx.rank;
    dim3 grid(cx.nbk - gFirst, count);
    updateKernel<<<grid, 256, 0, s>>>(Lp, ldp, r0, cx.dA, N, c0, cx.rank, cx.P);
}

// Distributed factorization. Returns the failing diagonal index or -1 on success.
static long long distributedCholesky(Ctx& cx) {
    const int nbk = cx.nbk, P = cx.P, rank = cx.rank, N = cx.N;
    if (nbk == 0) return -1;

    auto bcastCount = [&](int K) { return 1 + (size_t)(N - (K + 1) * NB) * NB; };

    // Step 0 panel
    if (rank == 0) {
        factorPanel(cx, 0, cx.sCrit, cx.hbuf[0]);
        CUDA_CHECK(cudaEventRecord(cx.ready[0], cx.sCrit));
        CUDA_CHECK(cudaStreamSynchronize(cx.sCrit));
    }
    MPI_Bcast(cx.hbuf[0], (int)bcastCount(0), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0 && N - NB > 0) {
        CUDA_CHECK(cudaMemcpyAsync(cx.dLp[0], cx.hbuf[0] + 1,
                                   (size_t)(N - NB) * NB * sizeof(double),
                                   cudaMemcpyHostToDevice, cx.sCopy));
    }
    if (rank != 0) CUDA_CHECK(cudaEventRecord(cx.ready[0], cx.sCopy));

    for (int K = 0; K < nbk; ++K) {
        if (cx.hbuf[K & 1][0] >= 0.0) return (long long)cx.hbuf[K & 1][0];
        if (K == nbk - 1) break;

        const int o1 = (K + 1) % P;
        // host buffer of parity (K+1)&1 was last used for panel K-1 (possibly by a pending H2D)
        if (K >= 1) CUDA_CHECK(cudaEventSynchronize(cx.ready[K - 1]));

        if (rank == o1) {
            CUDA_CHECK(cudaStreamWaitEvent(cx.sCrit, cx.ready[K], 0));
            if (K >= 1) CUDA_CHECK(cudaStreamWaitEvent(cx.sCrit, cx.bulkDone[K - 1], 0));
            launchUpdate(cx, K, (K + 1) / P, 1, cx.sCrit);
            factorPanel(cx, K + 1, cx.sCrit, cx.hbuf[(K + 1) & 1]);
            CUDA_CHECK(cudaEventRecord(cx.ready[K + 1], cx.sCrit));
        }

        // bulk update of all remaining local block columns (global index >= K+2)
        CUDA_CHECK(cudaStreamWaitEvent(cx.sMain, cx.ready[K], 0));
        const int c0 = firstLocalBlock(cx, K + 2);
        launchUpdate(cx, K, c0, cx.nlb - c0, cx.sMain);
        CUDA_CHECK(cudaEventRecord(cx.bulkDone[K], cx.sMain));

        if (rank == o1) CUDA_CHECK(cudaStreamSynchronize(cx.sCrit));

        MPI_Bcast(cx.hbuf[(K + 1) & 1], (int)bcastCount(K + 1), MPI_DOUBLE, o1, MPI_COMM_WORLD);

        if (rank != o1) {
            const int m = N - (K + 2) * NB;
            CUDA_CHECK(cudaStreamWaitEvent(cx.sCopy, cx.bulkDone[K > 0 ? K - 1 : 0], 0));
            if (m > 0) {
                CUDA_CHECK(cudaMemcpyAsync(cx.dLp[(K + 1) & 1], cx.hbuf[(K + 1) & 1] + 1,
                                           (size_t)m * NB * sizeof(double),
                                           cudaMemcpyHostToDevice, cx.sCopy));
            }
            CUDA_CHECK(cudaEventRecord(cx.ready[K + 1], cx.sCopy));
        }
    }
    return -1;
}

// Gather the factor on rank 0 into a row-major n x n matrix (upper triangle zero).
static void gatherResult(Ctx& cx, std::vector<double>& A) {
    const int N = cx.N, n = cx.n, P = cx.P, rank = cx.rank;
    const size_t blk = (size_t)N * NB;
    if (rank != 0) {
        std::vector<double> hl(blk * std::max(cx.nlb, 1));
        if (cx.nlb > 0) CUDA_CHECK(cudaMemcpy(hl.data(), cx.dA, blk * cx.nlb * sizeof(double),
                                              cudaMemcpyDeviceToHost));
        for (int c = 0; c < cx.nlb; ++c)
            MPI_Send(hl.data() + blk * c, (int)blk, MPI_DOUBLE, 0, c, MPI_COMM_WORLD);
        return;
    }
    A.assign((size_t)n * n, 0.0);
    std::vector<double> buf(blk);
    auto scatter = [&](int G, const double* p) {
        const int j0 = G * NB;
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = j0; i < n; ++i) {
            const int jmax = std::min(std::min(i, j0 + NB - 1), n - 1);
            for (int j = j0; j <= jmax; ++j) A[(size_t)i * n + j] = p[(size_t)N * (j - j0) + i];
        }
    };
    for (int q = 0; q < P; ++q) {
        const int cnt = (cx.nbk > q) ? (cx.nbk - q + P - 1) / P : 0;
        if (q == 0) {
            std::vector<double> hl(blk * std::max(cnt, 1));
            if (cnt > 0) CUDA_CHECK(cudaMemcpy(hl.data(), cx.dA, blk * cnt * sizeof(double),
                                               cudaMemcpyDeviceToHost));
            for (int c = 0; c < cnt; ++c) scatter(c * P, hl.data() + blk * c);
        } else {
            for (int c = 0; c < cnt; ++c) {
                MPI_Recv(buf.data(), (int)blk, MPI_DOUBLE, q, c, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                scatter(c * P + q, buf.data());
            }
        }
    }
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
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

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

    // Bind one GPU per rank (per node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    Ctx cx;
    cx.rank = rank;
    cx.P = P;
    cx.n = (int)n;
    cx.nbk = (int)((n + NB - 1) / NB);
    cx.N = cx.nbk * NB;
    cx.nlb = (cx.nbk > rank) ? (cx.nbk - rank + P - 1) / P : 0;
    const int N = cx.N;

    CUDA_CHECK(cudaStreamCreateWithPriority(&cx.sCrit, cudaStreamNonBlocking, -1));
    CUDA_CHECK(cudaStreamCreateWithFlags(&cx.sMain, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&cx.sCopy, cudaStreamNonBlocking));
    cx.bulkDone.resize(cx.nbk + 1);
    cx.ready.resize(cx.nbk + 1);
    for (auto& e : cx.bulkDone) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    for (auto& e : cx.ready) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

    const size_t localElems = (size_t)N * NB * std::max(cx.nlb, 1);
    CUDA_CHECK(cudaMalloc(&cx.dA, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(cx.dA, 0, localElems * sizeof(double)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&cx.dLp[b], (size_t)std::max(N, NB) * NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&cx.hbuf[b], ((size_t)N * NB + 1) * sizeof(double)));
        cx.hbuf[b][0] = -1.0;
    }
    CUDA_CHECK(cudaMalloc(&cx.dStat, sizeof(double)));
    {
        const double minus1 = -1.0;
        CUDA_CHECK(cudaMemcpy(cx.dStat, &minus1, sizeof(double), cudaMemcpyHostToDevice));
    }

    // Generate the matrix: B on the host (identical on all ranks), then A = B*B^T + n*I on GPU
    double* dB = nullptr;
    {
        std::vector<double> B(n * n);
        generateB(B, n);
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        }
    }
    if (n > 0 && cx.nlb > 0) {
        dim3 grid(cx.nbk, cx.nlb);
        xxtKernel<false><<<grid, 256>>>(dB, dB, cx.n, N, cx.dA, rank, P, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!validate && dB) {
        CUDA_CHECK(cudaFree(dB));
        dB = nullptr;
    }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const long long failIdx = distributedCholesky(cx);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;
    if (failIdx >= 0) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %lld\n", failIdx);
            printf("Cholesky decomposition failed\n");
        }
        exitCode = 1;
    } else {
        if (rank == 0) {
            printf("Computation time: %ld ms\n", (long)duration.count());

            // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
            double ops = (double)n * n * n / 3.0;
            double gflops = ops / (duration.count() / 1000.0) / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        }

        std::vector<double> A;
        if (printResults || validate) gatherResult(cx, A);

        // Print results for external validation
        if (printResults && rank == 0) {
            print_results(A, "CholeskyL");
        }

        // Validation (on rank 0's GPU)
        if (validate) {
            int valid = 1;
            if (rank == 0) {
                printf("Validating result...\n");
                double* dL = nullptr;
                unsigned long long* dErr = nullptr;
                unsigned long long hErr[2] = {0, 0};
                if (n > 0) {
                    CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
                    CUDA_CHECK(cudaMemcpy(dL, A.data(), n * n * sizeof(double),
                                          cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMalloc(&dErr, 2 * sizeof(unsigned long long)));
                    CUDA_CHECK(cudaMemset(dErr, 0, 2 * sizeof(unsigned long long)));
                    dim3 grid(cx.nbk, cx.nbk);
                    xxtKernel<true><<<grid, 256>>>(dL, dB, cx.n, N, nullptr, 0, 1, dErr);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpy(hErr, dErr, sizeof(hErr), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaFree(dL));
                    CUDA_CHECK(cudaFree(dErr));
                }
                double maxError, relError;
                memcpy(&maxError, &hErr[0], sizeof(double));
                memcpy(&relError, &hErr[1], sizeof(double));
                printf("Max absolute error: %.10e\n", maxError);
                printf("Max relative error: %.10e\n", relError);
                if (relError > 1e-6) {
                    printf("Validation failed: relative error too large\n");
                    valid = 0;
                }
                printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            exitCode = valid ? 0 : 1;
        }
    }

    if (dB) CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(cx.dA));
    CUDA_CHECK(cudaFree(cx.dStat));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(cx.dLp[b]));
        CUDA_CHECK(cudaFreeHost(cx.hbuf[b]));
    }
    MPI_Finalize();
    return exitCode;
}
