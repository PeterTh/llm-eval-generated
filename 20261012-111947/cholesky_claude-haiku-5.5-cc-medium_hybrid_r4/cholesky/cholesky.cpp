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

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Data distribution: the matrix is cut into kBlock-row blocks that are dealt out
// cyclically over MPI ranks (block R lives on rank R % nprocs). Each rank drives one
// GPU and keeps its rows resident in device memory.
//
// Algorithm: blocked left-looking Cholesky. For column block t, the owning rank
// factors the diagonal block on the host (OpenMP over rows) and broadcasts the
// finished L panel. Every rank then updates its trailing rows on its GPU: a tiled
// GEMM computes the partial sums over k < j0, and a per-row kernel finishes the
// block columns in order. The next diagonal block is updated first, so the owner's
// diagonal work overlaps the bulk of the trailing update (two CUDA streams).
//
// Semantics: every element of L is accumulated over k in the same increasing order
// as the sequential reference (no reassociation). The generated matrix is bit-identical
// to the reference. Factor terms use an explicit fused multiply-add on both host and
// device; the reference's compiler contracts these products only partly, so L agrees
// with it to rounding (about 1e-16 absolute) rather than bit for bit.

constexpr size_t kBlock = 256;  // row block size and panel width
constexpr int kTile = 64;       // GEMM output columns per CUDA block
constexpr int kRowTile = 16;    // GEMM output rows per CUDA block (sets occupancy)
constexpr int kKTile = 16;      // GEMM depth staged in shared memory per step

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// P[i][j] = sum_{k<K} X[i][k] * Y[j][k], accumulated in increasing k per element.
// kFused selects a fused multiply-add per term. Generation uses separate multiply and
// add, which matches the reference generator bit for bit.
template <bool kFused>
__global__ void gemmNtKernel(const double* __restrict__ X, int ldx,
                             const double* __restrict__ Y, int ldy,
                             double* __restrict__ P, int ldp,
                             int M, int N, int K) {
    __shared__ double As[kKTile][kRowTile + 1];
    __shared__ double Bs[kKTile][kTile + 1];
    const int tx = threadIdx.x % 16;  // column group: 4 columns each
    const int ty = threadIdx.x / 16;  // row group: 4 rows each
    const int i0 = blockIdx.y * kRowTile;
    const int j0 = blockIdx.x * kTile;
    double acc[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += kKTile) {
        for (int e = threadIdx.x; e < kRowTile * kKTile; e += blockDim.x) {
            const int r = e / kKTile;
            const int c = e % kKTile;
            const int gk = k0 + c;
            const int gi = i0 + r;
            As[c][r] = (gi < M && gk < K) ? X[(size_t)gi * ldx + gk] : 0.0;
        }
        for (int e = threadIdx.x; e < kTile * kKTile; e += blockDim.x) {
            const int r = e / kKTile;
            const int c = e % kKTile;
            const int gk = k0 + c;
            const int gj = j0 + r;
            Bs[c][r] = (gj < N && gk < K) ? Y[(size_t)gj * ldy + gk] : 0.0;
        }
        __syncthreads();

        const int kmax = min(kKTile, K - k0);
        for (int kk = 0; kk < kmax; ++kk) {
            double a[4], b[4];
            for (int x = 0; x < 4; ++x) {
                a[x] = As[kk][ty * 4 + x];
                b[x] = Bs[kk][tx * 4 + x];
            }
            for (int x = 0; x < 4; ++x)
                for (int y = 0; y < 4; ++y)
                    acc[x][y] = kFused ? fma(a[x], b[y], acc[x][y])
                                       : __dadd_rn(acc[x][y], __dmul_rn(a[x], b[y]));
        }
        __syncthreads();
    }

    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y) {
            const int gi = i0 + ty * 4 + x;
            const int gj = j0 + tx * 4 + y;
            if (gi < M && gj < N) P[(size_t)gi * ldp + gj] = acc[x][y];
        }
}

// Finishes block columns [j0, j0+w) for M rows. P holds the partial sums over k < j0
// (ld kBlock); panel rows hold L[j][0..j0+w) with ld ldpanel. Each row is independent
// and its block columns are resolved in order, since later columns use earlier ones.
__global__ void finishRowsKernel(double* __restrict__ L, int ldl,
                                 const double* __restrict__ P,
                                 const double* __restrict__ panel, int ldpanel,
                                 int j0, int w, int M) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    double* row = L + (size_t)i * ldl + j0;
    double vals[kBlock];
    for (int j = 0; j < w; ++j) {
        const double* pj = panel + (size_t)j * ldpanel + j0;
        double sum = P[i * kBlock + j];
        for (int k = 0; k < j; ++k) sum = fma(vals[k], pj[k], sum);
        vals[j] = (row[j] - sum) / pj[j];
        row[j] = vals[j];
    }
}

// Adds n to the diagonal of `rows` rows whose global indices start at g0.
__global__ void addDiagKernel(double* __restrict__ L, int n, int g0, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) L[(size_t)r * n + g0 + r] += (double)n;
}

template <bool kFused>
void gemmNt(const double* X, int ldx, const double* Y, int ldy, double* P, int ldp,
            int M, int N, int K, cudaStream_t stream = 0) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + kTile - 1) / kTile, (M + kRowTile - 1) / kRowTile);
    gemmNtKernel<kFused><<<grid, kRowTile * 4, 0, stream>>>(X, ldx, Y, ldy, P, ldp, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// Number of matrix rows owned by `rank` in the cyclic block distribution.
size_t validRowsOf(int rank, int nprocs, size_t n) {
    size_t rows = 0;
    for (size_t R = rank; R * kBlock < n; R += nprocs)
        rows += std::min(kBlock, n - R * kBlock);
    return rows;
}

// Global row index of local row `lr` on `rank`.
size_t globalRowOf(int rank, int nprocs, size_t lr) {
    const size_t R = rank + (lr / kBlock) * nprocs;
    return R * kBlock + lr % kBlock;
}

// Factors the w x w diagonal block D (ld kBlock) in place. On entry the lower triangle
// holds the original values and P holds the partial sums over k < j0. On exit the
// lower triangle holds L and the upper triangle is zero. Returns -1 on success, or the
// global index of a non-positive pivot.
long long factorDiagonalBlock(double* D, const double* P, size_t w, size_t j0) {
    for (size_t j = 0; j < w; ++j) {
        double sum = P[j * kBlock + j];
        for (size_t k = 0; k < j; ++k) sum = std::fma(D[j * kBlock + k], D[j * kBlock + k], sum);
        const double val = D[j * kBlock + j] - sum;
        if (val <= 0.0) return (long long)(j0 + j);
        const double diag = std::sqrt(val);
        D[j * kBlock + j] = diag;

#pragma omp parallel for if (w - j > 32)
        for (size_t i = j + 1; i < w; ++i) {
            double s = P[i * kBlock + j];
            for (size_t k = 0; k < j; ++k) s = std::fma(D[i * kBlock + k], D[j * kBlock + k], s);
            D[i * kBlock + j] = (D[i * kBlock + j] - s) / diag;
        }
    }
    for (size_t i = 0; i < w; ++i)
        for (size_t j = i + 1; j < w; ++j) D[i * kBlock + j] = 0.0;
    return -1;
}

// Distributed in-place decomposition. dL holds this rank's validRows rows (row-major,
// ld n). Returns false if the matrix is not positive definite.
//
// Pipeline per column block t (panel width w, panel columns [0, j0+w)):
//  - The owner of block t factors it on sMain. Its rows were already brought up to date
//    by the lead update of step t-1, so it does not wait for its bulk trailing update.
//  - The finished panel is broadcast; every rank copies it to the device and updates its
//    trailing rows. The rows of the next diagonal block (lead) run on sMain ahead of the
//    rest (bulk, on sBulk), so the owner's diagonal work overlaps the bulk update.
// Panel buffers are double-buffered on the host and the device; events guard reuse.
bool choleskyDistributed(double* dL, size_t n, size_t validRows, int rank, int nprocs,
                         MPI_Comm comm) {
    const size_t nblocks = (n + kBlock - 1) / kBlock;
    const size_t rowsMax = std::max<size_t>(validRows, 1);

    cudaStream_t sMain = nullptr, sBulk = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sMain, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sBulk, cudaStreamNonBlocking));
    cudaEvent_t evBulk[2], evH2D[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evBulk[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evH2D[b], cudaEventDisableTiming));
    }

    double *dPd = nullptr, *dPLead = nullptr, *dPBulk = nullptr;
    double* dPanel[2];
    double* hPanel[2];
    CUDA_CHECK(cudaMalloc(&dPd, kBlock * kBlock * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPLead, kBlock * kBlock * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPBulk, rowsMax * kBlock * sizeof(double)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dPanel[b], kBlock * n * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hPanel[b], kBlock * n * sizeof(double)));
    }

    std::vector<double> diagP(kBlock * kBlock), diagL(kBlock * kBlock);
    bool ok = true;

    for (size_t t = 0; t < nblocks; ++t) {
        const size_t j0 = t * kBlock;
        const size_t w = std::min(kBlock, n - j0);
        const size_t Kp = j0 + w;  // panel width: columns [0, Kp)
        const int owner = (int)(t % nprocs);
        const size_t ownerRow = (t / nprocs) * kBlock;  // local row of block t on owner
        const int b = (int)(t % 2);

        // Host staging buffer b was last read by the H2D copy of step t-2.
        if (t >= 2) CUDA_CHECK(cudaEventSynchronize(evH2D[b]));

        long long failIndex = -1;
        if (rank == owner) {
            // Partial sums over k < j0 for the diagonal block, then host factorization.
            gemmNt<true>(dL + ownerRow * n, (int)n, dL + ownerRow * n, (int)n, dPd, kBlock,
                         (int)w, (int)w, (int)j0, sMain);
            CUDA_CHECK(cudaMemcpy2DAsync(diagP.data(), kBlock * sizeof(double), dPd,
                                         kBlock * sizeof(double), w * sizeof(double), w,
                                         cudaMemcpyDeviceToHost, sMain));
            CUDA_CHECK(cudaMemcpy2DAsync(diagL.data(), kBlock * sizeof(double),
                                         dL + ownerRow * n + j0, n * sizeof(double),
                                         w * sizeof(double), w, cudaMemcpyDeviceToHost, sMain));
            CUDA_CHECK(cudaStreamSynchronize(sMain));
            failIndex = factorDiagonalBlock(diagL.data(), diagP.data(), w, j0);
        }

        MPI_Bcast(&failIndex, 1, MPI_LONG_LONG, owner, comm);
        if (failIndex >= 0) {
            if (rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %lld\n",
                       failIndex);
            ok = false;
            break;
        }

        if (rank == owner) {
            // Store the finished diagonal block; zero everything right of the diagonal.
            CUDA_CHECK(cudaMemcpy2DAsync(dL + ownerRow * n + j0, n * sizeof(double),
                                         diagL.data(), kBlock * sizeof(double),
                                         w * sizeof(double), w, cudaMemcpyHostToDevice, sMain));
            if (Kp < n)
                CUDA_CHECK(cudaMemset2DAsync(dL + ownerRow * n + Kp, n * sizeof(double), 0,
                                             (n - Kp) * sizeof(double), w, sMain));
            // Panel rows [j0, j0+w) over columns [0, Kp): prefix from the device, block from host.
            if (j0 > 0)
                CUDA_CHECK(cudaMemcpy2DAsync(hPanel[b], Kp * sizeof(double),
                                             dL + ownerRow * n, n * sizeof(double),
                                             j0 * sizeof(double), w, cudaMemcpyDeviceToHost,
                                             sMain));
            CUDA_CHECK(cudaStreamSynchronize(sMain));
#pragma omp parallel for
            for (size_t r = 0; r < w; ++r)
                for (size_t c = 0; c < w; ++c)
                    hPanel[b][r * Kp + j0 + c] = diagL[r * kBlock + c];
        }

        MPI_Bcast(hPanel[b], (int)(w * Kp), MPI_DOUBLE, owner, comm);

        // Device panel buffer b was last read by the updates of step t-2.
        CUDA_CHECK(cudaStreamWaitEvent(sMain, evBulk[b], 0));
        CUDA_CHECK(cudaMemcpyAsync(dPanel[b], hPanel[b], w * Kp * sizeof(double),
                                   cudaMemcpyHostToDevice, sMain));
        CUDA_CHECK(cudaEventRecord(evH2D[b], sMain));

        // Lead update: the owner of the next diagonal block brings its rows up to date.
        const size_t tn = t + 1;
        if (tn < nblocks && rank == (int)(tn % nprocs)) {
            const size_t lr0 = (tn / nprocs) * kBlock;
            const int wn = (int)std::min(kBlock, n - tn * kBlock);
            gemmNt<true>(dL + lr0 * n, (int)n, dPanel[b], (int)Kp, dPLead, kBlock, wn, (int)w,
                         (int)j0, sMain);
            finishRowsKernel<<<(wn + 255) / 256, 256, 0, sMain>>>(
                dL + lr0 * n, (int)n, dPLead, dPanel[b], (int)Kp, (int)j0, (int)w, wn);
            CUDA_CHECK(cudaGetLastError());
        }

        // Bulk update: local rows of blocks beyond tn (their lead is done or not ours).
        if (tn < nblocks) {
            const size_t lbStart = (tn < (size_t)rank) ? 0 : (tn - rank) / nprocs + 1;
            const size_t rowStart = lbStart * kBlock;
            if (rowStart < validRows) {
                const int M = (int)(validRows - rowStart);
                double* Lt = dL + rowStart * n;
                CUDA_CHECK(cudaStreamWaitEvent(sBulk, evH2D[b], 0));
                gemmNt<true>(Lt, (int)n, dPanel[b], (int)Kp, dPBulk, kBlock, M, (int)w, (int)j0,
                             sBulk);
                finishRowsKernel<<<(M + 255) / 256, 256, 0, sBulk>>>(
                    Lt, (int)n, dPBulk, dPanel[b], (int)Kp, (int)j0, (int)w, M);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(evBulk[b], sBulk));
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventDestroy(evBulk[b]));
        CUDA_CHECK(cudaEventDestroy(evH2D[b]));
        CUDA_CHECK(cudaFree(dPanel[b]));
        CUDA_CHECK(cudaFreeHost(hPanel[b]));
    }
    CUDA_CHECK(cudaStreamDestroy(sMain));
    CUDA_CHECK(cudaStreamDestroy(sBulk));
    CUDA_CHECK(cudaFree(dPd));
    CUDA_CHECK(cudaFree(dPLead));
    CUDA_CHECK(cudaFree(dPBulk));
    return ok;
}

// Copies this rank's rows to the host and gathers the full n x n matrix on every rank.
std::vector<double> gatherFull(const double* dL, size_t n, size_t validRows, int nprocs) {
    std::vector<double> local(validRows * n);
    if (!local.empty())
        CUDA_CHECK(cudaMemcpy(local.data(), dL, local.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));

    std::vector<int> counts(nprocs), displs(nprocs);
    int total = 0;
    for (int r = 0; r < nprocs; ++r) {
        counts[r] = (int)(validRowsOf(r, nprocs, n) * n);
        displs[r] = total;
        total += counts[r];
    }
    std::vector<double> gathered(total);
    MPI_Allgatherv(local.data(), (int)local.size(), MPI_DOUBLE, gathered.data(),
                   counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<double> full(n * n);
    for (int r = 0; r < nprocs; ++r) {
        const size_t rows = validRowsOf(r, nprocs, n);
        const double* src = gathered.data() + displs[r];
#pragma omp parallel for
        for (size_t lr = 0; lr < rows; ++lr) {
            const size_t g = globalRowOf(r, nprocs, lr);
            std::copy(src + lr * n, src + (lr + 1) * n, full.begin() + g * n);
        }
    }
    return full;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = rank == 0;

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

    // One rank per GPU: bind each rank to a device on its node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);
    // Split the node's cores between its ranks so host OpenMP teams do not oversubscribe.
    omp_set_num_threads(std::max(1, omp_get_num_procs() / nodeSize));
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % ndev));

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    const size_t nblocks = (n + kBlock - 1) / kBlock;
    const size_t validRows = validRowsOf(rank, nprocs, n);
    double* dL = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, std::max<size_t>(validRows * n, 1) * sizeof(double)));

    // Generate B with the same sequence as the reference, then A = B * B^T + n*I
    // with each rank computing only the rows it owns.
    {
        std::vector<double> B(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

        double* dB = nullptr;
        CUDA_CHECK(cudaMalloc(&dB, std::max<size_t>(n * n, 1) * sizeof(double)));
        if (n > 0)
            CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

        for (size_t lb = 0;; ++lb) {
            const size_t R = rank + lb * nprocs;
            if (R >= nblocks) break;
            const size_t g0 = R * kBlock;
            const int rows = (int)std::min(kBlock, n - g0);
            double* Lb = dL + lb * kBlock * n;
            gemmNt<false>(dB + g0 * n, (int)n, dB, (int)n, Lb, (int)n, rows, (int)n, (int)n);
            addDiagKernel<<<(rows + 255) / 256, 256>>>(Lb, (int)n, (int)g0, rows);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dB));
    }

    double* dA0 = nullptr;  // original rows, kept only for validation
    if (validate) {
        CUDA_CHECK(cudaMalloc(&dA0, std::max<size_t>(validRows * n, 1) * sizeof(double)));
        if (validRows > 0)
            CUDA_CHECK(cudaMemcpy(dA0, dL, validRows * n * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const bool success = choleskyDistributed(dL, n, validRows, rank, nprocs, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        CUDA_CHECK(cudaFree(dL));
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> L;
    if (printResults || validate) L = gatherFull(dL, n, validRows, nprocs);

    // Print results for external validation
    if (printResults && root) print_results(L, "CholeskyL");

    // Validation: max |L * L^T - A| over all rows, each rank checking its own rows
    if (validate) {
        if (root) printf("Validating result...\n");

        double* dFull = nullptr;
        double* dR = nullptr;
        CUDA_CHECK(cudaMalloc(&dFull, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dFull, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&dR, std::max<size_t>(validRows * n, 1) * sizeof(double)));
        gemmNt<true>(dL, (int)n, dFull, (int)n, dR, (int)n, (int)validRows, (int)n, (int)n);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> recon(validRows * n), orig(validRows * n);
        if (!recon.empty()) {
            CUDA_CHECK(cudaMemcpy(recon.data(), dR, recon.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(orig.data(), dA0, orig.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(dFull));
        CUDA_CHECK(cudaFree(dR));
        CUDA_CHECK(cudaFree(dA0));
        CUDA_CHECK(cudaFree(dL));

        double maxError = 0.0;
        double relError = 0.0;
#pragma omp parallel for reduction(max : maxError, relError)
        for (size_t i = 0; i < recon.size(); ++i) {
            const double error = std::fabs(recon[i] - orig[i]);
            maxError = std::max(maxError, error);
            const double rel = error / (std::fabs(orig[i]) + 1e-10);
            relError = std::max(relError, rel);
        }
        MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        bool valid = true;
        if (root) {
            printf("Max absolute error: %.10e\n", maxError);
            printf("Max relative error: %.10e\n", relError);
        }
        if (relError > 1e-6) {
            if (root) printf("Validation failed: relative error too large\n");
            valid = false;
        }

        if (valid) {
            if (root) printf("Validation: PASSED\n");
            return 0;
        }
        if (root) printf("Validation: FAILED\n");
        return 1;
    }

    CUDA_CHECK(cudaFree(dL));
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
