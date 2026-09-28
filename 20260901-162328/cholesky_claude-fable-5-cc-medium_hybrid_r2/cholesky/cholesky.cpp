#include <algorithm>
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

// Hybrid parallel Cholesky decomposition (MPI + OpenMP + CUDA)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Parallelization strategy:
//  - Block rows of A are distributed 1D block-cyclically across MPI ranks;
//    each rank drives one GPU (round-robin over the GPUs of its node).
//  - Right-looking blocked algorithm: the diagonal block is factorized on the
//    host with OpenMP, the panel below it is solved with a CUDA TRSM kernel,
//    the panel is allgathered across ranks, and the trailing submatrix is
//    updated with a tiled CUDA GEMM kernel on each rank's GPU.
//  - OpenMP additionally parallelizes matrix generation staging, panel
//    packing/unpacking, and validation on the host.

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Block size for the distributed blocked algorithm
constexpr int NB = 256;

static int g_rank = 0;
static int g_nranks = 1;

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Tiled GEMM: C (M x N) op= A (M x K) * B (N x K)^T, all row-major.
// MODE 0: C = A * B^T   (used for matrix generation)
// MODE 1: C -= A * B^T  (used for the trailing update)
template <int MODE>
__global__ void gemm_nt_kernel(int M, int N, int K,
                               const double* __restrict__ A, size_t lda,
                               const double* __restrict__ B, size_t ldb,
                               double* __restrict__ C, size_t ldc) {
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    __shared__ double sA[BK][BM];
    __shared__ double sB[BK][BN];

    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int tid = ty * 16 + tx;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int t = tid; t < BM * BK; t += 256) {
            const int i = t / BK, kk = t % BK;
            const int r = rowBase + i, c = k0 + kk;
            sA[kk][i] = (r < M && c < K) ? A[(size_t)r * lda + c] : 0.0;
        }
        for (int t = tid; t < BN * BK; t += 256) {
            const int j = t / BK, kk = t % BK;
            const int r = colBase + j, c = k0 + kk;
            sB[kk][j] = (r < N && c < K) ? B[(size_t)r * ldb + c] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = sA[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = sB[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = rowBase + ty * TM + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = colBase + tx * TN + j;
            if (c >= N) continue;
            if (MODE == 0)
                C[(size_t)r * ldc + c] = acc[i][j];
            else
                C[(size_t)r * ldc + c] -= acc[i][j];
        }
    }
}

template <int MODE>
static void launch_gemm_nt(int M, int N, int K,
                           const double* A, size_t lda,
                           const double* B, size_t ldb,
                           double* C, size_t ldc) {
    if (M <= 0 || N <= 0) return;
    dim3 threads(16, 16);
    dim3 blocks((N + 63) / 64, (M + 63) / 64);
    gemm_nt_kernel<MODE><<<blocks, threads>>>(M, N, K, A, lda, B, ldb, C, ldc);
    CUDA_CHECK(cudaGetLastError());
}

// TRSM (right, lower, transposed): solve X * L^T = A in place for a panel of
// M rows and kb columns. One thread per panel row; the row is staged through
// per-thread local memory so accesses are coalesced across the warp.
__global__ void trsm_rlt_kernel(int M, int kb,
                                double* __restrict__ X, size_t ldx,
                                const double* __restrict__ L, int ldl) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;

    double x[NB];
    double* row = X + (size_t)r * ldx;
    for (int j = 0; j < kb; ++j) x[j] = row[j];

    for (int j = 0; j < kb; ++j) {
        double s = x[j];
        for (int p = 0; p < j; ++p) s -= x[p] * L[j * ldl + p];
        x[j] = s / L[j * ldl + j];
    }

    for (int j = 0; j < kb; ++j) row[j] = x[j];
}

// Add `val` to the diagonal entries of an owned block row: local row t of the
// block corresponds to global row (gRow0 + t) and thus global column gRow0 + t.
__global__ void add_diag_kernel(double* __restrict__ Ablock, size_t lda,
                                int rows, size_t gRow0, double val) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < rows) Ablock[(size_t)t * lda + gRow0 + t] += val;
}

// ---------------------------------------------------------------------------
// Block-cyclic distribution helpers
// ---------------------------------------------------------------------------

struct Distribution {
    size_t n = 0;
    int nblocks = 0;                 // total number of block rows
    std::vector<int> localBlocks;    // global block indices owned by this rank
    std::vector<size_t> localOff;    // local row offset of each owned block (+ total)
    size_t localRows = 0;

    void init(size_t n_) {
        n = n_;
        nblocks = (int)((n + NB - 1) / NB);
        localBlocks.clear();
        localOff.assign(1, 0);
        for (int b = g_rank; b < nblocks; b += g_nranks) {
            localBlocks.push_back(b);
            localOff.push_back(localOff.back() + blockRows(b));
        }
        localRows = localOff.back();
    }

    size_t blockRows(int b) const {
        return std::min((size_t)NB, n - (size_t)b * NB);
    }
    static int owner(int b) { return b % g_nranks; }
    // Rows owned by rank r that belong to blocks with global index > k
    size_t tailRowsOf(int r, int k, size_t n_) const {
        size_t rows = 0;
        for (int b = (k + 1) + ((r - (k + 1)) % g_nranks + g_nranks) % g_nranks;
             b < nblocks; b += g_nranks) {
            if (b % g_nranks == r) rows += blockRows(b);
        }
        (void)n_;
        return rows;
    }
};

// ---------------------------------------------------------------------------
// Host-side factorization of the diagonal block (OpenMP)
// ---------------------------------------------------------------------------

static bool potrfHost(double* D, int kb, int ld, size_t globalOffset) {
    for (int j = 0; j < kb; ++j) {
        double s = D[(size_t)j * ld + j];
        for (int p = 0; p < j; ++p) s -= D[(size_t)j * ld + p] * D[(size_t)j * ld + p];
        if (s <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                   globalOffset + (size_t)j);
            return false;
        }
        const double d = sqrt(s);
        D[(size_t)j * ld + j] = d;
        #pragma omp parallel for schedule(static)
        for (int i = j + 1; i < kb; ++i) {
            double t = D[(size_t)i * ld + j];
            for (int p = 0; p < j; ++p) t -= D[(size_t)i * ld + p] * D[(size_t)j * ld + p];
            D[(size_t)i * ld + j] = t / d;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Distributed Cholesky decomposition
// ---------------------------------------------------------------------------

bool choleskyDecomposition(double* dA, const Distribution& dist) {
    const size_t n = dist.n;
    const int nblocks = dist.nblocks;

    double* hDiag = nullptr;
    double* hPanelSend = nullptr;
    double* hPanelRecv = nullptr;
    double* hPanel = nullptr;
    CUDA_CHECK(cudaMallocHost(&hDiag, (size_t)NB * NB * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hPanelSend, dist.localRows * NB * sizeof(double) + sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hPanelRecv, n * NB * sizeof(double) + sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hPanel, n * NB * sizeof(double) + sizeof(double)));

    double* dLkk = nullptr;
    double* dPanel = nullptr;
    CUDA_CHECK(cudaMalloc(&dLkk, (size_t)NB * NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPanel, (n * NB + 1) * sizeof(double)));

    std::vector<int> counts(g_nranks), displs(g_nranks);
    bool ok = true;

    for (int k = 0; k < nblocks && ok; ++k) {
        const int kb = (int)dist.blockRows(k);
        const int owner = Distribution::owner(k);
        const size_t colOff = (size_t)k * NB;

        // Factor the diagonal block on the owner's host CPU (OpenMP)
        int status = 1;
        if (g_rank == owner) {
            const size_t loff = dist.localOff[k / g_nranks];
            CUDA_CHECK(cudaMemcpy2D(hDiag, (size_t)kb * sizeof(double),
                                    dA + loff * n + colOff, n * sizeof(double),
                                    (size_t)kb * sizeof(double), kb,
                                    cudaMemcpyDeviceToHost));
            status = potrfHost(hDiag, kb, kb, colOff) ? 1 : 0;
            if (status) {
                CUDA_CHECK(cudaMemcpy2D(dA + loff * n + colOff, n * sizeof(double),
                                        hDiag, (size_t)kb * sizeof(double),
                                        (size_t)kb * sizeof(double), kb,
                                        cudaMemcpyHostToDevice));
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!status) {
            ok = false;
            break;
        }
        MPI_Bcast(hDiag, kb * kb, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dLkk, hDiag, (size_t)kb * kb * sizeof(double),
                              cudaMemcpyHostToDevice));

        // Panel solve: rows of all owned blocks below k form a contiguous tail
        // of the local buffer.
        int firstTail = (int)dist.localBlocks.size();
        for (size_t lb = 0; lb < dist.localBlocks.size(); ++lb) {
            if (dist.localBlocks[lb] > k) {
                firstTail = (int)lb;
                break;
            }
        }
        const size_t tailOff = dist.localOff[firstTail];
        const size_t myTailRows = dist.localRows - tailOff;

        if (myTailRows > 0) {
            const int threads = 128;
            const int blocks = (int)((myTailRows + threads - 1) / threads);
            trsm_rlt_kernel<<<blocks, threads>>>((int)myTailRows, kb,
                                                 dA + tailOff * n + colOff, n,
                                                 dLkk, kb);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy2D(hPanelSend, (size_t)kb * sizeof(double),
                                    dA + tailOff * n + colOff, n * sizeof(double),
                                    (size_t)kb * sizeof(double), myTailRows,
                                    cudaMemcpyDeviceToHost));
        }

        // Allgather the panel so every rank can update its trailing blocks
        const size_t panelRow0 = (size_t)(k + 1) * NB;
        const size_t panelRows = (panelRow0 < n) ? n - panelRow0 : 0;
        if (panelRows > 0) {
            size_t total = 0;
            for (int r = 0; r < g_nranks; ++r) {
                const size_t rows = dist.tailRowsOf(r, k, n);
                counts[r] = (int)(rows * kb);
                displs[r] = (int)total;
                total += rows * kb;
            }
            MPI_Allgatherv(hPanelSend, (int)(myTailRows * kb), MPI_DOUBLE,
                           hPanelRecv, counts.data(), displs.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);

            // Reorder rank-ordered chunks into global row order
            std::vector<size_t> chunkOff(g_nranks);
            for (int r = 0; r < g_nranks; ++r) chunkOff[r] = (size_t)displs[r];
            std::vector<size_t> srcOff(nblocks - (k + 1));
            for (int b = k + 1; b < nblocks; ++b) {
                const int r = Distribution::owner(b);
                srcOff[b - (k + 1)] = chunkOff[r];
                chunkOff[r] += dist.blockRows(b) * kb;
            }
            #pragma omp parallel for schedule(static)
            for (int b = k + 1; b < nblocks; ++b) {
                const size_t rows = dist.blockRows(b);
                memcpy(hPanel + ((size_t)b * NB - panelRow0) * kb,
                       hPanelRecv + srcOff[b - (k + 1)],
                       rows * kb * sizeof(double));
            }
            CUDA_CHECK(cudaMemcpy(dPanel, hPanel,
                                  panelRows * kb * sizeof(double),
                                  cudaMemcpyHostToDevice));

            // Trailing update on the GPU: A_i,(k+1..i) -= L_ik * Panel^T
            for (size_t lb = firstTail; lb < dist.localBlocks.size(); ++lb) {
                const int b = dist.localBlocks[lb];
                const size_t loff = dist.localOff[lb];
                const int M = (int)dist.blockRows(b);
                const int N = (int)((size_t)b * NB + dist.blockRows(b) - panelRow0);
                launch_gemm_nt<1>(M, N, kb,
                                  dA + loff * n + colOff, n,
                                  dPanel, kb,
                                  dA + loff * n + panelRow0, n);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(dLkk));
    CUDA_CHECK(cudaFree(dPanel));
    CUDA_CHECK(cudaFreeHost(hDiag));
    CUDA_CHECK(cudaFreeHost(hPanelSend));
    CUDA_CHECK(cudaFreeHost(hPanelRecv));
    CUDA_CHECK(cudaFreeHost(hPanel));

    return ok;
}

// ---------------------------------------------------------------------------
// Matrix generation (distributed, GPU GEMM per owned block row)
// ---------------------------------------------------------------------------

void generatePositiveDefiniteMatrix(double* dA, const Distribution& dist) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    const size_t n = dist.n;

    // Every rank generates the identical B (deterministic seed), then computes
    // only its own block rows of A = B * B^T on its GPU.
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    for (size_t lb = 0; lb < dist.localBlocks.size(); ++lb) {
        const int b = dist.localBlocks[lb];
        const size_t loff = dist.localOff[lb];
        const int rows = (int)dist.blockRows(b);
        launch_gemm_nt<0>(rows, (int)n, (int)n,
                          dB + (size_t)b * NB * n, n,
                          dB, n,
                          dA + loff * n, n);
        // Add diagonal dominance to ensure positive definiteness
        add_diag_kernel<<<(rows + 127) / 128, 128>>>(dA + loff * n, n, rows,
                                                     (size_t)b * NB, (double)n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

// ---------------------------------------------------------------------------
// Gather the distributed matrix onto rank 0 (host)
// ---------------------------------------------------------------------------

void gatherMatrix(const double* dA, const Distribution& dist,
                  std::vector<double>& A) {
    const size_t n = dist.n;
    std::vector<double> local(dist.localRows * n);
    if (dist.localRows > 0) {
        CUDA_CHECK(cudaMemcpy(local.data(), dA,
                              dist.localRows * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts(g_nranks), displs(g_nranks);
    std::vector<double> recv;
    if (g_rank == 0) recv.resize(n * n);
    size_t total = 0;
    for (int r = 0; r < g_nranks; ++r) {
        size_t rows = 0;
        for (int b = r; b < dist.nblocks; b += g_nranks) rows += dist.blockRows(b);
        counts[r] = (int)(rows * n);
        displs[r] = (int)total;
        total += rows * n;
    }
    MPI_Gatherv(local.data(), (int)(dist.localRows * n), MPI_DOUBLE,
                recv.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (g_rank == 0) {
        std::vector<size_t> chunkOff(g_nranks);
        for (int r = 0; r < g_nranks; ++r) chunkOff[r] = (size_t)displs[r];
        std::vector<size_t> srcOff(dist.nblocks);
        for (int b = 0; b < dist.nblocks; ++b) {
            const int r = Distribution::owner(b);
            srcOff[b] = chunkOff[r];
            chunkOff[r] += dist.blockRows(b) * n;
        }
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < dist.nblocks; ++b) {
            memcpy(A.data() + (size_t)b * NB * n, recv.data() + srcOff[b],
                   dist.blockRows(b) * n * sizeof(double));
        }
    }
}

// ---------------------------------------------------------------------------
// Validation (rank 0, OpenMP)
// ---------------------------------------------------------------------------

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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    // Bind each rank to a GPU (round-robin over the GPUs of its node)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank,
                        MPI_INFO_NULL, &nodeComm);
    int localRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);
    // Avoid host thread oversubscription: split the node's cores among ranks
    // unless the user explicitly set OMP_NUM_THREADS.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / nodeSize));
    }
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(localRank % ndev));

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
            if (g_rank == 0) printUsage(argv[0]);
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
        printf("MPI ranks: %d, GPUs per node: %d, OpenMP threads: %d\n",
               g_nranks, ndev, omp_get_max_threads());
    }

    Distribution dist;
    dist.init(n);

    // Allocate the distributed matrix on the GPU
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, (dist.localRows * n + 1) * sizeof(double)));

    // Generate positive definite matrix
    if (g_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dA, dist);

    std::vector<double> A;
    if (g_rank == 0) A.resize(n * n);
    std::vector<double> A_orig;
    if (validate) {
        if (g_rank == 0) A_orig.resize(n * n);
        gatherMatrix(dA, dist, A_orig); // Save original for validation
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(dA, dist);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) printf("Cholesky decomposition failed\n");
        CUDA_CHECK(cudaFree(dA));
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

    // Collect the factor on rank 0 and zero out the upper triangular part
    gatherMatrix(dA, dist, A);
    CUDA_CHECK(cudaFree(dA));
    if (g_rank == 0) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
            }
        }
    }

    int exitCode = 0;
    if (g_rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
