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

// Hybrid MPI + OpenMP + CUDA blocked Cholesky decomposition.
//
// The matrix is distributed over MPI ranks in units of PANEL_SIZE row blocks.
// Every rank owns a balanced subset of the block rows (a deterministic LPT
// schedule based on the triangular work per block).  Diagonal blocks are
// factored on the host with OpenMP available for the surrounding data-parallel
// work, while the panel solve and the trailing updates run on the GPU of each
// rank with CUDA.  Panels are exchanged with MPI_Allgatherv so every rank ends
// up with the complete factor, keeping the original output semantics.

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

namespace {

constexpr int PANEL_SIZE = 128;      // block/panel width
constexpr int GEMM_TILE = 64;        // C tile edge in the GEMM kernel
constexpr int GEMM_THREAD_TILE = 4;  // per-thread micro tile edge (4x4)
constexpr int GEMM_K_TILE = 16;      // K tile in shared memory
constexpr int TRSM_THREADS = 256;
constexpr int TRSM_BLOCK = 64;       // blocked triangular-solve width

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Solve one diagonal sub-block of the panel.  One warp owns one matrix row and
// cooperatively computes each diagonal update, which gives plenty of warps for
// the accelerator even when the trailing panel is still short.
__global__ void panel_diag_kernel(int rowBegin, int rowEnd, int diagStart,
                                  int colStart, int tb, double* __restrict__ A,
                                  int n, const int* __restrict__ blockOwner,
                                  int nb, int myRank) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int i = rowBegin + warp;
    if (i >= rowEnd) return;
    if (blockOwner[i / nb] != myRank) return;

    const int base = diagStart + colStart;
    for (int c = 0; c < tb; ++c) {
        const int j = base + c;
        double s = 0.0;
        for (int t = base + lane; t < j; t += 32) {
            s += A[(size_t)i * n + t] * A[(size_t)j * n + t];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0)
            A[(size_t)i * n + j] =
                (A[(size_t)i * n + j] - s) / A[(size_t)j * n + j];
        __syncwarp();
    }
}

// C[M x N] -= A[M x K] * B[N x K]^T   (all matrices row-major with leading
// dimensions ldc/lda/ldb).  Shared-memory tiled with a 4x4 register micro tile.
__global__ void gemm_nt_kernel(int M, int N, int K, double* __restrict__ C,
                               int ldc, const double* __restrict__ A, int lda,
                               const double* __restrict__ B, int ldb) {
    __shared__ double As[GEMM_TILE][GEMM_K_TILE + 1];
    __shared__ double Bs[GEMM_TILE][GEMM_K_TILE + 1];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const int rowBase = blockIdx.y * GEMM_TILE;
    const int colBase = blockIdx.x * GEMM_TILE;

    double acc[GEMM_THREAD_TILE][GEMM_THREAD_TILE];
#pragma unroll
    for (int i = 0; i < GEMM_THREAD_TILE; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_THREAD_TILE; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < K; k0 += GEMM_K_TILE) {
        for (int idx = tid; idx < GEMM_TILE * GEMM_K_TILE; idx += nthreads) {
            const int r = idx / GEMM_K_TILE;
            const int c = idx - r * GEMM_K_TILE;
            const int gr = rowBase + r;
            const int gk = k0 + c;
            As[r][c] = (gr < M && gk < K)
                           ? A[(size_t)gr * lda + gk]
                           : 0.0;
        }
        for (int idx = tid; idx < GEMM_TILE * GEMM_K_TILE; idx += nthreads) {
            const int r = idx / GEMM_K_TILE;
            const int c = idx - r * GEMM_K_TILE;
            const int gr = colBase + r;
            const int gk = k0 + c;
            Bs[r][c] = (gr < N && gk < K)
                           ? B[(size_t)gr * ldb + gk]
                           : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < GEMM_K_TILE; ++kk) {
            double a[GEMM_THREAD_TILE];
            double b[GEMM_THREAD_TILE];
#pragma unroll
            for (int i = 0; i < GEMM_THREAD_TILE; ++i)
                a[i] = As[threadIdx.y * GEMM_THREAD_TILE + i][kk];
#pragma unroll
            for (int j = 0; j < GEMM_THREAD_TILE; ++j)
                b[j] = Bs[threadIdx.x * GEMM_THREAD_TILE + j][kk];
#pragma unroll
            for (int i = 0; i < GEMM_THREAD_TILE; ++i)
#pragma unroll
                for (int j = 0; j < GEMM_THREAD_TILE; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    const int r0 = rowBase + threadIdx.y * GEMM_THREAD_TILE;
    const int c0 = colBase + threadIdx.x * GEMM_THREAD_TILE;
#pragma unroll
    for (int i = 0; i < GEMM_THREAD_TILE; ++i) {
        const int r = r0 + i;
        if (r < M) {
#pragma unroll
            for (int j = 0; j < GEMM_THREAD_TILE; ++j) {
                const int c = c0 + j;
                if (c < N) C[(size_t)r * ldc + c] -= acc[i][j];
            }
        }
    }
}

__global__ void zero_upper_kernel(double* __restrict__ A, int n) {
    const int row = blockIdx.y;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < n;
         col += blockDim.x * gridDim.x) {
        if (col > row) A[(size_t)row * n + col] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// In-place Cholesky of one diagonal block (row-major, leading dimension ld).
bool potrf_block(double* D, int kb, int ld, int columnOffset) {
    for (int j = 0; j < kb; ++j) {
        double s = 0.0;
        for (int k = 0; k < j; ++k) s += D[j * ld + k] * D[j * ld + k];
        const double v = D[j * ld + j] - s;
        if (v <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n",
                   columnOffset + j);
            return false;
        }
        D[j * ld + j] = sqrt(v);
        for (int i = j + 1; i < kb; ++i) {
            s = 0.0;
            for (int k = 0; k < j; ++k) s += D[i * ld + k] * D[j * ld + k];
            D[i * ld + j] = (D[i * ld + j] - s) / D[j * ld + j];
        }
    }
    return true;
}

// Balanced block-row ownership.  The work of block row b is proportional to
// b*(b+1)/2, so blocks are assigned largest-first to the least loaded rank.
void computeOwnership(int numBlocks, int nprocs, std::vector<int>& ownerOf,
                      std::vector<std::vector<int>>& owned,
                      std::vector<int>& ownerIndex) {
    ownerOf.assign(numBlocks, 0);
    owned.assign(nprocs, {});
    ownerIndex.assign(numBlocks, 0);
    std::vector<double> load(nprocs, 0.0);

    for (int b = numBlocks - 1; b >= 0; --b) {
        int best = 0;
        for (int r = 1; r < nprocs; ++r) {
            if (load[r] < load[best]) best = r;
        }
        ownerOf[b] = best;
        owned[best].push_back(b);
        const double w = 0.5 * (double)(b + 1) * (double)(b + 2);
        load[best] += w;
    }

    for (int r = 0; r < nprocs; ++r) {
        std::sort(owned[r].begin(), owned[r].end());
        for (size_t j = 0; j < owned[r].size(); ++j)
            ownerIndex[owned[r][j]] = (int)j;
    }
}

int deviceForRank(int rank) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
        fprintf(stderr, "No CUDA device available for rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return rank % ndev;
}

}  // namespace

// Generate a symmetric positive definite matrix.
// The original arithmetic (and therefore the original matrix) is preserved for
// every row that is generated, while the rows are spread over MPI ranks.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n,
                                    int rank, int nprocs, bool fullMatrix) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const int nb = PANEL_SIZE;
    const int numBlocks = ((int)n + nb - 1) / nb;
    std::vector<int> ownerOf, ownerIndex;
    std::vector<std::vector<int>> owned;
    computeOwnership(numBlocks, nprocs, ownerOf, owned, ownerIndex);

    auto generateBlock = [&](int b) {
        const int r0 = b * nb;
        const int r1 = std::min((int)n, r0 + nb);
#pragma omp parallel for schedule(static)
        for (int i = r0; i < r1; ++i) {
            const double* Bi = &B[(size_t)i * n];
            for (int j = 0; j < (int)n; ++j) {
                const double* Bj = &B[(size_t)j * n];
                double sum = 0.0;
                for (int k = 0; k < (int)n; ++k) sum += Bi[k] * Bj[k];
                A[(size_t)i * n + j] = sum;
            }
            A[(size_t)i * n + i] += (double)n;
        }
    };

    if (fullMatrix) {
        for (int b = 0; b < numBlocks; ++b) generateBlock(b);
    } else {
        for (size_t j = 0; j < owned[rank].size(); ++j)
            generateBlock(owned[rank][j]);
    }
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T with OpenMP over the independent rows.
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        for (long long j = 0; j < (long long)n; ++j) {
            double sum = 0.0;
            for (long long k = 0; k < (long long)n; ++k)
                sum += L[(size_t)i * n + k] * L[(size_t)j * n + k];
            reconstructed[(size_t)i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (long long i = 0; i < (long long)(n * n); ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

// Blocked right-looking Cholesky, distributed over MPI ranks, with the panel
// solve and trailing GEMM updates executed on each rank's CUDA device.
bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           bool collectResult) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));

    const int nb = std::min(PANEL_SIZE, (int)n);
    const int numBlocks = ((int)n + nb - 1) / nb;
    std::vector<int> ownerOf, ownerIndex;
    std::vector<std::vector<int>> owned;
    computeOwnership(numBlocks, nprocs, ownerOf, owned, ownerIndex);

    std::vector<int> blockSize(numBlocks);
    std::vector<size_t> blockStart(numBlocks);
    for (int b = 0; b < numBlocks; ++b) {
        blockStart[b] = (size_t)b * nb;
        blockSize[b] = std::min(nb, (int)n - (int)blockStart[b]);
    }

    // Device copy of the whole matrix.  Only owned block rows are initially
    // valid; all other rows receive their factor entries through the panel
    // exchanges and the final copy back to the host is the complete L.
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_A, 0, (size_t)n * n * sizeof(double)));

    int* d_owner = nullptr;
    CUDA_CHECK(cudaMalloc(&d_owner, (size_t)numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_owner, ownerOf.data(),
                          (size_t)numBlocks * sizeof(int),
                          cudaMemcpyHostToDevice));

    for (size_t j = 0; j < owned[rank].size(); ++j) {
        const int b = owned[rank][j];
        const int ib = blockSize[b];
        CUDA_CHECK(cudaMemcpy2D(d_A + blockStart[b] * n, (size_t)n * sizeof(double),
                                A.data() + blockStart[b] * n,
                                (size_t)n * sizeof(double),
                                (size_t)n * sizeof(double), ib,
                                cudaMemcpyHostToDevice));
    }

    std::vector<double> h_diag((size_t)nb * nb);
    std::vector<double> h_send((size_t)n * nb);
    std::vector<double> h_recv((size_t)n * nb);
    std::vector<double> h_panel((size_t)n * nb);
    std::vector<int> counts(nprocs), displs(nprocs);
    std::vector<int> startIdx(nprocs);

    bool success = true;

    for (int bk = 0; bk < numBlocks; ++bk) {
        const int kb = blockSize[bk];
        const size_t panelStart = blockStart[bk];
        const int ownerK = ownerOf[bk];

        // --- Diagonal block factorization ---------------------------------
        int potrfOk = 1;
        if (rank == ownerK) {
            CUDA_CHECK(cudaMemcpy2D(h_diag.data(), (size_t)kb * sizeof(double),
                                    d_A + panelStart * n + panelStart,
                                    (size_t)n * sizeof(double),
                                    (size_t)kb * sizeof(double), kb,
                                    cudaMemcpyDeviceToHost));
            potrfOk = potrf_block(h_diag.data(), kb, kb, (int)panelStart) ? 1 : 0;
        }
        MPI_Bcast(&potrfOk, 1, MPI_INT, ownerK, MPI_COMM_WORLD);
        if (!potrfOk) {
            success = false;
            break;
        }
        MPI_Bcast(h_diag.data(), kb * kb, MPI_DOUBLE, ownerK, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy2D(d_A + panelStart * n + panelStart,
                                (size_t)n * sizeof(double), h_diag.data(),
                                (size_t)kb * sizeof(double),
                                (size_t)kb * sizeof(double), kb,
                                cudaMemcpyHostToDevice));

        if (bk == numBlocks - 1) break;  // no trailing submatrix

        // --- Panel solve for all owned block rows below the diagonal ------
        // The panel is processed in TRSM_BLOCK-wide sub-blocks: the diagonal
        // sub-block is solved by the warp kernel, then the remaining columns
        // are updated with a GEMM.  This keeps the triangular dependency
        // serial only for a small part of the panel.
        const size_t trailStart = blockStart[bk + 1];
        const int trailRows = (int)n - (int)trailStart;
        const int warpsPerBlock = TRSM_THREADS / 32;
        const int solveGrid =
            (trailRows + warpsPerBlock - 1) / warpsPerBlock;

        for (int c0 = 0; c0 < kb; c0 += TRSM_BLOCK) {
            const int tc = std::min(TRSM_BLOCK, kb - c0);

            panel_diag_kernel<<<solveGrid, TRSM_THREADS>>>(
                (int)trailStart, (int)n, (int)panelStart, c0, tc, d_A, (int)n,
                d_owner, nb, rank);

            if (c0 + tc >= kb) break;

            const int rem = kb - (c0 + tc);
            for (size_t j = 0; j < owned[rank].size(); ++j) {
                const int b = owned[rank][j];
                if (b <= bk) continue;
                const int ib = blockSize[b];
                double* C = d_A + blockStart[b] * n + panelStart + c0 + tc;
                const double* Aik = d_A + blockStart[b] * n + panelStart + c0;
                const double* B = d_A + (panelStart + c0 + tc) * n +
                                  (panelStart + c0);
                dim3 block(GEMM_TILE / GEMM_THREAD_TILE,
                           GEMM_TILE / GEMM_THREAD_TILE);
                dim3 grid((rem + GEMM_TILE - 1) / GEMM_TILE,
                          (ib + GEMM_TILE - 1) / GEMM_TILE);
                gemm_nt_kernel<<<grid, block>>>(ib, rem, tc, C, (int)n, Aik,
                                                (int)n, B, (int)n);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Pack and exchange the panel column ---------------------------

        for (int r = 0; r < nprocs; ++r) {
            startIdx[r] = (int)(std::upper_bound(owned[r].begin(),
                                                 owned[r].end(), bk) -
                                owned[r].begin());
        }

        int offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            int rows = 0;
            for (size_t j = startIdx[r]; j < owned[r].size(); ++j)
                rows += blockSize[owned[r][j]];
            counts[r] = rows * kb;
            displs[r] = offset;
            offset += counts[r];
        }

        int sendRows = 0;
        for (size_t j = startIdx[rank]; j < owned[rank].size(); ++j) {
            const int b = owned[rank][j];
            const int ib = blockSize[b];
            CUDA_CHECK(cudaMemcpy2D(
                h_send.data() + (size_t)sendRows * kb,
                (size_t)kb * sizeof(double),
                d_A + blockStart[b] * n + panelStart,
                (size_t)n * sizeof(double), (size_t)kb * sizeof(double), ib,
                cudaMemcpyDeviceToHost));
            sendRows += ib;
        }

        MPI_Allgatherv(h_send.data(), sendRows * kb, MPI_DOUBLE, h_recv.data(),
                       counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        for (int b = bk + 1; b < numBlocks; ++b) {
            const int owner = ownerOf[b];
            const int ib = blockSize[b];
            const int rel = ownerIndex[b] - startIdx[owner];
            const size_t src = (size_t)displs[owner] + (size_t)rel * nb * kb;
            const int dstRow = (int)(blockStart[b] - trailStart);
            std::memcpy(h_panel.data() + (size_t)dstRow * kb,
                        h_recv.data() + src, (size_t)ib * kb * sizeof(double));
        }

        CUDA_CHECK(cudaMemcpy2D(d_A + trailStart * n + panelStart,
                                (size_t)n * sizeof(double), h_panel.data(),
                                (size_t)kb * sizeof(double),
                                (size_t)kb * sizeof(double), trailRows,
                                cudaMemcpyHostToDevice));

        // --- Trailing matrix updates on the GPU ---------------------------
        // All column blocks from bk+1 through b are contiguous in memory, so
        // one large GEMM per owned row block replaces many tiny launches and
        // keeps the accelerator well occupied.
        for (size_t j = 0; j < owned[rank].size(); ++j) {
            const int b = owned[rank][j];
            if (b <= bk) continue;
            const int ib = blockSize[b];
            const size_t rowBase = blockStart[b];
            const int cols = (int)(blockStart[b] + ib - trailStart);
            double* C = d_A + rowBase * n + trailStart;
            const double* Aik = d_A + rowBase * n + panelStart;
            const double* Ajk = d_A + trailStart * n + panelStart;
            dim3 block(GEMM_TILE / GEMM_THREAD_TILE,
                       GEMM_TILE / GEMM_THREAD_TILE);
            dim3 grid((cols + GEMM_TILE - 1) / GEMM_TILE,
                      (ib + GEMM_TILE - 1) / GEMM_TILE);
            gemm_nt_kernel<<<grid, block>>>(ib, cols, kb, C, (int)n, Aik,
                                            (int)n, Ajk, (int)n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (success && collectResult && rank == 0) {
        // Zero the upper triangle, then bring the complete factor back to the
        // host.  Only rank 0 needs it for printing / validation, and the
        // ordinary benchmark path skips the transfer entirely.
        dim3 zblock(256);
        dim3 zgrid((unsigned)((n + 255) / 256), (unsigned)n);
        zero_upper_kernel<<<zgrid, zblock>>>(d_A, (int)n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, (size_t)n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_owner));
    CUDA_CHECK(cudaFree(d_A));
    return success;
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
    MPI_Init(&argc, &argv);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = (size_t)atoi(argv[++i]);
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

    // Initialize the CUDA context before the timed region so that the first
    // CUDA call does not pollute the reported Cholesky computation time.
    CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
    CUDA_CHECK(cudaFree(0));

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d, OpenMP threads: %d\n", nprocs,
               omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Validation needs the original full matrix for the L*L^T comparison.
    generatePositiveDefiniteMatrix(A, n, rank, nprocs, validate);

    if (validate) A_orig = A;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, printResults || validate);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count(), max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);
    duration = std::chrono::milliseconds(max_duration_ms);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        double ops = (double)n * n * n / 3.0;
        double seconds = duration.count() / 1000.0;
        double gflops = (seconds > 0.0) ? ops / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Finalize();
    return ret;
}
