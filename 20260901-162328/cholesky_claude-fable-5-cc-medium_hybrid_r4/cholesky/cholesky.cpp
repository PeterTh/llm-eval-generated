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

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Parallelization scheme (blocked right-looking algorithm):
//  - Block rows of the matrix are distributed block-cyclically across MPI ranks,
//    each rank driving one GPU (round-robin over the node's devices).
//  - The small diagonal block is factored on the CPU with OpenMP.
//  - The panel triangular solve (TRSM) and the O(n^3) trailing-matrix update
//    (SYRK/GEMM) run as CUDA kernels on each rank's GPU.
//  - The factored panel is exchanged with MPI_Allgatherv each step.
//  - OpenMP additionally parallelizes host-side packing, validation and cleanup.

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                    __FILE__, __LINE__);                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

static constexpr int NB = 128;   // block size (panel width)
static constexpr int TILE = 64;  // GEMM tile size (divides NB)
static constexpr int KT = 16;    // GEMM k-tile depth

static int g_rank = 0;
static int g_nranks = 1;

// ---------------------------------------------------------------------------
// Block-cyclic distribution helpers: global block row b is owned by rank b % P
// and stored at local block slot (b - rank) / P.
// ---------------------------------------------------------------------------

static inline size_t numOwnedBlocks(size_t nblocks, int rank, int nranks) {
    return (nblocks > (size_t)rank) ? (nblocks - rank + nranks - 1) / nranks : 0;
}

static inline size_t blockRows(size_t b, size_t n) {
    return std::min((size_t)NB, n - b * NB);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Generate the locally owned block rows of A = B * B^T + n*I (lower triangle only)
__global__ void genKernel(double* __restrict__ A, const double* __restrict__ B,
                          size_t n, int rank, int nranks, size_t paddedRows) {
    __shared__ double Bi[16][17];
    __shared__ double Bj[16][17];

    const size_t lr0 = (size_t)blockIdx.y * 16;
    const size_t j0 = (size_t)blockIdx.x * 16;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Tiles never straddle block-row boundaries (16 divides NB)
    const size_t lb = lr0 / NB;
    const size_t g0 = ((size_t)rank + lb * (size_t)nranks) * NB + (lr0 % NB);
    if (j0 > g0 + 15) return;  // tile lies entirely in the upper triangle

    double sum = 0.0;
    for (size_t k0 = 0; k0 < n; k0 += 16) {
        const size_t gi = g0 + ty;
        const size_t gj = j0 + ty;
        Bi[ty][tx] = (gi < n && k0 + tx < n) ? B[gi * n + k0 + tx] : 0.0;
        Bj[ty][tx] = (gj < n && k0 + tx < n) ? B[gj * n + k0 + tx] : 0.0;
        __syncthreads();
        const int kl = (int)min((size_t)16, n - k0);
        for (int t = 0; t < kl; ++t) {
            sum += Bi[ty][t] * Bj[tx][t];
        }
        __syncthreads();
    }

    const size_t lr = lr0 + ty;
    const size_t gi = g0 + ty;
    const size_t j = j0 + tx;
    if (lr < paddedRows && gi < n && j <= gi) {
        if (j == gi) sum += (double)n;
        A[lr * n + j] = sum;
    }
}

// Panel triangular solve: X := X * L^{-T} for the local panel rows.
// One thread per row; L is the factored kb x kb diagonal block.
__global__ void trsmKernel(double* __restrict__ A, const double* __restrict__ L,
                           size_t n, int kb, size_t colOffset, size_t rowOffset,
                           size_t mRows) {
    const size_t r = rowOffset + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rowOffset + mRows) return;

    double x[NB];
    double* __restrict__ row = A + r * n + colOffset;
    for (int j = 0; j < kb; ++j) x[j] = row[j];
    for (int j = 0; j < kb; ++j) {
        double s = x[j];
        for (int t = 0; t < j; ++t) {
            s -= x[t] * __ldg(&L[(size_t)j * kb + t]);
        }
        x[j] = s / __ldg(&L[(size_t)j * kb + j]);
    }
    for (int j = 0; j < kb; ++j) row[j] = x[j];
}

// Trailing-matrix update: for locally owned rows i in the trailing region,
//   A[i][j] -= panel[i] . panel[j]   for panelBase <= j <= i
// panel holds the factored panel rows for global rows panelBase-NB..n packed
// tightly with leading dimension kb (row g at index g - panelBase).
__global__ void trailingUpdateKernel(double* __restrict__ A,
                                     const double* __restrict__ panel, size_t n,
                                     int kb, size_t panelBase, size_t rowOffset,
                                     size_t validRows, int rank, int nranks) {
    __shared__ double As[TILE][KT + 1];
    __shared__ double Bs[TILE][KT + 1];

    const size_t lr0 = rowOffset + (size_t)blockIdx.y * TILE;
    const size_t c0 = (size_t)blockIdx.x * TILE;

    // Row tiles never straddle block-row boundaries (TILE divides NB)
    const size_t lb = lr0 / NB;
    const size_t g0 = ((size_t)rank + lb * (size_t)nranks) * NB + (lr0 % NB);
    if (panelBase + c0 > g0 + TILE - 1) return;  // tile entirely above diagonal

    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15

    double acc[4][4] = {};

    for (int kk0 = 0; kk0 < kb; kk0 += KT) {
        // Cooperative load of 64x16 tiles of the panel (A side: rows, B side: cols)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = ty * 4 + i;
            const int kk = kk0 + tx;
            const size_t gr = g0 + r;
            const size_t gc = panelBase + c0 + r;
            As[r][tx] = (kk < kb && gr < n) ? panel[(gr - panelBase) * kb + kk] : 0.0;
            Bs[r][tx] = (kk < kb && gc < n) ? panel[(c0 + r) * kb + kk] : 0.0;
        }
        __syncthreads();

        const int kl = min(KT, kb - kk0);
        for (int t = 0; t < kl; ++t) {
            double a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[ty * 4 + i][t];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[tx * 4 + j][t];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t lr = lr0 + ty * 4 + i;
        const size_t gi = g0 + ty * 4 + i;
        if (lr >= validRows || gi >= n) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t gj = panelBase + c0 + tx * 4 + j;
            if (gj <= gi && gj < n) {
                A[lr * n + gj] -= acc[i][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side pieces
// ---------------------------------------------------------------------------

// Unblocked Cholesky of the kb x kb diagonal block (row-major, lower part),
// with OpenMP-parallel column updates. Returns false if not positive definite.
static bool potrfHost(double* a, int kb, size_t globalBase) {
    // The diagonal block is small; a few threads suffice and avoid
    // fork/join overhead and oversubscription under MPI process binding.
    const int nthreads = std::min(omp_get_max_threads(), 8);
    for (int j = 0; j < kb; ++j) {
        double sum = 0.0;
        for (int t = 0; t < j; ++t) {
            sum += a[(size_t)j * kb + t] * a[(size_t)j * kb + t];
        }
        const double val = a[(size_t)j * kb + j] - sum;
        if (val <= 0.0) {
            // Matrix is not positive definite
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                   globalBase + (size_t)j);
            return false;
        }
        const double d = sqrt(val);
        a[(size_t)j * kb + j] = d;
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
            if (kb - j > 32)
        for (int i = j + 1; i < kb; ++i) {
            double s = 0.0;
            for (int t = 0; t < j; ++t) {
                s += a[(size_t)i * kb + t] * a[(size_t)j * kb + t];
            }
            a[(size_t)i * kb + j] = (a[(size_t)i * kb + j] - s) / d;
        }
    }
    return true;
}

// Gather the distributed block rows onto rank 0 into the full matrix `full`.
// If zeroUpper is set, the strictly upper triangular part is cleared and the
// lower triangle is returned as-is; otherwise the (symmetric) lower triangle
// is mirrored into the upper triangle.
static void gatherMatrix(const double* d_A, size_t n, size_t validRows,
                         std::vector<double>& full, bool zeroUpper) {
    const size_t nblocks = (n + NB - 1) / NB;
    std::vector<double> local(std::max(validRows * n, (size_t)1));
    CUDA_CHECK(cudaMemcpy(local.data(), d_A, validRows * n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::vector<int> counts(g_nranks), displs(g_nranks);
    for (int r = 0; r < g_nranks; ++r) {
        size_t rows = 0;
        for (size_t b = r; b < nblocks; b += g_nranks) rows += blockRows(b, n);
        counts[r] = (int)(rows * n);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }

    std::vector<double> gath;
    if (g_rank == 0) gath.resize(n * n);
    MPI_Gatherv(local.data(), (int)(validRows * n), MPI_DOUBLE, gath.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (g_rank != 0) return;

    full.assign(n * n, 0.0);
    std::vector<size_t> cursor(g_nranks);
    for (int r = 0; r < g_nranks; ++r) cursor[r] = (size_t)displs[r];
    std::vector<std::pair<size_t, size_t>> placement(nblocks);  // block -> src offset
    for (size_t b = 0; b < nblocks; ++b) {
        const int o = (int)(b % g_nranks);
        placement[b] = {cursor[o], blockRows(b, n)};
        cursor[o] += blockRows(b, n) * n;
    }
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < nblocks; ++b) {
        memcpy(full.data() + b * NB * n, gath.data() + placement[b].first,
               placement[b].second * n * sizeof(double));
    }

    if (zeroUpper) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) full[i * n + j] = 0.0;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) full[i * n + j] = full[j * n + i];
        }
    }
}

// Distributed blocked right-looking Cholesky. d_A holds this rank's block rows.
static bool choleskyDecomposition(double* d_A, const size_t n) {
    const size_t nblocks = (n + NB - 1) / NB;
    const size_t owned = numOwnedBlocks(nblocks, g_rank, g_nranks);
    size_t validRows = 0;
    for (size_t b = g_rank; b < nblocks; b += g_nranks) validRows += blockRows(b, n);

    double* d_diag = nullptr;
    double* d_panel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_diag, (size_t)NB * NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel, std::max(n * NB, (size_t)1) * sizeof(double)));

    double *h_diag = nullptr, *h_send = nullptr, *h_recv = nullptr, *h_panel = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_diag, (size_t)NB * NB * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send, std::max(owned * NB * NB, (size_t)1) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv, std::max(n * NB, (size_t)1) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_panel, std::max(n * NB, (size_t)1) * sizeof(double)));

    std::vector<int> counts(g_nranks), displs(g_nranks);
    bool ok = true;

    for (size_t k = 0; k < nblocks; ++k) {
        const int kb = (int)blockRows(k, n);
        const int owner = (int)(k % g_nranks);

        // --- Factor the diagonal block on the owner's CPU (OpenMP) ---
        int fail = 0;
        if (g_rank == owner) {
            const size_t lRow = ((k - owner) / g_nranks) * NB;
            CUDA_CHECK(cudaMemcpy2D(h_diag, kb * sizeof(double),
                                    d_A + lRow * n + k * NB, n * sizeof(double),
                                    kb * sizeof(double), kb, cudaMemcpyDeviceToHost));
            if (!potrfHost(h_diag, kb, k * NB)) fail = 1;
        }
        MPI_Bcast(&fail, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (fail) {
            ok = false;
            break;
        }
        MPI_Bcast(h_diag, kb * kb, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_diag, h_diag, (size_t)kb * kb * sizeof(double),
                              cudaMemcpyHostToDevice));
        if (g_rank == owner) {
            const size_t lRow = ((k - owner) / g_nranks) * NB;
            CUDA_CHECK(cudaMemcpy2D(d_A + lRow * n + k * NB, n * sizeof(double),
                                    h_diag, kb * sizeof(double), kb * sizeof(double),
                                    kb, cudaMemcpyHostToDevice));
        }

        // --- Panel TRSM on the GPU for locally owned block rows below k ---
        const size_t lbStart = (k >= (size_t)g_rank) ? (k - g_rank) / g_nranks + 1 : 0;
        const size_t rowOff = lbStart * NB;
        const size_t mLocal = (validRows > rowOff) ? validRows - rowOff : 0;
        if (mLocal > 0) {
            const int threads = 128;
            const int blocks = (int)((mLocal + threads - 1) / threads);
            trsmKernel<<<blocks, threads>>>(d_A, d_diag, n, kb, k * NB, rowOff, mLocal);
            CUDA_CHECK(cudaGetLastError());
        }

        if (k + 1 == nblocks) break;  // no trailing matrix left

        // --- Allgather the factored panel across ranks ---
        const size_t panelBase = (k + 1) * NB;
        const size_t panelRows = n - panelBase;
        for (int r = 0; r < g_nranks; ++r) {
            size_t rows = 0;
            for (size_t b = (size_t)r; b < nblocks; b += g_nranks) {
                if (b > k) rows += blockRows(b, n);
            }
            counts[r] = (int)(rows * kb);
            displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
        }
        if (mLocal > 0) {
            CUDA_CHECK(cudaMemcpy2D(h_send, kb * sizeof(double),
                                    d_A + rowOff * n + k * NB, n * sizeof(double),
                                    kb * sizeof(double), mLocal,
                                    cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(h_send, (int)(mLocal * kb), MPI_DOUBLE, h_recv, counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Reorder rank-ordered chunks into global block order
        std::vector<size_t> used(g_nranks, 0);
        size_t dstRow = 0;
        for (size_t j = k + 1; j < nblocks; ++j) {
            const int o = (int)(j % g_nranks);
            const size_t mj = blockRows(j, n);
            memcpy(h_panel + dstRow * kb, h_recv + displs[o] + used[o],
                   mj * kb * sizeof(double));
            used[o] += mj * kb;
            dstRow += mj;
        }
        CUDA_CHECK(cudaMemcpy(d_panel, h_panel, panelRows * kb * sizeof(double),
                              cudaMemcpyHostToDevice));

        // --- Trailing-matrix update on the GPU ---
        if (mLocal > 0) {
            const size_t rowsPadded = owned * NB - rowOff;
            dim3 grid((unsigned)((panelRows + TILE - 1) / TILE),
                      (unsigned)((rowsPadded + TILE - 1) / TILE));
            dim3 block(16, 16);
            trailingUpdateKernel<<<grid, block>>>(d_A, d_panel, n, kb, panelBase,
                                                  rowOff, validRows, g_rank, g_nranks);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFreeHost(h_panel));
    CUDA_CHECK(cudaFreeHost(h_recv));
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_diag));
    CUDA_CHECK(cudaFree(d_panel));
    CUDA_CHECK(cudaFree(d_diag));
    return ok;
}

// Generate a symmetric positive definite matrix (locally owned block rows)
// Method: A = B * B^T + n*I with B pseudo-random, identical on all ranks.
static void generatePositiveDefiniteMatrix(double* d_A, const size_t n,
                                           size_t paddedRows) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential PRNG, identical to reference)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    if (paddedRows == 0) return;

    double* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((unsigned)((n + 15) / 16), (unsigned)((paddedRows + 15) / 16));
    genKernel<<<grid, block>>>(d_A, d_B, n, g_rank, g_nranks, paddedRows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_B));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const size_t n) {
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

    // Bind each rank to a GPU (round-robin over the devices on its node)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL,
                        &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices/node: %d\n",
               g_nranks, omp_get_max_threads(), deviceCount);
    }

    // Allocate this rank's block rows of the matrix on its GPU
    const size_t nblocks = (n + NB - 1) / NB;
    const size_t owned = numOwnedBlocks(nblocks, g_rank, g_nranks);
    const size_t paddedRows = owned * NB;
    size_t validRows = 0;
    for (size_t b = g_rank; b < nblocks; b += g_nranks) validRows += blockRows(b, n);

    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, std::max(paddedRows * n, (size_t)1) * sizeof(double)));

    std::vector<double> A;  // full result on rank 0
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (g_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(d_A, n, paddedRows);

    if (validate) {
        gatherMatrix(d_A, n, validRows, A_orig, false);  // save original (rank 0)
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d_A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) printf("Cholesky decomposition failed\n");
        CUDA_CHECK(cudaFree(d_A));
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

    // Collect the factor L on rank 0 (with the upper triangle zeroed)
    gatherMatrix(d_A, n, validRows, A, true);
    CUDA_CHECK(cudaFree(d_A));

    int rc = 0;
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
                rc = 1;
            }
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return rc;
}
