#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA blocked (right-looking) Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is padded on the device to a multiple of the tile size NB. The padding
// region is set to the identity so the padded matrix stays positive definite and the
// factorization of the original n x n block is unaffected (the padding rows/cols are
// exactly zero off the diagonal and never contribute to the real block).

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

constexpr int NB = 64;          // tile size (rows/cols per block)
constexpr int KC = 16;          // k-chunk for the tiled GEMM
constexpr int TILE_THREADS = 256;
constexpr int LDS = NB + 1;     // padded shared-memory leading dimension

// ---------------------------------------------------------------------------------------
// Tiled GEMM-NT kernel: C_tile (=|-=) X_rows * Y_rows^T, summing k over [k0, k1).
// X, Y, C all row-major with leading dimension ld. Each block computes one NB x NB tile,
// each thread a 4x4 micro-tile. The k loop runs in increasing order per thread.
// If lowerOnly is set the 1-D grid enumerates only tiles with tileCol <= tileRow.
// ---------------------------------------------------------------------------------------
template <bool SUBTRACT, bool LOWER_ONLY>
__global__ void __launch_bounds__(TILE_THREADS)
gemm_nt_kernel(const double* __restrict__ X, const double* __restrict__ Y, double* __restrict__ C,
               int ld, int k0, int k1, int rowBase, int colBase) {
    __shared__ double Xs[NB][KC + 1];
    __shared__ double Ys[NB][KC + 1];

    int bi, bj;
    if (LOWER_ONLY) {
        // Triangular decode of the 1-D block index into (bi, bj) with bj <= bi
        const int idx = blockIdx.x;
        int r = (int)((sqrt(8.0 * (double)idx + 1.0) - 1.0) * 0.5);
        while (r * (r + 1) / 2 > idx) --r;
        while ((r + 1) * (r + 2) / 2 <= idx) ++r;
        bi = r;
        bj = idx - r * (r + 1) / 2;
    } else {
        bi = blockIdx.y;
        bj = blockIdx.x;
    }

    const int row0 = rowBase + bi * NB;
    const int col0 = colBase + bj * NB;

    const int tid = threadIdx.x;
    const int tx = tid & 15;   // 0..15 -> 4 columns each
    const int ty = tid >> 4;   // 0..15 -> 4 rows each

    // load mapping: each thread loads 4 consecutive k values of one row
    const int lr = tid >> 2;          // 0..63 row within tile
    const int lk = (tid & 3) * 4;     // 0,4,8,12

    double acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0;

    const double* xrow = X + (size_t)(row0 + lr) * ld + lk;
    const double* yrow = Y + (size_t)(col0 + lr) * ld + lk;

    for (int k = k0; k < k1; k += KC) {
        const double4 xv = *reinterpret_cast<const double4*>(xrow + k);
        const double4 yv = *reinterpret_cast<const double4*>(yrow + k);
        Xs[lr][lk + 0] = xv.x; Xs[lr][lk + 1] = xv.y; Xs[lr][lk + 2] = xv.z; Xs[lr][lk + 3] = xv.w;
        Ys[lr][lk + 0] = yv.x; Ys[lr][lk + 1] = yv.y; Ys[lr][lk + 2] = yv.z; Ys[lr][lk + 3] = yv.w;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < KC; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = Xs[ty * 4 + r][kk];
#pragma unroll
            for (int c = 0; c < 4; ++c) b[c] = Ys[tx * 4 + c][kk];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        double* crow = C + (size_t)(row0 + ty * 4 + r) * ld + col0 + tx * 4;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (SUBTRACT) crow[c] -= acc[r][c];
            else          crow[c]  = acc[r][c];
        }
    }
}

// ---------------------------------------------------------------------------------------
// Unblocked Cholesky of the NB x NB diagonal tile starting at (kb, kb). Single block.
// Threads are grouped 4 per row; partial dot products are reduced with warp shuffles.
// On a non-positive pivot the global index is recorded in *flag (first failure only)
// and the kernel stops; later panels return immediately once a failure is recorded.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
potrf_tile_kernel(double* __restrict__ A, int ld, int kb, int* __restrict__ flag) {
    __shared__ double s[NB][LDS];
    __shared__ int stop;

    const int tid = threadIdx.x;
    if (tid == 0) stop = (*flag >= 0) ? 1 : 0;
    __syncthreads();
    if (stop) return;

    double* tile = A + (size_t)kb * ld + kb;
    for (int e = tid; e < NB * NB; e += TILE_THREADS) {
        s[e / NB][e % NB] = tile[(size_t)(e / NB) * ld + (e % NB)];
    }
    __syncthreads();

    const int i = tid >> 2;   // row handled by this thread group
    const int c = tid & 3;    // lane within group
    const unsigned gmask = 0xFu << ((tid & 31) & ~3);

    for (int j = 0; j < NB; ++j) {
        // diagonal element: computed by the group that owns row j
        if (i == j) {
            double part = 0.0;
            for (int t = c; t < j; t += 4) part = fma(s[j][t], s[j][t], part);
            part += __shfl_xor_sync(gmask, part, 1);
            part += __shfl_xor_sync(gmask, part, 2);
            if (c == 0) {
                const double val = s[j][j] - part;
                if (val <= 0.0) {
                    stop = 1;
                    atomicCAS(flag, -1, kb + j);
                } else {
                    s[j][j] = sqrt(val);
                }
            }
        }
        __syncthreads();
        if (stop) return;

        // off-diagonal elements of column j (rows i > j)
        if (i > j) {
            double part = 0.0;
            for (int t = c; t < j; t += 4) part = fma(s[i][t], s[j][t], part);
            part += __shfl_xor_sync(gmask, part, 1);
            part += __shfl_xor_sync(gmask, part, 2);
            if (c == 0) s[i][j] = (s[i][j] - part) / s[j][j];
        }
        __syncthreads();
    }

    for (int e = tid; e < NB * NB; e += TILE_THREADS) {
        tile[(size_t)(e / NB) * ld + (e % NB)] = s[e / NB][e % NB];
    }
}

// ---------------------------------------------------------------------------------------
// Triangular solve for the panel below the diagonal tile:
//   L21 = A21 * L11^{-T}   (L11 lower triangular, NB x NB)
// Block b handles row tile (kb + NB + b*NB). Each row is processed by a group of 4 lanes.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
trsm_panel_kernel(double* __restrict__ A, int ld, int kb) {
    __shared__ double Ls[NB][LDS];

    const int tid = threadIdx.x;
    const double* L11 = A + (size_t)kb * ld + kb;
    double* tile = A + (size_t)(kb + NB + blockIdx.x * NB) * ld + kb;

    for (int e = tid; e < NB * NB; e += TILE_THREADS) {
        const int r = e / NB, q = e % NB;
        Ls[r][q] = L11[(size_t)r * ld + q];
    }

    const int i = tid >> 2;   // row of the tile handled by this lane group
    const int c = tid & 3;    // lane within group: owns columns t = c + 4*q
    const unsigned gmask = 0xFu << ((tid & 31) & ~3);

    // Row i of the tile, distributed over the 4 lanes of the group (stride 4)
    double x[NB / 4];
    double* row = tile + (size_t)i * ld;
#pragma unroll
    for (int q = 0; q < NB / 4; ++q) x[q] = row[c + 4 * q];
    __syncthreads();

#pragma unroll
    for (int j = 0; j < NB; ++j) {
        double part = 0.0;
#pragma unroll
        for (int q = 0; q < NB / 4; ++q) {
            const int t = c + 4 * q;
            if (t < j) part = fma(x[q], Ls[j][t], part);
        }
        part += __shfl_xor_sync(gmask, part, 1);
        part += __shfl_xor_sync(gmask, part, 2);
        if (c == (j & 3)) x[j >> 2] = (x[j >> 2] - part) / Ls[j][j];
    }

#pragma unroll
    for (int q = 0; q < NB / 4; ++q) row[c + 4 * q] = x[q];
}

// Zero the strictly upper triangular part of an m x m matrix with leading dimension ld.
__global__ void zero_upper_kernel(double* __restrict__ A, int ld, int m) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (i < m && j < m && j > i) A[(size_t)i * ld + j] = 0.0;
}

// Set up the padded device matrix: copy of the n x n block, identity in the padding.
__global__ void pad_identity_kernel(double* __restrict__ A, int ld, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (i >= ld || j >= ld) return;
    if (i >= n || j >= n) A[(size_t)i * ld + j] = (i == j) ? 1.0 : 0.0;
}

__global__ void add_diag_kernel(double* __restrict__ A, int ld, int n, double v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[(size_t)i * ld + i] += v;
}

// Error reduction for validation: max |R - A| and max |R - A| / (|A| + 1e-10) over n x n.
// Non-negative doubles order like their bit patterns, so atomicMax on uint64 works.
__global__ void error_reduce_kernel(const double* __restrict__ R, const double* __restrict__ A,
                                    int ld, int n, unsigned long long* __restrict__ maxAbs,
                                    unsigned long long* __restrict__ maxRel) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    double e = 0.0, rel = 0.0;
    if (i < n && j < n) {
        const double a = A[(size_t)i * ld + j];
        e = fabs(R[(size_t)i * ld + j] - a);
        rel = e / (fabs(a) + 1e-10);
    }
    // warp-level reduction, then one atomic per warp
    for (int off = 16; off > 0; off >>= 1) {
        e = fmax(e, __shfl_xor_sync(0xFFFFFFFFu, e, off));
        rel = fmax(rel, __shfl_xor_sync(0xFFFFFFFFu, rel, off));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(maxAbs, __double_as_longlong(e));
        atomicMax(maxRel, __double_as_longlong(rel));
    }
}

// ---------------------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------------------
static inline int paddedSize(size_t n) {
    return (int)((n + NB - 1) / NB * NB);
}

// Upload an n x n row-major host matrix into a padded ld x ld device matrix.
static void uploadPadded(double* d_A, const std::vector<double>& A, size_t n, int ld) {
    CUDA_CHECK(cudaMemcpy2D(d_A, (size_t)ld * sizeof(double), A.data(), n * sizeof(double),
                            n * sizeof(double), n, cudaMemcpyHostToDevice));
    dim3 blk(256), grd((ld + 255) / 256, ld);
    pad_identity_kernel<<<grd, blk>>>(d_A, ld, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

static void downloadPadded(std::vector<double>& A, const double* d_A, size_t n, int ld) {
    CUDA_CHECK(cudaMemcpy2D(A.data(), n * sizeof(double), d_A, (size_t)ld * sizeof(double),
                            n * sizeof(double), n, cudaMemcpyDeviceToHost));
}

// C = X * Y^T for the full padded ld x ld matrices (k over [0, ld)).
static void gemmNTFull(const double* d_X, const double* d_Y, double* d_C, int ld) {
    const int tiles = ld / NB;
    dim3 grd(tiles, tiles);
    gemm_nt_kernel<false, false><<<grd, TILE_THREADS>>>(d_X, d_Y, d_C, ld, 0, ld, 0, 0);
    CUDA_CHECK(cudaGetLastError());
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    const int ld = paddedSize(n);

    double* d_A = nullptr;
    int* d_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)ld * ld * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0xFF, sizeof(int)));  // -1: no failure
    uploadPadded(d_A, A, n, ld);

    const int ntiles = ld / NB;
    for (int t = 0; t < ntiles; ++t) {
        const int kb = t * NB;
        const int rem = ntiles - t - 1;  // tiles below / right of the diagonal tile

        // 1. factor the diagonal tile
        potrf_tile_kernel<<<1, TILE_THREADS>>>(d_A, ld, kb, d_flag);
        CUDA_CHECK(cudaGetLastError());
        if (rem == 0) break;

        // 2. triangular solve of the panel below the diagonal tile
        trsm_panel_kernel<<<rem, TILE_THREADS>>>(d_A, ld, kb);
        CUDA_CHECK(cudaGetLastError());

        // 3. trailing update of the lower triangle: A22 -= L21 * L21^T
        const int nblocks = rem * (rem + 1) / 2;
        const double* panel = d_A + (size_t)(kb + NB) * ld;
        gemm_nt_kernel<true, true><<<nblocks, TILE_THREADS>>>(panel, panel, d_A + (size_t)(kb + NB) * ld + (kb + NB),
                                                              ld, kb, kb + NB, 0, 0);
        CUDA_CHECK(cudaGetLastError());
    }

    // Zero out upper triangular part
    {
        dim3 blk(256), grd((ld + 255) / 256, ld);
        zero_upper_kernel<<<grd, blk>>>(d_A, ld, ld);
        CUDA_CHECK(cudaGetLastError());
    }

    int flag = -1;
    CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    if (flag >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)flag);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_flag));
        return false;
    }

    downloadPadded(A, d_A, n, ld);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_flag));
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential PRNG stream, identical to the reference)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T on the GPU (padding of B is zero so it does not contribute)
    const int ld = paddedSize(n);
    double* d_B = nullptr;
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)ld * ld * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)ld * ld * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_B, 0, (size_t)ld * ld * sizeof(double)));
    CUDA_CHECK(cudaMemcpy2D(d_B, (size_t)ld * sizeof(double), B.data(), n * sizeof(double),
                            n * sizeof(double), n, cudaMemcpyHostToDevice));
    gemmNTFull(d_B, d_B, d_A, ld);

    // Add diagonal dominance to ensure positive definiteness
    add_diag_kernel<<<(unsigned)((n + 255) / 256), 256>>>(d_A, ld, (int)n, (double)n);
    CUDA_CHECK(cudaGetLastError());

    downloadPadded(A, d_A, n, ld);
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    const int ld = paddedSize(n);
    double* d_L = nullptr;
    double* d_R = nullptr;
    double* d_A = nullptr;
    unsigned long long* d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_L, (size_t)ld * ld * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_R, (size_t)ld * ld * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)ld * ld * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_max, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_L, 0, (size_t)ld * ld * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_max, 0, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy2D(d_L, (size_t)ld * sizeof(double), L.data(), n * sizeof(double),
                            n * sizeof(double), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_A, (size_t)ld * sizeof(double), A_orig.data(), n * sizeof(double),
                            n * sizeof(double), n, cudaMemcpyHostToDevice));

    // Compute L * L^T
    gemmNTFull(d_L, d_L, d_R, ld);

    // Compare with original
    {
        dim3 blk(256), grd((unsigned)((n + 255) / 256), (unsigned)n);
        error_reduce_kernel<<<grd, blk>>>(d_R, d_A, ld, (int)n, d_max, d_max + 1);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long bits[2];
    CUDA_CHECK(cudaMemcpy(bits, d_max, sizeof(bits), cudaMemcpyDeviceToHost));
    double maxError, relError;
    memcpy(&maxError, &bits[0], sizeof(double));
    memcpy(&relError, &bits[1], sizeof(double));

    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_max));

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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initialize the CUDA context up front so it is not attributed to the computation
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition (timing includes host<->device transfers)
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %ld ms\n", (long)duration.count());

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
