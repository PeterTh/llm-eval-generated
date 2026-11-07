#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA blocked Cholesky decomposition (right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Per panel of NB columns:
//   1. POTRF : factor the NB x NB diagonal tile (single thread block, shared memory)
//   2. TRSM  : solve the sub-diagonal panel  X * L_kk^T = A_ik
//   3. SYRK  : trailing update  A_22 -= X * X^T  (lower triangle only, tiled GEMM)
// The whole matrix stays resident on the device; the final upper triangle is zeroed
// on the device as well.

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                            \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// Panel / tile size of the blocked factorization
constexpr int NB = 64;
constexpr int NB_PAD = NB + 1;  // shared-memory padding to avoid bank conflicts

// GEMM tiling parameters: 64x64 output tile, 16-deep K chunks, 256 threads, 4x4 per thread
constexpr int GEMM_TILE = 64;
constexpr int GEMM_K = 16;
constexpr int GEMM_THREADS = 256;
constexpr int GEMM_TILE_PAD = GEMM_TILE + 1;

// Rows handled per thread block in the TRSM kernel (16 threads cooperate per row)
constexpr int TRSM_ROWS = 16;
constexpr int TRSM_TPR = 16;

// Index of the first non-positive pivot (NO_FAIL if the matrix is positive definite so far).
// Recorded with atomicMin, so the sentinel must be larger than any valid index.
constexpr int NO_FAIL = 0x7fffffff;
__device__ int g_fail = NO_FAIL;

// ---------------------------------------------------------------------------------------
// POTRF: Cholesky factorization of the diagonal tile A[k0..k0+w, k0..k0+w] in shared memory.
// Launched with a single block of 256 threads.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) potrf_tile_kernel(double* __restrict__ A, const int n,
                                                         const int k0, const int w) {
    __shared__ double tile[NB][NB_PAD];
    __shared__ int s_fail;

    if (g_fail != NO_FAIL) return;  // an earlier panel already failed

    const int tid = threadIdx.x;
    if (tid == 0) s_fail = -1;

    // Load lower triangle of the tile (upper part unused)
    for (int e = tid; e < NB * NB; e += blockDim.x) {
        const int i = e / NB, j = e % NB;
        tile[i][j] = (i < w && j < w && j <= i) ? A[(size_t)(k0 + i) * n + (k0 + j)] : 0.0;
    }
    __syncthreads();

    for (int j = 0; j < w; ++j) {
        if (tid == 0) {
            const double val = tile[j][j];
            if (val <= 0.0) {
                s_fail = k0 + j;
            } else {
                tile[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        if (s_fail >= 0) break;

        // Scale column j below the diagonal
        const double d = tile[j][j];
        if (tid > j && tid < w) {
            tile[tid][j] /= d;
        }
        __syncthreads();

        // Rank-1 update of the remaining lower-right part: (i,k) with j < k <= i < w
        for (int e = tid; e < NB * NB; e += blockDim.x) {
            const int i = e / NB, k = e % NB;
            if (k > j && k <= i && i < w) {
                tile[i][k] -= tile[i][j] * tile[k][j];
            }
        }
        __syncthreads();
    }

    if (s_fail >= 0) {
        if (tid == 0) atomicMin(&g_fail, s_fail);
        return;
    }

    // Store lower triangle back
    for (int e = tid; e < NB * NB; e += blockDim.x) {
        const int i = e / NB, j = e % NB;
        if (i < w && j < w && j <= i) {
            A[(size_t)(k0 + i) * n + (k0 + j)] = tile[i][j];
        }
    }
}

// ---------------------------------------------------------------------------------------
// TRSM: for rows r in [r0, n):  A[r, k0..k0+w] <- A[r, k0..k0+w] * L_kk^{-T}
// i.e. forward substitution with the lower-triangular tile L_kk = A[k0..k0+w, k0..k0+w].
// Block: 256 threads = TRSM_ROWS rows x TRSM_TPR threads per row.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TRSM_ROWS* TRSM_TPR) trsm_panel_kernel(double* __restrict__ A,
                                                                        const int n, const int k0,
                                                                        const int w, const int r0) {
    __shared__ double L[NB][NB_PAD];
    __shared__ double X[TRSM_ROWS][NB_PAD];

    if (g_fail != NO_FAIL) return;

    const int tid = threadIdx.x;
    const int lr = tid / TRSM_TPR;  // local row
    const int lt = tid % TRSM_TPR;  // thread within row
    const int row = r0 + blockIdx.x * TRSM_ROWS + lr;
    const bool rowValid = row < n;

    // Load L_kk (lower triangle incl. diagonal)
    for (int e = tid; e < NB * NB; e += blockDim.x) {
        const int i = e / NB, j = e % NB;
        L[i][j] = (i < w && j <= i) ? A[(size_t)(k0 + i) * n + (k0 + j)] : 0.0;
    }
    // Load panel rows (coalesced along columns)
    #pragma unroll
    for (int q = 0; q < NB / TRSM_TPR; ++q) {
        const int c = lt + q * TRSM_TPR;
        X[lr][c] = (rowValid && c < w) ? A[(size_t)row * n + (k0 + c)] : 0.0;
    }
    __syncthreads();

    // Forward substitution, right-looking: after x_j is known, eliminate it from x_k, k > j.
    // Threads of one row live in the same half-warp, so __syncwarp suffices between steps.
    for (int j = 0; j < w; ++j) {
        const double xj = X[lr][j] / L[j][j];
        __syncwarp();
        if (lt == 0) X[lr][j] = xj;
        #pragma unroll
        for (int q = 0; q < NB / TRSM_TPR; ++q) {
            const int k = lt + q * TRSM_TPR;
            if (k > j) {
                X[lr][k] -= xj * L[k][j];
            }
        }
        __syncwarp();
    }

    // Store back
    if (rowValid) {
        #pragma unroll
        for (int q = 0; q < NB / TRSM_TPR; ++q) {
            const int c = lt + q * TRSM_TPR;
            if (c < w) A[(size_t)row * n + (k0 + c)] = X[lr][c];
        }
    }
}

// ---------------------------------------------------------------------------------------
// Tiled GEMM with transposed B:
//   C[i][j] = beta * C[i][j] + alpha * sum_{p<K} Aop[i][p] * Bop[j][p]
// for i in [0, m), j in [0, m) (square output), row-major with leading dimension ld.
// Aop/Bop point to the (0,0) element of their panels, each with leading dimension ld.
// If LOWER_ONLY, tiles strictly above the diagonal are skipped (output is symmetric
// and only the lower triangle is needed).
// Each thread accumulates a 4x4 register block; K is traversed in order per output element.
// ---------------------------------------------------------------------------------------
template <bool LOWER_ONLY, bool CHECK_FAIL>
__global__ void __launch_bounds__(GEMM_THREADS) gemm_nt_kernel(double* __restrict__ C,
                                                               const double* __restrict__ Aop,
                                                               const double* __restrict__ Bop,
                                                               const int m, const int K,
                                                               const int ld, const double alpha,
                                                               const double beta) {
    if (CHECK_FAIL && g_fail != NO_FAIL) return;
    if (LOWER_ONLY && blockIdx.y > blockIdx.x) return;  // blockIdx.x = row tile, y = col tile

    __shared__ double As[GEMM_K][GEMM_TILE_PAD];
    __shared__ double Bs[GEMM_K][GEMM_TILE_PAD];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int rowBase = blockIdx.x * GEMM_TILE;
    const int colBase = blockIdx.y * GEMM_TILE;

    double acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0;

    // Loader mapping: thread loads (row = ldr + 16*q, p = ldp) for q = 0..3 -> coalesced along p
    const int ldp = tid % GEMM_K;
    const int ldr = tid / GEMM_K;

    for (int p0 = 0; p0 < K; p0 += GEMM_K) {
        const int p = p0 + ldp;
        #pragma unroll
        for (int q = 0; q < GEMM_TILE / GEMM_K; ++q) {
            const int lr = ldr + q * (GEMM_THREADS / GEMM_K);
            const int gi = rowBase + lr;
            const int gj = colBase + lr;
            As[ldp][lr] = (gi < m && p < K) ? Aop[(size_t)gi * ld + p] : 0.0;
            Bs[ldp][lr] = (gj < m && p < K) ? Bop[(size_t)gj * ld + p] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int pp = 0; pp < GEMM_K; ++pp) {
            double a[4], b[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = As[pp][ty + 16 * r];
            #pragma unroll
            for (int c = 0; c < 4; ++c) b[c] = Bs[pp][tx + 16 * c];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                #pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int gi = rowBase + ty + 16 * r;
        if (gi >= m) continue;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int gj = colBase + tx + 16 * c;
            if (gj >= m) continue;
            if (LOWER_ONLY && gj > gi) continue;
            double* cp = C + (size_t)gi * ld + gj;
            const double old = (beta == 0.0) ? 0.0 : beta * (*cp);
            *cp = old + alpha * acc[r][c];
        }
    }
}

// Add `diag` to every diagonal element
__global__ void add_diagonal_kernel(double* __restrict__ A, const int n, const double diag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[(size_t)i * n + i] += diag;
}

// Zero the strictly upper triangular part
__global__ void zero_upper_kernel(double* __restrict__ A, const int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (i < n && j < n && j > i) A[(size_t)i * n + j] = 0.0;
}

static inline unsigned divUp(size_t a, size_t b) { return (unsigned)((a + b - 1) / b); }

// Cholesky factorization of the device-resident row-major matrix dA (n x n).
// On success the lower triangle holds L and the strict upper triangle is zero.
bool choleskyDecomposition(double* dA, const size_t n) {
    const int ni = (int)n;
    const int failInit = NO_FAIL;
    CUDA_CHECK(cudaMemcpyToSymbol(g_fail, &failInit, sizeof(int)));

    for (int k0 = 0; k0 < ni; k0 += NB) {
        const int w = std::min(NB, ni - k0);
        const int r0 = k0 + w;  // first row of the trailing part
        const int rem = ni - r0;

        // 1. Factor the diagonal tile
        potrf_tile_kernel<<<1, 256>>>(dA, ni, k0, w);

        if (rem <= 0) break;

        // 2. Solve the panel below the diagonal tile
        trsm_panel_kernel<<<divUp(rem, TRSM_ROWS), TRSM_ROWS * TRSM_TPR>>>(dA, ni, k0, w, r0);

        // 3. Trailing update: A22 -= L21 * L21^T (lower triangle)
        const unsigned nt = divUp(rem, GEMM_TILE);
        double* base = dA + (size_t)r0 * n;  // row r0
        gemm_nt_kernel<true, true><<<dim3(nt, nt), GEMM_THREADS>>>(
            base + r0, base + k0, base + k0, rem, w, ni, -1.0, 1.0);
    }

    // Zero the upper triangle (semantics of the reference implementation)
    zero_upper_kernel<<<dim3(divUp(n, 256), (unsigned)n), 256>>>(dA, ni);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int fail = NO_FAIL;
    CUDA_CHECK(cudaMemcpyFromSymbol(&fail, g_fail, sizeof(int)));
    if (fail != NO_FAIL) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)fail);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix directly on the device.
void generatePositiveDefiniteMatrix(double* dA, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential PRNG stream, same as the reference)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Compute A = B * B^T
    const unsigned nt = divUp(n, GEMM_TILE);
    gemm_nt_kernel<false, false><<<dim3(nt, nt), GEMM_THREADS>>>(dA, dB, dB, (int)n, (int)n, (int)n,
                                                                 1.0, 0.0);

    // Add diagonal dominance to ensure positive definiteness
    add_diagonal_kernel<<<divUp(n, 256), 256>>>(dA, (int)n, (double)n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

bool validateCholesky(const double* dL, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T on the device and comparing with original matrix

    double* dR = nullptr;
    CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));

    // Compute L * L^T
    const unsigned nt = divUp(n, GEMM_TILE);
    gemm_nt_kernel<false, false><<<dim3(nt, nt), GEMM_THREADS>>>(dR, dL, dL, (int)n, (int)n, (int)n,
                                                                 1.0, 0.0);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> reconstructed(n * n);
    CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dR));

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

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

    // Allocate matrix (host copy for output, device copy for computation)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    CUDA_CHECK(cudaFree(0));  // initialize the CUDA context up front
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dA, n);

    if (validate) {
        A_orig.resize(n * n);  // Save original for validation
        CUDA_CHECK(cudaMemcpy(A_orig.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(dA, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy the factor back to the host
    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(dA, A_orig, n);

        CUDA_CHECK(cudaFree(dA));
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(dA));
    return 0;
}
