#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA Cholesky decomposition (right-looking blocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// For each panel of NB columns:
//   1. potrf_diag_kernel : factor the NB x NB diagonal block (single thread block)
//   2. trsm_kernel       : L21 = A21 * L11^{-T} for all rows below the panel
//   3. syrk_kernel       : A22 -= L21 * L21^T (lower triangle only, register-tiled)
// Finally the upper triangle is zeroed, matching the reference implementation.

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                       \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

namespace {

constexpr int NB = 64;         // panel / tile width
constexpr int TILE = 64;       // GEMM/SYRK output tile (TILE x TILE)
constexpr int KCHUNK = 16;     // K-dimension chunk staged in shared memory
constexpr int GEMM_THREADS = 256;  // 16 x 16 threads, each computing 4 x 4 outputs
constexpr int TRSM_ROWS = 64;      // rows handled per TRSM thread block
constexpr int TRSM_LANES = 4;      // threads cooperating on one TRSM row

// ---------------------------------------------------------------------------
// Factor the nb x nb diagonal block starting at A[k*n + k] in place.
// One thread block; the tile is staged in shared memory.
// On failure, records the (global) diagonal index via atomicMin into *errIdx.
// ---------------------------------------------------------------------------
__global__ void potrf_diag_kernel(double* __restrict__ A, const size_t n, const int k,
                                  const int nb, int* __restrict__ errIdx) {
    __shared__ double tile[NB][NB + 1];
    __shared__ double diag;
    __shared__ int failed;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    double* Ablk = A + (size_t)k * n + k;

    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int r = idx / nb, c = idx % nb;
        if (c <= r) tile[r][c] = Ablk[(size_t)r * n + c];
    }
    if (tid == 0) failed = 0;
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        // Diagonal element
        if (tid == 0) {
            double sum = 0.0;
            for (int kk = 0; kk < j; ++kk) sum += tile[j][kk] * tile[j][kk];
            const double val = tile[j][j] - sum;
            if (val <= 0.0) {
                failed = 1;
                atomicMin(errIdx, k + j);
                diag = 1.0;  // avoid NaN propagation inside the block; result is discarded
            } else {
                diag = sqrt(val);
            }
            tile[j][j] = diag;
        }
        __syncthreads();
        if (failed) return;

        // Column j below the diagonal
        for (int i = j + 1 + tid; i < nb; i += nthreads) {
            double sum = 0.0;
            for (int kk = 0; kk < j; ++kk) sum += tile[i][kk] * tile[j][kk];
            tile[i][j] = (tile[i][j] - sum) / diag;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int r = idx / nb, c = idx % nb;
        if (c <= r) Ablk[(size_t)r * n + c] = tile[r][c];
    }
}

// ---------------------------------------------------------------------------
// Triangular solve for the panel rows below the diagonal block:
//   L21 = A21 * L11^{-T}, i.e. for each row i:  L21[i][j] = (A21[i][j] - sum_{kk<j} L21[i][kk]*L11[j][kk]) / L11[j][j]
// Each block handles TRSM_ROWS rows; TRSM_LANES threads cooperate per row.
// Dynamic shared memory: L11 (NB x (NB+1)) + row tile (TRSM_ROWS x (NB+1)).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TRSM_ROWS * TRSM_LANES)
trsm_kernel(double* __restrict__ A, const size_t n, const int k, const int nb) {
    extern __shared__ double smem[];
    double (*L11)[NB + 1] = reinterpret_cast<double (*)[NB + 1]>(smem);
    double (*rows)[NB + 1] = reinterpret_cast<double (*)[NB + 1]>(smem + NB * (NB + 1));

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int row0 = k + nb + blockIdx.x * TRSM_ROWS;
    const int nrows = min(TRSM_ROWS, (int)n - row0);

    const double* Ldiag = A + (size_t)k * n + k;
    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int r = idx / nb, c = idx % nb;
        if (c <= r) L11[r][c] = Ldiag[(size_t)r * n + c];
    }
    for (int idx = tid; idx < nrows * nb; idx += nthreads) {
        const int r = idx / nb, c = idx % nb;
        rows[r][c] = A[(size_t)(row0 + r) * n + k + c];
    }
    __syncthreads();

    const int r = tid / TRSM_LANES;
    const int lane = tid % TRSM_LANES;
    const bool active = r < nrows;

    for (int j = 0; j < nb; ++j) {
        double sum = 0.0;
        if (active) {
            for (int kk = lane; kk < j; kk += TRSM_LANES) sum += rows[r][kk] * L11[j][kk];
        }
        // Reduce across the TRSM_LANES consecutive lanes (lanes are within a warp)
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        if (active && lane == 0) rows[r][j] = (rows[r][j] - sum) / L11[j][j];
        __syncthreads();
    }

    for (int idx = tid; idx < nrows * nb; idx += nthreads) {
        const int rr = idx / nb, c = idx % nb;
        A[(size_t)(row0 + rr) * n + k + c] = rows[rr][c];
    }
}

// ---------------------------------------------------------------------------
// Trailing update:  A22 -= L21 * L21^T  for the lower triangle of A22.
// A22 starts at row/col (k + nb); L21 is the nb-wide panel at column k.
// Register-tiled: 64x64 tile, 256 threads, 4x4 outputs each.
// Tiles strictly above the diagonal are skipped by the host-side launch mapping.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(GEMM_THREADS)
syrk_kernel(double* __restrict__ A, const size_t n, const int k, const int nb) {
    __shared__ double sA[KCHUNK][TILE + 1];  // L21 rows for tile-row  (transposed: [kk][i])
    __shared__ double sB[KCHUNK][TILE + 1];  // L21 rows for tile-col  (transposed: [kk][j])

    // Map linear block index to lower-triangular tile (bi >= bj)
    const int bid = blockIdx.x;
    // bi = floor((sqrt(8*bid+1)-1)/2) via integer-safe computation
    int bi = (int)((sqrtf(8.0f * (float)bid + 1.0f) - 1.0f) * 0.5f);
    while (bi * (bi + 1) / 2 > bid) --bi;
    while ((bi + 1) * (bi + 2) / 2 <= bid) ++bi;
    const int bj = bid - bi * (bi + 1) / 2;

    const int off = k + nb;
    const int rowBase = off + bi * TILE;
    const int colBase = off + bj * TILE;

    const int tid = threadIdx.x;
    const int tx = tid % 16;  // column group
    const int ty = tid / 16;  // row group

    double acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0;

    const double* P = A + (size_t)0 * n + k;  // panel base (column k)

    for (int k0 = 0; k0 < nb; k0 += KCHUNK) {
        // Each thread loads 4 elements of sA and 4 of sB.
        // Load layout: 256 threads cover 64 rows x 16 kk -> 4 elements per thread.
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int idx = tid + e * GEMM_THREADS;  // 0..1023
            const int kk = idx % KCHUNK;
            const int r = idx / KCHUNK;  // 0..63
            const int gk = k0 + kk;
            const int gr = rowBase + r;
            const int gc = colBase + r;
            sA[kk][r] = (gk < nb && gr < (int)n) ? P[(size_t)gr * n + gk] : 0.0;
            sB[kk][r] = (gk < nb && gc < (int)n) ? P[(size_t)gc * n + gk] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < KCHUNK; ++kk) {
            double ra[4], rb[4];
#pragma unroll
            for (int a = 0; a < 4; ++a) ra[a] = sA[kk][ty + a * 16];
#pragma unroll
            for (int b = 0; b < 4; ++b) rb[b] = sB[kk][tx + b * 16];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] += ra[a] * rb[b];
        }
        __syncthreads();
    }

    // Write back (lower triangle only)
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int gi = rowBase + ty + a * 16;
        if (gi >= (int)n) continue;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int gj = colBase + tx + b * 16;
            if (gj <= gi && gj < (int)n) {
                A[(size_t)gi * n + gj] -= acc[a][b];
            }
        }
    }
}

// Zero the strictly upper triangular part
__global__ void zero_upper_kernel(double* __restrict__ A, const size_t n) {
    const size_t i = blockIdx.y;
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i) A[i * n + j] = 0.0;
}

// ---------------------------------------------------------------------------
// Full product C = X * X^T (+ alpha on the diagonal). Used for matrix
// generation (A = B*B^T + n*I) and validation (L*L^T).
// Register-tiled 64x64 tile, K = n in chunks of KCHUNK.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(GEMM_THREADS)
xxt_kernel(const double* __restrict__ X, double* __restrict__ C, const size_t n,
           const double diagAdd) {
    __shared__ double sA[KCHUNK][TILE + 1];
    __shared__ double sB[KCHUNK][TILE + 1];

    const int rowBase = blockIdx.y * TILE;
    const int colBase = blockIdx.x * TILE;
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;

    double acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0;

    for (size_t k0 = 0; k0 < n; k0 += KCHUNK) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int idx = tid + e * GEMM_THREADS;
            const int kk = idx % KCHUNK;
            const int r = idx / KCHUNK;
            const size_t gk = k0 + kk;
            const int gr = rowBase + r;
            const int gc = colBase + r;
            sA[kk][r] = (gk < n && gr < (int)n) ? X[(size_t)gr * n + gk] : 0.0;
            sB[kk][r] = (gk < n && gc < (int)n) ? X[(size_t)gc * n + gk] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < KCHUNK; ++kk) {
            double ra[4], rb[4];
#pragma unroll
            for (int a = 0; a < 4; ++a) ra[a] = sA[kk][ty + a * 16];
#pragma unroll
            for (int b = 0; b < 4; ++b) rb[b] = sB[kk][tx + b * 16];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] += ra[a] * rb[b];
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int gi = rowBase + ty + a * 16;
        if (gi >= (int)n) continue;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int gj = colBase + tx + b * 16;
            if (gj < (int)n) {
                C[(size_t)gi * n + gj] = acc[a][b] + (gi == gj ? diagAdd : 0.0);
            }
        }
    }
}

// Host helper: C = X * X^T + diagAdd * I, all on device; copies result to host.
void deviceXXt(const std::vector<double>& X, std::vector<double>& C, const size_t n,
               const double diagAdd) {
    double *dX = nullptr, *dC = nullptr;
    const size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), bytes, cudaMemcpyHostToDevice));
    const unsigned tiles = (unsigned)((n + TILE - 1) / TILE);
    xxt_kernel<<<dim3(tiles, tiles), GEMM_THREADS>>>(dX, dC, n, diagAdd);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dC));
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    const size_t bytes = n * n * sizeof(double);
    double* dA = nullptr;
    int* dErr = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dErr, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    const int noErr = 0x7fffffff;
    CUDA_CHECK(cudaMemcpy(dErr, &noErr, sizeof(int), cudaMemcpyHostToDevice));

    const size_t trsmSmem = (size_t)(NB + TRSM_ROWS) * (NB + 1) * sizeof(double);
    CUDA_CHECK(cudaFuncSetAttribute(trsm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)trsmSmem));

    for (size_t k = 0; k < n; k += NB) {
        const int nb = (int)std::min((size_t)NB, n - k);

        potrf_diag_kernel<<<1, 256>>>(dA, n, (int)k, nb, dErr);

        const size_t rem = n - k - nb;  // rows/cols in the trailing matrix
        if (rem > 0) {
            const unsigned trsmBlocks = (unsigned)((rem + TRSM_ROWS - 1) / TRSM_ROWS);
            trsm_kernel<<<trsmBlocks, TRSM_ROWS * TRSM_LANES, trsmSmem>>>(dA, n, (int)k, nb);

            const unsigned t = (unsigned)((rem + TILE - 1) / TILE);
            const unsigned triBlocks = t * (t + 1) / 2;
            syrk_kernel<<<triBlocks, GEMM_THREADS>>>(dA, n, (int)k, nb);
        }
    }

    {
        const unsigned threads = 256;
        dim3 grid((unsigned)((n + threads - 1) / threads), (unsigned)n);
        zero_upper_kernel<<<grid, threads>>>(dA, n);
    }
    CUDA_CHECK(cudaGetLastError());

    int err = noErr;
    CUDA_CHECK(cudaMemcpy(&err, dErr, sizeof(int), cudaMemcpyDeviceToHost));
    if (err != noErr) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)err);
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dErr));
        return false;
    }

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dErr));
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential RNG to keep the exact same matrix)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T + n * I on the GPU
    deviceXXt(B, A, n, (double)n);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T on the GPU
    deviceXXt(L, reconstructed, n, 0.0);

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

    // Initialize the CUDA context up front so it is not attributed to the timed region
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

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

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
