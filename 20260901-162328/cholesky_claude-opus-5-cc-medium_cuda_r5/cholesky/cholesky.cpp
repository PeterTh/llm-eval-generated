#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Cholesky decomposition (CUDA, blocked right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// The matrix is factorized panel by panel:
//   1) POTF2: factorize the NB x NB diagonal block (single thread block, shared memory)
//   2) TRSM:  L21 = A21 * L11^-T for all rows below the diagonal block
//   3) SYRK:  A22 -= L21 * L21^T (blocked GEMM, only the lower triangle is touched)
// All three steps run on the GPU; the panel loop only enqueues kernels, so no
// host/device synchronization happens inside the factorization.

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                            \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

// Panel width / tile geometry of the trailing update
static constexpr int NB = 64;   // panel width (also diagonal block size)
static constexpr int TS = 64;   // tile of the trailing update handled by one block
static constexpr int TK = 16;   // k-chunk staged in shared memory
static constexpr int TRSM_ROWS = 64;  // rows of the panel handled by one TRSM block

// ---------------------------------------------------------------------------
// Step 1: factorize the nbc x nbc diagonal block in shared memory (one block).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void potf2Kernel(double* __restrict__ A, const int n, const int k0,
                                                   const int nbc, int* __restrict__ info) {
    __shared__ double s[NB][NB + 1];

    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < nbc * nbc; idx += nthreads) {
        const int r = idx / nbc;
        const int c = idx - r * nbc;
        s[r][c] = A[(size_t)(k0 + r) * n + (k0 + c)];
    }
    __syncthreads();

    for (int j = 0; j < nbc; ++j) {
        if (tid == 0) {
            const double val = s[j][j];
            if (!(val > 0.0)) {
                atomicMin(info, k0 + j);
            }
            s[j][j] = sqrt(val);
        }
        __syncthreads();

        const double djj = s[j][j];
        for (int i = j + 1 + tid; i < nbc; i += nthreads) {
            s[i][j] /= djj;
        }
        __syncthreads();

        // Rank-1 update of the trailing lower triangle of the block
        for (int i = j + 1 + (int)threadIdx.y; i < nbc; i += (int)blockDim.y) {
            const double lij = s[i][j];
            for (int c = j + 1 + (int)threadIdx.x; c <= i; c += (int)blockDim.x) {
                s[i][c] -= lij * s[c][j];
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < nbc * nbc; idx += nthreads) {
        const int r = idx / nbc;
        const int c = idx - r * nbc;
        A[(size_t)(k0 + r) * n + (k0 + c)] = s[r][c];
    }
}

// ---------------------------------------------------------------------------
// Step 2: triangular solve for the rows below the diagonal block.
// Each thread owns one row; its NB values live in shared memory (column major
// within the tile) so that global loads/stores stay fully coalesced.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(TRSM_ROWS) void trsmKernel(double* __restrict__ A, const int n, const int k0,
                                                        const int nbc, const int kend) {
    __shared__ double xs[NB][TRSM_ROWS + 1];

    const int tid = threadIdx.x;
    const int row0 = kend + blockIdx.x * TRSM_ROWS;

    for (int r = 0; r < TRSM_ROWS; ++r) {
        const int gr = row0 + r;
        if (gr < n && tid < nbc) {
            xs[tid][r] = A[(size_t)gr * n + k0 + tid];
        }
    }
    __syncthreads();

    const int myRow = row0 + tid;
    if (myRow < n) {
        const double* __restrict__ L11 = A + (size_t)k0 * n + k0;
        for (int j = 0; j < nbc; ++j) {
            const double* __restrict__ lj = L11 + (size_t)j * n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            int c = 0;
            for (; c + 3 < j; c += 4) {
                s0 = fma(xs[c][tid], lj[c], s0);
                s1 = fma(xs[c + 1][tid], lj[c + 1], s1);
                s2 = fma(xs[c + 2][tid], lj[c + 2], s2);
                s3 = fma(xs[c + 3][tid], lj[c + 3], s3);
            }
            for (; c < j; ++c) {
                s0 = fma(xs[c][tid], lj[c], s0);
            }
            const double sum = (s0 + s1) + (s2 + s3);
            xs[j][tid] = (xs[j][tid] - sum) / lj[j];
        }
    }
    __syncthreads();

    for (int r = 0; r < TRSM_ROWS; ++r) {
        const int gr = row0 + r;
        if (gr < n && tid < nbc) {
            A[(size_t)gr * n + k0 + tid] = xs[tid][r];
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3: trailing update A22 -= L21 * L21^T.
// 64x64 output tile per block, 4x4 accumulators per thread (16x16 threads).
// Only tiles in the lower triangle are computed.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void syrkUpdateKernel(double* __restrict__ A, const int n, const int k0,
                                                        const int nbc, const int kend, const int colTileOffset) {
    const int tileCol = blockIdx.x + colTileOffset;
    if (tileCol > (int)blockIdx.y) return;

    __shared__ double As[TK][TS + 1];
    __shared__ double Bs[TK][TS + 1];

    const int row0 = kend + blockIdx.y * TS;
    const int col0 = kend + tileCol * TS;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = tx + 16 * ty;

    double acc[4][4] = {{0.0}};

    for (int kk = 0; kk < nbc; kk += TK) {
        const int kc = k0 + kk;
        const int klen = min(TK, nbc - kk);
#pragma unroll
        for (int l = 0; l < (TS * TK) / 256; ++l) {
            const int idx = tid + l * 256;
            const int r = idx / TK;
            const int c = idx - r * TK;
            const int gr = row0 + r;
            const int gc = col0 + r;
            As[c][r] = (gr < n && c < klen) ? A[(size_t)gr * n + kc + c] : 0.0;
            Bs[c][r] = (gc < n && c < klen) ? A[(size_t)gc * n + kc + c] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < TK; ++t) {
            double a[4], b[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) a[p] = As[t][ty + 16 * p];
#pragma unroll
            for (int q = 0; q < 4; ++q) b[q] = Bs[t][tx + 16 * q];
#pragma unroll
            for (int p = 0; p < 4; ++p)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[p][q] = fma(a[p], b[q], acc[p][q]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int i = row0 + ty + 16 * p;
        if (i >= n) continue;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int j = col0 + tx + 16 * q;
            if (j < n) {
                A[(size_t)i * n + j] -= acc[p][q];
            }
        }
    }
}

// Zero out the upper triangular part of the result
__global__ void zeroUpperKernel(double* __restrict__ A, const int n) {
    const size_t total = (size_t)n * n;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = idx / n;
        const size_t j = idx - i * n;
        if (j > i) A[idx] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// C = B * B^T (full symmetric product, computed on the lower triangle and
// mirrored). Used for matrix generation and for validation.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void symProductKernel(const double* __restrict__ B, double* __restrict__ C,
                                                        const int n) {
    if (blockIdx.x > blockIdx.y) return;

    __shared__ double As[TK][TS + 1];
    __shared__ double Bs[TK][TS + 1];

    const int row0 = blockIdx.y * TS;
    const int col0 = blockIdx.x * TS;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = tx + 16 * ty;

    double acc[4][4] = {{0.0}};

    for (int kk = 0; kk < n; kk += TK) {
        const int klen = min(TK, n - kk);
#pragma unroll
        for (int l = 0; l < (TS * TK) / 256; ++l) {
            const int idx = tid + l * 256;
            const int r = idx / TK;
            const int c = idx - r * TK;
            const int gr = row0 + r;
            const int gc = col0 + r;
            As[c][r] = (gr < n && c < klen) ? B[(size_t)gr * n + kk + c] : 0.0;
            Bs[c][r] = (gc < n && c < klen) ? B[(size_t)gc * n + kk + c] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < TK; ++t) {
            double a[4], b[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) a[p] = As[t][ty + 16 * p];
#pragma unroll
            for (int q = 0; q < 4; ++q) b[q] = Bs[t][tx + 16 * q];
#pragma unroll
            for (int p = 0; p < 4; ++p)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[p][q] = fma(a[p], b[q], acc[p][q]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int i = row0 + ty + 16 * p;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int j = col0 + tx + 16 * q;
            if (i < n && j < n) {
                C[(size_t)i * n + j] = acc[p][q];
                C[(size_t)j * n + i] = acc[p][q];
            }
        }
    }
}

__global__ void addDiagonalKernel(double* __restrict__ A, const int n, const double value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[(size_t)i * n + i] += value;
}

// Per-block reduction of the absolute / relative reconstruction error
__global__ __launch_bounds__(256) void errorReduceKernel(const double* __restrict__ rec,
                                                         const double* __restrict__ orig, const size_t total,
                                                         double* __restrict__ maxAbs, double* __restrict__ maxRel) {
    __shared__ double sAbs[256];
    __shared__ double sRel[256];

    double localAbs = 0.0;
    double localRel = 0.0;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const double error = fabs(rec[idx] - orig[idx]);
        localAbs = fmax(localAbs, error);
        localRel = fmax(localRel, error / (fabs(orig[idx]) + 1e-10));
    }

    sAbs[threadIdx.x] = localAbs;
    sRel[threadIdx.x] = localRel;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            sAbs[threadIdx.x] = fmax(sAbs[threadIdx.x], sAbs[threadIdx.x + stride]);
            sRel[threadIdx.x] = fmax(sRel[threadIdx.x], sRel[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        maxAbs[blockIdx.x] = sAbs[0];
        maxRel[blockIdx.x] = sRel[0];
    }
}

// Small pool of cached device buffers: allocations are reused across the
// generation / factorization / validation phases so that no cudaMalloc (which
// implicitly synchronizes) happens inside the measured region.
static double* deviceBuffer(const int slot, const size_t bytes) {
    static double* ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
    static size_t sizes[4] = {0, 0, 0, 0};

    if (sizes[slot] < bytes) {
        if (ptrs[slot]) CUDA_CHECK(cudaFree(ptrs[slot]));
        CUDA_CHECK(cudaMalloc(&ptrs[slot], bytes));
        sizes[slot] = bytes;
    }
    return ptrs[slot];
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    const int N = (int)n;
    const size_t bytes = n * n * sizeof(double);

    double* const dA = deviceBuffer(0, bytes);
    int* const dInfo = (int*)deviceBuffer(3, sizeof(int));

    // Two streams implement a look-ahead schedule: the factorization of the next
    // panel runs concurrently with the bulk of the current trailing update.
    static cudaStream_t sMain = nullptr;
    static cudaStream_t sPanel = nullptr;
    static cudaEvent_t eColumn = nullptr;
    static cudaEvent_t ePanel = nullptr;
    if (sMain == nullptr) {
        CUDA_CHECK(cudaStreamCreate(&sMain));
        CUDA_CHECK(cudaStreamCreate(&sPanel));
        CUDA_CHECK(cudaEventCreateWithFlags(&eColumn, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ePanel, cudaEventDisableTiming));
    }

    const int infoInit = INT_MAX;
    CUDA_CHECK(cudaMemcpyAsync(dInfo, &infoInit, sizeof(int), cudaMemcpyHostToDevice, sMain));
    CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), bytes, cudaMemcpyHostToDevice, sMain));

    const dim3 blockDims(16, 16);

    // Factorize the first panel; all later panels are factorized ahead of time
    // inside the loop below.
    {
        const int nbc = std::min(NB, N);
        potf2Kernel<<<1, dim3(32, 8), 0, sMain>>>(dA, N, 0, nbc, dInfo);
        if (N - nbc > 0) {
            const int rows = N - nbc;
            trsmKernel<<<(rows + TRSM_ROWS - 1) / TRSM_ROWS, TRSM_ROWS, 0, sMain>>>(dA, N, 0, nbc, nbc);
        }
    }

    for (int k = 0; k < N; k += NB) {
        const int nbc = std::min(NB, N - k);
        const int kend = k + nbc;
        const int m = N - kend;
        if (m <= 0) break;

        const int tiles = (m + TS - 1) / TS;

        // Update the column block that the next panel lives in
        syrkUpdateKernel<<<dim3(1, tiles), blockDims, 0, sMain>>>(dA, N, k, nbc, kend, 0);

        // Factorize the next panel concurrently with the remaining update
        CUDA_CHECK(cudaEventRecord(eColumn, sMain));
        CUDA_CHECK(cudaStreamWaitEvent(sPanel, eColumn, 0));
        const int nbc2 = std::min(NB, N - kend);
        potf2Kernel<<<1, dim3(32, 8), 0, sPanel>>>(dA, N, kend, nbc2, dInfo);
        const int rows2 = N - kend - nbc2;
        if (rows2 > 0) {
            trsmKernel<<<(rows2 + TRSM_ROWS - 1) / TRSM_ROWS, TRSM_ROWS, 0, sPanel>>>(dA, N, kend, nbc2,
                                                                                      kend + nbc2);
        }
        CUDA_CHECK(cudaEventRecord(ePanel, sPanel));

        // Remaining trailing update (disjoint from the columns of the next panel)
        if (tiles > 1) {
            syrkUpdateKernel<<<dim3(tiles - 1, tiles), blockDims, 0, sMain>>>(dA, N, k, nbc, kend, 1);
        }

        CUDA_CHECK(cudaStreamWaitEvent(sMain, ePanel, 0));
    }

    zeroUpperKernel<<<1024, 256, 0, sMain>>>(dA, N);

    int info = INT_MAX;
    CUDA_CHECK(cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, sMain));
    CUDA_CHECK(cudaMemcpyAsync(A.data(), dA, bytes, cudaMemcpyDeviceToHost, sMain));
    CUDA_CHECK(cudaStreamSynchronize(sMain));
    CUDA_CHECK(cudaGetLastError());

    if (info != INT_MAX) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info);
        return false;
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    if (n == 0) return;

    const int N = (int)n;
    const size_t bytes = n * n * sizeof(double);

    double* const dB = deviceBuffer(1, bytes);
    double* const dA = deviceBuffer(0, bytes);

    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Compute A = B * B^T
    const int tiles = (N + TS - 1) / TS;
    symProductKernel<<<dim3(tiles, tiles), dim3(16, 16)>>>(dB, dA, N);

    // Add diagonal dominance to ensure positive definiteness
    addDiagonalKernel<<<(N + 255) / 256, 256>>>(dA, N, (double)n);

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    double maxError = 0.0;
    double relError = 0.0;

    if (n > 0) {
        const int N = (int)n;
        const size_t bytes = n * n * sizeof(double);

        double* const dL = deviceBuffer(0, bytes);
        double* const dOrig = deviceBuffer(1, bytes);
        double* const dRec = deviceBuffer(2, bytes);

        CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dOrig, A_orig.data(), bytes, cudaMemcpyHostToDevice));

        // Compute L * L^T
        const int tiles = (N + TS - 1) / TS;
        symProductKernel<<<dim3(tiles, tiles), dim3(16, 16)>>>(dL, dRec, N);

        // Compare with original
        const int nblocks = 1024;
        double* dAbs = nullptr;
        double* dRel = nullptr;
        CUDA_CHECK(cudaMalloc(&dAbs, nblocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dRel, nblocks * sizeof(double)));
        errorReduceKernel<<<nblocks, 256>>>(dRec, dOrig, n * n, dAbs, dRel);

        std::vector<double> hAbs(nblocks), hRel(nblocks);
        CUDA_CHECK(cudaMemcpy(hAbs.data(), dAbs, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hRel.data(), dRel, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        for (int i = 0; i < nblocks; ++i) {
            maxError = std::max(maxError, hAbs[i]);
            relError = std::max(relError, hRel[i]);
        }

        CUDA_CHECK(cudaFree(dAbs));
        CUDA_CHECK(cudaFree(dRel));
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

    // Initialize the CUDA context up front so that it is not attributed to the
    // measured decomposition time
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
