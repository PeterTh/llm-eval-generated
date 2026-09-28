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

// Cholesky decomposition (CUDA, right-looking blocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err_ = (call);                                                                      \
        if (err_ != cudaSuccess) {                                                                            \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__,                \
                   cudaGetErrorString(err_));                                                                 \
            exit(1);                                                                                          \
        }                                                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Tiled matrix multiply: C = alpha * A * B^T (+ C)
//
// A is M x K, B is N x K, C is M x N, all row-major. Both operands are read
// along their rows, which is exactly the access pattern of the dot products in
// the sequential code (A[i*n+k] * B[j*n+k]).
//
// The k dimension is accumulated in strictly increasing order into a single
// register per output element, so for alpha == 1 / beta == 0 and unfused
// multiply-add this reproduces the sequential summation order bit for bit.
// ---------------------------------------------------------------------------

#define BM 64  // rows of C per block
#define BK 16  // k-slice per iteration
#define TM 4   // rows of C per thread
#define GEMM_THREADS 256

// If skipUpper is true, blocks that lie entirely above the diagonal of the
// enclosing symmetric matrix are dropped (used for the trailing update, where
// only the lower part matters). diagOffset is the column index of C that
// corresponds to its first row, i.e. globalCol - globalRow of C[0][0].
// If fused is false, multiply and add are kept separate, matching the rounding
// of the scalar reference loop exactly; the factorization uses the (twice as
// fast) fused form.
// BNc/TNc give the column extent of the tile; a narrow variant is used for the
// skinny look-ahead update.
template <bool skipUpper, bool fused, int BNc, int TNc>
__global__ __launch_bounds__(GEMM_THREADS) void gemmNTKernel(const int M, const int N, const int K,
                                                             const double alpha, const double* __restrict__ A,
                                                             const int lda, const double* __restrict__ B,
                                                             const int ldb, double* __restrict__ C, const int ldc,
                                                             const bool betaZero, const int diagOffset) {
    static_assert((BM / TM) * (BNc / TNc) == GEMM_THREADS, "tile shape must match the block size");

    if (skipUpper && (int)(blockIdx.y * BM + BM - 1) < diagOffset + (int)(blockIdx.x * BNc)) {
        return;
    }

    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BNc + 1];

    const int tid = threadIdx.x;
    const int ty = tid / (BNc / TNc);  // row group of this thread
    const int tx = tid % (BNc / TNc);  // col group of this thread

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BNc;

    // Cooperative loads: every thread reads EA (resp. EB) contiguous k-values
    // of one row, so that neighbouring threads read neighbouring addresses.
    constexpr int EA = BM * BK / GEMM_THREADS;
    constexpr int EB = BNc * BK / GEMM_THREADS;
    const int loadRowA = tid / (BK / EA);
    const int loadKA = (tid % (BK / EA)) * EA;
    const int loadRowB = tid / (BK / EB);
    const int loadKB = (tid % (BK / EB)) * EB;

    double acc[TM][TNc];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TNc; ++j) {
            acc[i][j] = 0.0;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int aRow = rowBase + loadRowA;
        const int bRow = colBase + loadRowB;
#pragma unroll
        for (int e = 0; e < EA; ++e) {
            const int kk = loadKA + e;
            As[kk][loadRowA] = (aRow < M && k0 + kk < K) ? A[(size_t)aRow * lda + k0 + kk] : 0.0;
        }
#pragma unroll
        for (int e = 0; e < EB; ++e) {
            const int kk = loadKB + e;
            Bs[kk][loadRowB] = (bRow < N && k0 + kk < K) ? B[(size_t)bRow * ldb + k0 + kk] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM];
            double b[TNc];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][ty * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TNc; ++j) {
                b[j] = Bs[kk][tx * TNc + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TNc; ++j) {
                    acc[i][j] = fused ? fma(a[i], b[j], acc[i][j])
                                      : __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = rowBase + ty * TM + i;
        if (row >= M) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < TNc; ++j) {
            const int col = colBase + tx * TNc + j;
            if (col < N) {
                double* c = &C[(size_t)row * ldc + col];
                *c = betaZero ? alpha * acc[i][j] : *c + alpha * acc[i][j];
            }
        }
    }
}

// Trailing update of the factorization: C -= A * B^T, fused mul-add.
// The lower triangle of the symmetric trailing matrix is enough, so blocks
// fully above its diagonal (at column offset diagOffset) are skipped.
static void launchTrailingUpdate(const int M, const int N, const int K, const double* A, const int lda,
                                 const double* B, const int ldb, double* C, const int ldc, const int diagOffset,
                                 cudaStream_t stream) {
    if (M <= 0 || N <= 0) {
        return;
    }
    if (N <= 32) {
        // Skinny look-ahead update: half-width tiles keep all threads busy
        const dim3 grid(1, (M + BM - 1) / BM);
        gemmNTKernel<true, true, 32, 2><<<grid, GEMM_THREADS, 0, stream>>>(M, N, K, -1.0, A, lda, B, ldb, C, ldc,
                                                                           false, diagOffset);
    } else {
        const dim3 grid((N + 63) / 64, (M + BM - 1) / BM);
        gemmNTKernel<true, true, 64, 4><<<grid, GEMM_THREADS, 0, stream>>>(M, N, K, -1.0, A, lda, B, ldb, C, ldc,
                                                                           false, diagOffset);
    }
}

// C = A * B^T with the exact rounding of the scalar reference loop.
static void launchGemmNTExact(const int M, const int N, const int K, const double* A, const int lda, const double* B,
                              const int ldb, double* C, const int ldc) {
    if (M <= 0 || N <= 0) {
        return;
    }
    const dim3 grid((N + 63) / 64, (M + BM - 1) / BM);
    gemmNTKernel<false, false, 64, 4><<<grid, GEMM_THREADS>>>(M, N, K, 1.0, A, lda, B, ldb, C, ldc, true, 0);
}

// ---------------------------------------------------------------------------
// Blocked Cholesky building blocks
// ---------------------------------------------------------------------------

#define NB 48          // block size of the factorization
#define TRSM_ROWS 32   // panel rows handled per thread block

// Unblocked Cholesky of the nb x nb diagonal block, in shared memory.
// Executed by a single block; nb <= NB.
__global__ __launch_bounds__(NB) void potrfDiagKernel(double* __restrict__ A, const int n, const int k, const int nb,
                                                      int* __restrict__ info) {
    __shared__ double s[NB][NB + 1];
    __shared__ double diag[NB];

    const int t = threadIdx.x;

    if (t < nb) {
        for (int j = 0; j < nb; ++j) {
            s[t][j] = A[(size_t)(k + t) * n + k + j];
        }
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        // All dot products of column j at once: thread j produces the diagonal
        // sum, threads below it the off-diagonal ones. Keeping p sequential
        // preserves the summation order of the scalar algorithm.
        double sum = 0.0;
        if (t >= j && t < nb) {
            for (int p = 0; p < j; ++p) {
                sum = __dadd_rn(sum, __dmul_rn(s[t][p], s[j][p]));
            }
        }
        __syncthreads();

        if (t == j) {
            const double val = s[j][j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite; remember the first such column
                atomicMin(info, k + j);
            }
            diag[j] = sqrt(val);
            s[j][j] = diag[j];
        }
        __syncthreads();

        if (t > j && t < nb) {
            s[t][j] = (s[t][j] - sum) / diag[j];
        }
        __syncthreads();
    }

    if (t < nb) {
        for (int j = 0; j <= t; ++j) {
            A[(size_t)(k + t) * n + k + j] = s[t][j];
        }
    }
}

// Panel solve: for the rows below the diagonal block, X * Lkk^T = B, i.e. the
// off-diagonal update of the sequential algorithm applied one row per thread.
__global__ __launch_bounds__(TRSM_ROWS) void trsmPanelKernel(double* __restrict__ A, const int n, const int k,
                                                             const int nb, const int rows) {
    __shared__ double sL[NB][NB + 1];
    __shared__ double sX[TRSM_ROWS][NB + 1];

    const int tid = threadIdx.x;
    const int r0 = k + nb + blockIdx.x * TRSM_ROWS;
    const int myRows = min(TRSM_ROWS, rows - blockIdx.x * TRSM_ROWS);

    // Diagonal block (lower triangle) into shared memory
    for (int idx = tid; idx < nb * nb; idx += TRSM_ROWS) {
        const int r = idx / nb;
        const int c = idx % nb;
        sL[r][c] = A[(size_t)(k + r) * n + k + c];
    }
    // Panel tile, coalesced along rows
    for (int idx = tid; idx < myRows * nb; idx += TRSM_ROWS) {
        const int r = idx / nb;
        const int c = idx % nb;
        sX[r][c] = A[(size_t)(r0 + r) * n + k + c];
    }
    __syncthreads();

    if (tid < myRows) {
        for (int j = 0; j < nb; ++j) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                sum = __dadd_rn(sum, __dmul_rn(sX[tid][p], sL[j][p]));
            }
            sX[tid][j] = (sX[tid][j] - sum) / sL[j][j];
        }
    }
    __syncthreads();

    for (int idx = tid; idx < myRows * nb; idx += TRSM_ROWS) {
        const int r = idx / nb;
        const int c = idx % nb;
        A[(size_t)(r0 + r) * n + k + c] = sX[r][c];
    }
}

__global__ void zeroUpperKernel(double* __restrict__ A, const int n) {
    const int row = blockIdx.y;
    const int stride = gridDim.x * blockDim.x;
    for (int col = row + 1 + blockIdx.x * blockDim.x + threadIdx.x; col < n; col += stride) {
        A[(size_t)row * n + col] = 0.0;
    }
}

__global__ void addDiagonalKernel(double* __restrict__ A, const int n, const double value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[(size_t)i * n + i] += value;
    }
}

// Element-wise error reduction for validation
__global__ void errorKernel(const double* __restrict__ recon, const double* __restrict__ orig, const size_t count,
                            double* __restrict__ maxOut, double* __restrict__ relOut) {
    __shared__ double sMax[256];
    __shared__ double sRel[256];

    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const double error = fabs(recon[i] - orig[i]);
        maxError = fmax(maxError, error);
        relError = fmax(relError, error / (fabs(orig[i]) + 1e-10));
    }
    sMax[threadIdx.x] = maxError;
    sRel[threadIdx.x] = relError;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sMax[threadIdx.x] = fmax(sMax[threadIdx.x], sMax[threadIdx.x + s]);
            sRel[threadIdx.x] = fmax(sRel[threadIdx.x], sRel[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maxOut[blockIdx.x] = sMax[0];
        relOut[blockIdx.x] = sRel[0];
    }
}

// ---------------------------------------------------------------------------

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const int ni = (int)n;
    const size_t bytes = n * n * sizeof(double);

    double* dA = nullptr;
    int* dInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));

    const int infoInit = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dInfo, &infoInit, sizeof(int), cudaMemcpyHostToDevice));

    // Only the lower triangle is read by the factorization, so upload it in
    // block rows and skip the (roughly half of the) upper part entirely.
    const int uploadRows = 512;
    for (int r0 = 0; r0 < ni; r0 += uploadRows) {
        const int rows = min(uploadRows, ni - r0);
        const int cols = min(ni, r0 + rows);
        CUDA_CHECK(cudaMemcpy2D(dA + (size_t)r0 * ni, ni * sizeof(double), A.data() + (size_t)r0 * ni,
                                ni * sizeof(double), cols * sizeof(double), rows, cudaMemcpyHostToDevice));
    }

    // Two streams implement a one-step look-ahead: the panel stream factors the
    // next block column while the update stream is still busy with the (much
    // larger) rest of the trailing submatrix of the current step.
    cudaStream_t panelStream, updateStream;
    CUDA_CHECK(cudaStreamCreate(&panelStream));
    CUDA_CHECK(cudaStreamCreate(&updateStream));
    cudaEvent_t evPanel[2], evUpdate[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evPanel[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evUpdate[i], cudaEventDisableTiming));
    }

    // The upper triangle is not part of the input (and ends up zero anyway)
    const int zeroBlocks = min(64, (ni + 255) / 256);
    if (ni > 1) {
        zeroUpperKernel<<<dim3(zeroBlocks, ni - 1), 256, 0, panelStream>>>(dA, ni);
    }

    auto factorPanel = [&](const int k, const int nb) {
        potrfDiagKernel<<<1, NB, 0, panelStream>>>(dA, ni, k, nb, dInfo);
        const int rows = ni - k - nb;
        if (rows > 0) {
            trsmPanelKernel<<<(rows + TRSM_ROWS - 1) / TRSM_ROWS, TRSM_ROWS, 0, panelStream>>>(dA, ni, k, nb, rows);
        }
    };

    factorPanel(0, min(NB, ni));
    CUDA_CHECK(cudaEventRecord(evPanel[0], panelStream));

    for (int k = 0, step = 0; k < ni; k += NB, ++step) {
        const int nb = min(NB, ni - k);
        const int rows = ni - k - nb;  // rows below the diagonal block
        if (rows == 0) {
            break;
        }
        const int nbNext = min(NB, rows);       // width of the next block column
        const int restCols = rows - nbNext;     // trailing columns beyond it
        const double* panel = dA + (size_t)(k + nb) * ni + k;

        // The look-ahead update writes into the region the previous step's
        // trailing update produced.
        if (step > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(panelStream, evUpdate[(step - 1) & 1], 0));
        }
        // Bring the next block column up to date, then factor it
        launchTrailingUpdate(rows, nbNext, nb, panel, ni, panel, ni, dA + (size_t)(k + nb) * ni + (k + nb), ni, 0,
                             panelStream);
        factorPanel(k + nb, nbNext);
        CUDA_CHECK(cudaEventRecord(evPanel[(step + 1) & 1], panelStream));

        // Rest of the trailing update, concurrently; it needs this step's panel
        if (restCols > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(updateStream, evPanel[step & 1], 0));
            launchTrailingUpdate(rows, restCols, nb, panel, ni, panel + (size_t)nbNext * ni, ni,
                                 dA + (size_t)(k + nb) * ni + (k + nb + nbNext), ni, nbNext, updateStream);
        }
        CUDA_CHECK(cudaEventRecord(evUpdate[step & 1], updateStream));
    }

    // Zero out upper triangular part
    if (ni > 1) {
        CUDA_CHECK(cudaStreamWaitEvent(panelStream, evUpdate[0], 0));
        CUDA_CHECK(cudaStreamWaitEvent(panelStream, evUpdate[1], 0));
        zeroUpperKernel<<<dim3(zeroBlocks, ni - 1), 256, 0, panelStream>>>(dA, ni);
    }

    CUDA_CHECK(cudaStreamSynchronize(panelStream));
    CUDA_CHECK(cudaStreamSynchronize(updateStream));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventDestroy(evPanel[i]));
        CUDA_CHECK(cudaEventDestroy(evUpdate[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(panelStream));
    CUDA_CHECK(cudaStreamDestroy(updateStream));

    int info = INT_MAX;
    CUDA_CHECK(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dInfo));

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

    if (n == 0) {
        return;
    }

    const int ni = (int)n;
    const size_t bytes = n * n * sizeof(double);

    double* dB = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Compute A = B * B^T
    launchGemmNTExact(ni, ni, ni, dB, ni, dB, ni, dA, ni);

    // Add diagonal dominance to ensure positive definiteness
    addDiagonalKernel<<<(ni + 255) / 256, 256>>>(dA, ni, (double)n);

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    double maxError = 0.0;
    double relError = 0.0;

    if (n > 0) {
        const int ni = (int)n;
        const size_t bytes = n * n * sizeof(double);
        const int redBlocks = 256;

        double* dL = nullptr;
        double* dOrig = nullptr;
        double* dRecon = nullptr;
        double* dMax = nullptr;
        double* dRel = nullptr;
        CUDA_CHECK(cudaMalloc(&dL, bytes));
        CUDA_CHECK(cudaMalloc(&dOrig, bytes));
        CUDA_CHECK(cudaMalloc(&dRecon, bytes));
        CUDA_CHECK(cudaMalloc(&dMax, redBlocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dRel, redBlocks * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dOrig, A_orig.data(), bytes, cudaMemcpyHostToDevice));

        // Compute L * L^T
        launchGemmNTExact(ni, ni, ni, dL, ni, dL, ni, dRecon, ni);

        // Compare with original
        errorKernel<<<redBlocks, 256>>>(dRecon, dOrig, n * n, dMax, dRel);

        std::vector<double> hMax(redBlocks);
        std::vector<double> hRel(redBlocks);
        CUDA_CHECK(cudaMemcpy(hMax.data(), dMax, redBlocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hRel.data(), dRel, redBlocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        for (int i = 0; i < redBlocks; ++i) {
            maxError = std::max(maxError, hMax[i]);
            relError = std::max(relError, hRel[i]);
        }

        CUDA_CHECK(cudaFree(dL));
        CUDA_CHECK(cudaFree(dOrig));
        CUDA_CHECK(cudaFree(dRecon));
        CUDA_CHECK(cudaFree(dMax));
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

    // Initialize the CUDA context up front so that it is not part of any timing
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
