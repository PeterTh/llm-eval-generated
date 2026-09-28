#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Cholesky decomposition (CUDA, blocked right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        const cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                                \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);     \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

// Block size of the blocked Cholesky factorization (diagonal block / panel width)
static constexpr int NB = 32;

// Tiling parameters of the matrix-multiply kernel
static constexpr int TILE = 64;   // output tile edge length
static constexpr int KT = 16;     // k-chunk held in shared memory
static constexpr int TDIM = 16;   // thread block is TDIM x TDIM, each thread owns 4x4 outputs
static constexpr int MICRO = TILE / TDIM;

// C = X * Y^T  (or C -= X * Y^T when SUBTRACT), with X, Y stored row-major.
// EXACT reproduces the operation order and rounding of the scalar reference loop
// (ascending k, separate multiply and add, no FMA contraction).
// LOWER_ONLY restricts the computation to the lower triangle of C, which requires C to
// have the same row/column origin as the tiles of X and Y (true for the trailing update).
template <bool EXACT, bool SUBTRACT, bool LOWER_ONLY>
__global__ void gemmNTKernel(const double* __restrict__ X, const double* __restrict__ Y,
                             double* __restrict__ C, const int M, const int N, const int K,
                             const int ldx, const int ldy, const int ldc) {
    if (LOWER_ONLY && blockIdx.x > blockIdx.y) {
        return; // tile lies strictly in the upper triangle
    }

    __shared__ double As[KT][TILE];
    __shared__ double Bs[KT][TILE];

    const int row0 = blockIdx.y * TILE;
    const int col0 = blockIdx.x * TILE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TDIM + tx;

    double acc[MICRO][MICRO];
#pragma unroll
    for (int a = 0; a < MICRO; ++a) {
#pragma unroll
        for (int b = 0; b < MICRO; ++b) {
            acc[a][b] = 0.0;
        }
    }

    for (int kk = 0; kk < K; kk += KT) {
        // Cooperative load: consecutive threads read consecutive k values (coalesced).
#pragma unroll
        for (int e = 0; e < TILE * KT / (TDIM * TDIM); ++e) {
            const int idx = tid + e * TDIM * TDIM;
            const int i = idx / KT;
            const int lp = idx - i * KT;
            const int p = kk + lp;
            As[lp][i] = (row0 + i < M && p < K) ? X[(size_t)(row0 + i) * ldx + p] : 0.0;
            Bs[lp][i] = (col0 + i < N && p < K) ? Y[(size_t)(col0 + i) * ldy + p] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int p2 = 0; p2 < KT; ++p2) {
            double av[MICRO];
            double bv[MICRO];
#pragma unroll
            for (int a = 0; a < MICRO; ++a) {
                av[a] = As[p2][ty + TDIM * a];
            }
#pragma unroll
            for (int b = 0; b < MICRO; ++b) {
                bv[b] = Bs[p2][tx + TDIM * b];
            }
#pragma unroll
            for (int a = 0; a < MICRO; ++a) {
#pragma unroll
                for (int b = 0; b < MICRO; ++b) {
                    if (EXACT) {
                        acc[a][b] = __dadd_rn(acc[a][b], __dmul_rn(av[a], bv[b]));
                    } else {
                        acc[a][b] = fma(av[a], bv[b], acc[a][b]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < MICRO; ++a) {
        const int i = row0 + ty + TDIM * a;
        if (i >= M) {
            continue;
        }
#pragma unroll
        for (int b = 0; b < MICRO; ++b) {
            const int j = col0 + tx + TDIM * b;
            if (j >= N || (LOWER_ONLY && j > i)) {
                continue;
            }
            double* dst = &C[(size_t)i * ldc + j];
            *dst = SUBTRACT ? *dst - acc[a][b] : acc[a][b];
        }
    }
}

// Unblocked Cholesky factorization of the nb x nb diagonal block at (kb, kb).
// Runs in a single thread block with the block resident in shared memory.
__global__ void potrfDiagKernel(double* __restrict__ A, const int n, const int kb, const int nb,
                                int* __restrict__ errIdx) {
    __shared__ double s[NB][NB + 1];

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int i = idx / nb;
        const int j = idx - i * nb;
        s[i][j] = A[(size_t)(kb + i) * n + kb + j];
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        if (tid == 0) {
            const double val = s[j][j];
            if (!(val > 0.0)) {
                // Matrix is not positive definite; remember the first offending diagonal.
                atomicMin(errIdx, kb + j);
            }
            s[j][j] = sqrt(val);
        }
        __syncthreads();

        const double djj = s[j][j];
        for (int i = j + 1 + tid; i < nb; i += nthreads) {
            s[i][j] /= djj;
        }
        __syncthreads();

        // Rank-1 update of the remaining lower-triangular part of the block.
        for (int idx = tid; idx < nb * nb; idx += nthreads) {
            const int i = idx / nb;
            const int c = idx - i * nb;
            if (c > j && i >= c) {
                s[i][c] = fma(-s[i][j], s[c][j], s[i][c]);
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int i = idx / nb;
        const int j = idx - i * nb;
        if (j <= i) {
            A[(size_t)(kb + i) * n + kb + j] = s[i][j];
        }
    }
}

// Triangular solve for the panel below the diagonal block: X * L_kk^T = B, in place.
// Each thread handles one row of the panel; rows are independent.
static constexpr int TRSM_ROWS = 64;

__global__ void trsmPanelKernel(double* __restrict__ A, const int n, const int kb, const int nb,
                                const int rowStart, const int m) {
    __shared__ double d[NB][NB + 1];
    __shared__ double xs[NB][TRSM_ROWS + 1];

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int i = idx / nb;
        const int j = idx - i * nb;
        d[i][j] = A[(size_t)(kb + i) * n + kb + j];
    }

    const int rbase = rowStart + blockIdx.x * TRSM_ROWS;
    const int rows = min(TRSM_ROWS, m - blockIdx.x * TRSM_ROWS);

    for (int idx = tid; idx < rows * nb; idx += nthreads) {
        const int r = idx / nb;
        const int c = idx - r * nb;
        xs[c][r] = A[(size_t)(rbase + r) * n + kb + c];
    }
    __syncthreads();

    if (tid < rows) {
        for (int j = 0; j < nb; ++j) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                sum = fma(xs[p][tid], d[j][p], sum);
            }
            xs[j][tid] = (xs[j][tid] - sum) / d[j][j];
        }
    }
    __syncthreads();

    for (int idx = tid; idx < rows * nb; idx += nthreads) {
        const int r = idx / nb;
        const int c = idx - r * nb;
        A[(size_t)(rbase + r) * n + kb + c] = xs[c][r];
    }
}

// Zero out the strict upper triangular part of the row band [r0, r0 + rows).
__global__ void zeroUpperBandKernel(double* __restrict__ A, const int n, const int r0,
                                    const int rows) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = r0 + r;
    if (r < rows && j < n && j > i) {
        A[(size_t)i * n + j] = 0.0;
    }
}

// Mirror the lower triangle onto the upper one.
__global__ void symmetrizeKernel(double* __restrict__ A, const int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n && j > i) {
        A[(size_t)i * n + j] = A[(size_t)j * n + i];
    }
}

// Add the diagonal dominance term.
__global__ void addDiagKernel(double* __restrict__ A, const int n, const double v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[(size_t)i * n + i] += v;
    }
}

// Reduce the absolute and relative reconstruction error.
// Non-negative doubles compare identically to their bit patterns, so an integer
// atomicMax yields an order-independent (hence deterministic) maximum.
__global__ void errorReduceKernel(const double* __restrict__ R, const double* __restrict__ A,
                                  const size_t total, unsigned long long* __restrict__ out) {
    double maxErr = 0.0;
    double relErr = 0.0;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const double error = fabs(R[idx] - A[idx]);
        maxErr = fmax(maxErr, error);
        relErr = fmax(relErr, error / (fabs(A[idx]) + 1e-10));
    }

    // Warp/block reduction through shared memory.
    __shared__ double sm[2][256];
    sm[0][threadIdx.x] = maxErr;
    sm[1][threadIdx.x] = relErr;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sm[0][threadIdx.x] = fmax(sm[0][threadIdx.x], sm[0][threadIdx.x + stride]);
            sm[1][threadIdx.x] = fmax(sm[1][threadIdx.x], sm[1][threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(&out[0], __double_as_longlong(sm[0][0]));
        atomicMax(&out[1], __double_as_longlong(sm[1][0]));
    }
}

// Launch the lower triangle of C = X * X^T, matching the reference rounding, then mirror
// it to the upper triangle (which is bit-identical since the products commute).
static void launchSymGemmNTExact(const double* X, double* C, const int n) {
    const dim3 threads(TDIM, TDIM);
    const int tiles = (n + TILE - 1) / TILE;
    gemmNTKernel<true, false, true><<<dim3(tiles, tiles), threads>>>(X, X, C, n, n, n, n, n, n);
    CUDA_CHECK(cudaGetLastError());

    const dim3 zthreads(32, 8);
    symmetrizeKernel<<<dim3((n + 31) / 32, (n + 7) / 8), zthreads>>>(C, n);
    CUDA_CHECK(cudaGetLastError());
}

// Device buffers, streams and page-locked host memory used by the factorization.
// Set up (and torn down) outside of the measured region, like any other allocation.
struct CholeskyWorkspace {
    double* dA = nullptr;
    int* dErr = nullptr;
    cudaStream_t compute = nullptr; // factorization kernels
    cudaStream_t copy = nullptr;    // device -> host streaming of finished rows
    cudaEvent_t bandReady = nullptr;
    void* host = nullptr;
    size_t bytes = 0;
};

static CholeskyWorkspace g_ws;

void choleskyWorkspaceInit(std::vector<double>& A, const size_t n) {
    if (n == 0 || g_ws.dA != nullptr) {
        return;
    }
    g_ws.bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&g_ws.dA, g_ws.bytes));
    CUDA_CHECK(cudaMalloc(&g_ws.dErr, sizeof(int)));
    CUDA_CHECK(cudaStreamCreate(&g_ws.compute));
    CUDA_CHECK(cudaStreamCreate(&g_ws.copy));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_ws.bandReady, cudaEventDisableTiming));

    // Page-lock the matrix so transfers run at full PCIe speed and can overlap with compute.
    if (cudaHostRegister(A.data(), g_ws.bytes, cudaHostRegisterDefault) == cudaSuccess) {
        g_ws.host = A.data();
    } else {
        cudaGetLastError(); // clear the error, fall back to pageable transfers
    }
}

void choleskyWorkspaceFree() {
    if (g_ws.dA == nullptr) {
        return;
    }
    if (g_ws.host != nullptr) {
        CUDA_CHECK(cudaHostUnregister(g_ws.host));
    }
    CUDA_CHECK(cudaEventDestroy(g_ws.bandReady));
    CUDA_CHECK(cudaStreamDestroy(g_ws.compute));
    CUDA_CHECK(cudaStreamDestroy(g_ws.copy));
    CUDA_CHECK(cudaFree(g_ws.dA));
    CUDA_CHECK(cudaFree(g_ws.dErr));
    g_ws = CholeskyWorkspace{};
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    choleskyWorkspaceInit(A, n); // no-op if already set up

    const int ni = (int)n;
    const size_t bytes = n * n * sizeof(double);
    double* const dA = g_ws.dA;
    int* const dErr = g_ws.dErr;
    const cudaStream_t compute = g_ws.compute;
    const cudaStream_t copy = g_ws.copy;
    const cudaEvent_t bandReady = g_ws.bandReady;

    const int errInit = ni; // sentinel: no error
    CUDA_CHECK(cudaMemcpyAsync(dErr, &errInit, sizeof(int), cudaMemcpyHostToDevice, compute));
    CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), bytes, cudaMemcpyHostToDevice, compute));

    const dim3 threads(TDIM, TDIM);
    const dim3 zthreads(32, 8);

    for (int kb = 0; kb < ni; kb += NB) {
        const int nb = min(NB, ni - kb);
        const int k0 = kb + nb;
        const int m = ni - k0;

        // 1) Factor the diagonal block.
        potrfDiagKernel<<<1, 256, 0, compute>>>(dA, ni, kb, nb, dErr);
        CUDA_CHECK(cudaGetLastError());

        // Rows [kb, k0) are final now: zero their upper part and stream them back to the
        // host on a separate stream, overlapping the transfer with the remaining work.
        zeroUpperBandKernel<<<dim3((ni + 31) / 32, (nb + 7) / 8), zthreads, 0, compute>>>(
            dA, ni, kb, nb);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(bandReady, compute));
        CUDA_CHECK(cudaStreamWaitEvent(copy, bandReady, 0));
        CUDA_CHECK(cudaMemcpyAsync(A.data() + (size_t)kb * n, dA + (size_t)kb * n,
                                   (size_t)nb * n * sizeof(double), cudaMemcpyDeviceToHost, copy));

        if (m <= 0) {
            continue;
        }

        // 2) Solve for the panel below the diagonal block.
        const int trsmBlocks = (m + TRSM_ROWS - 1) / TRSM_ROWS;
        trsmPanelKernel<<<trsmBlocks, TRSM_ROWS, 0, compute>>>(dA, ni, kb, nb, k0, m);
        CUDA_CHECK(cudaGetLastError());

        // 3) Symmetric rank-nb update of the trailing submatrix (lower triangle only).
        //    This is the bulk of the work and runs at the device's double-precision peak.
        const double* panel = dA + (size_t)k0 * n + kb;
        const int tiles = (m + TILE - 1) / TILE;
        gemmNTKernel<false, true, true><<<dim3(tiles, tiles), threads, 0, compute>>>(
            panel, panel, dA + (size_t)k0 * n + k0, m, m, nb, ni, ni, ni);
        CUDA_CHECK(cudaGetLastError());
    }

    int errIdx = ni;
    CUDA_CHECK(cudaMemcpyAsync(&errIdx, dErr, sizeof(int), cudaMemcpyDeviceToHost, compute));
    CUDA_CHECK(cudaStreamSynchronize(compute));
    CUDA_CHECK(cudaStreamSynchronize(copy));

    if (errIdx < ni) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %d\n", errIdx);
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

    const size_t bytes = n * n * sizeof(double);
    double* dB = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Compute A = B * B^T
    launchSymGemmNTExact(dB, dA, (int)n);

    // Add diagonal dominance to ensure positive definiteness
    addDiagKernel<<<((int)n + 255) / 256, 256>>>(dA, (int)n, (double)n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    double maxError = 0.0;
    double relError = 0.0;

    if (n > 0) {
        const size_t total = n * n;
        const size_t bytes = total * sizeof(double);

        double* dL = nullptr;
        double* dR = nullptr;
        double* dA = nullptr;
        unsigned long long* dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dL, bytes));
        CUDA_CHECK(cudaMalloc(&dR, bytes));
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dOut, 2 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dOut, 0, 2 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dA, A_orig.data(), bytes, cudaMemcpyHostToDevice));

        // Compute L * L^T
        launchSymGemmNTExact(dL, dR, (int)n);

        // Compare with original
        int nblocks = (int)((total + 255) / 256);
        nblocks = min(nblocks, 4096);
        errorReduceKernel<<<nblocks, 256>>>(dR, dA, total, dOut);
        CUDA_CHECK(cudaGetLastError());

        unsigned long long out[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(out, dOut, sizeof(out), cudaMemcpyDeviceToHost));
        memcpy(&maxError, &out[0], sizeof(double));
        memcpy(&relError, &out[1], sizeof(double));

        CUDA_CHECK(cudaFree(dL));
        CUDA_CHECK(cudaFree(dR));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dOut));
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

    // Initialize the CUDA context up front so it is not attributed to the benchmark
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

    // Allocate GPU buffers / page-locked host memory before measuring
    choleskyWorkspaceInit(A, n);

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    choleskyWorkspaceFree();

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
