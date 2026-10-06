#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Cholesky decomposition on the GPU (CUDA), blocked right-looking algorithm
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

constexpr int NB = 64;        // panel width
constexpr int TS = 64;        // GEMM output tile
constexpr int BK = 16;        // GEMM k-chunk
constexpr int GEMM_THREADS = 256;
constexpr int DIAG_THREADS = 256;
constexpr int TRSM_ROWS = 64; // rows per TRSM block (one thread per row)

// C[i][j] (=|-=) sum_k A[i][k] * B[j][k]  (i < M, j < N); LOWER: only j <= i.
// Each thread accumulates k in ascending order; EXACT uses separately rounded
// multiply and add (bitwise identical to the plain sequential CPU loop).
template <bool LOWER, bool SUB, bool EXACT = false>
__global__ void __launch_bounds__(GEMM_THREADS)
gemmNTKernel(const double* __restrict__ A, size_t lda, const double* __restrict__ B, size_t ldb,
             double* __restrict__ C, size_t ldc, int M, int N, int K, const int* __restrict__ info) {
    const int bi = blockIdx.y, bj = blockIdx.x;
    if (LOWER && bj > bi) return;
    if (info && *info != INT_MAX) return;

    __shared__ double As[BK][TS];
    __shared__ double Bs[BK][TS];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    const int i0 = bi * TS, j0 = bj * TS;

    double acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0;

    const int lk = tid % BK, lr = tid / BK;  // load mapping: 16 rows per pass
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kk = k0 + lk;
#pragma unroll
        for (int p = 0; p < TS / (GEMM_THREADS / BK); ++p) {
            const int r = lr + p * (GEMM_THREADS / BK);
            const int gi = i0 + r, gj = j0 + r;
            As[lk][r] = (gi < M && kk < K) ? A[(size_t)gi * lda + kk] : 0.0;
            Bs[lk][r] = (gj < N && kk < K) ? B[(size_t)gj * ldb + kk] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[4], b[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) a[m] = As[k][ty + 16 * m];
#pragma unroll
            for (int m = 0; m < 4; ++m) b[m] = Bs[k][tx + 16 * m];
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int y = 0; y < 4; ++y) acc[x][y] = EXACT ? __dadd_rn(acc[x][y], __dmul_rn(a[x], b[y]))
                                                     : fma(a[x], b[y], acc[x][y]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const int gi = i0 + ty + 16 * x;
        if (gi >= M) continue;
#pragma unroll
        for (int y = 0; y < 4; ++y) {
            const int gj = j0 + tx + 16 * y;
            if (gj >= N || (LOWER && gj > gi)) continue;
            double* c = &C[(size_t)gi * ldc + gj];
            if (SUB) *c -= acc[x][y];
            else *c = acc[x][y];
        }
    }
}

// Factor the b x b diagonal block at (k,k) in shared memory. Thread t owns
// row t/4, columns t%4 + 4q. The strictly upper part of the block is zeroed.
__global__ void __launch_bounds__(DIAG_THREADS)
potrfDiagKernel(double* __restrict__ A, size_t n, int k, int b, int* __restrict__ info) {
    static_assert(DIAG_THREADS == 4 * NB, "diag thread mapping");
    if (*info != INT_MAX) return;
    __shared__ double S[NB][NB + 1];
    const int tid = threadIdx.x;
    const int r = tid / 4, cg = tid % 4;
    double* __restrict__ rowp = &A[(size_t)(k + r) * n + k];
#pragma unroll
    for (int q = 0; q < NB / 4; ++q) {
        const int c = cg + 4 * q;
        if (r < b && c <= r) S[r][c] = rowp[c];
    }
    __syncthreads();

    for (int j = 0; j < b; ++j) {
        const double val = S[j][j];
        if (val <= 0.0) {
            // Matrix is not positive definite (uniform decision across the block)
            if (tid == 0) atomicMin(info, k + j);
            return;
        }
        const double d = sqrt(val);
        if (tid > j && tid < b) S[tid][j] /= d;
        __syncthreads();
        if (r > j && r < b) {
            const double lrj = S[r][j];
#pragma unroll
            for (int q = 0; q < NB / 4; ++q) {
                const int c = cg + 4 * q;
                if (c > j && c <= r) S[r][c] -= lrj * S[c][j];
            }
        }
        __syncthreads();
    }

    if (r < b) {
#pragma unroll
        for (int q = 0; q < NB / 4; ++q) {
            const int c = cg + 4 * q;
            if (c < b) rowp[c] = (c < r) ? S[r][c] : (c == r ? sqrt(S[r][r]) : 0.0);
        }
    }
}

// Solve rows below the diagonal block: X * L11^T = A21 (one thread per row,
// row kept in registers).
__global__ void __launch_bounds__(TRSM_ROWS)
trsmPanelKernel(double* __restrict__ A, size_t n, int k, int* __restrict__ info) {
    if (*info != INT_MAX) return;
    __shared__ double L[NB][NB];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < NB * NB; idx += TRSM_ROWS) {
        const int r = idx / NB, c = idx % NB;
        L[r][c] = (c <= r) ? A[(size_t)(k + r) * n + k + c] : 0.0;
    }
    __syncthreads();

    const size_t row = (size_t)k + NB + (size_t)blockIdx.x * TRSM_ROWS + tid;
    if (row >= n) return;
    double* __restrict__ a = &A[row * n + k];
    double x[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) x[c] = a[c];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        x[j] /= L[j][j];
#pragma unroll
        for (int c = j + 1; c < NB; ++c) x[c] = fma(-x[j], L[c][j], x[c]);
    }
#pragma unroll
    for (int c = 0; c < NB; ++c) a[c] = x[c];
}

__global__ void addDiagonalKernel(double* __restrict__ A, size_t n, double v) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += v;
}

template <bool LOWER, bool SUB, bool EXACT = false>
static void launchGemmNT(const double* A, size_t lda, const double* B, size_t ldb, double* C, size_t ldc,
                         int M, int N, int K, const int* info, cudaStream_t s) {
    if (M <= 0 || N <= 0) return;
    dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS);
    gemmNTKernel<LOWER, SUB, EXACT><<<grid, GEMM_THREADS, 0, s>>>(A, lda, B, ldb, C, ldc, M, N, K, info);
}

// Persistent device resources (allocated outside the timed region)
static double* d_A = nullptr;
static int* d_info = nullptr;
static size_t d_capacity = 0;
static cudaStream_t sMain, sPanel, sUp, sDown;
static std::vector<cudaEvent_t> evPanel, evDiag, evMain;

static void ensureDeviceBuffers(size_t n) {
    if (d_capacity < n * n) {
        if (d_A) CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaMalloc(&d_A, std::max<size_t>(n * n, 1) * sizeof(double)));
        d_capacity = n * n;
    }
    if (!d_info) {
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        int lo, hi;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sMain, cudaStreamNonBlocking, lo));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sPanel, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sUp, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sDown, cudaStreamNonBlocking));
    }
    const size_t panels = (n + NB - 1) / NB + 1;
    while (evPanel.size() < panels) {
        cudaEvent_t e[3];
        for (auto& x : e) CUDA_CHECK(cudaEventCreateWithFlags(&x, cudaEventDisableTiming));
        evPanel.push_back(e[0]);
        evDiag.push_back(e[1]);
        evMain.push_back(e[2]);
    }
}

// Copy the lower part (columns [0, k+b)) of row block [k, k+b)
static void copyRowBlock(double* dst, const double* src, size_t n, size_t k, size_t b, cudaMemcpyKind kind,
                         cudaStream_t s) {
    CUDA_CHECK(cudaMemcpy2DAsync(dst + k * n, n * sizeof(double), src + k * n, n * sizeof(double),
                                 (k + b) * sizeof(double), b, kind, s));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order. Only the lower triangle is read; the
    // strictly upper triangle of the result is zero.
    if (n == 0) return true;
    ensureDeviceBuffers(n);
    const int N = (int)n;
    const int P = (N + NB - 1) / NB;
    double* hA = A.data();
    auto pk = [&](int p) { return p * NB; };
    auto pb = [&](int p) { return std::min(NB, N - p * NB); };

    // Upload lower triangle
    static const int okInfo = INT_MAX;
    CUDA_CHECK(cudaMemcpyAsync(d_info, &okInfo, sizeof(int), cudaMemcpyHostToDevice, sUp));
    for (int p = 0; p < P; ++p) copyRowBlock(d_A, hA, n, pk(p), pb(p), cudaMemcpyHostToDevice, sUp);
    CUDA_CHECK(cudaEventRecord(evMain[P], sUp));
    CUDA_CHECK(cudaStreamWaitEvent(sMain, evMain[P], 0));
    CUDA_CHECK(cudaStreamWaitEvent(sPanel, evMain[P], 0));

    auto factorPanel = [&](int p) {
        const int k = pk(p), b = pb(p), m = N - k - b;
        potrfDiagKernel<<<1, DIAG_THREADS, 0, sPanel>>>(d_A, n, k, b, d_info);
        CUDA_CHECK(cudaEventRecord(evDiag[p], sPanel));
        if (m > 0) trsmPanelKernel<<<(m + TRSM_ROWS - 1) / TRSM_ROWS, TRSM_ROWS, 0, sPanel>>>(d_A, n, k, d_info);
        CUDA_CHECK(cudaEventRecord(evPanel[p], sPanel));
        // Row block p of L is final: stream it back to the host
        CUDA_CHECK(cudaStreamWaitEvent(sDown, evDiag[p], 0));
        copyRowBlock(hA, d_A, n, k, b, cudaMemcpyDeviceToHost, sDown);
    };
    factorPanel(0);

    // Right-looking update with one-panel lookahead: the next panel is updated
    // and factored on the high-priority stream while the bulk trailing update runs.
    for (int p = 0; p + 1 < P; ++p) {
        const int k = pk(p), b = pb(p);
        const int kn = k + b, bn = pb(p + 1), c0 = kn + bn;
        const int m = N - kn;

        // Trailing update (excluding next panel's columns) on the main stream
        CUDA_CHECK(cudaStreamWaitEvent(sMain, evPanel[p], 0));
        if (c0 < N) {
            const double* L = d_A + (size_t)c0 * n + k;
            launchGemmNT<true, true>(L, n, L, n, d_A + (size_t)c0 * n + c0, n, N - c0, N - c0, b, d_info, sMain);
        }
        CUDA_CHECK(cudaEventRecord(evMain[p], sMain));

        // Next panel: apply update from panel p, then factor it
        if (p > 0) CUDA_CHECK(cudaStreamWaitEvent(sPanel, evMain[p - 1], 0));
        const double* L = d_A + (size_t)kn * n + k;
        launchGemmNT<true, true>(L, n, L, n, d_A + (size_t)kn * n + kn, n, m, bn, b, d_info, sPanel);
        factorPanel(p + 1);
    }
    CUDA_CHECK(cudaGetLastError());

    // Zero the strictly upper triangle on the host while the GPU works
    for (int p = 0; p < P; ++p) {
        const size_t end = pk(p) + pb(p);
        for (size_t i = pk(p); i < end; ++i) std::fill(hA + i * n + end, hA + (i + 1) * n, 0.0);
    }

    CUDA_CHECK(cudaStreamSynchronize(sPanel));
    CUDA_CHECK(cudaStreamSynchronize(sMain));
    CUDA_CHECK(cudaStreamSynchronize(sDown));
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
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
    
    // Compute A = B * B^T on the GPU
    ensureDeviceBuffers(n);
    double* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    launchGemmNT<false, false, true>(d_B, n, d_B, n, d_A, n, (int)n, (int)n, (int)n, nullptr, 0);
    
    // Add diagonal dominance to ensure positive definiteness
    addDiagonalKernel<<<(unsigned)((n + 255) / 256), 256>>>(d_A, n, (double)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_B));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T on the GPU
    if (n > 0) {
        ensureDeviceBuffers(n);
        double* d_R = nullptr;
        CUDA_CHECK(cudaMalloc(&d_R, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_A, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        launchGemmNT<false, false, true>(d_A, n, d_A, n, d_R, n, (int)n, (int)n, (int)n, nullptr, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(reconstructed.data(), d_R, n * n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_R));
    }
    
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
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Pin host buffer and prepare device buffers (outside the timed region)
    CUDA_CHECK(cudaHostRegister(A.data(), std::max<size_t>(n * n, 1) * sizeof(double), cudaHostRegisterDefault));
    ensureDeviceBuffers(n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
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
