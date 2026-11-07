#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition on the GPU (CUDA)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Per panel of NB columns:
//   1. potrf_diag_kernel : factor the NB x NB diagonal block (single thread block, shared memory)
//   2. trsm_kernel       : L21 = A21 * L11^-T for all rows below the diagonal block
//   3. syrk_kernel       : A22 -= L21 * L21^T (lower-triangular tiles only, register-blocked GEMM)

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                              \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

namespace {

constexpr int NB = 64;           // panel / tile width
constexpr int SYRK_THREADS = 256;  // 16 x 16 threads, each computing a 4 x 4 register tile
constexpr int SYRK_KCHUNK = 16;    // k-depth staged in shared memory per iteration
constexpr int DIAG_THREADS = 256;
constexpr int TRSM_THREADS = NB;   // one thread per row

// Zero the strictly upper triangular part of A (the algorithm never reads it).
__global__ void zero_upper_kernel(double* __restrict__ A, size_t n) {
    const size_t i = blockIdx.y;
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}

// Factor the bs x bs diagonal block starting at (k0, k0). Lower triangle only.
// On a non-positive pivot the global column index is recorded in *flag (first failure wins).
__global__ void potrf_diag_kernel(double* __restrict__ A, size_t n, size_t k0, int bs,
                                  int* __restrict__ flag) {
    __shared__ double S[NB][NB + 1];
    __shared__ int sfail;

    const int tid = threadIdx.x;
    if (tid == 0) sfail = 0;

    for (int e = tid; e < bs * bs; e += blockDim.x) {
        const int i = e / bs;
        const int j = e - i * bs;
        if (j <= i) S[i][j] = A[(k0 + i) * n + (k0 + j)];
    }
    __syncthreads();

    for (int j = 0; j < bs; ++j) {
        if (tid == 0) {
            const double v = S[j][j];
            if (v <= 0.0) {
                sfail = 1;
                if (*flag < 0) *flag = (int)(k0 + j);
            } else {
                S[j][j] = sqrt(v);
            }
        }
        __syncthreads();
        if (sfail) break;

        const double d = S[j][j];
        for (int i = j + 1 + tid; i < bs; i += blockDim.x) {
            S[i][j] /= d;
        }
        __syncthreads();

        // Trailing update of the remaining lower triangle: S[i][c] -= S[i][j] * S[c][j], j < c <= i
        const int rem = bs - j - 1;
        for (int e = tid; e < rem * rem; e += blockDim.x) {
            const int r = e / rem;
            const int c = e - r * rem;
            if (c <= r) {
                const int gi = j + 1 + r;
                const int gc = j + 1 + c;
                S[gi][gc] -= S[gi][j] * S[gc][j];
            }
        }
        __syncthreads();
    }

    for (int e = tid; e < bs * bs; e += blockDim.x) {
        const int i = e / bs;
        const int j = e - i * bs;
        if (j <= i) A[(k0 + i) * n + (k0 + j)] = S[i][j];
    }
}

// Solve X * L11^T = A21 for the rows below the diagonal block; one thread per row.
__global__ void trsm_kernel(double* __restrict__ A, size_t n, size_t k0, int bs) {
    __shared__ double L[NB][NB + 1];

    const int tid = threadIdx.x;
    for (int e = tid; e < bs * bs; e += blockDim.x) {
        const int i = e / bs;
        const int j = e - i * bs;
        if (j <= i) L[i][j] = A[(k0 + i) * n + (k0 + j)];
    }
    __syncthreads();

    const size_t row = k0 + bs + (size_t)blockIdx.x * blockDim.x + tid;
    if (row >= n) return;

    double* __restrict__ arow = A + row * n + k0;

    double x[NB];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        x[j] = (j < bs) ? arow[j] : 0.0;
    }

#pragma unroll
    for (int j = 0; j < NB; ++j) {
        if (j >= bs) break;
        double s = x[j];
#pragma unroll
        for (int k = 0; k < j; ++k) {
            s -= x[k] * L[j][k];
        }
        x[j] = s / L[j][j];
    }

#pragma unroll
    for (int j = 0; j < NB; ++j) {
        if (j < bs) arow[j] = x[j];
    }
}

// Trailing update: C -= P * P^T where P is the n x bs panel below the diagonal block
// (rows >= base, columns [k0, k0 + bs)) and C is the trailing lower triangle of A.
// Each thread block computes a 64 x 64 tile; only tiles on or below the diagonal are processed.
__global__ void __launch_bounds__(SYRK_THREADS)
syrk_kernel(double* __restrict__ A, size_t n, size_t k0, int bs) {
    const int tr = blockIdx.y;
    const int tc = blockIdx.x;
    if (tc > tr) return;

    __shared__ double As[SYRK_KCHUNK][NB];
    __shared__ double Bs[SYRK_KCHUNK][NB];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;

    const size_t base = k0 + bs;
    const size_t row0 = base + (size_t)tr * NB;
    const size_t col0 = base + (size_t)tc * NB;

    double c[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int s = 0; s < 4; ++s) c[r][s] = 0.0;

    for (int kc = 0; kc < bs; kc += SYRK_KCHUNK) {
#pragma unroll
        for (int q = 0; q < (NB * SYRK_KCHUNK) / SYRK_THREADS; ++q) {
            const int e = tid + q * SYRK_THREADS;
            const int i = e >> 4;
            const int kk = e & 15;
            const int k = kc + kk;
            const size_t gi = row0 + i;
            const size_t gj = col0 + i;
            As[kk][i] = (k < bs && gi < n) ? A[gi * n + k0 + k] : 0.0;
            Bs[kk][i] = (k < bs && gj < n) ? A[gj * n + k0 + k] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < SYRK_KCHUNK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = As[kk][tx + 16 * r];
#pragma unroll
            for (int s = 0; s < 4; ++s) b[s] = Bs[kk][ty + 16 * s];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int s = 0; s < 4; ++s) c[r][s] += a[r] * b[s];
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const size_t gi = row0 + tx + 16 * r;
        if (gi >= n) continue;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const size_t gj = col0 + ty + 16 * s;
            if (gj <= gi) {
                A[gi * n + gj] -= c[r][s];
            }
        }
    }
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order; the result L overwrites the lower triangle and the
    // strictly upper triangle is zeroed, exactly as in the sequential reference.
    if (n == 0) return true;

    const size_t bytes = n * n * sizeof(double);
    double* dA = nullptr;
    int* dFlag = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dFlag, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    const int negOne = -1;
    CUDA_CHECK(cudaMemcpy(dFlag, &negOne, sizeof(int), cudaMemcpyHostToDevice));

    {
        const dim3 grid((unsigned)((n + 255) / 256), (unsigned)n);
        zero_upper_kernel<<<grid, 256>>>(dA, n);
    }

    for (size_t k0 = 0; k0 < n; k0 += NB) {
        const int bs = (int)std::min<size_t>(NB, n - k0);
        const size_t rem = n - k0 - bs;  // rows/cols in the trailing submatrix

        potrf_diag_kernel<<<1, DIAG_THREADS>>>(dA, n, k0, bs, dFlag);

        if (rem > 0) {
            const unsigned rowBlocks = (unsigned)((rem + TRSM_THREADS - 1) / TRSM_THREADS);
            trsm_kernel<<<rowBlocks, TRSM_THREADS>>>(dA, n, k0, bs);

            const unsigned tiles = (unsigned)((rem + NB - 1) / NB);
            syrk_kernel<<<dim3(tiles, tiles), SYRK_THREADS>>>(dA, n, k0, bs);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    int flag = -1;
    CUDA_CHECK(cudaMemcpy(&flag, dFlag, sizeof(int), cudaMemcpyDeviceToHost));
    if (flag >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)flag);
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dFlag));
        return false;
    }

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dFlag));
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
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
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
    
    // Initialize the CUDA context and load the kernel modules up front so that one-time
    // setup cost is not attributed to the timed region
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, zero_upper_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, potrf_diag_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, trsm_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, syrk_kernel));
    }
    
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
