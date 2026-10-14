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

// Blocked, right-looking Cholesky decomposition on the GPU.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// A is stored in row-major order; L overwrites the lower triangle and the upper triangle is zeroed.
//
// Each panel of NB columns is processed in three phases:
//   1. factor the NB x NB diagonal block               (potrfDiagKernel, one thread block)
//   2. solve for the rows of the panel below it        (trsmPanelKernel)
//   3. update the trailing lower triangle              (gemmNTKernel<true>, tiled GEMM)

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

namespace {

constexpr int NB = 64;            // panel width / diagonal block size
constexpr int TRSM_THREADS = 128; // one thread per row of the panel
constexpr int GEMM_DIM = 16;      // threads per tile dimension
constexpr int GEMM_MICRO = 4;     // outputs computed per thread per dimension
constexpr int GEMM_TILE = GEMM_DIM * GEMM_MICRO; // 64 x 64 output tile
constexpr int GEMM_BK = 16;       // K-step staged in shared memory

// Copy the lower triangle of the b x b diagonal block at (k0, k0) into shared memory.
__device__ void loadLowerBlock(double (*S)[NB], const double* A, size_t n, int k0, int b) {
    for (int idx = threadIdx.x; idx < b * b; idx += blockDim.x) {
        const int r = idx / b, c = idx % b;
        if (c <= r) S[r][c] = A[(size_t)(k0 + r) * n + (k0 + c)];
    }
}

// Unblocked Cholesky of the diagonal block at (k0, k0), executed by a single thread block.
// Records the smallest failing global diagonal index in *failIdx (INT_MAX if none).
__global__ void potrfDiagKernel(double* A, size_t n, int k0, int b, int* failIdx) {
    __shared__ double S[NB][NB];
    loadLowerBlock(S, A, n, k0, b);
    __syncthreads();

    for (int j = 0; j < b; ++j) {
        if (threadIdx.x == 0) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += S[j][k] * S[j][k];
            }
            const double val = S[j][j] - sum;
            if (val <= 0.0) {
                atomicMin(failIdx, k0 + j);
            }
            S[j][j] = sqrt(val);
        }
        __syncthreads();

        for (int i = j + 1 + threadIdx.x; i < b; i += blockDim.x) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += S[i][k] * S[j][k];
            }
            S[i][j] = (S[i][j] - sum) / S[j][j];
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < b * b; idx += blockDim.x) {
        const int r = idx / b, c = idx % b;
        if (c <= r) A[(size_t)(k0 + r) * n + (k0 + c)] = S[r][c];
    }
}

// Solve X * L_kk^T = A_panel for every row below the diagonal block, in place.
__global__ void trsmPanelKernel(double* A, size_t n, int k0, int b, size_t rowStart) {
    __shared__ double L[NB][NB];
    loadLowerBlock(L, A, n, k0, b);
    __syncthreads();

    const size_t r = rowStart + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;

    double* row = A + r * n + k0;
    for (int j = 0; j < b; ++j) {
        double v = row[j];
        for (int m = 0; m < j; ++m) {
            v -= row[m] * L[j][m];
        }
        row[j] = v / L[j][j];
    }
}

// C = alpha * X * Y^T + beta * C, with X (M x K) and Y (N x K) and C (M x N), all row-major.
// If LOWER, only elements with column <= row are computed and stored; tiles entirely above
// the diagonal are skipped.
template <bool LOWER>
__global__ void gemmNTKernel(double* C, size_t ldc, const double* X, const double* Y, size_t ld,
                             int M, int N, int K, double alpha, double beta) {
    if (LOWER && blockIdx.x > blockIdx.y) return;

    __shared__ double As[GEMM_BK][GEMM_TILE + 1];
    __shared__ double Bs[GEMM_BK][GEMM_TILE + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * GEMM_DIM + tx;
    const int rowBase = blockIdx.y * GEMM_TILE;
    const int colBase = blockIdx.x * GEMM_TILE;

    double acc[GEMM_MICRO][GEMM_MICRO] = {};

    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        for (int idx = tid; idx < GEMM_TILE * GEMM_BK; idx += GEMM_DIM * GEMM_DIM) {
            const int r = idx / GEMM_BK, kk = idx % GEMM_BK;
            const int k = k0 + kk;
            const int xr = rowBase + r, yr = colBase + r;
            As[kk][r] = (xr < M && k < K) ? X[(size_t)xr * ld + k] : 0.0;
            Bs[kk][r] = (yr < N && k < K) ? Y[(size_t)yr * ld + k] : 0.0;
        }
        __syncthreads();

        for (int kk = 0; kk < GEMM_BK; ++kk) {
            double a[GEMM_MICRO], b[GEMM_MICRO];
            for (int r = 0; r < GEMM_MICRO; ++r) a[r] = As[kk][ty + GEMM_DIM * r];
            for (int c = 0; c < GEMM_MICRO; ++c) b[c] = Bs[kk][tx + GEMM_DIM * c];
            for (int r = 0; r < GEMM_MICRO; ++r)
                for (int c = 0; c < GEMM_MICRO; ++c)
                    acc[r][c] += a[r] * b[c];
        }
        __syncthreads();
    }

    for (int r = 0; r < GEMM_MICRO; ++r) {
        for (int c = 0; c < GEMM_MICRO; ++c) {
            const int i = rowBase + ty + GEMM_DIM * r;
            const int j = colBase + tx + GEMM_DIM * c;
            if (i < M && j < N && (!LOWER || j <= i)) {
                double* out = &C[(size_t)i * ldc + j];
                const double v = alpha * acc[r][c];
                *out = (beta == 0.0) ? v : v + beta * (*out);
            }
        }
    }
}

__global__ void addDiagonalKernel(double* A, size_t n, double value) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += value;
}

__global__ void zeroUpperKernel(double* A, size_t n) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    const size_t i = idx / n, j = idx % n;
    if (j > i) A[idx] = 0.0;
}

void gemmNT(double* C, size_t ldc, const double* X, const double* Y, size_t ld,
            int M, int N, int K, double alpha, double beta) {
    if (M == 0 || N == 0) return;
    const dim3 block(GEMM_DIM, GEMM_DIM);
    const dim3 grid((N + GEMM_TILE - 1) / GEMM_TILE, (M + GEMM_TILE - 1) / GEMM_TILE);
    gemmNTKernel<false><<<grid, block>>>(C, ldc, X, Y, ld, M, N, K, alpha, beta);
}

} // namespace

// Factor the lower triangle of dA in place (device memory). Returns false if not positive definite.
bool choleskyDecomposition(double* dA, const size_t n) {
    if (n == 0) return true;

    int* dFail = nullptr;
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));
    const int noFail = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dFail, &noFail, sizeof(int), cudaMemcpyHostToDevice));

    for (size_t k0 = 0; k0 < n; k0 += NB) {
        const int b = (int)std::min<size_t>(NB, n - k0);
        const size_t r0 = k0 + b; // first row below the diagonal block
        const int rows = (int)(n - r0);

        potrfDiagKernel<<<1, NB>>>(dA, n, (int)k0, b, dFail);

        if (rows > 0) {
            trsmPanelKernel<<<(rows + TRSM_THREADS - 1) / TRSM_THREADS, TRSM_THREADS>>>(
                dA, n, (int)k0, b, r0);

            // Trailing update: A22 -= P * P^T, where P = A[r0.., k0..k0+b)
            double* C = dA + r0 * n + r0;
            const double* P = dA + r0 * n + k0;
            gemmNTKernel<true><<<dim3((rows + GEMM_TILE - 1) / GEMM_TILE,
                                      (rows + GEMM_TILE - 1) / GEMM_TILE),
                                 dim3(GEMM_DIM, GEMM_DIM)>>>(
                C, n, P, P, n, rows, rows, b, -1.0, 1.0);
        }
    }

    const size_t total = n * n;
    zeroUpperKernel<<<(total + 255) / 256, 256>>>(dA, n);

    CUDA_CHECK(cudaGetLastError());
    int fail = noFail;
    CUDA_CHECK(cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dFail));

    if (fail != INT_MAX) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)fail);
        return false;
    }

    return true;
}

// Generate a symmetric positive definite matrix directly in device memory
void generatePositiveDefiniteMatrix(double* dA, const size_t n) {
    if (n == 0) return;

    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Compute A = B * B^T
    gemmNT(dA, n, dB, dB, n, (int)n, (int)n, (int)n, 1.0, 0.0);

    // Add diagonal dominance to ensure positive definiteness
    addDiagonalKernel<<<(n + 255) / 256, 256>>>(dA, n, (double)n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(dB));
}

bool validateCholesky(const std::vector<double>& reconstructed, const std::vector<double>& A_orig, const size_t n) {
    // Compare L * L^T (computed on the GPU) with the original matrix

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

    // Allocate matrix on the device
    const size_t count = n * n;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, count * sizeof(double)));

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dA, n);

    std::vector<double> A_orig;
    if (validate) {
        A_orig.resize(count);
        CUDA_CHECK(cudaMemcpy(A_orig.data(), dA, count * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Perform Cholesky decomposition (device-resident; timing covers all GPU work)
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(dA, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        CUDA_CHECK(cudaFree(dA));
        return 1;
    }

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Print results for external validation
    std::vector<double> A;
    if (printResults) {
        A.resize(count);
        CUDA_CHECK(cudaMemcpy(A.data(), dA, count * sizeof(double), cudaMemcpyDeviceToHost));
        print_results(A, "CholeskyL");
    }

    // Validation
    int status = 0;
    if (validate) {
        printf("Validating result...\n");

        // Reconstruct L * L^T on the GPU
        double* dR = nullptr;
        CUDA_CHECK(cudaMalloc(&dR, count * sizeof(double)));
        gemmNT(dR, n, dA, dA, n, (int)n, (int)n, (int)n, 1.0, 0.0);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> reconstructed(count);
        CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dR));

        bool valid = validateCholesky(reconstructed, A_orig, n);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }

    CUDA_CHECK(cudaFree(dA));
    return status;
}
