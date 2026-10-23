#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define ALGO_TS 32
#define DIAG_THREADS 256
#define FACTOR_SUB 4
#define PANEL_WARPS 4
#define GEMM_TILE 64
#define GEMM_THREADS 16
#define GEMM_PER_THREAD 4
#define GEMM_KPAD (ALGO_TS + 1)

__device__ __forceinline__ double fastRecip(double b) {
    double r = (double)(1.0f / (float)b);
    r = r * (2.0 - b * r);
    r = r * (2.0 - b * r);
    return r;
}

__device__ __forceinline__ void fastSqrtInv(double x, double& root, double& invRoot) {
    double r = (double)rsqrtf((float)x);
    r = r * (1.5 - 0.5 * x * r * r);
    invRoot = r;
    root = x * r;
}

__global__ void factorDiagonalBlockKernel(double* __restrict__ A, const int n,
                                          const int diagStart, const int diagSize,
                                          int* __restrict__ failFlag,
                                          int* __restrict__ failIdx) {
    __shared__ double s[ALGO_TS][ALGO_TS + 1];
    __shared__ double invdiag[ALGO_TS];
    const int tid = threadIdx.x;
    const int count = diagSize * diagSize;

    for (int idx = tid; idx < count; idx += blockDim.x) {
        const int r = idx / diagSize;
        const int c = idx - r * diagSize;
        s[r][c] = A[(size_t)(diagStart + r) * n + diagStart + c];
    }
    __syncthreads();

    for (int base = 0; base < diagSize; base += FACTOR_SUB) {
        const int size = min(FACTOR_SUB, diagSize - base);
        const int end = base + size;

        for (int jj = 0; jj < size; ++jj) {
            const int j = base + jj;
            if (tid == 0) {
                double sum = 0.0;
                for (int k = base; k < j; ++k) {
                    sum += s[j][k] * s[j][k];
                }
                const double val = s[j][j] - sum;
                if (val <= 0.0) {
                    if (*failFlag == 0) {
                        *failFlag = 1;
                        *failIdx = diagStart + j;
                    }
                    s[j][j] = 1.0;
                    invdiag[j] = 1.0;
                } else {
                    double root;
                    fastSqrtInv(val, root, invdiag[j]);
                    s[j][j] = root;
                }
            }
            __syncthreads();

            for (int i = j + 1 + tid; i < end; i += blockDim.x) {
                double sum = 0.0;
                for (int k = base; k < j; ++k) {
                    sum += s[i][k] * s[j][k];
                }
                s[i][j] = (s[i][j] - sum) * invdiag[j];
            }
            __syncthreads();
        }

        if (end < diagSize) {
            for (int jj = 0; jj < size; ++jj) {
                const int j = base + jj;
                for (int i = end + tid; i < diagSize; i += blockDim.x) {
                    double sum = 0.0;
                    for (int k = base; k < j; ++k) {
                        sum += s[i][k] * s[j][k];
                    }
                    s[i][j] = (s[i][j] - sum) * invdiag[j];
                }
                __syncthreads();
            }

            const int trailing = diagSize - end;
            for (int idx = tid; idx < trailing * trailing; idx += blockDim.x) {
                const int r = idx / trailing;
                const int c = idx - r * trailing;
                if (r >= c) {
                    const int i = end + r;
                    const int j = end + c;
                    double sum = 0.0;
                    for (int k = base; k < end; ++k) {
                        sum += s[i][k] * s[j][k];
                    }
                    s[i][j] -= sum;
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < count; idx += blockDim.x) {
        const int r = idx / diagSize;
        const int c = idx - r * diagSize;
        if (r >= c) {
            A[(size_t)(diagStart + r) * n + diagStart + c] = s[r][c];
        }
    }
}

__global__ void panelSolveKernel(double* __restrict__ A, const int n,
                                 const int diagStart, const int diagEnd,
                                 const int* __restrict__ failFlag) {
    if (*failFlag) {
        return;
    }
    __shared__ double diag[ALGO_TS][ALGO_TS + 1];
    __shared__ double invdiag[ALGO_TS];
    __shared__ double rows[PANEL_WARPS][ALGO_TS + 1];

    const int tid = threadIdx.x;
    const int diagSize = diagEnd - diagStart;
    const int count = diagSize * diagSize;

    for (int idx = tid; idx < count; idx += blockDim.x) {
        const int r = idx / diagSize;
        const int c = idx - r * diagSize;
        diag[r][c] = A[(size_t)(diagStart + r) * n + diagStart + c];
    }
    __syncthreads();

    if (tid < diagSize) {
        invdiag[tid] = fastRecip(diag[tid][tid]);
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int i = diagEnd + blockIdx.x * (blockDim.x >> 5) + warp;
    if (i >= n) {
        return;
    }

    double* __restrict__ rowbuf = rows[warp];
    double* __restrict__ Ai = A + (size_t)i * n;
    if (lane < diagSize) {
        rowbuf[lane] = Ai[diagStart + lane];
    }
    __syncwarp();

    for (int jj = 0; jj < diagSize; ++jj) {
        double sum = (lane < jj) ? rowbuf[lane] * diag[jj][lane] : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        }
        if (lane == 0) {
            rowbuf[jj] = (rowbuf[jj] - sum) * invdiag[jj];
        }
        __syncwarp();
    }

    if (lane < diagSize) {
        Ai[diagStart + lane] = rowbuf[lane];
    }
}

__global__ void updateTrailingKernel(double* __restrict__ A, const int n,
                                     const int base, const int diagStart,
                                     const int K,
                                     const int* __restrict__ failFlag) {
    if (*failFlag) {
        return;
    }
    const int rowTile = blockIdx.y;
    const int colTile = blockIdx.x;
    if (colTile > rowTile) {
        return;
    }

    __shared__ double As[GEMM_TILE][GEMM_KPAD];
    __shared__ double Bs[GEMM_TILE][GEMM_KPAD];

    const int tid = threadIdx.y * GEMM_THREADS + threadIdx.x;
    const int tileElems = GEMM_TILE * K;

    for (int idx = tid; idx < tileElems; idx += GEMM_THREADS * GEMM_THREADS) {
        const int r = idx / K;
        const int kk = idx - r * K;
        const int gi = base + rowTile * GEMM_TILE + r;
        As[r][kk] = (gi < n) ? A[(size_t)gi * n + diagStart + kk] : 0.0;
    }
    for (int idx = tid; idx < tileElems; idx += GEMM_THREADS * GEMM_THREADS) {
        const int c = idx / K;
        const int kk = idx - c * K;
        const int gj = base + colTile * GEMM_TILE + c;
        Bs[c][kk] = (gj < n) ? A[(size_t)gj * n + diagStart + kk] : 0.0;
    }
    __syncthreads();

    const int r0 = threadIdx.y * GEMM_PER_THREAD;
    const int c0 = threadIdx.x * GEMM_PER_THREAD;

    double acc[GEMM_PER_THREAD][GEMM_PER_THREAD];
#pragma unroll
    for (int rr = 0; rr < GEMM_PER_THREAD; ++rr) {
#pragma unroll
        for (int cc = 0; cc < GEMM_PER_THREAD; ++cc) {
            acc[rr][cc] = 0.0;
        }
    }

    for (int kk = 0; kk < K; ++kk) {
        double a[GEMM_PER_THREAD];
        double b[GEMM_PER_THREAD];
#pragma unroll
        for (int rr = 0; rr < GEMM_PER_THREAD; ++rr) {
            a[rr] = As[r0 + rr][kk];
        }
#pragma unroll
        for (int cc = 0; cc < GEMM_PER_THREAD; ++cc) {
            b[cc] = Bs[c0 + cc][kk];
        }
#pragma unroll
        for (int rr = 0; rr < GEMM_PER_THREAD; ++rr) {
#pragma unroll
            for (int cc = 0; cc < GEMM_PER_THREAD; ++cc) {
                acc[rr][cc] += a[rr] * b[cc];
            }
        }
    }

#pragma unroll
    for (int rr = 0; rr < GEMM_PER_THREAD; ++rr) {
        const int r = r0 + rr;
        const int gi = base + rowTile * GEMM_TILE + r;
        if (gi >= n) {
            continue;
        }
#pragma unroll
        for (int cc = 0; cc < GEMM_PER_THREAD; ++cc) {
            const int c = c0 + cc;
            const int gj = base + colTile * GEMM_TILE + c;
            if (gj >= n) {
                continue;
            }
            if (rowTile == colTile && c > r) {
                continue;
            }
            A[(size_t)gi * n + gj] -= acc[rr][cc];
        }
    }
}

__global__ void zeroUpperKernel(double* __restrict__ A, const int n) {
    const int i = blockIdx.x;
    double* __restrict__ row = A + (size_t)i * n;
    for (int j = i + 1 + threadIdx.x; j < n; j += blockDim.x) {
        row[j] = 0.0;
    }
}

static bool checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        printf("CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        return false;
    }
    return true;
}

__global__ void warmupKernel() {}

// Simple Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(double* dA, const size_t n,
                           int* dFail, int* dFailIdx) {
    if (n == 0) {
        return true;
    }

    const int ni = (int)n;
    const int nb = (ni + ALGO_TS - 1) / ALGO_TS;
    for (int b = 0; b < nb; ++b) {
        const int diagStart = b * ALGO_TS;
        const int diagEnd = std::min(ni, diagStart + ALGO_TS);
        const int diagSize = diagEnd - diagStart;

        factorDiagonalBlockKernel<<<1, DIAG_THREADS>>>(
            dA, ni, diagStart, diagSize, dFail, dFailIdx);

        if (diagEnd < ni) {
            const int rows = ni - diagEnd;
            const int panelWarps = PANEL_WARPS;
            const int panelBlocks = (rows + panelWarps - 1) / panelWarps;
            panelSolveKernel<<<panelBlocks, panelWarps * 32>>>(
                dA, ni, diagStart, diagEnd, dFail);

            const int m = ni - diagEnd;
            const int tiles = (m + GEMM_TILE - 1) / GEMM_TILE;
            dim3 grid(tiles, tiles);
            dim3 block(GEMM_THREADS, GEMM_THREADS);
            updateTrailingKernel<<<grid, block>>>(
                dA, ni, diagEnd, diagStart, diagSize, dFail);
        }
    }

    zeroUpperKernel<<<ni, 256>>>(dA, ni);

    return checkCuda(cudaDeviceSynchronize(), "kernel execution");
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
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    cudaFree(nullptr);
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    const size_t bytes = n * n * sizeof(double);
    double* dA = nullptr;
    int* dFail = nullptr;
    int* dFailIdx = nullptr;
    bool setupOk = checkCuda(cudaMalloc(&dA, bytes), "cudaMalloc A") &&
                   checkCuda(cudaMalloc(&dFail, sizeof(int)), "cudaMalloc fail") &&
                   checkCuda(cudaMalloc(&dFailIdx, sizeof(int)), "cudaMalloc failIdx");
    if (setupOk) {
        setupOk = checkCuda(cudaMemset(dFail, 0, sizeof(int)), "memset fail") &&
                  checkCuda(cudaMemset(dFailIdx, 0, sizeof(int)), "memset failIdx") &&
                  checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "H2D");
    }
    if (!setupOk) {
        cudaFree(dA);
        cudaFree(dFail);
        cudaFree(dFailIdx);
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(dA, n, dFail, dFailIdx);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto durationUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    int fail = 0;
    int failIdx = 0;
    if (success) {
        success = checkCuda(cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost), "D2H fail") &&
                  checkCuda(cudaMemcpy(&failIdx, dFailIdx, sizeof(int), cudaMemcpyDeviceToHost), "D2H failIdx") &&
                  checkCuda(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost), "D2H A");
    }

    cudaFree(dA);
    cudaFree(dFail);
    cudaFree(dFailIdx);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (fail) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failIdx);
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    const double seconds = durationUs.count() / 1e6;
    double gflops = (seconds > 0.0) ? ops / seconds / 1e9 : 0.0;
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
