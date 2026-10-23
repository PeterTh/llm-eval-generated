#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA Cholesky decomposition (blocked, right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

namespace {

constexpr int kPanel = 32;
constexpr int kTile = 32;
constexpr int kThreads = 256;

void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

__device__ __forceinline__ double invSqrt(double x) {
    double r = static_cast<double>(rsqrtf(static_cast<float>(x)));
    r = r * (1.5 - 0.5 * x * r * r);
    r = r * (1.5 - 0.5 * x * r * r);
    return r;
}

// Row-major input -> column-major working matrix (lower triangle only).
__global__ void inputToColumnMajor(const double* __restrict__ src,
                                   double* __restrict__ dst,
                                   const int n) {
    __shared__ double tile[kTile][kTile + 1];
    const int br = blockIdx.y * kTile;
    const int bc = blockIdx.x * kTile;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    for (int j = 0; j < kTile; j += 8) {
        const int r = br + ty + j;
        const int c = bc + tx;
        tile[ty + j][tx] = (r < n && c < n) ? src[(size_t)r * n + c] : 0.0;
    }
    __syncthreads();
    for (int j = 0; j < kTile; j += 8) {
        const int cc = bc + ty + j;
        const int rr = br + tx;
        if (rr < n && cc < n) {
            dst[(size_t)cc * n + rr] = (rr >= cc) ? tile[tx][ty + j] : 0.0;
        }
    }
}

// Column-major result -> row-major result with a zeroed upper triangle.
__global__ void columnMajorToOutput(const double* __restrict__ src,
                                    double* __restrict__ dst,
                                    const int n) {
    __shared__ double tile[kTile][kTile + 1];
    const int br = blockIdx.y * kTile;
    const int bc = blockIdx.x * kTile;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    for (int j = 0; j < kTile; j += 8) {
        const int r = br + ty + j;
        const int c = bc + tx;
        tile[ty + j][tx] = (r < n && c < n) ? src[(size_t)r * n + c] : 0.0;
    }
    __syncthreads();
    for (int j = 0; j < kTile; j += 8) {
        const int rr = bc + ty + j;
        const int cc = br + tx;
        if (rr < n && cc < n) {
            dst[(size_t)rr * n + cc] = (rr >= cc) ? tile[tx][ty + j] : 0.0;
        }
    }
}

// Factor a diagonal panel of width b (single block).
__global__ void panelFactor(double* __restrict__ l,
                            double* __restrict__ invDiag,
                            int* __restrict__ errFlag,
                            int* __restrict__ errIdx,
                            const int n, const int k, const int b) {
    __shared__ double sh[kPanel][kPanel + 1];
    __shared__ double invd[kPanel];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int j = 0; j < b; ++j) {
        for (int i = j + tid; i < b; i += kThreads) {
            sh[j][i] = l[(size_t)(k + j) * n + (k + i)];
        }
    }
    __syncthreads();

    for (int j = 0; j < b; ++j) {
        if (warp == 0) {
            double s = 0.0;
            for (int t = lane; t < j; t += 32) {
                const double v = sh[t][j];
                s += v * v;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_down_sync(0xffffffffu, s, off);
            }
            if (lane == 0) {
                const double val = sh[j][j] - s;
                double d;
                if (val <= 0.0) {
                    *errFlag = 1;
                    atomicMin(errIdx, k + j);
                    d = 0.0;
                    invd[j] = 0.0;
                } else {
                    invd[j] = invSqrt(val);
                    d = val * invd[j];
                }
                sh[j][j] = d;
                invDiag[k + j] = invd[j];
            }
        }
        __syncthreads();

        const double dinv = invd[j];
        const int sub = tid & 7;
        const int group = tid >> 3;
        for (int rowBase = j + 1; rowBase < b; rowBase += 32) {
            const int row = rowBase + group;
            double s0 = 0.0;
            double s1 = 0.0;
            if (row < b) {
                for (int t = sub; t < j; t += 16) {
                    s0 += sh[t][row] * sh[t][j];
                    if (t + 8 < j) {
                        s1 += sh[t + 8][row] * sh[t + 8][j];
                    }
                }
            }
            double s = s0 + s1;
            s += __shfl_down_sync(0xffffffffu, s, 4, 8);
            s += __shfl_down_sync(0xffffffffu, s, 2, 8);
            s += __shfl_down_sync(0xffffffffu, s, 1, 8);
            if (sub == 0 && row < b) {
                sh[j][row] = (sh[j][row] - s) * dinv;
            }
        }
        __syncthreads();
    }

    for (int j = 0; j < b; ++j) {
        for (int i = j + tid; i < b; i += kThreads) {
            l[(size_t)(k + j) * n + (k + i)] = sh[j][i];
        }
    }
}

// Triangular solve of the panel below the diagonal block.
__global__ void panelSolve(double* __restrict__ l,
                           const double* __restrict__ invDiag,
                           const int n, const int k, const int b,
                           const int start) {
    __shared__ double sh[kPanel][kPanel + 1];
    __shared__ double rows[8][kPanel];
    __shared__ double rhs[8][kPanel];
    __shared__ double dinv[kPanel];

    const int tid = threadIdx.x;
    if (tid < b) {
        dinv[tid] = invDiag[k + tid];
    }
    for (int idx = tid; idx < kPanel * kPanel; idx += kThreads) {
        const int c = idx / kPanel;
        const int r = idx % kPanel;
        if (c < b && r < b) {
            sh[c][r] = l[(size_t)(k + c) * n + (k + r)];
        }
    }
    for (int idx = tid; idx < 8 * kPanel; idx += kThreads) {
        const int w = idx & 7;
        const int j = idx >> 3;
        const int i = start + blockIdx.x * 8 + w;
        if (j < b && i < n) {
            rhs[w][j] = l[(size_t)(k + j) * n + i];
        }
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int i = start + blockIdx.x * 8 + warp;
    if (i >= n) {
        return;
    }

    for (int j = 0; j < b; ++j) {
        double s = 0.0;
        for (int t = lane; t < j; t += 32) {
            s += rows[warp][t] * sh[t][j];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off);
        }
        if (lane == 0) {
            const double v = (rhs[warp][j] - s) * dinv[j];
            l[(size_t)(k + j) * n + i] = v;
            rows[warp][j] = v;
        }
        __syncwarp();
    }
}

// Rank-b update of the trailing lower submatrix.
__global__ void trailingUpdate(double* __restrict__ l,
                               const int n, const int k, const int b,
                               const int start) {
    if (blockIdx.y > blockIdx.x) {
        return;
    }

    __shared__ double at[kPanel][kTile];
    __shared__ double bt[kPanel][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int i0 = start + blockIdx.x * kTile;
    const int j0 = start + blockIdx.y * kTile;

    for (int idx = tid; idx < kPanel * kTile; idx += kThreads) {
        const int rr = idx % kTile;
        const int t = idx / kTile;
        const int ii = i0 + rr;
        const int jj = j0 + rr;
        at[t][rr] = (t < b && ii < n) ? l[(size_t)(k + t) * n + ii] : 0.0;
        bt[t][rr] = (t < b && jj < n) ? l[(size_t)(k + t) * n + jj] : 0.0;
    }
    __syncthreads();

    const int r0 = ty * 2;
    const int r1 = r0 + 1;
    const int c0 = tx * 2;
    const int c1 = c0 + 1;
    double acc00 = 0.0, acc01 = 0.0, acc10 = 0.0, acc11 = 0.0;

    for (int t = 0; t < b; ++t) {
        const double a0 = at[t][r0];
        const double a1 = at[t][r1];
        const double b0 = bt[t][c0];
        const double b1 = bt[t][c1];
        acc00 += a0 * b0;
        acc01 += a0 * b1;
        acc10 += a1 * b0;
        acc11 += a1 * b1;
    }

    const int gi0 = i0 + r0;
    const int gi1 = i0 + r1;
    const int gj0 = j0 + c0;
    const int gj1 = j0 + c1;

    if (gi0 < n && gj0 < n && gi0 >= gj0) l[(size_t)gj0 * n + gi0] -= acc00;
    if (gi0 < n && gj1 < n && gi0 >= gj1) l[(size_t)gj1 * n + gi0] -= acc01;
    if (gi1 < n && gj0 < n && gi1 >= gj0) l[(size_t)gj0 * n + gi1] -= acc10;
    if (gi1 < n && gj1 < n && gi1 >= gj1) l[(size_t)gj1 * n + gi1] -= acc11;
}

__global__ void rowToColumnMajor(const double* __restrict__ src,
                                 double* __restrict__ dst,
                                 const int n) {
    __shared__ double tile[kTile][kTile + 1];
    const int br = blockIdx.y * kTile;
    const int bc = blockIdx.x * kTile;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    for (int j = 0; j < kTile; j += 8) {
        const int r = br + ty + j;
        const int c = bc + tx;
        tile[ty + j][tx] = (r < n && c < n) ? src[(size_t)r * n + c] : 0.0;
    }
    __syncthreads();
    for (int j = 0; j < kTile; j += 8) {
        const int cc = bc + ty + j;
        const int rr = br + tx;
        if (rr < n && cc < n) {
            dst[(size_t)cc * n + rr] = tile[tx][ty + j];
        }
    }
}

__global__ void btbLower(const double* __restrict__ bc,
                         double* __restrict__ a,
                         const int n) {
    if (blockIdx.y > blockIdx.x) {
        return;
    }

    __shared__ double as[kTile][kTile];
    __shared__ double bs[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int i0 = blockIdx.x * kTile;
    const int j0 = blockIdx.y * kTile;
    const int r0 = ty * 2;
    const int r1 = r0 + 1;
    const int c0 = tx * 2;
    const int c1 = c0 + 1;
    double acc00 = 0.0, acc01 = 0.0, acc10 = 0.0, acc11 = 0.0;

    for (int kc = 0; kc < n; kc += kTile) {
        for (int idx = tid; idx < kTile * kTile; idx += kThreads) {
            const int rr = idx % kTile;
            const int t = idx / kTile;
            const int kk = kc + t;
            as[rr][t] = (kk < n && i0 + rr < n) ? bc[(size_t)kk * n + i0 + rr] : 0.0;
            bs[rr][t] = (kk < n && j0 + rr < n) ? bc[(size_t)kk * n + j0 + rr] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < kTile; ++t) {
            const double a0 = as[r0][t];
            const double a1 = as[r1][t];
            const double b0 = bs[c0][t];
            const double b1 = bs[c1][t];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        __syncthreads();
    }

    const int gi0 = i0 + r0;
    const int gi1 = i0 + r1;
    const int gj0 = j0 + c0;
    const int gj1 = j0 + c1;

    if (gi0 < n && gj0 < n && gi0 > gj0) {
        a[(size_t)gi0 * n + gj0] = acc00;
        a[(size_t)gj0 * n + gi0] = acc00;
    } else if (gi0 < n && gi0 == gj0) {
        a[(size_t)gi0 * n + gj0] = acc00 + static_cast<double>(n);
    }
    if (gi0 < n && gj1 < n && gi0 > gj1) {
        a[(size_t)gi0 * n + gj1] = acc01;
        a[(size_t)gj1 * n + gi0] = acc01;
    } else if (gi0 < n && gi0 == gj1) {
        a[(size_t)gi0 * n + gj1] = acc01 + static_cast<double>(n);
    }
    if (gi1 < n && gj0 < n && gi1 > gj0) {
        a[(size_t)gi1 * n + gj0] = acc10;
        a[(size_t)gj0 * n + gi1] = acc10;
    } else if (gi1 < n && gi1 == gj0) {
        a[(size_t)gi1 * n + gj0] = acc10 + static_cast<double>(n);
    }
    if (gi1 < n && gj1 < n && gi1 > gj1) {
        a[(size_t)gi1 * n + gj1] = acc11;
        a[(size_t)gj1 * n + gi1] = acc11;
    } else if (gi1 < n && gi1 == gj1) {
        a[(size_t)gi1 * n + gj1] = acc11 + static_cast<double>(n);
    }
}

double* g_residentLower = nullptr;
size_t g_residentSize = 0;

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t nSize) {
    if (nSize == 0) {
        return true;
    }
    const int n = static_cast<int>(nSize);
    const size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);

    double* dL = nullptr;
    double* dWork = nullptr;
    double* dInvDiag = nullptr;
    int* dErrFlag = nullptr;
    int* dErrIdx = nullptr;

    const dim3 transposeBlock(kTile, 8);
    const dim3 transposeGrid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    const dim3 tileBlock(16, 16);

    const bool resident = (g_residentLower != nullptr && g_residentSize == elements);
    if (resident) {
        dL = g_residentLower;
        g_residentLower = nullptr;
        g_residentSize = 0;
    } else {
        checkCuda(cudaMalloc(&dL, elements * sizeof(double)), "cudaMalloc(L)");
    }
    checkCuda(cudaMalloc(&dWork, elements * sizeof(double)), "cudaMalloc(work)");
    checkCuda(cudaMalloc(&dInvDiag, static_cast<size_t>(n) * sizeof(double)), "cudaMalloc(invDiag)");
    checkCuda(cudaMalloc(&dErrFlag, sizeof(int)), "cudaMalloc(errFlag)");
    checkCuda(cudaMalloc(&dErrIdx, sizeof(int)), "cudaMalloc(errIdx)");

    if (!resident) {
        checkCuda(cudaMemcpy(dWork, A.data(), elements * sizeof(double), cudaMemcpyHostToDevice),
                  "copy input");
        inputToColumnMajor<<<transposeGrid, transposeBlock>>>(dWork, dL, n);
    }
    checkCuda(cudaMemset(dErrFlag, 0, sizeof(int)), "memset(errFlag)");
    checkCuda(cudaMemset(dErrIdx, 0x7f, sizeof(int)), "memset(errIdx)");

    for (int k = 0; k < n; ) {
        const int b = std::min(kPanel, n - k);
        panelFactor<<<1, kThreads>>>(dL, dInvDiag, dErrFlag, dErrIdx, n, k, b);

        const int start = k + b;
        if (start < n) {
            const int rows = n - start;
            const dim3 solveGrid((rows + 7) / 8);
            panelSolve<<<solveGrid, kThreads>>>(dL, dInvDiag, n, k, b, start);

            const int trailTiles = (rows + kTile - 1) / kTile;
            const dim3 updateGrid(trailTiles, trailTiles);
            trailingUpdate<<<updateGrid, tileBlock>>>(dL, n, k, b, start);
        }
        k += b;
    }

    columnMajorToOutput<<<transposeGrid, transposeBlock>>>(dL, dWork, n);
    checkCuda(cudaGetLastError(), "kernel launch");

    checkCuda(cudaMemcpy(A.data(), dWork, elements * sizeof(double), cudaMemcpyDeviceToHost),
              "copy output");

    int hErrFlag = 0;
    int hErrIdx = 0;
    checkCuda(cudaMemcpy(&hErrFlag, dErrFlag, sizeof(int), cudaMemcpyDeviceToHost), "copy errFlag");
    checkCuda(cudaMemcpy(&hErrIdx, dErrIdx, sizeof(int), cudaMemcpyDeviceToHost), "copy errIdx");

    checkCuda(cudaFree(dErrIdx), "cudaFree(errIdx)");
    checkCuda(cudaFree(dErrFlag), "cudaFree(errFlag)");
    checkCuda(cudaFree(dInvDiag), "cudaFree(invDiag)");
    checkCuda(cudaFree(dWork), "cudaFree(work)");
    checkCuda(cudaFree(dL), "cudaFree(L)");

    if (hErrFlag != 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", hErrIdx);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t nSize) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    if (nSize == 0) {
        return;
    }
    const int n = static_cast<int>(nSize);
    const size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);

    std::vector<double> B(elements);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < elements; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    double* dBC = nullptr;
    double* dA = nullptr;
    checkCuda(cudaMalloc(&dB, elements * sizeof(double)), "cudaMalloc(B)");
    checkCuda(cudaMalloc(&dBC, elements * sizeof(double)), "cudaMalloc(BC)");
    checkCuda(cudaMalloc(&dA, elements * sizeof(double)), "cudaMalloc(A)");
    checkCuda(cudaMemcpy(dB, B.data(), elements * sizeof(double), cudaMemcpyHostToDevice),
              "copy B");

    const dim3 transposeBlock(kTile, 8);
    const dim3 transposeGrid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    rowToColumnMajor<<<transposeGrid, transposeBlock>>>(dB, dBC, n);

    const int tiles = (n + kTile - 1) / kTile;
    const dim3 gemmBlock(16, 16);
    const dim3 gemmGrid(tiles, tiles);
    btbLower<<<gemmGrid, gemmBlock>>>(dBC, dA, n);
    checkCuda(cudaGetLastError(), "generation kernel launch");

    checkCuda(cudaFree(dBC), "cudaFree(BC)");
    checkCuda(cudaFree(dB), "cudaFree(B)");

    if (g_residentLower != nullptr && g_residentSize != elements) {
        checkCuda(cudaFree(g_residentLower), "cudaFree(resident)");
        g_residentLower = nullptr;
    }
    if (g_residentLower == nullptr) {
        checkCuda(cudaMalloc(&g_residentLower, elements * sizeof(double)), "cudaMalloc(resident)");
        g_residentSize = elements;
    }
    inputToColumnMajor<<<transposeGrid, transposeBlock>>>(dA, g_residentLower, n);

    checkCuda(cudaMemcpy(A.data(), dA, elements * sizeof(double), cudaMemcpyDeviceToHost),
              "copy A");

    checkCuda(cudaFree(dA), "cudaFree(A)");
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

    checkCuda(cudaFree(nullptr), "CUDA initialization");

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
