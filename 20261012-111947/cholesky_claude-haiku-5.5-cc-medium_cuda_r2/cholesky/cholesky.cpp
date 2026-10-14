#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked, left-looking Cholesky decomposition executed on the GPU.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// A is stored in row-major order and L overwrites the lower triangle. Every entry is
// computed as L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j], where the sum
// is accumulated in increasing k with fused multiply-add. All GPU kernels keep exactly this
// per-entry accumulation order, so the result matches the sequential algorithm bit for bit,
// while independent entries are computed in parallel.

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err_));                                       \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

// Column-block width of the blocked factorization
constexpr int BLOCK = 64;
// Dot-product tile: TILE x TILE outputs per block, 16 x 16 threads, 2 x 2 outputs per thread
constexpr int TILE = 32;
constexpr int KCHUNK = 16;
constexpr int DOT_THREADS = 256;
constexpr int ELEM_THREADS = 256;
constexpr int PANEL_THREADS = 128;

static int blocksFor(size_t count, int threads) {
    return std::max<int>(1, static_cast<int>((count + threads - 1) / threads));
}

// out[r * ldOut + c] = sum_{k<K} X[r * ld + k] * Y[c * ld + k] for r < M, c < N.
// The k-sum is accumulated strictly in increasing k order (chunks are processed in order),
// so each output matches the sequential reference exactly.
__global__ void dotTileKernel(const double* X, const double* Y, int ld, int M, int N, int K,
                              double* out, int ldOut) {
    __shared__ double Xs[TILE][KCHUNK + 1];
    __shared__ double Ys[TILE][KCHUNK + 1];

    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;
    const int rowBase = blockIdx.y * TILE;
    const int colBase = blockIdx.x * TILE;

    double acc[2][2] = {{0.0, 0.0}, {0.0, 0.0}};

    for (int kc = 0; kc < K; kc += KCHUNK) {
        const int kLen = min(KCHUNK, K - kc);
        for (int idx = threadIdx.x; idx < TILE * KCHUNK; idx += blockDim.x) {
            const int r = idx / KCHUNK;
            const int kk = idx % KCHUNK;
            Xs[r][kk] = (rowBase + r < M && kk < kLen) ? X[(size_t)(rowBase + r) * ld + kc + kk] : 0.0;
            Ys[r][kk] = (colBase + r < N && kk < kLen) ? Y[(size_t)(colBase + r) * ld + kc + kk] : 0.0;
        }
        __syncthreads();

        for (int kk = 0; kk < kLen; ++kk) {
            const double a0 = Xs[2 * ty][kk];
            const double a1 = Xs[2 * ty + 1][kk];
            const double b0 = Ys[2 * tx][kk];
            const double b1 = Ys[2 * tx + 1][kk];
            acc[0][0] = fma(a0, b0, acc[0][0]);
            acc[0][1] = fma(a0, b1, acc[0][1]);
            acc[1][0] = fma(a1, b0, acc[1][0]);
            acc[1][1] = fma(a1, b1, acc[1][1]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const int r = rowBase + 2 * ty + i;
            const int c = colBase + 2 * tx + j;
            if (r < M && c < N) {
                out[(size_t)r * ldOut + c] = acc[i][j];
            }
        }
    }
}

// Factor the diagonal block [c0, c0 + nb) x [c0, c0 + nb) of A.
// S holds, for each row r = c0 + t, the partial sums over k < c0 (computed by dotTileKernel).
// Single block, one thread per row of the diagonal block; the columns are processed in order.
template <int B>
__global__ void diagonalBlockKernel(double* A, int n, int c0, int nb, const double* S, int* failIndex) {
    __shared__ double L[B][B];
    __shared__ int failed;

    const int t = threadIdx.x;
    if (t == 0) {
        failed = -1;
    }

    for (int jj = 0; jj < B; ++jj) {
        __syncthreads();
        if (jj < nb && t == jj) {
            // Diagonal element: L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)
            double sum = S[t * B + jj];
            for (int k = 0; k < jj; ++k) {
                sum = fma(L[t][k], L[t][k], sum);
            }
            const int j = c0 + jj;
            const double val = A[(size_t)j * n + j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                if (failed < 0) {
                    failed = j;
                }
            }
            L[t][t] = sqrt(val);
        }
        __syncthreads();
        if (jj < nb && t > jj && t < nb) {
            // Off-diagonal element: L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]
            double sum = S[t * B + jj];
            for (int k = 0; k < jj; ++k) {
                sum = fma(L[t][k], L[jj][k], sum);
            }
            L[t][jj] = (A[(size_t)(c0 + t) * n + (c0 + jj)] - sum) / L[jj][jj];
        }
    }
    __syncthreads();

    if (t < nb) {
        for (int k = 0; k <= t; ++k) {
            A[(size_t)(c0 + t) * n + (c0 + k)] = L[t][k];
        }
    }
    if (t == 0) {
        *failIndex = failed;
    }
}

// Compute the rows below the diagonal block (rows >= c0 + nb) for the columns of the block.
// Each thread owns one row; the factored diagonal block is staged in shared memory.
template <int B>
__global__ void panelKernel(double* A, int n, int c0, int nb, const double* S) {
    __shared__ double L[B][B];

    for (int idx = threadIdx.x; idx < nb * nb; idx += blockDim.x) {
        const int jj = idx / nb;
        const int k = idx % nb;
        if (k <= jj) {
            L[jj][k] = A[(size_t)(c0 + jj) * n + (c0 + k)];
        }
    }
    __syncthreads();

    const int r = c0 + nb + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) {
        return;
    }

    double row[B];
#pragma unroll
    for (int jj = 0; jj < B; ++jj) {
        row[jj] = 0.0;
    }

    double* Arow = A + (size_t)r * n + c0;
    const double* Srow = S + (size_t)(r - c0) * B;

#pragma unroll
    for (int jj = 0; jj < B; ++jj) {
        if (jj < nb) {
            double sum = Srow[jj];
#pragma unroll
            for (int k = 0; k < B; ++k) {
                if (k < jj) {
                    sum = fma(row[k], L[jj][k], sum);
                }
            }
            row[jj] = (Arow[jj] - sum) / L[jj][jj];
        }
    }

#pragma unroll
    for (int jj = 0; jj < B; ++jj) {
        if (jj < nb) {
            Arow[jj] = row[jj];
        }
    }
}

__global__ void addDiagonalKernel(double* A, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[(size_t)i * n + i] += n;
    }
}

__global__ void zeroUpperKernel(double* A, int n) {
    const size_t total = (size_t)n * n;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t r = idx / n;
        const size_t c = idx % n;
        if (c > r) {
            A[idx] = 0.0;
        }
    }
}

// Max absolute and max relative error. Both are non-negative doubles, so their bit patterns
// order like unsigned integers and atomicMax on the bits yields the exact maximum.
__global__ void errorKernel(const double* rec, const double* orig, size_t total,
                            unsigned long long* maxAbs, unsigned long long* maxRel) {
    double localAbs = 0.0;
    double localRel = 0.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const double error = fabs(rec[i] - orig[i]);
        if (error > localAbs) {
            localAbs = error;
        }
        const double rel = error / (fabs(orig[i]) + 1e-10);
        if (rel > localRel) {
            localRel = rel;
        }
    }
    if (localAbs > 0.0) {
        atomicMax(maxAbs, static_cast<unsigned long long>(__double_as_longlong(localAbs)));
    }
    if (localRel > 0.0) {
        atomicMax(maxRel, static_cast<unsigned long long>(__double_as_longlong(localRel)));
    }
}

bool choleskyDecomposition(double* dA, const int n) {
    double* dS = nullptr;
    int* dFail = nullptr;
    CUDA_CHECK(cudaMalloc(&dS, sizeof(double) * static_cast<size_t>(n) * BLOCK));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));

    bool ok = true;
    for (int c0 = 0; c0 < n; c0 += BLOCK) {
        const int nb = std::min(BLOCK, n - c0);
        const int c1 = c0 + nb;
        const double* blockCols = dA + (size_t)c0 * n;

        // Contributions of all columns k < c0 to rows >= c0 of this column block
        dotTileKernel<<<dim3(blocksFor(nb, TILE), blocksFor(n - c0, TILE)), DOT_THREADS>>>(
            blockCols, blockCols, n, n - c0, nb, c0, dS, BLOCK);

        diagonalBlockKernel<BLOCK><<<1, BLOCK>>>(dA, n, c0, nb, dS, dFail);
        int fail = -1;
        CUDA_CHECK(cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost));
        if (fail >= 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", fail);
            ok = false;
            break;
        }

        if (c1 < n) {
            panelKernel<BLOCK><<<blocksFor(n - c1, PANEL_THREADS), PANEL_THREADS>>>(dA, n, c0, nb, dS);
        }
    }

    if (ok) {
        zeroUpperKernel<<<blocksFor(static_cast<size_t>(n) * n, ELEM_THREADS), ELEM_THREADS>>>(dA, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dFail));
    return ok;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(double* dA, const int n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(static_cast<size_t>(n) * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < B.size(); ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, sizeof(double) * B.size()));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice));

    // Compute A = B * B^T
    dotTileKernel<<<dim3(blocksFor(n, TILE), blocksFor(n, TILE)), DOT_THREADS>>>(dB, dB, n, n, n, n, dA, n);

    // Add diagonal dominance to ensure positive definiteness
    addDiagonalKernel<<<blocksFor(n, ELEM_THREADS), ELEM_THREADS>>>(dA, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(dB));
}

bool validateCholesky(const double* dL, const double* dA_orig, const int n) {
    // Validate by computing L * L^T and comparing with original matrix
    const size_t total = static_cast<size_t>(n) * n;

    double* dRec = nullptr;
    unsigned long long* dMax = nullptr;
    CUDA_CHECK(cudaMalloc(&dRec, sizeof(double) * total));
    CUDA_CHECK(cudaMalloc(&dMax, sizeof(unsigned long long) * 2));
    CUDA_CHECK(cudaMemset(dMax, 0, sizeof(unsigned long long) * 2));

    // Compute L * L^T
    dotTileKernel<<<dim3(blocksFor(n, TILE), blocksFor(n, TILE)), DOT_THREADS>>>(dL, dL, n, n, n, n, dRec, n);

    // Compare with original
    errorKernel<<<blocksFor(total, ELEM_THREADS), ELEM_THREADS>>>(dRec, dA_orig, total, dMax, dMax + 1);

    unsigned long long hMax[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(hMax, dMax, sizeof(hMax), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dRec));
    CUDA_CHECK(cudaFree(dMax));

    double maxError = 0.0;
    double relError = 0.0;
    std::memcpy(&maxError, &hMax[0], sizeof(double));
    std::memcpy(&relError, &hMax[1], sizeof(double));

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

    const int ni = static_cast<int>(n);
    const size_t total = n * n;

    // Allocate matrix on the device
    double* dA = nullptr;
    double* dA_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(double) * total));

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dA, ni);

    if (validate) {
        // Save original for validation
        CUDA_CHECK(cudaMalloc(&dA_orig, sizeof(double) * total));
        CUDA_CHECK(cudaMemcpy(dA_orig, dA, sizeof(double) * total, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(dA, ni);

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
        std::vector<double> A(total);
        CUDA_CHECK(cudaMemcpy(A.data(), dA, sizeof(double) * total, cudaMemcpyDeviceToHost));
        print_results(A, "CholeskyL");
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(dA, dA_orig, ni);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    CUDA_CHECK(cudaFree(dA));
    if (dA_orig) {
        CUDA_CHECK(cudaFree(dA_orig));
    }
    return exitCode;
}
