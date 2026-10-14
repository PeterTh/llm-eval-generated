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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// GPU tiling parameters
constexpr size_t kPanel = 64;     // Columns per panel of the factorization
constexpr size_t kTile = 64;      // Output tile edge of the GEMM kernel
constexpr size_t kTileK = 16;     // K step of the GEMM kernel
constexpr size_t kReg = 4;        // Outputs per thread along each tile edge
constexpr size_t kTrsmThreads = 128;

// C[r * ldc + c] = sum_{k=0}^{K-1} X[r * ldx + k] * Y[c * ldy + k]
// Every output is accumulated in increasing k order, exactly as the sequential
// reference does, so the tiling only reorders independent outputs, never the
// floating-point operations that produce a given value.
__global__ void gemmNTKernel(const double* __restrict__ X, size_t ldx,
                             const double* __restrict__ Y, size_t ldy,
                             double* __restrict__ C, size_t ldc,
                             size_t M, size_t N, size_t K) {
    __shared__ double Xs[kTile][kTileK + 1];
    __shared__ double Ys[kTile][kTileK + 1];

    const size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const size_t rowBase = blockIdx.y * kTile;
    const size_t colBase = blockIdx.x * kTile;

    double acc[kReg][kReg] = {};

    for (size_t k0 = 0; k0 < K; k0 += kTileK) {
        const size_t kLen = min(kTileK, K - k0);
        for (size_t e = tid; e < kTile * kTileK; e += blockDim.x * blockDim.y) {
            const size_t r = e / kTileK;
            const size_t kk = e % kTileK;
            Xs[r][kk] = (rowBase + r < M && kk < kLen) ? X[(rowBase + r) * ldx + k0 + kk] : 0.0;
            Ys[r][kk] = (colBase + r < N && kk < kLen) ? Y[(colBase + r) * ldy + k0 + kk] : 0.0;
        }
        __syncthreads();

        for (size_t kk = 0; kk < kLen; ++kk) {
            double a[kReg];
            double b[kReg];
            for (size_t x = 0; x < kReg; ++x) {
                a[x] = Xs[threadIdx.y * kReg + x][kk];
                b[x] = Ys[threadIdx.x * kReg + x][kk];
            }
            for (size_t x = 0; x < kReg; ++x) {
                for (size_t y = 0; y < kReg; ++y) {
                    acc[x][y] += a[x] * b[y];
                }
            }
        }
        __syncthreads();
    }

    for (size_t x = 0; x < kReg; ++x) {
        for (size_t y = 0; y < kReg; ++y) {
            const size_t r = rowBase + threadIdx.y * kReg + x;
            const size_t c = colBase + threadIdx.x * kReg + y;
            if (r < M && c < N) {
                C[r * ldc + c] = acc[x][y];
            }
        }
    }
}

static void gemmNT(const double* X, size_t ldx, const double* Y, size_t ldy,
                   double* C, size_t ldc, size_t M, size_t N, size_t K) {
    if (M == 0 || N == 0) {
        return;
    }
    const dim3 block(kTile / kReg, kTile / kReg);
    const dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    gemmNTKernel<<<grid, block>>>(X, ldx, Y, ldy, C, ldc, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// Factor the w x w diagonal block of the panel starting at column J.
// Columns are sequential; within one column the rows below the diagonal run in parallel.
// S holds the partial sums over k < J for every row of the panel.
__global__ void potrfDiagKernel(double* A, size_t n, size_t J, size_t w,
                                const double* S, size_t ldS,
                                unsigned long long* failIdx) {
    __shared__ double Ls[kPanel][kPanel];

    const size_t tid = threadIdx.x;
    for (size_t idx = tid; idx < w * w; idx += blockDim.x) {
        const size_t r = idx / w;
        const size_t c = idx % w;
        if (c <= r) {
            Ls[r][c] = A[(J + r) * n + J + c];
        }
    }
    __syncthreads();

    for (size_t jj = 0; jj < w; ++jj) {
        if (tid == 0) {
            double sum = S[jj * ldS + jj];
            for (size_t kk = 0; kk < jj; ++kk) {
                sum += Ls[jj][kk] * Ls[jj][kk];
            }
            const double val = Ls[jj][jj] - sum;
            if (val <= 0.0) {
                // Keep the smallest failing diagonal index, matching the sequential order
                atomicMin(failIdx, static_cast<unsigned long long>(J + jj));
            }
            Ls[jj][jj] = sqrt(val);
        }
        __syncthreads();

        for (size_t ii = jj + 1 + tid; ii < w; ii += blockDim.x) {
            double sum = S[ii * ldS + jj];
            for (size_t kk = 0; kk < jj; ++kk) {
                sum += Ls[ii][kk] * Ls[jj][kk];
            }
            Ls[ii][jj] = (Ls[ii][jj] - sum) / Ls[jj][jj];
        }
        __syncthreads();
    }

    for (size_t idx = tid; idx < w * w; idx += blockDim.x) {
        const size_t r = idx / w;
        const size_t c = idx % w;
        if (c <= r) {
            A[(J + r) * n + J + c] = Ls[r][c];
        }
    }
}

// Solve for the rows below the panel: each row is independent given the diagonal block.
__global__ void potrfTrsmKernel(double* A, size_t n, size_t J, size_t w,
                                const double* S, size_t ldS) {
    __shared__ double Ls[kPanel][kPanel];

    const size_t tid = threadIdx.x;
    for (size_t idx = tid; idx < w * w; idx += blockDim.x) {
        const size_t r = idx / w;
        const size_t c = idx % w;
        if (c <= r) {
            Ls[r][c] = A[(J + r) * n + J + c];
        }
    }
    __syncthreads();

    const size_t i = J + w + blockIdx.x * blockDim.x + tid;
    if (i >= n) {
        return;
    }

    double* row = A + i * n + J;
    const double* Srow = S + (i - J) * ldS;
    for (size_t jj = 0; jj < w; ++jj) {
        double sum = Srow[jj];
        for (size_t kk = 0; kk < jj; ++kk) {
            sum += row[kk] * Ls[jj][kk];
        }
        row[jj] = (row[jj] - sum) / Ls[jj][jj];
    }
}

__global__ void zeroStrictUpperKernel(double* A, size_t n) {
    const size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n && c > r) {
        A[r * n + c] = 0.0;
    }
}

// Sequential unblocked semantics, computed as a left-looking blocked factorization on the GPU.
// Each entry is still a sum over k in increasing order, so the results match the reference.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const size_t bytes = n * n * sizeof(double);
    double* dA = nullptr;
    double* dS = nullptr;
    unsigned long long* dFail = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dS, n * kPanel * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(unsigned long long)));

    const unsigned long long noFail = ULLONG_MAX;
    CUDA_CHECK(cudaMemcpy(dFail, &noFail, sizeof(noFail), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));

    for (size_t J = 0; J < n; J += kPanel) {
        const size_t w = std::min(kPanel, n - J);

        // Partial sums over all columns left of the panel
        gemmNT(dA + J * n, n, dA + J * n, n, dS, kPanel, n - J, w, J);
        potrfDiagKernel<<<1, kPanel>>>(dA, n, J, w, dS, kPanel, dFail);
        CUDA_CHECK(cudaGetLastError());

        const size_t rows = n - J - w;
        if (rows > 0) {
            potrfTrsmKernel<<<(rows + kTrsmThreads - 1) / kTrsmThreads, kTrsmThreads>>>(
                dA, n, J, w, dS, kPanel);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    zeroStrictUpperKernel<<<grid, block>>>(dA, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    unsigned long long failIdx = noFail;
    CUDA_CHECK(cudaMemcpy(&failIdx, dFail, sizeof(failIdx), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dFail));

    if (failIdx != noFail) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               static_cast<size_t>(failIdx));
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

    // Compute A = B * B^T on the GPU
    const size_t bytes = n * n * sizeof(double);
    double* dB = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    gemmNT(dB, n, dB, n, dA, n, n, n, n);
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);
    const size_t bytes = n * n * sizeof(double);
    double* dL = nullptr;
    double* dR = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, bytes));
    CUDA_CHECK(cudaMalloc(&dR, bytes));
    CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
    gemmNT(dL, n, dL, n, dR, n, n, n, n);
    CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dR));

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
