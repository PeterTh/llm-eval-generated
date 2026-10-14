#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA parallel Cholesky decomposition (blocked, left-looking, unblocked within each panel)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Every element keeps the sequential accumulation order of the original
// unblocked algorithm (k = 0, 1, ..., j-1). Parallelism comes from computing
// independent elements concurrently. Results agree with the CPU version to
// floating-point rounding (FMA contraction differs), not bit-for-bit.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

constexpr size_t TILE = 32;             // Block edge, k-chunk, and panel width
constexpr size_t MICRO = 2;             // Outputs per thread along each dimension
constexpr size_t BLOCK = TILE / MICRO;  // Threads per block dimension

// C[r*ldc + c] = sum_{k<K} X[r*ldx + k] * Y[c*ldy + k]
// Tiled over k, but k is still accumulated strictly in increasing order per element.
__global__ void gemmNT(const double* __restrict__ X, size_t ldx,
                       const double* __restrict__ Y, size_t ldy,
                       double* __restrict__ C, size_t ldc,
                       size_t rows, size_t cols, size_t K) {
    __shared__ double Xs[TILE][TILE + 1];
    __shared__ double Ys[TILE][TILE + 1];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t tid = ty * BLOCK + tx;
    const size_t rowBase = blockIdx.y * TILE;
    const size_t colBase = blockIdx.x * TILE;

    double acc[MICRO][MICRO] = {};

    for (size_t k0 = 0; k0 < K; k0 += TILE) {
        const size_t rem = K - k0;
        const size_t kl = rem < TILE ? rem : TILE;

        for (size_t idx = tid; idx < TILE * TILE; idx += BLOCK * BLOCK) {
            const size_t r = idx / TILE;
            const size_t kc = idx % TILE;
            const bool inK = kc < kl;
            const size_t xr = rowBase + r;
            const size_t yc = colBase + r;
            Xs[r][kc] = (inK && xr < rows) ? X[xr * ldx + k0 + kc] : 0.0;
            Ys[r][kc] = (inK && yc < cols) ? Y[yc * ldy + k0 + kc] : 0.0;
        }
        __syncthreads();

        for (size_t kk = 0; kk < kl; ++kk) {
            for (size_t a = 0; a < MICRO; ++a) {
                for (size_t b = 0; b < MICRO; ++b) {
                    acc[a][b] += Xs[ty + a * BLOCK][kk] * Ys[tx + b * BLOCK][kk];
                }
            }
        }
        __syncthreads();
    }

    for (size_t a = 0; a < MICRO; ++a) {
        for (size_t b = 0; b < MICRO; ++b) {
            const size_t r = rowBase + ty + a * BLOCK;
            const size_t c = colBase + tx + b * BLOCK;
            if (r < rows && c < cols) {
                C[r * ldc + c] = acc[a][b];
            }
        }
    }
}

// Factor the TILE x TILE diagonal block of the current panel [J0, J0 + w).
// P holds the partial sums (k < J0) for every panel element, so each element's
// sum is completed in the same order as the sequential algorithm.
__global__ void factorDiagBlock(double* A, size_t n, size_t J0, size_t w,
                                const double* P, int* status) {
    __shared__ bool stop;
    if (threadIdx.x == 0) {
        stop = (*status != 0);
    }
    __syncthreads();
    if (stop) return;

    for (size_t jl = 0; jl < w; ++jl) {
        const size_t j = J0 + jl;

        if (threadIdx.x == 0) {
            double sum = P[jl * TILE + jl];
            for (size_t k = J0; k < j; ++k) {
                sum += A[j * n + k] * A[j * n + k];
            }
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite; record the first failing index
                *status = static_cast<int>(j) + 1;
            } else {
                A[j * n + j] = sqrt(val);
            }
        }
        __syncthreads();
        if (*status) return;

        for (size_t i = j + 1 + threadIdx.x; i < J0 + w; i += blockDim.x) {
            double sum = P[(i - J0) * TILE + jl];
            for (size_t k = J0; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
        __syncthreads();
    }
}

// Rows below the panel are independent of each other, so each thread
// finishes one row across all panel columns.
__global__ void factorBelowPanel(double* A, size_t n, size_t J0, size_t w,
                                 const double* P, const int* status) {
    const size_t i = J0 + w + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || *status) return;

    for (size_t jl = 0; jl < w; ++jl) {
        const size_t j = J0 + jl;
        double sum = P[(i - J0) * TILE + jl];
        for (size_t k = J0; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
    }
}

__global__ void zeroUpper(double* A, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    const size_t i = idx / n;
    const size_t j = idx % n;
    if (j > i) {
        A[idx] = 0.0;
    }
}

static void launchGemmNT(const double* dX, size_t ldx, const double* dY, size_t ldy,
                         double* dC, size_t ldc, size_t rows, size_t cols, size_t K) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    gemmNT<<<grid, block>>>(dX, ldx, dY, ldy, dC, ldc, rows, cols, K);
    CUDA_CHECK(cudaGetLastError());
}

// Returns X * Y^T for n x n row-major matrices, computed on the GPU
std::vector<double> multiplyNT(const std::vector<double>& X, const std::vector<double>& Y, const size_t n) {
    std::vector<double> C(n * n);
    if (n == 0) return C;

    const size_t bytes = n * n * sizeof(double);
    double *dX, *dY, *dC;
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, Y.data(), bytes, cudaMemcpyHostToDevice));

    launchGemmNT(dX, n, dY, n, dC, n, n, n, n);

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dC));
    return C;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;

    const size_t bytes = n * n * sizeof(double);
    double* dA;
    double* dP;
    int* dStatus;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dP, n * TILE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dStatus, sizeof(int)));
    CUDA_CHECK(cudaMemset(dStatus, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));

    for (size_t J0 = 0; J0 < n; J0 += TILE) {
        const size_t J1 = std::min(J0 + TILE, n);
        const size_t w = J1 - J0;

        // Partial sums over k < J0 for all rows at or below J0 in this panel
        launchGemmNT(dA + J0 * n, n, dA + J0 * n, n, dP, TILE, n - J0, w, J0);

        // Sequential work on the diagonal block of the panel
        factorDiagBlock<<<1, TILE>>>(dA, n, J0, w, dP, dStatus);
        CUDA_CHECK(cudaGetLastError());

        // Parallel work on all rows below the panel
        if (J1 < n) {
            const size_t threads = 256;
            const size_t blocks = (n - J1 + threads - 1) / threads;
            factorBelowPanel<<<blocks, threads>>>(dA, n, J0, w, dP, dStatus);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    const size_t zThreads = 256;
    const size_t zBlocks = (n * n + zThreads - 1) / zThreads;
    zeroUpper<<<zBlocks, zThreads>>>(dA, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    int status = 0;
    CUDA_CHECK(cudaMemcpy(&status, dStatus, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dStatus));

    if (status != 0) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               static_cast<size_t>(status - 1));
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

    // Compute A = B * B^T
    A = multiplyNT(B, B, n);

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    // Compute L * L^T
    const std::vector<double> reconstructed = multiplyNT(L, L, n);

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

    // Generate positive definite matrix (GPU initializes the CUDA context here, outside timing)
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
