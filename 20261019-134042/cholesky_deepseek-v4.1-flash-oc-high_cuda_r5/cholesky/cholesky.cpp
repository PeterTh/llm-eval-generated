#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// GPU Cholesky decomposition (blocked right-looking algorithm).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is stored in row-major order.  The decomposition is split into
// panels of CH_BLOCK columns.  For every panel three steps are executed on the
// GPU:
//   1. factorization of the diagonal block,
//   2. triangular solve for the panel below the diagonal block,
//   3. rank-CH_BLOCK update of the trailing sub-matrix.
// Steps 1 and 2 expose parallelism over rows while step 3 is a tiled
// triangular GEMM which contains the bulk of the floating point work.

namespace {

constexpr int CH_BLOCK = 32;    // panel width
constexpr int PANEL_ROWS = 64;  // rows handled by one panel-solve block
constexpr int TR_TILE = 32;     // trailing-update tile size

// Zero the strictly upper triangular part of the matrix.
__global__ void zeroUpperKernel(double* A, int n) {
    const int i = blockIdx.x;
    if (i >= n) {
        return;
    }
    for (int j = i + 1 + threadIdx.x; j < n; j += blockDim.x) {
        A[(size_t)i * n + j] = 0.0;
    }
}

// Factor the b x b diagonal block that starts at column s.
// The block is kept in shared memory and factorized right-looking: after a
// column is completed, the remaining lower sub-block is updated with an outer
// product.  Keeping the block in shared memory avoids the long dependent dot
// products of the unblocked formulation.
__global__ void factorDiagonalKernel(double* A, int n, int s, int b, int* flag) {
    __shared__ double S[CH_BLOCK][CH_BLOCK + 1];

    const int tid = threadIdx.x;

    // Load the lower triangle of the block into shared memory.
    for (int idx = tid; idx < b * b; idx += blockDim.x) {
        const int r = idx / b;
        const int c = idx % b;
        if (c <= r) {
            S[r][c] = A[(size_t)(s + r) * n + (s + c)];
        }
    }
    __syncthreads();

    for (int j = 0; j < b; ++j) {
        if (tid == 0) {
            const double val = S[j][j];
            double diag;
            if (val <= 0.0) {
                atomicMin(flag, s + j);
                diag = 1.0;
            } else {
                diag = sqrt(val);
            }
            S[j][j] = diag;
        }
        __syncthreads();

        const double inv = 1.0 / S[j][j];
        for (int i = j + 1 + tid; i < b; i += blockDim.x) {
            S[i][j] *= inv;
        }
        __syncthreads();

        for (int i = j + 1 + tid; i < b; i += blockDim.x) {
            const double f = S[i][j];
            for (int l = j + 1; l <= i; ++l) {
                S[i][l] -= f * S[l][j];
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < b * b; idx += blockDim.x) {
        const int r = idx / b;
        const int c = idx % b;
        if (c <= r) {
            A[(size_t)(s + r) * n + (s + c)] = S[r][c];
        }
    }
}

// Solve the panel rows below the diagonal block: rows [s + b, n), columns [s, s + b).
// Every row is independent, so one thread handles one row and no block-wide
// synchronization is required.
__global__ void panelSolveKernel(double* A, int n, int s, int b) {
    __shared__ double R[PANEL_ROWS][CH_BLOCK + 1];
    __shared__ double D[CH_BLOCK][CH_BLOCK + 1];

    const int tid = threadIdx.x;
    const int row = s + b + blockIdx.x * PANEL_ROWS + tid;
    const bool valid = row < n;

    // Cache the factorized diagonal block (broadcast to all rows).
    for (int idx = tid; idx < b * b; idx += blockDim.x) {
        const int r = idx / b;
        const int c = idx % b;
        if (c <= r) {
            D[r][c] = A[(size_t)(s + r) * n + (s + c)];
        }
    }
    __syncthreads();

    if (valid) {
        for (int k = 0; k < b; ++k) {
            R[tid][k] = A[(size_t)row * n + s + k];
        }

        for (int j = 0; j < b; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += R[tid][k] * D[j][k];
            }
            R[tid][j] = (R[tid][j] - sum) / D[j][j];
        }

        for (int k = 0; k < b; ++k) {
            A[(size_t)row * n + s + k] = R[tid][k];
        }
    }
}

// Rank-b update of the trailing lower-triangular sub-matrix, starting at t0.
// blockIdx.x enumerates the lower-triangular tile pairs, so no idle upper
// blocks are launched.
__global__ void trailingUpdateKernel(double* A, int n, int s, int b, int t0) {
    const int p = blockIdx.x;
    int ti = static_cast<int>((sqrt(8.0 * p + 1.0) - 1.0) * 0.5);
    while (ti * (ti + 1) / 2 > p) {
        --ti;
    }
    while ((ti + 1) * (ti + 2) / 2 <= p) {
        ++ti;
    }
    const int tj = p - ti * (ti + 1) / 2;

    const int i0 = t0 + ti * TR_TILE;
    const int j0 = t0 + tj * TR_TILE;

    __shared__ double Li[TR_TILE][CH_BLOCK + 1];
    __shared__ double Lj[TR_TILE][CH_BLOCK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TR_TILE + tx;

    for (int idx = tid; idx < TR_TILE * b; idx += TR_TILE * TR_TILE) {
        const int rr = idx / b;
        const int kk = idx % b;
        const int gi = i0 + rr;
        const int gj = j0 + rr;
        const int gk = s + kk;
        Li[rr][kk] = (gi < n) ? A[(size_t)gi * n + gk] : 0.0;
        Lj[rr][kk] = (gj < n) ? A[(size_t)gj * n + gk] : 0.0;
    }
    __syncthreads();

    const int gi = i0 + ty;
    const int gj = j0 + tx;
    if (gi < n && gj < n && gi >= gj) {
        double acc = 0.0;
        for (int k = 0; k < b; ++k) {
            acc += Li[ty][k] * Lj[tx][k];
        }
        A[(size_t)gi * n + gj] -= acc;
    }
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const int ni = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);

    double* dA = nullptr;
    int* dflag = nullptr;
    cudaError_t err = cudaMalloc(&dA, bytes);
    if (err == cudaSuccess) {
        err = cudaMalloc(&dflag, sizeof(int));
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        if (dA) cudaFree(dA);
        if (dflag) cudaFree(dflag);
        return false;
    }

    // flag is -1 while the matrix remains positive definite.
    cudaMemset(dflag, 0xFF, sizeof(int));
    err = cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(dA);
        cudaFree(dflag);
        return false;
    }

    zeroUpperKernel<<<ni, 256>>>(dA, ni);

    for (int s = 0; s < ni; s += CH_BLOCK) {
        const int b = std::min(CH_BLOCK, ni - s);
        factorDiagonalKernel<<<1, 2 * CH_BLOCK>>>(dA, ni, s, b, dflag);

        const int t0 = s + b;
        if (t0 < ni) {
            const int panelRows = ni - t0;
            const int pblocks = (panelRows + PANEL_ROWS - 1) / PANEL_ROWS;
            panelSolveKernel<<<pblocks, PANEL_ROWS>>>(dA, ni, s, b);

            const int tiles = (panelRows + TR_TILE - 1) / TR_TILE;
            const int triTiles = tiles * (tiles + 1) / 2;
            const dim3 block(TR_TILE, TR_TILE);
            trailingUpdateKernel<<<triTiles, block>>>(dA, ni, s, b, t0);
        }
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(dA);
        cudaFree(dflag);
        return false;
    }

    int hflag = -1;
    err = cudaMemcpy(&hflag, dflag, sizeof(int), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        err = cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(dA);
    cudaFree(dflag);

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return false;
    }

    if (hflag >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", hflag);
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

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");

    // Initialize the CUDA context and warm up every kernel ahead of the timed
    // region so that device/module startup is not attributed to the
    // decomposition itself.
    cudaFree(nullptr);
    {
        const size_t wn = 2 * CH_BLOCK + 1;
        std::vector<double> warm(wn * wn, 0.0);
        for (size_t i = 0; i < wn; ++i) {
            warm[i * wn + i] = 1.0;
        }
        choleskyDecomposition(warm, wn);
    }

    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %lld ms\n", (long long)(duration.count() * 1000.0));

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / duration.count() / 1e9;
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
