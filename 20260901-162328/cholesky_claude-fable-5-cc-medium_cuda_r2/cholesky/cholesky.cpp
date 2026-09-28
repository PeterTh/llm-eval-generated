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

// CUDA-parallelized Cholesky decomposition (blocked right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

constexpr int NB = 64;    // panel (block) width
constexpr int TILE = 32;  // tile size for GEMM-like kernels

// Factor the NB x NB diagonal block starting at (k, k) in place (single thread block).
// Records the first non-positive-definite diagonal index in errFlag (initialized to INT_MAX).
__global__ void potrfDiagKernel(double* A, size_t n, size_t k, int nb, int* errFlag) {
    __shared__ double s[NB][NB + 1];
    const int t = threadIdx.x;

    if (t < nb) {
        for (int j = 0; j <= t; ++j) {
            s[t][j] = A[(k + t) * n + (k + j)];
        }
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        if (t == j) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                sum += s[j][p] * s[j][p];
            }
            double val = s[j][j] - sum;
            if (val <= 0.0) {
                atomicMin(errFlag, (int)(k + j));
                val = 1.0;  // keep computing; failure is reported on the host
            }
            s[j][j] = sqrt(val);
        }
        __syncthreads();
        if (t > j && t < nb) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                sum += s[t][p] * s[j][p];
            }
            s[t][j] = (s[t][j] - sum) / s[j][j];
        }
        __syncthreads();
    }

    if (t < nb) {
        for (int j = 0; j <= t; ++j) {
            A[(k + t) * n + (k + j)] = s[t][j];
        }
    }
}

// Triangular solve for the panel below the diagonal block:
// A[k+nb : n, k : k+nb] = A[k+nb : n, k : k+nb] * L11^{-T}, one thread per row.
__global__ void trsmPanelKernel(double* A, size_t n, size_t k, int nb) {
    __shared__ double L[NB][NB + 1];
    const int tid = threadIdx.x;

    for (int idx = tid; idx < nb * nb; idx += blockDim.x) {
        const int r = idx / nb;
        const int c = idx % nb;
        L[r][c] = A[(k + r) * n + (k + c)];
    }
    __syncthreads();

    const size_t row = k + nb + (size_t)blockIdx.x * blockDim.x + tid;
    if (row >= n) return;

    double x[NB];
    double* a = &A[row * n + k];
    for (int j = 0; j < nb; ++j) {
        x[j] = a[j];
    }
    for (int j = 0; j < nb; ++j) {
        double sum = x[j];
        for (int p = 0; p < j; ++p) {
            sum -= x[p] * L[j][p];
        }
        x[j] = sum / L[j][j];
    }
    for (int j = 0; j < nb; ++j) {
        a[j] = x[j];
    }
}

// Trailing submatrix update (SYRK): A22 -= L21 * L21^T, lower triangle only.
// The trailing matrix starts at (k+nb, k+nb) and has size m x m.
__global__ void syrkUpdateKernel(double* A, size_t n, size_t k, int nb, size_t m) {
    const size_t by = (size_t)blockIdx.y * TILE;  // row tile within trailing matrix
    const size_t bx = (size_t)blockIdx.x * TILE;  // column tile
    if (bx > by + (TILE - 1)) return;             // tile entirely above the diagonal

    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t base = k + nb;

    double acc = 0.0;
    for (int kk = 0; kk < nb; kk += TILE) {
        const size_t ar = by + ty;
        const size_t br = bx + ty;
        As[ty][tx] = (ar < m && kk + tx < nb) ? A[(base + ar) * n + k + kk + tx] : 0.0;
        Bs[ty][tx] = (br < m && kk + tx < nb) ? A[(base + br) * n + k + kk + tx] : 0.0;
        __syncthreads();
#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            acc += As[ty][p] * Bs[tx][p];
        }
        __syncthreads();
    }

    const size_t row = by + ty;
    const size_t col = bx + tx;
    if (row < m && col <= row) {
        A[(base + row) * n + (base + col)] -= acc;
    }
}

// Zero out the strictly upper triangular part.
__global__ void zeroUpperKernel(double* A, size_t n) {
    const size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}

// C = B * B^T (full matrix), used for matrix generation and validation.
__global__ void aatKernel(const double* B, double* C, size_t n) {
    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t row = (size_t)blockIdx.y * TILE + ty;
    const size_t col = (size_t)blockIdx.x * TILE + tx;

    double acc = 0.0;
    for (size_t kk = 0; kk < n; kk += TILE) {
        const size_t cb = (size_t)blockIdx.x * TILE + ty;
        As[ty][tx] = (row < n && kk + tx < n) ? B[row * n + kk + tx] : 0.0;
        Bs[ty][tx] = (cb < n && kk + tx < n) ? B[cb * n + kk + tx] : 0.0;
        __syncthreads();
#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            acc += As[ty][p] * Bs[tx][p];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* dA = nullptr;
    int* dErr = nullptr;
    const int errInit = INT_MAX;

    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dErr, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dErr, &errInit, sizeof(int), cudaMemcpyHostToDevice));

    for (size_t k = 0; k < n; k += NB) {
        const int nb = (int)std::min((size_t)NB, n - k);

        potrfDiagKernel<<<1, NB>>>(dA, n, k, nb, dErr);

        const size_t m = n - k - nb;  // rows below the diagonal block
        if (m > 0) {
            const int threads = 256;
            const int blocks = (int)((m + threads - 1) / threads);
            trsmPanelKernel<<<blocks, threads>>>(dA, n, k, nb);

            const unsigned int tiles = (unsigned int)((m + TILE - 1) / TILE);
            dim3 grid(tiles, tiles);
            dim3 block(TILE, TILE);
            syrkUpdateKernel<<<grid, block>>>(dA, n, k, nb, m);
        }
    }

    {
        const unsigned int tiles = (unsigned int)((n + TILE - 1) / TILE);
        dim3 grid(tiles, tiles);
        dim3 block(TILE, TILE);
        zeroUpperKernel<<<grid, block>>>(dA, n);
    }

    int errIdx = INT_MAX;
    CUDA_CHECK(cudaMemcpy(&errIdx, dErr, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dErr));

    if (errIdx != INT_MAX) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               (size_t)errIdx);
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
    double* dB = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int tiles = (unsigned int)((n + TILE - 1) / TILE);
    dim3 grid(tiles, tiles);
    dim3 block(TILE, TILE);
    aatKernel<<<grid, block>>>(dB, dA, n);

    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());
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

    // Compute L * L^T on the GPU
    double* dL = nullptr;
    double* dR = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dL, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int tiles = (unsigned int)((n + TILE - 1) / TILE);
    dim3 grid(tiles, tiles);
    dim3 block(TILE, TILE);
    aatKernel<<<grid, block>>>(dL, dR, n);

    CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());
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
