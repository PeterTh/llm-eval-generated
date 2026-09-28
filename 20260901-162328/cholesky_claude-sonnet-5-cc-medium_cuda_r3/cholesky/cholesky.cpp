#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition parallelized with CUDA.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// The matrix is kept resident on the GPU for the entire factorization; only the
// small diagonal-block "not positive definite" flag is round-tripped to the host.

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err__));                                     \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// Block size used for the panel/trailing-update decomposition. Must fit the
// static shared-memory tiles declared in the kernels below.
static constexpr int kBlockSize = 64;
static constexpr int kUpdateTile = 32;

// Factorizes a single diagSize x diagSize diagonal block in place (unblocked
// Cholesky). Runs as a single thread block; columns are processed serially
// but each column's updates are spread across the block's threads.
__global__ void factorizeDiagBlockKernel(double* A, size_t n, size_t diagOff, size_t diagSize,
                                          int* errorFlag, unsigned long long* errorIndex) {
    __shared__ double tile[kBlockSize][kBlockSize];
    __shared__ int localError;

    const int tid = threadIdx.x;
    const int nThreads = blockDim.x;

    for (int idx = tid; idx < (int)(diagSize * diagSize); idx += nThreads) {
        size_t r = idx / diagSize;
        size_t c = idx % diagSize;
        tile[r][c] = A[(diagOff + r) * n + diagOff + c];
    }
    if (tid == 0) localError = 0;
    __syncthreads();

    for (size_t j = 0; j < diagSize; ++j) {
        if (tid == 0) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += tile[j][k] * tile[j][k];
            const double val = tile[j][j] - sum;
            if (val <= 0.0) {
                localError = 1;
            } else {
                tile[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        if (localError) {
            if (tid == 0) {
                *errorFlag = 1;
                *errorIndex = (unsigned long long)(diagOff + j);
            }
            return;
        }
        const double diagVal = tile[j][j];
        for (size_t i = j + 1 + tid; i < diagSize; i += nThreads) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += tile[i][k] * tile[j][k];
            tile[i][j] = (tile[i][j] - sum) / diagVal;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < (int)(diagSize * diagSize); idx += nThreads) {
        size_t r = idx / diagSize;
        size_t c = idx % diagSize;
        A[(diagOff + r) * n + diagOff + c] = (c <= r) ? tile[r][c] : 0.0;
    }
}

// Solves L_diag * row^T = A_row^T for every row below the current diagonal
// block (forward substitution), one row per thread.
__global__ void triangularSolveKernel(double* A, size_t n, size_t diagOff, size_t diagSize,
                                       size_t panelOff, size_t panelRows) {
    __shared__ double diagBlock[kBlockSize][kBlockSize];

    const int tid = threadIdx.x;
    for (int idx = tid; idx < (int)(diagSize * diagSize); idx += blockDim.x) {
        size_t r = idx / diagSize;
        size_t c = idx % diagSize;
        diagBlock[r][c] = A[(diagOff + r) * n + diagOff + c];
    }
    __syncthreads();

    const size_t row = panelOff + (size_t)blockIdx.x * blockDim.x + tid;
    if (row >= panelOff + panelRows) return;

    for (size_t jj = 0; jj < diagSize; ++jj) {
        double sum = 0.0;
        for (size_t k = 0; k < jj; ++k) sum += A[row * n + diagOff + k] * diagBlock[jj][k];
        A[row * n + diagOff + jj] = (A[row * n + diagOff + jj] - sum) / diagBlock[jj][jj];
    }
}

// Applies the rank-diagSize trailing update C -= P * P^T to the lower
// triangle of the trailing panelRows x panelRows submatrix, tiled through
// shared memory (P is the panelRows x diagSize block just solved above).
__global__ void updateTrailingKernel(double* A, size_t n, size_t panelOff, size_t diagSize,
                                      size_t panelRows) {
    __shared__ double sA[kUpdateTile][kBlockSize];
    __shared__ double sB[kUpdateTile][kBlockSize];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t rowBase = panelOff + (size_t)blockIdx.y * kUpdateTile;
    const size_t colBase = panelOff + (size_t)blockIdx.x * kUpdateTile;
    const size_t n_total = panelOff + panelRows;

    // Skip tiles that lie entirely in the strict upper triangle.
    if (colBase >= rowBase + kUpdateTile) return;

    const size_t row = rowBase + ty;
    const size_t col = colBase + tx;
    const size_t kOff = panelOff - diagSize;

    for (int k = tx; k < (int)diagSize; k += kUpdateTile)
        sA[ty][k] = (row < n_total) ? A[row * n + kOff + k] : 0.0;
    for (int k = ty; k < (int)diagSize; k += kUpdateTile)
        sB[tx][k] = (col < n_total) ? A[col * n + kOff + k] : 0.0;
    __syncthreads();

    if (row < n_total && col < n_total && row >= col) {
        double sum = 0.0;
        for (size_t k = 0; k < diagSize; ++k) sum += sA[ty][k] * sB[tx][k];
        A[row * n + col] -= sum;
    }
}

__global__ void zeroUpperTriangleKernel(double* A, size_t n) {
    const size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n && col > row) A[row * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    double* d_A = nullptr;
    int* d_errorFlag = nullptr;
    unsigned long long* d_errorIndex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_errorFlag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_errorIndex, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    bool posDef = true;
    size_t errIdx = 0;

    for (size_t off = 0; off < n; off += kBlockSize) {
        const size_t diagSize = std::min((size_t)kBlockSize, n - off);

        const int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_errorFlag, &zero, sizeof(int), cudaMemcpyHostToDevice));
        factorizeDiagBlockKernel<<<1, kBlockSize>>>(d_A, n, off, diagSize, d_errorFlag, d_errorIndex);

        int hostFlag = 0;
        CUDA_CHECK(cudaMemcpy(&hostFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost));
        if (hostFlag) {
            unsigned long long hostIdx = 0;
            CUDA_CHECK(cudaMemcpy(&hostIdx, d_errorIndex, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            errIdx = (size_t)hostIdx;
            posDef = false;
            break;
        }

        const size_t panelOff = off + diagSize;
        const size_t panelRows = n - panelOff;
        if (panelRows > 0) {
            const int threads = 128;
            const int blocks = (int)((panelRows + threads - 1) / threads);
            triangularSolveKernel<<<blocks, threads>>>(d_A, n, off, diagSize, panelOff, panelRows);

            const dim3 tblock(kUpdateTile, kUpdateTile);
            const dim3 tgrid((unsigned)((panelRows + kUpdateTile - 1) / kUpdateTile),
                              (unsigned)((panelRows + kUpdateTile - 1) / kUpdateTile));
            updateTrailingKernel<<<tgrid, tblock>>>(d_A, n, panelOff, diagSize, panelRows);
        }
    }

    if (posDef) {
        const dim3 zblock(32, 32);
        const dim3 zgrid((unsigned)((n + 31) / 32), (unsigned)((n + 31) / 32));
        zeroUpperTriangleKernel<<<zgrid, zblock>>>(d_A, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_A);
    cudaFree(d_errorFlag);
    cudaFree(d_errorIndex);

    if (!posDef) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", errIdx);
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
