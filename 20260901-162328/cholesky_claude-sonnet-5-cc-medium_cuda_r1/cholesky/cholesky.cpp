#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition (right-looking, Cholesky-Crout) parallelized with CUDA.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// Only the lower triangular part of A is ever read or written until the final
// zeroing pass, matching the semantics of the original sequential algorithm.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                          \
    } while (0)

// Block size used for diagonal-block factorization and panel solve.
static constexpr int NB = 32;
// Tile size used for the trailing-matrix rank-NB update and the generic X*X^T GEMM.
static constexpr int TILE = 16;

// Factorizes the cur x cur diagonal block A[k:k+cur, k:k+cur] in place (lower
// triangular Cholesky-Crout), using a single thread block with `cur` threads.
__global__ void factorizeDiagonalBlockKernel(double* A, size_t n, size_t k, int cur,
                                              int* failFlag, unsigned long long* failIndex) {
    extern __shared__ double s[]; // cur x cur, row-major

    const int tid = threadIdx.x;

    for (int c = 0; c < cur; ++c) {
        s[tid * cur + c] = A[(k + tid) * n + (k + c)];
    }
    __syncthreads();

    for (int j = 0; j < cur; ++j) {
        if (tid == j) {
            double sum = 0.0;
            for (int kk = 0; kk < j; ++kk) {
                const double v = s[j * cur + kk];
                sum += v * v;
            }
            const double val = s[j * cur + j] - sum;
            if (val <= 0.0) {
                atomicCAS((unsigned long long*)failIndex, (unsigned long long)-1,
                          (unsigned long long)(k + j));
                *failFlag = 1;
                s[j * cur + j] = 0.0;
            } else {
                s[j * cur + j] = sqrt(val);
            }
        }
        __syncthreads();
        if (*failFlag) {
            break;
        }
        if (tid > j) {
            double sum = 0.0;
            for (int kk = 0; kk < j; ++kk) {
                sum += s[tid * cur + kk] * s[j * cur + kk];
            }
            s[tid * cur + j] = (s[tid * cur + j] - sum) / s[j * cur + j];
        }
        __syncthreads();
    }

    if (!*failFlag) {
        for (int c = 0; c <= tid; ++c) {
            A[(k + tid) * n + (k + c)] = s[tid * cur + c];
        }
    }
}

// Solves the panel below the diagonal block: for each row of A[k+cur:n, k:k+cur],
// forward-substitutes against the (already factorized) lower-triangular diagonal
// block so that A[i][k:k+cur] * Ldiag^-T is stored back in place.
__global__ void panelSolveKernel(double* A, size_t n, size_t k, int cur, size_t rows) {
    __shared__ double Ldiag[NB][NB];

    const int tid = threadIdx.x;
    if (tid < cur) {
        for (int c = 0; c <= tid; ++c) {
            Ldiag[tid][c] = A[(k + tid) * n + (k + c)];
        }
    }
    __syncthreads();

    const size_t r = (size_t)blockIdx.x * NB + tid;
    if (r < rows) {
        const size_t globalRow = k + cur + r;
        double a[NB];
        double x[NB];
        for (int c = 0; c < cur; ++c) {
            a[c] = A[globalRow * n + (k + c)];
        }
        for (int c = 0; c < cur; ++c) {
            double sum = 0.0;
            for (int kk = 0; kk < c; ++kk) {
                sum += x[kk] * Ldiag[c][kk];
            }
            x[c] = (a[c] - sum) / Ldiag[c][c];
        }
        for (int c = 0; c < cur; ++c) {
            A[globalRow * n + (k + c)] = x[c];
        }
    }
}

// Updates the trailing submatrix: A[k+cur:n, k+cur:n] -= L21 * L21^T, where L21 is
// the just-solved panel of width `cur`. Only the lower-triangular part is computed.
__global__ void trailingUpdateKernel(double* A, size_t n, size_t k, int cur, size_t rows) {
    __shared__ double sA[TILE][NB];
    __shared__ double sB[TILE][NB];

    const int bi = blockIdx.y;
    const int bj = blockIdx.x;
    if (bi < bj) return; // only lower-triangular tiles

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const size_t li = (size_t)bi * TILE + ty; // local row within panel/trailing block
    const size_t lj = (size_t)bj * TILE + tx; // local col within panel/trailing block

    for (int c = tx; c < cur; c += TILE) {
        sA[ty][c] = (li < rows) ? A[(k + cur + li) * n + (k + c)] : 0.0;
    }
    for (int c = ty; c < cur; c += TILE) {
        sB[tx][c] = (lj < rows) ? A[(k + cur + lj) * n + (k + c)] : 0.0;
    }
    __syncthreads();

    if (li < rows && lj < rows) {
        const size_t gi = k + cur + li;
        const size_t gj = k + cur + lj;
        if (gi >= gj) {
            double sum = 0.0;
            for (int kk = 0; kk < cur; ++kk) {
                sum += sA[ty][kk] * sB[tx][kk];
            }
            A[gi * n + gj] -= sum;
        }
    }
}

__global__ void zeroUpperKernel(double* A, size_t n) {
    const size_t i = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}

// Generic tiled GEMM computing C = X * X^T for an n x n row-major matrix X.
__global__ void gemmXXTKernel(const double* X, double* C, size_t n) {
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    const size_t row = (size_t)blockIdx.y * TILE + threadIdx.y;
    const size_t col = (size_t)blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;
    const size_t numTiles = (n + TILE - 1) / TILE;
    for (size_t t = 0; t < numTiles; ++t) {
        const size_t aCol = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? X[row * n + aCol] : 0.0;

        const size_t bCol = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (col < n && bCol < n) ? X[col * n + bCol] : 0.0;

        __syncthreads();
        for (int kk = 0; kk < TILE; ++kk) {
            sum += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Computes out = X * X^T on the GPU.
static void gpuMatXXT(const std::vector<double>& X, std::vector<double>& out, size_t n) {
    if (n == 0) {
        out.clear();
        return;
    }
    double *d_X = nullptr, *d_C = nullptr;
    const size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_X, X.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE);
    const dim3 grid((unsigned)((n + TILE - 1) / TILE), (unsigned)((n + TILE - 1) / TILE));
    gemmXXTKernel<<<grid, block>>>(d_X, d_C, n);
    CUDA_CHECK(cudaGetLastError());

    out.resize(n * n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    double* d_A = nullptr;
    const size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));

    int* d_failFlag = nullptr;
    unsigned long long* d_failIndex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_failFlag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failIndex, sizeof(unsigned long long)));
    const unsigned long long initIndex = (unsigned long long)-1;
    CUDA_CHECK(cudaMemcpy(d_failIndex, &initIndex, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    bool success = true;

    for (size_t k = 0; k < n; k += NB) {
        const int cur = (int)std::min<size_t>(NB, n - k);

        CUDA_CHECK(cudaMemset(d_failFlag, 0, sizeof(int)));
        const size_t sharedBytes = (size_t)cur * cur * sizeof(double);
        factorizeDiagonalBlockKernel<<<1, cur, sharedBytes>>>(d_A, n, k, cur, d_failFlag, d_failIndex);
        CUDA_CHECK(cudaGetLastError());

        int hostFailFlag = 0;
        CUDA_CHECK(cudaMemcpy(&hostFailFlag, d_failFlag, sizeof(int), cudaMemcpyDeviceToHost));
        if (hostFailFlag) {
            unsigned long long hostFailIndex = 0;
            CUDA_CHECK(cudaMemcpy(&hostFailIndex, d_failIndex, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            printf("Error: Matrix is not positive definite at diagonal element %llu\n", hostFailIndex);
            success = false;
            break;
        }

        const size_t rows = n - (k + cur);
        if (rows > 0) {
            const unsigned numRowTiles = (unsigned)((rows + NB - 1) / NB);
            panelSolveKernel<<<numRowTiles, NB>>>(d_A, n, k, cur, rows);
            CUDA_CHECK(cudaGetLastError());

            const unsigned numTiles = (unsigned)((rows + TILE - 1) / TILE);
            const dim3 grid(numTiles, numTiles);
            const dim3 block(TILE, TILE);
            trailingUpdateKernel<<<grid, block>>>(d_A, n, k, cur, rows);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (success) {
        const dim3 block(32, 32);
        const dim3 grid((unsigned)((n + 31) / 32), (unsigned)((n + 31) / 32));
        zeroUpperKernel<<<grid, block>>>(d_A, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_failFlag));
    CUDA_CHECK(cudaFree(d_failIndex));

    return success;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential to preserve deterministic output)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T on the GPU
    gpuMatXXT(B, A, n);

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed;
    gpuMatXXT(L, reconstructed, n);

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
