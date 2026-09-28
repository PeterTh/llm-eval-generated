#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition on the GPU
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(err_));                                     \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// Panel/block size for the blocked factorization
constexpr int NB = 64;
// Tile size for the trailing-matrix update kernel
constexpr int TILE = 32;

// Factorize the nb x nb diagonal block starting at (k, k) in place.
// Runs as a single thread block with NB threads. On a non-positive-definite
// pivot, records the global diagonal index (+1) in *info and aborts the block.
__global__ void potf2Kernel(double* A, size_t n, size_t k, int nb, int* info) {
    __shared__ double s[NB][NB];
    __shared__ int failed;
    const int tx = threadIdx.x;

    if (tx == 0) {
        failed = 0;
    }
    // Load block into shared memory (coalesced: tx indexes columns)
    for (int i = 0; i < nb; ++i) {
        if (tx < nb) {
            s[i][tx] = A[(k + i) * n + k + tx];
        }
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        if (tx == 0) {
            const double val = s[j][j];
            if (val <= 0.0) {
                *info = (int)(k + j) + 1;
                failed = 1;
            } else {
                s[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        if (failed) {
            return;
        }
        // Scale column j
        if (tx > j && tx < nb) {
            s[tx][j] /= s[j][j];
        }
        __syncthreads();
        // Rank-1 update of the trailing block
        if (tx > j && tx < nb) {
            const double lij = s[tx][j];
            for (int t = j + 1; t <= tx; ++t) {
                s[tx][t] -= lij * s[t][j];
            }
        }
        __syncthreads();
    }

    // Store the factored block back (lower triangle; upper is zeroed later)
    for (int i = 0; i < nb; ++i) {
        if (tx < nb) {
            A[(k + i) * n + k + tx] = s[i][tx];
        }
    }
}

// Panel solve: A21 = A21 * L11^{-T}, where L11 is the nb x nb factored
// diagonal block at (k, k) and A21 is the m x nb panel below it.
// One thread per panel row.
__global__ void trsmKernel(double* A, size_t n, size_t k, int nb, size_t m) {
    __shared__ double L[NB][NB];
    for (int idx = threadIdx.x; idx < nb * nb; idx += blockDim.x) {
        L[idx / nb][idx % nb] = A[(k + idx / nb) * n + k + idx % nb];
    }
    __syncthreads();

    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) {
        return;
    }
    double* row = A + (k + nb + i) * n + k;

    double r[NB];
    for (int t = 0; t < nb; ++t) {
        r[t] = row[t];
    }
    for (int j = 0; j < nb; ++j) {
        double sum = r[j];
        for (int t = 0; t < j; ++t) {
            sum -= r[t] * L[j][t];
        }
        r[j] = sum / L[j][j];
    }
    for (int t = 0; t < nb; ++t) {
        row[t] = r[t];
    }
}

// Trailing-matrix update (SYRK): A22 -= P * P^T, where P is the m x nb panel
// at (k + nb, k) and A22 is the m x m trailing matrix at (k + nb, k + nb).
// Only the lower triangle of A22 is updated. Shared-memory tiled.
__global__ void syrkKernel(double* A, size_t n, size_t k, int nb, size_t m) {
    const size_t bx = blockIdx.x;
    const size_t by = blockIdx.y;
    if (by < bx) {
        return; // strictly above the diagonal: nothing to update
    }
    __shared__ double Ps[TILE][TILE + 1];
    __shared__ double Qs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t off = k + nb;
    const size_t i = by * TILE + ty; // row within trailing matrix
    const size_t j = bx * TILE + tx; // column within trailing matrix

    double acc = 0.0;
    for (int kk = 0; kk < nb; kk += TILE) {
        Ps[ty][tx] = (by * TILE + ty < m && kk + tx < nb)
                         ? A[(off + by * TILE + ty) * n + k + kk + tx]
                         : 0.0;
        Qs[ty][tx] = (bx * TILE + ty < m && kk + tx < nb)
                         ? A[(off + bx * TILE + ty) * n + k + kk + tx]
                         : 0.0;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            acc += Ps[ty][t] * Qs[tx][t];
        }
        __syncthreads();
    }

    if (i < m && j < m && i >= j) {
        A[(off + i) * n + off + j] -= acc;
    }
}

// C = M * M^T (full matrix), shared-memory tiled. Used for generating the
// positive definite input matrix and for validating the factorization.
__global__ void multMMtKernel(const double* M, double* C, size_t n) {
    __shared__ double Ps[TILE][TILE + 1];
    __shared__ double Qs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t i = (size_t)blockIdx.y * TILE + ty;
    const size_t j = (size_t)blockIdx.x * TILE + tx;

    double acc = 0.0;
    for (size_t kk = 0; kk < n; kk += TILE) {
        Ps[ty][tx] = (i < n && kk + tx < n) ? M[i * n + kk + tx] : 0.0;
        Qs[ty][tx] = ((size_t)blockIdx.x * TILE + ty < n && kk + tx < n)
                         ? M[((size_t)blockIdx.x * TILE + ty) * n + kk + tx]
                         : 0.0;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            acc += Ps[ty][t] * Qs[tx][t];
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        C[i * n + j] = acc;
    }
}

// Compute out = M * M^T on the GPU
static void gpuMultMMt(const std::vector<double>& M, std::vector<double>& out, size_t n) {
    double* dM = nullptr;
    double* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dM, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dM, M.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned nb = (unsigned)((n + TILE - 1) / TILE);
    dim3 grid(nb, nb);
    dim3 block(TILE, TILE);
    multMMtKernel<<<grid, block>>>(dM, dC, n);

    CUDA_CHECK(cudaMemcpy(out.data(), dC, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dM));
    CUDA_CHECK(cudaFree(dC));
}

// Zero out the strictly upper triangular part of A
__global__ void zeroUpperKernel(double* A, size_t n) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* dA = nullptr;
    int* dInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dInfo, 0, sizeof(int)));

    for (size_t k = 0; k < n; k += NB) {
        const int nb = (int)std::min((size_t)NB, n - k);

        // Factorize diagonal block
        potf2Kernel<<<1, NB>>>(dA, n, k, nb, dInfo);

        int info = 0;
        CUDA_CHECK(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) {
            // Matrix is not positive definite
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                   (size_t)(info - 1));
            cudaFree(dA);
            cudaFree(dInfo);
            return false;
        }

        const size_t m = n - k - nb;
        if (m > 0) {
            // Panel solve below the diagonal block
            const int trsmThreads = 128;
            const unsigned trsmBlocks = (unsigned)((m + trsmThreads - 1) / trsmThreads);
            trsmKernel<<<trsmBlocks, trsmThreads>>>(dA, n, k, nb, m);

            // Trailing-matrix update
            const unsigned mb = (unsigned)((m + TILE - 1) / TILE);
            dim3 grid(mb, mb);
            dim3 block(TILE, TILE);
            syrkKernel<<<grid, block>>>(dA, n, k, nb, m);
        }
    }

    {
        dim3 block(32, 8);
        dim3 grid((unsigned)((n + block.x - 1) / block.x),
                  (unsigned)((n + block.y - 1) / block.y));
        zeroUpperKernel<<<grid, block>>>(dA, n);
    }

    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dInfo));

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
    gpuMultMMt(B, A, n);
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T on the GPU
    gpuMultMMt(L, reconstructed, n);
    
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
    
    // Initialize the CUDA context up front so it is not attributed
    // to the timed decomposition
    CUDA_CHECK(cudaFree(nullptr));

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
