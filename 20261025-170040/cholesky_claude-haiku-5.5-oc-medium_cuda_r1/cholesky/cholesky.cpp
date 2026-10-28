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

// Cholesky decomposition on the GPU (CUDA).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization strategy (column-oriented, left-looking):
//   For each column j:
//     1. diagonalKernel: L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)   (one block)
//     2. updateKernel:   for all rows i > j in parallel,
//                        L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]
//   Every element's dot product is accumulated in the same sequential k-order as
//   the original CPU algorithm, so results are bit-identical to it.
//
// Device data is stored column-major (LT[c * n + r] = A[r][c]) so that threads
// of a warp (which own consecutive rows i) access memory contiguously.

namespace {

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                   \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int UPDATE_THREADS = 32;
constexpr size_t UPDATE_BATCH = 16;

// Tiled transpose: out[x * n + y] = in[y * n + x].
// If LowerOnly is set, output entries with y > x (strict upper triangle) are zeroed.
template <bool LowerOnly>
__global__ void transposeKernel(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    __shared__ double tile[TILE][TILE + 1];
    const size_t bx = (size_t)blockIdx.x * TILE;
    const size_t by = (size_t)blockIdx.y * TILE;

    for (int s = 0; s < TILE; s += BLOCK_ROWS) {
        const size_t y = by + threadIdx.y + s;
        const size_t x = bx + threadIdx.x;
        if (y < n && x < n) tile[threadIdx.y + s][threadIdx.x] = in[y * n + x];
    }
    __syncthreads();

    for (int s = 0; s < TILE; s += BLOCK_ROWS) {
        const size_t x = bx + threadIdx.y + s;   // output row
        const size_t y = by + threadIdx.x;       // output column
        if (x < n && y < n) {
            const double v = tile[threadIdx.x][threadIdx.y + s];
            out[x * n + y] = (!LowerOnly || y <= x) ? v : 0.0;
        }
    }
}

// Computes the diagonal element of column j. The squares are formed in parallel
// (exact per-term rounding), while the accumulation is strictly sequential in k
// on thread 0 to preserve the original summation order.
constexpr int DIAG_THREADS = 256;
constexpr size_t DIAG_CHUNK = 2048;

__global__ void diagonalKernel(double* __restrict__ LT, size_t n, size_t j, int* __restrict__ failIdx) {
    __shared__ double sq[DIAG_CHUNK];
    double sum = 0.0;  // meaningful in thread 0 only

    for (size_t base = 0; base < j; base += DIAG_CHUNK) {
        const size_t len = (j - base < DIAG_CHUNK) ? (j - base) : DIAG_CHUNK;
        for (size_t t = threadIdx.x; t < len; t += blockDim.x) {
            const double v = LT[(base + t) * n + j];
            sq[t] = __dmul_rn(v, v);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
#pragma unroll 8
            for (size_t t = 0; t < len; ++t) {
                sum = __dadd_rn(sum, sq[t]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const double val = LT[j * n + j] - sum;
        if (val <= 0.0) {
            // Matrix is not positive definite; record the first (smallest) failing column
            atomicMin(failIdx, (int)j);
        } else {
            LT[j * n + j] = sqrt(val);
        }
    }
}

// Computes column j below the diagonal: one thread per row i > j.
// Each element's dot product is accumulated sequentially in k (same order as the
// original algorithm); products are formed with explicit rounding (no FMA).
// Loads are batched so that several memory requests are in flight per thread.
__global__ void updateKernel(double* __restrict__ LT, size_t n, size_t j) {
    const size_t i = j + 1 + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double* pa = LT + i;  // LT[k * n + i]
    const double* pb = LT + j;  // LT[k * n + j]
    double sum = 0.0;
    size_t k = 0;
    for (; k + UPDATE_BATCH <= j; k += UPDATE_BATCH) {
        double a[UPDATE_BATCH], b[UPDATE_BATCH];
#pragma unroll
        for (int t = 0; t < UPDATE_BATCH; ++t) {
            a[t] = pa[t * n];
            b[t] = pb[t * n];
        }
#pragma unroll
        for (int t = 0; t < UPDATE_BATCH; ++t) {
            sum = __dadd_rn(sum, __dmul_rn(a[t], b[t]));
        }
        pa += UPDATE_BATCH * n;
        pb += UPDATE_BATCH * n;
    }
    for (; k < j; ++k) {
        sum = __dadd_rn(sum, __dmul_rn(*pa, *pb));
        pa += n;
        pb += n;
    }
    LT[j * n + i] = (LT[j * n + i] - sum) / LT[j * n + j];
}

}  // namespace

// Initializes the CUDA device/context so that one-time driver setup is not
// attributed to the timed decomposition.
void initCudaDevice() {
    CUDA_CHECK(cudaFree(nullptr));
}

// Simple Cholesky decomposition (GPU, column-oriented)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;

    const size_t bytes = n * n * sizeof(double);
    double* dA = nullptr;   // row-major input/output
    double* dLT = nullptr;  // column-major working buffer
    int* dFail = nullptr;   // smallest failing diagonal index (INT_MAX if none)

    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dLT, bytes));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));

    const int noFail = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dFail, &noFail, sizeof(int), cudaMemcpyHostToDevice));

    const dim3 tBlock(TILE, BLOCK_ROWS);
    const dim3 tGrid((unsigned)((n + TILE - 1) / TILE), (unsigned)((n + TILE - 1) / TILE));

    transposeKernel<false><<<tGrid, tBlock>>>(dA, dLT, n);

    for (size_t j = 0; j < n; ++j) {
        diagonalKernel<<<1, DIAG_THREADS>>>(dLT, n, j, dFail);
        const size_t rows = n - j - 1;
        if (rows > 0) {
            const unsigned blocks = (unsigned)((rows + UPDATE_THREADS - 1) / UPDATE_THREADS);
            updateKernel<<<blocks, UPDATE_THREADS>>>(dLT, n, j);
        }
    }

    transposeKernel<true><<<tGrid, tBlock>>>(dLT, dA, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));

    int failIdx = noFail;
    CUDA_CHECK(cudaMemcpy(&failIdx, dFail, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dLT));
    CUDA_CHECK(cudaFree(dFail));

    if (failIdx != noFail) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failIdx);
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
    
    // Initialize CUDA context outside the timed region
    initCudaDevice();

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
