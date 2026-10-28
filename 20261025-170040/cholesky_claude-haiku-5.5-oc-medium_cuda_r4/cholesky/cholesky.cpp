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

// Blocked, right-looking Cholesky decomposition on the GPU.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// Each block step k0 (width NB) performs:
//   1. factor the NB x NB diagonal block (one thread block, sequential pivots)
//   2. solve the panel rows below it (one thread per row, fully parallel)
//   3. update the trailing lower triangle: A22 -= P * P^T (tiled, parallel)

namespace {

constexpr int NB = 64;               // block (panel) width
constexpr int DIAG_THREADS = NB;     // threads for diagonal factorization
constexpr int PANEL_THREADS = 128;   // threads per panel block (one row each)
constexpr int TILE = 64;             // trailing-update output tile edge
constexpr int TILE_K = 16;           // K-chunk staged in shared memory
constexpr int GEMM_THREADS = 256;    // 16 x 16 threads, each computes 4 x 4 outputs
constexpr int UPPER_THREADS = 256;

// Factor the diagonal block A[k0:k0+kb, k0:k0+kb] in place (lower triangle).
// Pivot failures record the smallest failing global index into failIdx.
__global__ void choleskyDiagKernel(double* __restrict__ A, size_t n, size_t k0, int kb,
                                   int* __restrict__ failIdx) {
    __shared__ double L[NB][NB + 1];
    const int tid = threadIdx.x;

    for (int idx = tid; idx < kb * kb; idx += blockDim.x) {
        const int r = idx / kb;
        const int c = idx % kb;
        if (c <= r) {
            L[r][c] = A[(k0 + r) * n + (k0 + c)];
        }
    }
    __syncthreads();

    for (int j = 0; j < kb; ++j) {
        if (tid == 0) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[j][k] * L[j][k];
            }
            const double val = L[j][j] - sum;
            if (val <= 0.0) {
                atomicMin(failIdx, static_cast<int>(k0) + j);
            }
            L[j][j] = sqrt(val);
        }
        __syncthreads();

        if (tid > j && tid < kb) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[tid][k] * L[j][k];
            }
            L[tid][j] = (L[tid][j] - sum) / L[j][j];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < kb * kb; idx += blockDim.x) {
        const int r = idx / kb;
        const int c = idx % kb;
        if (c <= r) {
            A[(k0 + r) * n + (k0 + c)] = L[r][c];
        }
    }
}

// Solve X * L11^T = A21 for all rows below the diagonal block (in place).
// Requires kb == NB (guaranteed by the caller when rows remain below).
__global__ void choleskyPanelKernel(double* __restrict__ A, size_t n, size_t k0) {
    __shared__ double L[NB][NB + 1];
    const int tid = threadIdx.x;

    for (int idx = tid; idx < NB * NB; idx += blockDim.x) {
        const int r = idx / NB;
        const int c = idx % NB;
        if (c <= r) {
            L[r][c] = A[(k0 + r) * n + (k0 + c)];
        }
    }
    __syncthreads();

    const size_t row = k0 + NB + static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    if (row >= n) {
        return;
    }

    double* rowPtr = A + row * n + k0;
    double x[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        x[c] = rowPtr[c];
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        double sum = 0.0;
#pragma unroll
        for (int k = 0; k < j; ++k) {
            sum += x[k] * L[j][k];
        }
        x[j] = (x[j] - sum) / L[j][j];
    }
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        rowPtr[c] = x[c];
    }
}

// Trailing update of the lower triangle: A[i][j] -= sum_k A[i][k] * A[j][k]
// over the panel columns k in [k0, k0+NB), for k1 <= j <= i.
// blockIdx.y = row tile, blockIdx.x = column tile; upper tiles exit early.
__global__ void choleskyUpdateKernel(double* __restrict__ A, size_t n, size_t k0) {
    const int ti = blockIdx.y;
    const int tj = blockIdx.x;
    if (tj > ti) {
        return;
    }

    const size_t k1 = k0 + NB;
    const size_t r0 = k1 + static_cast<size_t>(ti) * TILE;
    const size_t c0 = k1 + static_cast<size_t>(tj) * TILE;

    __shared__ double As[TILE_K][TILE];
    __shared__ double Bs[TILE_K][TILE];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;

    double acc[4][4];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
#pragma unroll
        for (int y = 0; y < 4; ++y) {
            acc[x][y] = 0.0;
        }
    }

    for (int kc = 0; kc < NB; kc += TILE_K) {
#pragma unroll
        for (int s = 0; s < (TILE * TILE_K) / GEMM_THREADS; ++s) {
            const int idx = tid + s * GEMM_THREADS;
            const int r = idx / TILE_K;
            const int kk = idx % TILE_K;
            const size_t gk = k0 + kc + kk;
            const size_t ga = r0 + r;
            const size_t gb = c0 + r;
            As[kk][r] = (ga < n) ? A[ga * n + gk] : 0.0;
            Bs[kk][r] = (gb < n) ? A[gb * n + gk] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            double a[4];
            double b[4];
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                a[x] = As[kk][ty * 4 + x];
                b[x] = Bs[kk][tx * 4 + x];
            }
#pragma unroll
            for (int x = 0; x < 4; ++x) {
#pragma unroll
                for (int y = 0; y < 4; ++y) {
                    acc[x][y] = fma(a[x], b[y], acc[x][y]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const size_t gi = r0 + ty * 4 + x;
#pragma unroll
        for (int y = 0; y < 4; ++y) {
            const size_t gj = c0 + tx * 4 + y;
            if (gi < n && gj <= gi) {
                A[gi * n + gj] -= acc[x][y];
            }
        }
    }
}

// Zero the strictly upper triangular part (the result is lower triangular L).
__global__ void zeroUpperKernel(double* __restrict__ A, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * n) {
        return;
    }
    const size_t r = idx / n;
    const size_t c = idx % n;
    if (c > r) {
        A[idx] = 0.0;
    }
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    // A is stored in row-major order
    double* dA = nullptr;
    int* dFail = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const int noFail = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dFail, &noFail, sizeof(int), cudaMemcpyHostToDevice));

    bool ok = true;
    for (size_t k0 = 0; k0 < n; k0 += NB) {
        const int kb = static_cast<int>(std::min<size_t>(NB, n - k0));
        const size_t k1 = k0 + kb;

        choleskyDiagKernel<<<1, DIAG_THREADS>>>(dA, n, k0, kb, dFail);

        // Also synchronizes; detects non-positive pivots in order.
        int fail = INT_MAX;
        CUDA_CHECK(cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost));
        if (fail != INT_MAX) {
            // Matrix is not positive definite
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                   static_cast<size_t>(fail));
            ok = false;
            break;
        }

        if (k1 < n) {
            const size_t m = n - k1;
            const size_t panelBlocks = (m + PANEL_THREADS - 1) / PANEL_THREADS;
            choleskyPanelKernel<<<static_cast<unsigned int>(panelBlocks), PANEL_THREADS>>>(dA, n, k0);

            const unsigned int tiles = static_cast<unsigned int>((m + TILE - 1) / TILE);
            choleskyUpdateKernel<<<dim3(tiles, tiles), GEMM_THREADS>>>(dA, n, k0);
        }
    }

    if (ok) {
        const size_t total = n * n;
        const unsigned int upperBlocks = static_cast<unsigned int>((total + UPPER_THREADS - 1) / UPPER_THREADS);
        zeroUpperKernel<<<upperBlocks, UPPER_THREADS>>>(dA, n);
        CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(dFail));
    CUDA_CHECK(cudaFree(dA));
    return ok;
}

// Initialize the CUDA context up front so device setup is not charged to the timed region.
void initCudaDevice() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaFree(nullptr));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
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
