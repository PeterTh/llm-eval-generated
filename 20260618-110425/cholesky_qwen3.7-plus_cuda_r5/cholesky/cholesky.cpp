#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define BS 32

// Factor diagonal block A[k:k+bs, k:k+bs] in-place using shared memory.
// One thread per row; column-by-column right-looking factorization.
__global__ void factor_diag_block(double* __restrict__ A, size_t n, size_t k, size_t bs) {
    int tid = threadIdx.x;
    if (tid >= (int)bs) return;

    extern __shared__ double smem[];
    // smem layout: [bs][bs] row-major, smem[r*bs + c] = L[k+r, k+c]

    size_t row = k + tid;
    if (row < n) {
        for (size_t c = 0; c < bs; c++)
            smem[tid * bs + c] = A[row * n + (k + c)];
    }
    __syncthreads();

    for (size_t j = 0; j < bs; j++) {
        // Diagonal element
        if (tid == (int)j) {
            double s = 0.0;
            for (size_t m = 0; m < j; m++)
                s += smem[j * bs + m] * smem[j * bs + m];
            double v = smem[j * bs + j] - s;
            smem[j * bs + j] = (v > 0.0) ? sqrt(v) : -1.0;
        }
        __syncthreads();

        // Off-diagonal below
        if (tid > (int)j && row < n) {
            double s = 0.0;
            for (size_t m = 0; m < j; m++)
                s += smem[tid * bs + m] * smem[j * bs + m];
            smem[tid * bs + j] = (smem[tid * bs + j] - s) / smem[j * bs + j];
        }
        __syncthreads();
    }

    // Zero upper triangle
    if (row < n) {
        for (size_t c = tid + 1; c < bs; c++)
            smem[tid * bs + c] = 0.0;
    }
    __syncthreads();

    if (row < n) {
        for (size_t c = 0; c < bs; c++)
            A[row * n + (k + c)] = smem[tid * bs + c];
    }
}

// Panel triangular solve: compute L[k+bs:n, k:k+bs].
// Each thread processes one row. The sum only runs over columns within the
// current block because contributions from earlier blocks were already
// subtracted by the trailing-matrix update of previous iterations.
__global__ void panel_trsm(double* __restrict__ A, size_t n, size_t k, size_t bs) {
    size_t i = k + bs + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    for (size_t jj = 0; jj < bs; jj++) {
        size_t j = k + jj;
        if (j >= n) break;
        double s = 0.0;
        for (size_t m = k; m < j; m++)
            s += A[i * n + m] * A[j * n + m];
        A[i * n + j] = (A[i * n + j] - s) / A[j * n + j];
    }
}

// Trailing submatrix symmetric rank-bs update:
//   A[k+bs:n, k+bs:n] -= L[k+bs:n, k:k+bs] * L[k+bs:n, k:k+bs]^T
// One thread per lower-triangular element.
__global__ void trailing_syrk(double* __restrict__ A, size_t n, size_t k, size_t bs) {
    size_t j = k + bs + blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = k + bs + blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n || j > i) return;

    double s = 0.0;
    size_t end = k + bs;
    if (end > n) end = n;
    for (size_t m = k; m < end; m++)
        s += A[i * n + m] * A[j * n + m];
    A[i * n + j] -= s;
}

// Zero out the strict upper triangle
__global__ void zero_upper(double* __restrict__ A, size_t n) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i)
        A[i * n + j] = 0.0;
}

// Blocked right-looking Cholesky on GPU
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const size_t bs = BS;

    for (size_t k = 0; k < n; k += bs) {
        size_t cur = std::min(bs, n - k);

        // 1) Factor diagonal block
        factor_diag_block<<<1, (unsigned)cur, cur * cur * sizeof(double)>>>(d_A, n, k, cur);

        if (k + cur < n) {
            // 2) Panel triangular solve
            size_t panel_rows = n - k - cur;
            unsigned pt = 256;
            unsigned pb = (unsigned)((panel_rows + pt - 1) / pt);
            panel_trsm<<<pb, pt>>>(d_A, n, k, cur);

            // 3) Trailing symmetric rank-cur update
            size_t trail = n - k - cur;
            dim3 tt(16, 16);
            dim3 tb((unsigned)((trail + 15) / 16), (unsigned)((trail + 15) / 16));
            trailing_syrk<<<tb, tt>>>(d_A, n, k, cur);
        }
    }

    // Zero upper triangle
    dim3 zt(16, 16);
    dim3 zb((unsigned)((n + 15) / 16), (unsigned)((n + 15) / 16));
    zero_upper<<<zb, zt>>>(d_A, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
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
