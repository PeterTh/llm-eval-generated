#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA Cholesky decomposition (blocked, right-looking). A is row-major.

static inline bool cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        printf("CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        return false;
    }
    return true;
}

constexpr int CHOLESKY_NB = 32;

__global__ void potrf_diag_tile(double* __restrict__ A, int lda, int k, int bk, int* __restrict__ info) {
    __shared__ double s[CHOLESKY_NB * CHOLESKY_NB];

    for (int idx = threadIdx.x; idx < bk * bk; idx += blockDim.x) {
        int r = idx / bk;
        int c = idx - r * bk;
        s[r * CHOLESKY_NB + c] = A[(k + r) * lda + (k + c)];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = 0; j < bk; ++j) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                double v = s[j * CHOLESKY_NB + p];
                sum += v * v;
            }
            double val = s[j * CHOLESKY_NB + j] - sum;
            if (val <= 0.0) {
                atomicCAS(info, 0, k + j + 1);
                val = 1.0; // avoid NaNs propagating in case caller keeps going
            }
            s[j * CHOLESKY_NB + j] = sqrt(val);

            for (int i = j + 1; i < bk; ++i) {
                sum = 0.0;
                for (int p = 0; p < j; ++p) {
                    sum += s[i * CHOLESKY_NB + p] * s[j * CHOLESKY_NB + p];
                }
                s[i * CHOLESKY_NB + j] = (s[i * CHOLESKY_NB + j] - sum) / s[j * CHOLESKY_NB + j];
            }

            // Keep upper triangle clean (not required for correctness, but helps validation/output).
            for (int c = j + 1; c < bk; ++c) {
                s[j * CHOLESKY_NB + c] = 0.0;
            }
        }
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < bk * bk; idx += blockDim.x) {
        int r = idx / bk;
        int c = idx - r * bk;
        A[(k + r) * lda + (k + c)] = s[r * CHOLESKY_NB + c];
    }
}

__global__ void trsm_panel(double* __restrict__ A, int lda, int k, int bk) {
    __shared__ double L[CHOLESKY_NB * CHOLESKY_NB];

    for (int idx = threadIdx.x; idx < bk * bk; idx += blockDim.x) {
        int r = idx / bk;
        int c = idx - r * bk;
        L[r * CHOLESKY_NB + c] = A[(k + r) * lda + (k + c)];
    }
    __syncthreads();

    int row = k + bk + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= lda) return;

    for (int col = 0; col < bk; ++col) {
        double sum = 0.0;
        for (int p = 0; p < col; ++p) {
            sum += A[row * lda + (k + p)] * L[col * CHOLESKY_NB + p];
        }
        A[row * lda + (k + col)] = (A[row * lda + (k + col)] - sum) / L[col * CHOLESKY_NB + col];
    }
}

__global__ void syrk_trailing_update(double* __restrict__ A, int lda, int k, int bk) {
    const int start = k + bk;

    const int localRow = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int localCol = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = start + localRow;
    const int c = start + localCol;

    __shared__ double aSh[16][CHOLESKY_NB];
    __shared__ double bSh[16][CHOLESKY_NB];

    // Load the panel vectors needed for this (r,c) block.
    if (r < lda) {
        for (int t = (int)threadIdx.x; t < bk; t += (int)blockDim.x) {
            aSh[threadIdx.y][t] = A[r * lda + (k + t)];
        }
    }

    if (c < lda) {
        for (int t = (int)threadIdx.y; t < bk; t += (int)blockDim.y) {
            bSh[threadIdx.x][t] = A[c * lda + (k + t)];
        }
    }

    __syncthreads();

    if (r < lda && c < lda && c <= r) {
        double sum = 0.0;
        for (int t = 0; t < bk; ++t) {
            sum += aSh[threadIdx.y][t] * bSh[threadIdx.x][t];
        }
        A[r * lda + c] -= sum;
    }
}

__global__ void zero_upper_triangle(double* __restrict__ A, int lda) {
    int r = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r < lda && c < lda && c > r) {
        A[r * lda + c] = 0.0;
    }
}

static float g_cholesky_kernel_ms = 0.0f;

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int deviceCount = 0;
    if (!cuda_check(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) return false;
    if (deviceCount <= 0) {
        printf("Error: No CUDA devices found\n");
        return false;
    }
    if (!cuda_check(cudaSetDevice(0), "cudaSetDevice")) return false;

    double* dA = nullptr;
    int* dInfo = nullptr;

    const size_t bytes = n * n * sizeof(double);
    if (!cuda_check(cudaMalloc((void**)&dA, bytes), "cudaMalloc(dA)")) return false;
    if (!cuda_check(cudaMalloc((void**)&dInfo, sizeof(int)), "cudaMalloc(dInfo)")) {
        cudaFree(dA);
        return false;
    }

    if (!cuda_check(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "H2D memcpy")) {
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }
    if (!cuda_check(cudaMemset(dInfo, 0, sizeof(int)), "memset info")) {
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }

    cudaEvent_t startEv{}, stopEv{};
    if (!cuda_check(cudaEventCreate(&startEv), "cudaEventCreate(start)")) {
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }
    if (!cuda_check(cudaEventCreate(&stopEv), "cudaEventCreate(stop)")) {
        cudaEventDestroy(startEv);
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }

    const int lda = (int)n;
    cuda_check(cudaEventRecord(startEv), "cudaEventRecord(start)");

    for (int k = 0; k < lda; k += CHOLESKY_NB) {
        const int bk = (k + CHOLESKY_NB <= lda) ? CHOLESKY_NB : (lda - k);

        potrf_diag_tile<<<1, 256>>>(dA, lda, k, bk, dInfo);

        const int rem = lda - (k + bk);
        if (rem > 0) {
            const int threads = 256;
            const int blocks = (rem + threads - 1) / threads;
            trsm_panel<<<blocks, threads>>>(dA, lda, k, bk);

            dim3 block(16, 16);
            dim3 grid((rem + block.x - 1) / block.x, (rem + block.y - 1) / block.y);
            syrk_trailing_update<<<grid, block>>>(dA, lda, k, bk);
        }

        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            printf("CUDA launch error: %s\n", cudaGetErrorString(launchErr));
            cudaEventDestroy(stopEv);
            cudaEventDestroy(startEv);
            cudaFree(dInfo);
            cudaFree(dA);
            return false;
        }
    }

    {
        dim3 block(16, 16);
        dim3 grid((lda + block.x - 1) / block.x, (lda + block.y - 1) / block.y);
        zero_upper_triangle<<<grid, block>>>(dA, lda);
    }

    cuda_check(cudaEventRecord(stopEv), "cudaEventRecord(stop)");
    if (!cuda_check(cudaEventSynchronize(stopEv), "cudaEventSynchronize")) {
        cudaEventDestroy(stopEv);
        cudaEventDestroy(startEv);
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }
    cuda_check(cudaEventElapsedTime(&g_cholesky_kernel_ms, startEv, stopEv), "cudaEventElapsedTime");
    cudaEventDestroy(stopEv);
    cudaEventDestroy(startEv);

    if (!cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }

    int info = 0;
    if (!cuda_check(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost), "D2H info")) {
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }
    if (info != 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }

    if (!cuda_check(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost), "D2H memcpy")) {
        cudaFree(dInfo);
        cudaFree(dA);
        return false;
    }

    cudaFree(dInfo);
    cudaFree(dA);
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

    bool success = choleskyDecomposition(A, n);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    const long ms = (long)llround((double)g_cholesky_kernel_ms);
    printf("Computation time: %ld ms\n", ms);

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double secs = (double)g_cholesky_kernel_ms / 1000.0;
    double gflops = (secs > 0.0) ? (ops / secs / 1e9) : INFINITY;
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
