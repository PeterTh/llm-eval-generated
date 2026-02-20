#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA-accelerated Cholesky decomposition (right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

namespace {
constexpr int kBlockSize = 256;
constexpr int kUpdateBlock = 16;

bool checkCuda(cudaError_t result, const char* message) {
    if (result == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(result));
    return false;
}

__global__ void reduceRowSquaresKernel(const double* __restrict__ A,
                                       double* __restrict__ blockSums,
                                       int n,
                                       int k) {
    extern __shared__ double shared[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    const int rowOffset = k * n;
    double sum = 0.0;

    for (int j = idx; j < k; j += stride) {
        const double val = A[rowOffset + j];
        sum += val * val;
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = shared[0];
    }
}

__global__ void reduceBlockSumsKernel(const double* __restrict__ blockSums,
                                      double* __restrict__ result,
                                      int numBlocks) {
    extern __shared__ double shared[];
    const int tid = threadIdx.x;
    double sum = 0.0;

    for (int i = tid; i < numBlocks; i += blockDim.x) {
        sum += blockSums[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = shared[0];
    }
}

__global__ void computeDiagKernel(double* A,
                                  const double* sum,
                                  int n,
                                  int k,
                                  int* status) {
    if (threadIdx.x == 0) {
        const double val = A[k * n + k] - sum[0];
        if (val <= 0.0) {
            status[0] = 1;
            A[k * n + k] = 0.0;
        } else {
            A[k * n + k] = sqrt(val);
        }
    }
}

__global__ void computeColumnKernel(double* A, int n, int k) {
    extern __shared__ double rowTile[];
    const int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const int rowOffset = i * n;
    const int kOffset = k * n;
    double sum = 0.0;

    for (int tile = 0; tile < k; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        rowTile[threadIdx.x] = (j < k) ? A[kOffset + j] : 0.0;
        __syncthreads();

        const int tileEnd = ((k - tile) < blockDim.x) ? (k - tile) : blockDim.x;
        if (active) {
            for (int t = 0; t < tileEnd; ++t) {
                sum += A[rowOffset + tile + t] * rowTile[t];
            }
        }
        __syncthreads();
    }

    if (active) {
        A[rowOffset + k] = (A[rowOffset + k] - sum) / A[kOffset + k];
    }
}

__global__ void updateTrailingKernel(double* A, int n, int k) {
    const int i = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    const int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && i >= j) {
        A[i * n + j] -= A[i * n + k] * A[j * n + k];
    }
}

__global__ void zeroUpperKernel(double* A, int n) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n, double* durationMs) {
    if (n == 0) {
        if (durationMs) {
            *durationMs = 0.0;
        }
        return true;
    }

    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        printf("Error: Matrix size too large for GPU execution\n");
        return false;
    }

    const int nInt = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double* d_A = nullptr;
    double* d_blockSums = nullptr;
    double* d_sum = nullptr;
    int* d_status = nullptr;
    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};

    auto cleanup = [&]() {
        if (d_A) {
            cudaFree(d_A);
        }
        if (d_blockSums) {
            cudaFree(d_blockSums);
        }
        if (d_sum) {
            cudaFree(d_sum);
        }
        if (d_status) {
            cudaFree(d_status);
        }
        if (startEvent) {
            cudaEventDestroy(startEvent);
        }
        if (stopEvent) {
            cudaEventDestroy(stopEvent);
        }
    };

    if (!checkCuda(cudaMalloc(&d_A, bytes), "allocate device matrix")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "copy matrix to device")) {
        cleanup();
        return false;
    }

    const int maxBlocks = (nInt + kBlockSize - 1) / kBlockSize;
    if (!checkCuda(cudaMalloc(&d_blockSums, maxBlocks * sizeof(double)), "allocate block sums")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaMalloc(&d_sum, sizeof(double)), "allocate sum")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaMalloc(&d_status, sizeof(int)), "allocate status")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaMemset(d_status, 0, sizeof(int)), "reset status")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaEventCreate(&startEvent), "create start event")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaEventCreate(&stopEvent), "create stop event")) {
        cleanup();
        return false;
    }

    if (!checkCuda(cudaEventRecord(startEvent), "record start event")) {
        cleanup();
        return false;
    }

    for (int k = 0; k < nInt; ++k) {
        if (k > 0) {
            const int numBlocks = (k + kBlockSize - 1) / kBlockSize;
            const size_t sharedBytes = kBlockSize * sizeof(double);
            reduceRowSquaresKernel<<<numBlocks, kBlockSize, sharedBytes>>>(d_A, d_blockSums, nInt, k);
            if (!checkCuda(cudaGetLastError(), "launch reduceRowSquaresKernel")) {
                cleanup();
                return false;
            }
            reduceBlockSumsKernel<<<1, kBlockSize, sharedBytes>>>(d_blockSums, d_sum, numBlocks);
            if (!checkCuda(cudaGetLastError(), "launch reduceBlockSumsKernel")) {
                cleanup();
                return false;
            }
        } else {
            if (!checkCuda(cudaMemset(d_sum, 0, sizeof(double)), "reset sum")) {
                cleanup();
                return false;
            }
        }

        computeDiagKernel<<<1, 1>>>(d_A, d_sum, nInt, k, d_status);
        if (!checkCuda(cudaGetLastError(), "launch computeDiagKernel")) {
            cleanup();
            return false;
        }

        int hostStatus = 0;
        if (!checkCuda(cudaMemcpy(&hostStatus, d_status, sizeof(int), cudaMemcpyDeviceToHost), "copy status")) {
            cleanup();
            return false;
        }
        if (hostStatus != 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", k);
            cleanup();
            return false;
        }

        const int remaining = nInt - k - 1;
        if (remaining > 0) {
            const int colBlocks = (remaining + kBlockSize - 1) / kBlockSize;
            const size_t colShared = kBlockSize * sizeof(double);
            computeColumnKernel<<<colBlocks, kBlockSize, colShared>>>(d_A, nInt, k);
            if (!checkCuda(cudaGetLastError(), "launch computeColumnKernel")) {
                cleanup();
                return false;
            }

            dim3 block(kUpdateBlock, kUpdateBlock);
            dim3 grid((remaining + block.x - 1) / block.x,
                      (remaining + block.y - 1) / block.y);
            updateTrailingKernel<<<grid, block>>>(d_A, nInt, k);
            if (!checkCuda(cudaGetLastError(), "launch updateTrailingKernel")) {
                cleanup();
                return false;
            }
        }
    }

    dim3 block(kUpdateBlock, kUpdateBlock);
    dim3 grid((nInt + block.x - 1) / block.x,
              (nInt + block.y - 1) / block.y);
    zeroUpperKernel<<<grid, block>>>(d_A, nInt);
    if (!checkCuda(cudaGetLastError(), "launch zeroUpperKernel")) {
        cleanup();
        return false;
    }

    if (!checkCuda(cudaEventRecord(stopEvent), "record stop event")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaEventSynchronize(stopEvent), "sync stop event")) {
        cleanup();
        return false;
    }

    float elapsedMs = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "elapsed time")) {
        cleanup();
        return false;
    }

    if (durationMs) {
        *durationMs = static_cast<double>(elapsedMs);
    }

    if (!checkCuda(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost), "copy matrix to host")) {
        cleanup();
        return false;
    }

    cleanup();
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
    double durationMs = 0.0;
    bool success = choleskyDecomposition(A, n, &durationMs);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    const double durationSeconds = durationMs / 1000.0;
    double gflops = durationSeconds > 0.0 ? (ops / durationSeconds / 1e9) : 0.0;
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
