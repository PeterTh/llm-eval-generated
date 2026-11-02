#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// All matrices use row-major storage. CUDA errors are fatal, as a partial
// factorization or matrix product cannot be used by the benchmark.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

constexpr int TILE = 16;
constexpr int PANEL = 32;

// Form C = B * B^T, optionally adding n to the diagonal. Each thread
// computes one output element and consumes k in the original order.
__global__ void symmetricProduct(const double* B, double* C, size_t n, bool addDiagonal) {
    __shared__ double left[TILE][TILE];
    __shared__ double right[TILE][TILE + 1];
    const size_t row = size_t(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = size_t(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t base = 0; base < n; base += TILE) {
        const size_t kx = base + threadIdx.x;
        const size_t rightRow = size_t(blockIdx.x) * TILE + threadIdx.y;
        left[threadIdx.y][threadIdx.x] = row < n && kx < n ? B[row * n + kx] : 0.0;
        right[threadIdx.y][threadIdx.x] = rightRow < n && kx < n ? B[rightRow * n + kx] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += left[threadIdx.y][k] * right[threadIdx.x][k];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum + (addDiagonal && row == col ? double(n) : 0.0);
}

// Factor one diagonal panel in shared memory. The preceding panels have
// already updated this block in the original matrix.
__global__ void factorDiagonal(double* A, size_t n, size_t base, int width, unsigned long long* failure) {
    __shared__ double d[PANEL][PANEL + 1];
    const int lane = threadIdx.x;
    for (int index = lane; index < width * width; index += blockDim.x) {
        const int row = index / width, col = index % width;
        d[row][col] = A[(base + row) * n + base + col];
    }
    __syncthreads();
    for (int j = 0; j < width; ++j) {
        if (lane == j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += d[j][k] * d[j][k];
            double val = d[j][j] - sum;
            if (val <= 0.0) {
                atomicMin(failure, static_cast<unsigned long long>(base + j));
                val = 1.0; // Allow the remaining kernels to finish safely.
            }
            d[j][j] = sqrt(val);
        }
        __syncthreads();
        if (lane > j && lane < width) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += d[lane][k] * d[j][k];
            d[lane][j] = (d[lane][j] - sum) / d[j][j];
        }
        __syncthreads();
    }
    for (int index = lane; index < width * width; index += blockDim.x) {
        const int row = index / width, col = index % width;
        if (col <= row) A[(base + row) * n + base + col] = d[row][col];
    }
}

// Solve the block column against the triangular diagonal panel.
__global__ void solvePanel(double* A, size_t n, size_t base, int width) {
    const size_t row = base + width + size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int j = 0; j < width; ++j) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) sum += A[row * n + base + k] * A[(base + j) * n + base + k];
        A[row * n + base + j] = (A[row * n + base + j] - sum) / A[(base + j) * n + base + j];
    }
}

// Update the lower trailing matrix with the product of the panel and its
// transpose. Shared tiles provide reuse across the output block.
__global__ void updateTrailing(double* A, size_t n, size_t base, int width) {
    __shared__ double left[TILE][TILE];
    __shared__ double right[TILE][TILE + 1];
    const size_t row = base + width + size_t(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = base + width + size_t(blockIdx.x) * TILE + threadIdx.x;
    if (blockIdx.x > blockIdx.y) return;
    double sum = 0.0;
    for (int kbase = 0; kbase < width; kbase += TILE) {
        const int kx = kbase + threadIdx.x;
        const size_t rightRow = base + width + size_t(blockIdx.x) * TILE + threadIdx.y;
        left[threadIdx.y][threadIdx.x] = row < n && kx < width ? A[row * n + base + kx] : 0.0;
        right[threadIdx.y][threadIdx.x] = rightRow < n && kx < width ? A[rightRow * n + base + kx] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += left[threadIdx.y][k] * right[threadIdx.x][k];
        __syncthreads();
    }
    if (row < n && col < n && col <= row) A[row * n + col] -= sum;
}

__global__ void clearUpper(double* A, size_t n) {
    const size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) A[row * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    const size_t bytes = A.size() * sizeof(double);
    double* deviceA;
    unsigned long long* deviceFailure;
    checkCuda(cudaMalloc(&deviceA, bytes), "allocating the factor matrix");
    checkCuda(cudaMalloc(&deviceFailure, sizeof(*deviceFailure)), "allocating the status");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying the matrix to the GPU");
    checkCuda(cudaMemset(deviceFailure, 0xff, sizeof(*deviceFailure)), "initializing the status");
    for (size_t base = 0; base < n; base += PANEL) {
        const int width = int(std::min(size_t(PANEL), n - base));
        factorDiagonal<<<1, PANEL>>>(deviceA, n, base, width, deviceFailure);
        const size_t remaining = n - base - width;
        if (remaining) {
            solvePanel<<<(remaining + 127) / 128, 128>>>(deviceA, n, base, width);
            const unsigned int blocks = unsigned((remaining + TILE - 1) / TILE);
            updateTrailing<<<dim3(blocks, blocks), dim3(TILE, TILE)>>>(deviceA, n, base, width);
        }
    }
    const unsigned int blocks = unsigned((n + TILE - 1) / TILE);
    clearUpper<<<dim3(blocks, blocks), dim3(TILE, TILE)>>>(deviceA, n);
    checkCuda(cudaGetLastError(), "launching Cholesky kernels");
    unsigned long long failure;
    checkCuda(cudaMemcpy(&failure, deviceFailure, sizeof(failure), cudaMemcpyDeviceToHost), "reading the factorization status");
    if (failure == ~0ULL) checkCuda(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "copying the factor to the host");
    checkCuda(cudaFree(deviceFailure), "freeing the status");
    checkCuda(cudaFree(deviceA), "freeing the factor matrix");
    if (failure != ~0ULL) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", size_t(failure));
        return false;
    }
    return true;
}

// Generate the same rand_r sequence as the original benchmark on the host;
// the expensive symmetric matrix product runs on the GPU.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    if (n == 0) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) value = (rand_r(&seed) / double(RAND_MAX)) - 0.5;
    const size_t bytes = A.size() * sizeof(double);
    double *deviceB, *deviceA;
    checkCuda(cudaMalloc(&deviceB, bytes), "allocating the input matrix");
    checkCuda(cudaMalloc(&deviceA, bytes), "allocating the product matrix");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying the input matrix");
    const unsigned int blocks = unsigned((n + TILE - 1) / TILE);
    symmetricProduct<<<dim3(blocks, blocks), dim3(TILE, TILE)>>>(deviceB, deviceA, n, true);
    checkCuda(cudaGetLastError(), "launching matrix generation");
    checkCuda(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "copying the generated matrix");
    checkCuda(cudaFree(deviceA), "freeing the product matrix");
    checkCuda(cudaFree(deviceB), "freeing the input matrix");
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    if (n != 0) {
        const size_t bytes = L.size() * sizeof(double);
        double *deviceL, *deviceProduct;
        checkCuda(cudaMalloc(&deviceL, bytes), "allocating the validation factor");
        checkCuda(cudaMalloc(&deviceProduct, bytes), "allocating the validation product");
        checkCuda(cudaMemcpy(deviceL, L.data(), bytes, cudaMemcpyHostToDevice), "copying the validation factor");
        const unsigned int blocks = unsigned((n + TILE - 1) / TILE);
        symmetricProduct<<<dim3(blocks, blocks), dim3(TILE, TILE)>>>(deviceL, deviceProduct, n, false);
        checkCuda(cudaGetLastError(), "launching validation");
        checkCuda(cudaMemcpy(reconstructed.data(), deviceProduct, bytes, cudaMemcpyDeviceToHost), "copying the validation product");
        checkCuda(cudaFree(deviceProduct), "freeing the validation product");
        checkCuda(cudaFree(deviceL), "freeing the validation factor");
    }
    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
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
