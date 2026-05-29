#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// CUDA kernel: generate random matrix B with deterministic seed
__global__ void generateRandomKernel(double* B, size_t n, unsigned int seed) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = n * n;
    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        unsigned int temp = seed + static_cast<unsigned int>(i);
        temp = temp * 1103515245u + 12345u;
        double val = static_cast<double>((temp >> 16) & 0x7FFF) / static_cast<double>(0x7FFF) - 0.5;
        B[i] = val;
    }
}

// CUDA kernel: add diagonal dominance (add n to diagonal)
__global__ void addDiagonalKernel(double* A, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        A[i * n + i] += static_cast<double>(n);
    }
}

// CUDA kernel: zero out upper triangle after Cholesky
__global__ void zeroUpperTriangle(double* A, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
}

// CUDA kernel: tile-based matrix transpose (16x16 tiles)
__global__ void transposeKernel(const double* __restrict__ src, double* __restrict__ dst, size_t n) {
    __shared__ double tile[16][17];

    size_t row = static_cast<size_t>(blockIdx.y) * 16 + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * 16 + threadIdx.x;

    if (row < n && col < n) {
        tile[threadIdx.y][threadIdx.x] = src[row * n + col];
    }
    __syncthreads();

    row = static_cast<size_t>(blockIdx.x) * 16 + threadIdx.y;
    col = static_cast<size_t>(blockIdx.y) * 16 + threadIdx.x;

    if (row < n && col < n) {
        dst[row * n + col] = tile[threadIdx.x][threadIdx.y];
    }
}

// CUDA kernel: compute max absolute and relative error for validation
__global__ void computeErrorKernel(const double* reconstructed, const double* A_orig,
                                   size_t n, double* d_maxAbs, double* d_maxRel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = n * n;

    double localMaxAbs = 0.0;
    double localMaxRel = 0.0;

    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        double diff = fabs(reconstructed[i] - A_orig[i]);
        if (diff > localMaxAbs) localMaxAbs = diff;
        double rel = diff / (fabs(A_orig[i]) + 1e-10);
        if (rel > localMaxRel) localMaxRel = rel;
    }

    extern __shared__ char sharedMem[];
    double* s_abs = reinterpret_cast<double*>(sharedMem);
    double* s_rel = s_abs + blockDim.x;

    s_abs[threadIdx.x] = localMaxAbs;
    s_rel[threadIdx.x] = localMaxRel;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_abs[threadIdx.x] = fmax(s_abs[threadIdx.x], s_abs[threadIdx.x + s]);
            s_rel[threadIdx.x] = fmax(s_rel[threadIdx.x], s_rel[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(d_maxAbs),
                  __double_as_longlong(s_abs[0]));
        atomicMax(reinterpret_cast<unsigned long long*>(d_maxRel),
                  __double_as_longlong(s_rel[0]));
    }
}

// Generate symmetric positive definite matrix on GPU: A = B*B^T + n*I
void generatePositiveDefiniteMatrix(double* d_A, double* d_B, cublasHandle_t cublasHandle,
                                    size_t n) {
    size_t nn = n * n;
    int blockSize = 256;
    int gridSize = (static_cast<int>(nn) + blockSize - 1) / blockSize;

    // Generate random matrix B on GPU
    generateRandomKernel<<<gridSize, blockSize>>>(d_B, n, 42);

    // A = B * B^T using cuBLAS GEMM
    // cuBLAS expects column-major; our data is row-major.
    // Row-major B is seen as B^T by cuBLAS.
    // To compute B*B^T: use gemm(T,N) = (B^T)^T * B^T = B * B^T
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(cublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                &alpha, d_B, static_cast<int>(n),
                d_B, static_cast<int>(n),
                &beta, d_A, static_cast<int>(n));

    // Add diagonal dominance
    addDiagonalKernel<<<gridSize, blockSize>>>(d_A, n);
}

// Perform Cholesky decomposition using cuSOLVER potrf
bool choleskyDecomposition(double* d_A, size_t n, cusolverDnHandle_t cusolverHandle) {
    size_t nn = n * n;
    int lda = static_cast<int>(n);

    // For a symmetric matrix, memcpy from row-major to column-major buffer
    // gives correct values because A[i*n+j] == A[j*n+i].
    // potrf operates in-place on the lower triangle (column-major).
    // After potrf, L is in column-major. We transpose back to row-major.
    double* d_A_col = nullptr;
    cudaMalloc(&d_A_col, nn * sizeof(double));
    cudaMemcpy(d_A_col, d_A, nn * sizeof(double), cudaMemcpyDeviceToDevice);

    // Workspace query for potrf
    int workspaceSize = 0;
    cusolverDnDpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_LOWER, lda, d_A_col, lda, &workspaceSize);
    double* d_workspace = nullptr;
    if (workspaceSize > 0) {
        cudaMalloc(&d_workspace, workspaceSize * sizeof(double));
    }

    int* d_info = nullptr;
    cudaMalloc(&d_info, sizeof(int));
    cusolverDnDpotrf(cusolverHandle, CUBLAS_FILL_MODE_LOWER, lda,
                     d_A_col, lda, d_workspace, workspaceSize, d_info);

    int info = 0;
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_info);

    if (info != 0) {
        printf("Error: Matrix is not positive definite (potrf info=%d)\n", info);
        cudaFree(d_A_col);
        if (d_workspace) cudaFree(d_workspace);
        return false;
    }

    // Transpose L_col back to row-major
    int tileSize = 16;
    dim3 tBlock(tileSize, tileSize);
    dim3 tGrid((lda + tileSize - 1) / tileSize, (lda + tileSize - 1) / tileSize);
    transposeKernel<<<tGrid, tBlock>>>(d_A_col, d_A, n);

    // Zero out upper triangle
    int blockSize = 256;
    int zeroGrid = (static_cast<int>(n) + blockSize - 1) / blockSize;
    zeroUpperTriangle<<<zeroGrid, blockSize>>>(d_A, n);

    cudaFree(d_A_col);
    if (d_workspace) cudaFree(d_workspace);
    return true;
}

bool validateCholesky(double* d_L, double* d_A_orig, size_t n, cublasHandle_t cublasHandle) {
    size_t nn = n * n;

    // Compute L * L^T on GPU using cuBLAS
    double* d_reconstructed = nullptr;
    cudaMalloc(&d_reconstructed, nn * sizeof(double));

    const double alpha = 1.0;
    const double beta = 0.0;
    // cuBLAS is column-major; our L is row-major.
    // cuBLAS sees L as L^T. cublasDgemm(T,N) = (L^T)^T * L^T = L * L^T
    cublasDgemm(cublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                &alpha, d_L, static_cast<int>(n),
                d_L, static_cast<int>(n),
                &beta, d_reconstructed, static_cast<int>(n));

    // Compute max absolute and relative error
    double h_maxAbs = 0.0, h_maxRel = 0.0;
    double* dd_maxAbs = nullptr, *dd_maxRel = nullptr;
    cudaMalloc(&dd_maxAbs, sizeof(double));
    cudaMalloc(&dd_maxRel, sizeof(double));
    cudaMemcpy(dd_maxAbs, &h_maxAbs, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_maxRel, &h_maxRel, sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (static_cast<int>(nn) + blockSize - 1) / blockSize;
    size_t sharedMemSize = 2 * blockSize * sizeof(double);
    computeErrorKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_reconstructed, d_A_orig, n, dd_maxAbs, dd_maxRel);

    cudaMemcpy(&h_maxAbs, dd_maxAbs, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_maxRel, dd_maxRel, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Max absolute error: %.10e\n", h_maxAbs);
    printf("Max relative error: %.10e\n", h_maxRel);

    bool valid = (h_maxRel <= 1e-6);
    if (!valid) {
        printf("Validation failed: relative error too large\n");
    }

    cudaFree(d_reconstructed);
    cudaFree(dd_maxAbs);
    cudaFree(dd_maxRel);
    return valid;
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

    size_t nn = n * n;

    // Select GPU device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("Error: No CUDA devices found\n");
        return 1;
    }

    // Pick the largest GPU for best performance
    int selectedDevice = 0;
    size_t maxMem = 0;
    for (int d = 0; d < deviceCount; d++) {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, d);
        if (dprop.totalGlobalMem > maxMem) {
            maxMem = dprop.totalGlobalMem;
            selectedDevice = d;
        }
    }
    cudaSetDevice(selectedDevice);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, selectedDevice);
    printf("Using GPU: %s (Compute %d.%d, %zu MB)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024*1024));

    // Allocate GPU memory
    double* d_A = nullptr;
    double* d_B = nullptr;
    cudaMalloc(&d_A, nn * sizeof(double));
    cudaMalloc(&d_B, nn * sizeof(double));

    // Create cuBLAS and cuSOLVER handles
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(d_A, d_B, cublasHandle, n);

    // Save original for validation (on GPU)
    double* d_A_orig = nullptr;
    if (validate) {
        cudaMalloc(&d_A_orig, nn * sizeof(double));
        cudaMemcpy(d_A_orig, d_A, nn * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d_A, n, cusolverHandle);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        cudaFree(d_A);
        cudaFree(d_B);
        if (d_A_orig) cudaFree(d_A_orig);
        cublasDestroy(cublasHandle);
        cusolverDnDestroy(cusolverHandle);
        return 1;
    }

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy result back to host for printing/validation
    std::vector<double> h_A(nn);
    cudaMemcpy(h_A.data(), d_A, nn * sizeof(double), cudaMemcpyDeviceToHost);

    // Print results for external validation
    if (printResults) {
        print_results(h_A, "CholeskyL");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(d_A, d_A_orig, n, cublasHandle);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }

        cudaFree(d_A_orig);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    return 0;
}
