#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

static cublasHandle_t handle = nullptr;
static bool cuda_initialized = false;

extern "C" {

void cuda_init() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA device initialization failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        return;
    }
    
    cuda_initialized = true;
}

bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0 && cuda_initialized);
}

void cuda_finalize() {
    if (handle != nullptr) {
        cublasDestroy(handle);
        handle = nullptr;
    }
    cuda_initialized = false;
}

// CUDA kernel for matrix multiplication using cuBLAS
void cuda_matmul(double* A, double* B, double* C, size_t N, size_t block_start, size_t block_end) {
    if (!cuda_initialized || handle == nullptr) {
        return;
    }
    
    size_t local_rows = block_end - block_start;
    
    // Allocate GPU memory
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    cudaMalloc(&d_A, local_rows * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, local_rows * N * sizeof(double));
    
    // Copy data to GPU
    cudaMemcpy(d_A, A, local_rows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Perform matrix multiplication: C = A * B
    // A: local_rows x N, B: N x N, C: local_rows x N
    const double alpha = 1.0;
    const double beta = 0.0;
    
    cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        N, local_rows, N,
                                        &alpha,
                                        d_B, N,
                                        d_A, N,
                                        &beta,
                                        d_C, N);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS DGEMM failed with status %d\n", status);
    }
    
    // Copy result back to host
    cudaMemcpy(C, d_C, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

} // extern "C"
